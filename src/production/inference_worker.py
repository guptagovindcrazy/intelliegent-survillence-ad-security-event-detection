"""
production/inference_worker.py
--------------------------------
Consumer side of the producer/consumer architecture.

Design notes for viva:
    A pool of worker threads polls across a per-camera set of frame queues
    (one queue per camera_id, owned by that camera's CameraSource -- see
    camera_source.py). Any worker can pick up the next frame for any camera;
    all camera-specific state (tracker, zone cache, dwell timers, metrics) is
    looked up from the central StateStore by camera_id, so the worker itself
    holds nothing camera-specific across iterations. That is what "stateless
    worker" means in this design and is what lets you add worker threads/
    replicas as camera count grows, without re-architecting how cameras map
    to workers.

    Per-camera locking: two workers CAN legitimately pick up two frames for
    the same camera back-to-back (e.g. one camera producing faster than the
    others). PerCameraState.lock (state_store.py) serializes the actual
    detection/tracking/zone/dwell/alert work for a given camera across
    whichever worker(s) grab its frames, since Ultralytics' persist=True
    tracker and the per-track dwell FSM are not safe to touch from two
    threads at once. Different cameras still run fully in parallel, since
    each has its own independent lock.

    Wall-clock timers: TemporalAnalysisLayer already keys dwell time off
    `time.time()` rather than frame count (see src/layers/temporal_analysis.py),
    so dwell duration stays correct even if a camera's FPS drops under load --
    important now that frames can be dropped for backpressure.
"""

import queue
import threading
import time

from src.alert_engine import AlertEngine
from src.config import DET_CFG, TRACK_CFG, TEMPORAL_CFG
from src.layers.detection_tracking import DetectionTrackingLayer
from src.frame_geometry import rescale_shapes
from src.layers.spatial_zones import SpatialZoneLayer
from src.layers.temporal_analysis import TemporalAnalysisLayer
from src.production.errors import RetryExhausted, model_load_failed, retry_with_fallback
from src.production.state_store import CameraMetrics, PerCameraState, StateStore


def make_layer_factory(cameras_by_id: dict, prod_cfg, error_sink):
    """Returns a factory StateStore can call to build a fresh PerCameraState the
    first time a given camera_id is seen. Model loading is retried per
    ProductionConfig.model_load_retries before raising ERR_MODEL_500."""

    def factory(camera_id: str) -> PerCameraState:
        cam_cfg = cameras_by_id[camera_id]

        def _load():
            return DetectionTrackingLayer(DET_CFG, TRACK_CFG)

        def _on_failure(attempt, exc):
            error_sink(model_load_failed(camera_id, f"attempt {attempt}: {exc}"))

        detection_tracking = retry_with_fallback(_load, prod_cfg.model_load_retries, _on_failure)

        return PerCameraState(
            detection_tracking=detection_tracking,
            spatial_zone=SpatialZoneLayer(cam_cfg.zones, cam_cfg.tripwires),
            zone_frame_size=cam_cfg.zone_reference_size,
            temporal_analysis=TemporalAnalysisLayer(TEMPORAL_CFG),
            alert_engine=AlertEngine(),
            metrics=CameraMetrics(),
        )

    return factory


class InferenceWorker(threading.Thread):
    def __init__(self, worker_id: int, frame_queues: "Dict[str, queue.Queue]", state_store: StateStore,
                 zone_severity_lookup: dict, alert_dispatcher, error_sink=None, on_fatal_error=None,
                 idle_poll_sec: float = 0.02):
        super().__init__(daemon=True, name=f"InferenceWorker-{worker_id}")
        # One queue PER CAMERA (built once by live_main.py and shared,
        # read-only as a dict, across every CameraSource/InferenceWorker) --
        # NOT one shared queue for every camera. This is what makes each
        # camera's backpressure/eviction genuinely scoped to itself: see
        # camera_source.py, which only ever touches its own entry here.
        self.frame_queues = frame_queues
        self.state_store = state_store
        self.zone_severity_lookup = zone_severity_lookup
        self.alert_dispatcher = alert_dispatcher
        self.error_sink = error_sink or (lambda err: None)
        self.on_fatal_error = on_fatal_error or (lambda camera_id, err: None)
        self.idle_poll_sec = idle_poll_sec
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def _fit_zones_to_frame(state, frame) -> None:
        """Keeps a camera's zones in the coordinate space of the frames it
        actually delivers. Zones drawn at another resolution are rescaled the
        first time a frame arrives (and again if the stream's resolution ever
        changes); the common case -- same size as last time -- is one tuple
        comparison. Caller holds state.lock."""
        ref = getattr(state, "zone_frame_size", None)
        if ref is None:
            return  # no reference recorded: coordinates are taken as-is
        size = (frame.shape[1], frame.shape[0])
        if size == tuple(ref):
            return
        zones, wires = rescale_shapes(state.spatial_zone.zones, state.spatial_zone.tripwires, ref, size)
        state.spatial_zone = SpatialZoneLayer(zones, wires)
        state.zone_frame_size = size

    def _next_frame(self):
        """Round-robins a non-blocking check across every camera's own
        queue and returns the first frame found, or None if all are
        currently empty. Never blocks on any single camera, so one idle
        camera can't stall a worker that could be serving a busy one."""
        for camera_id, q in list(self.frame_queues.items()):
            try:
                frame, captured_at = q.get_nowait()
                return camera_id, frame, captured_at
            except queue.Empty:
                continue
        return None

    def run(self):
        while not self._stop_event.is_set():
            item = self._next_frame()
            if item is None:
                time.sleep(self.idle_poll_sec)
                continue
            camera_id, frame, captured_at = item

            start = time.time()
            try:
                state = self.state_store.get_or_create(camera_id)
            except RetryExhausted as exc:
                # Model load never succeeded for this camera after every retry --
                # this is the ERR_MODEL_500 / CRITICAL path: log it, mark the camera
                # unhealthy, and let the orchestrator's HealthMonitor push the whole
                # system to CRITICAL rather than silently killing this worker thread.
                err = exc.last_error
                err.camera_id = camera_id
                self.error_sink(err)
                self.state_store.mark_unhealthy(camera_id)
                self.on_fatal_error(camera_id, err)
                continue

            # Serialize this camera's frame through its own tracker/zone/dwell
            # state -- see PerCameraState.lock's docstring. A second worker
            # that grabs this same camera's NEXT frame will simply block here
            # until this one finishes, which is correct: a tracker must see
            # frames for one camera in order, one at a time.
            with state.lock:
                self._fit_zones_to_frame(state, frame)
                tracked_objects = state.detection_tracking.process(frame)
                alert_raised = False
                now = time.time()

                for obj in tracked_objects:
                    spatial_state = state.spatial_zone.evaluate(obj.track_id, obj.foot_point)
                    temporal_event = state.temporal_analysis.update(
                        spatial_state, obj.foot_point, self.zone_severity_lookup, now=now
                    )
                    alert = state.alert_engine.evaluate(spatial_state, temporal_event, now=now)
                    if alert:
                        alert_raised = True
                        self.alert_dispatcher(camera_id, alert)

            latency = time.time() - start
            self.state_store.record_frame_processed(camera_id, latency, alert_raised)

