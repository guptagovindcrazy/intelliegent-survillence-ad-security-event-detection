"""
production/state_store.py
---------------------------
Central store of per-camera pipeline state, keyed by camera_id.

Design notes for viva:
    The inference workers themselves are "stateless" in the sense that they
    carry no camera-specific data in the worker thread itself -- any worker
    can pick up any camera's frame from the shared queue and process it
    correctly, because all the state that makes tracking/zone/dwell logic
    work (track IDs, zone occupancy, FSM timers, per-camera metrics) lives
    here, keyed by camera_id, instead of inside a particular worker.

    This is what allows horizontal scaling: adding a worker thread (or, in a
    real distributed deployment, a worker *process* on another machine) never
    requires re-partitioning cameras across workers by hand -- the state store
    is the single source of truth every worker reads from and writes to.

    NOTE ON SCOPE: this in-process, dict-based store is intentionally simple
    for a single-machine academic demo. A real multi-machine deployment would
    back this with Redis or a similar shared store so state_store methods work
    the same way across process/machine boundaries -- the calling code in
    inference_worker.py would not need to change.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class CameraMetrics:
    frames_processed: int = 0
    alerts_raised: int = 0
    dropped_frames: int = 0
    last_frame_time: Optional[float] = None
    last_latency_sec: float = 0.0
    is_healthy: bool = True


@dataclass
class PerCameraState:
    """Bundle of everything one camera's pipeline needs between frames.

    `lock` must be held by whichever worker thread is running this camera's
    frame through detection/tracking/zone/dwell/alert logic. Multiple
    InferenceWorker threads all pull from shared queues and can pick up two
    frames for the SAME camera back-to-back (this is normal, not a bug --
    see camera_source.py's per-camera queue), but the tracker/zone/dwell
    state below is NOT safe to touch from two threads at once:
    DetectionTrackingLayer wraps an Ultralytics tracker with persist=True
    (explicitly documented as not thread-safe to share), and
    TemporalAnalysisLayer's per-track FSM would corrupt dwell timers if two
    threads updated the same track's history concurrently. The lock makes
    one camera's frames process strictly one-at-a-time (which tracking
    correctness requires anyway -- frame order matters for a tracker) while
    leaving different cameras fully parallel across workers."""
    detection_tracking: object
    spatial_zone: object
    temporal_analysis: object
    alert_engine: object
    metrics: CameraMetrics = field(default_factory=CameraMetrics)
    # Frame size the spatial layer's zones are currently expressed in (None =
    # unknown/as-is, never rescaled). InferenceWorker keeps it in step with
    # the camera's real frame size.
    zone_frame_size: Optional[tuple] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class StateStore:
    def __init__(self, layer_factory: Callable[[str], PerCameraState]):
        """`layer_factory(camera_id) -> PerCameraState` builds a fresh set of
        per-camera layer instances the first time that camera is seen."""
        self._lock = threading.Lock()
        self._layer_factory = layer_factory
        self._cameras: Dict[str, PerCameraState] = {}

    def get_or_create(self, camera_id: str) -> PerCameraState:
        with self._lock:
            if camera_id not in self._cameras:
                self._cameras[camera_id] = self._layer_factory(camera_id)
            return self._cameras[camera_id]

    def record_frame_processed(self, camera_id: str, latency_sec: float, alert_raised: bool):
        with self._lock:
            state = self._cameras.get(camera_id)
            if state is None:
                return
            m = state.metrics
            m.frames_processed += 1
            m.last_frame_time = time.time()
            m.last_latency_sec = latency_sec
            m.is_healthy = True
            if alert_raised:
                m.alerts_raised += 1

    def record_dropped_frame(self, camera_id: str):
        with self._lock:
            state = self._cameras.get(camera_id)
            if state:
                state.metrics.dropped_frames += 1

    def mark_unhealthy(self, camera_id: str):
        with self._lock:
            state = self._cameras.get(camera_id)
            if state:
                state.metrics.is_healthy = False

    def snapshot(self) -> Dict[str, CameraMetrics]:
        with self._lock:
            return {cam_id: state.metrics for cam_id, state in self._cameras.items()}
