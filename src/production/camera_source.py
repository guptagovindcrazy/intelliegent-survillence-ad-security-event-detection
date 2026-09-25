"""
production/camera_source.py
-----------------------------
Producer side of the producer/consumer architecture.

Design notes for viva:
    Each camera gets one lightweight thread whose only job is: read a frame,
    push it onto ITS OWN queue, repeat. It does NOT run any detection,
    tracking, or alert logic -- that separation is exactly what stops one slow
    or misbehaving camera from blocking every other camera in the system: a
    stuck capture thread only ever affects its own queue slot, never the
    inference workers or other cameras.

    Backpressure: each camera's queue is bounded (frame_queue_maxsize) and
    belongs to that camera alone -- NOT shared with any other camera's queue.
    If inference is slower than capture, we drop the OLDEST queued frame for
    THIS camera rather than blocking capture or growing memory without bound
    -- an explicit design decision (see ProductionConfig), not an accidental
    leak. Because the queue is per-camera, one busy/fast camera can never
    evict or starve another camera's frames; the isolation claim above is
    now actually enforced by the type, not just the intent.
"""

import queue
import threading
import time

import cv2

from src.camera_capture import open_camera_capture
from src.config import CameraConfig, ProductionConfig
from src.production.errors import cam_not_found, stream_timeout


class CameraSource(threading.Thread):
    def __init__(self, cam_cfg: CameraConfig, prod_cfg: ProductionConfig,
                 own_queue: "queue.Queue", error_sink, state_store):
        super().__init__(daemon=True, name=f"CameraSource-{cam_cfg.camera_id}")
        self.cam_cfg = cam_cfg
        self.prod_cfg = prod_cfg
        # This camera's OWN queue -- not shared with any other camera. See
        # module docstring: this is what makes backpressure/eviction
        # actually scoped to this camera alone.
        self.own_queue = own_queue
        self.error_sink = error_sink
        self.state_store = state_store
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _open_capture(self):
        """Returns (cap, error_message). cap is None on failure. See
        src/camera_capture.py for why this isn't a plain cv2.VideoCapture()
        call: for a live camera index it verifies a real frame actually
        arrives before trusting the capture, instead of only checking
        isOpened() -- which is exactly what can be True while a camera
        silently delivers nothing (the classic Windows MSMF failure mode)."""
        cap, backend_used, err = open_camera_capture(self.cam_cfg.source)
        if cap is not None:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.prod_cfg.capture_buffersize)
        return cap, err

    def run(self):
        cap, open_err = self._open_capture()
        if cap is None:
            self.error_sink(cam_not_found(self.cam_cfg.camera_id, f"{self.cam_cfg.source} ({open_err})"))
            self.state_store.mark_unhealthy(self.cam_cfg.camera_id)
            return

        last_frame_at = time.time()
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            now = time.time()

            if not ok:
                if now - last_frame_at > self.prod_cfg.stream_timeout_sec:
                    self.error_sink(stream_timeout(self.cam_cfg.camera_id, self.prod_cfg.stream_timeout_sec))
                    self.state_store.mark_unhealthy(self.cam_cfg.camera_id)
                    # attempt a reconnect rather than dying silently
                    cap.release()
                    time.sleep(1.0)
                    cap, open_err = self._open_capture()
                    if cap is None:
                        self.error_sink(cam_not_found(self.cam_cfg.camera_id, f"{self.cam_cfg.source} ({open_err})"))
                        self.state_store.mark_unhealthy(self.cam_cfg.camera_id)
                        return
                    last_frame_at = time.time()
                continue

            last_frame_at = now

            # Backpressure: drop the oldest frame in THIS camera's own queue
            # if it's full, so capture never blocks and memory never grows
            # unbounded -- and so another camera's frames are never touched.
            try:
                self.own_queue.put_nowait((frame, now))
            except queue.Full:
                try:
                    self.own_queue.get_nowait()  # discard oldest, this camera's own
                except queue.Empty:
                    pass
                self.state_store.record_dropped_frame(self.cam_cfg.camera_id)
                try:
                    self.own_queue.put_nowait((frame, now))
                except queue.Full:
                    pass  # extremely bursty -- just skip this frame

        cap.release()
