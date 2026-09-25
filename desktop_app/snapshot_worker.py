"""
desktop_app/snapshot_worker.py
--------------------------------
Grabs ONE clean frame from a video source off the GUI thread, for the zone
editor to draw on when no pipeline run is active (a running pipeline supplies
its own via PipelineWorker.request_snapshot()).

The frame is returned at *processing* resolution -- the same size the pipeline
would run at -- so coordinates drawn on it mean the same thing at run time.
"""

from PySide6.QtCore import QThread, Signal

from src.camera_capture import open_camera_capture
from src.config import DET_CFG
from src.frame_geometry import to_processing_frame
from src.source_resolver import SourceError, resolve_source

# A live camera's first frames are often dark/blurry while auto-exposure
# settles, and buffered ones are stale -- read a few and keep the last.
LIVE_WARMUP_READS = 5


class SnapshotWorker(QThread):
    snapshot_ready = Signal(object)   # np.ndarray (BGR)
    failed = Signal(str)

    def __init__(self, source: str, parent=None):
        super().__init__(parent)
        self.source = source

    def run(self):
        try:
            resolved = resolve_source(self.source)
        except SourceError as exc:
            self.failed.emit(str(exc))
            return
        is_live = resolved.is_live
        cap, _backend, err = open_camera_capture(resolved.location)
        if cap is None:
            self.failed.emit(f"Could not open {self.source!r}: {err}")
            return
        frame = None
        try:
            for _ in range(LIVE_WARMUP_READS if is_live else 1):
                ok, f = cap.read()
                if ok and f is not None and f.size > 0:
                    frame = f
        finally:
            cap.release()
        if frame is None:
            self.failed.emit("The source opened but delivered no frame.")
            return
        self.snapshot_ready.emit(to_processing_frame(frame, is_live, DET_CFG.max_processing_dim).copy())
