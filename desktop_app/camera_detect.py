"""
desktop_app/camera_detect.py
-------------------------------
Runs src.camera_capture.probe_camera_indices() off the GUI thread and
reports results back via a Signal, so scanning several camera indices
(each potentially trying 2 backends) never freezes the window -- the
exact class of bug this whole desktop app has been about avoiding.
"""

from PySide6.QtCore import QThread, Signal

from src.camera_capture import probe_camera_indices


class CameraDetectWorker(QThread):
    finished_detect = Signal(list)  # list of (index, backend) tuples

    def __init__(self, max_index: int = 4, parent=None):
        super().__init__(parent)
        self.max_index = max_index

    def run(self):
        found = probe_camera_indices(max_index=self.max_index)
        self.finished_detect.emit(found)
