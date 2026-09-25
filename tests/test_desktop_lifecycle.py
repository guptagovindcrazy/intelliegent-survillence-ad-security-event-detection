"""
tests/test_desktop_lifecycle.py
----------------------------------
Regression tests for two real crash bugs found by actually RUNNING the
desktop app with a real QApplication/QThread event loop (not just reading
the code):

  1. Calling Start again while a run was already in progress created a
     second PipelineWorker and orphaned the first, still-running QThread.
     Qt aborts the process when an orphaned running QThread is garbage
     collected ("QThread: Destroyed while thread is still running").
     Fixed with a re-entrancy guard at the top of MainWindow._start().

  2. Quitting the QApplication any way OTHER than the window's close button
     (e.g. QApplication.quit() called directly) bypassed closeEvent's
     worker-cleanup logic entirely, hitting the same crash. Fixed with an
     app.aboutToQuit handler in desktop_app/main.py that stops a running
     worker regardless of how the quit was triggered.

These tests need PySide6 and a Qt platform plugin. In a headless/CI
environment, run with QT_QPA_PLATFORM=offscreen. They stub out the
ultralytics/torch imports so no real model weights or GPU are required --
same pattern used for manual smoke testing during development.
"""

import tempfile
import os
import sys
import types
import unittest

try:
    from PySide6.QtWidgets import QApplication
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False


def _stub_heavy_deps():
    """Stub ultralytics/torch in sys.modules so desktop_app.pipeline_worker
    can be imported without the real (heavy) ML dependencies installed."""
    if "ultralytics" not in sys.modules:
        fake_ultra = types.ModuleType("ultralytics")

        class _FakeModel:
            def __init__(self, *a, **k):
                pass

        fake_ultra.YOLO = _FakeModel
        fake_ultra.RTDETR = _FakeModel
        sys.modules["ultralytics"] = fake_ultra
    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = fake_torch


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed")
class TestDesktopAppLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _stub_heavy_deps()
        import src.layers.detection_tracking as dt_module

        class _SlowFakeLayer:
            """A frame or two of artificial delay so a run is genuinely still
            in progress when the test checks it mid-run, instead of the real
            work finishing before the check runs (a mistake made once during
            manual testing that produced a false-positive bug report)."""
            def __init__(self, *a, **k):
                pass

            def process(self, frame):
                import time
                time.sleep(0.05)
                return []

            def close(self):
                pass

        dt_module.DetectionTrackingLayer = _SlowFakeLayer
        import desktop_app.pipeline_worker as pw_module
        pw_module.DetectionTrackingLayer = _SlowFakeLayer
        cls._pw_module = pw_module

        cls.app = QApplication.instance() or QApplication(sys.argv)

        import cv2
        import numpy as np
        cls.clip_path = os.path.join(tempfile.gettempdir(), "_lifecycle_test_clip.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cls.clip_path, fourcc, 15, (320, 240))
        for _ in range(120):  # 120 frames * 50ms/frame = ~6s of genuine processing time
            writer.write(np.zeros((240, 320, 3), dtype="uint8"))
        writer.release()

    def test_double_start_does_not_orphan_a_running_worker(self):
        from desktop_app.main_window import MainWindow
        from PySide6.QtCore import QTimer

        w = MainWindow()
        w.file_path_edit.setText(self.clip_path)
        w._start()
        first_worker = w.worker
        self.assertTrue(first_worker.isRunning())

        w._start()  # re-entrant call while first run is still genuinely active
        self.assertIs(w.worker, first_worker, "a second call to _start() while running must be a no-op")

        w._stop()
        QTimer.singleShot(0, self.app.quit)
        self.app.exec()
        first_worker.wait(2000)

    def test_about_to_quit_stops_a_running_worker(self):
        from desktop_app.main_window import MainWindow
        from PySide6.QtCore import QTimer

        w = MainWindow()
        cleanup_called = {"value": False}

        def cleanup():
            cleanup_called["value"] = True
            if w.worker is not None and w.worker.isRunning():
                w.worker.stop()
                w.worker.wait(2000)

        self.app.aboutToQuit.connect(cleanup)
        w.file_path_edit.setText(self.clip_path)
        w._start()
        self.assertTrue(w.worker.isRunning())

        QTimer.singleShot(100, self.app.quit)  # quit while genuinely still running
        self.app.exec()

        self.assertTrue(cleanup_called["value"])
        self.assertFalse(w.worker.isRunning())
        self.app.aboutToQuit.disconnect(cleanup)


if __name__ == "__main__":
    unittest.main()
