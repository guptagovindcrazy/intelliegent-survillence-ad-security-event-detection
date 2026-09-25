"""
tests/test_pipeline_worker_progress.py
----------------------------------------
Regression tests for the "UI looks frozen on a slow, large-frame run"
usability bug: PipelineWorker.run() previously emitted status_changed only
at the very start ("Running -- <source>") and the very end ("Finished") of
a file-based run, with no feedback in between -- so a CPU-only run on a
large/crowded 4K video that takes many seconds per frame looked hung.

These tests call PipelineWorker.run() directly (not via .start()/a real
QThread) so signal emission is synchronous and deterministic to assert on;
the existing tests/test_desktop_lifecycle.py already covers the QThread
lifecycle itself (start/stop, orphaned-thread crash) with a real thread.

Like test_desktop_lifecycle.py, ultralytics/torch are stubbed out so no real
model weights or GPU are required, and PySide6 is required to run at all.
"""

import tempfile
import os
import sys
import time
import types
import unittest

try:
    from PySide6.QtWidgets import QApplication
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False


def _stub_heavy_deps():
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


class _SlowFakeLayer:
    """Simulates real per-frame RT-DETR-L/CPU latency, and records the shape
    of every frame it's asked to process so tests can confirm what
    resolution detection actually ran at."""

    seen_shapes = []
    per_frame_delay_sec = 0.05

    def __init__(self, *a, **k):
        pass

    def process(self, frame):
        time.sleep(type(self).per_frame_delay_sec)
        type(self).seen_shapes.append(frame.shape[:2])
        return []

    def close(self):
        pass


def _make_video(path, width, height, n_frames, fps=30):
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(n_frames):
        writer.write(np.zeros((height, width, 3), dtype="uint8"))
    writer.release()


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed")
class TestPipelineWorkerProgressAndDownscale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _stub_heavy_deps()
        import src.layers.detection_tracking as dt_module
        dt_module.DetectionTrackingLayer = _SlowFakeLayer
        import desktop_app.pipeline_worker as pw_module
        pw_module.DetectionTrackingLayer = _SlowFakeLayer
        cls.pw_module = pw_module

        cls.app = QApplication.instance() or QApplication(sys.argv)

        cls.large_clip = os.path.join(tempfile.gettempdir(), "_progress_test_large.mp4")
        _make_video(cls.large_clip, width=3840, height=2160, n_frames=12)

        cls.normal_clip = os.path.join(tempfile.gettempdir(), "_progress_test_normal.mp4")
        _make_video(cls.normal_clip, width=640, height=480, n_frames=6)

    def _run_and_capture(self, source, per_frame_delay):
        _SlowFakeLayer.seen_shapes = []
        _SlowFakeLayer.per_frame_delay_sec = per_frame_delay

        worker = self.pw_module.PipelineWorker(source)
        statuses, preview_shapes = [], []
        worker.status_changed.connect(statuses.append)
        worker.frame_ready.connect(lambda f: preview_shapes.append(f.shape[:2]))
        # Call run() directly (synchronous, same thread) rather than start()
        # (a real QThread) -- deterministic, and signals still fire, since a
        # direct call/same-thread emit uses a direct (not queued) connection.
        worker.run()
        return statuses, preview_shapes, list(_SlowFakeLayer.seen_shapes)

    def test_large_slow_file_run_gets_progress_updates_not_silence(self):
        statuses, _, _ = self._run_and_capture(self.large_clip, per_frame_delay=0.3)

        progress_msgs = [s for s in statuses if s.startswith("Processing frame")]
        self.assertGreater(
            len(progress_msgs), 3,
            "a slow multi-second run produced no intermediate progress -- "
            "this is the exact 'looks frozen' bug",
        )
        self.assertIn("Processing frame 1 / 12", progress_msgs[0])
        self.assertIn("%", progress_msgs[0])

    def test_large_frame_is_downscaled_before_detection_and_preview(self):
        _, preview_shapes, detector_shapes = self._run_and_capture(
            self.large_clip, per_frame_delay=0.01
        )
        max_det_dim = max(detector_shapes[0])
        max_preview_dim = max(preview_shapes[0])
        self.assertLessEqual(max_det_dim, self.pw_module.DETECTION_MAX_DIM)
        self.assertLessEqual(max_preview_dim, self.pw_module.PREVIEW_MAX_DIM)
        self.assertLess(max_det_dim, 3840, "4K frame reached the detector at native resolution")

    def test_normal_resolution_video_is_completely_unaffected(self):
        statuses, preview_shapes, detector_shapes = self._run_and_capture(
            self.normal_clip, per_frame_delay=0.01
        )
        self.assertEqual(detector_shapes[0], (480, 640))
        self.assertEqual(preview_shapes[0], (480, 640))
        # still get progress feedback, just no resizing needed
        self.assertTrue(any(s.startswith("Processing frame") for s in statuses))


if __name__ == "__main__":
    unittest.main()
