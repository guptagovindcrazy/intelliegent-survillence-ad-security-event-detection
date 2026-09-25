"""
tests/test_batch_run_resilience.py
-------------------------------------
Regression tests for: run_batch() had no per-video exception handling, so
one bad/corrupt video anywhere in a folder crashed the whole batch, wrote
NO summary_report.md at all, and silently skipped every video queued after
the bad one -- even perfectly fine ones.

Follows the same ultralytics/torch stubbing convention as
test_desktop_lifecycle.py so no real model weights or GPU are required.
"""

import os
import shutil
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np


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


def _make_clip(path, n_frames=5):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(path, fourcc, 15, (64, 64))
    for _ in range(n_frames):
        w.write(np.zeros((64, 64, 3), dtype="uint8"))
    w.release()


class _FailsOnNthConstruction:
    """Stands in for DetectionTrackingLayer: raises during __init__ for one
    specific video (simulating a video-level fatal failure -- corrupted
    container metadata, an unreadable file, etc.), succeeds for every
    other video."""
    construct_count = 0
    fail_on = None  # set per-test

    def __init__(self, *a, **k):
        type(self).construct_count += 1
        if type(self).construct_count == type(self).fail_on:
            raise RuntimeError("simulated fatal failure for this video")

    def process(self, frame):
        return []

    def close(self):
        pass


class _FailsOnNthFrame:
    """Stands in for DetectionTrackingLayer: process() raises on specific
    frame-call numbers (simulating one bad frame inside an otherwise fine
    video), succeeds on every other call."""
    call_count = 0
    fail_on_calls = frozenset()

    def __init__(self, *a, **k):
        pass

    def process(self, frame):
        type(self).call_count += 1
        if type(self).call_count in type(self).fail_on_calls:
            raise RuntimeError("simulated bad frame")
        return []

    def close(self):
        pass


class TestBatchRunResilience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _stub_heavy_deps()
        import src.batch_run as batch_run
        cls.batch_run = batch_run

    def setUp(self):
        self.in_dir = tempfile.mkdtemp()
        self.out_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.in_dir, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.out_dir, ignore_errors=True)

    def test_one_video_raising_fatally_does_not_abort_the_batch(self):
        for name in ["01_good.mp4", "02_good.mp4", "03_bad.mp4", "04_good.mp4"]:
            _make_clip(os.path.join(self.in_dir, name))

        _FailsOnNthConstruction.construct_count = 0
        _FailsOnNthConstruction.fail_on = 3  # 03_bad.mp4
        self.batch_run.DetectionTrackingLayer = _FailsOnNthConstruction

        # must not raise -- this is the actual bug under test
        self.batch_run.run_batch(self.in_dir, self.out_dir)

        summary_path = os.path.join(self.out_dir, "summary_report.md")
        self.assertTrue(os.path.exists(summary_path), "no summary was written at all")
        with open(summary_path) as f:
            summary = f.read()

        # the bad video is reported as failed...
        self.assertIn("03_bad.mp4", summary)
        self.assertIn("Could not process", summary)
        # ...but every OTHER video, including the one queued AFTER the bad
        # one, was still processed and reported.
        self.assertIn("01_good.mp4", summary)
        self.assertIn("02_good.mp4", summary)
        self.assertIn("04_good.mp4", summary)
        self.assertIn("Videos processed successfully: 3/4", summary)

        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "alerts.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "batch_errors.log")))

    def test_one_bad_frame_does_not_lose_the_rest_of_its_own_video(self):
        _make_clip(os.path.join(self.in_dir, "clip.mp4"), n_frames=6)

        _FailsOnNthFrame.call_count = 0
        _FailsOnNthFrame.fail_on_calls = frozenset({3})  # frame 3 of 6 fails
        self.batch_run.DetectionTrackingLayer = _FailsOnNthFrame

        result = self.batch_run.process_one_video(os.path.join(self.in_dir, "clip.mp4"))
        self.assertNotIn("error", result)
        self.assertEqual(result["frames_processed"], 6, "a bad frame should be skipped, not stop the video early")
        self.assertEqual(len(result["frame_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
