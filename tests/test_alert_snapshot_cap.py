"""
tests/test_alert_snapshot_cap.py
------------------------------------
Regression test for: summary.alerts (each carrying a full PNG snapshot)
grew without bound for the whole run, with no cap. A long unattended run
on a busy scene could accumulate hundreds of MB to low GB in RAM, then
get base64-inflated again into the saved HTML report.

The fix keeps every alert's METADATA (needed for an accurate audit trail/
count) but stops retaining new snapshot images once MAX_ALERT_SNAPSHOTS is
reached, and notifies the user exactly once when that happens.
"""

import tempfile
import os
import sys
import types
import unittest


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


class _OneObjectDetector:
    def __init__(self, *a, **k):
        pass

    def process(self, frame):
        class Obj:
            track_id = 1
            bbox_xyxy = (5, 5, 20, 20)
            foot_point = (12, 20)
        return [Obj()]

    def close(self):
        pass


class _AlwaysAlertEngine:
    """Fires a fresh alert on every call -- real AlertEngine's cooldowns/
    dwell timers make many distinct alerts hard to generate quickly in a
    short test clip, and aren't what this test checks (see
    test_alert_engine_tripwire.py for cooldown behavior). This test only
    checks pipeline_worker's snapshot-retention cap."""

    def evaluate(self, spatial_state, temporal_event, now=None):
        from src.alert_engine import SecurityAlert
        return SecurityAlert(
            track_id=1, alert_type="DWELL_VIOLATION", zone_or_wire="TestZone",
            message="test alert", severity="high",
        )


def _make_clip(path, n_frames=30):
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(path, fourcc, 15, (64, 64))
    for _ in range(n_frames):
        w.write(np.zeros((64, 64, 3), dtype="uint8"))
    w.release()


@unittest.skipUnless(
    __import__("importlib").util.find_spec("PySide6") is not None,
    "PySide6 not installed",
)
class TestAlertSnapshotCap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_snapshots_stop_growing_but_alert_metadata_does_not(self):
        from PySide6.QtCore import QTimer

        _stub_heavy_deps()
        import desktop_app.pipeline_worker as pw

        original_cap = pw.MAX_ALERT_SNAPSHOTS
        pw.MAX_ALERT_SNAPSHOTS = 5
        pw.DetectionTrackingLayer = _OneObjectDetector
        pw.AlertEngine = _AlwaysAlertEngine
        self.addCleanup(setattr, pw, "MAX_ALERT_SNAPSHOTS", original_cap)

        n_frames = 30
        clip_path = os.path.join(tempfile.gettempdir(), "_test_alert_snapshot_cap.mp4")
        _make_clip(clip_path, n_frames=n_frames)

        worker = pw.PipelineWorker(clip_path, capture_snapshots=True)
        statuses = []
        results = {}
        worker.status_changed.connect(statuses.append)

        def on_finished(summary):
            results["summary"] = summary
            self.app.quit()

        worker.finished_run.connect(on_finished)
        worker.start()

        # Real QThread.start() is asynchronous, and Signal delivery to the
        # main thread is queued -- pumping a real event loop (like
        # test_desktop_lifecycle.py does) is what actually delivers
        # finished_run here, not just waiting on the thread object.
        QTimer.singleShot(10000, self.app.quit)  # safety timeout
        self.app.exec()
        worker.wait(2000)

        self.assertIn("summary", results, "finished_run never fired -- worker likely hung or crashed")
        summary = results["summary"]
        n_with_snapshot = sum(1 for a in summary.alerts if a.snapshot_png is not None)
        n_without_snapshot = len(summary.alerts) - n_with_snapshot

        self.assertEqual(len(summary.alerts), n_frames,
                          "every alert's metadata should be recorded regardless of the cap")
        self.assertEqual(n_with_snapshot, 5, "snapshot retention should stop exactly at the cap")
        self.assertEqual(n_without_snapshot, n_frames - 5)

        cap_notices = [s for s in statuses if "snapshot memory cap" in s]
        self.assertEqual(len(cap_notices), 1, "the cap notice should fire exactly once")


if __name__ == "__main__":
    unittest.main()
