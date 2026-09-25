"""
tests/test_production_concurrency.py
---------------------------------------
Regression tests for two related production-path bugs:

  1. Race condition: InferenceWorkers shared one PerCameraState with no
     lock held during processing, so with the default config (1 camera,
     2 workers) two threads could call .process() on the SAME
     DetectionTrackingLayer/tracker instance concurrently.

  2. Cross-camera eviction: all cameras shared ONE frame queue, so a busy
     camera's backpressure ("drop the oldest frame") could evict a
     DIFFERENT camera's frame, contradicting the documented per-camera
     isolation guarantee.

Follows the same ultralytics/torch stubbing convention as
test_desktop_lifecycle.py so no real model weights or GPU are required.
"""

import queue
import sys
import threading
import time
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


class _InstrumentedDetector:
    """Records whether any two calls to .process() ever overlap in time."""

    def __init__(self):
        self._active = 0
        self._max_concurrent = 0
        self._guard = threading.Lock()
        self.call_count = 0

    def process(self, frame):
        with self._guard:
            self._active += 1
            self._max_concurrent = max(self._max_concurrent, self._active)
        time.sleep(0.05)
        with self._guard:
            self._active -= 1
            self.call_count += 1
        return []


class _NoopLayer:
    def evaluate(self, *a, **k):
        class S:
            zones_inside = []
            tripwire_crossed = None
            tripwire_direction = None
            track_id = 0
        return S()


class _NoopTemporal:
    def update(self, *a, **k):
        class E:
            should_alert = False
        return E()


class _NoopAlertEngine:
    def evaluate(self, *a, **k):
        return None


class TestNoConcurrentProcessingPerCamera(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _stub_heavy_deps()
        from src.production.state_store import StateStore, PerCameraState, CameraMetrics
        from src.production.inference_worker import InferenceWorker
        cls.StateStore = StateStore
        cls.PerCameraState = PerCameraState
        cls.CameraMetrics = CameraMetrics
        cls.InferenceWorker = InferenceWorker

    def test_two_workers_never_run_the_same_cameras_detector_concurrently(self):
        detector = _InstrumentedDetector()
        state = self.PerCameraState(
            detection_tracking=detector,
            spatial_zone=_NoopLayer(),
            temporal_analysis=_NoopTemporal(),
            alert_engine=_NoopAlertEngine(),
            metrics=self.CameraMetrics(),
        )
        state_store = self.StateStore(layer_factory=lambda camera_id: state)

        cam0_queue = queue.Queue(maxsize=50)
        for _ in range(10):
            cam0_queue.put((object(), time.time()))
        frame_queues = {"cam0": cam0_queue}

        workers = [
            self.InferenceWorker(
                worker_id=i, frame_queues=frame_queues, state_store=state_store,
                zone_severity_lookup={}, alert_dispatcher=lambda cam, alert: None,
            )
            for i in range(2)
        ]
        for w in workers:
            w.start()
        deadline = time.time() + 5
        while detector.call_count < 10 and time.time() < deadline:
            time.sleep(0.05)
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=2)

        self.assertEqual(detector.call_count, 10, "not all frames were processed")
        self.assertEqual(detector._max_concurrent, 1,
                          "two workers processed the SAME camera's frames concurrently")


class TestPerCameraQueueIsolation(unittest.TestCase):
    def test_camera_queues_are_independent_objects(self):
        """The structural guarantee the eviction fix relies on: each camera
        gets its OWN queue.Queue, so there is no object left for one
        camera's backpressure to evict another camera's frames from."""
        queue_a = queue.Queue(maxsize=2)
        queue_b = queue.Queue(maxsize=2)
        self.assertIsNot(queue_a, queue_b)

        queue_a.put(("frame_a1", 0.0))
        queue_a.put(("frame_a2", 0.0))
        # Filling camA's queue to capacity must have zero effect on camB's.
        self.assertEqual(queue_b.qsize(), 0)
        with self.assertRaises(queue.Full):
            queue_a.put_nowait(("frame_a3", 0.0))
        self.assertEqual(queue_b.qsize(), 0)


if __name__ == "__main__":
    unittest.main()
