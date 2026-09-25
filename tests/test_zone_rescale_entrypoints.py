"""
Zones drawn at one resolution must land correctly at another in EVERY entry
point (single-video CLI, batch runner, live multi-camera workers), not just the
desktop app. All of them go through src/zone_runtime.py / rescale_shapes.
"""

import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

import numpy as np


def _stub_heavy_deps():
    if "ultralytics" not in sys.modules:
        m = types.ModuleType("ultralytics")
        m.YOLO = m.RTDETR = type("_M", (), {"__init__": lambda self, *a, **k: None})
        sys.modules["ultralytics"] = m
    if "torch" not in sys.modules:
        t = types.ModuleType("torch")
        t.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = t


class _NoDetections:
    def __init__(self, *a, **k):
        pass

    def process(self, frame):
        return []

    def close(self):
        pass


def _config(ref):
    from src.config import Zone, ZoneConfig
    return ZoneConfig([Zone("Door", [(100, 100), (200, 100), (200, 200)], "high")], [], ref)


class TestZoneRuntime(unittest.TestCase):
    def test_rescales_to_frame_and_builds_lookup(self):
        from src.zone_runtime import ZoneRuntime
        rt = ZoneRuntime.build(_config((640, 480)), (320, 240))
        self.assertEqual(rt.zones[0].polygon, [(50, 50), (100, 50), (100, 100)])
        self.assertEqual(rt.severity_lookup, {"Door": "high"})

    def test_no_reference_or_same_size_leaves_coordinates_alone(self):
        from src.zone_runtime import ZoneRuntime
        self.assertEqual(ZoneRuntime.build(_config(None), (320, 240)).zones[0].polygon[0], (100, 100))
        self.assertEqual(ZoneRuntime.build(_config((320, 240)), (320, 240)).zones[0].polygon[0], (100, 100))

    def test_invalid_config_raises(self):
        from src.config import Zone, ZoneConfig
        from src.zone_runtime import ZoneRuntime
        with self.assertRaises(ValueError):
            ZoneRuntime.build(ZoneConfig([Zone("Bad", [(0, 0), (1, 1)])], []), (320, 240))


class TestEntryPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _stub_heavy_deps()
        import cv2
        cls.clip = os.path.join(tempfile.gettempdir(), "_rescale_clip.mp4")
        wr = cv2.VideoWriter(cls.clip, cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 240))
        for _ in range(5):
            wr.write(np.zeros((240, 320, 3), dtype="uint8"))
        wr.release()

    def _spy(self, module):
        built = []
        real = module.ZoneRuntime

        class Spy:
            @staticmethod
            def build(cfg, size):
                rt = real.build(cfg, size)
                built.append(rt)
                return rt

        module.ZoneRuntime = Spy
        self.addCleanup(setattr, module, "ZoneRuntime", real)
        return built

    def _patch(self, module, name, value):
        old = getattr(module, name)
        setattr(module, name, value)
        self.addCleanup(setattr, module, name, old)

    def test_batch_run_rescales_per_video(self):
        import src.batch_run as br
        self._patch(br, "DetectionTrackingLayer", _NoDetections)
        self._patch(br, "ZONE_CONFIG", _config((640, 480)))
        built = self._spy(br)
        br.process_one_video(self.clip)
        self.assertEqual(len(built), 1, "zones are built once per video, not per frame")
        self.assertEqual(built[0].zones[0].polygon[0], (50, 50))

    def test_single_video_cli_rescales(self):
        import src.main as main_mod
        self._patch(main_mod, "DetectionTrackingLayer", _NoDetections)
        self._patch(main_mod, "ZONE_CONFIG", _config((640, 480)))
        built = self._spy(main_mod)
        main_mod.run(self.clip, display=False)
        self.assertEqual(built[0].zones[0].polygon[0], (50, 50))

    def test_live_worker_fits_zones_once_then_is_a_cheap_no_op(self):
        from src.layers.spatial_zones import SpatialZoneLayer
        from src.production.inference_worker import InferenceWorker
        cfg = _config(None)
        state = SimpleNamespace(spatial_zone=SpatialZoneLayer(cfg.zones, cfg.tripwires),
                                zone_frame_size=(640, 480))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        InferenceWorker._fit_zones_to_frame(state, frame)
        self.assertEqual(state.spatial_zone.zones[0].polygon[0], (50, 50))
        self.assertEqual(state.zone_frame_size, (320, 240))
        layer = state.spatial_zone
        InferenceWorker._fit_zones_to_frame(state, frame)
        self.assertIs(state.spatial_zone, layer, "same-size frames must not rebuild the layer")
        InferenceWorker._fit_zones_to_frame(state, np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(state.spatial_zone.zones[0].polygon[0], (100, 100), "resolution change is followed")

    def test_live_worker_leaves_legacy_zones_untouched(self):
        from src.layers.spatial_zones import SpatialZoneLayer
        from src.production.inference_worker import InferenceWorker
        cfg = _config(None)
        layer = SpatialZoneLayer(cfg.zones, cfg.tripwires)
        state = SimpleNamespace(spatial_zone=layer, zone_frame_size=None)
        InferenceWorker._fit_zones_to_frame(state, np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertIs(state.spatial_zone, layer)


if __name__ == "__main__":
    unittest.main()
