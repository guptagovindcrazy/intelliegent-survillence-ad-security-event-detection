"""Headless tests for src/frame_geometry.py and reference-size handling."""

import os
import tempfile
import unittest

import numpy as np

from src.config import Tripwire, Zone, load_zone_config
from src.frame_geometry import (
    FrameViewTransform, downscale_to_max_dim, rescale_shapes, to_processing_frame,
)
from src.zone_editor import ZoneEditorModel


class TestDownscale(unittest.TestCase):
    def test_small_frame_is_returned_untouched(self):
        f = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertIs(downscale_to_max_dim(f, 1280), f)

    def test_large_frame_is_capped_preserving_aspect(self):
        f = np.zeros((2160, 3840, 3), dtype=np.uint8)
        out = downscale_to_max_dim(f, 1280)
        self.assertEqual(out.shape[:2], (720, 1280))

    def test_live_frames_are_never_resized(self):
        f = np.zeros((2160, 3840, 3), dtype=np.uint8)
        self.assertIs(to_processing_frame(f, True, 1280), f)
        self.assertEqual(to_processing_frame(f, False, 1280).shape[1], 1280)


class TestRescale(unittest.TestCase):
    def test_scales_x_and_y_independently_without_mutating_input(self):
        z = [Zone("Z", [(100, 100), (200, 100), (200, 200)], "medium")]
        w = [Tripwire("W", (0, 50), (640, 50), False)]
        nz, nw = rescale_shapes(z, w, (640, 480), (1280, 960))
        self.assertEqual(nz[0].polygon, [(200, 200), (400, 200), (400, 400)])
        self.assertEqual((nz[0].name, nz[0].severity), ("Z", "medium"))
        self.assertEqual((nw[0].p1, nw[0].p2, nw[0].direction_sensitive), ((0, 100), (1280, 100), False))
        self.assertEqual(z[0].polygon[0], (100, 100))  # input untouched

    def test_round_trip_is_stable_for_integer_ratios(self):
        z = [Zone("Z", [(11, 7), (90, 7), (90, 50)])]
        up, _ = rescale_shapes(z, [], (100, 100), (400, 400))
        back, _ = rescale_shapes(up, [], (400, 400), (100, 100))
        self.assertEqual(back[0].polygon, z[0].polygon)

    def test_rejects_non_positive_sizes(self):
        with self.assertRaises(ValueError):
            rescale_shapes([], [], (0, 480), (640, 480))


class TestFrameViewTransform(unittest.TestCase):
    def test_letterboxed_mapping_and_round_trip(self):
        tf = FrameViewTransform((1000, 400), (640, 480))  # pillarboxed: scale 400/480
        x, y, w, h = tf.image_rect()
        self.assertAlmostEqual(h, 400)
        self.assertAlmostEqual(x, (1000 - w) / 2)
        for pt in [(0, 0), (320, 240), (639, 479), (100, 33)]:
            wx, wy = tf.to_widget(*pt)
            self.assertEqual(tf.to_image(wx, wy), pt)

    def test_margin_clicks_return_none_unless_clamped(self):
        tf = FrameViewTransform((1000, 400), (640, 480))
        self.assertIsNone(tf.to_image(5, 200))
        self.assertEqual(tf.to_image(5, 200, clamp=True)[0], 0)

    def test_degenerate_sizes_do_not_divide_by_zero(self):
        self.assertIsNone(FrameViewTransform((0, 0), (640, 480)).to_image(1, 1))
        self.assertIsNone(FrameViewTransform((100, 100), (0, 0)).to_image(1, 1))


class TestReferenceSize(unittest.TestCase):
    def _yaml(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_loader_reads_reference_size(self):
        cfg = load_zone_config(self._yaml("reference_size: [1280, 720]\nzones: []\n"))
        self.assertEqual(cfg.reference_size, (1280, 720))

    def test_bad_reference_size_is_ignored_not_fatal(self):
        for bad in ("[0, 720]", "[1, 2, 3]", "abc"):
            self.assertIsNone(load_zone_config(self._yaml(f"reference_size: {bad}\n")).reference_size)

    def test_no_reference_size_means_none(self):
        self.assertIsNone(load_zone_config(self._yaml("zones: []\n")).reference_size)

    def test_model_rebase_rescales_existing_shapes_and_records_size(self):
        m = ZoneEditorModel([Zone("Z", [(100, 100), (200, 100), (200, 200)])], [], reference_size=(640, 480))
        m.add_point((1, 1))
        m.rebase((1280, 960))
        self.assertEqual(m.zones[0].polygon[0], (200, 200))
        self.assertEqual(m.reference_size, (1280, 960))
        self.assertEqual(m.pending, [])

    def test_rebase_without_prior_size_keeps_coordinates(self):
        m = ZoneEditorModel([Zone("Z", [(100, 100), (200, 100), (200, 200)])], [])
        m.rebase((1280, 960))
        self.assertEqual(m.zones[0].polygon[0], (100, 100))
        self.assertEqual(m.reference_size, (1280, 960))

    def test_reference_size_survives_save_and_reload(self):
        m = ZoneEditorModel()
        m.rebase((1280, 720))
        f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        m.save(f.name)
        self.assertEqual(ZoneEditorModel.from_yaml(f.name).reference_size, (1280, 720))

    def test_remove_at_and_to_config_copy(self):
        m = ZoneEditorModel([Zone("A", [(0, 0), (1, 0), (1, 1)]), Zone("B", [(0, 0), (2, 0), (2, 2)])], [])
        self.assertEqual(m.remove_at("zone", 0), "A")
        self.assertIsNone(m.remove_at("zone", 5))
        cfg = m.to_config()
        cfg.zones[0].polygon.append((9, 9))
        self.assertEqual(len(m.zones[0].polygon), 3)


if __name__ == "__main__":
    unittest.main()
