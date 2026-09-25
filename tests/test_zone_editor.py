"""Headless tests for the draw_zones.py editing model and overlay renderer."""

import os
import tempfile
import unittest

import numpy as np

from src.config import _load_zones_from_yaml
from src.zone_editor import ZoneEditorModel, render_overlay


class TestZoneEditorModel(unittest.TestCase):
    def test_zone_needs_three_points(self):
        m = ZoneEditorModel()
        m.add_point((0, 0)); m.add_point((10, 0))
        self.assertFalse(m.can_finish())
        m.add_point((10, 10))
        self.assertTrue(m.can_finish())

    def test_commit_adds_zone_and_clears_pending(self):
        m = ZoneEditorModel()
        for p in [(0, 0), (10, 0), (10, 10)]:
            m.add_point(p)
        m.cycle_severity()  # high -> low
        self.assertIsNone(m.commit("Door"))
        self.assertEqual(m.zones[0].name, "Door")
        self.assertEqual(m.zones[0].severity, "low")
        self.assertEqual(m.pending, [])
        self.assertTrue(m.dirty)

    def test_blank_and_duplicate_names_rejected_and_points_kept(self):
        m = ZoneEditorModel()
        for p in [(0, 0), (10, 0), (10, 10)]:
            m.add_point(p)
        self.assertIsNotNone(m.commit("   "))
        self.assertEqual(m.commit("A"), None)
        for p in [(0, 0), (10, 0), (10, 10)]:
            m.add_point(p)
        self.assertIn("Duplicate", m.commit("A"))
        self.assertEqual(len(m.pending), 3)
        self.assertEqual(len(m.zones), 1)

    def test_tripwire_caps_at_two_points(self):
        m = ZoneEditorModel()
        m.set_mode("tripwire")
        for p in [(0, 5), (20, 5), (30, 5)]:
            m.add_point(p)
        self.assertEqual(len(m.pending), 2)
        self.assertTrue(m.can_finish())
        m.toggle_direction()
        self.assertIsNone(m.commit("Line"))
        self.assertFalse(m.tripwires[0].direction_sensitive)

    def test_mode_switch_discards_pending_and_undo_delete(self):
        m = ZoneEditorModel()
        m.add_point((1, 1))
        m.set_mode("tripwire")
        self.assertEqual(m.pending, [])
        m.add_point((0, 0)); m.add_point((5, 5))
        m.commit("W")
        self.assertEqual(m.delete_last(), "W")
        self.assertIsNone(m.delete_last())

    def test_save_roundtrips_through_config_loader(self):
        m = ZoneEditorModel()
        for p in [(0, 0), (10, 0), (10, 10)]:
            m.add_point(p)
        m.commit("Door")
        m.set_mode("tripwire")
        m.add_point((0, 5)); m.add_point((20, 5))
        m.commit("Line")
        f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        m.save(f.name)
        self.assertFalse(m.dirty)
        zones, wires = _load_zones_from_yaml(f.name)
        self.assertEqual(zones[0].polygon, [(0, 0), (10, 0), (10, 10)])
        self.assertEqual(wires[0].p2, (20, 5))
        # and the editor can re-open what it wrote
        again = ZoneEditorModel.from_yaml(f.name)
        self.assertEqual([z.name for z in again.zones], ["Door"])

    def test_render_overlay_runs_and_does_not_mutate_input(self):
        m = ZoneEditorModel()
        for p in [(10, 10), (100, 10), (100, 100)]:
            m.add_point(p)
        m.commit("Z")
        m.add_point((5, 5))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = render_overlay(frame, m, naming="ab", message="hi")
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(int(frame.sum()), 0)
        self.assertGreater(int(out.sum()), 0)


class TestBoxDrawing(unittest.TestCase):
    def test_set_box_orders_corners_whatever_the_drag_direction(self):
        m = ZoneEditorModel()
        self.assertTrue(m.set_box((100, 80), (20, 10)))
        self.assertEqual(m.pending, [(20, 10), (100, 10), (100, 80), (20, 80)])
        self.assertTrue(m.can_finish())
        self.assertIsNone(m.commit("Box"))
        self.assertEqual(len(m.zones[0].polygon), 4)

    def test_flat_box_is_refused_and_pending_untouched(self):
        m = ZoneEditorModel()
        m.add_point((1, 1))
        self.assertFalse(m.set_box((5, 5), (50, 5)))
        self.assertFalse(m.set_box((5, 5), (5, 50)))
        self.assertEqual(m.pending, [(1, 1)])

    def test_changing_zone_shape_clears_pending_and_validates_the_value(self):
        m = ZoneEditorModel()
        m.add_point((1, 1))
        m.set_zone_shape("box")
        self.assertEqual(m.pending, [])
        m.set_zone_shape("box")            # same shape: nothing to clear
        with self.assertRaises(ValueError):
            m.set_zone_shape("circle")


class TestVertexEditing(unittest.TestCase):
    def _model(self):
        from src.config import Tripwire, Zone
        return ZoneEditorModel([Zone("Z", [(10, 10), (100, 10), (100, 100)])],
                               [Tripwire("W", (0, 50), (200, 50))])

    def test_move_zone_vertex_marks_dirty(self):
        m = self._model()
        self.assertTrue(m.move_vertex("zone", 0, 1, (120, 20)))
        self.assertEqual(m.zones[0].polygon[1], (120, 20))
        self.assertTrue(m.dirty)

    def test_move_to_same_spot_is_a_no_op(self):
        m = self._model()
        self.assertFalse(m.move_vertex("zone", 0, 0, (10, 10)))
        self.assertFalse(m.dirty)

    def test_tripwire_cannot_collapse_to_zero_length(self):
        m = self._model()
        self.assertFalse(m.move_vertex("tripwire", 0, 0, (200, 50)))
        self.assertEqual(m.tripwires[0].p1, (0, 50))
        self.assertTrue(m.move_vertex("tripwire", 0, 1, (250, 60)))
        self.assertEqual(m.tripwires[0].p2, (250, 60))

    def test_flip_tripwire_swaps_endpoints_and_is_reversible(self):
        m = self._model()
        self.assertTrue(m.flip_tripwire(0))
        self.assertEqual((m.tripwires[0].p1, m.tripwires[0].p2), ((200, 50), (0, 50)))
        self.assertTrue(m.dirty)
        m.flip_tripwire(0)
        self.assertEqual(m.tripwires[0].p1, (0, 50))
        self.assertFalse(m.flip_tripwire(4))

    def test_out_of_range_indices_are_refused(self):
        m = self._model()
        self.assertFalse(m.move_vertex("zone", 3, 0, (1, 1)))
        self.assertFalse(m.move_vertex("zone", 0, 9, (1, 1)))
        self.assertFalse(m.move_vertex("tripwire", 0, 2, (1, 1)))

    def test_nearest_vertex_uses_screen_distance_and_radius(self):
        from src.frame_geometry import FrameViewTransform
        from src.zone_editor import nearest_vertex
        m = self._model()
        tf = FrameViewTransform((400, 400), (200, 200))  # 2x zoom
        wx, wy = tf.to_widget(100, 10)
        self.assertEqual(nearest_vertex(m, tf, wx + 3, wy - 2), ("zone", 0, 1))
        self.assertIsNone(nearest_vertex(m, tf, wx + 30, wy))
        wx, wy = tf.to_widget(200, 50)
        self.assertEqual(nearest_vertex(m, tf, wx, wy), ("tripwire", 0, 1))


if __name__ == "__main__":
    unittest.main()
