"""
tests/test_spatial_zones.py
-----------------------------
Unit tests for Layer 2's pure geometry: zone containment and tripwire-crossing
detection (including direction). No video, no model, no I/O -- just the math.
"""

import unittest

from src.config import Tripwire, Zone
from src.layers.spatial_zones import SpatialZoneLayer


class TestZoneContainment(unittest.TestCase):
    def setUp(self):
        self.zone = Zone(name="TestZone", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], severity="high")
        self.layer = SpatialZoneLayer(zones=[self.zone], tripwires=[])

    def test_point_clearly_inside(self):
        state = self.layer.evaluate(track_id=1, foot_point=(50, 50))
        self.assertIn("TestZone", state.zones_inside)

    def test_point_clearly_outside(self):
        state = self.layer.evaluate(track_id=1, foot_point=(500, 500))
        self.assertNotIn("TestZone", state.zones_inside)

    def test_point_on_boundary_counts_as_inside(self):
        # cv2.pointPolygonTest returns 0 (not negative) exactly on the edge --
        # the layer treats that as inside, per _point_in_zone's `result >= 0`.
        state = self.layer.evaluate(track_id=1, foot_point=(0, 50))
        self.assertIn("TestZone", state.zones_inside)

    def test_multiple_zones_independent(self):
        zone_b = Zone(name="OtherZone", polygon=[(200, 200), (300, 200), (300, 300), (200, 300)], severity="low")
        layer = SpatialZoneLayer(zones=[self.zone, zone_b], tripwires=[])
        state = layer.evaluate(track_id=1, foot_point=(250, 250))
        self.assertIn("OtherZone", state.zones_inside)
        self.assertNotIn("TestZone", state.zones_inside)


class TestTripwireCrossing(unittest.TestCase):
    def setUp(self):
        # horizontal tripwire at y=250
        self.wire = Tripwire(name="Gate", p1=(0, 250), p2=(640, 250), direction_sensitive=True)
        self.layer = SpatialZoneLayer(zones=[], tripwires=[self.wire])

    def test_no_crossing_on_first_observation(self):
        # first-ever observation has nothing to compare against -- must not
        # spuriously report a crossing
        state = self.layer.evaluate(track_id=1, foot_point=(100, 100))
        self.assertIsNone(state.tripwire_crossed)

    def test_crossing_detected_top_to_bottom(self):
        self.layer.evaluate(track_id=1, foot_point=(100, 100))   # above the line
        state = self.layer.evaluate(track_id=1, foot_point=(100, 400))  # now below it
        self.assertEqual(state.tripwire_crossed, "Gate")
        self.assertIsNotNone(state.tripwire_direction)

    def test_no_crossing_when_staying_on_same_side(self):
        self.layer.evaluate(track_id=1, foot_point=(100, 100))
        state = self.layer.evaluate(track_id=1, foot_point=(100, 120))  # still above
        self.assertIsNone(state.tripwire_crossed)

    def test_opposite_crossings_report_opposite_directions(self):
        # track 1 crosses top -> bottom
        self.layer.evaluate(track_id=1, foot_point=(100, 100))
        state_a = self.layer.evaluate(track_id=1, foot_point=(100, 400))
        # track 2 crosses bottom -> top
        self.layer.evaluate(track_id=2, foot_point=(100, 400))
        state_b = self.layer.evaluate(track_id=2, foot_point=(100, 100))
        self.assertNotEqual(state_a.tripwire_direction, state_b.tripwire_direction)

    def test_independent_tracks_do_not_interfere(self):
        # track 1 establishes a side; track 2 crossing shouldn't be affected by
        # track 1's history (per-track side cache, keyed by (track_id, wire.name))
        self.layer.evaluate(track_id=1, foot_point=(100, 100))
        state = self.layer.evaluate(track_id=2, foot_point=(100, 400))
        self.assertIsNone(state.tripwire_crossed)  # track 2's first observation


if __name__ == "__main__":
    unittest.main()
