"""
Tripwire behaviour, end to end through the real SpatialZoneLayer + AlertEngine:
every crossing is detected, only the right direction alerts (unless the wire is
two-way), jitter on the line is ignored, and the on-screen arrow always matches
what the alert logic does.
"""

import unittest

from src.alert_engine import AlertEngine
from src.config import TRIPWIRE_CFG, Tripwire
from src.layers.spatial_zones import SpatialZoneLayer, violation_direction


class _NoDwell:
    should_alert = False
    track_id = 0
    zone = None
    reason = ""


def _crossings(layer, points, track_id=1):
    """Feeds successive foot points; returns the list of directions reported."""
    out = []
    for pt in points:
        st = layer.evaluate(track_id, pt)
        if st.tripwire_crossed:
            out.append(st.tripwire_direction)
    return out


class TestArrowMatchesLogic(unittest.TestCase):
    def test_moving_along_the_arrow_is_reported_as_in_and_against_it_as_out(self):
        wires = [((0, 250), (640, 250)), ((100, 0), (100, 480)), ((0, 0), (400, 300)), ((500, 20), (50, 400))]
        for p1, p2 in wires:
            ux, uy = violation_direction(p1, p2)
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            before, after = (mx - ux * 60, my - uy * 60), (mx + ux * 60, my + uy * 60)
            layer = SpatialZoneLayer([], [Tripwire("W", p1, p2)])
            self.assertEqual(_crossings(layer, [before, after]), ["in"], f"wire {p1}->{p2}")
            layer = SpatialZoneLayer([], [Tripwire("W", p1, p2)])
            self.assertEqual(_crossings(layer, [after, before]), ["out"], f"wire {p1}->{p2}")

    def test_swapping_endpoints_reverses_the_direction(self):
        a = SpatialZoneLayer([], [Tripwire("W", (0, 250), (640, 250))])
        b = SpatialZoneLayer([], [Tripwire("W", (640, 250), (0, 250))])
        path = [(100, 400), (100, 100)]
        self.assertNotEqual(_crossings(a, path), _crossings(b, path))
        self.assertEqual(violation_direction((0, 250), (640, 250)), (0.0, -1.0))

    def test_zero_length_wire_does_not_crash(self):
        layer = SpatialZoneLayer([], [Tripwire("W", (5, 5), (5, 5))])
        self.assertEqual(_crossings(layer, [(0, 0), (10, 10)]), [])


class TestCrossingDetection(unittest.TestCase):
    def setUp(self):
        self.layer = SpatialZoneLayer([], [Tripwire("W", (0, 250), (640, 250))], tripwire_margin_px=5.0)

    def test_every_repeated_crossing_is_detected(self):
        ys = [400, 100] * 5                      # ten crossings in a row
        self.assertEqual(len(_crossings(self.layer, [(100, y) for y in ys])), 9)  # first point only sets the side
        self.layer = SpatialZoneLayer([], [Tripwire("W", (0, 250), (640, 250))], tripwire_margin_px=5.0)
        ys = [400, 100] * 5 + [400]
        self.assertEqual(len(_crossings(self.layer, [(100, y) for y in ys])), 10)

    def test_jitter_inside_the_dead_band_is_not_a_crossing(self):
        pts = [(100, 400)] + [(100, 250 + d) for d in (3, -3, 4, -4, 2, -2)]
        self.assertEqual(_crossings(self.layer, pts), [])

    def test_wobble_at_the_line_then_a_real_crossing_counts_once(self):
        pts = [(100, 400), (100, 253), (100, 247), (100, 252), (100, 100)]
        self.assertEqual(_crossings(self.layer, pts), ["in"])

    def test_a_point_exactly_on_the_line_does_not_hide_the_crossing(self):
        layer = SpatialZoneLayer([], [Tripwire("W", (0, 250), (640, 250))], tripwire_margin_px=0.0)
        self.assertEqual(_crossings(layer, [(100, 100), (100, 250), (100, 400)]), ["out"])

    def test_default_margin_comes_from_config(self):
        self.assertEqual(SpatialZoneLayer([], []).tripwire_margin_px, TRIPWIRE_CFG.margin_px)


class TestAlertingThroughTheRealPipeline(unittest.TestCase):
    def _alerts(self, wire, ys, cooldown, dt=1.0):
        layer = SpatialZoneLayer([], [wire], tripwire_margin_px=5.0)
        engine = AlertEngine(tripwire_cooldown_sec=cooldown)
        fired = []
        for i, y in enumerate(ys):
            st = layer.evaluate(1, (100, y))
            a = engine.evaluate(st, _NoDwell(), now=i * dt)
            if a:
                fired.append(a)
        return fired

    YS = [400, 100] * 5 + [400]   # ten crossings, alternating direction

    def test_one_way_wire_alerts_only_on_the_violating_direction(self):
        wire = Tripwire("W", (0, 250), (640, 250), direction_sensitive=True)
        alerts = self._alerts(wire, self.YS, cooldown=0.0)
        self.assertEqual(len(alerts), 5)
        self.assertTrue(all("inbound" in a.message for a in alerts))

    def test_two_way_wire_alerts_on_every_crossing(self):
        wire = Tripwire("W", (0, 250), (640, 250), direction_sensitive=False)
        alerts = self._alerts(wire, self.YS, cooldown=0.0)
        self.assertEqual(len(alerts), 10)
        self.assertEqual(sum("outbound" in a.message for a in alerts), 5)

    def test_cooldown_holds_back_bursts(self):
        wire = Tripwire("W", (0, 250), (640, 250), direction_sensitive=False)
        many = self._alerts(wire, self.YS, cooldown=0.0)
        few = self._alerts(wire, self.YS, cooldown=20.0)
        self.assertEqual(len(few), 1)
        self.assertGreater(len(many), len(few))

    def test_engine_default_cooldown_comes_from_config(self):
        self.assertEqual(AlertEngine().tripwire_cooldown_sec, TRIPWIRE_CFG.cooldown_sec)


if __name__ == "__main__":
    unittest.main()
