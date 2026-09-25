"""
"Alert on entry" zones: stepping into the zone raises an alert immediately (no
time limit), jitter on the boundary doesn't repeat it, and a zone is EITHER
entry-alerted OR time-limited, never both.
"""

import os
import tempfile
import unittest

import numpy as np

from src.alert_engine import AlertEngine
from src.config import TEMPORAL_CFG, Tripwire, Zone, load_zone_config
from src.frame_geometry import rescale_shapes
from src.layers.spatial_zones import SpatialState, SpatialZoneLayer
from src.layers.temporal_analysis import TemporalAnalysisLayer
from src.zone_editor import ZoneEditorModel

BOX = [(100, 100), (300, 100), (300, 300), (100, 300)]


def _layer(entry=True, margin=5.0):
    return SpatialZoneLayer([Zone("Window", BOX, "high", alert_on_entry=entry)], [], entry_margin_px=margin)


def _entries(layer, points, track_id=1):
    return [layer.evaluate(track_id, pt).zones_entered for pt in points]


class TestEntryDetection(unittest.TestCase):
    def test_stepping_in_fires_once_not_every_frame(self):
        seen = _entries(_layer(), [(50, 200), (150, 200), (200, 200), (250, 200)])
        self.assertEqual(seen, [[], ["Window"], [], []])

    def test_first_seen_already_inside_counts_as_an_entry(self):
        self.assertEqual(_entries(_layer(), [(200, 200), (210, 200)]), [["Window"], []])

    def test_wobble_on_the_boundary_is_not_repeated_entries(self):
        pts = [(50, 200)] + [(100 + d, 200) for d in (3, -3, 4, -2, 3, -4)]
        self.assertEqual(sum(bool(e) for e in _entries(_layer(), pts)), 0)

    def test_leaving_and_coming_back_is_a_new_entry(self):
        seen = _entries(_layer(), [(50, 200), (200, 200), (50, 200), (200, 200)])
        self.assertEqual([bool(e) for e in seen], [False, True, False, True])

    def test_each_person_is_tracked_separately(self):
        layer = _layer()
        layer.evaluate(1, (50, 200))
        layer.evaluate(2, (50, 200))
        self.assertEqual(layer.evaluate(1, (200, 200)).zones_entered, ["Window"])
        self.assertEqual(layer.evaluate(2, (200, 200)).zones_entered, ["Window"])

    def test_ordinary_zones_never_report_entries(self):
        self.assertEqual(_entries(_layer(entry=False), [(50, 200), (200, 200)]), [[], []])

    def test_entry_zone_is_visible_as_inside_but_not_timed(self):
        st = _layer().evaluate(1, (200, 200))
        self.assertEqual(st.zones_inside, ["Window"])
        self.assertEqual(st.dwell_zones, [])
        st = _layer(entry=False).evaluate(1, (200, 200))
        self.assertEqual(st.dwell_zones, ["Window"])

    def test_default_margin_comes_from_config(self):
        from src.config import ENTRY_CFG
        self.assertEqual(SpatialZoneLayer([], []).entry_margin_px, ENTRY_CFG.margin_px)

    def test_overlay_labels_an_entry_zone_without_error(self):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        _layer().draw_overlays(frame)
        self.assertGreater(int(frame.sum()), 0)


class TestEntryAlerts(unittest.TestCase):
    def _run(self, zone_entry, points, cooldown=5.0, dt=1.0):
        layer = _layer(entry=zone_entry)
        tl = TemporalAnalysisLayer(TEMPORAL_CFG)
        engine = AlertEngine(entry_cooldown_sec=cooldown)
        lookup = {"Window": "high"}
        alerts = []
        for i, pt in enumerate(points):
            st = layer.evaluate(1, pt)
            ev = tl.update(st, pt, lookup, now=i * dt)
            a = engine.evaluate(st, ev, now=i * dt)
            if a:
                alerts.append((i, a))
        return alerts

    def test_entry_zone_alerts_on_the_step_in_frame(self):
        alerts = self._run(True, [(50, 200), (200, 200), (200, 200)])
        self.assertEqual([(i, a.alert_type) for i, a in alerts], [(1, "ZONE_ENTRY_VIOLATION")])
        self.assertIn("entered restricted zone 'Window'", alerts[0][1].message)

    def test_entry_zone_never_also_raises_a_dwell_alert(self):
        alerts = self._run(True, [(50, 200)] + [(200, 200)] * 40)   # stays 40 s
        self.assertEqual({a.alert_type for _, a in alerts}, {"ZONE_ENTRY_VIOLATION"})

    def test_time_limit_zone_still_needs_the_time_limit(self):
        alerts = self._run(False, [(50, 200)] + [(200, 200)] * 20)
        types = {a.alert_type for _, a in alerts}
        self.assertEqual(types, {"DWELL_VIOLATION"})
        self.assertGreater(alerts[0][0], 3)          # not on the step-in frame

    def test_a_quick_step_in_and_out_still_alerts_for_entry_zones_only(self):
        quick = [(50, 200), (200, 200), (50, 200)]
        self.assertEqual(len(self._run(True, quick)), 1)
        self.assertEqual(len(self._run(False, quick)), 0)

    def test_cooldown_holds_back_rapid_re_entries(self):
        path = [(50, 200), (200, 200)] * 4
        self.assertEqual(len(self._run(True, path, cooldown=0.0)), 4)
        self.assertEqual(len(self._run(True, path, cooldown=60.0)), 1)

    def test_entry_alert_beats_a_tripwire_crossed_on_the_same_frame(self):
        engine = AlertEngine()
        st = SpatialState(track_id=1, zones_entered=["Window"], tripwire_crossed="Wall", tripwire_direction="in")
        ev = type("E", (), {"should_alert": False, "track_id": 1, "zone": None, "reason": ""})()
        self.assertEqual(engine.evaluate(st, ev, now=0.0).alert_type, "ZONE_ENTRY_VIOLATION")


class TestConfigAndEditorRoundTrip(unittest.TestCase):
    def _yaml(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_loader_reads_the_flag_and_defaults_to_false(self):
        cfg = load_zone_config(self._yaml(
            "zones:\n"
            "  - {name: A, polygon: [[0,0],[9,0],[9,9]], alert_on_entry: true}\n"
            "  - {name: B, polygon: [[0,0],[9,0],[9,9]]}\n"))
        self.assertEqual([z.alert_on_entry for z in cfg.zones], [True, False])

    def test_a_non_boolean_flag_skips_that_zone_with_a_warning_not_a_crash(self):
        cfg = load_zone_config(self._yaml(
            "zones:\n"
            "  - {name: Bad, polygon: [[0,0],[9,0],[9,9]], alert_on_entry: 'nope'}\n"
            "  - {name: Good, polygon: [[0,0],[9,0],[9,9]]}\n"))
        self.assertEqual([z.name for z in cfg.zones], ["Good"])

    def test_rescaling_keeps_the_flag(self):
        z, _ = rescale_shapes([Zone("W", BOX, "high", True)], [], (640, 480), (320, 240))
        self.assertTrue(z[0].alert_on_entry)

    def test_editor_saves_the_flag_only_when_set_and_reloads_it(self):
        m = ZoneEditorModel()
        m.alert_on_entry = True
        self.assertTrue(m.set_box((10, 10), (90, 90)))
        self.assertIsNone(m.commit("Window"))
        m.alert_on_entry = False
        m.set_box((100, 100), (190, 190))
        self.assertIsNone(m.commit("Yard"))
        import yaml
        written = {z["name"]: z for z in yaml.safe_load(m.to_yaml())["zones"]}
        self.assertIs(written["Window"]["alert_on_entry"], True)
        self.assertNotIn("alert_on_entry", written["Yard"], "ordinary zones stay uncluttered")
        f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        f.close()
        self.addCleanup(os.unlink, f.name)
        m.save(f.name)
        again = ZoneEditorModel.from_yaml(f.name)
        self.assertEqual({z.name: z.alert_on_entry for z in again.zones}, {"Window": True, "Yard": False})
        self.assertTrue(again.to_config().zones[0].alert_on_entry)


class TestReportsAndLabels(unittest.TestCase):
    def _summary(self, *types):
        from desktop_app.models import AlertRecord, RunSummary
        alerts = [AlertRecord(frame_index=i, video_time_sec=float(i), track_id=1, alert_type=t,
                              zone_or_wire="Window", message="m") for i, t in enumerate(types, 1)]
        return RunSummary(source="cam", is_live=True, alerts=alerts,
                          zones=[Zone("Window", BOX, "high", True), Zone("Yard", BOX, "low", False)],
                          tripwires=[Tripwire("Wall", (0, 0), (9, 9))])

    def test_report_describes_entry_alerts_and_entry_zone_rules(self):
        from desktop_app.report_generator import build_html_report
        html = build_html_report(self._summary("ZONE_ENTRY_VIOLATION", "DWELL_VIOLATION"))
        self.assertIn("zone entry violations", html)
        self.assertIn("Zone entry violation", html)
        self.assertIn("badge entry", html)
        self.assertIn("alert the moment someone enters", html)
        self.assertIn("8.0s", html)          # the ordinary zone still shows its time limit

    def test_unknown_alert_types_still_render_instead_of_crashing(self):
        from desktop_app.models import alert_type_info
        from desktop_app.report_generator import build_html_report
        self.assertEqual(alert_type_info("BAG_LEFT_BEHIND").short, "Bag Left Behind")
        self.assertIn("Bag Left Behind", build_html_report(self._summary("BAG_LEFT_BEHIND")))

    def test_each_known_type_has_a_distinct_icon_and_colour(self):
        from desktop_app.models import ALERT_TYPES
        self.assertEqual(len({i.icon for i in ALERT_TYPES.values()}), len(ALERT_TYPES))
        self.assertEqual(len({i.css for i in ALERT_TYPES.values()}), len(ALERT_TYPES))


if __name__ == "__main__":
    unittest.main()
