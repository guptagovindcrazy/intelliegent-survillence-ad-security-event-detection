"""
tests/test_temporal_analysis.py
---------------------------------
Unit tests for Layer 3's state machine: debounce, dwell thresholds, severity
lookup, and the false-alarm-suppressing transition chain. Uses a fake clock
(explicit `now` values) so timing is exact and deterministic -- no sleeping.
"""

import unittest

from src.config import TemporalConfig
from src.layers.spatial_zones import SpatialState
from src.layers.temporal_analysis import TemporalAnalysisLayer, TrackState


def make_cfg(**overrides):
    cfg = TemporalConfig()
    cfg.min_consecutive_frames_in_zone = overrides.get("debounce", 3)
    cfg.dwell_threshold_sec = overrides.get("thresholds", {"low": 8.0, "medium": 5.0, "high": 2.0})
    cfg.alert_cooldown_sec = overrides.get("cooldown", 10.0)
    cfg.stationary_speed_px_s = overrides.get("stationary_speed", 15.0)
    return cfg


class TestDebounce(unittest.TestCase):
    def setUp(self):
        self.layer = TemporalAnalysisLayer(make_cfg(debounce=3))
        self.severity = {"Zone": "high"}

    def test_dwell_does_not_start_before_debounce_satisfied(self):
        state = SpatialState(track_id=1, zones_inside=["Zone"])
        # frames 1 and 2 are below the debounce count (3) -- state should stay ZONE_ENTERED
        ev1 = self.layer.update(state, (10, 10), self.severity, now=0.0)
        ev2 = self.layer.update(state, (10, 10), self.severity, now=0.1)
        self.assertEqual(ev1.state, TrackState.ZONE_ENTERED)
        self.assertEqual(ev2.state, TrackState.ZONE_ENTERED)
        self.assertEqual(ev1.dwell_time_sec, 0.0)

    def test_dwell_starts_once_debounce_satisfied(self):
        state = SpatialState(track_id=1, zones_inside=["Zone"])
        self.layer.update(state, (10, 10), self.severity, now=0.0)
        self.layer.update(state, (10, 10), self.severity, now=0.1)
        ev3 = self.layer.update(state, (10, 10), self.severity, now=0.2)  # 3rd consecutive frame
        self.assertIn(ev3.state, (TrackState.DWELLING, TrackState.ALERT_CANDIDATE, TrackState.ALERTED))


class TestDwellThresholdAndAlert(unittest.TestCase):
    def setUp(self):
        self.layer = TemporalAnalysisLayer(make_cfg(debounce=2, thresholds={"high": 2.0}, cooldown=5.0))
        self.severity = {"Door": "high"}

    def test_no_alert_before_threshold(self):
        state = SpatialState(track_id=1, zones_inside=["Door"])
        self.layer.update(state, (0, 0), self.severity, now=0.0)
        ev = self.layer.update(state, (0, 0), self.severity, now=1.0)  # debounce satisfied, dwell=0
        self.assertFalse(ev.should_alert)

    def test_alert_fires_once_threshold_exceeded(self):
        state = SpatialState(track_id=1, zones_inside=["Door"])
        self.layer.update(state, (0, 0), self.severity, now=0.0)
        self.layer.update(state, (0, 0), self.severity, now=0.1)  # debounce satisfied here, dwell clock starts
        ev = self.layer.update(state, (0, 0), self.severity, now=3.0)  # 2.9s dwell > 2.0s threshold
        self.assertTrue(ev.should_alert)
        self.assertEqual(ev.state, TrackState.ALERTED)

    def test_cooldown_suppresses_duplicate_alert(self):
        state = SpatialState(track_id=1, zones_inside=["Door"])
        self.layer.update(state, (0, 0), self.severity, now=0.0)
        self.layer.update(state, (0, 0), self.severity, now=0.1)
        ev1 = self.layer.update(state, (0, 0), self.severity, now=3.0)
        self.assertTrue(ev1.should_alert)
        # still dwelling well past threshold, but within cooldown window -- must NOT re-alert
        ev2 = self.layer.update(state, (0, 0), self.severity, now=4.0)
        self.assertFalse(ev2.should_alert)

    def test_alert_can_fire_again_after_cooldown_and_re_entry(self):
        state = SpatialState(track_id=1, zones_inside=["Door"])
        self.layer.update(state, (0, 0), self.severity, now=0.0)
        self.layer.update(state, (0, 0), self.severity, now=0.1)
        ev1 = self.layer.update(state, (0, 0), self.severity, now=3.0)
        self.assertTrue(ev1.should_alert)

        # leaves the zone -- resets debounce/dwell state back toward NORMAL
        empty_state = SpatialState(track_id=1, zones_inside=[])
        ev_exit = self.layer.update(empty_state, (0, 0), self.severity, now=3.1)
        self.assertEqual(ev_exit.state, TrackState.NORMAL)

        # re-enters after the cooldown has elapsed (now=10.0, cooldown=5.0 from t=3.0)
        self.layer.update(state, (0, 0), self.severity, now=10.0)
        self.layer.update(state, (0, 0), self.severity, now=10.1)
        ev2 = self.layer.update(state, (0, 0), self.severity, now=13.0)
        self.assertTrue(ev2.should_alert)


class TestSeverityAwareThresholds(unittest.TestCase):
    def test_high_severity_alerts_faster_than_low_severity(self):
        cfg = make_cfg(debounce=1, thresholds={"low": 8.0, "high": 1.0}, cooldown=0.0)
        layer = TemporalAnalysisLayer(cfg)

        high_state = SpatialState(track_id=1, zones_inside=["HighZone"])
        low_state = SpatialState(track_id=2, zones_inside=["LowZone"])
        severity = {"HighZone": "high", "LowZone": "low"}

        layer.update(high_state, (0, 0), severity, now=0.0)
        ev_high = layer.update(high_state, (0, 0), severity, now=1.5)  # past 1.0s high threshold
        self.assertTrue(ev_high.should_alert)

        layer.update(low_state, (0, 0), severity, now=0.0)
        ev_low = layer.update(low_state, (0, 0), severity, now=1.5)  # well under 8.0s low threshold
        self.assertFalse(ev_low.should_alert)


if __name__ == "__main__":
    unittest.main()
