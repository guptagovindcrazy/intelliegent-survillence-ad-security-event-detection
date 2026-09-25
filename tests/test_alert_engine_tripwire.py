"""
tests/test_alert_engine_tripwire.py
-------------------------------------
Regression test for: AlertEngine declared a _tripwire_alerted_recently dict
but never used it, so a track whose foot-point jittered back and forth
across a tripwire line fired a fresh TRIPWIRE_VIOLATION on every crossing,
with zero debounce (unlike dwell violations, which already had a cooldown).
"""

import unittest

from src.alert_engine import AlertEngine


class _FakeSpatialState:
    def __init__(self, track_id, tripwire_crossed, tripwire_direction):
        self.track_id = track_id
        self.tripwire_crossed = tripwire_crossed
        self.tripwire_direction = tripwire_direction
        self.zones_inside = []


class _NoDwellEvent:
    should_alert = False
    track_id = 0
    zone = None
    reason = ""


class TestTripwireCooldown(unittest.TestCase):
    def setUp(self):
        self.engine = AlertEngine(tripwire_cooldown_sec=5.0)
        self.te = _NoDwellEvent()

    def _cross(self, track_id, wire, now):
        ss = _FakeSpatialState(track_id=track_id, tripwire_crossed=wire, tripwire_direction="in")
        return self.engine.evaluate(ss, self.te, now=now)

    def test_repeated_jitter_crossings_within_cooldown_fire_once(self):
        alerts = [self._cross(7, "EntryGate", t) for t in (0.0, 0.2, 0.5, 1.0)]
        fired = [a for a in alerts if a is not None]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].alert_type, "TRIPWIRE_VIOLATION")

    def test_new_crossing_after_cooldown_elapses_fires_again(self):
        self._cross(7, "EntryGate", 0.0)
        second = self._cross(7, "EntryGate", 10.0)  # well past the 5s cooldown
        self.assertIsNotNone(second)

    def test_cooldown_is_scoped_per_track_not_global(self):
        self._cross(7, "EntryGate", 0.0)
        other_track = self._cross(99, "EntryGate", 0.1)  # different track, same wire, same instant
        self.assertIsNotNone(other_track)

    def test_cooldown_is_scoped_per_wire_not_global(self):
        self._cross(7, "EntryGate", 0.0)
        other_wire = self._cross(7, "SideDoor", 0.1)  # same track, different wire
        self.assertIsNotNone(other_wire)

    def test_outbound_crossings_are_never_alerted_cooldown_or_not(self):
        ss = _FakeSpatialState(track_id=1, tripwire_crossed="EntryGate", tripwire_direction="out")
        self.assertIsNone(self.engine.evaluate(ss, self.te, now=0.0))


if __name__ == "__main__":
    unittest.main()
