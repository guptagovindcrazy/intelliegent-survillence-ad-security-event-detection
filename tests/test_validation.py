"""
tests/test_validation.py
--------------------------
Unit tests for src/validation.py -- what ERR_ZONE_CONFIG_400 actually checks.
"""

import unittest

from src.config import Tripwire, Zone
from src.validation import assert_valid_or_raise, validate_zones_and_tripwires


def zone(name="Z", polygon=None, severity="high"):
    return Zone(name=name, polygon=polygon or [(0, 0), (1, 0), (1, 1)], severity=severity)


def wire(name="W", p1=(0, 0), p2=(1, 1)):
    return Tripwire(name=name, p1=p1, p2=p2)


class TestZoneValidation(unittest.TestCase):
    def test_valid_config_has_no_problems(self):
        problems = validate_zones_and_tripwires([zone()], [wire()])
        self.assertEqual(problems, [])

    def test_too_few_polygon_points_is_flagged(self):
        problems = validate_zones_and_tripwires([zone(polygon=[(0, 0), (1, 1)])], [])
        self.assertTrue(any("at least 3" in p for p in problems))

    def test_blank_zone_name_is_flagged(self):
        problems = validate_zones_and_tripwires([zone(name="")], [])
        self.assertTrue(any("blank name" in p for p in problems))

    def test_duplicate_zone_names_flagged(self):
        problems = validate_zones_and_tripwires([zone(name="A"), zone(name="A")], [])
        self.assertTrue(any("Duplicate zone name" in p for p in problems))

    def test_unknown_severity_flagged(self):
        problems = validate_zones_and_tripwires([zone(severity="urgent")], [])
        self.assertTrue(any("unknown severity" in p for p in problems))


class TestTripwireValidation(unittest.TestCase):
    def test_degenerate_tripwire_is_flagged(self):
        problems = validate_zones_and_tripwires([], [wire(p1=(5, 5), p2=(5, 5))])
        self.assertTrue(any("identical endpoints" in p for p in problems))

    def test_duplicate_tripwire_names_flagged(self):
        problems = validate_zones_and_tripwires([], [wire(name="G"), wire(name="G", p1=(2, 2), p2=(9, 9))])
        self.assertTrue(any("Duplicate tripwire name" in p for p in problems))


class TestAssertValidOrRaise(unittest.TestCase):
    def test_raises_on_invalid_config(self):
        with self.assertRaises(ValueError):
            assert_valid_or_raise([zone(polygon=[(0, 0)])], [])

    def test_does_not_raise_on_valid_config(self):
        try:
            assert_valid_or_raise([zone()], [wire()])
        except ValueError:
            self.fail("assert_valid_or_raise raised ValueError on a valid config")


if __name__ == "__main__":
    unittest.main()
