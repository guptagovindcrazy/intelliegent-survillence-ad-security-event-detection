"""Tests for config/zones.yaml loading: empty by default, malformed entries skipped."""

import os
import tempfile
import unittest

from src.config import RESTRICTED_ZONES, TRIPWIRES, ZONES_YAML_PATH, _load_zones_from_yaml


class TestZonesYamlLoading(unittest.TestCase):
    def _write(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        f.write(content)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_shipped_default_is_empty(self):
        self.assertEqual(_load_zones_from_yaml(ZONES_YAML_PATH), ([], []))
        self.assertEqual(RESTRICTED_ZONES, [])
        self.assertEqual(TRIPWIRES, [])

    def test_missing_file_is_empty(self):
        self.assertEqual(_load_zones_from_yaml("/no/such/zones.yaml"), ([], []))

    def test_valid_file_loads(self):
        path = self._write(
            "zones:\n"
            "  - name: Door\n"
            "    severity: medium\n"
            "    polygon: [[0, 0], [10, 0], [10, 10]]\n"
            "tripwires:\n"
            "  - name: Line\n"
            "    p1: [0, 5]\n"
            "    p2: [20, 5]\n"
            "    direction_sensitive: false\n"
        )
        zones, wires = _load_zones_from_yaml(path)
        self.assertEqual(zones[0].name, "Door")
        self.assertEqual(zones[0].severity, "medium")
        self.assertEqual(zones[0].polygon, [(0, 0), (10, 0), (10, 10)])
        self.assertEqual(wires[0].p2, (20, 5))
        self.assertFalse(wires[0].direction_sensitive)

    def test_invalid_yaml_does_not_raise(self):
        path = self._write("this: is: not: valid: yaml: [[[\n")
        self.assertEqual(_load_zones_from_yaml(path), ([], []))

    def test_malformed_entries_skipped_good_ones_kept(self):
        path = self._write(
            "zones:\n"
            "  - name: NoPolygon\n"
            "  - name: BadPoint\n"
            "    polygon: [[1, 2, 3], [4, 5], [6, 7]]\n"
            "  - name: Good\n"
            "    polygon: [[0, 0], [5, 0], [5, 5]]\n"
            "tripwires:\n"
            "  - name: NoEnds\n"
        )
        zones, wires = _load_zones_from_yaml(path)
        self.assertEqual([z.name for z in zones], ["Good"])
        self.assertEqual(wires, [])


if __name__ == "__main__":
    unittest.main()
