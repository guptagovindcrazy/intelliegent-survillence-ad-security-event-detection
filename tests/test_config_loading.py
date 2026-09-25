"""
tests/test_config_loading.py
------------------------------
Regression tests for a real bug found during a code audit: a malformed
config/cameras.yaml (invalid YAML syntax, or a camera entry missing required
keys) used to raise an uncaught exception at IMPORT TIME (src/config.py runs
_load_cameras_from_yaml() at module load), which would crash the entire
application before main() even started. It now degrades gracefully: a
warning is printed and the hardcoded default camera list is used instead.
"""

import os
import tempfile
import unittest

from src.config import _load_cameras_from_yaml


class TestCameraYamlLoading(unittest.TestCase):
    def _write(self, content: str) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        f.write(content)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_missing_file_returns_none(self):
        self.assertIsNone(_load_cameras_from_yaml("/no/such/path/cameras.yaml"))

    def test_valid_yaml_loads_cameras(self):
        path = self._write("cameras:\n  - camera_id: cam1\n    source: \"0\"\n")
        result = _load_cameras_from_yaml(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].camera_id, "cam1")

    def test_invalid_yaml_syntax_does_not_raise(self):
        path = self._write("this: is: not: valid: yaml: [[[\n")
        result = _load_cameras_from_yaml(path)  # must not raise
        self.assertIsNone(result)

    def test_entry_missing_source_is_skipped_not_fatal(self):
        path = self._write("cameras:\n  - camera_id: cam1\n")  # no 'source'
        result = _load_cameras_from_yaml(path)  # must not raise
        self.assertIsNone(result)

    def test_partial_valid_entries_still_load(self):
        path = self._write(
            "cameras:\n"
            "  - camera_id: cam1\n"
            "    source: \"0\"\n"
            "  - camera_id: cam2\n"  # missing source -- should be skipped, not fatal
        )
        result = _load_cameras_from_yaml(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].camera_id, "cam1")

    def test_empty_cameras_list_returns_none(self):
        path = self._write("cameras: []\n")
        self.assertIsNone(_load_cameras_from_yaml(path))


if __name__ == "__main__":
    unittest.main()
