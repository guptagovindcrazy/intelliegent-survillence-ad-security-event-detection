"""
tests/test_report_generator.py
---------------------------------
Unit tests for desktop_app/report_generator.py. This module has no
ultralytics/torch/PySide6 dependency (see desktop_app/models.py), so these
tests run in the same lightweight environment as the rest of the suite.
"""

import unittest

from desktop_app.models import AlertRecord, RunSummary
from desktop_app.report_generator import build_html_report


class TestReportGenerator(unittest.TestCase):
    def test_report_with_no_alerts_says_no_concerns(self):
        summary = RunSummary(source="cam.mp4", is_live=False, frames_processed=100, duration_sec=5.0)
        html = build_html_report(summary)
        self.assertIn("No security concerns were found", html)
        self.assertNotIn("None", html)  # Optional[...] fields must never leak Python's None into the page

    def test_report_with_alerts_includes_each_one(self):
        summary = RunSummary(
            source="cam.mp4", is_live=False, frames_processed=500, duration_sec=20.0,
            alerts=[
                AlertRecord(10, 1.0, 3, "DWELL_VIOLATION", "ServerRoomDoor", "Track 3 dwelled 3.2s", "high"),
                AlertRecord(50, 5.0, 7, "TRIPWIRE_VIOLATION", "MainGateLine", "Track 7 crossed inbound", "high"),
            ],
        )
        html = build_html_report(summary)
        self.assertIn("ServerRoomDoor", html)
        self.assertIn("MainGateLine", html)
        self.assertIn("Track #3", html)
        self.assertIn("Track #7", html)
        self.assertIn("2 alert(s)", html)

    def test_report_on_a_failed_run_shows_the_error_not_a_crash(self):
        summary = RunSummary(source="bad.mp4", is_live=False, error="Could not open video source:\nbad.mp4")
        html = build_html_report(summary)  # must not raise
        self.assertIn("did not complete normally", html)
        self.assertIn("bad.mp4", html)

    def test_html_special_characters_in_messages_are_escaped(self):
        # A message containing HTML-like text must not break the page or inject markup.
        summary = RunSummary(
            source="cam.mp4", is_live=False, frames_processed=10, duration_sec=1.0,
            alerts=[AlertRecord(1, 0.1, 1, "DWELL_VIOLATION", "<script>Zone</script>",
                                 "<b>malicious</b> message & \"quotes\"", "high")],
        )
        html = build_html_report(summary)
        self.assertNotIn("<script>Zone</script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_missing_snapshot_does_not_crash_report(self):
        # snapshot_png is Optional -- an alert with no snapshot must still render.
        summary = RunSummary(
            source="cam.mp4", is_live=False, frames_processed=10, duration_sec=1.0,
            alerts=[AlertRecord(1, 0.1, 1, "DWELL_VIOLATION", "Zone", "message", "high", snapshot_png=None)],
        )
        html = build_html_report(summary)  # must not raise
        self.assertIn("Zone", html)

    def test_live_source_labelled_differently_from_file_source(self):
        live_html = build_html_report(RunSummary(source="0", is_live=True))
        file_html = build_html_report(RunSummary(source="video.mp4", is_live=False))
        self.assertIn("Live camera", live_html)
        self.assertIn("Recorded video file", file_html)


if __name__ == "__main__":
    unittest.main()


class TestReportAccessibilityAndSize(unittest.TestCase):
    """Alt text, image size, contrast, mobile layout and favicon/meta tags."""

    def _report_with_snapshot(self, w=1280, h=720, alert_type="DWELL_VIOLATION"):
        import io
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (30, 60, 90)).save(buf, format="PNG")
        record = AlertRecord(frame_index=42, video_time_sec=12.5, track_id=3, alert_type=alert_type,
                             zone_or_wire="ServerRoom", message="m", snapshot_png=buf.getvalue())
        summary = RunSummary(source="cam.mp4", is_live=False, alerts=[record])
        return build_html_report(summary), record

    def test_snapshot_alt_text_describes_the_alert_not_generic(self):
        html, record = self._report_with_snapshot()
        self.assertIn("alt='", html)
        self.assertNotIn("alt='snapshot'", html, "alt text must describe the alert, not be a placeholder")
        self.assertIn("track #3", html.lower())
        self.assertIn("servername" if False else "serverroom", html.lower())

    def test_large_snapshot_is_shrunk_and_recompressed_smaller(self):
        html, record = self._report_with_snapshot(1920, 1080)
        import base64
        self.assertIn("data:image/jpeg;base64,", html)
        b64 = html.split("data:image/jpeg;base64,", 1)[1].split("'", 1)[0]
        encoded = base64.b64decode(b64)
        self.assertLess(len(encoded), len(record.snapshot_png))

        from PIL import Image
        import io
        img = Image.open(io.BytesIO(encoded))
        self.assertLessEqual(max(img.size), 480)

    def test_small_snapshot_is_not_upscaled(self):
        html, record = self._report_with_snapshot(200, 150)
        import base64, io
        from PIL import Image
        b64 = html.split("data:image/jpeg;base64,", 1)[1].split("'", 1)[0]
        img = Image.open(io.BytesIO(base64.b64decode(b64)))
        self.assertEqual(img.size, (200, 150))

    def test_missing_pillow_falls_back_to_the_original_png(self):
        import sys
        import desktop_app.report_generator as rg
        real_import = __import__("builtins").__import__

        def blocked(name, *a, **k):
            if name == "PIL":
                raise ImportError("blocked for test")
            return real_import(name, *a, **k)

        import builtins
        builtins.__import__ = blocked
        try:
            mime, data = rg._compress_snapshot(b"not a real png but bytes")
        finally:
            builtins.__import__ = real_import
        self.assertEqual((mime, data), ("image/png", b"not a real png but bytes"))

    def test_page_has_mobile_viewport_meta_description_and_favicon(self):
        html = build_html_report(RunSummary(source="cam.mp4", is_live=False))
        self.assertIn('name="viewport"', html)
        self.assertIn('name="description"', html)
        self.assertIn("rel=\"icon\"", html)
        self.assertIn('lang="en"', html)

    def test_wide_tables_are_wrapped_for_horizontal_scroll_not_page_overflow(self):
        record = AlertRecord(frame_index=1, video_time_sec=1.0, track_id=1,
                             alert_type="TRIPWIRE_VIOLATION", zone_or_wire="Gate", message="m")
        html = build_html_report(RunSummary(source="cam.mp4", is_live=False, alerts=[record]))
        self.assertIn('class="table-scroll"', html)

    def test_badge_colors_meet_aa_contrast_against_white_text(self):
        import re
        def relative_luminance(hexcolor):
            hexcolor = hexcolor.lstrip("#")
            r, g, b = (int(hexcolor[i:i + 2], 16) / 255 for i in (0, 2, 4))
            def lin(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

        def contrast(hex1, hex2):
            l1, l2 = sorted((relative_luminance(hex1), relative_luminance(hex2)))
            return (l2 + 0.05) / (l1 + 0.05)

        from desktop_app.report_generator import CSS
        vars_ = dict(re.findall(r"--([\w-]+):\s*(#[0-9A-Fa-f]{6});", CSS))
        badge_rules = re.findall(r"\.badge\.\w+ \{ background: (?:var\(--([\w-]+)\)|(#[0-9A-Fa-f]{6})); \}", CSS)
        self.assertTrue(badge_rules, "badge color rules not found in CSS -- test needs updating")
        for var_ref, literal in badge_rules:
            color = vars_[var_ref] if var_ref else literal
            self.assertGreaterEqual(contrast(color, "#FFFFFF"), 4.5,
                                    f"badge background {color} fails WCAG AA against white text")

    def test_footer_text_meets_aa_contrast_against_its_background(self):
        import re
        from desktop_app.report_generator import CSS

        def relative_luminance(hexcolor):
            hexcolor = hexcolor.lstrip("#")
            r, g, b = (int(hexcolor[i:i + 2], 16) / 255 for i in (0, 2, 4))
            def lin(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

        def contrast(hex1, hex2):
            l1, l2 = sorted((relative_luminance(hex1), relative_luminance(hex2)))
            return (l2 + 0.05) / (l1 + 0.05)

        vars_ = dict(re.findall(r"--([\w-]+):\s*(#[0-9A-Fa-f]{6});", CSS))
        self.assertGreaterEqual(contrast(vars_["footer-grey"], vars_["light-bg"]), 4.5)
