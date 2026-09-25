"""Source classification and YouTube resolution (yt-dlp is faked; no network)."""

import sys
import types
import unittest

from src.source_resolver import (
    SourceError, classify_is_live, is_youtube_url, resolve_source,
)


def _fake_yt_dlp(info=None, error=None, seen_options=None):
    mod = types.ModuleType("yt_dlp")

    class YoutubeDL:
        def __init__(self, options):
            if seen_options is not None:
                seen_options.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            assert download is False, "must only look up the address, never download"
            if error:
                raise error
            return info

    mod.YoutubeDL = YoutubeDL
    return mod


class TestClassification(unittest.TestCase):
    def test_webcam_and_rtsp_are_live(self):
        self.assertTrue(classify_is_live("0"))
        self.assertTrue(classify_is_live("rtsp://user:pw@10.0.0.5:554/stream"))

    def test_links_to_video_files_are_recorded_video(self):
        for url in ("https://host/a/clip.mp4", "http://host/clip.MOV?token=1", "https://host/x.webm"):
            self.assertFalse(classify_is_live(url), url)

    def test_other_http_streams_are_live(self):
        self.assertTrue(classify_is_live("http://192.168.1.9:8080/video"))
        self.assertTrue(classify_is_live("https://host/live.mjpg"))

    def test_file_paths_are_recorded_video(self):
        self.assertFalse(classify_is_live(r"C:\videos\clip.mp4"))
        self.assertFalse(classify_is_live("/home/me/clip.avi"))


class TestYoutubeDetection(unittest.TestCase):
    def test_recognises_real_youtube_links(self):
        for url in ("https://www.youtube.com/watch?v=abc", "https://youtu.be/abc", "http://m.youtube.com/watch?v=1",
                    "https://youtube.com/shorts/abc", "  https://youtu.be/abc  "):
            self.assertTrue(is_youtube_url(url), url)

    def test_rejects_lookalikes_and_other_things(self):
        for src in ("https://notyoutube.com/watch?v=1", "https://youtube.com.evil.example/x",
                    "https://example.com/?u=youtube.com", "youtube.com/watch?v=1", "0", "rtsp://youtube.com/x", ""):
            self.assertFalse(is_youtube_url(src), src)


class TestYoutubeResolution(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("yt_dlp", None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            sys.modules.pop("yt_dlp", None)
        else:
            sys.modules["yt_dlp"] = self._saved

    def test_video_link_resolves_to_the_media_address_as_recorded_video(self):
        opts = {}
        sys.modules["yt_dlp"] = _fake_yt_dlp({"url": "https://media.example/v.mp4", "is_live": False}, seen_options=opts)
        r = resolve_source("https://youtu.be/abc")
        self.assertEqual((r.location, r.is_live, r.label), ("https://media.example/v.mp4", False, "https://youtu.be/abc"))
        self.assertTrue(opts["noplaylist"])
        self.assertIn("720", opts["format"])

    def test_a_youtube_live_stream_is_treated_as_live(self):
        sys.modules["yt_dlp"] = _fake_yt_dlp({"url": "https://media.example/live.m3u8", "is_live": True})
        self.assertTrue(resolve_source("https://youtu.be/abc").is_live)

    def test_missing_yt_dlp_gives_an_actionable_message(self):
        sys.modules["yt_dlp"] = None      # makes `import yt_dlp` raise ImportError
        with self.assertRaises(SourceError) as cm:
            resolve_source("https://youtu.be/abc")
        self.assertIn("pip install yt-dlp", str(cm.exception))

    def test_lookup_failure_becomes_a_readable_error_not_a_raw_exception(self):
        sys.modules["yt_dlp"] = _fake_yt_dlp(error=RuntimeError("Video unavailable\nsecond line"))
        with self.assertRaises(SourceError) as cm:
            resolve_source("https://youtu.be/abc")
        msg = str(cm.exception)
        self.assertIn("Video unavailable", msg)
        self.assertNotIn("second line", msg)

    def test_no_playable_address_is_an_error(self):
        sys.modules["yt_dlp"] = _fake_yt_dlp({"title": "x"})
        with self.assertRaises(SourceError):
            resolve_source("https://youtu.be/abc")

    def test_ordinary_sources_never_touch_yt_dlp(self):
        sys.modules["yt_dlp"] = None      # would blow up if imported
        self.assertEqual(resolve_source("0").location, "0")
        self.assertEqual(resolve_source("clip.mp4").is_live, False)


if __name__ == "__main__":
    unittest.main()
