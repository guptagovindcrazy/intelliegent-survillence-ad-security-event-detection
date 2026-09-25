"""
source_resolver.py
-------------------
Turns whatever the user typed as a video source into something OpenCV can open,
and says whether it behaves like a LIVE feed or a recorded video.

    "0"                              webcam index          -> live
    "rtsp://cam/stream"              network camera        -> live
    "https://host/clip.mp4"          link to a video file  -> recorded (own timeline)
    "https://host/stream.mjpg"       other http(s) stream  -> live
    "https://youtu.be/xxxx"          YouTube page link     -> resolved with yt-dlp
    "C:\\videos\\clip.mp4"            file path             -> recorded

A YouTube link is a web PAGE, not a video file, so OpenCV can't open it
directly -- that is why pasting one used to fail. yt-dlp (optional dependency)
looks up the actual media address for it. Only use videos you have the right to
use: YouTube's terms restrict downloading/processing other people's content.
"""

from dataclasses import dataclass
from urllib.parse import urlparse

VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm")
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}


class SourceError(Exception):
    """A source couldn't be resolved; the message is written for the user."""


@dataclass(frozen=True)
class ResolvedSource:
    location: str    # what to hand to cv2.VideoCapture
    is_live: bool    # True: wall-clock timing, native resolution. False: the video's own timeline.
    label: str       # what the user typed, for display


def is_youtube_url(source: str) -> bool:
    try:
        parsed = urlparse(source.strip())
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and (parsed.hostname or "").lower() in YOUTUBE_HOSTS


def classify_is_live(source: str) -> bool:
    """Webcam indices and network streams deliver frames in real time, so the
    dwell clock uses wall-clock time for them. Files -- including a plain link
    to a video FILE -- use the video's own timestamps instead, so results don't
    depend on how fast the video is processed."""
    s = source.strip()
    lower = s.lower()
    if s.isdigit() or lower.startswith("rtsp://"):
        return True
    if lower.startswith(("http://", "https://")):
        path = urlparse(s).path.lower()
        return not path.endswith(VIDEO_EXTENSIONS)
    return False


def _resolve_youtube(url: str, max_height: int) -> ResolvedSource:
    try:
        import yt_dlp
    except ImportError as exc:
        raise SourceError(
            "YouTube links need the 'yt-dlp' package. Install it with:  pip install yt-dlp"
        ) from exc
    options = {
        # One file with video AND audio up to max_height (what OpenCV can read), else the best available.
        "format": f"best[height<={max_height}]/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 -- yt-dlp raises many types; all mean "couldn't get it"
        reason = str(exc).splitlines()[0][:200] if str(exc) else exc.__class__.__name__
        raise SourceError(
            f"Could not get the video from that YouTube link ({reason}). It may be private, "
            f"age-restricted, region-locked or removed, or yt-dlp may need updating "
            f"(pip install -U yt-dlp)."
        ) from exc
    direct = (info or {}).get("url")
    if not direct:
        raise SourceError("YouTube didn't return a playable address for that video.")
    return ResolvedSource(direct, bool(info.get("is_live")), url)


def resolve_source(source: str, max_height: int = 720) -> ResolvedSource:
    """Raises SourceError (with a user-readable message) if a YouTube link can't be resolved."""
    s = source.strip()
    if is_youtube_url(s):
        return _resolve_youtube(s, max_height)
    return ResolvedSource(s, classify_is_live(s), s)
