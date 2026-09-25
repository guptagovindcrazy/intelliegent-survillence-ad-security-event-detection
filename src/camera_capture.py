"""
src/camera_capture.py
-----------------------
Shared logic for opening a video/camera source with cv2.VideoCapture.

Why this exists: `cv2.VideoCapture(source)` with no explicit backend lets
OpenCV auto-pick one, which on Windows is usually MSMF. MSMF is a
well-documented source of "opens fine (isOpened() == True) but .read()
never returns a real frame, or hangs" for a meaningful fraction of
webcams -- especially ones whose drivers only properly support
DirectShow. From the caller's side this is indistinguishable from "the
camera just isn't working": no exception, no isOpened()==False, just
silence -- exactly the symptom of "camera count shows up but nothing
comes through."

open_camera_capture() only changes behavior for LIVE camera indices (e.g.
source="0"): it tries CAP_DSHOW first on Windows and actually confirms a
real frame arrives before trusting the capture, falling back to the
default backend if DSHOW doesn't work on a given machine. File/RTSP-path
sources are untouched -- this specific failure mode is a live-webcam-index
problem, not a file/container-decoding one, and probing a file source
would consume its first frame for nothing.
"""

import platform
import time

import cv2


def open_camera_capture(source: str, read_timeout_sec: float = 3.0):
    """Opens `source` (a camera index given as a string, or a file/RTSP
    path) and returns (cap, backend_used, error_message):
      - On success: (an opened, frame-confirmed VideoCapture, backend int
        or None, None).
      - On failure: (None, None, a human-readable reason).
    Never returns a cap that claims isOpened()==True but hasn't actually
    delivered a frame, for a live source -- that silent-lie state is
    exactly what this function exists to catch instead of hand back.
    """
    is_live = source.isdigit()

    if not is_live:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            return None, None, f"could not open {source!r}"
        return cap, None, None

    src = int(source)
    backends_to_try = [cv2.CAP_ANY]
    if platform.system() == "Windows":
        # Try DirectShow FIRST for a live index on Windows -- see module
        # docstring. CAP_ANY (OpenCV's auto-pick, usually MSMF here) is
        # kept as a fallback in case a given machine's driver genuinely
        # prefers MSMF.
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_ANY]

    last_err = None
    for backend in backends_to_try:
        cap = cv2.VideoCapture(src) if backend == cv2.CAP_ANY else cv2.VideoCapture(src, backend)
        if not cap.isOpened():
            cap.release()
            last_err = f"backend={backend}: isOpened() was False"
            continue

        # isOpened() lying (True but no real frames ever arrive) is exactly
        # the MSMF failure mode this exists to catch -- confirm a real
        # frame comes through before trusting this capture. Discarding
        # this one probe frame is a non-issue for a LIVE source (there is
        # no "first frame" to preserve the way there is for a file).
        deadline = time.time() + read_timeout_sec
        got_real_frame = False
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                got_real_frame = True
                break
        if got_real_frame:
            return cap, backend, None

        cap.release()
        last_err = (
            f"backend={backend}: isOpened()==True but no real frame arrived "
            f"within {read_timeout_sec}s -- this is the classic 'camera looks "
            f"open but delivers nothing' symptom, usually a backend/driver "
            f"mismatch or another app holding the camera"
        )

    return None, None, last_err or f"could not open camera index {src}"


def probe_camera_indices(max_index: int = 4, timeout_sec: float = 0.5):
    """Scans camera indices 0..max_index (inclusive) and returns a list of
    (index, backend_used) for every one that actually delivered a real
    frame -- i.e. would work if the user selected it. Releases each
    capture immediately after confirming, so nothing stays held open.

    Deliberately uses a much shorter timeout_sec than open_camera_capture's
    default (0.5s vs 3s): this runs a whole small range of indices back to
    back, purely to populate a "here's what's actually available" list, so
    it should be fast even when several indices in the range don't exist --
    a real, present camera responds far faster than 0.5s, and a genuinely
    working-but-slow-to-init camera is better confirmed once, for real, when
    the user actually selects and starts it (open_camera_capture's longer
    default timeout handles that case).

    Intended to run off the GUI thread (see desktop_app/camera_detect.py) --
    scanning several indices, each potentially trying 2 backends, is not
    fast enough to call directly from a UI event handler without freezing it.
    """
    found = []
    for index in range(max_index + 1):
        cap, backend, _err = open_camera_capture(str(index), read_timeout_sec=timeout_sec)
        if cap is not None:
            cap.release()
            found.append((index, backend))
    return found
