"""
camera_diagnostic.py
----------------------
Standalone script -- only needs opencv-python (`pip install opencv-python`
if you don't already have it). Run this directly:

    python camera_diagnostic.py

It does NOT need the rest of the surveillance project. It tries every
camera index 0-4 with every backend OpenCV has on your machine, actually
reads real frames (not just checking isOpened()), and tells you exactly
which index/backend combination works -- so you know exactly what to put
as the camera source, and why the current one is silently failing.
"""

import platform
import time

import cv2

BACKENDS = [
    ("CAP_ANY (auto)", cv2.CAP_ANY),
]
if platform.system() == "Windows":
    BACKENDS += [
        ("CAP_DSHOW", cv2.CAP_DSHOW),
        ("CAP_MSMF", cv2.CAP_MSMF),
    ]

print(f"OpenCV version: {cv2.__version__}")
print(f"Platform: {platform.system()} {platform.release()}\n")

found_any_working = False

for index in range(5):
    for backend_name, backend in BACKENDS:
        cap = cv2.VideoCapture(index, backend) if backend != cv2.CAP_ANY else cv2.VideoCapture(index)
        opened = cap.isOpened()

        if not opened:
            cap.release()
            print(f"  index={index}  backend={backend_name:15s}  isOpened=False")
            continue

        # The real test: isOpened() can lie. Try to get an actual frame.
        deadline = time.time() + 2.0
        got_frame = False
        frame_shape = None
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                got_frame = True
                frame_shape = frame.shape
                break

        status = "WORKS -- got a real frame" if got_frame else "isOpened=True but NO real frame arrived (this is the bug)"
        print(f"  index={index}  backend={backend_name:15s}  isOpened=True   {status}"
              + (f"  shape={frame_shape}" if got_frame else ""))

        if got_frame:
            found_any_working = True

        cap.release()
    print()

print("=" * 70)
if found_any_working:
    print("At least one index/backend combination above got a real frame.")
    print("Use THAT index, and if CAP_DSHOW is what worked, that confirms")
    print("the MSMF-default-backend issue -- the project's camera_capture.py")
    print("fix (tries CAP_DSHOW first on Windows) should now handle this")
    print("automatically for you.")
else:
    print("NO index/backend combination produced a real frame on this machine.")
    print("This means it's not a backend-selection issue -- check:")
    print("  1. Windows Settings > Privacy & security > Camera > ")
    print("     'Let desktop apps access your camera' is ON")
    print("  2. No other app (Teams/Zoom/Discord/browser) currently has")
    print("     the camera open")
    print("  3. The camera works in the built-in Windows Camera app at all")
