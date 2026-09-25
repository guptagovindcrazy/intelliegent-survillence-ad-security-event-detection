"""
draw_zones.py
--------------
Click-to-draw setup tool for restricted zones and tripwires. Shows a frame from
your camera (or an image/video), you click points, name the shape, and it saves
straight into config/zones.yaml -- no guessing pixel coordinates.

Usage:
    python draw_zones.py                       # first camera in config/cameras.yaml
    python draw_zones.py --camera cam1         # a specific camera from cameras.yaml
    python draw_zones.py --source 0            # a webcam index, video file, or RTSP URL
    python draw_zones.py --image frame.png     # draw on a still image

Keys:
    z / t      zone (polygon) mode / tripwire (line) mode
    left-click add a point (zone: 3+ points, then Enter; tripwire: 2 points)
    Enter      finish the shape and type its name (Enter again to confirm)
    u          undo last point        x   delete last saved shape of this mode
    v          cycle zone severity    d   toggle tripwire direction-sensitivity
    r          grab a fresh frame     s   save to zones.yaml      q / Esc  quit
"""

import argparse
import sys

import cv2

from src.camera_capture import open_camera_capture
from src.config import CAMERAS, ZONES_YAML_PATH
from src.zone_editor import ZoneEditorModel, render_overlay

WINDOW = "Draw zones"
MAX_W, MAX_H = 1280, 720


def _grab(cap):
    frame = None
    for _ in range(5):  # skip a few so we get a recent frame from a live camera
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            frame = f
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw restricted zones / tripwires and save to zones.yaml")
    ap.add_argument("--camera", help="camera_id from config/cameras.yaml (default: first camera)")
    ap.add_argument("--source", help="webcam index, video file path or RTSP URL (overrides --camera)")
    ap.add_argument("--image", help="draw on a still image instead of a live source")
    ap.add_argument("--output", default=ZONES_YAML_PATH, help="YAML file to read/write")
    args = ap.parse_args()

    cap = None
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image {args.image!r}")
            return 1
    else:
        source = args.source
        if source is None:
            cams = {c.camera_id: c for c in CAMERAS}
            if args.camera:
                if args.camera not in cams:
                    print(f"Unknown camera {args.camera!r}. Available: {', '.join(cams)}")
                    return 1
                source = cams[args.camera].source
            else:
                source = CAMERAS[0].source
        cap, _backend, err = open_camera_capture(str(source))
        if cap is None:
            print(f"Could not open source {source!r}: {err}")
            return 1
        frame = _grab(cap)
        if frame is None:
            print("Opened the source but got no frame.")
            return 1

    model = ZoneEditorModel.from_yaml(args.output)
    # Existing zones are rescaled onto THIS frame, and the file records its
    # size so the pipeline can rescale to whatever resolution it runs at.
    model.rebase((frame.shape[1], frame.shape[0]))
    state = {"naming": None, "msg": "", "frame": frame, "confirm_quit": False}

    def scale():
        h, w = state["frame"].shape[:2]
        return min(1.0, MAX_W / w, MAX_H / h)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and state["naming"] is None:
            s = scale()
            model.add_point((round(x / s), round(y / s)))  # display -> frame coordinates
            state["msg"] = ""
            state["confirm_quit"] = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        img = render_overlay(state["frame"], model, state["naming"], state["msg"])
        s = scale()
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == 255:  # no key
            # a tripwire is complete at 2 points -- go straight to naming
            if model.mode == "tripwire" and model.can_finish() and state["naming"] is None:
                state["naming"] = ""
            continue

        if state["naming"] is not None:  # typing a name
            if key == 13:
                err = model.commit(state["naming"])
                if err:
                    state["msg"] = err
                else:
                    state["msg"] = "Added. Press s to save."
                    state["naming"] = None
            elif key == 27:
                state["naming"] = None
                if model.mode == "tripwire":
                    model.pending = []
                state["msg"] = "Cancelled"
            elif key in (8, 127):
                state["naming"] = state["naming"][:-1]
            elif 32 <= key <= 126:
                state["naming"] += chr(key)
            continue

        state["confirm_quit"] = state["confirm_quit"] and key in (ord("q"), 27)
        if key == ord("z"):
            model.set_mode("zone")
        elif key == ord("t"):
            model.set_mode("tripwire")
        elif key in (13, 10):
            if model.can_finish():
                state["naming"] = ""
            else:
                state["msg"] = "A zone needs at least 3 points"
        elif key in (ord("u"), 8):
            model.undo_point()
        elif key == ord("x"):
            name = model.delete_last()
            state["msg"] = f"Deleted '{name}'" if name else "Nothing to delete"
        elif key == ord("v"):
            state["msg"] = f"Severity: {model.cycle_severity()}"
        elif key == ord("d"):
            state["msg"] = "1-way" if model.toggle_direction() else "2-way"
        elif key == ord("r") and cap is not None:
            f = _grab(cap)
            if f is not None:
                state["frame"] = f
        elif key == ord("s"):
            model.save(args.output)
            state["msg"] = f"Saved to {args.output}"
        elif key in (ord("q"), 27):
            if model.dirty and not state["confirm_quit"]:
                state["msg"] = "Unsaved changes -- press q again to quit without saving, or s to save"
                state["confirm_quit"] = True
            else:
                break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
