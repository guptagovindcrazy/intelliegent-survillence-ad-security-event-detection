"""
diagnose.py -- finds out why the desktop app won't start.

    python diagnose.py            (or:  run.bat diagnose)

Checks each dependency and project import in turn, in a separate process so a
hard crash (e.g. a missing Windows DLL for torch) is reported instead of
silently killing this script. Prints everything and saves it to
diagnose_report.txt -- send that file's contents if something is marked FAIL.
"""

import os
import platform
import subprocess
import sys

STEPS = [
    ("PySide6 (GUI toolkit)", "import PySide6; from PySide6.QtWidgets import QApplication; print(PySide6.__version__)"),
    ("OpenCV", "import cv2; print(cv2.__version__)"),
    ("NumPy", "import numpy; print(numpy.__version__)"),
    ("PyYAML", "import yaml; print(yaml.__version__)"),
    ("torch (ML runtime)", "import torch; print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')"),
    ("ultralytics (YOLO)", "import ultralytics; print(ultralytics.__version__)"),
    ("project config", "import src.config as c; print('cameras:', len(c.CAMERAS), '| zones:', len(c.RESTRICTED_ZONES), '| tripwires:', len(c.TRIPWIRES))"),
    ("pipeline import", "import src.main, desktop_app.pipeline_worker; print('ok')"),
    ("main window (built, not shown)",
     "from PySide6.QtWidgets import QApplication\n"
     "app = QApplication([])\n"
     "from desktop_app.main_window import MainWindow\n"
     "w = MainWindow(); print('MainWindow created')"),
]


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    lines = [f"Python {sys.version.split()[0]} ({platform.architecture()[0]}) on {platform.platform()}",
             f"Executable: {sys.executable}", f"Project: {root}", ""]
    failed = 0
    for name, code in STEPS:
        proc = subprocess.run([sys.executable, "-X", "faulthandler", "-c", code], cwd=root,
                              capture_output=True, text=True)
        out = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 0:
            lines.append(f"[ OK ] {name}: {out.splitlines()[-1] if out else ''}")
        else:
            failed += 1
            lines.append(f"[FAIL] {name} (exit code {proc.returncode})")
            lines.extend("       " + ln for ln in out.splitlines()[-25:])
    lines.append("")
    lines.append("Everything imports -- if the window still doesn't appear, run "
                 "`python -X faulthandler -m desktop_app.main` and send what it prints."
                 if not failed else f"{failed} step(s) failed -- send this report.")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(root, "diagnose_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
