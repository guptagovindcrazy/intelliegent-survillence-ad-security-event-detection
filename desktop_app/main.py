"""
desktop_app/main.py
---------------------
Entry point for the desktop GUI.

Run:
    python -m desktop_app.main
"""

import sys

from PySide6.QtWidgets import QApplication

from desktop_app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Safety net for EVERY quit path, not just the window's X button: closeEvent
    # already stops a running worker cleanly, but QApplication.quit() can be
    # triggered other ways (a future menu action, Ctrl+C, OS session end) that
    # bypass closeEvent entirely. Without this, quitting while a worker thread
    # is still running crashes the process (QThread destroyed while running).
    def _stop_worker_before_quit():
        if window.worker is not None and window.worker.isRunning():
            window.worker.stop()
            window.worker.wait(2000)

    app.aboutToQuit.connect(_stop_worker_before_quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
