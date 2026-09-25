"""
desktop_app/error_panel.py
-----------------------------
A proper in-window "error page" shown when a run fails, replacing the
video preview area -- not a modal dialog you can miss or dismiss without
reading. Gives plain-English, category-specific guidance (what likely
went wrong, what to try) plus the raw technical error in a collapsible
details section for anyone who wants it.

Categories match RunSummary.error_category, set by pipeline_worker.py at
each of its three failure points. An unrecognized/None category still
works -- it just shows a generic title and skips the tailored suggestions.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

ACCENT = "#E24C4B"
NAVY_DARK = "#141B45"

# (title, plain-English explanation, [actionable suggestions])
_CATEGORY_INFO = {
    "source_open": (
        "Couldn't open the video source",
        "The app couldn't get a real video frame from this source. For a live "
        "camera, this is almost always one of a few specific, fixable things "
        "rather than a broken camera.",
        [
            "Run camera_diagnostic.py (in the project folder) \u2014 it tests every "
            "camera index/backend and tells you exactly which one works.",
            "Windows: Settings \u2192 Privacy & security \u2192 Camera \u2192 make sure "
            "\u201CLet desktop apps access your camera\u201D is on.",
            "Close other apps that might be holding the camera (Teams, Zoom, "
            "Discord, browser tabs).",
            "For a network stream: double-check the RTSP/HTTP URL and that the "
            "camera is reachable on the network.",
            "Try a different camera index (0, 1, 2\u2026) \u2014 index 0 isn't always "
            "your camera if a virtual camera is installed.",
        ],
    ),
    "model_load": (
        "Couldn't load the detection model",
        "The RT-DETR/YOLO weights failed to load. This is usually a missing "
        "file, a bad path, or a device (CPU/GPU) mismatch.",
        [
            "Check that the weights file referenced in config.py actually "
            "exists at that path.",
            "If this machine has no GPU, make sure the device setting isn't "
            "forcing CUDA \u2014 that fails loudly on a CPU-only machine.",
            "Check there's enough free disk space and RAM to load the model.",
            "See the technical details below for the exact underlying error.",
        ],
    ),
    "zone_config": (
        "Zone/tripwire configuration is invalid",
        "The restricted zones or tripwires defined in config.py failed "
        "validation before the run even started \u2014 nothing was processed.",
        [
            "Check for duplicate zone or tripwire names.",
            "Every zone polygon needs at least 3 points.",
            "Every zone needs a recognized severity level.",
            "See the technical details below for exactly which rule failed.",
        ],
    ),
    "processing": (
        "Processing failed on the very first frame",
        "Every frame would have hit this same error, so the run was stopped "
        "immediately rather than \"finishing\" having silently done nothing. "
        "The #1 cause is a device/GPU mismatch: the detection config asking "
        "for a GPU that isn't available on this machine.",
        [
            "Check config.py's DetectionConfig.device \u2014 \u201Cauto\u201D "
            "(recommended) picks CPU automatically when no GPU is found; an "
            "explicit \u201Ccuda\u201D on a machine with no CUDA GPU will fail "
            "exactly like this.",
            "If you do have an NVIDIA GPU, confirm its drivers and CUDA "
            "toolkit are installed and that `torch.cuda.is_available()` "
            "returns True in a Python shell.",
            "See the technical details below for the exact underlying error.",
        ],
    ),
}

_GENERIC_INFO = (
    "Something went wrong",
    "The run stopped before finishing. See the technical details below for "
    "the exact error.",
    [],
)


class ErrorPanel(QWidget):
    """Drop-in replacement for the preview area when a run fails. Call
    show_error(summary) to populate it, dismissed.connect(...) to be
    notified when the user clicks "Dismiss" (main_window.py uses this to
    swap the preview panel back in)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dismiss_callback = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setStyleSheet(f"""
            ErrorPanel {{ background: {NAVY_DARK}; border-radius: 8px; }}
            QLabel {{ color: white; }}
            QPlainTextEdit {{
                background: #0B0F2B; color: #C9CEE8; border: 1px solid #333B6B;
                border-radius: 6px; font-family: Consolas, monospace; font-size: 11px;
            }}
        """)

        icon_and_title = QHBoxLayout()
        icon = QLabel("\u26A0\uFE0F")
        icon.setStyleSheet("font-size: 32px;")
        icon_and_title.addWidget(icon)

        title_col = QVBoxLayout()
        self.title_label = QLabel()
        self.title_label.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {ACCENT};")
        self.explanation_label = QLabel()
        self.explanation_label.setWordWrap(True)
        title_col.addWidget(self.title_label)
        title_col.addWidget(self.explanation_label)
        icon_and_title.addLayout(title_col, 1)
        outer.addLayout(icon_and_title)

        self.suggestions_label = QLabel()
        self.suggestions_label.setWordWrap(True)
        self.suggestions_label.setStyleSheet("margin-top: 10px;")
        outer.addWidget(self.suggestions_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background:#333B6B; margin-top: 10px; margin-bottom: 4px;")
        outer.addWidget(divider)

        details_row = QHBoxLayout()
        self.details_toggle = QPushButton("Show technical details \u25BE")
        self.details_toggle.setFlat(True)
        self.details_toggle.setStyleSheet(f"color:{ACCENT}; text-align:left; background:transparent; padding:2px;")
        self.details_toggle.clicked.connect(self._toggle_details)
        details_row.addWidget(self.details_toggle)
        details_row.addStretch()
        outer.addLayout(details_row)

        self.details_box = QPlainTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setMaximumHeight(120)
        self.details_box.setVisible(False)
        outer.addWidget(self.details_box)

        outer.addStretch()

        dismiss_row = QHBoxLayout()
        dismiss_row.addStretch()
        self.dismiss_btn = QPushButton("Dismiss")
        self.dismiss_btn.clicked.connect(self._on_dismiss)
        dismiss_row.addWidget(self.dismiss_btn)
        outer.addLayout(dismiss_row)

    def _toggle_details(self):
        visible = not self.details_box.isVisible()
        self.details_box.setVisible(visible)
        self.details_toggle.setText(("Hide" if visible else "Show") + " technical details " + ("\u25B4" if visible else "\u25BE"))

    def _on_dismiss(self):
        if self._dismiss_callback:
            self._dismiss_callback()

    def dismissed(self, callback):
        self._dismiss_callback = callback

    def show_error(self, summary):
        """Populates the panel from a failed RunSummary (summary.error set,
        summary.error_category optionally set to one of _CATEGORY_INFO's keys)."""
        title, explanation, suggestions = _CATEGORY_INFO.get(summary.error_category, _GENERIC_INFO)
        self.title_label.setText(title)
        self.explanation_label.setText(explanation)
        if suggestions:
            bullets = "\n".join(f"\u2022  {s}" for s in suggestions)
            self.suggestions_label.setText(bullets)
            self.suggestions_label.setVisible(True)
        else:
            self.suggestions_label.setVisible(False)
        self.details_box.setPlainText(summary.error or "(no details available)")
        self.details_box.setVisible(False)
        self.details_toggle.setText("Show technical details \u25BE")
