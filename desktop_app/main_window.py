"""
desktop_app/main_window.py
----------------------------
The whole UI in one window:
  - pick a source: recorded video file, a live local camera, or an RTSP/HTTP stream
  - Start/Stop control the PipelineWorker background thread
  - a live preview panel shows the annotated video as it's processed
  - a live alert list shows each flag the moment it's raised
  - "Save Report" writes a self-contained HTML report once a run has data
  - "Edit Zones" swaps the preview for a click-to-draw zone editor working on a
    clean frame from the selected source (or from the running feed). Saved
    zones are hot-applied to a running pipeline, or used on the next run.
"""

import os

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QRadioButton, QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
)

from desktop_app.camera_detect import CameraDetectWorker
from desktop_app.error_panel import ErrorPanel
from desktop_app.pipeline_worker import PipelineWorker
from desktop_app.models import alert_type_info
from desktop_app.report_generator import save_report
from desktop_app.snapshot_worker import SnapshotWorker
from desktop_app.zone_editor_panel import ZoneEditorPanel

NAVY = "#1E2761"
NAVY_DARK = "#141B45"
ACCENT = "#E24C4B"
GREEN = "#1C7C54"


def cv_frame_to_pixmap(frame) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spatio-Temporal Surveillance \u2014 Desktop")
        self.resize(1180, 720)

        self.worker: PipelineWorker | None = None
        self.last_summary = None
        self._run_active = False            # between Start and the run's finished signal
        self._editing = False               # zone editor is showing
        self._awaiting_snapshot = False     # asked for a frame, not delivered yet
        self._snapshot_worker: SnapshotWorker | None = None

        self._build_ui()
        self._apply_styles()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        # ---- source selection ----
        source_box = QGroupBox("Video Source")
        source_layout = QHBoxLayout(source_box)

        self.radio_file = QRadioButton("Recorded video file")
        self.radio_file.setChecked(True)
        self.radio_camera = QRadioButton("Live camera")
        self.radio_stream = QRadioButton("Network stream (RTSP/HTTP)")

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Choose a video file\u2026")
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.clicked.connect(self._browse_file)

        self.camera_combo = QComboBox()
        self.camera_combo.setEnabled(False)
        self.camera_combo.setMinimumWidth(160)
        self.camera_combo.addItem("Click Detect \u2192", userData=None)
        self.detect_cameras_btn = QPushButton("Detect")
        self.detect_cameras_btn.setEnabled(False)
        self.detect_cameras_btn.clicked.connect(self._detect_cameras)
        self._camera_detect_worker = None

        self.stream_url_edit = QLineEdit()
        self.stream_url_edit.setPlaceholderText("rtsp://\u2026 stream, video link, or YouTube link")
        self.stream_url_edit.setEnabled(False)

        self.radio_file.toggled.connect(self._on_source_type_changed)
        self.radio_camera.toggled.connect(self._on_source_type_changed)
        self.radio_stream.toggled.connect(self._on_source_type_changed)

        source_layout.addWidget(self.radio_file)
        source_layout.addWidget(self.file_path_edit, 2)
        source_layout.addWidget(browse_btn)
        source_layout.addWidget(self.radio_camera)
        source_layout.addWidget(self.camera_combo)
        source_layout.addWidget(self.detect_cameras_btn)
        source_layout.addWidget(self.radio_stream)
        source_layout.addWidget(self.stream_url_edit, 2)

        outer.addWidget(source_box)

        # ---- controls ----
        controls = QHBoxLayout()
        self.start_btn = QPushButton("\u25B6  Start")
        self.stop_btn = QPushButton("\u25A0  Stop")
        self.stop_btn.setEnabled(False)
        self.report_btn = QPushButton("\U0001F4C4  Save Report\u2026")
        self.report_btn.setEnabled(False)
        self.edit_zones_btn = QPushButton("\u270E  Edit Zones")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.report_btn.clicked.connect(self._save_report)
        self.edit_zones_btn.clicked.connect(self._toggle_edit_zones)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.edit_zones_btn)
        controls.addStretch()
        controls.addWidget(self.report_btn)
        outer.addLayout(controls)

        # ---- main split: preview | alerts ----
        split = QHBoxLayout()

        self.preview_label = QLabel("Preview will appear here once you press Start")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(720, 480)
        self.preview_label.setStyleSheet(f"background:{NAVY_DARK}; color:white; border-radius:8px;")

        self.error_panel = ErrorPanel()
        self.error_panel.dismissed(self._on_error_dismissed)

        # Stacked so the error page can fully replace the preview (not a
        # modal popup you can miss/dismiss without reading) and swap back
        # cleanly once acknowledged or once a new run starts.
        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.preview_label)   # index 0
        self.preview_stack.addWidget(self.error_panel)      # index 1
        self.zone_panel = ZoneEditorPanel()
        self.zone_panel.saved.connect(self._on_zones_saved)
        self.zone_panel.closed.connect(self._exit_edit_mode)
        self.preview_stack.addWidget(self.zone_panel)       # index 2
        self.preview_stack.setMinimumSize(720, 480)
        split.addWidget(self.preview_stack, 3)

        alerts_box = QGroupBox("Live Alerts")
        alerts_layout = QVBoxLayout(alerts_box)
        self.alert_list = QListWidget()
        alerts_layout.addWidget(self.alert_list)
        split.addWidget(alerts_box, 2)

        outer.addLayout(split, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

    def _apply_styles(self):
        self.setStyleSheet(f"""
            QPushButton {{ background: {NAVY}; color: white; padding: 8px 14px; border-radius: 6px; font-weight: 600; }}
            QPushButton:disabled {{ background: #B7BDD6; }}
            QPushButton:hover:!disabled {{ background: {NAVY_DARK}; }}
            QGroupBox {{ font-weight: 600; border: 1px solid #D8DEEE; border-radius: 8px; margin-top: 8px; padding-top: 12px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        """)

    # ------------------------------------------------------------- helpers
    def _on_source_type_changed(self):
        self.file_path_edit.setEnabled(self.radio_file.isChecked())
        camera_selected = self.radio_camera.isChecked()
        self.camera_combo.setEnabled(camera_selected)
        self.detect_cameras_btn.setEnabled(camera_selected)
        self.stream_url_edit.setEnabled(self.radio_stream.isChecked())

    def _detect_cameras(self):
        self.detect_cameras_btn.setEnabled(False)
        self.detect_cameras_btn.setText("Detecting\u2026")
        self.camera_combo.clear()
        self.camera_combo.addItem("Detecting\u2026", userData=None)
        self.status_bar.showMessage("Scanning for cameras (indices 0\u20134)\u2026")

        self._camera_detect_worker = CameraDetectWorker(max_index=4)
        self._camera_detect_worker.finished_detect.connect(self._on_cameras_detected)
        self._camera_detect_worker.start()

    def _on_cameras_detected(self, found: list):
        self.detect_cameras_btn.setEnabled(True)
        self.detect_cameras_btn.setText("Detect")
        self.camera_combo.clear()

        if not found:
            self.camera_combo.addItem("No camera found", userData=None)
            self.status_bar.showMessage(
                "No working camera found on indices 0\u20134. Check Windows camera "
                "privacy settings, or that no other app is using it."
            )
            return

        for index, backend in found:
            self.camera_combo.addItem(f"Camera {index}", userData=index)
        self.status_bar.showMessage(f"Found {len(found)} camera(s).")

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video file", "", "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)")
        if path:
            self.file_path_edit.setText(path)

    def _resolve_source(self) -> str | None:
        if self.radio_file.isChecked():
            path = self.file_path_edit.text().strip()
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "No file selected", "Choose a valid video file first.")
                return None
            return path
        if self.radio_camera.isChecked():
            index = self.camera_combo.currentData()
            if index is None:
                QMessageBox.warning(
                    self, "No camera selected",
                    "Click \"Detect\" first and pick a camera from the list.",
                )
                return None
            return str(index)
        url = self.stream_url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "No stream URL", "Enter an RTSP/HTTP stream URL or a YouTube link first.")
            return None
        return url

    # -------------------------------------------------------------- start/stop
    def _start(self):
        # Guard against re-entrant Start calls (e.g. a stray double-click or a
        # programmatic call) creating a second worker while one is still running --
        # that orphans the first QThread and crashes the app (QThread destroyed
        # while still running). The disabled button covers normal clicking, but
        # this guard is the real safety net.
        if self.worker is not None and self.worker.isRunning():
            return
        if self._editing or self._awaiting_snapshot:
            return  # finish/close the zone editor first

        source = self._resolve_source()
        if source is None:
            return

        self.alert_list.clear()
        self.last_summary = None
        self.report_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_source_controls_enabled(False)
        self.preview_stack.setCurrentWidget(self.preview_label)

        self._run_active = True
        self.worker = PipelineWorker(source)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.alert_raised.connect(self._on_alert)
        self.worker.status_changed.connect(self.status_bar.showMessage)
        self.worker.finished_run.connect(self._on_finished)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
        self.stop_btn.setEnabled(False)

    def _set_source_controls_enabled(self, enabled: bool):
        for w in (self.radio_file, self.radio_camera, self.radio_stream,
                  self.file_path_edit, self.camera_combo, self.detect_cameras_btn,
                  self.stream_url_edit):
            w.setEnabled(enabled)
        if enabled:
            self._on_source_type_changed()

    # -------------------------------------------------------------- signals
    def _on_frame(self, frame):
        if self._editing:
            return  # the editor is on screen; don't spend time painting a hidden label
        pix = cv_frame_to_pixmap(frame)
        self.preview_label.setPixmap(pix.scaled(
            self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    def _on_alert(self, record):
        badge = alert_type_info(record.alert_type).icon
        text = f"{badge}  [{record.video_time_sec:6.1f}s]  Track #{record.track_id} \u2014 {record.message}"
        item = QListWidgetItem(text)
        self.alert_list.insertItem(0, item)  # newest first

    def _on_finished(self, summary):
        self._run_active = False  # (isRunning() can still be True for a moment after this signal)
        self.last_summary = summary
        self.stop_btn.setEnabled(False)
        if self._awaiting_snapshot:  # run ended before it could hand over a frame
            self._awaiting_snapshot = False
            self.edit_zones_btn.setEnabled(True)
        self._refresh_controls()

        if summary.error:
            self.status_bar.showMessage("Error \u2014 see details")
            self.error_panel.show_error(summary)
            if not self._editing:  # otherwise shown when the editor closes
                self.preview_stack.setCurrentWidget(self.error_panel)
            return

        self.report_btn.setEnabled(True)
        frame_errors_note = f", {summary.frame_errors} frame(s) skipped due to errors" if summary.frame_errors else ""
        self.status_bar.showMessage(
            f"Finished \u2014 {summary.frames_processed} frames, {len(summary.alerts)} alert(s){frame_errors_note}."
        )

    # ---------------------------------------------------------- zone editing
    def _running(self) -> bool:
        return self._run_active and self.worker is not None

    def _refresh_controls(self):
        """Start/source controls follow one rule: usable only when nothing is
        running and the zone editor isn't open."""
        idle = not self._running() and not self._editing and not self._awaiting_snapshot
        self.start_btn.setEnabled(idle)
        self._set_source_controls_enabled(idle)

    def _toggle_edit_zones(self):
        if self._editing:
            self.zone_panel.request_close()
        else:
            self._begin_edit_zones()

    def _begin_edit_zones(self):
        if self._awaiting_snapshot:
            return
        if self._running():
            # The live pipeline hands over one clean frame; nothing else changes.
            self._awaiting_snapshot = True
            self.edit_zones_btn.setEnabled(False)
            self.status_bar.showMessage("Grabbing a frame from the running feed\u2026")
            self.worker.request_snapshot()
            return
        source = self._resolve_source()
        if source is None:
            return
        self._awaiting_snapshot = True
        self.edit_zones_btn.setEnabled(False)
        self.status_bar.showMessage("Grabbing a frame\u2026")
        old = self._snapshot_worker
        if old is not None and old.isRunning():
            old.wait(2000)  # never drop a QThread that's still running -- that crashes Qt
        self._snapshot_worker = SnapshotWorker(source)
        self._snapshot_worker.snapshot_ready.connect(self._on_snapshot)
        self._snapshot_worker.failed.connect(self._on_snapshot_failed)
        self._snapshot_worker.start()

    def _on_snapshot(self, frame):
        if not self._awaiting_snapshot:
            return  # unsolicited / stale
        self._awaiting_snapshot = False
        self._editing = True
        self.zone_panel.load(frame)
        self.preview_stack.setCurrentWidget(self.zone_panel)
        self.edit_zones_btn.setText("\u2715  Close Editor")
        self.edit_zones_btn.setEnabled(True)
        self._refresh_controls()
        self.status_bar.showMessage(
            "Editing zones \u2014 the live feed keeps running in the background." if self._running()
            else "Editing zones.")

    def _on_snapshot_failed(self, message: str):
        self._awaiting_snapshot = False
        self.edit_zones_btn.setEnabled(True)
        self.status_bar.showMessage("Could not grab a frame.")
        QMessageBox.warning(self, "Could not grab a frame", message)

    def _on_zones_saved(self, config):
        if self._running():
            self.worker.update_zones(config)
            note = "applied to the running feed"
        else:
            note = "they'll be used on the next run"
        self.status_bar.showMessage(
            f"Zones saved \u2014 {len(config.zones)} zone(s), {len(config.tripwires)} tripwire(s); {note}.")

    def _exit_edit_mode(self):
        self._editing = False
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.edit_zones_btn.setText("\u270E  Edit Zones")
        self.edit_zones_btn.setEnabled(True)
        self._refresh_controls()
        if not self._running() and self.last_summary is not None and self.last_summary.error:
            self.preview_stack.setCurrentWidget(self.error_panel)  # a run failed while we were editing

    def _on_error_dismissed(self):
        self.preview_stack.setCurrentWidget(self.preview_label)
        self.status_bar.showMessage("Ready.")

    # -------------------------------------------------------------- report
    def _save_report(self):
        if not self.last_summary:
            return
        default_name = "surveillance_report.html"
        path, _ = QFileDialog.getSaveFileName(self, "Save report", default_name, "HTML Report (*.html)")
        if not path:
            return
        save_report(self.last_summary, path)
        QMessageBox.information(self, "Report saved", f"Report saved to:\n{path}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        if self._snapshot_worker is not None and self._snapshot_worker.isRunning():
            self._snapshot_worker.wait(2000)
        event.accept()
