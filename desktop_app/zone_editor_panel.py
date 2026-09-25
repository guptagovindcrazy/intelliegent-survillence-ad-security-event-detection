"""
desktop_app/zone_editor_panel.py
----------------------------------
In-app zone editor: draw restricted zones and tripwires directly on a frame
from the feed you're watching.

  ZoneCanvas       -- paints one frame plus the shapes, turns mouse clicks into
                      image-pixel points on the shared ZoneEditorModel.
  ZoneEditorPanel  -- canvas + side controls (mode, severity, direction, shape
                      list, save/cancel). Emits `saved(ZoneConfig)` / `closed()`.

Performance/design notes:
  * The frame is converted to a QPixmap once (set_frame) and its scaled copy is
    cached per widget size, so repaints while drawing are just a blit plus a
    handful of vector shapes -- cost doesn't grow with frame size or with how
    long you edit.
  * Nothing here touches the pipeline. The editor works on a still frame, and
    the result reaches a running pipeline only via PipelineWorker.update_zones,
    so drawing never slows detection and detection never slows drawing.
  * All geometry/validation lives in the GUI-free ZoneEditorModel and
    FrameViewTransform (unit-tested headlessly); this file is the thin Qt layer.
"""

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QMessageBox, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from src.config import ZONES_YAML_PATH, ZoneConfig
from src.frame_geometry import FrameViewTransform
from src.layers.spatial_zones import violation_direction
from src.zone_editor import SEVERITIES, ZoneEditorModel, nearest_vertex

_SEV_COLOR = {"low": QColor(255, 200, 0), "medium": QColor(255, 140, 0), "high": QColor(255, 0, 0)}
_WIRE_COLOR = QColor(0, 128, 255)
_PENDING_COLOR = QColor(0, 220, 0)
_ENTRY_COLOR = QColor(190, 0, 210)   # "alert on entry" zones


def _to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy())


class ZoneCanvas(QWidget):
    changed = Signal()            # points/shapes changed -> panel refreshes its list/buttons
    finish_requested = Signal()   # a shape is complete and needs a name

    def __init__(self, model: ZoneEditorModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._pixmap: Optional[QPixmap] = None
        self._scaled: Optional[QPixmap] = None
        self._scaled_key = None
        self._frame_size = (0, 0)
        self._hover: Optional[tuple] = None   # last cursor position in image px
        self._drag: Optional[tuple] = None    # (kind, shape_index, vertex_index) being dragged
        self._box_anchor: Optional[tuple] = None   # image px where a box drag started
        self._box_cursor: Optional[tuple] = None   # image px the box drag has reached
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(720, 480)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # -- state ---------------------------------------------------------------
    def set_frame(self, frame_bgr: np.ndarray) -> None:
        self._pixmap = _to_pixmap(frame_bgr)
        self._frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
        self._scaled = self._scaled_key = None
        self._hover = None
        self._drag = None
        self._box_anchor = self._box_cursor = None
        self.update()

    def set_model(self, model: ZoneEditorModel) -> None:
        self.model = model
        self.update()

    @property
    def frame_size(self):
        return self._frame_size

    def transform(self) -> FrameViewTransform:
        return FrameViewTransform((self.width(), self.height()), self._frame_size)

    # -- painting ---------------------------------------------------------------
    def _scaled_pixmap(self, w: int, h: int) -> QPixmap:
        if self._scaled is None or self._scaled_key != (w, h):
            self._scaled = self._pixmap.scaled(
                w, h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self._scaled_key = (w, h)
        return self._scaled

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#141B45"))
        if self._pixmap is None:
            return
        tf = self.transform()
        x, y, w, h = tf.image_rect()
        p.drawPixmap(int(round(x)), int(round(y)), self._scaled_pixmap(max(1, int(round(w))), max(1, int(round(h)))))

        def poly(points):
            return QPolygonF([QPointF(*tf.to_widget(px, py)) for px, py in points])

        p.setFont(QFont(self.font().family(), 9, QFont.Weight.DemiBold))
        for z in self.model.zones:
            color = _ENTRY_COLOR if z.alert_on_entry else _SEV_COLOR.get(z.severity, _SEV_COLOR["high"])
            fill = QColor(color)
            fill.setAlpha(60)
            p.setPen(QPen(color, 2))
            p.setBrush(fill)
            p.drawPolygon(poly(z.polygon))
            lx, ly = tf.to_widget(*z.polygon[0])
            p.setPen(QColor("white"))
            p.drawText(QPointF(lx + 5, ly + 15), f"{z.name} [{'entry alert' if z.alert_on_entry else z.severity}]")
        for t in self.model.tripwires:
            p.setPen(QPen(_WIRE_COLOR, 2))
            a, b = QPointF(*tf.to_widget(*t.p1)), QPointF(*tf.to_widget(*t.p2))
            p.drawLine(a, b)
            if t.direction_sensitive:  # arrow = the direction that raises an alert
                ux, uy = violation_direction(t.p1, t.p2)
                mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
                tip = QPointF(mid.x() + ux * 40, mid.y() + uy * 40)
                p.drawLine(mid, tip)
                for wing in (0.5, -0.5):  # two short strokes make the arrow head
                    hx = -(ux * 0.87 - uy * wing) * 12
                    hy = -(uy * 0.87 + ux * wing) * 12
                    p.drawLine(tip, QPointF(tip.x() + hx, tip.y() + hy))
            p.setPen(QColor("white"))
            p.drawText(QPointF(a.x() + 5, a.y() - 6), f"{t.name} ({'1-way' if t.direction_sensitive else '2-way'})")

        if not self.model.pending:  # vertex handles: grab one to reshape a saved shape
            p.setPen(QPen(QColor("black"), 1))
            p.setBrush(QColor("white"))
            handles = [pt for z in self.model.zones for pt in z.polygon]
            handles += [pt for t in self.model.tripwires for pt in (t.p1, t.p2)]
            for hx, hy in handles:
                wx, wy = tf.to_widget(hx, hy)
                p.drawRect(int(wx) - 4, int(wy) - 4, 8, 8)

        if self._box_anchor is not None and self._box_cursor is not None:  # box being dragged out
            ax, ay = tf.to_widget(*self._box_anchor)
            bx, by = tf.to_widget(*self._box_cursor)
            fill = QColor(_PENDING_COLOR)
            fill.setAlpha(50)
            p.setPen(QPen(_PENDING_COLOR, 2, Qt.PenStyle.DashLine))
            p.setBrush(fill)
            p.drawRect(QRectF(QPointF(min(ax, bx), min(ay, by)), QPointF(max(ax, bx), max(ay, by))))

        pts = self.model.pending
        if pts:
            path = [QPointF(*tf.to_widget(px, py)) for px, py in pts]
            p.setPen(QPen(_PENDING_COLOR, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for a, b in zip(path, path[1:]):
                p.drawLine(a, b)
            if self._hover is not None:  # rubber band to the cursor
                p.setPen(QPen(_PENDING_COLOR, 1, Qt.PenStyle.DashLine))
                p.drawLine(path[-1], QPointF(*tf.to_widget(*self._hover)))
            p.setPen(QPen(QColor("black"), 1))
            p.setBrush(_PENDING_COLOR)
            for pt in path:
                p.drawEllipse(pt, 4, 4)

    # -- input ---------------------------------------------------------------------
    def mousePressEvent(self, e):
        self.setFocus()
        if e.button() == Qt.MouseButton.RightButton:
            self.model.undo_point()
            self.changed.emit()
            self.update()
            return
        if e.button() != Qt.MouseButton.LeftButton or self._pixmap is None:
            return
        tf = self.transform()
        # Grabbing a vertex takes priority over starting a shape -- unless a
        # shape is mid-draw, or Shift is held to force a new point right on top of one.
        if not self.model.pending and not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            hit = nearest_vertex(self.model, tf, e.position().x(), e.position().y())
            if hit:
                self._drag = hit
                return
        pt = tf.to_image(e.position().x(), e.position().y())
        if pt is None:      # click in the letterbox margin, outside the image
            return
        if self.model.mode == "zone" and self.model.zone_shape == "box" and not self.model.pending:
            self._box_anchor = self._box_cursor = pt   # press-drag-release draws a rectangle
            self.update()
            return
        self.model.add_point(pt)
        self.changed.emit()
        self.update()
        if self.model.mode == "tripwire" and self.model.can_finish():
            self.finish_requested.emit()

    def mouseDoubleClickEvent(self, e):
        # The first click of the double-click already added the point; the
        # double-click just means "that was the last one".
        if e.button() == Qt.MouseButton.LeftButton and self.model.mode == "zone" and self.model.can_finish():
            self.finish_requested.emit()

    def mouseMoveEvent(self, e):
        if self._pixmap is None:
            return
        tf = self.transform()
        if self._box_anchor is not None:
            self._box_cursor = tf.to_image(e.position().x(), e.position().y(), clamp=True)
            self.update()
            return
        if self._drag is not None:
            pt = tf.to_image(e.position().x(), e.position().y(), clamp=True)
            if pt is not None and self.model.move_vertex(*self._drag, pt):
                self.changed.emit()
                self.update()
            return
        if self.model.pending:
            self._hover = tf.to_image(e.position().x(), e.position().y(), clamp=True)
            self.update()
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            over = nearest_vertex(self.model, tf, e.position().x(), e.position().y())
            self.setCursor(Qt.CursorShape.SizeAllCursor if over else Qt.CursorShape.CrossCursor)

    MIN_BOX_PX = 6   # a smaller drag is treated as a stray click, not a box

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = None
        if self._box_anchor is not None:
            a, b = self._box_anchor, self._box_cursor or self._box_anchor
            self._box_anchor = self._box_cursor = None
            big_enough = abs(a[0] - b[0]) >= self.MIN_BOX_PX and abs(a[1] - b[1]) >= self.MIN_BOX_PX
            if big_enough and self.model.set_box(a, b):
                self.changed.emit()
                self.update()
                self.finish_requested.emit()
            else:
                self.update()

    def leaveEvent(self, _e):
        if self._hover is not None:
            self._hover = None
            self.update()

    def keyPressEvent(self, e):
        key = e.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.model.can_finish():
                self.finish_requested.emit()
        elif key in (Qt.Key.Key_Backspace, Qt.Key.Key_U):
            self.model.undo_point()
            self.changed.emit()
            self.update()
        elif key == Qt.Key.Key_Escape:
            self._box_anchor = self._box_cursor = None
            self.model.pending = []
            self.changed.emit()
            self.update()
        else:
            super().keyPressEvent(e)


class ZoneEditorPanel(QWidget):
    saved = Signal(object)   # ZoneConfig -- already written to disk
    closed = Signal()        # editor should be dismissed (after save or cancel)

    def __init__(self, path: str = ZONES_YAML_PATH, parent=None):
        super().__init__(parent)
        self.path = path
        self.model = ZoneEditorModel()
        self.canvas = ZoneCanvas(self.model)
        self.canvas.changed.connect(self._refresh)
        self.canvas.finish_requested.connect(self._name_and_commit)

        side = QVBoxLayout()
        side.addWidget(QLabel("<b>Edit zones</b>"))
        self.box_radio = QRadioButton("Zone: box (drag)")
        self.zone_radio = QRadioButton("Zone: polygon (click points)")
        self.wire_radio = QRadioButton("Tripwire: line")
        self.box_radio.setChecked(True)
        self._shape_group = QButtonGroup(self)
        for radio in (self.box_radio, self.zone_radio, self.wire_radio):
            self._shape_group.addButton(radio)
            radio.toggled.connect(self._on_mode_changed)
            side.addWidget(radio)

        self.severity_combo = QComboBox()
        self.severity_combo.addItems(SEVERITIES)
        self.severity_combo.setCurrentText(self.model.severity)
        self.severity_combo.currentTextChanged.connect(self._on_severity_changed)
        self.entry_check = QCheckBox("Alert the moment someone steps in")
        self.entry_check.setToolTip("No time limit: entering the zone is already the violation "
                                    "(good for a window or a door). Untick to use a time limit instead.")
        self.entry_check.toggled.connect(self._on_entry_changed)
        self.direction_check = QCheckBox("Tripwire counts one direction only")
        self.direction_check.setChecked(True)
        self.direction_check.toggled.connect(self._on_direction_changed)
        side.addWidget(self.entry_check)
        side.addWidget(QLabel("Zone severity (sets the time limit)"))
        side.addWidget(self.severity_combo)
        side.addWidget(self.direction_check)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        side.addWidget(self.hint)

        row = QHBoxLayout()
        self.finish_btn = QPushButton("Finish shape")
        self.undo_btn = QPushButton("Undo point")
        self.finish_btn.clicked.connect(self._name_and_commit)
        self.undo_btn.clicked.connect(self._undo_point)
        row.addWidget(self.finish_btn)
        row.addWidget(self.undo_btn)
        side.addLayout(row)

        side.addWidget(QLabel("Saved shapes"))
        self.shape_list = QListWidget()
        self.shape_list.itemSelectionChanged.connect(self._refresh)
        side.addWidget(self.shape_list, 1)
        self.flip_btn = QPushButton("Flip tripwire direction")
        self.flip_btn.setToolTip("The arrow on a tripwire points the way that counts as a violation.")
        self.flip_btn.clicked.connect(self._flip_selected)
        side.addWidget(self.flip_btn)
        self.delete_btn = QPushButton("Delete selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        side.addWidget(self.delete_btn)

        self.save_btn = QPushButton("Save")
        self.cancel_btn = QPushButton("Close")
        self.save_btn.clicked.connect(self.save)
        self.cancel_btn.clicked.connect(self.request_close)
        side.addWidget(self.save_btn)
        side.addWidget(self.cancel_btn)

        outer = QHBoxLayout(self)
        outer.addWidget(self.canvas, 1)
        side_host = QWidget()
        side_host.setLayout(side)
        side_host.setFixedWidth(240)
        outer.addWidget(side_host)
        self._refresh()

    # -- lifecycle ---------------------------------------------------------------
    def load(self, frame_bgr: np.ndarray) -> None:
        """Starts an editing session on `frame_bgr` (a processing-resolution
        frame). Existing shapes are read fresh from disk and rebased into this
        frame's coordinate space."""
        self.model = ZoneEditorModel.from_yaml(self.path)
        self._sync_mode_from_radios()
        self.model.severity = self.severity_combo.currentText()
        self.model.alert_on_entry = self.entry_check.isChecked()
        self.model.direction_sensitive = self.direction_check.isChecked()
        self.model.rebase((frame_bgr.shape[1], frame_bgr.shape[0]))
        self.canvas.set_model(self.model)
        self.canvas.set_frame(frame_bgr)
        self._refresh()
        self.canvas.setFocus()

    @property
    def dirty(self) -> bool:
        return self.model.dirty

    # -- actions -----------------------------------------------------------------
    def _sync_mode_from_radios(self):
        if self.wire_radio.isChecked():
            self.model.set_mode("tripwire")
        else:
            self.model.set_mode("zone")
            self.model.set_zone_shape("box" if self.box_radio.isChecked() else "polygon")

    def _on_mode_changed(self, checked: bool = True):
        if not checked:   # each switch fires twice (old radio off, new radio on); act once
            return
        self._sync_mode_from_radios()
        self._refresh()
        self.canvas.update()

    def _on_severity_changed(self, value: str):
        self.model.severity = value

    def _on_entry_changed(self, checked: bool):
        self.model.alert_on_entry = checked
        self._refresh()

    def _on_direction_changed(self, checked: bool):
        self.model.direction_sensitive = checked

    def _undo_point(self):
        self.model.undo_point()
        self._refresh()
        self.canvas.update()

    def _ask_name(self, default: str, error: str = "") -> Optional[str]:
        label = (error + "\n" if error else "") + "Name:"
        text, ok = QInputDialog.getText(self, "Name this shape", label, text=default)
        return text if ok else None

    def _suggest_name(self) -> str:
        kind, existing = ("Zone", self.model.zones) if self.model.mode == "zone" else ("Tripwire", self.model.tripwires)
        taken = {s.name for s in existing}
        n = len(existing) + 1
        while f"{kind} {n}" in taken:
            n += 1
        return f"{kind} {n}"

    def _name_and_commit(self):
        if not self.model.can_finish():
            return
        default, error = self._suggest_name(), ""
        while True:
            name = self._ask_name(default, error)
            if name is None:                       # cancelled
                # A finished wire or box can't be extended point by point -- drop it.
                if self.model.mode == "tripwire" or self.model.zone_shape == "box":
                    self.model.pending = []
                break
            error = self.model.commit(name) or ""
            if not error:
                break
            default = name
        self._refresh()
        self.canvas.update()

    def _selected(self):
        row = self.shape_list.currentRow()
        if row < 0:
            return None
        return ("zone", row) if row < len(self.model.zones) else ("tripwire", row - len(self.model.zones))

    def _delete_selected(self):
        sel = self._selected()
        if sel:
            self.model.remove_at(*sel)
            self._refresh()
            self.canvas.update()

    def _flip_selected(self):
        sel = self._selected()
        if sel and sel[0] == "tripwire":
            self.model.flip_tripwire(sel[1])
            self._refresh()
            self.canvas.update()

    def save(self) -> bool:
        try:
            self.model.save(self.path)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save zones", f"{self.path}\n\n{exc}")
            return False
        self.saved.emit(self.model.to_config())
        self.closed.emit()
        return True

    def _confirm_discard(self) -> bool:
        return QMessageBox.question(
            self, "Discard changes?", "You have unsaved zone changes. Close without saving?",
        ) == QMessageBox.StandardButton.Yes

    def request_close(self) -> bool:
        """Closes the editor, asking first if there are unsaved changes.
        Returns True if it closed."""
        if self.model.dirty and not self._confirm_discard():
            return False
        self.closed.emit()
        return True

    # -- view refresh -------------------------------------------------------------------
    def _refresh(self):
        m = self.model
        selected = self.shape_list.currentRow()
        self.shape_list.blockSignals(True)
        self.shape_list.clear()
        for z in m.zones:
            self.shape_list.addItem(f"Zone: {z.name} [{'entry alert' if z.alert_on_entry else z.severity}]")
        for t in m.tripwires:
            self.shape_list.addItem(f"Tripwire: {t.name}")
        if 0 <= selected < self.shape_list.count():
            self.shape_list.setCurrentRow(selected)
        self.shape_list.blockSignals(False)

        is_zone = m.mode == "zone"
        self.entry_check.setEnabled(is_zone)
        self.severity_combo.setEnabled(is_zone and not m.alert_on_entry)  # severity only sets the time limit
        self.direction_check.setEnabled(not is_zone)
        self.finish_btn.setEnabled(m.can_finish())
        self.undo_btn.setEnabled(bool(m.pending))
        self.delete_btn.setEnabled(self.shape_list.currentRow() >= 0)
        sel = self._selected()
        self.flip_btn.setEnabled(bool(sel and sel[0] == "tripwire"))
        if is_zone and m.zone_shape == "box":
            hint = ("Press, drag and release to draw a box, then name it. "
                    "Drag a white corner handle to reshape a saved zone.")
        elif is_zone:
            hint = ("Click to add points (3+), then double-click or press Finish. Right-click undoes a point. "
                    "Drag a white handle to reshape a saved shape (Shift-click to start a new one on top of a handle).")
        else:
            hint = None
        self.hint.setText(
            hint if hint else
            "Click two points to place the line. The arrow shows which way counts as a violation "
            "(use Flip to reverse it, or untick the box to count both ways). Right-click undoes a point.")
