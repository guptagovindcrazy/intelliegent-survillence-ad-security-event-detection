"""
zone_editor.py
---------------
GUI-free logic behind draw_zones.py: the editing model (points -> named
zone/tripwire -> validated -> saved to config/zones.yaml) and the overlay
renderer. Kept separate from the OpenCV window loop so it can be unit-tested
headlessly.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from src.config import ZONES_YAML_PATH, Tripwire, Zone, ZoneConfig, load_zone_config
from src.frame_geometry import FrameViewTransform, Size, rescale_shapes
from src.layers.spatial_zones import violation_direction
from src.validation import validate_zones_and_tripwires

SEVERITIES = ("low", "medium", "high")

YAML_HEADER = (
    "# Restricted zones and tripwires -- edited by draw_zones.py, or by hand.\n"
    "# Coordinates are pixels [x, y] in the camera frame (origin top-left).\n"
    "# severity: low | medium | high (higher = shorter dwell time before an alert)\n"
    "# alert_on_entry: true = alert the instant someone steps in (no time limit); default = time limit\n"
    "# direction_sensitive: true = only one crossing direction counts as a violation\n"
    "# Empty lists (the default) mean a clean feed with no boxes or alerts.\n"
)


class ZoneEditorModel:
    def __init__(self, zones: Optional[List[Zone]] = None, tripwires: Optional[List[Tripwire]] = None,
                 reference_size: Optional[Size] = None):
        self.zones: List[Zone] = list(zones or [])
        self.tripwires: List[Tripwire] = list(tripwires or [])
        # (width, height) of the frame the shapes' coordinates are in. Saved
        # with the file so the pipeline can rescale to any other resolution.
        self.reference_size: Optional[Size] = reference_size
        self.mode = "zone"                  # "zone" | "tripwire"
        self.zone_shape = "polygon"         # how a zone is drawn: "box" (drag) | "polygon" (click points)
        self.severity = "high"
        self.alert_on_entry = False         # new zones: alert on stepping in (True) or after a time limit (False)
        self.direction_sensitive = True
        self.pending: List[Tuple[int, int]] = []
        self.dirty = False

    @classmethod
    def from_yaml(cls, path: str = ZONES_YAML_PATH) -> "ZoneEditorModel":
        cfg = load_zone_config(path)
        return cls(cfg.zones, cfg.tripwires, cfg.reference_size)

    def rebase(self, frame_size: Size) -> None:
        """Puts the model in the coordinate space of the frame being edited.
        Existing shapes drawn on a different-sized frame are rescaled so they
        appear in the right place; afterwards `reference_size` is that frame's
        size, so a save records what the coordinates now mean. Any in-progress
        (pending) points belong to the old space and are dropped."""
        frame_size = (int(frame_size[0]), int(frame_size[1]))
        if self.reference_size and self.reference_size != frame_size:
            self.zones, self.tripwires = rescale_shapes(
                self.zones, self.tripwires, self.reference_size, frame_size)
            self.pending = []
        self.reference_size = frame_size

    # -- drawing state ----------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.mode = mode
            self.pending = []

    def set_zone_shape(self, shape: str) -> None:
        if shape not in ("box", "polygon"):
            raise ValueError(f"zone shape must be 'box' or 'polygon', got {shape!r}")
        if shape != self.zone_shape:
            self.zone_shape = shape
            self.pending = []

    def set_box(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Sets the pending points to the rectangle spanned by two opposite
        corners, whichever way it was dragged. Returns False (and leaves the
        pending points alone) for a zero-width or zero-height box."""
        (x1, y1), (x2, y2) = a, b
        if x1 == x2 or y1 == y2:
            return False
        left, right = sorted((int(x1), int(x2)))
        top, bottom = sorted((int(y1), int(y2)))
        self.pending = [(left, top), (right, top), (right, bottom), (left, bottom)]
        return True

    def cycle_severity(self) -> str:
        self.severity = SEVERITIES[(SEVERITIES.index(self.severity) + 1) % len(SEVERITIES)]
        return self.severity

    def toggle_direction(self) -> bool:
        self.direction_sensitive = not self.direction_sensitive
        return self.direction_sensitive

    def add_point(self, pt: Tuple[int, int]) -> None:
        if self.mode == "tripwire" and len(self.pending) >= 2:
            return
        self.pending.append((int(pt[0]), int(pt[1])))

    def undo_point(self) -> None:
        if self.pending:
            self.pending.pop()

    def can_finish(self) -> bool:
        if self.mode == "zone":
            return len(self.pending) >= 3
        return len(self.pending) == 2

    # -- committing / deleting ---------------------------------------------
    def commit(self, name: str) -> Optional[str]:
        """Turns the pending points into a named shape. Returns an error
        message (shape NOT added, points kept so the user can retry), or None."""
        name = name.strip()
        if not name:
            return "Name can't be blank"
        if not self.can_finish():
            return "Not enough points"
        zones, wires = list(self.zones), list(self.tripwires)
        if self.mode == "zone":
            zones.append(Zone(name=name, polygon=list(self.pending), severity=self.severity,
                              alert_on_entry=self.alert_on_entry))
        else:
            wires.append(Tripwire(name=name, p1=self.pending[0], p2=self.pending[1],
                                  direction_sensitive=self.direction_sensitive))
        problems = validate_zones_and_tripwires(zones, wires)
        if problems:
            return problems[0]
        self.zones, self.tripwires = zones, wires
        self.pending = []
        self.dirty = True
        return None

    def remove_at(self, kind: str, index: int) -> Optional[str]:
        """Removes one saved shape by kind ("zone" | "tripwire") and index.
        Returns its name, or None if the index is out of range."""
        items = self.zones if kind == "zone" else self.tripwires
        if not 0 <= index < len(items):
            return None
        removed = items.pop(index)
        self.dirty = True
        return removed.name

    def move_vertex(self, kind: str, index: int, vertex: int, pt: Tuple[int, int]) -> bool:
        """Moves one vertex of a saved shape (a zone's polygon point, or a
        tripwire's endpoint: vertex 0 = p1, 1 = p2). Returns True if the shape
        changed. A move that would collapse a tripwire to a zero-length line is
        refused, so the model never holds a shape validation would reject."""
        pt = (int(pt[0]), int(pt[1]))
        if kind == "zone":
            if not (0 <= index < len(self.zones)) or not (0 <= vertex < len(self.zones[index].polygon)):
                return False
            poly = self.zones[index].polygon
            if poly[vertex] == pt:
                return False
            poly[vertex] = pt
        else:
            if not (0 <= index < len(self.tripwires)) or vertex not in (0, 1):
                return False
            wire = self.tripwires[index]
            old, other = (wire.p1, wire.p2) if vertex == 0 else (wire.p2, wire.p1)
            if pt == old or pt == other:
                return False
            if vertex == 0:
                wire.p1 = pt
            else:
                wire.p2 = pt
        self.dirty = True
        return True

    def flip_tripwire(self, index: int) -> bool:
        """Reverses which way counts as a violation by swapping the wire's
        endpoints (the "inside" side is defined by p1 -> p2 order)."""
        if not 0 <= index < len(self.tripwires):
            return False
        wire = self.tripwires[index]
        wire.p1, wire.p2 = wire.p2, wire.p1
        self.dirty = True
        return True

    def delete_last(self) -> Optional[str]:
        """Removes the most recently added shape of the current mode."""
        items = self.zones if self.mode == "zone" else self.tripwires
        if not items:
            return None
        removed = items.pop()
        self.dirty = True
        return removed.name

    # -- persistence ---------------------------------------------------------
    @staticmethod
    def _zone_to_dict(z: Zone) -> dict:
        d = {"name": z.name, "severity": z.severity, "polygon": [list(p) for p in z.polygon]}
        if z.alert_on_entry:          # written only when set, so ordinary zones stay uncluttered
            d["alert_on_entry"] = True
        return d

    def to_config(self) -> ZoneConfig:
        """Snapshot of the saved shapes as a ZoneConfig (copies, so later
        edits don't leak into whoever receives it)."""
        return ZoneConfig(
            zones=[Zone(z.name, list(z.polygon), z.severity, z.alert_on_entry) for z in self.zones],
            tripwires=[Tripwire(t.name, t.p1, t.p2, t.direction_sensitive) for t in self.tripwires],
            reference_size=self.reference_size,
        )

    def to_yaml(self) -> str:
        data = {}
        if self.reference_size:
            data["reference_size"] = list(self.reference_size)
        data |= {
            "zones": [self._zone_to_dict(z) for z in self.zones],
            "tripwires": [{"name": t.name, "p1": list(t.p1), "p2": list(t.p2),
                           "direction_sensitive": t.direction_sensitive} for t in self.tripwires],
        }
        return YAML_HEADER + "\n" + yaml.safe_dump(data, sort_keys=False, default_flow_style=None)

    def save(self, path: str = ZONES_YAML_PATH) -> None:
        with open(path, "w") as f:
            f.write(self.to_yaml())
        self.dirty = False


# ---------------------------------------------------------------------------
# Rendering (BGR)
# ---------------------------------------------------------------------------
_SEV_COLOR = {"low": (0, 200, 255), "medium": (0, 140, 255), "high": (0, 0, 255)}
_ENTRY_COLOR = (200, 0, 200)   # "alert on entry" zones
_WIRE_COLOR = (255, 128, 0)
_PENDING_COLOR = (0, 255, 0)


def _label(img, text, org, color):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def render_overlay(frame: np.ndarray, model: ZoneEditorModel, naming: Optional[str] = None,
                   message: str = "") -> np.ndarray:
    """Returns a copy of `frame` with saved shapes, in-progress points and a
    help/status bar drawn on it. All coordinates are in frame pixels."""
    out = frame.copy()

    fill = out.copy()
    for z in model.zones:
        cv2.fillPoly(fill, [np.array(z.polygon, dtype=np.int32)],
                     _ENTRY_COLOR if z.alert_on_entry else _SEV_COLOR.get(z.severity, (0, 0, 255)))
    out = cv2.addWeighted(fill, 0.25, out, 0.75, 0)
    for z in model.zones:
        color = _ENTRY_COLOR if z.alert_on_entry else _SEV_COLOR.get(z.severity, (0, 0, 255))
        cv2.polylines(out, [np.array(z.polygon, dtype=np.int32)], True, color, 2)
        tag = "entry alert" if z.alert_on_entry else z.severity
        _label(out, f"{z.name} [{tag}]", (z.polygon[0][0] + 4, z.polygon[0][1] + 18), color)
    for t in model.tripwires:
        cv2.line(out, t.p1, t.p2, _WIRE_COLOR, 2)
        if t.direction_sensitive:  # arrow points the way that counts as a violation
            ux, uy = violation_direction(t.p1, t.p2)
            mx, my = (t.p1[0] + t.p2[0]) // 2, (t.p1[1] + t.p2[1]) // 2
            cv2.arrowedLine(out, (mx, my), (int(mx + ux * 40), int(my + uy * 40)), _WIRE_COLOR, 2, tipLength=0.4)
        tag = "1-way" if t.direction_sensitive else "2-way"
        _label(out, f"{t.name} ({tag})", (t.p1[0] + 4, t.p1[1] - 6), _WIRE_COLOR)

    pts = model.pending
    for i, p in enumerate(pts):
        cv2.circle(out, p, 4, _PENDING_COLOR, -1)
        if i:
            cv2.line(out, pts[i - 1], p, _PENDING_COLOR, 1, cv2.LINE_AA)

    # status / help bar
    if naming is not None:
        lines = [f"Name: {naming}_", "Enter = confirm   Esc = cancel"]
    else:
        opts = (f"mode: {model.mode.upper()}"
                + (f"  severity: {model.severity}" if model.mode == "zone"
                   else f"  {'1-way' if model.direction_sensitive else '2-way'}"))
        lines = [opts,
                 "click=add point  Enter=finish  u=undo  x=delete last  z/t=zone/tripwire  "
                 "v=severity  d=direction  s=save  q=quit"]
    if message:
        lines.append(message)
    h = 22 * len(lines) + 8
    out[:h, :] = (out[:h, :] * 0.3).astype(np.uint8)
    for i, text in enumerate(lines):
        _label(out, text, (8, 22 * (i + 1)), (255, 255, 255))
    return out


def nearest_vertex(model: ZoneEditorModel, transform: FrameViewTransform, wx: float, wy: float,
                   radius: float = 9.0) -> Optional[Tuple[str, int, int]]:
    """The saved-shape vertex closest to a widget-space point, within `radius`
    widget pixels, as (kind, shape_index, vertex_index) -- or None. Distance is
    measured on screen (not in image pixels) so the grab area feels the same at
    any zoom. Ties go to the later-drawn shape, which is the one on top."""
    best, best_d2 = None, radius * radius
    shapes = [("zone", i, z.polygon) for i, z in enumerate(model.zones)]
    shapes += [("tripwire", i, [t.p1, t.p2]) for i, t in enumerate(model.tripwires)]
    for kind, i, pts in shapes:
        for v, (px, py) in enumerate(pts):
            sx, sy = transform.to_widget(px, py)
            d2 = (sx - wx) ** 2 + (sy - wy) ** 2
            if d2 <= best_d2:
                best, best_d2 = (kind, i, v), d2
    return best
