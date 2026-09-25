"""
layers/spatial_zones.py
------------------------
LAYER 2 — Spatial Zone Layer

Responsibility:
    * Represent restricted zones as polygons and tripwires as line segments.
    * For each tracked object, determine:
        (a) whether its ground-point lies inside any restricted polygon
        (b) whether its trajectory just crossed a tripwire, and in which direction
    * Emit a per-track SpatialState consumed by the Temporal Analysis Layer.

Design notes for viva:
    - Point-in-polygon uses cv2.pointPolygonTest — O(n) per query on the polygon
      vertex count, negligible cost versus YOLO inference.
    - We test the bbox's *foot point* (bottom-center), not the raw centroid, because
      a person's centroid can visually sit outside a doorway polygon while their feet
      are already inside it. This cuts a class of false negatives at doorway zones.
    - Tripwire crossing is detected with the sign of the cross product between the
      wire vector and the vector to the object's position, evaluated at t-1 and t.
      A sign flip = crossing; the sign itself tells us the direction (in vs out).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import ENTRY_CFG, TRIPWIRE_CFG, Tripwire, Zone


@dataclass
class SpatialState:
    track_id: int
    zones_inside: List[str] = field(default_factory=list)     # zone names currently occupied (all zones, any rule)
    # The subset of zones_inside that use the TIME-LIMIT rule -- what the
    # dwell timer should watch. None (older callers) means "all of zones_inside".
    dwell_zones: Optional[List[str]] = None
    zones_entered: List[str] = field(default_factory=list)    # "alert on entry" zones stepped into just now
    tripwire_crossed: Optional[str] = None                     # name of wire crossed this frame
    tripwire_direction: Optional[str] = None                   # "in" | "out"
    tripwire_direction_sensitive: bool = True                  # False: the wire counts in BOTH directions


def violation_direction(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Unit vector pointing from the "outside" to the "inside" of a tripwire:
    the way a person must move to be reported as crossing "in". It is derived
    from the same cross-product sign the crossing test uses, so an arrow drawn
    with it always matches what the alert logic actually does. Swapping p1 and
    p2 reverses it. Returns (0, 0) for a zero-length wire."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return (0.0, 0.0)
    return (dy / length, -dx / length)


class SpatialZoneLayer:
    def __init__(self, zones: List[Zone], tripwires: List[Tripwire], tripwire_margin_px: Optional[float] = None,
                 entry_margin_px: Optional[float] = None):
        self.zones = zones
        self.tripwires = tripwires
        self.tripwire_margin_px = TRIPWIRE_CFG.margin_px if tripwire_margin_px is None else tripwire_margin_px
        self.entry_margin_px = ENTRY_CFG.margin_px if entry_margin_px is None else entry_margin_px
        # (track_id, zone_name) -> True if the track is currently INSIDE that
        # "alert on entry" zone (with hysteresis; see _check_entry).
        self._entry_inside: Dict[Tuple[int, str], bool] = {}
        # cache polygons as int32 numpy arrays for cv2
        self._zone_polys = {z.name: np.array(z.polygon, dtype=np.int32) for z in zones}
        # previous-position cache for tripwire crossing detection: {(track_id, wire_name): side}
        self._prev_side: Dict[Tuple[int, str], float] = {}

    # -- polygon test ---------------------------------------------------
    def _point_in_zone(self, point: Tuple[float, float], zone_name: str) -> bool:
        poly = self._zone_polys[zone_name]
        result = cv2.pointPolygonTest(poly, point, measureDist=False)
        return result >= 0  # >=0 means inside or exactly on the boundary

    # -- zone-entry test ---------------------------------------------------
    def _check_entry(self, track_id: int, point: Tuple[float, float], zone: Zone) -> bool:
        """True exactly when this track has just stepped INTO `zone`.

        Hysteresis: the person only becomes "inside" once more than
        entry_margin_px past the edge, and only "outside" once more than that
        far beyond it. Between the two they keep their previous state, so
        jitter along the boundary can't fire repeated entries, while walking
        out and back in later still counts as a new entry."""
        key = (track_id, zone.name)
        dist = cv2.pointPolygonTest(self._zone_polys[zone.name], (float(point[0]), float(point[1])), True)
        was_inside = self._entry_inside.get(key, False)
        if dist > self.entry_margin_px:
            now_inside = True
        elif dist < -self.entry_margin_px:
            now_inside = False
        else:
            now_inside = was_inside
        self._entry_inside[key] = now_inside
        return now_inside and not was_inside

    # -- tripwire test ----------------------------------------------------
    @staticmethod
    def _side_of_line(p1, p2, point) -> float:
        """Cross product sign: >0 one side, <0 the other side, 0 = exactly on the line."""
        (x1, y1), (x2, y2) = p1, p2
        (px, py) = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def _check_tripwire(self, track_id: int, point: Tuple[float, float], wire: Tripwire):
        key = (track_id, wire.name)
        (x1, y1), (x2, y2) = wire.p1, wire.p2
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length == 0:
            return None, None
        # Signed distance from the line in pixels (+ one side, - the other).
        dist = self._side_of_line(wire.p1, wire.p2, point) / length

        # Dead band: too close to the line to say which side we're on. Keep the
        # last confirmed side untouched, so wobble around the line can't flip
        # it, while a real crossing (out of the band, on the other side) still
        # registers however many times it repeats.
        if abs(dist) <= self.tripwire_margin_px:
            return None, None

        current_side = 1 if dist > 0 else -1
        prev_side = self._prev_side.get(key)
        self._prev_side[key] = current_side
        if prev_side is None or prev_side == current_side:
            return None, None  # first observation of this track, or no side change

        direction = "in" if current_side < 0 else "out"
        return wire.name, direction

    # -- public API -------------------------------------------------------
    def evaluate(self, track_id: int, foot_point: Tuple[float, float]) -> SpatialState:
        state = SpatialState(track_id=track_id)

        dwell_zones = []
        for zone in self.zones:
            if self._point_in_zone(foot_point, zone.name):
                state.zones_inside.append(zone.name)
                if not zone.alert_on_entry:
                    dwell_zones.append(zone.name)
            if zone.alert_on_entry and self._check_entry(track_id, foot_point, zone):
                state.zones_entered.append(zone.name)
        state.dwell_zones = dwell_zones

        for wire in self.tripwires:
            name, direction = self._check_tripwire(track_id, foot_point, wire)
            if name is not None:
                state.tripwire_crossed = name
                state.tripwire_direction = direction
                state.tripwire_direction_sensitive = wire.direction_sensitive
                break  # one crossing event per frame is enough for the alert logic

        return state

    # -- drawing helpers (OpenCV) ------------------------------------------
    def draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        for zone in self.zones:
            pts = self._zone_polys[zone.name].reshape((-1, 1, 2))
            if zone.alert_on_entry:
                color, label = (200, 0, 200), f"{zone.name} (entry alert)"
            else:
                color, label = ((0, 0, 255) if zone.severity == "high" else (0, 165, 255)), zone.name
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, label, zone.polygon[0], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        for wire in self.tripwires:
            cv2.line(frame, wire.p1, wire.p2, (255, 0, 0), 2)
            if wire.direction_sensitive:  # arrow = the direction that raises an alert
                ux, uy = violation_direction(wire.p1, wire.p2)
                mx, my = (wire.p1[0] + wire.p2[0]) // 2, (wire.p1[1] + wire.p2[1]) // 2
                cv2.arrowedLine(frame, (mx, my), (int(mx + ux * 40), int(my + uy * 40)),
                                (255, 0, 0), 2, tipLength=0.4)
            cv2.putText(frame, wire.name, wire.p1, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
        return frame
