"""
frame_geometry.py
------------------
Pure (GUI-free, torch-free) geometry helpers shared by the pipeline, the
zone-editing tools and their tests:

  * downscale_to_max_dim / to_processing_frame -- the ONE definition of "what
    size frame does the pipeline actually run on", so the pipeline and the
    zone editor can never disagree about it.
  * rescale_shapes -- maps zones/tripwires between two frame sizes, so a zone
    drawn on one resolution still lands in the right place on another.
  * FrameViewTransform -- maps between widget pixels and image pixels for a
    letterboxed (aspect-preserving) display, used for click-to-draw.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from src.config import Tripwire, Zone

Size = Tuple[int, int]  # (width, height) in pixels


def downscale_to_max_dim(frame, max_dim: int):
    """Shrinks `frame` so its longer side is at most `max_dim`, preserving
    aspect ratio. Returns `frame` UNCHANGED (same object, no copy, no resize
    call) when it's already within budget."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / float(longest)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_processing_frame(frame, is_live: bool, max_dim: int):
    """The frame the detection/zone pipeline runs on: recorded video is capped
    at `max_dim`, live camera/stream frames are left at native resolution."""
    return frame if is_live else downscale_to_max_dim(frame, max_dim)


def _check_size(size: Size, what: str) -> None:
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"{what} must have positive width and height, got {size}")


def rescale_shapes(zones: List[Zone], tripwires: List[Tripwire],
                   from_size: Size, to_size: Size) -> Tuple[List[Zone], List[Tripwire]]:
    """Returns NEW zones/tripwires with every point mapped from a frame of
    `from_size` to one of `to_size` (x and y scaled independently, so a change
    of aspect ratio still maps the same relative position). The inputs are
    never mutated. Equal sizes return equal copies."""
    _check_size(from_size, "from_size")
    _check_size(to_size, "to_size")
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]

    def pt(p: Tuple[int, int]) -> Tuple[int, int]:
        return (int(round(p[0] * sx)), int(round(p[1] * sy)))

    new_zones = [Zone(z.name, [pt(p) for p in z.polygon], z.severity, z.alert_on_entry) for z in zones]
    new_wires = [Tripwire(t.name, pt(t.p1), pt(t.p2), t.direction_sensitive) for t in tripwires]
    return new_zones, new_wires


@dataclass(frozen=True)
class FrameViewTransform:
    """Maps between widget coordinates and image coordinates when an image of
    `image_size` is drawn centred and aspect-preserved inside a widget of
    `widget_size` (KeepAspectRatio letterboxing)."""
    widget_size: Size
    image_size: Size

    @property
    def scale(self) -> float:
        (ww, wh), (iw, ih) = self.widget_size, self.image_size
        if min(ww, wh, iw, ih) <= 0:
            return 0.0
        return min(ww / iw, wh / ih)

    @property
    def offset(self) -> Tuple[float, float]:
        s = self.scale
        return ((self.widget_size[0] - self.image_size[0] * s) / 2.0,
                (self.widget_size[1] - self.image_size[1] * s) / 2.0)

    def image_rect(self) -> Tuple[float, float, float, float]:
        """(x, y, w, h) of the displayed image in widget coordinates."""
        s, (ox, oy) = self.scale, self.offset
        return (ox, oy, self.image_size[0] * s, self.image_size[1] * s)

    def to_image(self, wx: float, wy: float, clamp: bool = False) -> Optional[Tuple[int, int]]:
        """Widget point -> image pixel. Points in the letterbox margin return
        None, or are clamped to the nearest image edge when `clamp` is True."""
        s = self.scale
        if s <= 0:
            return None
        (ox, oy), (iw, ih) = self.offset, self.image_size
        x, y = (wx - ox) / s, (wy - oy) / s
        inside = 0 <= x <= iw and 0 <= y <= ih
        if not inside and not clamp:
            return None
        x, y = min(max(x, 0), iw - 1), min(max(y, 0), ih - 1)
        return (int(round(x)), int(round(y)))

    def to_widget(self, ix: float, iy: float) -> Tuple[float, float]:
        s, (ox, oy) = self.scale, self.offset
        return (ox + ix * s, oy + iy * s)
