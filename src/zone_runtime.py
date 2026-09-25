"""
zone_runtime.py
----------------
The zone-dependent pieces of a pipeline, built together from a ZoneConfig and
the size of the frame actually being processed:

    ZoneConfig (+ optional reference_size) + real frame size
        -> rescale if the sizes differ -> validate -> SpatialZoneLayer + severity lookup

Every entry point (desktop app, single-video CLI, batch runner) goes through
this one function, so "zones drawn at one resolution work at another" is
implemented once, not per caller.
"""

from dataclasses import dataclass
from typing import Dict, List

from src.config import Tripwire, Zone, ZoneConfig
from src.frame_geometry import Size, rescale_shapes
from src.layers.spatial_zones import SpatialZoneLayer
from src.validation import assert_valid_or_raise


@dataclass
class ZoneRuntime:
    zones: List[Zone]
    tripwires: List[Tripwire]
    spatial_layer: SpatialZoneLayer
    severity_lookup: Dict[str, str]

    @classmethod
    def build(cls, config: ZoneConfig, frame_size: Size) -> "ZoneRuntime":
        """Raises ValueError if the (rescaled) configuration is invalid."""
        zones, wires = config.zones, config.tripwires
        if config.reference_size and tuple(config.reference_size) != tuple(frame_size):
            zones, wires = rescale_shapes(zones, wires, config.reference_size, frame_size)
        assert_valid_or_raise(zones, wires)
        return cls(zones, wires, SpatialZoneLayer(zones, wires), {z.name: z.severity for z in zones})
