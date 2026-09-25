"""
alert_engine.py
----------------
Fusion logic: an alert is only raised when Layer 2 (spatial) AND Layer 3 (temporal)
BOTH agree, plus a tripwire crossing is treated as an independent, instant trigger
path (since a tripwire violation has no meaningful "dwell time" -- crossing it once
in the wrong direction is itself the violation).

This module is intentionally thin: it is the single place that decides "alert or not",
so during a viva you can point to one function and explain the entire false-alarm
suppression policy instead of tracing logic scattered across layers.
"""

from dataclasses import dataclass
from typing import Optional

import time

from src.config import ENTRY_CFG, TRIPWIRE_CFG
from src.layers.spatial_zones import SpatialState
from src.layers.temporal_analysis import TemporalEvent

# Kept for callers that import it; the live default is TRIPWIRE_CFG.cooldown_sec
# (src/config.py), which is where to change it.
DEFAULT_TRIPWIRE_COOLDOWN_SEC = TRIPWIRE_CFG.cooldown_sec


@dataclass
class SecurityAlert:
    track_id: int
    alert_type: str        # "DWELL_VIOLATION" | "ZONE_ENTRY_VIOLATION" | "TRIPWIRE_VIOLATION"
    zone_or_wire: str
    message: str
    severity: str = "high"


class AlertEngine:
    def __init__(self, tripwire_cooldown_sec: Optional[float] = None, entry_cooldown_sec: Optional[float] = None):
        self.entry_cooldown_sec = ENTRY_CFG.cooldown_sec if entry_cooldown_sec is None else entry_cooldown_sec
        # (track_id, zone_name) -> last entry-alert time (same idea as tripwires).
        self._entry_alerted_recently: dict = {}
        self.tripwire_cooldown_sec = (
            TRIPWIRE_CFG.cooldown_sec if tripwire_cooldown_sec is None else tripwire_cooldown_sec)
        # (track_id, wire_name) -> last alert time. Without this, a track
        # whose foot-point jitters back and forth across a tripwire line
        # (ordinary detection/tracking noise, not repeated real crossings)
        # would fire a fresh TRIPWIRE_VIOLATION on every single crossing --
        # this dedups those the same way dwell violations already are.
        self._tripwire_alerted_recently: dict = {}

    def evaluate(self, spatial_state: SpatialState, temporal_event: TemporalEvent,
                 now: Optional[float] = None) -> Optional[SecurityAlert]:
        # Path 1: dwell-based violation, already fully debounced + cooldown-managed
        # inside TemporalAnalysisLayer. We simply surface it here.
        if temporal_event.should_alert:
            return SecurityAlert(
                track_id=temporal_event.track_id,
                alert_type="DWELL_VIOLATION",
                zone_or_wire=temporal_event.zone or "unknown_zone",
                message=temporal_event.reason,
                severity="high",
            )

        # Path 1b: stepping into an "alert on entry" zone -- instant, no time limit.
        for zone_name in getattr(spatial_state, "zones_entered", None) or []:
            effective_now = now if now is not None else time.time()
            key = (spatial_state.track_id, zone_name)
            last = self._entry_alerted_recently.get(key)
            if last is not None and (effective_now - last) < self.entry_cooldown_sec:
                continue
            self._entry_alerted_recently[key] = effective_now
            return SecurityAlert(
                track_id=spatial_state.track_id,
                alert_type="ZONE_ENTRY_VIOLATION",
                zone_or_wire=zone_name,
                message=f"Track {spatial_state.track_id} entered restricted zone '{zone_name}'",
                severity="high",
            )

        # Path 2: tripwire violation -- instant. A direction-sensitive wire only
        # counts "in" crossings (so e.g. staff leaving through an entry-only gate
        # line don't fire); a wire marked direction_sensitive=False counts both
        # ways. Cooldown-checked so a rapid back-and-forth can't spam alerts.
        direction = spatial_state.tripwire_direction
        sensitive = getattr(spatial_state, "tripwire_direction_sensitive", True)
        if spatial_state.tripwire_crossed and (direction == "in" or (direction == "out" and not sensitive)):
            effective_now = now if now is not None else time.time()
            key = (spatial_state.track_id, spatial_state.tripwire_crossed)
            last_alert_at = self._tripwire_alerted_recently.get(key)
            if last_alert_at is not None and (effective_now - last_alert_at) < self.tripwire_cooldown_sec:
                return None
            self._tripwire_alerted_recently[key] = effective_now
            return SecurityAlert(
                track_id=spatial_state.track_id,
                alert_type="TRIPWIRE_VIOLATION",
                zone_or_wire=spatial_state.tripwire_crossed,
                message=(
                    f"Track {spatial_state.track_id} crossed tripwire "
                    f"'{spatial_state.tripwire_crossed}' {'inbound' if direction == 'in' else 'outbound'}"
                ),
                severity="high",
            )

        return None
