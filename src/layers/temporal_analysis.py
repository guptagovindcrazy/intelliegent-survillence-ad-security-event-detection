"""
layers/temporal_analysis.py
----------------------------
LAYER 3 — Temporal Analysis Layer

Responsibility:
    * Maintain per-track history (positions + timestamps + zone occupancy).
    * Compute dwell duration inside restricted zones.
    * Compute instantaneous speed from trajectory, to distinguish loitering
      from someone briskly walking through a monitored area.
    * Run a small finite-state machine per track:

        NORMAL -> ZONE_ENTERED -> DWELLING -> ALERT_CANDIDATE -> ALERTED
                                                                    |
                                                                    v (cooldown expiry / zone exit)
                                                                  NORMAL

Design notes for viva:
    - Debouncing: a track must be observed inside a zone for
      `min_consecutive_frames_in_zone` consecutive frames before dwell timing
      even starts. This absorbs single-frame detection flicker (a common source
      of false positives in frame-level systems).
    - Dwell threshold is severity-dependent (from config), so a high-severity
      zone (e.g. a server room door) alerts faster than a low-severity one
      (e.g. an outer perimeter), matching real security priorities.
    - The state machine — not just "if in zone" — is what actually suppresses
      false alarms: an alert is only ever emitted on the ZONE_ENTERED -> DWELLING
      -> ALERT_CANDIDATE transition chain, never from a single frame's spatial
      reading in isolation.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

from src.config import TemporalConfig
from src.layers.spatial_zones import SpatialState


class TrackState(Enum):
    NORMAL = auto()
    ZONE_ENTERED = auto()
    DWELLING = auto()
    ALERT_CANDIDATE = auto()
    ALERTED = auto()


@dataclass
class TrackHistory:
    positions: Deque[Tuple[float, float, float]] = field(default_factory=deque)  # (x, y, t)
    consecutive_frames_in_zone: int = 0
    zone_entry_time: Optional[float] = None
    current_zone: Optional[str] = None
    state: TrackState = TrackState.NORMAL
    last_alert_time: Optional[float] = None


@dataclass
class TemporalEvent:
    track_id: int
    state: TrackState
    dwell_time_sec: float
    speed_px_s: float
    zone: Optional[str]
    should_alert: bool
    reason: str = ""


class TemporalAnalysisLayer:
    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self.histories: Dict[int, TrackHistory] = {}

    def _get_history(self, track_id: int) -> TrackHistory:
        if track_id not in self.histories:
            self.histories[track_id] = TrackHistory(
                positions=deque(maxlen=self.cfg.trajectory_history_len)
            )
        return self.histories[track_id]

    @staticmethod
    def _speed(history: TrackHistory) -> float:
        """Pixels/second over the most recent two samples."""
        if len(history.positions) < 2:
            return 0.0
        (x1, y1, t1) = history.positions[-2]
        (x2, y2, t2) = history.positions[-1]
        dt = max(t2 - t1, 1e-6)
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / dt

    def update(self, spatial_state: SpatialState, foot_point: Tuple[float, float],
               zone_severity_lookup: Dict[str, str], now: Optional[float] = None) -> TemporalEvent:
        now = now if now is not None else time.time()
        h = self._get_history(spatial_state.track_id)
        h.positions.append((foot_point[0], foot_point[1], now))
        speed = self._speed(h)

        # Only zones on the time-limit rule are timed; "alert on entry" zones
        # are handled by the alert engine and must not also trigger a dwell alert.
        timed_zones = (spatial_state.zones_inside if spatial_state.dwell_zones is None
                       else spatial_state.dwell_zones)
        in_restricted = len(timed_zones) > 0
        zone_name = timed_zones[0] if in_restricted else None

        should_alert = False
        reason = ""

        # ---- state machine transitions ----
        if not in_restricted:
            # exiting a zone (or was never in one) resets everything except cooldown memory
            if h.state == TrackState.ALERTED:
                pass  # keep cooldown timestamp so we don't immediately re-alert on flicker
            h.state = TrackState.NORMAL
            h.consecutive_frames_in_zone = 0
            h.zone_entry_time = None
            h.current_zone = None
            dwell = 0.0

        else:
            if h.current_zone != zone_name:
                # entered a *different* zone than before -> restart debounce/dwell
                h.current_zone = zone_name
                h.consecutive_frames_in_zone = 0
                h.zone_entry_time = None

            h.consecutive_frames_in_zone += 1

            if h.consecutive_frames_in_zone < self.cfg.min_consecutive_frames_in_zone:
                h.state = TrackState.ZONE_ENTERED
                dwell = 0.0
            else:
                if h.zone_entry_time is None:
                    h.zone_entry_time = now  # dwell clock starts only after debounce passes
                dwell = now - h.zone_entry_time
                severity = zone_severity_lookup.get(zone_name, "medium")
                threshold = self.cfg.dwell_threshold_sec.get(severity, 5.0)

                if dwell < threshold:
                    h.state = TrackState.DWELLING
                else:
                    h.state = TrackState.ALERT_CANDIDATE

                    cooldown_ok = (
                        h.last_alert_time is None
                        or (now - h.last_alert_time) > self.cfg.alert_cooldown_sec
                    )
                    if cooldown_ok:
                        h.state = TrackState.ALERTED
                        h.last_alert_time = now
                        should_alert = True
                        loitering = speed < self.cfg.stationary_speed_px_s
                        reason = (
                            f"Track {spatial_state.track_id} dwelled {dwell:.1f}s in "
                            f"'{zone_name}' (threshold {threshold:.1f}s)"
                            + (", stationary/loitering" if loitering else ", still moving")
                        )

        return TemporalEvent(
            track_id=spatial_state.track_id,
            state=h.state,
            dwell_time_sec=dwell,
            speed_px_s=speed,
            zone=zone_name,
            should_alert=should_alert,
            reason=reason,
        )

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        h = self.histories.get(track_id)
        if not h:
            return []
        return [(x, y) for (x, y, _t) in h.positions]
