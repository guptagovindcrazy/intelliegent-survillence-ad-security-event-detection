"""
production/health_monitor.py
------------------------------
Computes an overall system health level and drives graceful degradation,
instead of the system either working perfectly or crashing outright.

Levels (matching the roadmap discussed for this project):
    FULL      -- GPU available, every configured camera healthy, all zones active
    DEGRADED  -- GPU unavailable (CPU fallback) and/or some (but not all) cameras
                 are down; alerts still work for every camera that IS healthy
    CRITICAL  -- EVERY configured camera has inference down (e.g. model failed to
                 load after all retries); raw frames are still captured/logged for
                 later review, but no real-time alerting is happening anywhere

Design notes for viva:
    Observability is treated as a first-class feature, not an afterthought:
    this monitor periodically snapshots the StateStore's per-camera metrics
    (FPS proxy via last_latency_sec, dropped-frame counts, health flags) so
    problems are visible before they become a crash, and exposes a single
    `current_level()` the rest of the system (dashboards, alert dispatcher)
    can key off of.
"""

import threading
import time
from enum import Enum, auto

import torch

from src.production.state_store import StateStore


class SystemLevel(Enum):
    FULL = auto()
    DEGRADED = auto()
    CRITICAL = auto()


class HealthMonitor(threading.Thread):
    def __init__(self, state_store: StateStore, interval_sec: float, all_camera_ids,
                 on_level_change=None):
        super().__init__(daemon=True, name="HealthMonitor")
        self.state_store = state_store
        self.interval_sec = interval_sec
        self.all_camera_ids = set(all_camera_ids)
        self.on_level_change = on_level_change or (lambda old, new: None)
        self._stop_event = threading.Event()
        self._level = SystemLevel.FULL
        self._critical_cameras = set()
        self._lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def mark_inference_down(self, camera_id: str):
        """Called by the orchestrator when a SPECIFIC camera exhausts model-load
        retries (ERR_MODEL_500) with no working fallback. Only that camera is
        considered down -- the system as a whole only goes CRITICAL once every
        configured camera has failed this way; if some cameras are still healthy,
        the system stays DEGRADED, since alerting for those cameras still works."""
        with self._lock:
            self._critical_cameras.add(camera_id)

    def mark_camera_recovered(self, camera_id: str):
        with self._lock:
            self._critical_cameras.discard(camera_id)

    def current_level(self) -> SystemLevel:
        return self._level

    def _compute_level(self) -> SystemLevel:
        with self._lock:
            critical_cameras = set(self._critical_cameras)

        if self.all_camera_ids and critical_cameras >= self.all_camera_ids:
            # every configured camera has inference down -- no camera can alert
            return SystemLevel.CRITICAL

        gpu_ok = torch.cuda.is_available()
        metrics = self.state_store.snapshot()
        any_unhealthy = any(not m.is_healthy for m in metrics.values()) if metrics else False

        if critical_cameras or not gpu_ok or any_unhealthy:
            # some cameras down, and/or CPU fallback -- but at least one camera
            # is still alerting, so this is a degradation, not a full outage
            return SystemLevel.DEGRADED
        return SystemLevel.FULL

    def run(self):
        while not self._stop_event.is_set():
            new_level = self._compute_level()
            if new_level != self._level:
                self.on_level_change(self._level, new_level)
                self._level = new_level
            time.sleep(self.interval_sec)
