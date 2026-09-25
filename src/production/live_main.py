"""
production/live_main.py
-------------------------
Wires together the full production architecture and runs it until interrupted:

    CAMERAS (config)
        |
        v
    [CameraSource threads]  --(frame, ts)-->  [one queue PER camera_id]
                                                                |
                                                                v
                                                   [InferenceWorker pool] <--> [StateStore]
                                                                |                  ^
                                                                v                  |
                                                        [alert dispatcher]   [HealthMonitor]

Run:
    python -m src.production.live_main
    python -m src.production.live_main --duration 60   # auto-stop after N seconds (demo/testing)
"""

import argparse
import queue
import signal
import time

from src.config import CAMERAS, PROD_CFG, RESTRICTED_ZONES
from src.production.camera_source import CameraSource
from src.production.errors import SurveillanceError, zone_config_invalid
from src.production.health_monitor import HealthMonitor, SystemLevel
from src.production.inference_worker import InferenceWorker, make_layer_factory
from src.production.state_store import StateStore
from src.validation import validate_zones_and_tripwires


def build_zone_severity_lookup():
    return {z.name: z.severity for z in RESTRICTED_ZONES}


def default_error_sink(err: SurveillanceError):
    print(f"[ERROR] {err}")


def default_alert_dispatcher(camera_id: str, alert):
    print(f"[ALERT] camera={camera_id} {alert.alert_type} :: {alert.message}")


def on_level_change(old_level, new_level):
    print(f"[SYSTEM] health level changed: {old_level.name} -> {new_level.name}")
    if new_level == SystemLevel.CRITICAL:
        print("[SYSTEM] CRITICAL: real-time alerting is down. Raw frames are still "
              "being captured; this incident needs manual review.")
    elif new_level == SystemLevel.DEGRADED:
        print("[SYSTEM] DEGRADED: running on CPU fallback and/or some cameras are "
              "unhealthy. Alerts continue for all healthy cameras.")


def run(duration_sec: float = None):
    zone_severity_lookup = build_zone_severity_lookup()

    # Validate each camera's zone/tripwire config up front (ERR_ZONE_CONFIG_400).
    # A camera with bad config is excluded rather than crashing the whole system --
    # every other correctly-configured camera still runs.
    valid_cameras = []
    for cam_cfg in CAMERAS:
        problems = validate_zones_and_tripwires(cam_cfg.zones, cam_cfg.tripwires)
        if problems:
            default_error_sink(zone_config_invalid(cam_cfg.camera_id, "; ".join(problems)))
        else:
            valid_cameras.append(cam_cfg)

    if not valid_cameras:
        print("[SYSTEM] no camera has a valid zone/tripwire configuration -- nothing to run.")
        return

    cameras_by_id = {c.camera_id: c for c in valid_cameras}
    # One bounded queue PER CAMERA -- not one shared queue for everyone.
    # This is what makes each camera's backpressure/eviction genuinely
    # scoped to itself (see camera_source.py's module docstring): a busy
    # camera filling its own queue can never evict or starve another
    # camera's frames, because there's nothing shared left to contend over.
    frame_queues: "dict[str, queue.Queue]" = {
        c.camera_id: queue.Queue(maxsize=PROD_CFG.frame_queue_maxsize) for c in valid_cameras
    }

    error_sink = default_error_sink
    layer_factory = make_layer_factory(cameras_by_id, PROD_CFG, error_sink)
    state_store = StateStore(layer_factory)

    health_monitor = HealthMonitor(
        state_store, PROD_CFG.health_check_interval_sec,
        all_camera_ids=list(cameras_by_id.keys()), on_level_change=on_level_change,
    )

    camera_sources = [
        CameraSource(cam_cfg, PROD_CFG, frame_queues[cam_cfg.camera_id], error_sink, state_store)
        for cam_cfg in valid_cameras
    ]
    workers = [
        InferenceWorker(
            i, frame_queues, state_store, zone_severity_lookup, default_alert_dispatcher,
            error_sink=error_sink, on_fatal_error=lambda cam_id, err: health_monitor.mark_inference_down(cam_id),
        )
        for i in range(PROD_CFG.num_inference_workers)
    ]

    print(f"[SYSTEM] starting {len(camera_sources)} camera(s), "
          f"{len(workers)} inference worker(s), {PROD_CFG.frame_queue_maxsize} frames/camera queue capacity")

    for cs in camera_sources:
        cs.start()
    for w in workers:
        w.start()
    health_monitor.start()

    stop_flag = {"stop": False}

    def _handle_sigint(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    start_time = time.time()
    try:
        while not stop_flag["stop"]:
            if duration_sec is not None and (time.time() - start_time) > duration_sec:
                break
            time.sleep(1.0)
    finally:
        print("[SYSTEM] shutting down...")
        for cs in camera_sources:
            cs.stop()
        for w in workers:
            w.stop()
        health_monitor.stop()
        time.sleep(0.5)  # let threads notice the stop event

        # final metrics summary -- observability as a first-class feature
        for cam_id, m in state_store.snapshot().items():
            print(f"[SUMMARY] {cam_id}: frames={m.frames_processed} "
                  f"alerts={m.alerts_raised} dropped={m.dropped_frames} "
                  f"healthy={m.is_healthy} last_latency={m.last_latency_sec:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live multi-camera surveillance pipeline")
    parser.add_argument("--duration", type=float, default=None,
                         help="Auto-stop after this many seconds (omit to run until Ctrl+C)")
    args = parser.parse_args()
    run(duration_sec=args.duration)
