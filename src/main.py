"""
main.py
-------
Pipeline orchestrator. Wires together:

    Frame --> [Layer 1: Detection & Tracking] --> TrackedObjects
           --> [Layer 2: Spatial Zone Layer]   --> SpatialState (per track)
           --> [Layer 3: Temporal Analysis]    --> TemporalEvent (per track)
           --> [Alert Engine]                  --> SecurityAlert | None

Run:
    python -m src.main --source path/to/video.mp4
    python -m src.main --source 0                 # webcam
    python -m src.main --source rtsp://...         # IP camera stream
"""

import argparse
import time

import cv2

from src.alert_engine import AlertEngine
from src.camera_capture import open_camera_capture
from src.config import DET_CFG, RESTRICTED_ZONES, TEMPORAL_CFG, TRACK_CFG, TRIPWIRES, ZONE_CONFIG
from src.layers.detection_tracking import DetectionTrackingLayer
from src.layers.spatial_zones import SpatialZoneLayer
from src.layers.temporal_analysis import TemporalAnalysisLayer
from src.validation import assert_valid_or_raise
from src.source_resolver import classify_is_live, resolve_source
from src.zone_runtime import ZoneRuntime


def build_zone_severity_lookup():
    return {z.name: z.severity for z in RESTRICTED_ZONES}


def draw_track(frame, obj, spatial_state, temporal_event):
    x1, y1, x2, y2 = map(int, obj.bbox_xyxy)
    in_zone = len(spatial_state.zones_inside) > 0
    color = (0, 0, 255) if temporal_event.state.name in ("ALERT_CANDIDATE", "ALERTED") else (
        (0, 165, 255) if in_zone else (0, 255, 0)
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {obj.track_id} {obj.class_name} {temporal_event.state.name}"
    if temporal_event.dwell_time_sec > 0:
        label += f" {temporal_event.dwell_time_sec:.1f}s"
    cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def is_live_source(source: str) -> bool:
    """Webcam indices and network streams deliver frames in real time, so the
    dwell clock should use wall-clock time for them. A file path does not --
    processing it faster or slower than real-time would otherwise skew dwell
    duration, so for files we use the video's OWN timestamp instead (see
    `frame_clock_seconds`)."""
    return classify_is_live(source)


def frame_clock_seconds(cap, is_live: bool) -> float:
    """Returns the 'now' to feed the temporal-analysis layer. For live sources
    this is real wall-clock time. For a video FILE, this is the video's own
    timeline position (CAP_PROP_POS_MSEC), so dwell time reflects what actually
    happened in the footage regardless of how fast we process it."""
    if is_live:
        return time.time()
    return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


def run(source: str, display: bool = True, save_path: str = None):
    assert_valid_or_raise(RESTRICTED_ZONES, TRIPWIRES)
    detector_tracker = DetectionTrackingLayer(DET_CFG, TRACK_CFG)
    temporal_layer = TemporalAnalysisLayer(TEMPORAL_CFG)
    alert_engine = AlertEngine()
    zones = None  # built from the first frame: its size decides any zone rescaling

    resolved = resolve_source(source)   # a YouTube link becomes a real media address (SourceError if not)
    live = resolved.is_live
    cap, backend_used, open_err = open_camera_capture(resolved.location)
    if cap is None:
        raise RuntimeError(f"Could not open video source: {source}\n{open_err}")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            now = frame_clock_seconds(cap, live)
            if zones is None:
                zones = ZoneRuntime.build(ZONE_CONFIG, (frame.shape[1], frame.shape[0]))

            # ---------------- Layer 1 ----------------
            tracked_objects = detector_tracker.process(frame)

            # ---------------- Layer 2 & 3, per track ----------------
            for obj in tracked_objects:
                spatial_state = zones.spatial_layer.evaluate(obj.track_id, obj.foot_point)
                temporal_event = temporal_layer.update(
                    spatial_state, obj.foot_point, zones.severity_lookup, now=now
                )

                # ---------------- Alert fusion ----------------
                alert = alert_engine.evaluate(spatial_state, temporal_event, now=now)
                if alert:
                    print(f"[ALERT] frame={frame_idx} {alert.alert_type} :: {alert.message}")

                if display or save_path:
                    frame = draw_track(frame, obj, spatial_state, temporal_event)

            if display or save_path:
                frame = zones.spatial_layer.draw_overlays(frame)

            if writer:
                writer.write(frame)
            if display:
                cv2.imshow("Spatio-Temporal Surveillance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        detector_tracker.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatio-Temporal Surveillance Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Video path, webcam index, or RTSP URL")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow window")
    parser.add_argument("--save", type=str, default=None, help="Path to save annotated output video")
    args = parser.parse_args()

    run(source=args.source, display=not args.no_display, save_path=args.save)
