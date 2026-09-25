"""
desktop_app/pipeline_worker.py
--------------------------------
Runs the EXISTING 3-layer pipeline (Layers 1/2/3 + AlertEngine, unchanged) on
a background Qt thread so the UI never freezes while frames are processed.
Turns pipeline output into Qt signals the main window reacts to:

  frame_ready(np.ndarray)   -- next annotated frame, for the live preview panel
  alert_raised(AlertRecord) -- a new alert, for the live alert list
  status_changed(str)      -- short status text for the status bar
  finished_run(RunSummary) -- everything needed to build the report

This file does not reimplement any detection/tracking/zone/dwell logic --
it only wires the existing src/ modules to a QThread and to a video source
(file, webcam index, or RTSP URL), exactly like src/main.py's run() loop.
"""

import threading
import time
from typing import Optional

import cv2
from PySide6.QtCore import QThread, Signal

from desktop_app.models import AlertRecord, RunSummary
from src.alert_engine import AlertEngine
from src.camera_capture import open_camera_capture
from src.config import DET_CFG, TEMPORAL_CFG, TRACK_CFG, ZoneConfig, load_zone_config
from src.frame_geometry import downscale_to_max_dim, to_processing_frame
from src.layers.detection_tracking import DetectionTrackingLayer
from src.layers.spatial_zones import SpatialZoneLayer
from src.layers.temporal_analysis import TemporalAnalysisLayer
from src.main import frame_clock_seconds
from src.source_resolver import SourceError, resolve_source
from src.validation import assert_valid_or_raise
from src.zone_runtime import ZoneRuntime

# How often (wall-clock seconds) to push a "Processing frame N / M (P%)"
# status update for file-based runs. Throttled by time rather than frame
# count so it behaves sensibly at both extremes: a few seconds/frame on a
# slow CPU (still want to hear from it every ~0.5s, not once every 300
# frames) and dozens of fps on a fast decode (don't want to flood the
# status bar/Qt event queue with an emit on every single frame).
PROGRESS_UPDATE_INTERVAL_SEC = 0.5

# Longer side (pixels) a frame is allowed to reach before file-based runs
# downscale it prior to detection/drawing/zone-evaluation. Frames already
# at or under this are returned unchanged -- see config.py's
# DetectionConfig.max_processing_dim for why this exists separately from
# `imgsz`. Only applied to file sources: live camera/RTSP zone calibration
# behavior is left exactly as it was.
DETECTION_MAX_DIM = DET_CFG.max_processing_dim

# Longer side (pixels) a frame is allowed to reach before being handed to
# the live-preview panel. This is deliberately independent of
# DETECTION_MAX_DIM and applies to EVERY source (file, camera, stream):
# it only affects the copy of the frame sent to the UI for display, never
# the frame used for detection/zone-evaluation/alert-snapshot encoding, so
# it cannot change alert behavior -- only how expensive it is to paint the
# preview. Converting and painting a full 4K QPixmap on every frame is real,
# avoidable UI-thread cost even when detection itself is already cheap.
PREVIEW_MAX_DIM = 960

# Hard ceiling on how many alert snapshots (PNG-encoded annotated frames)
# stay in memory for one run. Every alert's METADATA (frame index, time,
# track, type, message) is always kept -- that's small (a few hundred
# bytes) and is the audit trail, so it's never dropped. The snapshot
# image is the expensive part (can be hundreds of KB to a few MB each at
# high resolution), and with no cap a long unattended run on a busy scene
# accumulates these without bound: at even 300KB/snapshot, a few thousand
# alerts over a multi-hour run is already into the hundreds of MB held in
# RAM, then base64-inflated (~1.33x) again into the HTML report on save.
# Past this cap, later alerts are still fully recorded, just without a
# retained image -- report_generator.py already handles a record with no
# snapshot_png gracefully (it simply doesn't render an image for that row).
MAX_ALERT_SNAPSHOTS = 500


# Kept under its old name: tests and callers import it from here. The
# implementation lives in src/frame_geometry.py so the zone editor shares the
# exact same definition of "the frame the pipeline runs on".
_downscale_to_max_dim = downscale_to_max_dim


_ZoneRuntime = ZoneRuntime  # shared implementation: src/zone_runtime.py


def _alert_report_time(now: float, is_live: bool, run_start: float) -> float:
    """Seconds into the run to show in the report. A video file's `now` is
    already its own timeline; a live source's `now` is wall-clock epoch time,
    which must be made relative to when the run began (otherwise the report
    prints absolute timestamps like '29830632m 20.8s')."""
    return now - run_start if is_live else now


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


class PipelineWorker(QThread):
    frame_ready = Signal(object)
    alert_raised = Signal(object)
    status_changed = Signal(str)
    finished_run = Signal(object)
    # A clean (un-annotated) frame at processing resolution, delivered once per
    # request_snapshot() call -- the picture the zone editor draws on.
    snapshot_ready = Signal(object)

    def __init__(self, source: str, capture_snapshots: bool = True,
                 zone_config: Optional[ZoneConfig] = None, parent=None):
        super().__init__(parent)
        self.source = source
        self.capture_snapshots = capture_snapshots
        self._stop_requested = False
        self._snapshots_retained = 0  # see MAX_ALERT_SNAPSHOTS
        # None -> read config/zones.yaml fresh when the run starts, so zones
        # saved since the app launched apply without a restart.
        self._initial_zone_config = zone_config
        # Cross-thread mailboxes. The UI thread only ever sets these; the
        # worker thread consumes them at a frame boundary, so no pipeline
        # object is ever touched from two threads.
        self._zone_lock = threading.Lock()
        self._pending_zone_config: Optional[ZoneConfig] = None
        self._snapshot_requested = threading.Event()

    def stop(self):
        self._stop_requested = True

    def update_zones(self, config: ZoneConfig) -> None:
        """Thread-safe hot reload: the new zones take effect on the next
        frame. Per-track dwell timers are reset (they were measured against
        the old zones); alert cooldowns are kept."""
        with self._zone_lock:
            self._pending_zone_config = config

    def request_snapshot(self) -> None:
        """Thread-safe: ask for one clean frame via snapshot_ready. Costs one
        frame copy, only when asked -- nothing per-frame otherwise."""
        self._snapshot_requested.set()

    def _take_pending_zone_config(self) -> Optional[ZoneConfig]:
        with self._zone_lock:
            cfg, self._pending_zone_config = self._pending_zone_config, None
        return cfg

    def run(self):
        summary = RunSummary(source=self.source, is_live=False)
        self.status_changed.emit("Preparing the video source\u2026")
        try:
            resolved = resolve_source(self.source)   # YouTube link -> real media address
        except SourceError as exc:
            summary.error = str(exc)
            summary.error_category = "source_open"
            self.finished_run.emit(summary)
            return
        summary.is_live = resolved.is_live

        zone_config = self._initial_zone_config or load_zone_config()
        try:
            assert_valid_or_raise(zone_config.zones, zone_config.tripwires)
        except ValueError as exc:
            summary.error = f"Zone/tripwire configuration is invalid:\n{exc}"
            summary.error_category = "zone_config"
            self.finished_run.emit(summary)
            return

        self.status_changed.emit("Loading detection model\u2026")
        try:
            detector_tracker = DetectionTrackingLayer(DET_CFG, TRACK_CFG)
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            summary.error = f"Could not load the detection model:\n{exc}"
            summary.error_category = "model_load"
            self.finished_run.emit(summary)
            return

        temporal_layer = TemporalAnalysisLayer(TEMPORAL_CFG)
        alert_engine = AlertEngine()
        # Built from the first real frame (its size decides any rescaling),
        # via the same path a hot reload uses.
        runtime: Optional[_ZoneRuntime] = None
        with self._zone_lock:
            # Don't clobber an update_zones() that arrived before run() began.
            if self._pending_zone_config is None:
                self._pending_zone_config = zone_config

        cap, backend_used, open_err = open_camera_capture(resolved.location)
        if cap is None:
            summary.error = f"Could not open video source:\n{self.source}\n\n{open_err}"
            summary.error_category = "source_open"
            self.finished_run.emit(summary)
            return

        self.status_changed.emit(f"Running \u2014 {self.source}")
        frame_idx = 0
        wall_start = time.time()
        last_progress_emit = wall_start
        total_frames = 0
        if not summary.is_live:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        stride = DET_CFG.frame_stride
        analysed = 0  # frames actually run through detection (frame_idx counts every source frame)
        first_frame_processed_ok = False
        frame_errors = 0
        fatal_processing_error = None
        zone_error = None

        try:
            while not self._stop_requested:
                # frame_stride > 1: frames we won't analyse are only grab()bed --
                # advanced past without being decoded into an image, which is
                # what makes skipping actually cheap. frame_idx still counts
                # every source frame so progress/ETA and alert frame numbers
                # stay in source-video terms.
                analyse = stride == 1 or frame_idx % stride == 0
                if analyse:
                    ok, frame = cap.read()
                else:
                    ok, frame = cap.grab(), None
                if not ok:
                    break
                frame_idx += 1
                if not analyse:
                    continue
                analysed += 1
                now = frame_clock_seconds(cap, summary.is_live)

                # File-based (recorded video) sources only: cap frame size
                # before detection so a 4K source doesn't force full-res
                # copies/drawing/encoding through the rest of this loop. A
                # no-op for frames already <= DETECTION_MAX_DIM (typical
                # webcam-resolution test clips included), and never applied
                # to live camera/RTSP so their zone calibration is untouched.
                frame = to_processing_frame(frame, summary.is_live, DETECTION_MAX_DIM)

                # Zone (re)load happens at a frame boundary, on this thread,
                # so the pipeline objects are never shared with the UI thread.
                pending = self._take_pending_zone_config()
                if pending is not None:
                    frame_size = (frame.shape[1], frame.shape[0])
                    try:
                        new_runtime = _ZoneRuntime.build(pending, frame_size)
                    except ValueError as exc:
                        if runtime is None:
                            zone_error = exc  # nothing valid to run with
                            break
                        self.status_changed.emit(f"Zone update rejected, keeping previous zones: {exc}")
                    else:
                        if runtime is not None:
                            temporal_layer = TemporalAnalysisLayer(TEMPORAL_CFG)
                            self.status_changed.emit(
                                f"Zones updated \u2014 {len(new_runtime.zones)} zone(s), "
                                f"{len(new_runtime.tripwires)} tripwire(s)")
                        runtime = new_runtime
                        summary.zones, summary.tripwires = runtime.zones, runtime.tripwires

                if self._snapshot_requested.is_set():
                    self._snapshot_requested.clear()
                    self.snapshot_ready.emit(frame.copy())  # before anything is drawn on it

                # Progress feedback for file-based runs -- this is the actual
                # bug fix: previously nothing updated the status bar between
                # "Running -- <source>" and "Finished", so a slow CPU run on
                # a large/crowded video looked frozen for its entire duration
                # even though it was genuinely working. Always emit on the
                # very first frame (so slow per-frame processing still shows
                # something immediately) and otherwise throttle by wall time.
                if not summary.is_live:
                    now_wall = time.time()
                    if frame_idx == 1 or now_wall - last_progress_emit >= PROGRESS_UPDATE_INTERVAL_SEC:
                        last_progress_emit = now_wall
                        if total_frames > 0:
                            pct = min(100, int(100 * frame_idx / total_frames))
                            elapsed = now_wall - wall_start
                            rate = frame_idx / elapsed if elapsed > 0 else 0
                            eta = f" \u2014 ~{_format_eta((total_frames - frame_idx) / rate)} remaining" if rate > 0 else ""
                            self.status_changed.emit(
                                f"Processing frame {frame_idx} / {total_frames} ({pct}%){eta}"
                            )
                        else:
                            # CAP_PROP_FRAME_COUNT is unreliable/unavailable for
                            # some containers -- still say something rather
                            # than going silent.
                            self.status_changed.emit(f"Processing frame {frame_idx}\u2026")

                # Everything from detection through alert-recording is
                # wrapped so one bad frame can't take the whole run down
                # silently -- this is the actual fix for "the UI just sits
                # on 'Processing frame 1/N' forever with no error shown".
                # Before this, an exception here (classic cause: device="cuda"
                # requested but no CUDA GPU present) propagated straight out
                # of run() -- QThread swallows an uncaught exception in its
                # run() method, so finished_run never fires and the app looks
                # permanently frozen with no error at all.
                try:
                    tracked_objects = detector_tracker.process(frame)
                    frame_alerts = []

                    for obj in tracked_objects:
                        x1, y1, x2, y2 = map(int, obj.bbox_xyxy)
                        spatial_state = runtime.spatial_layer.evaluate(obj.track_id, obj.foot_point)
                        temporal_event = temporal_layer.update(
                            spatial_state, obj.foot_point, runtime.severity_lookup, now=now
                        )
                        alert = alert_engine.evaluate(spatial_state, temporal_event, now=now)

                        in_zone = len(spatial_state.zones_inside) > 0
                        color = (0, 0, 255) if alert else ((0, 165, 255) if in_zone else (0, 255, 0))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID {obj.track_id} {temporal_event.state.name}"
                        cv2.putText(frame, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        if alert:
                            frame_alerts.append(alert)

                    frame = runtime.spatial_layer.draw_overlays(frame)

                    # Preview downscale is independent of, and in addition to,
                    # the detection-side cap above: it runs for every source
                    # (including live camera/RTSP) and only affects what's sent
                    # to the UI for painting. Zone evaluation and the alert
                    # snapshot below both already happened against `frame` at
                    # whatever resolution detection used, so this cannot change
                    # alert behavior.
                    preview_frame = downscale_to_max_dim(frame, PREVIEW_MAX_DIM)
                    self.frame_ready.emit(preview_frame.copy())

                    if frame_alerts:
                        snapshot_bytes = None
                        # Only bother encoding a PNG at all if we're still under
                        # the retention cap -- no point spending the encode cost
                        # on an image we're about to discard.
                        if self.capture_snapshots and self._snapshots_retained < MAX_ALERT_SNAPSHOTS:
                            ok_enc, buf = cv2.imencode(".png", frame)
                            if ok_enc:
                                snapshot_bytes = buf.tobytes()
                                self._snapshots_retained += 1
                                if self._snapshots_retained == MAX_ALERT_SNAPSHOTS:
                                    self.status_changed.emit(
                                        f"Note: reached the {MAX_ALERT_SNAPSHOTS}-snapshot memory cap for this "
                                        f"run \u2014 alerts are still being recorded and counted, just without a "
                                        f"retained image from here on."
                                    )
                        for alert in frame_alerts:
                            record = AlertRecord(
                                frame_index=frame_idx,
                                video_time_sec=_alert_report_time(now, summary.is_live, wall_start),
                                track_id=alert.track_id,
                                alert_type=alert.alert_type,
                                zone_or_wire=alert.zone_or_wire,
                                message=alert.message,
                                severity=alert.severity,
                                snapshot_png=snapshot_bytes,
                            )
                            summary.alerts.append(record)
                            self.alert_raised.emit(record)

                    first_frame_processed_ok = True
                except Exception as exc:  # noqa: BLE001 -- deliberate: see comment above
                    if not first_frame_processed_ok:
                        # Nothing has EVER succeeded -- this is a systemic
                        # failure (device/GPU mismatch is the classic cause),
                        # not a one-off bad frame. Every remaining frame would
                        # fail identically, so stop now with a clear error
                        # instead of silently "processing" all the way to
                        # 100% having done nothing.
                        fatal_processing_error = exc
                        break
                    # At least one frame already worked, so the pipeline is
                    # fundamentally fine -- treat this as one isolated bad
                    # frame (same principle as batch_run.py's per-frame
                    # guard) and keep going rather than losing the rest of
                    # an otherwise-working run.
                    frame_errors += 1
        finally:
            cap.release()
            detector_tracker.close()

        if zone_error is not None:
            summary.error = f"Zone/tripwire configuration is invalid for this video's frame size:\n{zone_error}"
            summary.error_category = "zone_config"
            self.finished_run.emit(summary)
            return

        if fatal_processing_error is not None:
            summary.error = (
                f"Processing failed on the first frame:\n{fatal_processing_error}\n\n"
                f"This usually means a device/GPU mismatch (the detection config "
                f"requesting a GPU that isn't available on this machine) rather than "
                f"a problem with this specific video or camera."
            )
            summary.error_category = "processing"
            self.finished_run.emit(summary)
            return

        summary.frames_processed = analysed
        summary.duration_sec = time.time() - wall_start
        summary.stopped_early = self._stop_requested
        summary.frame_errors = frame_errors
        if frame_errors:
            self.status_changed.emit(f"Finished \u2014 {frame_errors} frame(s) failed to process and were skipped.")
        else:
            self.status_changed.emit("Finished")
        self.finished_run.emit(summary)
