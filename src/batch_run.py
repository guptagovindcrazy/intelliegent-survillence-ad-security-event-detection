"""
batch_run.py
------------
Run the spatio-temporal surveillance pipeline over an entire FOLDER of videos
("a database of videos") instead of just one file, and produce:

  1. alerts.csv        -- one row per alert, machine-readable
  2. summary_report.md -- a plain-English summary per video + overall totals

This does NOT change any detection/tracking/zone/dwell logic -- it simply
loops the existing pipeline (Layers 1-3 + Alert Engine) over every video file
found in a directory, one at a time, and collects the results.

Usage:
    python -m src.batch_run --folder /path/to/videos --out /path/to/results
    python -m src.batch_run --folder /path/to/videos --out results --save-videos

Notes on public research datasets (UCF-Crime, ShanghaiTech, SOMPT22):
    These are large (tens of GB), gated behind an access request / license
    agreement with the original authors -- they are not `pip install`-able.
    Once you've downloaded and unzipped one locally, just point --folder at
    that directory; this script doesn't care where the videos came from.
"""

import argparse
import csv
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2

from src.alert_engine import AlertEngine
from src.config import DET_CFG, RESTRICTED_ZONES, TEMPORAL_CFG, TRACK_CFG, TRIPWIRES, ZONE_CONFIG
from src.layers.detection_tracking import DetectionTrackingLayer
from src.layers.spatial_zones import SpatialZoneLayer
from src.layers.temporal_analysis import TemporalAnalysisLayer
from src.validation import assert_valid_or_raise
from src.zone_runtime import ZoneRuntime

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def build_zone_severity_lookup():
    return {z.name: z.severity for z in RESTRICTED_ZONES}


def process_one_video(video_path: str, save_path: str = None):
    """Runs the exact same 3-layer pipeline as main.py on a single video file,
    but silently (no display window) and returns the list of alerts raised.

    Resilience notes:
      - The whole function body runs under try/finally so cap/writer/model
        resources always get released, even if something raises partway
        through -- otherwise a long batch leaks a cv2 handle (and a temp
        tracker-yaml file, see DetectionTrackingLayer.close()) per video
        that hits trouble, which compounds over a folder of hundreds of
        files into real OS-handle/disk exhaustion.
      - A single bad FRAME (a corrupt/truncated region in an otherwise fine
        video) is caught and skipped rather than aborting the whole video:
        frame_errors on the returned dict records how many, for visibility,
        without losing every alert already found in the rest of that file.
      - Note this function itself can still raise for something more
        fundamental (e.g. model load failure) -- callers (run_batch) are
        responsible for catching that at the per-video level so one file
        doesn't take down the rest of the batch. See run_batch's own
        docstring for why that boundary lives there and not in here.
    """
    detector_tracker = DetectionTrackingLayer(DET_CFG, TRACK_CFG)
    temporal_layer = TemporalAnalysisLayer(TEMPORAL_CFG)
    alert_engine = AlertEngine()
    zones = None  # built from this video's first frame: its size decides any zone rescaling

    cap = None
    writer = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open {video_path}"}

        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        alerts = []
        frame_errors = []
        frame_idx = 0
        start_time = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if zones is None:
                zones = ZoneRuntime.build(ZONE_CONFIG, (frame.shape[1], frame.shape[0]))
            # batch_run only ever processes video FILES (never a live source), so the
            # dwell clock always uses the video's own timeline (CAP_PROP_POS_MSEC),
            # not wall-clock time -- otherwise dwell duration would be wrong whenever
            # a file is processed faster or slower than its real playback duration.
            now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            try:
                tracked_objects = detector_tracker.process(frame)
                for obj in tracked_objects:
                    spatial_state = zones.spatial_layer.evaluate(obj.track_id, obj.foot_point)
                    temporal_event = temporal_layer.update(
                        spatial_state, obj.foot_point, zones.severity_lookup, now=now
                    )
                    alert = alert_engine.evaluate(spatial_state, temporal_event, now=now)
                    if alert:
                        alerts.append({
                            "frame": frame_idx,
                            "track_id": alert.track_id,
                            "alert_type": alert.alert_type,
                            "zone_or_wire": alert.zone_or_wire,
                            "message": alert.message,
                        })
            except Exception as exc:  # noqa: BLE001 -- one bad frame must not lose the whole video
                frame_errors.append(f"frame {frame_idx}: {exc.__class__.__name__}: {exc}")

            if writer:
                frame = zones.spatial_layer.draw_overlays(frame)
                writer.write(frame)

        elapsed = time.time() - start_time
        return {
            "frames_processed": frame_idx,
            "processing_seconds": round(elapsed, 1),
            "alerts": alerts,
            "frame_errors": frame_errors,
        }
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        detector_tracker.close()


def plain_english_summary(video_name: str, result: dict) -> str:
    """Turns raw alert data into a short, non-technical paragraph per video."""
    if "error" in result:
        return f"### {video_name}\n\u26A0\uFE0F Could not process this file: {result['error']}\n"

    alerts = result["alerts"]
    lines = [f"### {video_name}"]
    frame_errors = result.get("frame_errors") or []
    if frame_errors:
        lines.append(f"\u26A0\uFE0F {len(frame_errors)} frame(s) in this video failed to process "
                      f"and were skipped (results below are from the remaining frames).")
    if not alerts:
        lines.append("No security concerns were found in this video \u2014 nothing stayed in a "
                      "restricted area for long enough, nobody entered a no-entry zone, and no tripwire "
                      "was crossed the wrong way.")
        return "\n".join(lines) + "\n"

    by_type = defaultdict(int)
    for a in alerts:
        by_type[a["alert_type"]] += 1

    dwell_n = by_type.get("DWELL_VIOLATION", 0)
    trip_n = by_type.get("TRIPWIRE_VIOLATION", 0)
    entry_n = by_type.get("ZONE_ENTRY_VIOLATION", 0)

    if dwell_n:
        lines.append(f"\U0001F6A8 {dwell_n} time(s), someone stayed too long in a restricted area "
                      f"(a loitering-style violation).")
    if entry_n:
        lines.append(f"\U0001F6A8 {entry_n} time(s), someone stepped into a zone where nobody may enter.")
    if trip_n:
        lines.append(f"\U0001F6A8 {trip_n} time(s), someone crossed a virtual tripwire line they "
                      f"shouldn't have (e.g. entering through a forbidden gate).")

    first = alerts[0]
    lines.append(f"First flagged event: frame {first['frame']} \u2014 \u201C{first['message']}\u201D")
    return "\n".join(lines) + "\n"


def run_batch(folder: str, out_dir: str, save_videos: bool = False):
    """Processes every video in `folder` and writes alerts.csv + summary_report.md.

    Durability design (this is the actual fix for "one bad video loses the
    whole batch's output"):
      - Both output files are opened ONCE, at the top, and kept open with
        every video's results flushed to disk immediately after that video
        finishes -- never buffered in memory until "the end" and written
        in one shot. If video 90/100 crashes the whole process (or the
        machine loses power, or someone Ctrl+C's it), videos 1-89's rows
        and report sections are already safely on disk.
      - The summary report is built by APPENDING each video's section as
        it completes, not by collecting every section in a list and
        joining+rewriting the whole file at the end -- that pattern is
        O(n) work n times (O(n^2) total) as the batch grows, which is
        exactly the kind of thing that quietly turns "processes a folder
        of 20 videos" into "chokes on a folder of 5,000". Appending is
        O(1) work per video, so this scales the same way regardless of
        folder size.
      - A single video raising ANY exception (corrupt file, decoder crash,
        transient OOM, etc.) is caught, logged, and recorded as a failure
        in the report -- the loop moves on to the next file rather than
        losing everything already done and abandoning every video after
        it. See process_one_video's docstring for the matching per-FRAME
        version of this same principle.
    """
    assert_valid_or_raise(RESTRICTED_ZONES, TRIPWIRES)
    os.makedirs(out_dir, exist_ok=True)
    video_files = sorted(
        f for f in Path(folder).iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        print(f"No video files found in {folder} (looked for {VIDEO_EXTENSIONS})")
        return

    csv_path = os.path.join(out_dir, "alerts.csv")
    summary_path = os.path.join(out_dir, "summary_report.md")
    errors_log_path = os.path.join(out_dir, "batch_errors.log")

    total_alerts = 0
    failed_videos = []  # [(video_name, short_error_str), ...]

    with open(csv_path, "w", newline="") as csv_file, \
            open(summary_path, "w") as summary_file:

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["video", "frame", "track_id", "alert_type", "zone_or_wire", "message"])

        summary_file.write("# Batch Surveillance Report\n\n")
        summary_file.write(f"Processing {len(video_files)} video(s) from `{folder}`.\n\n")
        summary_file.write("_(Totals below are filled in as each video finishes; this report is "
                            "safe to read even if the batch is still running or was interrupted.)_\n\n")
        summary_file.flush()

        for i, vf in enumerate(video_files, start=1):
            print(f"[batch] ({i}/{len(video_files)}) processing {vf.name} ...")
            save_path = os.path.join(out_dir, f"annotated_{vf.stem}.mp4") if save_videos else None

            try:
                result = process_one_video(str(vf), save_path=save_path)
            except Exception as exc:  # noqa: BLE001 -- deliberate: isolate one video's failure from the rest of the batch
                print(f"[batch]   FAILED: {exc.__class__.__name__}: {exc}")
                failed_videos.append((vf.name, f"{exc.__class__.__name__}: {exc}"))
                with open(errors_log_path, "a") as err_log:
                    err_log.write(f"=== {vf.name} ===\n{traceback.format_exc()}\n\n")
                result = {"error": f"{exc.__class__.__name__}: {exc} (see batch_errors.log)"}

            if "error" not in result:
                for a in result["alerts"]:
                    csv_writer.writerow(
                        [vf.name, a["frame"], a["track_id"], a["alert_type"], a["zone_or_wire"], a["message"]]
                    )
                total_alerts += len(result["alerts"])
            csv_file.flush()

            summary_file.write(plain_english_summary(vf.name, result))
            summary_file.write("\n")
            summary_file.flush()

        n_ok = len(video_files) - len(failed_videos)
        summary_file.write("---\n\n## Totals\n\n")
        summary_file.write(f"- Videos processed successfully: {n_ok}/{len(video_files)}\n")
        summary_file.write(f"- Total alerts across all videos: {total_alerts}\n")
        if failed_videos:
            summary_file.write(f"- Videos that FAILED to process: {len(failed_videos)} "
                                f"(see `batch_errors.log` for full tracebacks)\n")
            for name, err in failed_videos:
                summary_file.write(f"  - {name}: {err}\n")
        summary_file.flush()

    n_ok = len(video_files) - len(failed_videos)
    print(f"\nDone. {n_ok}/{len(video_files)} video(s) processed successfully, "
          f"{total_alerts} total alerts.")
    if failed_videos:
        print(f"  {len(failed_videos)} video(s) FAILED -- see {errors_log_path} for details:")
        for name, err in failed_videos:
            print(f"    - {name}: {err}")
    print(f"  Detailed CSV  -> {csv_path}")
    print(f"  Plain summary -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-run the surveillance pipeline over a folder of videos")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing video files")
    parser.add_argument("--out", type=str, default="batch_results", help="Folder to write results into")
    parser.add_argument("--save-videos", action="store_true", help="Also save annotated output video per input")
    args = parser.parse_args()

    run_batch(folder=args.folder, out_dir=args.out, save_videos=args.save_videos)
