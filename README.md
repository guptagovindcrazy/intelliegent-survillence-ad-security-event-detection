# Spatio-Temporal Surveillance Pipeline

Context-aware CCTV pipeline: RT-DETR detection → ByteTrack persistent tracking →
polygon zone / tripwire checks → dwell-time & trajectory reasoning → fused alert engine.

## Structure
```
draw_zones.py                     # click-to-draw zone/tripwire setup tool -> config/zones.yaml
src/
  config.py                     # all tunables: cameras, thresholds, model settings (zones load from config/zones.yaml)
  zone_editor.py                 # GUI-free zone editing model + OpenCV overlay (draw_zones.py)
  frame_geometry.py              # processing-frame sizing, zone rescaling, widget<->image mapping
  alert_engine.py                # fusion logic: spatial AND temporal -> SecurityAlert
  main.py                        # single-video orchestrator / CLI entry point
  batch_run.py                    # process a whole folder of videos -> alerts.csv + summary_report.md
  layers/
    detection_tracking.py        # LAYER 1: RT-DETR (or YOLO) + ByteTrack
    spatial_zones.py              # LAYER 2: polygon zones + tripwires (OpenCV)
    temporal_analysis.py          # LAYER 3: dwell time, trajectory, state machine
  production/
    errors.py                     # structured error taxonomy (ERR_CAM_404, ERR_MODEL_500, ...)
    state_store.py                 # central per-camera state, shared by stateless workers
    camera_source.py               # producer: one capture thread per camera -> shared frame queue
    inference_worker.py            # consumer: worker pool running the 3-layer pipeline + alerts
    health_monitor.py              # FULL / DEGRADED / CRITICAL graceful-degradation levels
    live_main.py                   # orchestrator / CLI entry point for live, multi-camera mode
desktop_app/
  main.py                          # GUI entry point: python -m desktop_app.main
  main_window.py                    # the window: source picker, live preview, alert list, report button
  pipeline_worker.py                 # runs the same 3-layer pipeline on a background Qt thread
  report_generator.py                # builds the self-contained HTML report
  models.py                          # shared, dependency-free data classes (AlertRecord, RunSummary)
  zone_editor_panel.py               # in-app click-to-draw zone editor (canvas + controls)
  snapshot_worker.py                 # grabs one clean frame off the GUI thread for the editor
```

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python -m src.main --source path/to/video.mp4
python -m src.main --source 0                      # webcam
python -m src.main --source rtsp://camera-ip/stream --no-display --save out.mp4
```

## Run on a whole folder of videos ("a database of videos")
```bash
python -m src.batch_run --folder /path/to/videos --out results
python -m src.batch_run --folder /path/to/videos --out results --save-videos   # also saves annotated copies
```
This loops the same pipeline over every `.mp4/.avi/.mov/.mkv/.m4v` file in the folder
and writes two files into `results/`:
- `alerts.csv` — one row per alert, with the video name, frame number, track ID, and message (machine-readable, good for spreadsheets or further analysis)
- `summary_report.md` — a short, plain-English paragraph per video ("no concerns found" / "flagged 2 loitering events") plus a total count across all videos — good for a quick read or pasting into a report

Note on public benchmark datasets (UCF-Crime, ShanghaiTech, SOMPT22): these are large
research datasets (tens of GB) that require requesting access from the original authors —
they aren't `pip install`-able. Once you've downloaded and unzipped one locally, point
`--folder` at that directory and the batch runner works the same way regardless of where
the videos came from.

## Live, multi-camera production mode
```bash
python -m src.production.live_main                    # runs until Ctrl+C
python -m src.production.live_main --duration 60       # auto-stop after 60s (useful for demos)
```
This is the scaled-up architecture, separate from the single-file `main.py` / batch
`batch_run.py` paths above:

- **Producer/consumer**: each camera (configured in `CAMERAS` in `src/config.py`) runs
  its own lightweight capture thread (`CameraSource`) feeding a shared, bounded frame
  queue. A slow or dead camera never blocks the others.
- **Stateless inference workers**: a small pool of worker threads (`InferenceWorker`)
  pull frames from that shared queue. All camera-specific state — track IDs, zone
  occupancy, dwell timers, per-camera metrics — lives centrally in the `StateStore`,
  keyed by `camera_id`, not inside any one worker. That's what lets you add worker
  threads as camera count grows without re-architecting how cameras map to workers.
- **Structured errors** (`src/production/errors.py`): failures are `SurveillanceError`
  objects with a stable code (`ERR_CAM_404`, `ERR_MODEL_500`, `ERR_GPU_UNAVAILABLE`,
  `ERR_STREAM_TIMEOUT`, etc.), a message, and a `recoverable` flag, instead of raw
  stack traces — model loading retries automatically before giving up.
- **Graceful degradation** (`src/production/health_monitor.py`): the system reports
  one of three levels — **FULL** (GPU + all cameras healthy), **DEGRADED** (CPU
  fallback and/or some cameras down, but every healthy camera keeps alerting), or
  **CRITICAL** (inference itself is down; raw frames are still captured for later
  review, but there's no real-time alerting until it's fixed).

Add a camera by adding one line to the `CAMERAS` list in `src/config.py`
(`CameraConfig(camera_id="cam1", source="rtsp://...")`) — no other code changes needed.

### A note on RT-DETR vs. YOLO
`src/config.py`'s `DetectionConfig.model_type` defaults to `"rtdetr"` (RT-DETR-L),
a transformer-based, NMS-free detector, swapped in for the earlier YOLO version per
instructor feedback. Set it back to `"yolo"` with a YOLO `weights` path if you need
to compare the two.

### Batch mode's dwell-time caveat
`batch_run.py` and `main.py` compute dwell time against wall-clock time
(`time.time()`), same as the live pipeline. If you process a *recorded* video file
faster than its own real-time duration (very common in batch analysis), dwell time
will be under-counted relative to what actually happened in the footage, since the
clock is running in wall-clock seconds, not video-timeline seconds. The live pipeline
doesn't have this problem because frames genuinely arrive in real time. If this
matters for your evaluation, drive the dwell clock from the video's own timestamp
(`cap.get(cv2.CAP_PROP_POS_MSEC)`) instead of `time.time()` when processing files.

## Running the tests
```bash
python -m unittest discover -s tests -v
```
Covers the pure-logic pieces: zone containment, tripwire crossing + direction,
the dwell-time state machine (debounce, severity-aware thresholds, cooldown),
zone/tripwire config validation, camera YAML loading, the HTML report
generator, and desktop-app lifecycle (Start/Stop/quit safety while a run is
active). No video or model needed to run these — the desktop lifecycle tests
stub out RT-DETR/YOLO the same way manual smoke testing did during
development. On a headless machine (no display), set
`QT_QPA_PLATFORM=offscreen` first.

## Desktop app (upload/watch a video, see live alerts, save a report)
```bash
python -m desktop_app.main
```
A PySide6 GUI on top of the exact same pipeline underneath \u2014 no separate logic:
- Pick a **recorded video file** (Browse\u2026), a **live local camera** (device index), or a **network stream** (RTSP/HTTP URL).
- **Start** runs the pipeline on a background thread; the live preview panel shows the annotated video (boxes, zone/tripwire overlays, red boxes on alert) as it plays, and the **Live Alerts** panel logs each flag the instant it fires.
- **Stop** ends a live/long run early at any point.
- **Save Report\u2026** writes a single, self-contained HTML file (open it in any browser \u2014 no other files needed, snapshots are embedded). The report leads with a plain-English "How to read this report" section and a glossary (Track ID, restricted zone, dwell violation, tripwire violation, cooldown) before any data, then an executive summary, a snapshot card per alert, the full timeline, and the exact zone/tripwire configuration used for that run \u2014 so it's understandable on its own, with no other context needed.

Requires `PySide6` (in requirements.txt). For processing many recordings unattended without a GUI, use `batch_run.py` instead (Section above) \u2014 the desktop app is for watching one source at a time interactively.

### Packaging it as a standalone .exe (no Python/terminal needed to run it)
By default the desktop app runs as a Python script (`python -m desktop_app.main`),
which needs the venv active. To get a real double-click `SurveillanceApp.exe`
that a teammate can run with no Python install at all:
```bash
pip install pyinstaller
pyinstaller build_app.spec
```
Run this from the project root, inside your activated venv, on the same OS
you want the executable for (build on Windows for a Windows .exe). The first
build takes several minutes and produces a large output folder \u2014 that's
normal, since RT-DETR's dependencies (torch, ultralytics) aren't small.
The finished app is at `dist/SurveillanceApp/SurveillanceApp.exe`; zip the
whole `dist/SurveillanceApp/` folder to share it \u2014 it's self-contained.

**Windows one-command launcher:** `run.bat` (creates `.venv`, installs requirements on first run, then starts the app); `run.bat draw` and `run.bat test` run the zone-drawing tool and the tests.

## The HTML report

`Save Report...` writes one self-contained HTML file (photos included), also usable straight from
`batch_run.py`'s per-video output. It has a favicon and a page description, works down to phone width
(alert cards stack, wide tables scroll in place instead of the page), and every alert photo has real
alt text describing what happened, not a generic placeholder. Photos are shrunk to at most 480px and
re-saved as JPEG before being embedded, so a report with many alerts stays a small file; this needs
Pillow (`requirements.txt`), and falls back to the original image if Pillow isn't installed. Badge and
footer colors are chosen to keep readable contrast against their background (WCAG AA, 4.5:1), which the
test suite checks (`tests/test_report_generator.py`).

## Video sources

In the desktop app choose **Stream URL** and paste an RTSP address, a link to a video file, or a
**YouTube link**. A YouTube link is a web page, not a video file, so the app asks `yt-dlp` (in
`requirements.txt`; if it is missing run `pip install yt-dlp`) for the real media address, then
processes it like a recorded video (its own timeline, capped at 1280 px). A YouTube *live* stream is
treated as a live feed. Only use videos you have the right to use; YouTube's terms restrict
downloading other people's content. If a link fails (private, region-locked, removed, or yt-dlp out
of date: `pip install -U yt-dlp`), download the video yourself and choose it as a file instead.

## Zone rules: entry alert or time limit

Each zone follows ONE of two rules. The default is a **time limit**: an alert is raised once someone has
been inside longer than the zone's severity allows (high 3 s, medium 5 s, low 8 s, after 5 frames of
debounce). Tick **Alert the moment someone steps in** when drawing a zone (or set `alert_on_entry: true`
in `config/zones.yaml`) and the zone becomes an **entry zone**: entering it is already the violation, so
the alert fires on the step in, with no waiting, and no separate time-limit alert follows. Use this for
things like the ground under a window or a door. A person counts as inside only once more than
`ENTRY_CFG.margin_px` past the edge, so a wobbling detection box on the boundary can't fire repeated
entries, and `ENTRY_CFG.cooldown_sec` (5 s) holds back rapid re-entries by the same person.

## How tripwires alert

A tripwire is a line. Each tracked person's foot point is compared with it on every analysed frame; when
the person ends up on the other side (past a small dead band, `TRIPWIRE_CFG.margin_px`, so a wobbling
detection box can't fake crossings) that is one crossing. On a **direction-sensitive** wire only crossings
in the direction of the arrow raise an alert; a wire with `direction_sensitive: false` alerts both ways.
In the editor the arrow points the way that counts, and **Flip tripwire direction** reverses it. After an
alert, further alerts for the same person and line are held back for `TRIPWIRE_CFG.cooldown_sec`
(default 5 s; raise it for a quieter log). Crossings need a stable track ID, so on a very slow machine
(a few frames per second) a person the tracker loses can cause a missed crossing.

## Reducing CPU load / laptop heat

Detection dominates the cost of a run. In `src/config.py` (`DetectionConfig`):
`frame_stride` (analyse every Nth frame; skipped frames are not decoded -- 3 is a good start for
~30 fps sources; desktop app only), `cpu_threads` (cap torch's CPU threads to keep fans down),
and `imgsz` (smaller = faster, but small/far people are detected less reliably). A CUDA build of
torch moves detection to the GPU automatically (`device="auto"`).

## Customizing zones

Zones and tripwires live in `config/zones.yaml` -- no code changes needed. The
shipped file is **empty**, so a fresh run shows a clean feed with no boxes and no
alerts until you define your own. A commented example sits at the top of the file;
copy it, uncomment it, and change the name/coordinates. Polygon points are pixel
coordinates `[x, y]` in the source frame (origin top-left).

**Easier: draw them in the app.** Click **Edit Zones** in the desktop app. It shows a clean
frame from the selected source (or from the feed that is currently running), and you draw on
it: drag out a box for a zone (or pick *polygon* and click its corners, then double-click), two points for a
tripwire, then name it. Right-click undoes a point; drag a white handle to reshape a saved shape (Shift-click to start a
new shape on top of a handle); the side list deletes saved shapes;
*Save* writes `config/zones.yaml`. If a run is in progress the new zones are hot-applied on the
next frame (per-track dwell timers reset; alert cooldowns are kept) -- no restart. Otherwise
they are picked up on the next Start. The zones file is re-read at the start of every run.

**Resolution-independent.** The editor records the frame size it drew on (`reference_size`)
and the desktop pipeline rescales zones to whatever size it actually processes, so zones survive
switching between a 640x480 webcam and a 1080p/4K video. A hand-written file without
`reference_size` is used as-is. (`src.main`, `src.batch_run` and the live multi-camera CLI
read the zones unscaled; use them with zones drawn at the same resolution.)

**Standalone alternative:** `python draw_zones.py` (add `--camera cam1`, `--source <index|file|rtsp>`
or `--image frame.png`) offers the same drawing in an OpenCV window. Keys: `z`/`t` zone or
tripwire mode, `u` undo point, `x` delete last shape, `v` zone severity, `d` tripwire direction,
`r` fresh frame, `s` save, `q` quit.

## Why this design reduces false alarms
A single frame's detection is never enough to raise an alert. An alert requires
**all** of: (1) the object's foot-point inside a restricted polygon, (2) that
observation to be stable for `min_consecutive_frames_in_zone` consecutive frames
(debounce), and (3) dwell time inside the zone to exceed a severity-specific
threshold — OR a tripwire crossing in the disallowed direction. This mirrors how a
human guard reasons ("someone's standing at that door too long") instead of firing
on every raw detection.
"# ai_survillence_base-spy" 
