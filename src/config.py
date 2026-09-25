"""
config.py
---------
Central configuration for the spatio-temporal surveillance pipeline.
Keeping all tunables in one place makes the system auditable during a viva:
every threshold used to justify an alert traces back to a single, documented value.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import os
import yaml


def _resolve_device(requested: str) -> str:
    """Resolves "auto" to "cuda" only if a CUDA-capable GPU is actually
    importable/available right now; otherwise (including if torch isn't
    installed at all, e.g. some lightweight config-only test contexts)
    falls back to "cpu". An explicit "cpu"/"cuda"/"mps" request is
    returned as-is -- "auto" is the only value this touches, so anyone who
    deliberately wants to force a specific device still can. Imports torch
    lazily (only when actually resolving "auto") so config.py itself stays
    importable without torch installed, matching the rest of this module's
    "no heavy deps just to read a config value" design."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# 1. MODEL / DETECTION CONFIG
# ---------------------------------------------------------------------------
@dataclass
class DetectionConfig:
    model_type: str = "rtdetr"            # "rtdetr" or "yolo" -- RT-DETR chosen per teacher feedback
    weights: str = "rtdetr-l.pt"          # RT-DETR-L; use rtdetr-x for higher accuracy at more cost
    conf_threshold: float = 0.35         # minimum detection confidence
    iou_threshold: float = 0.45          # NMS IoU threshold (unused by RT-DETR's own head, kept for YOLO fallback)
    classes: List[int] = field(default_factory=lambda: [0])  # COCO class 0 = 'person'
    device: str = "auto"                 # "auto" (recommended), "cuda", "cpu", or "mps"
    # "auto" resolves to "cuda" only if a CUDA-capable GPU is actually
    # detected at construction time, else falls back to "cpu" -- see
    # _resolve_device() above. Hardcoding "cuda" as the literal default
    # used to mean this pipeline would try to force GPU inference on ANY
    # machine, CPU-only laptops included, with no fallback: the first
    # detection call would raise inside the per-frame processing loop
    # instead of at model-load time, which (before the per-frame try/except
    # added alongside this) could silently hang the desktop app's worker
    # thread rather than surfacing a clear error.
    imgsz: int = 640
    # `imgsz` already makes Ultralytics letterbox-resize each frame internally
    # before its own forward pass, regardless of the frame's native resolution
    # -- confirmed by reading layers/detection_tracking.py's model.track() call,
    # which passes `imgsz=self.det_cfg.imgsz`. So a 4K frame is NOT run through
    # the network at 4K.
    #
    # `max_processing_dim` is a separate, additive knob: it caps the longer
    # side of the frame WE hand to the whole per-frame pipeline (detection call,
    # box drawing, zone evaluation, and the desktop app's live-preview/alert-
    # snapshot encoding) for file-based (recorded video) runs. A native 4K frame
    # means 4x the pixels to copy/convert/encode at every one of those steps even
    # though the network itself never sees more than `imgsz` pixels on a side --
    # this cap removes that overhead. Frames already at or below this size are
    # left completely untouched (no resize call at all), so normal-resolution
    # videos behave exactly as before this option was added.
    max_processing_dim: int = 1280

    # --- load / heat knobs (defaults keep the original behaviour) -----------
    # Run detection on every Nth frame and merely skip past the rest without
    # decoding them into images. Detection dominates CPU/GPU cost, so load
    # (and laptop heat) falls roughly N-fold: 3 is a good starting point for
    # ~30 fps sources. The tracker and dwell debounce then count ANALYSED
    # frames, so their frame-based settings stretch N-fold in real time.
    frame_stride: int = 1
    # Cap on CPU threads used by torch (0 = torch's default, i.e. all cores).
    # Lower it to keep fans quiet on a laptop; it trades speed for heat.
    cpu_threads: int = 0

    def __post_init__(self):
        self.device = _resolve_device(self.device)
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}")
        if self.cpu_threads < 0:
            raise ValueError(f"cpu_threads must be >= 0, got {self.cpu_threads}")


# ---------------------------------------------------------------------------
# 2. TRACKER CONFIG (ByteTrack, invoked via Ultralytics' tracker YAML)
# ---------------------------------------------------------------------------
@dataclass
class TrackerConfig:
    tracker_type: str = "bytetrack.yaml"   # Ultralytics ships bytetrack.yaml & botsort.yaml
    track_high_thresh: float = 0.5         # confidence to start a track
    track_low_thresh: float = 0.1          # confidence to keep a track alive (2nd assoc. stage)
    new_track_thresh: float = 0.6          # confidence required to spawn a brand-new track
    track_buffer: int = 30                 # frames to keep a "lost" track before deletion
    match_thresh: float = 0.8              # IoU threshold for Hungarian-algorithm matching


# ---------------------------------------------------------------------------
# 3. SPATIAL ZONE CONFIG
# ---------------------------------------------------------------------------
@dataclass
class Zone:
    name: str
    polygon: List[Tuple[int, int]]   # ordered (x, y) pixel vertices, restricted region
    severity: str = "high"           # "low" | "medium" | "high" -> affects dwell threshold
    # False (default): the zone uses a TIME LIMIT -- alert after someone stays
    # longer than the severity's dwell threshold. True: alert the moment
    # someone steps in, with no waiting (use for "nobody may enter here"
    # areas such as a window or a door); severity is then irrelevant.
    alert_on_entry: bool = False


@dataclass
class Tripwire:
    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    direction_sensitive: bool = True  # if True, only one crossing direction counts as a violation


# Zones and tripwires live in config/zones.yaml (same idea as cameras.yaml:
# edit a file, no code change). With no file -- or an empty one -- the lists
# below are EMPTY: a fresh install shows a clean video with no boxes and no
# alerts until you define real zones. See config/zones.yaml for a commented
# example to copy.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONES_YAML_PATH = os.path.join(_PROJECT_ROOT, "config", "zones.yaml")


@dataclass
class ZoneConfig:
    """Everything config/zones.yaml describes. `reference_size` is the
    (width, height) of the frame the coordinates were drawn on; when set, the
    pipeline rescales the shapes to whatever frame size it actually runs at,
    so zones survive a change of camera/video resolution. When None the
    coordinates are taken as-is (hand-written / legacy files)."""
    zones: List[Zone] = field(default_factory=list)
    tripwires: List[Tripwire] = field(default_factory=list)
    reference_size: Optional[Tuple[int, int]] = None


def _as_point(value, what: str) -> Tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
        raise ValueError(f"{what} must be [x, y] numbers, got {value!r}")
    return (int(value[0]), int(value[1]))


def load_zone_config(path: str = ZONES_YAML_PATH) -> ZoneConfig:
    """Loads zones, tripwires and reference size from YAML. Missing/empty file
    -> an empty ZoneConfig. Like _load_cameras_from_yaml, this can run at
    import time, so a broken file or a malformed entry prints a warning and is
    skipped instead of crashing the app. Semantic problems (duplicate names,
    <3 polygon points, ...) are still caught by src/validation.py at pipeline
    start."""
    if not os.path.exists(path):
        return ZoneConfig()
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        print(f"[config] WARNING: {path} is not valid YAML ({exc}); using no zones/tripwires.")
        return ZoneConfig()
    if not isinstance(raw, dict):
        print(f"[config] WARNING: {path} must be a mapping with 'zones'/'tripwires' keys; "
              f"using no zones/tripwires.")
        return ZoneConfig()

    zones: List[Zone] = []
    for i, entry in enumerate(raw.get("zones") or []):
        try:
            if not isinstance(entry, dict) or "name" not in entry or "polygon" not in entry:
                raise ValueError("needs 'name' and 'polygon'")
            poly = entry["polygon"]
            if not isinstance(poly, (list, tuple)):
                raise ValueError("'polygon' must be a list of [x, y] points")
            on_entry = entry.get("alert_on_entry", False)
            if not isinstance(on_entry, bool):
                raise ValueError(f"'alert_on_entry' must be true or false, got {on_entry!r}")
            zones.append(Zone(
                name=str(entry["name"]),
                polygon=[_as_point(pt, "polygon point") for pt in poly],
                severity=str(entry.get("severity", "high")),
                alert_on_entry=on_entry,
            ))
        except ValueError as exc:
            print(f"[config] WARNING: {path} zone #{i + 1} skipped: {exc}")

    wires: List[Tripwire] = []
    for i, entry in enumerate(raw.get("tripwires") or []):
        try:
            if not isinstance(entry, dict) or not all(k in entry for k in ("name", "p1", "p2")):
                raise ValueError("needs 'name', 'p1' and 'p2'")
            wires.append(Tripwire(
                name=str(entry["name"]),
                p1=_as_point(entry["p1"], "p1"),
                p2=_as_point(entry["p2"], "p2"),
                direction_sensitive=bool(entry.get("direction_sensitive", True)),
            ))
        except ValueError as exc:
            print(f"[config] WARNING: {path} tripwire #{i + 1} skipped: {exc}")

    reference_size = None
    if raw.get("reference_size") is not None:
        try:
            w, h = _as_point(raw["reference_size"], "reference_size")
            if w <= 0 or h <= 0:
                raise ValueError("reference_size must be positive")
            reference_size = (w, h)
        except ValueError as exc:
            print(f"[config] WARNING: {path} reference_size ignored: {exc}")

    return ZoneConfig(zones, wires, reference_size)


def _load_zones_from_yaml(path: str = ZONES_YAML_PATH) -> Tuple[List[Zone], List[Tripwire]]:
    """(zones, tripwires) view of load_zone_config -- kept for callers/tests
    that don't care about the reference size."""
    cfg = load_zone_config(path)
    return cfg.zones, cfg.tripwires


ZONE_CONFIG = load_zone_config()
RESTRICTED_ZONES, TRIPWIRES = ZONE_CONFIG.zones, ZONE_CONFIG.tripwires


# ---------------------------------------------------------------------------
# 4. TEMPORAL / ALERT-FUSION CONFIG
# ---------------------------------------------------------------------------
@dataclass
class TemporalConfig:
    # dwell thresholds (seconds) per zone severity -- this is the core false-alarm reducer
    dwell_threshold_sec: dict = field(default_factory=lambda: {
        "low": 8.0,
        "medium": 5.0,
        "high": 3.0,
    })
    min_consecutive_frames_in_zone: int = 5   # debounce: filters single-frame detection flicker
    trajectory_history_len: int = 64          # how many past centroids to retain per track
    alert_cooldown_sec: float = 20.0          # suppress duplicate alerts for same track
    stationary_speed_px_s: float = 15.0       # below this speed (px/s) the object is "loitering"


@dataclass
class TripwireConfig:
    # A person must get more than this many pixels past the line before the
    # side change counts. Detection boxes wobble a few pixels frame to frame,
    # so without this a person standing ON the line would "cross" it over and
    # over. 0 disables the dead band (only points exactly on the line are ignored).
    margin_px: float = 5.0
    # After a tripwire alert for a (person, line) pair, further alerts for that
    # same pair are held back this long. With the dead band above doing the
    # jitter filtering, this can be short: it only stops a rapid back-and-forth
    # from producing a burst of alerts. Raise it (e.g. 20) for a quieter log.
    cooldown_sec: float = 5.0


@dataclass
class ZoneEntryConfig:
    # For "alert on entry" zones. A person counts as inside only once they are
    # more than this many pixels past the edge, and as outside only once more
    # than this far beyond it -- so a detection box wobbling on the boundary
    # can't produce a string of fake entries.
    margin_px: float = 5.0
    # After an entry alert for a (person, zone) pair, further entry alerts for
    # the same pair are held back this long.
    cooldown_sec: float = 5.0


DET_CFG = DetectionConfig()
TRACK_CFG = TrackerConfig()
TEMPORAL_CFG = TemporalConfig()
TRIPWIRE_CFG = TripwireConfig()
ENTRY_CFG = ZoneEntryConfig()


# ---------------------------------------------------------------------------
# 5. PRODUCTION / MULTI-CAMERA CONFIG
# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    camera_id: str
    source: str                     # "0" for default webcam, or an RTSP URL
    zones: List[Zone] = field(default_factory=lambda: RESTRICTED_ZONES)
    tripwires: List[Tripwire] = field(default_factory=lambda: TRIPWIRES)
    # Frame size the zones were drawn on (None = coordinates used as-is); the
    # live pipeline rescales to each camera's real frame size. See zone_runtime.py.
    zone_reference_size: Optional[Tuple[int, int]] = field(default_factory=lambda: ZONE_CONFIG.reference_size)


@dataclass
class ProductionConfig:
    frame_queue_maxsize: int = 8          # backpressure: bounded queue per camera
    num_inference_workers: int = 2        # worker threads pulling from the shared queue
    stream_timeout_sec: float = 10.0      # no frames for this long -> ERR_STREAM_TIMEOUT
    model_load_retries: int = 3           # retry count before ERR_MODEL_500 is raised
    health_check_interval_sec: float = 5.0
    capture_buffersize: int = 1           # cv2.CAP_PROP_BUFFERSIZE -- always read the newest frame


# Config-driven camera list: if config/cameras.yaml exists, cameras are loaded
# from THAT file (add a camera by editing the YAML -- no code change, no redeploy).
# If the file doesn't exist, we fall back to the single hardcoded default below,
# so the pipeline still runs out of the box with no extra setup.
def _load_cameras_from_yaml(path: str = "config/cameras.yaml"):
    """Loads CAMERAS from YAML if the file exists and is well-formed. A missing
    file is a normal, silent fall-through to the hardcoded default (see below).
    A file that EXISTS but is broken (bad YAML syntax, or a camera entry
    missing camera_id/source) is a config mistake worth surfacing loudly --
    but this function runs at import time, so raising here would crash the
    entire application before main() even starts. Instead we print a clear
    warning and fall back to the default camera list, exactly like an invalid
    zone/tripwire config excludes just that camera rather than crashing
    everything (see src/validation.py)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        print(f"[config] WARNING: {path} is not valid YAML ({exc}); "
              f"falling back to the default camera list.")
        return None

    entries = raw.get("cameras", [])
    if not entries:
        return None

    cams = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "camera_id" not in entry or "source" not in entry:
            print(f"[config] WARNING: {path} entry #{i + 1} is missing 'camera_id' or "
                  f"'source' ({entry!r}); skipping that entry.")
            continue
        cams.append(CameraConfig(camera_id=str(entry["camera_id"]), source=str(entry["source"])))

    if not cams:
        print(f"[config] WARNING: no valid camera entries found in {path}; "
              f"falling back to the default camera list.")
        return None
    return cams


CAMERAS: List[CameraConfig] = _load_cameras_from_yaml() or [
    CameraConfig(camera_id="cam0", source="0"),  # default webcam
]

PROD_CFG = ProductionConfig()

