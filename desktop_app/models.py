"""
desktop_app/models.py
-----------------------
Plain data containers shared between pipeline_worker.py and
report_generator.py. Deliberately dependency-free (no ultralytics/torch/PySide6
imports) so the report generator can be used/tested on its own.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class AlertTypeInfo:
    short: str      # table badge text
    long: str       # card badge text
    css: str        # report badge colour class
    icon: str       # marker used in the app's alert list
    plural: str     # "<n> were <plural> (<blurb>)" in the report summary
    blurb: str


# The one place that says how each alert type is worded and coloured, so the
# app, the report and the batch summary can't drift apart. Adding a new alert
# type means adding one entry here.
ALERT_TYPES = {
    "DWELL_VIOLATION": AlertTypeInfo(
        "Dwell", "Dwell violation", "dwell", "\U0001F534",
        "dwell violations", "someone lingered too long in a restricted zone"),
    "ZONE_ENTRY_VIOLATION": AlertTypeInfo(
        "Entry", "Zone entry violation", "entry", "\U0001F7E3",
        "zone entry violations", "someone stepped into a zone where nobody may enter"),
    "TRIPWIRE_VIOLATION": AlertTypeInfo(
        "Tripwire", "Tripwire violation", "tripwire", "\U0001F7E0",
        "tripwire violations", "someone crossed a line the wrong way"),
}


def alert_type_info(alert_type: str) -> AlertTypeInfo:
    """Wording for an alert type; unknown types get a readable generic entry."""
    known = ALERT_TYPES.get(alert_type)
    if known:
        return known
    nice = alert_type.replace("_", " ").title()
    return AlertTypeInfo(nice, nice, "other", "\U0001F7E1", nice.lower() + "s", "see details")


@dataclass
class AlertRecord:
    frame_index: int
    video_time_sec: float
    track_id: int
    alert_type: str          # a key of ALERT_TYPES
    zone_or_wire: str
    message: str
    severity: str = "high"
    snapshot_png: Optional[bytes] = None  # annotated frame at the moment of the alert


@dataclass
class RunSummary:
    source: str
    is_live: bool
    frames_processed: int = 0
    duration_sec: float = 0.0
    alerts: List[AlertRecord] = field(default_factory=list)
    stopped_early: bool = False
    error: Optional[str] = None
    # One of "zone_config" | "model_load" | "source_open" | "processing" |
    # None (generic/unclassified). Lets the UI show tailored guidance per
    # failure type instead of parsing the raw error string, which is
    # fragile and couples the UI to exact wording. See
    # desktop_app/error_panel.py.
    error_category: Optional[str] = None
    # How many individual frames failed to process and were skipped
    # (logged, not fatal) during an otherwise-successful run -- 0 for a
    # clean run. See pipeline_worker.py's per-frame try/except.
    frame_errors: int = 0
    # The zones/tripwires actually in effect for this run (after any
    # resolution rescaling or live edit), so the report describes what really
    # ran instead of whatever the config says later. None = not recorded
    # (older summaries) -> the report falls back to the current config.
    zones: Optional[List[Any]] = None
    tripwires: Optional[List[Any]] = None
