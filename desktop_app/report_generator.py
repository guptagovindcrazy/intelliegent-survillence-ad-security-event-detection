"""
desktop_app/report_generator.py
---------------------------------
Builds a single, self-contained HTML report (no separate image files --
snapshots are base64-embedded) from a finished PipelineWorker run.

Design goal: someone with NO context on this project should be able to open
the report and understand exactly what it's telling them, which is why the
report leads with a plain-English "How to read this report" section before
any data, and every alert is explained in the same words used to explain the
system generally, not raw internal field names.
"""

import base64
import html
import io
from collections import Counter
from datetime import datetime

from desktop_app.models import alert_type_info
from src.config import RESTRICTED_ZONES, TEMPORAL_CFG, TRIPWIRES

# A generic security-camera emoji, base64'd as the report's favicon so the file
# stays a single, self-contained HTML page (no second file to keep alongside it).
_FAVICON_SVG = base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    b'<rect width="32" height="32" rx="6" fill="#1E2761"/>'
    b'<circle cx="16" cy="16" r="9" fill="none" stroke="#E24C4B" stroke-width="2.5"/>'
    b'<circle cx="16" cy="16" r="3.5" fill="#E24C4B"/></svg>'
).decode("ascii")

# Longest edge (px) a snapshot is shrunk to before embedding, and the JPEG
# quality used. Alert photos only need to be big enough to recognise what
# happened -- shrinking them keeps a report with many alerts from becoming an
# unwieldy multi-hundred-MB file. Falls back to the original PNG bytes if
# Pillow isn't installed (it's already a dependency of ultralytics, so this
# should be rare) or the image data is unreadable.
_SNAPSHOT_MAX_DIM = 480
_SNAPSHOT_JPEG_QUALITY = 72


def _compress_snapshot(png_bytes: bytes):
    """Returns (mime_type, encoded_bytes) for embedding -- smaller where possible."""
    try:
        from PIL import Image
    except ImportError:
        return "image/png", png_bytes
    try:
        img = Image.open(io.BytesIO(png_bytes))
        img.load()
    except Exception:
        return "image/png", png_bytes
    img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > _SNAPSHOT_MAX_DIM:
        scale = _SNAPSHOT_MAX_DIM / longest
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=_SNAPSHOT_JPEG_QUALITY, optimize=True)
    return "image/jpeg", out.getvalue()

CSS = """
:root {
  --navy: #1E2761; --navy-dark: #141B45; --ice: #CADCFC; --accent: #E24C4B;
  /* Darker than --accent (used for the red border/heading, itself decorative)
     specifically so white badge text on it clears 4.5:1 contrast. */
  --accent-badge: #A93226;
  --green: #1C7C54; --grey: #3A3F55; --light-bg: #F4F6FB; --border: #E2E6F0;
  /* Darker than a plain mid-grey so the footer text clears 4.5:1 on --light-bg. */
  --footer-grey: #5B6172;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, Calibri, Arial, sans-serif; margin: 0; background: var(--light-bg); color: var(--grey); }
.wrap { max-width: 980px; margin: 0 auto; padding: 32px 24px 80px; }
header.hero { background: var(--navy-dark); color: white; padding: 40px 32px; border-radius: 12px; margin-bottom: 28px; }
header.hero h1 { margin: 0 0 6px; font-size: 26px; }
header.hero .meta { color: var(--ice); font-size: 14px; }
.stat-row { display: flex; gap: 16px; margin: 20px 0 32px; flex-wrap: wrap; }
.stat-card { background: white; border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; flex: 1; min-width: 150px; }
.stat-card .num { font-size: 28px; font-weight: 700; color: var(--navy-dark); }
.stat-card .label { font-size: 12.5px; color: var(--grey); margin-top: 2px; }
section { background: white; border: 1px solid var(--border); border-radius: 10px; padding: 24px 28px; margin-bottom: 24px; }
section h2 { color: var(--navy-dark); font-size: 19px; margin-top: 0; }
section h3 { color: var(--navy); font-size: 15px; }
.glossary dt { font-weight: 700; color: var(--navy-dark); margin-top: 10px; }
.glossary dd { margin: 2px 0 0; }
.table-scroll { width: 100%; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
th { background: var(--navy); color: white; text-align: left; padding: 8px 10px; }
td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
tr:nth-child(even) td { background: #F8F9FD; }
/* Badge colors are chosen to keep white text at or above a 4.5:1 contrast
   ratio against it (WCAG AA for normal-size text). */
.badge { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 11.5px; font-weight: 700; color: white; }
.badge.dwell { background: var(--accent-badge); }
.badge.tripwire { background: var(--navy); }
.badge.entry { background: #7C3AED; }
.badge.other { background: var(--grey); }
.alert-card { display: flex; gap: 16px; border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 14px; }
.alert-card img { width: 220px; max-width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border); flex-shrink: 0; }
.alert-card .details { font-size: 13.5px; }
.no-alerts { text-align: center; padding: 40px 0; color: var(--green); font-size: 16px; font-weight: 600; }
footer { color: var(--footer-grey); font-size: 12px; text-align: center; margin-top: 30px; }

/* Mobile: cards stack instead of sitting side by side, hero padding shrinks,
   and wide tables scroll within their own box instead of the whole page. */
@media (max-width: 640px) {
  .wrap { padding: 16px 12px 56px; }
  header.hero { padding: 24px 20px; border-radius: 8px; }
  header.hero h1 { font-size: 21px; }
  section { padding: 16px 16px; }
  .stat-row { gap: 10px; margin: 14px 0 22px; }
  .stat-card { min-width: 120px; padding: 12px 14px; }
  .stat-card .num { font-size: 22px; }
  .alert-card { flex-direction: column; }
  .alert-card img { width: 100%; }
}
"""

GLOSSARY = [
    ("Track ID", "A number the system assigns to one specific person the moment it first sees them, and keeps using for that same person as they move through the video. Two different numbers mean the system believes they are two different people."),
    ("Restricted zone", "An area on screen (drawn as a colored outline) that people are not supposed to linger in. Each zone has a severity (high / medium / low) that controls how quickly a lingering violation is flagged."),
    ("Tripwire", "An invisible line on screen. Crossing it in the wrong direction is flagged immediately \u2014 there's no waiting period, since crossing it once is already the complete violation."),
    ("Dwell violation", "Someone stayed inside a restricted zone longer than that zone's time limit. A brief walk-through does not count \u2014 only lingering past the threshold does."),
    ("Zone entry violation", "Someone stepped into a zone marked 'alert on entry' (for example the ground in front of a window). There is no time limit for these zones: entering is already the violation."),
    ("Tripwire violation", "Someone crossed a tripwire line in the disallowed direction (e.g. entering through an exit-only gate)."),
    ("Cooldown", "After an alert fires for someone, the system waits a short time before it will alert on that same person again for the same ongoing situation \u2014 this is why you won't see the same violation repeated every second."),
]


def _fmt_time(seconds: float) -> str:
    m, s = divmod(max(seconds, 0), 60)
    return f"{int(m)}m {s:04.1f}s"


def _executive_summary(summary) -> str:
    if not summary.alerts:
        return (
            "<div class='no-alerts'>\u2713 No security concerns were found.<br>"
            "<span style='font-weight:400;font-size:13px;color:var(--grey);'>"
            "Nothing stayed in a restricted area past its time limit, nobody entered a no-entry zone, and no tripwire was crossed the wrong way.</span></div>"
        )
    by_type = Counter(a.alert_type for a in summary.alerts)
    by_zone = Counter(a.zone_or_wire for a in summary.alerts)
    busiest_zone = by_zone.most_common(1)[0]
    lines = [f"<p><strong>{len(summary.alerts)} alert(s)</strong> were raised during this run.</p>", "<ul>"]
    for alert_type, n in by_type.most_common():
        info = alert_type_info(alert_type)
        lines.append(f"<li>{n} were <strong>{html.escape(info.plural)}</strong> ({html.escape(info.blurb)}).</li>")
    lines.append(f"<li>The area with the most activity was <strong>{html.escape(busiest_zone[0])}</strong> ({busiest_zone[1]} alert(s)).</li>")
    lines.append("</ul>")
    return "\n".join(lines)


def build_html_report(summary) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    source_label = "Live camera / stream" if summary.is_live else "Recorded video file"

    stat_cards = f"""
    <div class="stat-card"><div class="num">{summary.frames_processed}</div><div class="label">Frames processed</div></div>
    <div class="stat-card"><div class="num">{_fmt_time(summary.duration_sec)}</div><div class="label">Processing time</div></div>
    <div class="stat-card"><div class="num">{len(summary.alerts)}</div><div class="label">Total alerts</div></div>
    <div class="stat-card"><div class="num">{"Yes" if summary.stopped_early else "No"}</div><div class="label">Stopped early by user</div></div>
    """

    glossary_html = "".join(f"<dt>{html.escape(term)}</dt><dd>{html.escape(desc)}</dd>" for term, desc in GLOSSARY)

    timeline_rows = "".join(
        f"<tr><td>{_fmt_time(a.video_time_sec)}</td><td>{a.frame_index}</td><td>{a.track_id}</td>"
        f"<td><span class='badge {alert_type_info(a.alert_type).css}'>"
        f"{html.escape(alert_type_info(a.alert_type).short)}</span></td>"
        f"<td>{html.escape(a.zone_or_wire)}</td><td>{html.escape(a.message)}</td></tr>"
        for a in summary.alerts
    )
    timeline_section = (
        f"<div class='table-scroll'><table><tr><th>Time</th><th>Frame</th><th>Track</th><th>Type</th>"
        f"<th>Area</th><th>Details</th></tr>{timeline_rows}</table></div>"
        if summary.alerts else "<p>No alerts to list.</p>"
    )

    cards_html = ""
    for a in summary.alerts:
        img_tag = ""
        if a.snapshot_png:
            mime, encoded = _compress_snapshot(a.snapshot_png)
            b64 = base64.b64encode(encoded).decode("ascii")
            alt = html.escape(
                f"Camera snapshot at {_fmt_time(a.video_time_sec)}: track #{a.track_id} "
                f"triggering a {alert_type_info(a.alert_type).long.lower()} at '{a.zone_or_wire}'")
            img_tag = f"<img src='data:{mime};base64,{b64}' alt='{alt}' loading='lazy'/>"
        info = alert_type_info(a.alert_type)
        badge_cls, badge_txt = info.css, html.escape(info.long)
        cards_html += f"""
        <div class="alert-card">
          {img_tag}
          <div class="details">
            <span class="badge {badge_cls}">{badge_txt}</span>
            <p><strong>When:</strong> {_fmt_time(a.video_time_sec)} (frame {a.frame_index})<br>
               <strong>Who:</strong> Track #{a.track_id}<br>
               <strong>Where:</strong> {html.escape(a.zone_or_wire)}<br>
               <strong>Details:</strong> {html.escape(a.message)}</p>
          </div>
        </div>"""

    # The zones this run actually used; older summaries didn't record them.
    run_zones = summary.zones if summary.zones is not None else RESTRICTED_ZONES
    run_tripwires = summary.tripwires if summary.tripwires is not None else TRIPWIRES
    zone_rows = "".join(
        f"<tr><td>{html.escape(z.name)}</td>"
        + ("<td>-</td><td>alert the moment someone enters</td></tr>" if z.alert_on_entry else
           f"<td>{z.severity}</td><td>{TEMPORAL_CFG.dwell_threshold_sec.get(z.severity, '?')}s</td></tr>")
        for z in run_zones
    )
    wire_rows = "".join(
        f"<tr><td>{html.escape(t.name)}</td>"
        f"<td>{'direction-sensitive' if t.direction_sensitive else 'either direction'}</td>"
        f"<td>instant</td></tr>"
        for t in run_tripwires
    )

    error_banner = ""
    if summary.error:
        error_banner = (
            f"<section style='border-color:var(--accent);'><h2 style='color:var(--accent);'>Run did not complete normally</h2>"
            f"<p>{html.escape(summary.error)}</p></section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Automated surveillance analysis report for {html.escape(str(summary.source))}: {len(summary.alerts)} alert(s) across {summary.frames_processed} processed frames.">
<link rel="icon" href="data:image/svg+xml;base64,{_FAVICON_SVG}">
<title>Surveillance Report \u2014 {html.escape(str(summary.source))}</title>
<style>{CSS}</style></head>
<body><div class="wrap">

<header class="hero">
  <h1>Surveillance Analysis Report</h1>
  <div class="meta">Source: {html.escape(str(summary.source))} ({source_label}) &nbsp;\u00B7&nbsp; Generated: {now_str}</div>
</header>

{error_banner}

<section>
  <h2>How to read this report</h2>
  <p>This report explains what the surveillance system observed while watching this video or feed.
     In a time-limit zone it only raises an alert when someone is both (a) somewhere they shouldn't be, AND
     (b) has been there long enough for it to be a real concern \u2014 a brief walk-through never triggers one.
     Tripwires and "alert on entry" zones are stricter: they alert immediately. This keeps
     the report focused on things actually worth a human's attention.</p>
  <dl class="glossary">{glossary_html}</dl>
</section>

<div class="stat-row">{stat_cards}</div>

<section>
  <h2>Executive summary</h2>
  {_executive_summary(summary)}
</section>

<section>
  <h2>Alert details</h2>
  {cards_html if summary.alerts else "<p>No alerts to show.</p>"}
</section>

<section>
  <h2>Full timeline</h2>
  {timeline_section}
</section>

<section>
  <h3>Configuration used for this run</h3>
  <p style="font-size:13px;">These are the restricted areas and lines the system was watching for, and how long someone
     could stay in each before it counted as a violation.</p>
  <div class="table-scroll"><table><tr><th>Restricted zone</th><th>Severity</th><th>Rule / time limit before flagged</th></tr>{zone_rows}</table></div>
  <br>
  <div class="table-scroll"><table><tr><th>Tripwire</th><th>Rule</th><th>Time limit</th></tr>{wire_rows}</table></div>
</section>

<footer>Generated by the Spatio-Temporal Surveillance Pipeline desktop app.</footer>
</div></body></html>"""


def save_report(summary, path: str) -> None:
    html_text = build_html_report(summary)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
