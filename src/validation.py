"""
validation.py
--------------
Validates zone/tripwire configuration before the pipeline starts using it.

This exists so a malformed config (a two-point "polygon", a degenerate
zero-length tripwire, a duplicate or blank name) fails fast with a clear
message at startup, instead of silently producing wrong containment/crossing
results deep inside cv2.pointPolygonTest or the cross-product math. This is
what ERR_ZONE_CONFIG_400 (see src/production/errors.py) actually validates
against in the production pipeline; main.py and batch_run.py call the same
check directly and raise a plain ValueError, since they don't use the
structured-error machinery.
"""

from typing import List

from src.config import Tripwire, Zone


def validate_zones_and_tripwires(zones: List[Zone], tripwires: List[Tripwire]) -> List[str]:
    """Returns a list of human-readable problems. Empty list = valid config."""
    problems = []

    seen_zone_names = set()
    for z in zones:
        if not z.name or not z.name.strip():
            problems.append("A zone has a blank name")
        elif z.name in seen_zone_names:
            problems.append(f"Duplicate zone name: '{z.name}'")
        seen_zone_names.add(z.name)

        if len(z.polygon) < 3:
            problems.append(
                f"Zone '{z.name}' has only {len(z.polygon)} point(s) -- a polygon needs at least 3"
            )
        if z.severity not in ("low", "medium", "high"):
            problems.append(
                f"Zone '{z.name}' has an unknown severity '{z.severity}' (expected low/medium/high)"
            )

    seen_wire_names = set()
    for t in tripwires:
        if not t.name or not t.name.strip():
            problems.append("A tripwire has a blank name")
        elif t.name in seen_wire_names:
            problems.append(f"Duplicate tripwire name: '{t.name}'")
        seen_wire_names.add(t.name)

        if t.p1 == t.p2:
            problems.append(f"Tripwire '{t.name}' has identical endpoints {t.p1} -- a zero-length line can't be crossed")

    return problems


def assert_valid_or_raise(zones: List[Zone], tripwires: List[Tripwire]) -> None:
    """Convenience for entry points that just want to fail fast with ValueError."""
    problems = validate_zones_and_tripwires(zones, tripwires)
    if problems:
        raise ValueError("Invalid zone/tripwire configuration:\n  - " + "\n  - ".join(problems))
