"""Pure-function parsers used by the wizard stage probe.

Lives outside the handlers package so unit tests can import these
functions without triggering the `handlers/__init__.py` chain (which
imports the position handler -> microscope_control -> requires real
Pycromanager bindings).

The handler at `server.handlers.probe_stage_af` is the runtime caller;
it adds hardware glue (find the speed property, get_allowed_values,
time the round-trip move, viability arithmetic) but defers the
nameplate parsing and recommendation selection to this module.
"""

import re
from typing import List, Optional, Tuple


# ----- Constants used by recommendation logic ----------------------

# Prior-style 1-100 percent fallback values. Used when the speed
# property has no allowed-values list (continuous numeric scale where
# we can't distinguish percent from um/s without a live verify).
PRIOR_FALLBACK_SLOW = "1"
PRIOR_FALLBACK_NORMAL = "100"
# Empirically measured slow um/s on Prior MaxSpeed=1 (PPM rig).
PRIOR_FALLBACK_SLOW_UM_S = 11.5


# ----- Regex parsers -----------------------------------------------


_MM_PER_SEC_RE = re.compile(r"^\s*([\d.]+)\s*mm\s*/\s*sec\s*$", re.IGNORECASE)
_UM_PER_SEC_RE = re.compile(r"^\s*([\d.]+)\s*u?m\s*/\s*sec\s*$", re.IGNORECASE)


def parse_velocity_string(value: str) -> Optional[float]:
    """Parse '<X>mm/sec' or '<X>um/sec' to um/s.

    Returns None if the value is purely numeric or in an unrecognized
    format -- the caller should fall back to live verification.
    """
    m = _MM_PER_SEC_RE.match(value)
    if m:
        return float(m.group(1)) * 1000.0
    m = _UM_PER_SEC_RE.match(value)
    if m:
        return float(m.group(1))
    return None


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def classify_allowed_values(
    allowed: List[str],
) -> Tuple[str, List[Tuple[str, Optional[float]]]]:
    """Classify a stage's speed-property allowed-values list as one of:

        'velocity_enum' -- '<X>mm/sec' or '<X>um/sec' strings
        'numeric_enum'  -- numeric strings (could be percent or um/s,
                           we cannot tell from the list alone)
        'empty'         -- no allowed values reported (continuous prop)
        'unknown'       -- non-numeric, non-velocity strings

    Returns (kind, parsed). For velocity_enum, parsed is a list of
    (raw_value, derived_um_per_s) tuples sorted ascending by velocity.
    For numeric_enum, parsed is sorted ascending by numeric value
    (second tuple field is None -- velocity comes from live verify).
    """
    if not allowed:
        return "empty", []
    velocity_parsed = []
    all_numeric = True
    for v in allowed:
        ums = parse_velocity_string(v)
        if ums is not None:
            velocity_parsed.append((v, ums))
        else:
            all_numeric = all_numeric and _is_numeric(v)
    if velocity_parsed and len(velocity_parsed) == len(allowed):
        velocity_parsed.sort(key=lambda p: p[1])
        return "velocity_enum", velocity_parsed
    if all_numeric:
        numeric_parsed = sorted(
            [(v, float(v)) for v in allowed], key=lambda p: p[1],
        )
        return "numeric_enum", [(v, None) for v, _ in numeric_parsed]
    return "unknown", [(v, None) for v in allowed]


def pick_recommended_values(
    classification: str,
    parsed: List[Tuple[str, Optional[float]]],
    current_value: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    """Given a classified allowed list, return
    (slow_value, normal_value, slow_um_per_s_estimate, reason).

    `current_value` is the property's current setting; used as the
    fallback `normal_value` for stages whose enums don't make 'max'
    obvious (or whose values we don't recognize).
    """
    if classification == "velocity_enum" and parsed:
        slow_v, slow_ums = parsed[0]
        fast_v, _ = parsed[-1]
        return (slow_v, fast_v, slow_ums,
                f"velocity enum: slow={slow_v}, fast={fast_v}")
    if classification == "numeric_enum" and parsed:
        slow_v, _ = parsed[0]
        fast_v, _ = parsed[-1]
        return (slow_v, fast_v, None,
                f"numeric enum: slow={slow_v}, fast={fast_v} "
                f"(velocity to be measured)")
    if classification == "empty":
        # Continuous numeric property. Assume Prior-style 1-100
        # percent (the original streaming-AF constants). Live verify
        # will adjust um/s if needed.
        return (PRIOR_FALLBACK_SLOW, PRIOR_FALLBACK_NORMAL,
                PRIOR_FALLBACK_SLOW_UM_S,
                "no allowed-values; Prior-style 1-100 percent fallback")
    # Unknown classification -- best we can do is keep the current
    # value as 'normal' and decline to recommend a slow value.
    return (None, current_value, None,
            f"unrecognized allowed values ({len(parsed)} entries); "
            f"manual override required")
