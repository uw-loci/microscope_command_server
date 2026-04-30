"""Unit tests for the wizard stage probe's pure parsers.

The probe handler `handle_probe_stage_af` has three pieces:
    1. nameplate parsing of the speed property's allowed values
    2. live verification (timed 1-um round-trip) -- needs hardware
    3. viability calculation -- pure arithmetic

These tests cover #1 and #3. The live verify path is exercised at
runtime on the Setup Wizard's Probe step against real hardware.

The pure parsers live in `server.probe_parsers` (outside the handler
package) so this test imports them without triggering the handler
package's __init__ chain (which transitively requires
microscope_control / Pycromanager bindings).
"""

import pytest

from microscope_command_server.server.probe_parsers import (
    classify_allowed_values,
    parse_velocity_string,
    pick_recommended_values,
)


# ----- parse_velocity_string -----


@pytest.mark.parametrize("raw, expected_um_s", [
    ("0.50mm/sec", 500.0),
    ("2.50mm/sec", 2500.0),
    ("0.25mm/sec", 250.0),
    ("  1.00 mm/sec  ", 1000.0),    # whitespace tolerant
    ("1.50 MM/SEC", 1500.0),         # case insensitive
    ("500um/sec", 500.0),
    ("11.5 um/sec", 11.5),
    ("100m/sec", 100.0),             # 'm/sec' caught by um regex too
])
def test_parse_velocity_string_recognized(raw, expected_um_s):
    assert parse_velocity_string(raw) == pytest.approx(expected_um_s)


@pytest.mark.parametrize("raw", [
    "1",                  # numeric: not a velocity string
    "100",
    "0.5",
    "fast",
    "",
    "mm/sec",             # missing number
    "500 cm/sec",         # unsupported unit
])
def test_parse_velocity_string_unrecognized(raw):
    assert parse_velocity_string(raw) is None


# ----- classify_allowed_values -----


def test_classify_velocity_enum_ows3():
    """OWS3 ZDrive: a 9-value mm/sec enum."""
    allowed = [
        "0.50mm/sec", "0.75mm/sec", "1.00mm/sec",
        "1.25mm/sec", "1.50mm/sec", "1.70mm/sec",
        "2.00mm/sec", "2.25mm/sec", "2.50mm/sec",
    ]
    kind, parsed = classify_allowed_values(allowed)
    assert kind == "velocity_enum"
    # Sorted ascending by velocity.
    assert parsed[0] == ("0.50mm/sec", 500.0)
    assert parsed[-1] == ("2.50mm/sec", 2500.0)
    # Lengths match.
    assert len(parsed) == len(allowed)


def test_classify_numeric_enum():
    """Some adapters expose a numeric enum like '0', '50', '100'."""
    allowed = ["100", "50", "25", "10", "1"]
    kind, parsed = classify_allowed_values(allowed)
    assert kind == "numeric_enum"
    # Sorted ascending; um/s is unknown so second tuple field is None.
    assert parsed[0][0] == "1"
    assert parsed[-1][0] == "100"
    assert all(p[1] is None for p in parsed)


def test_classify_empty():
    """No allowed-values list -> empty classification (Prior-style)."""
    kind, parsed = classify_allowed_values([])
    assert kind == "empty"
    assert parsed == []


def test_classify_mixed_unknown():
    """Mixed strings (some velocity, some not) -> unknown."""
    allowed = ["0.50mm/sec", "fast", "slow"]
    kind, parsed = classify_allowed_values(allowed)
    assert kind == "unknown"


# ----- pick_recommended_values -----


def test_pick_velocity_enum_picks_slowest_and_fastest():
    allowed = ["0.50mm/sec", "1.00mm/sec", "2.50mm/sec"]
    _, parsed = classify_allowed_values(allowed)
    slow_v, normal_v, slow_ums, reason = pick_recommended_values(
        "velocity_enum", parsed, current_value="2.50mm/sec",
    )
    assert slow_v == "0.50mm/sec"
    assert normal_v == "2.50mm/sec"
    assert slow_ums == pytest.approx(500.0)
    assert "velocity enum" in reason.lower()


def test_pick_numeric_enum_picks_min_and_max_um_s_unknown():
    allowed = ["1", "5", "25", "50", "100"]
    _, parsed = classify_allowed_values(allowed)
    slow_v, normal_v, slow_ums, reason = pick_recommended_values(
        "numeric_enum", parsed, current_value="100",
    )
    assert slow_v == "1"
    assert normal_v == "100"
    # Numeric enum: velocity will come from live verify, so estimate is None.
    assert slow_ums is None
    assert "to be measured" in reason.lower()


def test_pick_empty_uses_prior_fallback():
    """Empty allowed -> Prior 1-100 percent fallback (the legacy default)."""
    slow_v, normal_v, slow_ums, reason = pick_recommended_values(
        "empty", [], current_value="50",
    )
    assert slow_v == "1"
    assert normal_v == "100"
    assert slow_ums == pytest.approx(11.5)
    assert "prior" in reason.lower()


def test_pick_unknown_returns_no_slow_value():
    """Unrecognized format -> caller has to override manually."""
    _, parsed = classify_allowed_values(["foo", "bar"])
    slow_v, normal_v, slow_ums, reason = pick_recommended_values(
        "unknown", parsed, current_value="bar",
    )
    assert slow_v is None
    assert normal_v == "bar"  # carry the current setting
    assert slow_ums is None
    assert "manual override" in reason.lower()


# ----- Realistic integration: OWS3 nameplate + viability -----


def test_ows3_nameplate_translates_to_unviable_streaming():
    """OWS3 slowest is 0.50mm/sec = 500 um/s. Over a 12 um sweep at
    30 fps that's ~0.024 s = 0.7 frames -- not viable for streaming."""
    allowed = [
        "0.50mm/sec", "0.75mm/sec", "1.00mm/sec",
        "1.25mm/sec", "1.50mm/sec", "1.70mm/sec",
        "2.00mm/sec", "2.25mm/sec", "2.50mm/sec",
    ]
    _, parsed = classify_allowed_values(allowed)
    _, _, slow_ums, _ = pick_recommended_values(
        "velocity_enum", parsed, current_value="2.50mm/sec",
    )
    sweep_um = 12.0
    fps = 30.0
    expected_frames = (sweep_um / slow_ums) * fps
    # Viability gate is "need >= 8" (VIABILITY_MIN_FRAMES). OWS3 fails it.
    assert expected_frames < 8.0


def test_ppm_prior_fallback_translates_to_viable_streaming():
    """Prior at MaxSpeed=1 is ~11.5 um/s. Over 12 um at 38 fps that's
    ~1.04 s = ~40 frames -- well above the viability floor."""
    _, _, slow_ums, _ = pick_recommended_values(
        "empty", [], current_value="50",
    )
    sweep_um = 12.0
    fps = 38.0
    expected_frames = (sweep_um / slow_ums) * fps
    assert expected_frames >= 8.0
