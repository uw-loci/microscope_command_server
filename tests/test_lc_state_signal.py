"""LCSIGNAL: every LC-PolScope state measured at one exposure, for the Camera tab.

Uses the polarimetric fake microscope from test_lc_calibration: the camera sees
the forward model at whatever retardances the crystals are set to, light scales
with exposure above a 102-count black level, and the 12-bit camera clips at 4095.
"""

import json
import struct
from types import SimpleNamespace

import pytest

pytest.importorskip("polscope_library")

from tests.test_lc_calibration import FakeCore, FakeHardware  # noqa: E402

LC_A = "Retardance LC-A [in waves]"
LC_B = "Retardance LC-B [in waves]"
#: Extinction at the simulator's identity point, swing 0.03 waves.
PALETTE = {
    "State0": (0.25, 0.50),
    "State1": (0.28, 0.50),
    "State2": (0.25, 0.53),
    "State3": (0.22, 0.50),
    "State4": (0.25, 0.47),
}


def _settings(state_exposures=None, overrides=None, black_level=102):
    channels = []
    for cid, (a, b) in PALETTE.items():
        entry = {
            "id": cid,
            "exposure_ms": (state_exposures or {}).get(cid, 50),
            "device_properties": [
                {"device": "MeadowlarkLC", "property": LC_A, "value": a},
                {"device": "MeadowlarkLC", "property": LC_B, "value": b},
            ],
        }
        channels.append(entry)
    profile = {"modality": "lcpolscope"}
    if overrides:
        profile["channel_overrides"] = overrides
    return {
        "modalities": {
            "lcpolscope": {"channels": channels, "reconstruction": {"black_level": black_level}}
        },
        "acquisition_profiles": {"lcpolscope_20x": profile},
    }


def _measure(hardware, **kw):
    from microscope_command_server.calibration.lc_state_signal import measure_lc_states

    return measure_lc_states(
        hardware, _settings(**kw.pop("settings_kw", {})), "lcpolscope_20x", **kw
    )


def test_every_state_is_measured_at_the_one_exposure_asked_for():
    """A state carrying its own exposure in the YAML must not be measured at it:
    the point is all five at the same exposure."""
    hw = FakeHardware()
    report = _measure(hw, exposure_ms=40.0, settings_kw={"state_exposures": {"State2": 10}})
    assert report["exposure_ms"] == 40.0
    assert [s["id"] for s in report["states"]] == list(PALETTE)
    assert hw.exposure == 40.0
    assert set(hw.exposure_log) == {40.0}


def test_extinction_is_flagged_and_the_brightest_swing_state_reported():
    report = _measure(FakeHardware(), exposure_ms=50.0)
    by_id = {s["id"]: s for s in report["states"]}
    assert by_id["State0"]["extinction"] is True
    assert by_id["State0"]["mean"] < min(
        by_id[c]["mean"] for c in ("State1", "State2", "State3", "State4")
    )
    assert report["brightest_state"] != "State0"
    assert report["full_scale"] == 4095.0
    assert report["brightest_fraction"] == pytest.approx(
        by_id[report["brightest_state"]]["p999"] / 4095.0, abs=1e-4
    )
    assert report["clipped"] is False


def test_the_suggested_exposure_lands_the_brightest_state_on_target():
    hw = FakeHardware()
    first = _measure(hw, exposure_ms=50.0, target_fraction=0.6)
    suggested = first["suggested_exposure_ms"]
    assert suggested > 50.0
    second = _measure(hw, exposure_ms=suggested, target_fraction=0.6)
    assert second["brightest_fraction"] == pytest.approx(0.6, abs=0.01)


def test_a_clipped_field_gets_no_suggestion_from_a_clipped_value():
    """A clipped number says only 'at least this bright'; scaling from it would
    under-correct, so the report says clipped and offers nothing."""
    report = _measure(FakeHardware(), exposure_ms=5000.0)
    assert report["clipped"] is True
    assert report["suggested_exposure_ms"] is None


def test_suggestion_arithmetic_respects_the_black_level():
    from microscope_command_server.calibration.lc_state_signal import suggest_exposure_ms

    # 100 counts of light above a 100 black at 10 ms; 0.5 of 4095 is 2047.5,
    # i.e. 1947.5 of light: 19.475x the exposure.
    assert suggest_exposure_ms(
        10.0, 200.0, 4095.0, black_level=100.0, target_fraction=0.5
    ) == pytest.approx(194.8, abs=0.05)
    assert suggest_exposure_ms(10.0, 90.0, 4095.0, black_level=100.0) is None  # no light
    assert suggest_exposure_ms(10.0, 4095.0, 4095.0) is None  # clipped


def test_unknown_profile_and_bad_target_are_reported_not_raised():
    from microscope_command_server.calibration.lc_state_signal import measure_lc_states

    assert (
        "unknown acquisition profile"
        in measure_lc_states(FakeHardware(), _settings(), "nope")["error"]
    )
    assert (
        "target fraction"
        in measure_lc_states(FakeHardware(), _settings(), "lcpolscope_20x", target_fraction=1.5)[
            "error"
        ]
    )


def test_profile_overrides_are_merged_like_applych():
    from microscope_command_server.server.handlers.illumination import (
        profile_channel_entry,
        resolve_profile_modality,
    )

    extra = {"device": "LED", "property": "Intensity", "value": "80"}
    settings = _settings(overrides={"State1": {"device_properties": [extra], "settle_ms": 20}})
    _, modality_cfg, profile_cfg = resolve_profile_modality(settings, "lcpolscope_20x")
    entry = profile_channel_entry(modality_cfg, profile_cfg, "State1")
    assert entry["device_properties"][-1] == extra
    assert len(entry["device_properties"]) == 3
    assert entry["settle_ms"] == 20
    # The library itself is not modified by the merge.
    assert len(modality_cfg["channels"][1]["device_properties"]) == 2
    assert profile_channel_entry(modality_cfg, profile_cfg, "State9") is None
    assert (
        resolve_profile_modality(settings, "lcpolscope_20x_3") is not None
    ), "trailing counter tolerated"


def test_the_socket_handler_round_trips_the_report():
    from microscope_command_server.server.handlers.calibration import handle_lcsignal

    payload = b"lcpolscope_20x".ljust(32, b"\x00") + struct.pack("!dd", 45.0, 0.0)
    sent = []
    conn = SimpleNamespace(
        recv=lambda n, buf=[payload]: buf.pop() if buf else b"", sendall=sent.append
    )
    hw = FakeHardware(core=FakeCore())
    hw.settings = _settings()
    handle_lcsignal(conn, SimpleNamespace(addr="t"), hw, None)
    (length,) = struct.unpack("!I", sent[0])
    report = json.loads(sent[1].decode("utf-8"))
    assert length == len(sent[1])
    assert report["exposure_ms"] == 45.0
    assert report["target_fraction"] == 0.75, "0 means the default"
    assert len(report["states"]) == 5


def test_a_short_payload_is_an_error_in_the_report():
    from microscope_command_server.server.handlers.calibration import handle_lcsignal

    sent = []
    conn = SimpleNamespace(
        recv=lambda n, buf=[b"short"]: buf.pop() if buf else b"", sendall=sent.append
    )
    handle_lcsignal(conn, SimpleNamespace(addr="t"), FakeHardware(), {})
    assert "short LCSIGNAL payload" in json.loads(sent[1].decode("utf-8"))["error"]
