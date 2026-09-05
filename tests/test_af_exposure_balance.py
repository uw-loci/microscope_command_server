"""Autofocus exposure must not collapse the JAI's per-channel balance.

On the JAI the colour balance comes from per-channel EXPOSURES as much as from the analog
gains -- the uncrossed calibration is R=0.1ms, G=0.2ms, B=0.6ms. Five sites in the AF path
used to disable individual-exposure mode and apply one unified exposure, keeping the gains
and discarding the exposures, which puts red at 2x its calibrated exposure and blue at 1/3.

Measured on the 2026-08-29 run, slide ppm_20x_22: AF frames came back with channel means of
[254.7, 119.6, 57.7] -- red pinned against the ceiling with no contrast left. With a third
of the luminance carrying no signal the texture metric fell just under threshold, and 469 of
1250 autofocus attempts were rejected at a median texture of 0.00836 against a bar of 0.01,
with none below 0.005. Each rejection deferred autofocus and acquired its tile at a stale Z.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_hardware_imports():
    """Mock hardware modules so workflow.py imports without pycromanager.

    Same approach as test_overlapped_io: the module under test is pure arithmetic over a
    calibration dict, but it lives in a file that pulls in the hardware stack at import.
    """
    mocks = {}
    for mod in [
        "pycromanager",
        "microscope_control",
        "microscope_control.hardware",
        "microscope_control.hardware.pycromanager",
        "microscope_control.autofocus",
        "microscope_control.autofocus.core",
        "ppm_library",
        "ppm_library.imaging",
        "ppm_library.imaging.writer",
        "ppm_library.imaging.background",
        "skimage",
        "skimage.filters",
    ]:
        if mod not in sys.modules:
            mocks[mod] = MagicMock()
            sys.modules[mod] = mocks[mod]
    yield
    for mod in mocks:
        del sys.modules[mod]


def _set_af_exposure():
    from microscope_command_server.acquisition.workflow import set_af_exposure

    return set_af_exposure


UNCROSSED_CAL = {
    "angles": {
        "uncrossed": {
            "exposures_ms": {"r": 0.1, "g": 0.2, "b": 0.6},
            "gains": {"unified_gain": 1.0, "analog_red": 0.969, "analog_blue": 4.0},
        }
    }
}


class _Camera:
    def __init__(self, per_channel=True):
        self.applied = None
        self._per_channel = per_channel

    def supports_per_channel_exposure(self):
        return self._per_channel

    def apply_settings(self, exposures, unified_gain, analog_red, analog_blue, individual_exposure):
        self.applied = dict(
            exposures=dict(exposures),
            unified_gain=unified_gain,
            analog_red=analog_red,
            analog_blue=analog_blue,
            individual_exposure=individual_exposure,
        )


class _Hardware:
    def __init__(self, per_channel=True):
        self.camera = _Camera(per_channel)
        self.unified_exposures = []

    def get_camera_name(self):
        return "JAI AP-3200T-USB"

    def set_exposure(self, ms):
        self.unified_exposures.append(ms)


class _Ctx:
    def __init__(self, jai=True, cal=UNCROSSED_CAL):
        self.is_jai_camera = jai
        self.jai_calibration = cal


def test_per_channel_ratio_is_preserved_and_green_lands_on_target():
    hw, ctx = _Hardware(), _Ctx()
    _set_af_exposure()(hw, ctx, 0.2)

    assert hw.camera.applied is not None, "must go through apply_settings"
    exps = hw.camera.applied["exposures"]
    assert hw.camera.applied["individual_exposure"] is True
    # Green lands exactly on the requested exposure -- the intent of the old unified call.
    assert exps["g"] == pytest.approx(0.2)
    # ...and red/blue keep their calibrated ratio to it, instead of being forced to green's.
    assert exps["r"] == pytest.approx(0.1)
    assert exps["b"] == pytest.approx(0.6)
    assert not hw.unified_exposures, "must not fall back to a unified exposure"


def test_scaling_preserves_the_ratio_rather_than_flattening_it():
    """The saturation guard and the brightness doubler both scale this exposure."""
    hw, ctx = _Hardware(), _Ctx()
    _set_af_exposure()(hw, ctx, 0.4)  # 2x the calibrated green
    exps = hw.camera.applied["exposures"]
    assert (exps["r"], exps["g"], exps["b"]) == pytest.approx((0.2, 0.4, 1.2))
    assert exps["b"] / exps["g"] == pytest.approx(0.6 / 0.2)
    assert exps["r"] / exps["g"] == pytest.approx(0.1 / 0.2)


def test_the_old_behaviour_would_have_doubled_red_and_thirded_blue():
    """States the defect as arithmetic, so a regression is recognisable as this bug."""
    cal = UNCROSSED_CAL["angles"]["uncrossed"]["exposures_ms"]
    unified = 0.2
    assert unified / cal["r"] == pytest.approx(2.0)  # red at 2x -> into the ceiling
    assert unified / cal["b"] == pytest.approx(1 / 3)  # blue at a third -> starved

    hw, ctx = _Hardware(), _Ctx()
    _set_af_exposure()(hw, ctx, unified)
    exps = hw.camera.applied["exposures"]
    assert exps["r"] != pytest.approx(unified)
    assert exps["b"] != pytest.approx(unified)


def test_analog_gains_are_still_applied():
    hw, ctx = _Hardware(), _Ctx()
    _set_af_exposure()(hw, ctx, 0.2)
    assert hw.camera.applied["analog_red"] == pytest.approx(0.969)
    assert hw.camera.applied["analog_blue"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "ctx",
    [
        _Ctx(jai=False),
        _Ctx(cal=None),
        _Ctx(cal={"angles": {}}),
        _Ctx(cal={"angles": {"uncrossed": {}}}),
    ],
    ids=["not-a-jai", "no-calibration", "no-uncrossed-angle", "no-exposures"],
)
def test_falls_back_to_a_unified_exposure_when_there_is_no_balance_to_preserve(ctx):
    """A camera without per-channel calibration should still get its exposure set."""
    hw = _Hardware(per_channel=bool(getattr(ctx, "is_jai_camera", False)))
    _set_af_exposure()(hw, ctx, 0.2)
    assert hw.unified_exposures == [0.2]
    assert hw.camera.applied is None
