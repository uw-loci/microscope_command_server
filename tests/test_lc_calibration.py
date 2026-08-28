"""The hardware half of the LC calibration, against a fake Micro-Manager.

The library half is tested on a simulated instrument. This covers the part
that talks to Micro-Manager: that it writes the right properties, reads back
rather than assuming, restores state after a dark frame, and returns a usable
dict on every path.
"""

import json

import numpy as np
import pytest

polscope = pytest.importorskip("polscope_library")

from microscope_command_server.calibration.lc_calibration_workflow import (  # noqa: E402
    run_lc_calibration,
)
from microscope_command_server.calibration.lc_instrument import MicroManagerLC  # noqa: E402


class FakeCore:
    """Records property traffic and answers reads consistently."""

    def __init__(self, failing_device=None, limits=None):
        self.properties = {
            ("MeadowlarkLC", "Retardance LC-A [in waves]"): "0.25",
            ("MeadowlarkLC", "Retardance LC-B [in waves]"): "0.50",
            ("LED", "Intensity"): "100",
        }
        self.writes = []
        self.waits = []
        self.failing_device = failing_device
        #: (device, property) -> (lower, upper). Empty by default, which is
        #: how an adapter that declares no limits behaves.
        self.limits = dict(limits or {})

    def has_property_limits(self, device, prop):
        return (device, prop) in self.limits

    def get_property_lower_limit(self, device, prop):
        return self.limits[(device, prop)][0]

    def get_property_upper_limit(self, device, prop):
        return self.limits[(device, prop)][1]

    def set_property(self, device, prop, value):
        if device == self.failing_device:
            raise RuntimeError("device on fire")
        self.properties[(device, prop)] = value
        self.writes.append((device, prop, value))

    def get_property(self, device, prop):
        return self.properties[(device, prop)]

    def wait_for_device(self, device):
        self.waits.append(device)


class FakeHardware:
    """A microscope whose camera sees the polarimetric forward model."""

    def __init__(self, core=None):
        self.core = core or FakeCore()
        self.exposure = 50.0

    def snap_image(self):
        from polscope_library.calibration.simulator import SimulatedPolScope

        lca = float(self.core.get_property("MeadowlarkLC", "Retardance LC-A [in waves]"))
        lcb = float(self.core.get_property("MeadowlarkLC", "Retardance LC-B [in waves]"))
        lamp_on = self.core.get_property("LED", "Intensity") != "0"
        scope = SimulatedPolScope(read_noise=0.0)
        value = scope.intensity_at(lca, lcb) if lamp_on else scope.black_level
        return np.full((4, 4), value, dtype=np.float64), {}

    def set_exposure(self, ms):
        self.exposure = float(ms)

    def get_exposure(self):
        return self.exposure


class TestInstrument:
    def test_writes_retardance_in_waves(self):
        hw = FakeHardware()
        MicroManagerLC(hw, settle_ms=0).set_retardance("LCA", 0.31)
        assert ("MeadowlarkLC", "Retardance LC-A [in waves]", "0.31") in hw.core.writes
        assert "MeadowlarkLC" in hw.core.waits

    def test_clamps_rather_than_raising(self):
        """A search near a rail must lose one measurement, not the whole run."""
        hw = FakeHardware()
        lc = MicroManagerLC(hw, settle_ms=0)
        lc.set_retardance("LCA", 99.0)
        assert lc.get_retardance("LCA") == pytest.approx(lc.limits.max_waves)

    def test_reads_back_rather_than_echoing_the_command(self):
        hw = FakeHardware()
        lc = MicroManagerLC(hw, settle_ms=0)
        lc.set_retardance("LCB", 0.47)
        hw.core.properties[("MeadowlarkLC", "Retardance LC-B [in waves]")] = "0.4699"
        assert lc.get_retardance("LCB") == pytest.approx(0.4699)

    def test_voltage_mode_without_a_curve_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="needs a retardance/voltage curve"):
            MicroManagerLC(FakeHardware(), mode="MM-Voltage")

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="unknown LC control mode"):
            MicroManagerLC(FakeHardware(), mode="DAC")

    def test_no_lamp_means_no_dark_frame(self):
        """So the library's black-level chain falls through rather than
        blocking on a person, which is what stopped recOrder working here."""
        with pytest.raises(NotImplementedError):
            MicroManagerLC(FakeHardware(), settle_ms=0).measure_dark()

    def test_dark_frame_darkens_and_restores(self):
        hw = FakeHardware()
        lc = MicroManagerLC(
            hw, settle_ms=0, lamp=("LED", "Intensity", 0, 100), dark_exposure_ms=1.0
        )
        dark = lc.measure_dark()
        assert dark == pytest.approx(102.0)
        assert hw.core.get_property("LED", "Intensity") == "100", "lamp left off"
        assert hw.exposure == pytest.approx(50.0), "exposure not restored"

    def test_lamp_is_restored_even_when_the_snap_fails(self):
        """Leaving the lamp off after an exception is the sort of state that
        makes the next acquisition mysterious."""
        hw = FakeHardware()

        def boom():
            raise RuntimeError("camera died")

        hw.snap_image = boom
        lc = MicroManagerLC(hw, settle_ms=0, lamp=("LED", "Intensity", 0, 100))
        with pytest.raises(RuntimeError, match="camera died"):
            lc.measure_dark()
        assert hw.core.get_property("LED", "Intensity") == "100"

    def test_missing_core_is_a_clear_error(self):
        class NoCore:
            pass

        with pytest.raises(RuntimeError, match="no .core"):
            MicroManagerLC(NoCore(), settle_ms=0).set_retardance("LCA", 0.25)


class TestWorkflow:
    def _yaml(self, **recon):
        base = {
            "swing_waves": 0.03,
            "scheme": "5-State",
            "wavelength_nm": 546.0,
            "lc_control_mode": "MM-Retardance",
            "lc_settle_ms": 0,
        }
        base.update(recon)
        return {"modalities": {"lcpolscope": {"reconstruction": base}}}

    def test_calibrates_and_writes_a_file(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(), self._yaml(), output_folder=str(tmp_path), black_level=102.0
        )
        assert result["success"] is True
        assert set(result["palette"]) == {f"State{i}" for i in range(5)}
        assert result["extinction_ratio"] > 100
        assert result["assessment"] == "good"

        written = json.loads((tmp_path / result["metadata_path"].split("/")[-1]).read_text())
        assert written["palette"] == result["palette"]
        assert written["trace"], "the per-exposure trace belongs in the file"
        assert "trace" not in result, "but not in the socket reply, which should stay small"

    def test_settings_come_from_the_yaml_so_they_cannot_drift(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(),
            self._yaml(swing_waves=0.05, scheme="4-State", wavelength_nm=549.0),
            output_folder=str(tmp_path),
            black_level=102.0,
        )
        assert result["swing_waves"] == 0.05
        assert result["scheme"] == "4-State"
        assert result["wavelength_nm"] == 549.0
        assert len(result["palette"]) == 4

    def test_voltage_mode_is_refused_not_silently_downgraded(self, tmp_path):
        """Quietly giving the operator retardance control would produce a
        working calibration stored the wrong way."""
        result = run_lc_calibration(
            FakeHardware(), self._yaml(lc_control_mode="MM-Voltage"), output_folder=str(tmp_path)
        )
        assert result["success"] is False
        assert "MM-Retardance" in result["error"]

    def test_failure_still_returns_a_dict(self, tmp_path):
        class Broken:
            pass

        result = run_lc_calibration(Broken(), self._yaml(), output_folder=str(tmp_path))
        assert result["success"] is False
        assert "error" in result and "warnings" in result

    def test_a_poor_calibration_is_returned_with_a_warning(self, tmp_path):
        hw = FakeHardware()
        original = hw.snap_image

        def flat(*_args, **_kw):
            image, meta = original()
            return image * 0 + 500.0, meta  # no contrast at all

        hw.snap_image = flat
        result = run_lc_calibration(
            hw, self._yaml(), output_folder=str(tmp_path), black_level=102.0
        )
        assert result["success"] is True, "a poor calibration is still a result"
        assert result["assessment"] in ("poor", "unmeasurable")

    def test_progress_callback_failures_do_not_kill_the_run(self, tmp_path):
        def bad(*_a):
            raise RuntimeError("UI went away")

        result = run_lc_calibration(
            FakeHardware(),
            self._yaml(),
            output_folder=str(tmp_path),
            black_level=102.0,
            progress_callback=bad,
        )
        assert result["success"] is True


def _waves_limits(lca, lcb):
    return {
        ("MeadowlarkLC", "Retardance LC-A [in waves]"): lca,
        ("MeadowlarkLC", "Retardance LC-B [in waves]"): lcb,
    }


class TestRetardanceLimitsComeFromTheDevice:
    """The 1.600-wave ceiling is recOrder's number, not this rig's.

    The D5020 on the LC-PolScope reports roughly 0 to 1 wave on the writable
    "[in waves]" properties. Keeping the inherited constant would let the
    clamp hand the adapter a value it refuses -- the clamp exists precisely
    so a search near a rail loses one measurement instead of the run.
    """

    def test_device_limits_replace_the_inherited_default(self):
        core = FakeCore(limits=_waves_limits((0.0, 1.0), (0.0, 1.0)))
        lc = MicroManagerLC(FakeHardware(core), settle_ms=0)

        assert lc.limits.max_waves == pytest.approx(1.0)
        # Not the library default, which is what this test exists to catch.
        assert lc.limits.max_waves != pytest.approx(1.600)
        # An exact zero is still avoided even though the device permits it.
        assert lc.limits.min_waves == pytest.approx(0.001)

    def test_the_narrower_axis_wins(self):
        """One RetardanceLimits clamps both crystals, so it must fit both."""
        core = FakeCore(limits=_waves_limits((0.0, 1.0), (0.05, 0.80)))
        lc = MicroManagerLC(FakeHardware(core), settle_ms=0)

        assert lc.limits.min_waves == pytest.approx(0.05)
        assert lc.limits.max_waves == pytest.approx(0.80)

    def test_a_commanded_value_is_clamped_to_what_the_device_accepts(self):
        core = FakeCore(limits=_waves_limits((0.0, 1.0), (0.0, 1.0)))
        lc = MicroManagerLC(FakeHardware(core), settle_ms=0)

        # 1.69 waves is where the config's leftover voltages land.
        lc.set_retardance("LCA", 1.69)

        written = float(core.properties[("MeadowlarkLC", "Retardance LC-A [in waves]")])
        assert written == pytest.approx(1.0)

    def test_an_adapter_declaring_no_limits_falls_back(self):
        """Not a reason to refuse: calibration works near a quarter wave."""
        lc = MicroManagerLC(FakeHardware(FakeCore()), settle_ms=0)

        assert lc.limits.max_waves == pytest.approx(1.600)

    def test_a_core_that_cannot_answer_falls_back(self):
        class Unhelpful(FakeCore):
            def has_property_limits(self, device, prop):
                raise RuntimeError("adapter does not implement this")

        lc = MicroManagerLC(FakeHardware(Unhelpful()), settle_ms=0)

        assert lc.limits.max_waves == pytest.approx(1.600)
