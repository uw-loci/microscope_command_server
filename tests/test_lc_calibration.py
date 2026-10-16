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

    def __init__(self, failing_device=None, limits=None, bit_depth=12):
        self.properties = {
            ("MeadowlarkLC", "Retardance LC-A [in waves]"): "0.25",
            ("MeadowlarkLC", "Retardance LC-B [in waves]"): "0.50",
            ("LED", "Intensity"): "100",
            ("FakeCam", "Binning"): "1",
            ("FakeCam", "PixelType"): "12bit",
            ("FakeCam", "Gain"): "0",
            ("FakeCam", "GainAuto"): "Off",
        }
        self.bit_depth = bit_depth
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

    def get_device_property_names(self, device):
        return [prop for (dev, prop) in self.properties if dev == device]

    def set_property(self, device, prop, value):
        if device == self.failing_device:
            raise RuntimeError("device on fire")
        self.properties[(device, prop)] = value
        self.writes.append((device, prop, value))

    def get_property(self, device, prop):
        return self.properties[(device, prop)]

    def wait_for_device(self, device):
        self.waits.append(device)

    # -- camera, as CMMCore exposes it --------------------------------------

    def get_camera_device(self):
        return "FakeCam"

    def get_image_bit_depth(self):
        return self.bit_depth

    def get_bytes_per_pixel(self):
        return 2

    def get_image_width(self):
        return 4

    def get_image_height(self):
        return 4


class FakeHardware:
    """A microscope whose camera sees the polarimetric forward model.

    ``transmittance`` is the camera counts at full transmission, so a dark
    field is one number away; ``black_level`` is the camera's own offset.
    """

    # 20000 counts at full transmission on a 12-bit camera: the far grid points
    # clip (allowed) and the swing states, at 0.9% of that, sit ~180 counts
    # above extinction -- a properly exposed calibration. The simulator's own
    # default of 4000 puts the contrast at 40 counts, 0.98% of range, right on
    # the warning threshold.
    def __init__(self, core=None, transmittance=20000.0, black_level=102.0):
        self.core = core or FakeCore()
        self.exposure = 50.0
        self.exposure_log = []
        self.transmittance = transmittance
        self.black_level = black_level

    def snap_image(self):
        from polscope_library.calibration.simulator import SimulatedPolScope

        lca = float(self.core.get_property("MeadowlarkLC", "Retardance LC-A [in waves]"))
        lcb = float(self.core.get_property("MeadowlarkLC", "Retardance LC-B [in waves]"))
        lamp_on = self.core.get_property("LED", "Intensity") != "0"
        scope = SimulatedPolScope(
            read_noise=0.0, transmittance=self.transmittance, black_level=self.black_level
        )
        value = scope.intensity_at(lca, lcb) if lamp_on else scope.black_level
        # A 12-bit camera delivering uint16: clip at full scale as the real one does.
        return np.full((4, 4), min(value, 4095.0), dtype=np.uint16), {}

    def get_camera_name(self):
        return "FakeCam"

    def set_exposure(self, ms):
        self.exposure = float(ms)
        self.exposure_log.append(float(ms))

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
    def _yaml(self, channel_exposures=None, **recon):
        base = {
            "swing_waves": 0.03,
            "scheme": "5-State",
            "wavelength_nm": 546.0,
            "lc_control_mode": "MM-Retardance",
            "lc_settle_ms": 0,
        }
        base.update(recon)
        modality = {"reconstruction": base}
        if channel_exposures is not None:
            modality["channels"] = [
                {"id": f"State{i}", "exposure_ms": ms} for i, ms in enumerate(channel_exposures)
            ]
        return {"modalities": {"lcpolscope": modality}}

    def test_calibrates_and_writes_a_file(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(), self._yaml(), output_folder=str(tmp_path), black_level=102.0
        )
        assert result["success"] is True
        assert set(result["palette_waves"]) == {f"State{i}" for i in range(5)}
        assert result["extinction_ratio"] > 100
        assert result["assessment"] == "good"

        written = json.loads((tmp_path / result["metadata_path"].split("/")[-1]).read_text())
        assert written["palette_waves"] == result["palette_waves"]
        assert written["trace"], "the per-exposure trace belongs in the file"
        assert {"lca_waves", "lcb_waves"} <= set(
            written["trace"][0]
        ), "trace values carry their unit"
        # QPSC writes the palette into the YAML against these names, so the
        # server, which owns the adapter's property names, has to send them.
        assert result["lc_device"] == "MeadowlarkLC"
        assert result["lc_retardance_properties_waves"] == {
            "LCA": "Retardance LC-A [in waves]",
            "LCB": "Retardance LC-B [in waves]",
        }
        assert result["calibrated_at"]
        assert "trace" not in result, "but not in the socket reply, which should stay small"

    def test_black_level_falls_back_to_the_yaml_before_assuming_zero(self, tmp_path):
        # The rig cannot darken the field, so without this the library would
        # report "0.0 (assumed)" with the camera's real offset sitting in the YAML.
        result = run_lc_calibration(
            FakeHardware(), self._yaml(black_level=103), output_folder=str(tmp_path)
        )
        assert result["black_level"] == 103.0
        assert result["black_level_source"].startswith("config:")
        assert "black_level" in result["black_level_source"]

    def test_requested_black_level_wins_over_the_yaml(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(),
            self._yaml(black_level=103),
            output_folder=str(tmp_path),
            black_level=102.0,
        )
        assert result["black_level"] == 102.0
        assert result["black_level_source"] == "request"

    def test_a_zero_black_level_in_the_yaml_is_not_treated_as_a_measurement(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(), self._yaml(black_level=0), output_folder=str(tmp_path)
        )
        assert result["black_level_source"] == "assumed"
        assert any("assuming 0" in w for w in result["warnings"])

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
        assert len(result["palette_waves"]) == 4

    def test_adapter_state_is_recorded_and_a_wavelength_disagreement_is_said(self, tmp_path):
        """The palette is in the ADAPTER's waves. Its wavelength setting is the
        one value in this chain nobody had been reading; now it is recorded,
        and compared with the reconstruction wavelength out loud."""
        hw = FakeHardware()
        hw.core.properties[("MeadowlarkLC", "Wavelength")] = "546.0"
        result = run_lc_calibration(
            hw, self._yaml(wavelength_nm=549.0), output_folder=str(tmp_path), black_level=102.0
        )
        assert result["success"] is True
        assert result["lc_adapter_wavelength_nm"] == 546.0
        assert any("546 nm" in w and "549 nm" in w and "percent" in w for w in result["warnings"])

        written = json.loads((tmp_path / result["metadata_path"].split("/")[-1]).read_text())
        snapshot = written["lc_device_properties"]
        assert snapshot["Wavelength"] == "546.0"
        assert "Retardance LC-A [in waves]" in snapshot
        assert "lc_device_properties" not in result, "the snapshot stays out of the socket reply"

    def test_matching_adapter_wavelength_raises_no_warning(self, tmp_path):
        hw = FakeHardware()
        hw.core.properties[("MeadowlarkLC", "Wavelength")] = "549.0"
        result = run_lc_calibration(
            hw, self._yaml(wavelength_nm=549.0), output_folder=str(tmp_path), black_level=102.0
        )
        assert result["lc_adapter_wavelength_nm"] == 549.0
        assert not any("percent" in w for w in result["warnings"])

    def test_an_adapter_without_a_wavelength_property_is_simply_unknown(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(), self._yaml(), output_folder=str(tmp_path), black_level=102.0
        )
        assert result["lc_adapter_wavelength_nm"] is None

    def test_camera_state_is_recorded_with_every_calibration(self, tmp_path):
        """A frame mean in counts is unreadable without the exposure and the
        pixel format. The first two rig runs were analysed without either."""
        result = run_lc_calibration(
            FakeHardware(), self._yaml(), output_folder=str(tmp_path), black_level=102.0
        )
        camera = result["camera"]
        assert camera["device"] == "FakeCam"
        assert camera["exposure_ms"] == 50.0
        assert camera["exposure_source"] == "micro-manager"
        assert camera["bit_depth"] == 12
        assert camera["binning"] == "1"
        assert camera["pixel_type"] == "12bit"
        assert camera["gain"] == "0", "GainAuto must not be mistaken for the gain"
        signal = result["signal"]
        assert signal["full_scale"] == 4095.0
        assert signal["frame_dtype"] == "uint16"
        assert signal["grid_peak"] > signal["brightest_state"] > signal["extinction"]
        assert 0 < signal["grid_peak_fraction"] <= 1

        written = json.loads((tmp_path / result["metadata_path"].split("/")[-1]).read_text())
        assert written["camera_properties"]["PixelType"] == "12bit"
        assert "camera_properties" not in result, "every camera property stays in the file"

    def test_a_requested_exposure_is_set_recorded_and_restored(self, tmp_path):
        hw = FakeHardware()
        result = run_lc_calibration(
            hw, self._yaml(), output_folder=str(tmp_path), black_level=102.0, exposure_ms=120.0
        )
        assert result["camera"]["exposure_ms"] == 120.0
        assert result["camera"]["exposure_source"] == "request"
        assert hw.exposure_log[0] == 120.0
        assert hw.exposure == 50.0, "the acquisition's exposure is put back"

    def test_a_dark_field_is_refused_early_with_the_camera_state(self, tmp_path):
        """The 2026-10-13 rig run: frame means of 1 to 17 counts, searched for
        32 s, and handed back a palette with an Apply button."""
        hw = FakeHardware(transmittance=15.0, black_level=1.0)
        result = run_lc_calibration(
            hw, self._yaml(), output_folder=str(tmp_path), black_level=1.0, exposure_ms=5.0
        )
        assert result["success"] is False
        assert "too dark" in result["error"]
        assert "5 ms" in result["error"] and "12-bit" in result["error"]
        assert result["camera"]["exposure_ms"] == 5.0
        assert hw.exposure == 50.0, "restored on the failure path too"
        assert hw.core.writes, "it did look before refusing"

    def test_a_black_level_above_the_darkest_frame_is_called_out(self, tmp_path):
        """103 against frames that never reached 17: not this camera's offset
        in these units. The ratio is unmeasurable and the warning says why."""
        result = run_lc_calibration(
            FakeHardware(transmittance=4000.0, black_level=5.0),
            self._yaml(black_level=103),
            output_folder=str(tmp_path),
        )
        assert result["success"] is True
        assert result["assessment_by_ratio"] == "unmeasurable"
        assert result["assessment"] == "unreliable"
        assert "black level above the darkest state" in result["reliability_issues"]
        assert any(
            "above the darkest state" in w and "pixel format" in w for w in result["warnings"]
        )

    def test_swing_contrast_and_leakage_are_reported(self, tmp_path):
        """The two numbers a palette stands or falls on, in raw counts."""
        result = run_lc_calibration(
            FakeHardware(), self._yaml(), output_folder=str(tmp_path), black_level=102.0
        )
        signal = result["signal"]
        assert signal["extinction_above_black"] == pytest.approx(
            signal["extinction"] - 102.0, abs=0.01
        )
        assert signal["swing_contrast"] == pytest.approx(
            signal["brightest_state"] - signal["extinction"], abs=0.01
        )
        assert signal["swing_contrast_fraction"] > 0.01
        assert not any("Swing contrast" in w for w in result["warnings"])
        assert set(signal["state_saturated_fraction"]) == {f"State{i}" for i in range(5)}
        assert all(v == 0 for v in signal["state_saturated_fraction"].values())

    def test_a_low_swing_contrast_passes_the_grid_guard_but_is_warned_about(self, tmp_path):
        # Bright enough for the grid (a tenth of 4095 is 410; the far grid
        # points reach ~half of transmittance) yet the swing states, at 0.9%
        # of transmittance, are a handful of counts apart.
        result = run_lc_calibration(
            FakeHardware(transmittance=1500.0, black_level=5.0),
            self._yaml(),
            output_folder=str(tmp_path),
            black_level=5.0,
        )
        assert result["success"] is True
        assert result["signal"]["swing_contrast"] < 41
        assert any("Swing contrast" in w and "noise" in w for w in result["warnings"])

    def test_a_clipped_state_is_warned_about(self, tmp_path):
        # Swing states at 0.9% of 500k transmittance sit at ~4500 counts, over
        # a 12-bit camera's 4095; the grid points clip too, which is allowed.
        result = run_lc_calibration(
            FakeHardware(transmittance=500_000.0, black_level=5.0),
            self._yaml(),
            output_folder=str(tmp_path),
            black_level=5.0,
        )
        assert result["success"] is True
        saturated = result["signal"]["state_saturated_fraction"]
        assert any(v > 0.001 for v in saturated.values())
        assert any("top of the camera's range" in w for w in result["warnings"])

    def test_the_acquisition_exposure_is_projected_and_a_dark_one_is_warned_about(self, tmp_path):
        """A palette taken at 500 ms is fine; an acquisition at 50 ms is dark.
        That is this week's problem moved one step along, so say it now."""
        result = run_lc_calibration(
            FakeHardware(),
            self._yaml(channel_exposures=[5, 5, 5, 5, 5]),
            output_folder=str(tmp_path),
            black_level=102.0,
            exposure_ms=50.0,
        )
        signal = result["signal"]
        assert signal["acquisition_exposure_ms"] == 5.0
        assert signal["projected_swing_contrast"] == pytest.approx(
            signal["swing_contrast"] / 10, rel=1e-6
        )
        low, high = signal["usable_exposure_ms"]
        assert low is not None and high is not None and low < high
        assert any(
            "acquisition exposure of 5 ms" in w and "at least" in w for w in result["warnings"]
        )

    def test_an_acquisition_exposure_that_would_clip_is_warned_about(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(),
            self._yaml(channel_exposures=[5000, 5000, 5000, 5000, 5000]),
            output_folder=str(tmp_path),
            black_level=102.0,
            exposure_ms=50.0,
        )
        assert result["signal"]["projected_brightest_pixel"] >= 4095
        assert any("would reach" in w and "at most" in w for w in result["warnings"])

    def test_a_matching_acquisition_exposure_raises_no_projection_warning(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(),
            self._yaml(channel_exposures=[50, 50, 50, 50, 50]),
            output_folder=str(tmp_path),
            black_level=102.0,
            exposure_ms=50.0,
        )
        assert result["signal"]["projected_swing_contrast"] == result["signal"]["swing_contrast"]
        assert not any("acquisition exposure" in w for w in result["warnings"])

    def test_disagreeing_channel_exposures_are_reported_not_projected(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(),
            self._yaml(channel_exposures=[50, 50, 60, 50, 50]),
            output_folder=str(tmp_path),
            black_level=102.0,
        )
        assert result["signal"]["acquisition_exposure_ms"] is None
        assert any("different exposures" in w for w in result["warnings"])

    def test_a_histogram_image_is_written_beside_the_json(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(), self._yaml(), output_folder=str(tmp_path), black_level=102.0
        )
        image = tmp_path / result["signal"]["histogram_path"].split("/")[-1]
        assert image.exists() and image.stat().st_size > 1000
        assert (
            image.stem
            == result["metadata_path"].split("/")[-1].replace(".json", "") + "_histograms"
        )

    def test_auto_gain_is_refused_before_the_search(self, tmp_path):
        """2026-10-16: Gain Auto Continuous; five states at 173 counts, the
        camera's target; "good". The frames were the camera's doing."""
        hw = FakeHardware()
        hw.core.properties[("FakeCam", "GainAuto")] = "Continuous"
        result = run_lc_calibration(
            hw, self._yaml(), output_folder=str(tmp_path), black_level=102.0, exposure_ms=20.0
        )
        assert result["success"] is False
        assert "GainAuto" in result["error"] and "Off" in result["error"]
        assert "camera" in result
        assert hw.exposure == 50.0, "exposure restored on this failure path too"
        assert not any(
            w[1] == "Retardance LC-A [in waves]" for w in hw.core.writes
        ), "refused before moving a crystal"

    def test_gamma_and_8_bit_mark_the_result_unreliable_without_refusing(self, tmp_path):
        hw = FakeHardware(core=FakeCore(bit_depth=8), transmittance=200.0, black_level=2.0)
        hw.core.properties[("FakeCam", "Gamma Enabled")] = "1"
        result = run_lc_calibration(hw, self._yaml(), output_folder=str(tmp_path), black_level=2.0)
        assert result["success"] is True
        assert result["assessment"] == "unreliable"
        assert result["assessment_by_ratio"] in ("good", "acceptable", "poor", "unmeasurable")
        assert "camera gamma curve" in result["reliability_issues"]
        assert "8-bit frames" in result["reliability_issues"]

    def test_a_clipped_state_makes_the_assessment_unreliable_not_good(self, tmp_path):
        """The 2026-10-16 headline: "Good. This calibration is ready to use."
        over a State1 that was 99.9% saturated."""
        result = run_lc_calibration(
            FakeHardware(transmittance=500_000.0, black_level=5.0),
            self._yaml(),
            output_folder=str(tmp_path),
            black_level=5.0,
        )
        assert result["assessment"] == "unreliable"
        assert any("clipped" in issue for issue in result["reliability_issues"])

    def test_a_clean_run_keeps_the_ratio_band_as_its_assessment(self, tmp_path):
        result = run_lc_calibration(
            FakeHardware(), self._yaml(), output_folder=str(tmp_path), black_level=102.0
        )
        assert result["assessment"] == result["assessment_by_ratio"] == "good"
        assert result["reliability_issues"] == []

    def test_an_unknown_bit_depth_disables_the_guard_and_says_so(self, tmp_path):
        class NoDepthCore(FakeCore):
            def get_image_bit_depth(self):
                raise RuntimeError("no camera")

        result = run_lc_calibration(
            FakeHardware(core=NoDepthCore(), transmittance=15.0, black_level=1.0),
            self._yaml(),
            output_folder=str(tmp_path),
            black_level=1.0,
        )
        assert result["success"] is True, "no bit depth means no threshold, not a refusal"
        assert result["signal"]["full_scale"] is None
        assert any("bit depth" in w for w in result["warnings"])

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
        assert result["assessment_by_ratio"] in ("poor", "unmeasurable")
        # Flat frames have no swing contrast, one of the things that makes the
        # ratio untrustworthy, so the headline says so rather than "poor".
        assert result["assessment"] == "unreliable"

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
