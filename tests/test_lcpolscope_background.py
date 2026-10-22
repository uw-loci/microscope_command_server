"""LC-PolScope backgrounds: collected averaged and with provenance, refused when stale.

A Stokes-space background describes the polarization states it was taken
with. After a recalibration those states change, and correcting new tiles
against the old background biases every one of them with nothing to see --
OpenPolScope requires a new background after every calibration. So the
collector records which calibration it belongs to, and the acquisition
refuses one that does not match.
"""

import json
import logging

import numpy as np
import pytest

from microscope_command_server.modality.lcpolscope import (
    BACKGROUND_PROVENANCE_FILE,
    background_problems,
)

RECON = {"lc_calibrated_at": "2026-10-16T14:45:18", "wavelength_nm": 549}


def _prov(**over):
    base = {
        "collected_at": "2026-10-16T15:30:00",
        "lc_calibrated_at": "2026-10-16T14:45:18",
        "frames": 10,
        "exposure_ms": {f"State{i}": 50.0 for i in range(5)},
        "lc_adapter_wavelength_nm": 549.0,
    }
    base.update(over)
    return base


class TestBackgroundProblems:
    def test_a_matching_background_is_clean(self):
        assert background_problems(_prov(), RECON, {f"State{i}": 50.0 for i in range(5)}) == (
            [],
            [],
        )

    def test_no_provenance_is_fatal(self):
        fatal, _ = background_problems({}, RECON)
        assert fatal and BACKGROUND_PROVENANCE_FILE in fatal[0]

    def test_a_different_calibration_is_fatal(self):
        fatal, _ = background_problems(_prov(lc_calibrated_at="2026-10-12T16:57:22"), RECON)
        assert fatal and "2026-10-12T16:57:22" in fatal[0] and "2026-10-16T14:45:18" in fatal[0]

    def test_a_background_for_a_calibration_when_none_is_applied_is_fatal(self):
        fatal, _ = background_problems(_prov(), {"wavelength_nm": 549})
        assert fatal and "Micro-Manager presets" in fatal[0]

    def test_exposure_and_wavelength_differences_are_advisory(self):
        fatal, advisory = background_problems(
            _prov(lc_adapter_wavelength_nm=546.0), RECON, {f"State{i}": 40.0 for i in range(5)}
        )
        assert fatal == []
        assert any("exposure" in a for a in advisory)
        assert any("546" in a for a in advisory)


class TestAcquisitionCheck:
    def _check(self, tmp_path, provenance, recon=RECON):
        from microscope_command_server.acquisition.workflow import _check_lcpolscope_background

        if provenance is not None:
            (tmp_path / BACKGROUND_PROVENANCE_FILE).write_text(json.dumps(provenance))
        images = {f"State{i}": np.ones((2, 2)) for i in range(5)}
        settings = {"modalities": {"lcpolscope": {"reconstruction": recon}}}
        params = {"channels": list(images), "channel_exposures": [50.0] * 5}
        result = _check_lcpolscope_background(
            "lcpolscope", tmp_path, settings, params, images, logging.getLogger("t")
        )
        return result, images

    def test_a_matching_background_is_kept_with_its_provenance(self, tmp_path):
        result, images = self._check(tmp_path, _prov())
        assert len(images) == 5
        assert result["lc_calibrated_at"] == RECON["lc_calibrated_at"]
        assert result["folder"] == str(tmp_path)

    def test_a_stale_background_is_dropped_and_said_so(self, tmp_path, caplog):
        with caplog.at_level(logging.ERROR):
            result, images = self._check(tmp_path, _prov(lc_calibrated_at="2026-10-12T16:57:22"))
        assert images == {} and result == {}
        assert any(
            "NOT used" in r.getMessage() and "WITHOUT" in r.getMessage() for r in caplog.records
        )

    def test_a_background_without_provenance_is_dropped(self, tmp_path):
        result, images = self._check(tmp_path, None)
        assert images == {} and result == {}

    def test_other_modalities_are_untouched(self, tmp_path):
        from microscope_command_server.acquisition.workflow import _check_lcpolscope_background

        images = {"DAPI": np.ones((2, 2))}
        assert (
            _check_lcpolscope_background(
                "fluorescence", tmp_path, {}, {}, images, logging.getLogger("t")
            )
            == {}
        )
        assert "DAPI" in images


class TestCollector:
    """The collector, against a fake microscope."""

    def _hardware(self, auto_gain="Off", value=1000.0, noise=0.0):
        from types import SimpleNamespace

        rng = np.random.default_rng(0)

        class Core:
            props = {
                ("Cam", "Gain Auto"): auto_gain,
                ("Cam", "Pixel Format"): "Mono12p",
                ("MeadowlarkLC", "Wavelength"): "549",
            }

            def get_camera_device(self):
                return "Cam"

            def get_image_bit_depth(self):
                return 12

            def get_device_property_names(self, device):
                return [p for (d, p) in self.props if d == device]

            def get_property(self, device, prop):
                return self.props[(device, prop)]

            def set_property(self, device, prop, value):
                self.props[(device, prop)] = str(value)

            def wait_for_device(self, device):
                pass

        snaps = []

        def snap_image(**_):
            snaps.append(1)
            return np.full((4, 4), value) + rng.normal(0, noise, (4, 4)), {}

        hw = SimpleNamespace(
            core=Core(),
            snap_image=snap_image,
            set_exposure=lambda ms: None,
            get_exposure=lambda: 50.0,
            get_camera_name=lambda: "Cam",
            get_pixel_size_um=lambda: 0.17,
        )
        hw.snaps = snaps
        return hw

    def _settings(self, frames=4):
        channels = [
            {
                "id": f"State{i}",
                "exposure_ms": 50,
                "device_properties": [
                    {
                        "device": "MeadowlarkLC",
                        "property": "Retardance LC-A [in waves]",
                        "value": 0.25,
                    }
                ],
            }
            for i in range(5)
        ]
        return {
            "acquisition_profiles": {"lcpolscope_20x": {"modality": "lcpolscope"}},
            "modalities": {
                "lcpolscope": {
                    "channels": channels,
                    "reconstruction": dict(RECON, background_frames=frames, black_level=103),
                }
            },
        }

    def _collect(self, tmp_path, hw, frames=4):
        from microscope_command_server.acquisition.workflow import _collect_channel_backgrounds

        return _collect_channel_backgrounds(
            [f"State{i}" for i in range(5)],
            str(tmp_path),
            "lcpolscope",
            "lcpolscope_20x",
            hw,
            self._settings(frames),
            lambda *a: None,
            logging.getLogger("t"),
            None,
            None,
        )

    def test_states_are_averaged_and_the_provenance_is_written(self, tmp_path):
        pytest.importorskip("tifffile")
        hw = self._hardware()
        self._collect(tmp_path, hw, frames=4)
        assert len(hw.snaps) == 20, "4 frames x 5 states"
        prov = json.loads((tmp_path / BACKGROUND_PROVENANCE_FILE).read_text())
        assert prov["frames"] == 4
        assert prov["lc_calibrated_at"] == RECON["lc_calibrated_at"]
        assert prov["lc_adapter_wavelength_nm"] == 549.0
        assert prov["states"] == [f"State{i}" for i in range(5)]
        assert prov["state_stats"]["State0"]["mean"] == pytest.approx(1000.0)
        assert prov["camera"]["significant_bits"] == 12
        for i in range(5):
            assert (tmp_path / f"State{i}.tif").exists()

    def test_a_camera_on_auto_gain_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError, match="Gain Auto"):
            self._collect(tmp_path, self._hardware(auto_gain="Continuous"))
        assert not (tmp_path / BACKGROUND_PROVENANCE_FILE).exists()


class TestStatePlan:
    LIB = [f"State{i}" for i in range(5)]

    def test_all_states_at_one_exposure_is_fine(self):
        from microscope_command_server.modality.lcpolscope import state_plan_problems

        order = ["State1", "State0", "State2", "State3", "State4"]  # focus state first
        assert state_plan_problems(order, [33.0] * 5, self.LIB) == []

    def test_a_missing_state_is_refused(self):
        from microscope_command_server.modality.lcpolscope import state_plan_problems

        problems = state_plan_problems(self.LIB[:4], [33.0] * 4, self.LIB)
        assert problems and "State4" in problems[0]

    def test_unequal_exposures_are_refused(self):
        from microscope_command_server.modality.lcpolscope import state_plan_problems

        problems = state_plan_problems(self.LIB, [33, 33, 40, 33, 33], self.LIB)
        assert problems and "State2=40 ms" in problems[0]


# --- swing-state spread: calibration imbalance vs specimen ---------------------------------


def _state_field(level, shape=(256, 256), patch=None):
    img = np.full(shape, float(level))
    if patch is not None:
        (r0, r1, c0, c1), factor = patch
        img[r0:r1, c0:c1] *= factor
    return img


def test_uniform_imbalance_has_a_flat_field_map():
    # 10-21 Hamamatsu means, uncalibrated palette: 28% apart, but every block alike.
    from microscope_command_server.acquisition.workflow import (
        _block_means,
        _swing_field_variation,
    )

    maps = [_block_means(_state_field(m)) for m in (28761, 33398, 25324, 29185)]
    assert _swing_field_variation(maps, black=103) < 0.001


def test_specimen_changes_the_balance_where_it_sits():
    from microscope_command_server.acquisition.workflow import (
        _block_means,
        _swing_field_variation,
    )

    patch = ((0, 128, 0, 128), 0.6)  # a birefringent particle dims one state in a quarter
    maps = [
        _block_means(_state_field(29000, patch=patch)),
        _block_means(_state_field(29000)),
        _block_means(_state_field(29000)),
        _block_means(_state_field(29000)),
    ]
    assert _swing_field_variation(maps, black=103) > 0.2


def test_spread_message_names_the_calibration_when_none_is_applied():
    from microscope_command_server.acquisition.workflow import _swing_spread_message

    msg = _swing_spread_message(0.28, {"State1": 1.0}, False, 0.004)
    assert "No QPSC LC calibration" in msg and "specimen" not in msg
    assert "0.4%" in msg
    calibrated = _swing_spread_message(0.28, {"State1": 1.0}, True, None)
    assert "specimen" in calibrated
    # Uncalibrated, but the balance changes across the field: something is in it.
    patchy = _swing_spread_message(0.28, {"State1": 1.0}, False, 2.0)
    assert "specimen" in patchy and "No QPSC LC calibration" not in patchy
