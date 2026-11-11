"""Each background profile keeps its own white balance beside its backgrounds.

``white_balance.yml`` in ``<profile>/<detector>/<family>/<mag>/<per_angle|simple>/``
holds the same keys as the scope-wide block in the imageprocessing YAML; readers given
that folder use it, and fall back to the scope-wide block when it holds none.
"""

import logging
from unittest.mock import MagicMock

import yaml

from microscope_command_server.acquisition import workflow

LOG = logging.getLogger("test")
OBJ = "LOCI_OBJECTIVE_OLYMPUS_20X_POL_001"
DET = "LOCI_DETECTOR_JAI_001"


def _per_angle(g):
    return {
        "exposures_ms": {
            name: {"all": g, "r": g / 2, "g": g, "b": g * 2}
            for name in ("positive", "negative", "crossed", "uncrossed")
        },
        "gains": {
            name: {"unified_gain": 1.0, "analog_red": 1.0, "analog_blue": 4.0, "wb_method": "x"}
            for name in ("positive", "negative", "crossed", "uncrossed")
        },
        "wb_last_modified": "2026-11-11T12:00:00",
    }


def _scope(tmp_path, g=10.0, simple=None):
    config = tmp_path / "config_PPM.yml"
    config.write_text("microscope: {name: PPM}\n")
    block = _per_angle(g)
    if simple:
        block["simple_wb"] = simple
    (tmp_path / "imageprocessing_PPM.yml").write_text(
        yaml.safe_dump({"imaging_profiles": {"ppm": {OBJ: {DET: block}}}})
    )
    return config


def test_a_profile_white_balance_is_used_instead_of_the_scope_wide_one(tmp_path):
    config = _scope(tmp_path, g=10.0)
    folder = tmp_path / "profiles" / "gel" / DET / "ppm" / "20x" / "per_angle"
    workflow.write_profile_white_balance(folder, _per_angle(3.0), "per_angle", "ppm", OBJ, DET)

    cal = workflow.load_jai_calibration_from_imageprocessing(
        config,
        per_angle=True,
        modality="ppm",
        objective=OBJ,
        detector=DET,
        logger=LOG,
        wb_folder=folder,
    )
    assert cal["angles"]["positive"]["exposures_ms"]["g"] == 3.0

    scope = workflow.load_jai_calibration_from_imageprocessing(
        config, per_angle=True, modality="ppm", objective=OBJ, detector=DET, logger=LOG
    )
    assert scope["angles"]["positive"]["exposures_ms"]["g"] == 10.0


def test_a_profile_without_white_balance_falls_back_to_the_scope_wide_one(tmp_path):
    config = _scope(tmp_path, g=10.0)
    empty = tmp_path / "profiles" / "new" / "per_angle"
    empty.mkdir(parents=True)
    cal = workflow.load_jai_calibration_from_imageprocessing(
        config,
        per_angle=True,
        modality="ppm",
        objective=OBJ,
        detector=DET,
        logger=LOG,
        wb_folder=empty,
    )
    assert cal["angles"]["positive"]["exposures_ms"]["g"] == 10.0


def test_simple_white_balance_comes_from_the_profile(tmp_path):
    scope_simple = {"base_angle": "uncrossed", "angles": {"uncrossed": {"scale": 1.0}}}
    config = _scope(tmp_path, simple=scope_simple)
    folder = tmp_path / "simple"
    block = {
        **_per_angle(2.0),
        "simple_wb": {"base_angle": "uncrossed", "angles": {"positive": {"scale": 9.0}}},
    }
    workflow.write_profile_white_balance(folder, block, "simple", "ppm", OBJ, DET)

    simple = workflow.load_simple_wb_from_imageprocessing(
        config, modality="ppm", objective=OBJ, detector=DET, logger=LOG, wb_folder=folder
    )
    assert simple["angles"]["positive"]["scale"] == 9.0


def test_the_file_records_what_it_is(tmp_path):
    path = workflow.write_profile_white_balance(
        tmp_path, _per_angle(3.0), "per_angle", "ppm", OBJ, DET
    )
    data = yaml.safe_load(path.read_text())
    assert data["mode"] == "per_angle"
    assert data["objective"] == OBJ and data["detector"] == DET
    assert data["white_balance"]["exposures_ms"]["positive"]["g"] == 3.0


def test_a_calibration_is_copied_into_the_profile(tmp_path):
    from microscope_command_server.server.handlers import calibration

    config = _scope(tmp_path, g=7.0, simple={"base_angle": "uncrossed", "angles": {}})
    folder = tmp_path / "profiles" / "gel" / "per_angle"
    params = {"yaml_file_path": str(config), "wb_folder": str(folder)}

    calibration._store_profile_white_balance(params, "per_angle", "ppm", OBJ, DET)

    stored = workflow.read_profile_white_balance(folder)
    assert stored["exposures_ms"]["positive"]["g"] == 7.0
    assert "simple_wb" not in stored, "per-angle stores only the per-angle values"


def test_the_acquisition_flag_is_parsed():
    params = workflow.parse_acquisition_message(
        "--yaml c.yml --projects p --sample s --scan-type ppm_20x_1 --region bounds "
        "--wb-mode per_angle --wb-folder D:/bg/profiles/gel/JAI/ppm/20x/per_angle END_MARKER"
    )
    assert params["wb_folder"] == "D:/bg/profiles/gel/JAI/ppm/20x/per_angle"


def test_stack_snaps_apply_per_angle_white_balance(monkeypatch):
    from microscope_command_server.acquisition import stack_timelapse

    seen = {}
    monkeypatch.setattr(
        workflow, "apply_jai_calibration_for_angle", lambda **kw: seen.update(kw) or (True, None)
    )
    stack_timelapse._apply_wb_for_snap(MagicMock(), {"angles": {}}, 7.0, "ppm", LOG)
    assert seen["per_angle"] is True and seen["angle"] == 7.0
