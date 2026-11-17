"""One config per microscope: the active camera picks the profile and the per-camera values.

The LC-PolScope keeps one config for its two cameras. Each acquisition profile names
its camera, and camera-dependent values sit in ``per_detector`` blocks. The server
resolves a background profile among the active camera's profiles (the same rule as
QPSC's MicroscopeConfigManager.resolveProfileKey) and, on CONFIG, merges the active
camera's per_detector values into the settings every handler sees.
"""

import logging
from types import SimpleNamespace

import pytest
import yaml

from microscope_command_server.acquisition.workflow import _resolve_background_profile_key
from microscope_command_server.server.handlers.system import apply_active_detector
from microscope_control.config import ConfigManager

ORYX = "LCPS_DETECTOR_FLIR_ORYX_001"
HAM = "LCPS_DETECTOR_HAMAMATSU_FLASH_001"
OBJ20 = "LCPS_OBJECTIVE_20X_POL_001"
OBJ4 = "LCPS_OBJECTIVE_4X_001"

PROFILES = {
    "lcpolscope_20x": {"modality": "lcpolscope", "detector": HAM, "objective": OBJ20},
    "brightfield_20x": {"modality": "brightfield", "detector": HAM, "objective": OBJ20},
    "lcpolscope_4x": {"modality": "lcpolscope", "detector": HAM, "objective": OBJ4},
    "lcpolscope_20x_oryx": {"modality": "lcpolscope", "detector": ORYX, "objective": OBJ20},
    "brightfield_20x_oryx": {"modality": "brightfield", "detector": ORYX, "objective": OBJ20},
    "lcpolscope_4x_oryx": {"modality": "lcpolscope", "detector": ORYX, "objective": OBJ4},
}

LOG = logging.getLogger("test")


def _hw(detector, profiles=PROFILES):
    return SimpleNamespace(
        settings={"acquisition_profiles": profiles}, _active_detector_id=detector
    )


@pytest.mark.parametrize(
    "modality, objective, detector, expected",
    [
        ("lcpolscope", OBJ20, HAM, "lcpolscope_20x"),
        ("lcpolscope", OBJ20, ORYX, "lcpolscope_20x_oryx"),
        ("lcpolscope", OBJ4, ORYX, "lcpolscope_4x_oryx"),
        ("Brightfield", OBJ20, ORYX, "brightfield_20x_oryx"),
        ("brightfield", OBJ20, HAM, "brightfield_20x"),
        # No camera known: the first in file order.
        ("lcpolscope", OBJ20, None, "lcpolscope_20x"),
    ],
)
def test_background_profile_follows_the_active_camera(modality, objective, detector, expected):
    assert _resolve_background_profile_key(modality, objective, _hw(detector), LOG) == expected


def test_name_rules_still_resolve_profiles_without_fields():
    # The OWS3 / PPM shape: keys only, no modality/objective fields, no detector.
    profiles = {"Brightfield_10x": {}, "Brightfield_20x": {}, "Fluorescence_10x": {}}
    hw = _hw("ANY_CAMERA", profiles)
    assert _resolve_background_profile_key("Brightfield", "0.5NA_AIR_10x", hw, LOG) == (
        "Brightfield_10x"
    )
    assert _resolve_background_profile_key("brightfield", "0.75NA_AIR_20x", hw, LOG) == (
        "Brightfield_20x"
    )


def test_name_rules_skip_the_other_cameras_profiles():
    profiles = {"Brightfield_10x": {"detector": HAM}}
    assert _resolve_background_profile_key("Brightfield", "x_10x", _hw(ORYX, profiles), LOG) is None
    assert _resolve_background_profile_key("Brightfield", "x_10x", _hw(HAM, profiles), LOG) == (
        "Brightfield_10x"
    )


def test_unresolvable_inputs_return_none():
    assert _resolve_background_profile_key("", OBJ20, _hw(HAM), LOG) is None
    assert _resolve_background_profile_key("shg", OBJ20, _hw(HAM), LOG) is None
    assert _resolve_background_profile_key("lcpolscope", OBJ20, _hw(HAM, {}), LOG) is None


# ---------------------------------------------------------------- CONFIG overlay

CONFIG = {
    "microscope": {"name": "LCPolScope"},
    "stage": {"stage_id": "S"},
    "modalities": {
        "lcpolscope": {
            "reconstruction": {
                "black_level": 51,
                "per_detector": {ORYX: {"black_level": 99}, HAM: {"black_level": 51}},
            },
            "channels": [
                {
                    "id": "State0",
                    "exposure_ms": 141.5,
                    "per_detector": {ORYX: {"exposure_ms": 30}, HAM: {"exposure_ms": 141.5}},
                }
            ],
        }
    },
}


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config_LCPolScope.yml"
    path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    return path


def _black_level(settings):
    return settings["modalities"]["lcpolscope"]["reconstruction"]["black_level"]


def test_config_applies_the_active_cameras_values_everywhere(config_path, tmp_path, caplog):
    manager = ConfigManager(str(tmp_path / "unused"))
    hardware = SimpleNamespace(_active_detector_id=ORYX, _camera_name="Oryx", settings=None)
    raw = manager.load_config_file(str(config_path), apply_detector_values=False)

    with caplog.at_level(logging.INFO):
        merged = apply_active_detector(manager, hardware, raw)

    assert _black_level(merged) == 99
    assert merged["modalities"]["lcpolscope"]["channels"][0]["exposure_ms"] == 30
    assert hardware.settings is merged
    assert _black_level(raw) == 51  # the raw load is left alone
    # Every later load (the acquisition workflows reload the YAML) gets the same camera.
    assert manager.active_detector == ORYX
    assert _black_level(manager.load_config_file(str(config_path))) == 99
    assert any("per_detector values applied" in r.getMessage() for r in caplog.records)


def test_config_with_an_unknown_camera_uses_the_shared_values(config_path, tmp_path, caplog):
    manager = ConfigManager(str(tmp_path / "unused"))
    manager.active_detector = ORYX  # from an earlier CONFIG
    hardware = SimpleNamespace(_active_detector_id=None, _camera_name="Mystery", settings=None)
    raw = manager.load_config_file(str(config_path), apply_detector_values=False)

    with caplog.at_level(logging.INFO):
        merged = apply_active_detector(manager, hardware, raw)

    assert _black_level(merged) == 51
    assert manager.active_detector is None
    assert any("matches no detector" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- mirror of QPSC's tests
# The three cases of QPSC's ProfileDetectorResolutionTest (qupath-extension-qpsc,
# MicroscopeConfigManager.matchProfileKey). Keep the two in step.

JAVA_PROFILES = {
    "lcpolscope_20x": {"modality": "lcpolscope", "objective": OBJ20, "detector": HAM},
    "lcpolscope_20x_oryx": {"modality": "lcpolscope", "objective": OBJ20, "detector": ORYX},
    "brightfield_20x": {"modality": "brightfield", "detector": HAM},
    "brightfield_20x_oryx": {"modality": "brightfield", "detector": ORYX},
}


def test_java_mirror_the_active_cameras_profile_wins_by_its_fields():
    hw_ham, hw_oryx = _hw(HAM, JAVA_PROFILES), _hw(ORYX, JAVA_PROFILES)
    assert _resolve_background_profile_key("lcpolscope", OBJ20, hw_ham, LOG) == "lcpolscope_20x"
    assert _resolve_background_profile_key("LCPolScope", OBJ20, hw_oryx, LOG) == (
        "lcpolscope_20x_oryx"
    )


def test_java_mirror_without_objective_fields_name_rules_run_on_the_cameras_profiles_only():
    hw_ham, hw_oryx = _hw(HAM, JAVA_PROFILES), _hw(ORYX, JAVA_PROFILES)
    assert _resolve_background_profile_key("brightfield", OBJ20, hw_ham, LOG) == "brightfield_20x"
    # "20x_oryx" is not in the objective id, and the Hamamatsu profile is not a candidate.
    assert _resolve_background_profile_key("brightfield", OBJ20, hw_oryx, LOG) is None


def test_java_mirror_an_unknown_camera_considers_every_profile_in_file_order():
    hw = _hw(None, JAVA_PROFILES)
    assert _resolve_background_profile_key("lcpolscope", OBJ20, hw, LOG) == "lcpolscope_20x"
