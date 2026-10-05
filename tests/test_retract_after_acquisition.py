"""Retracting to the declared safe Z when an acquisition finishes.

The behaviour that matters is the refusals. A retraction that moves the wrong way drives
the objective into the slide, and the PPM config carries the scar of exactly that: a safe
Z of -500 inferred from "focus is near -400, so retract further" without establishing
which direction is away from the sample.

``retract.py`` takes plain callables for the stage, so these drive the real decision rather
than a reimplementation of it.
"""

import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SRC = ROOT / "microscope_command_server" / "acquisition" / "workflow.py"


def _load(name: str):
    path = ROOT / "microscope_command_server" / "acquisition" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_rt_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rt = _load("retract")


class FakeConfig:
    """Stands in for ConfigManager, whose only dict accessor is get_config()."""

    def __init__(self, stage=None, raises=False):
        self._cfg = {"stage": stage} if stage is not None else {}
        self._raises = raises

    def get_config(self):
        if self._raises:
            raise RuntimeError("no config loaded")
        return self._cfg


# PPM: safe Z is 0 and increasing Z moves the stage away from the objective.
PPM = {"safe_z_um": 0, "focus": {"retract_sign": "positive"}}


class Stage:
    def __init__(self, z):
        self.z = z
        self.moves = []

    def read(self):
        return self.z

    def move(self, z):
        self.moves.append(z)
        self.z = z


def run(config, stage, **kw):
    return rt.retract_to_safe_z(config, read_z=stage.read, move_z=stage.move, **kw)


# --------------------------------------------------------------------------
# the normal case
# --------------------------------------------------------------------------


def test_retracts_from_focus_to_the_safe_z():
    # The 2026-10-02 run sat at -418.4 um for every traverse, with safe Z at 0.
    stage = Stage(-418.4)
    assert run(FakeConfig(PPM), stage) is True
    assert stage.moves == [0.0]


def test_already_retracted_is_not_moved():
    stage = Stage(0.0)
    assert run(FakeConfig(PPM), stage) is False
    assert stage.moves == []


def test_a_sub_micron_difference_is_not_worth_a_move():
    stage = Stage(-0.4)
    assert run(FakeConfig(PPM), stage) is False
    assert stage.moves == []


def test_a_negative_retract_sign_rig_retracts_downward():
    # The opposite convention must work without special-casing PPM.
    stage = Stage(-418.4)
    config = FakeConfig({"safe_z_um": -700, "focus": {"retract_sign": "negative"}})
    assert run(config, stage) is True
    assert stage.moves == [-700.0]


# --------------------------------------------------------------------------
# the refusals, which are the point
# --------------------------------------------------------------------------


def test_a_wrong_side_safe_z_is_refused_not_driven_to(caplog):
    """The -500 mistake the PPM config warns about, with focus near -400."""
    stage = Stage(-400.0)
    config = FakeConfig({"safe_z_um": -500, "focus": {"retract_sign": "positive"}})
    with caplog.at_level(logging.ERROR):
        assert run(config, stage) is False
    assert stage.moves == []
    assert "SAMPLE side" in caplog.text
    assert "retract_sign" in caplog.text


def test_an_undeclared_direction_is_refused(caplog):
    stage = Stage(-418.4)
    config = FakeConfig({"safe_z_um": 0})
    with caplog.at_level(logging.WARNING):
        assert run(config, stage) is False
    assert stage.moves == []
    assert "retract_sign is not declared" in caplog.text


def test_an_unrecognised_direction_is_refused():
    stage = Stage(-418.4)
    config = FakeConfig({"safe_z_um": 0, "focus": {"retract_sign": "up"}})
    assert run(config, stage) is False
    assert stage.moves == []


def test_no_declared_safe_z_does_nothing():
    stage = Stage(-418.4)
    assert run(FakeConfig({"focus": {"retract_sign": "positive"}}), stage) is False
    assert stage.moves == []


def test_a_non_numeric_safe_z_does_nothing():
    stage = Stage(-418.4)
    config = FakeConfig({"safe_z_um": "retracted", "focus": {"retract_sign": "positive"}})
    assert run(config, stage) is False
    assert stage.moves == []


# --------------------------------------------------------------------------
# it must never turn a finished acquisition into a failed one
# --------------------------------------------------------------------------


def test_an_unreadable_config_does_not_raise():
    stage = Stage(-418.4)
    assert run(FakeConfig(raises=True), stage) is False
    assert stage.moves == []


def test_an_unreadable_z_does_not_raise():
    def boom():
        raise RuntimeError("serial command failed")

    assert rt.retract_to_safe_z(FakeConfig(PPM), read_z=boom, move_z=lambda z: None) is False


def test_a_failed_move_does_not_raise(caplog):
    def boom(z):
        raise RuntimeError("stage busy")

    with caplog.at_level(logging.WARNING):
        assert rt.retract_to_safe_z(FakeConfig(PPM), read_z=lambda: -418.4, move_z=boom) is False
    assert "failed" in caplog.text


# --------------------------------------------------------------------------
# the call site
# --------------------------------------------------------------------------


def test_the_retraction_precedes_the_end_of_region_return_move():
    """Order is the whole point: that return move was itself up to 12.7 mm at focus height.

    Retracting after it would leave the one move this is meant to cover uncovered.
    """
    source = WORKFLOW_SRC.read_text()
    retract_at = source.index("retract_to_safe_z(")
    # The return move inside _cleanup_acquisition.
    return_at = source.index("Returning to %s: X=%.1f, Y=%.1f")
    assert retract_at < return_at
    # And it sits in the guaranteed-cleanup path, so a cancelled or failed acquisition
    # also leaves the objective clear.
    cleanup_at = source.index("def _cleanup_acquisition(")
    assert retract_at > cleanup_at


def test_the_return_move_no_longer_claims_to_preserve_z():
    # It now runs at the retracted Z; a log line saying otherwise would send the next
    # investigation looking in the wrong place.
    source = WORKFLOW_SRC.read_text()
    assert "(preserving Z)" not in source
    assert "(at the retracted Z)" in source
