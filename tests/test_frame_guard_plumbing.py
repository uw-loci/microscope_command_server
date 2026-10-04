"""The acquisition workflow's half of the stage frame guard.

The guard itself lives in microscope_control (``hardware/xy_motion.py``, applied
at the XY choke point in ``hardware/stage.py``). These tests cover what the
server contributes: arming the check for exactly the window where we are the
only thing moving the stage, and turning a suspect frame into a stopped run.

``frame_guard`` is loaded by path because ``acquisition/__init__`` imports
microscope_control, which is not installed in every environment this suite runs
in. The module itself has no hardware imports, which is why it was extracted.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SRC = ROOT / "microscope_command_server" / "acquisition" / "workflow.py"


def _load(name: str):
    path = ROOT / "microscope_command_server" / "acquisition" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_fg_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fg = _load("frame_guard")


class FakeStage:
    def __init__(self, suspect=None, setter_raises=False):
        self.xy_frame_suspect = suspect
        self.armed = None
        self._setter_raises = setter_raises

    def set_frame_check_armed(self, armed):
        if self._setter_raises:
            raise RuntimeError("adapter does not support this")
        self.armed = armed


# --------------------------------------------------------------------------
# arming
# --------------------------------------------------------------------------


def test_arming_reaches_the_stage():
    stage = FakeStage()
    hardware = SimpleNamespace(stage=stage)
    fg.set_frame_check_armed(hardware, True)
    assert stage.armed is True
    fg.set_frame_check_armed(hardware, False)
    assert stage.armed is False


@pytest.mark.parametrize(
    "hardware",
    [None, SimpleNamespace(), SimpleNamespace(stage=object()), SimpleNamespace(stage=None)],
)
def test_arming_tolerates_hardware_without_the_hook(hardware):
    # Mock rigs and any stage predating the hook must still acquire.
    fg.set_frame_check_armed(hardware, True)


def test_arming_tolerates_a_stage_that_refuses():
    fg.set_frame_check_armed(SimpleNamespace(stage=FakeStage(setter_raises=True)), True)


# --------------------------------------------------------------------------
# stopping the run
# --------------------------------------------------------------------------


def test_a_healthy_frame_does_not_stop_the_run():
    fg.raise_if_frame_suspect(SimpleNamespace(stage=FakeStage()))
    fg.raise_if_frame_suspect(SimpleNamespace(stage=FakeStage(suspect=None)))
    # An empty reason is not a reason.
    fg.raise_if_frame_suspect(SimpleNamespace(stage=FakeStage(suspect="")))


@pytest.mark.parametrize("hardware", [None, SimpleNamespace(), SimpleNamespace(stage=object())])
def test_hardware_without_the_hook_does_not_stop_the_run(hardware):
    fg.raise_if_frame_suspect(hardware)
    assert fg.frame_suspect_reason(hardware) is None


def test_a_suspect_frame_stops_the_run_and_says_why():
    stage = FakeStage(
        suspect="Stage reports (0.0, 0.0) um but the last commanded position was "
        "(-41453.0, -7012.0) um"
    )
    with pytest.raises(RuntimeError) as excinfo:
        fg.raise_if_frame_suspect(SimpleNamespace(stage=stage))
    message = str(excinfo.value)
    # The operator has to be able to act on this without reading the source.
    assert "no longer trustworthy" in message
    assert "last commanded position" in message


# --------------------------------------------------------------------------
# the call sites, which are what make it a guard rather than a function
# --------------------------------------------------------------------------


def test_the_guard_runs_once_per_tile_in_the_acquisition_loop():
    """A guard only at the start of a region would miss a mid-region change.

    That is precisely the 2026-10-02 shape: slot 3 was still finding tissue
    thirty seconds before slot 4's first autofocus came up blank.
    """
    source = WORKFLOW_SRC.read_text()
    loop = "for pos_idx, (pos, filename) in enumerate(ctx.positions):"
    assert source.count(loop) == 1
    body = source.split(loop, 1)[1]
    head = "\n".join(body.splitlines()[:12])
    assert "raise_if_frame_suspect(hardware)" in head


def test_the_check_is_armed_before_autofocus_and_disarmed_on_cleanup():
    source = WORKFLOW_SRC.read_text()
    # Armed ahead of Phase 7/8 so the pre-acquisition autofocus's own stage
    # moves are covered -- its tissue search moves a FOV at a time and is the
    # first thing to run after a slot jump.
    assert source.index("set_frame_check_armed(hardware, True)") < source.index(
        "_configure_autofocus(ctx)"
    )
    # Disarmed in the guaranteed-cleanup path, so a joystick nudge after the
    # run is not reported as an origin change.
    assert source.index("set_frame_check_armed(ctx.hardware, False)") > source.index(
        "def _cleanup_acquisition("
    )
