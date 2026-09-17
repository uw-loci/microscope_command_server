"""End-of-acquisition teardown leaves the light path dark for channel modalities.

The tile loop ends with the last acquired channel still applied. For fluorescence that
keeps exciting the sample and leaves the Camera tab describing a channel that is not the
one on the hardware. _cleanup_acquisition runs on success, cancel AND failure, so this is
where the teardown belongs.
"""

import ast
import pathlib

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "acquisition"
    / "workflow.py"
)
SOURCE = WORKFLOW.read_text()


def load_function(name, namespace):
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(WORKFLOW), "exec"), namespace
            )
            return namespace[name]
    raise AssertionError(f"{name} not found")


class FakeLogger:
    def __init__(self):
        self.lines = []

    def _log(self, msg, *a):
        self.lines.append(msg % a if a else msg)

    info = warning = debug = error = _log

    def text(self):
        return "\n".join(self.lines)


class FakeHardware:
    def __init__(self, fail_disable=False):
        self.disabled = 0
        self.moves = []
        self.fail_disable = fail_disable

    def _disable_all_modality_illuminations(self):
        if self.fail_disable:
            raise RuntimeError("light path busy")
        self.disabled += 1

    def move_to_position(self, pos):
        self.moves.append(pos)


class FakePool:
    def __init__(self):
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1


class Ctx:
    def __init__(self, channels=None, hardware=None):
        self.write_pool = FakePool()
        self.tile_measurements_stream = None
        self.starting_position = None
        self.params = {"channels": channels or []}
        self.hardware = hardware or FakeHardware()
        self.logger = FakeLogger()


def cleanup(ctx):
    # The annotation on the def line is evaluated at exec time, so the type must exist.
    ns = {"Position": lambda **kw: dict(kw), "AcquisitionContext": object}
    load_function("_cleanup_acquisition", ns)(ctx)
    return ctx


def test_channel_acquisition_switches_illumination_off():
    ctx = cleanup(Ctx(channels=["DAPI", "FITC"]))
    assert ctx.hardware.disabled == 1
    assert "Deactivated modality illumination" in ctx.logger.text()


def test_angle_acquisition_is_left_alone():
    # PPM and other angle modalities have no channel teardown to do here.
    ctx = cleanup(Ctx(channels=[]))
    assert ctx.hardware.disabled == 0


def test_a_failed_deactivation_does_not_break_cleanup():
    ctx = Ctx(channels=["DAPI"], hardware=FakeHardware(fail_disable=True))
    cleanup(ctx)
    assert "Could not deactivate illumination" in ctx.logger.text()
    # The rest of teardown still ran.
    assert ctx.write_pool.shutdowns == 1


def test_teardown_still_returns_the_stage():
    ctx = Ctx(channels=["DAPI"])

    class Pos:
        x, y, z = 1.0, 2.0, 3.0

    ctx.starting_position = Pos()
    cleanup(ctx)
    assert ctx.hardware.moves == [{"x": 1.0, "y": 2.0}]
