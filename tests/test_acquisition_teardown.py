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
        self.region_origin = None
        self.batch_acquire = False
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


class Pos:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def test_batch_run_parks_on_its_own_region_not_the_inherited_start():
    """A batch inherits its "starting position" from whichever slide ran before it.

    On the 4-slide PPM run of 2026-09-17 every acquisition ended with the same move,
    to a point on slide 4, including the one that finished slide 3. Parking on this
    region's own first tile keeps the stage on the slide it just acquired.
    """
    ctx = Ctx(channels=[])
    ctx.batch_acquire = True
    ctx.starting_position = Pos(-50564.0, 3523.0)  # a point on another slide
    ctx.region_origin = (-17448.0, 1025.0)  # this region's first tile
    cleanup(ctx)
    assert ctx.hardware.moves == [{"x": -17448.0, "y": 1025.0}]
    assert "this region's first tile" in ctx.logger.text()
    # The coordinates must be in the log; their absence is what made the original
    # incident unreadable.
    assert "-17448.0" in ctx.logger.text()


def test_single_slide_still_returns_to_the_operators_position():
    # The whole point of the flag is that this path is untouched.
    ctx = Ctx(channels=[])
    ctx.batch_acquire = False
    ctx.starting_position = Pos(-50564.0, 3523.0)
    ctx.region_origin = (-17448.0, 1025.0)
    cleanup(ctx)
    assert ctx.hardware.moves == [{"x": -50564.0, "y": 3523.0}]
    assert "starting XY" in ctx.logger.text()


def test_batch_run_without_a_region_origin_falls_back_to_the_start():
    # Defensive: an empty tile list leaves region_origin None. Better to park
    # somewhere known than to skip the move.
    ctx = Ctx(channels=[])
    ctx.batch_acquire = True
    ctx.starting_position = Pos(5.0, 6.0)
    ctx.region_origin = None
    cleanup(ctx)
    assert ctx.hardware.moves == [{"x": 5.0, "y": 6.0}]
