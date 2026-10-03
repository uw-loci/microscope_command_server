"""The focus surface has to be wired into every path that records a focus point.

Source-shape tests, in the style of test_acquisition_teardown.py: exercising the real
tile loop needs a microscope, and the invariants worth protecting here are structural.
A focus point that reaches ctx.completed_af_positions without reaching the surface is
a point the surface cannot vet and cannot learn from -- and the whole mechanism exists
because an unvetted focus point put a region out of focus.
"""

import pathlib
import re

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "acquisition"
    / "workflow.py"
)
SOURCE = WORKFLOW.read_text()


def test_the_flag_is_parsed_and_defaults_to_off():
    assert '"--focus-surface"' in SOURCE
    assert 'params["focus_surface_mode"] = parts[i + 1]' in SOURCE
    # parse_mode owns the default, and it is off -- a client that sends nothing gets
    # exactly the old behaviour.
    assert 'parse_mode(ctx.params.get("focus_surface_mode")' in SOURCE


def test_every_recorded_focus_point_also_reaches_the_surface():
    appends = len(re.findall(r"completed_af_positions\.append\(", SOURCE))
    feeds = len(re.findall(r"focus_surface\.add\(", SOURCE))
    assert appends == 3, (
        "the bootstrap adopt-current path, the pre-acquisition AF, and the per-tile AF; "
        "a new one must feed the surface too"
    )
    assert feeds == appends, (
        "a focus point recorded without being given to the surface is one the surface "
        "can neither learn from nor vet"
    )


def test_results_are_vetted_before_they_are_recorded():
    # Order matters: vetting can substitute the surface's prediction, and what gets
    # recorded must be the Z the stage was actually left at.
    vet = SOURCE.index("af_z = _vet_autofocus_result(")
    record = SOURCE.index("ctx.completed_af_positions.append((pos.x, pos.y, af_z))")
    assert vet < record


def test_observe_mode_cannot_move_the_stage():
    # The whole value of observe mode is that it is inert. The only stage move in the
    # vetting helper sits after the enforced check.
    helper = SOURCE[SOURCE.index("def _vet_autofocus_result(") :]
    helper = helper[: helper.index("\ndef ")]
    assert "if not verdict.enforced:" in helper
    inert_return = helper.index("if not verdict.enforced:")
    move = helper.index("move_to_position(Position(z=verdict.predicted_z))")
    assert inert_return < move, "observe mode must return before any stage move"


def test_a_failed_override_move_keeps_the_measurement():
    # If the stage will not go to the prediction, the tile is wherever autofocus left
    # it, and that is the Z that must be recorded. Recording the prediction would put
    # a number in the surface that describes nowhere the stage has been.
    helper = SOURCE[SOURCE.index("def _vet_autofocus_result(") :]
    helper = helper[: helper.index("\ndef ")]
    assert "keeping the " in helper and "return af_z" in helper


def test_the_settings_line_states_the_mode():
    # Every past autofocus investigation started by reading this line. A mode that
    # changes which Z a tile uses has to be on it.
    assert "focus_surface={params.get('focus_surface_mode') or 'off'}" in SOURCE


def test_the_surface_is_rebuilt_per_region_including_when_af_is_disabled():
    # Two regions on one slide sit at different heights; carrying a surface across
    # them would average them. And --af-disabled returns early, so it needs its own
    # rebuild or it would inherit the previous region's plane.
    builds = len(re.findall(r"ctx\.focus_surface = _build_focus_surface\(ctx\)", SOURCE))
    assert builds == 2
