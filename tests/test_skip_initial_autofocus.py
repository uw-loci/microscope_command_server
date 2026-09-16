"""--af-skip-initial must leave the same state the real search would have left.

The pre-acquisition autofocus is not just a stage move -- it seeds four fields that the
per-tile dispatcher then reads. If the skip path sets a different set, the loop does not
fail, it silently mis-seeds: `completed_af_positions` feeds the nearest-focused-tile seed,
and `dynamic_af_positions` decides whether the first AF position gets autofocused a second
time. So this pins the two paths to the same contract rather than testing the skip alone.

Scope matters too. This flag is NOT --af-disabled: per-tile AF, drift checks and the
manual fallback all still run, and the test asserts the flag does not touch the sets that
would turn those off.
"""

import io
import re

import pytest

SRC = "microscope_command_server/acquisition/workflow.py"


class _Pos:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Hardware:
    def __init__(self, pos):
        self._pos = pos

    def get_current_position(self):
        return self._pos


class _Ctx:
    """Only the fields the two AF paths touch."""

    def __init__(self, pos, af_positions):
        import logging

        self.logger = logging.getLogger("test")
        self.hardware = _Hardware(pos)
        self.af_positions = af_positions
        self.first_tissue_autofocus_done = False
        self.last_af_pos_idx = -1
        self.completed_af_positions = []
        self.dynamic_af_positions = set(af_positions)
        self.deferred_af_positions = set()


def _load_skip_fn():
    src = io.open(SRC, encoding="utf-8").read()
    start = src.index("def _adopt_current_focus_as_initial(")
    end = src.index("def _run_pre_acquisition_autofocus(", start)
    ns = {"AcquisitionContext": object}
    exec(src[start:end], ns)
    return ns["_adopt_current_focus_as_initial"]


def test_skip_sets_the_same_four_fields_the_real_search_sets():
    ctx = _Ctx(_Pos(1234.0, -5678.0, -331.5), [7, 19, 42])
    _load_skip_fn()(ctx)

    assert ctx.first_tissue_autofocus_done is True
    assert ctx.last_af_pos_idx == 7, "must claim the first AF position as done"
    assert 7 not in ctx.dynamic_af_positions, "else the first tile autofocuses anyway"
    # The recorded point is where the OPERATOR focused, not the diagonal AF position.
    assert ctx.completed_af_positions == [(1234.0, -5678.0, -331.5)]


def test_skip_does_not_disable_autofocus_for_the_remaining_positions():
    """The whole point of the flag: drift correction survives."""
    ctx = _Ctx(_Pos(0.0, 0.0, -100.0), [3, 11, 25])
    _load_skip_fn()(ctx)

    assert ctx.dynamic_af_positions == {11, 25}, "per-tile AF must still be scheduled"
    assert ctx.deferred_af_positions == set()


def test_the_real_path_sets_exactly_these_fields_and_no_others():
    """Guards the contract above against the real function growing a fifth field."""
    src = io.open(SRC, encoding="utf-8").read()
    start = src.index("def _run_pre_acquisition_autofocus(")
    end = src.index("def _handle_tile_autofocus(", start)
    body = src[start:end]
    tail = body[body.index("Initial autofocus completed") :]
    assigned = set(re.findall(r"ctx\.([a-z_]+)(?:\.append|\.discard| =)", tail))
    assert assigned == {
        "first_tissue_autofocus_done",
        "last_af_pos_idx",
        "completed_af_positions",
        "dynamic_af_positions",
    }, f"real AF path now sets {assigned}; mirror the change in the skip path"


def test_the_flag_is_parsed_off_the_wire():
    src = io.open(SRC, encoding="utf-8").read()
    assert '"--af-skip-initial"' in src
    assert 'params["af_skip_initial"] = True' in src
    assert 'params.get("af_skip_initial")' in src
