"""The pre-scan focus survey: point selection, and the ways it must not matter.

Two halves. The spread selector is real logic with a right answer, so it is
exec-extracted and exercised directly. The survey runner needs a microscope, a
strategy and a socket, so its invariants are asserted against the source -- and those
invariants are all about the survey being unable to hurt a run: every failure mode
skips a point rather than stopping, and it never prompts a human it cannot reach.
"""

import io
import pathlib
import re

import numpy as np
import pytest
from scipy.spatial.distance import cdist

SRC_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "acquisition"
    / "workflow.py"
)
SOURCE = io.open(SRC_PATH, encoding="utf-8").read()


def _load(name, after):
    """Exec one top-level function out of workflow.py, which cannot be imported here."""
    start = SOURCE.index(f"def {name}(")
    end = SOURCE.index(after, start)
    ns = {
        "np": np,
        "_cdist_scipy": cdist,
        "List": list,
        "Tuple": tuple,
        "AcquisitionContext": object,
    }
    exec(SOURCE[start:end], ns)
    return ns[name]


spread_survey_points = _load("_spread_survey_points", "def _run_focus_survey(")


def raster(n_cols=12, n_rows=10, pitch=300.0):
    """Tile centres in serpentine scan order, like TilingUtilities produces."""
    out = []
    for c in range(n_cols):
        rows = range(n_rows) if c % 2 == 0 else reversed(range(n_rows))
        for r in rows:
            out.append((c * pitch, r * pitch))
    return out


class TestSpreadSelection:
    def test_it_spreads_in_both_axes_unlike_scan_order(self):
        # The point of the selector. The first 9 tiles in scan order are one column:
        # they constrain the tilt along the column and say nothing across it, and the
        # focus surface refuses such a set outright.
        tiles = raster()
        scan_order = np.asarray(tiles[:9])
        picked = np.asarray([tiles[i] for i in spread_survey_points(tiles, [], 9)])
        assert np.ptp(scan_order[:, 0]) == 0, "scan order really is a single column"
        assert np.ptp(picked[:, 0]) > 0.8 * np.ptp(np.asarray(tiles)[:, 0])
        assert np.ptp(picked[:, 1]) > 0.8 * np.ptp(np.asarray(tiles)[:, 1])

    def test_the_first_pick_is_a_corner_not_the_middle(self):
        tiles = raster()
        first = tiles[spread_survey_points(tiles, [], 1)[0]]
        arr = np.asarray(tiles)
        centre = arr.mean(axis=0)
        corner_dist = np.abs(arr - centre).sum(axis=1).max()
        assert np.abs(np.asarray(first) - centre).sum() == pytest.approx(corner_dist, rel=0.01)

    def test_it_avoids_what_has_already_been_measured(self):
        # Phase 8 has already measured one point. The survey must spend its budget
        # somewhere else, not next door to it.
        tiles = raster()
        already = [tiles[0]]
        picked = [tiles[i] for i in spread_survey_points(tiles, already, 4)]
        for p in picked:
            assert np.hypot(p[0] - already[0][0], p[1] - already[0][1]) > 1000.0

    def test_picks_are_distinct(self):
        tiles = raster()
        picked = spread_survey_points(tiles, [], 12)
        assert len(picked) == len(set(picked))

    def test_it_cannot_ask_for_more_tiles_than_exist(self):
        tiles = raster(n_cols=2, n_rows=2)
        assert len(spread_survey_points(tiles, [], 25)) <= len(tiles)

    def test_degenerate_inputs_return_nothing_rather_than_raising(self):
        assert spread_survey_points([], [], 5) == []
        assert spread_survey_points(raster(), [], 0) == []


class TestTheSurveyCannotHurtARun:
    @staticmethod
    def body():
        start = SOURCE.index("def _run_focus_survey(")
        return SOURCE[start : SOURCE.index("\ndef ", start + 10)]

    def test_absent_flag_returns_immediately(self):
        # The default must be byte-identical behaviour to before.
        body = self.body()
        assert 'ctx.params.get("focus_survey_points") or 0' in body
        assert "if requested <= 0:\n        return" in body

    def test_a_survey_without_a_surface_is_refused_not_wasted(self):
        body = self.body()
        assert "if surface is None or not surface.active:" in body
        assert "nothing to fit" in body

    def test_it_never_prompts_for_manual_focus(self):
        # The survey's whole purpose is unattended time saving. A manual-focus dialog
        # in the middle of it would block an overnight batch on a point that could
        # simply have been skipped.
        body = self.body()
        assert "request_manual_focus=None" in body
        assert "max_retries=0" in body

    @pytest.mark.parametrize(
        "failure",
        ["could not move to tile", "tissue check failed", "autofocus failed at tile"],
    )
    def test_every_failure_skips_a_point_rather_than_stopping(self, failure):
        body = self.body()
        assert failure in body
        # Each failure path continues the loop.
        idx = body.index(failure)
        assert "continue" in body[idx : idx + 400]

    def test_a_point_with_no_tissue_is_skipped(self):
        # Autofocus on blank glass still finds a peak -- on the coverslip. A survey
        # point there would define the surface at the wrong height.
        body = self.body()
        assert "is_valid(" in body
        assert "no usable signal" in body

    def test_a_measured_point_feeds_both_the_surface_and_the_seed_list(self):
        # The existing nearest-neighbour path must benefit too, or a survey would
        # improve the surface and leave the fallback worse than it needed to be.
        body = self.body()
        assert "ctx.completed_af_positions.append((pos.x, pos.y, achieved))" in body
        assert "surface.add(pos.x, pos.y, achieved)" in body

    def test_it_records_the_achieved_z_not_the_returned_one(self):
        # autofocus_with_manual_fallback can return a skip-Z that the stage was never
        # driven to. The surface must only ever contain places the stage has been.
        body = self.body()
        assert "achieved = hardware.get_current_position().z" in body

    def test_an_unusable_surface_says_so_and_calls_it_safe(self):
        body = self.body()
        assert "if not surface.licensed:" in body
        assert "exactly as it does today" in body

    def test_it_is_cancellable(self):
        body = self.body()
        assert "ctx.is_cancelled" in body

    def test_it_runs_after_the_pre_acquisition_autofocus(self):
        # Phase 8 sets the AF rotation angle and exposure and measures the first
        # point. Running the survey first would duplicate all of it.
        pre = SOURCE.index("_run_pre_acquisition_autofocus(ctx)")
        survey = SOURCE.index("_run_focus_survey(ctx)")
        assert pre < survey

    def test_the_flags_are_parsed(self):
        assert '"--focus-survey"' in SOURCE
        assert '"--focus-survey-tiles"' in SOURCE
        assert re.search(r'params\["focus_survey_points"\] = int\(parts\[i \+ 1\]\)', SOURCE)
