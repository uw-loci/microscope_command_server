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
        "Optional": __import__("typing").Optional,
        "SURVEY_FIT_MIN_PLANE_POINTS": 5,
        "SURVEY_FIT_MIN_LINE_POINTS": 4,
        "SURVEY_FIT_MAX_RMS_UM": 2.0,
        "SURVEY_FIT_MAX_EXTRAPOLATION": 1.5,
    }
    exec(SOURCE[start:end], ns)
    return ns[name]


spread_survey_points = _load("_spread_survey_points", "def _run_focus_survey(")
survey_seed_z = _load("_survey_seed_z", "def _spread_survey_points(")


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

    def test_it_keeps_clear_of_the_region_edge(self):
        # Farthest-point sampling fills the extremes first, which is also where
        # autofocus is least trustworthy -- a boundary tile is the one most likely to be
        # half off the tissue, which is why the first AF position has been inset one
        # diagonal FOV since 2025-12-12. Replaying real regions measured the cost: a
        # 16-point farthest-point survey produced a worse point set than a 9-point one
        # on five regions, the extra picks all being edge.
        tiles = raster()
        arr = np.asarray(tiles)
        lo, hi = arr.min(axis=0), arr.max(axis=0)
        picked = np.asarray([tiles[i] for i in spread_survey_points(tiles, [], 9, 600.0)])
        assert (picked.min(axis=0) >= lo + 599).all()
        assert (picked.max(axis=0) <= hi - 599).all()

    def test_the_inset_is_abandoned_rather_than_returning_too_few(self):
        # A region barely wider than one FOV would inset to nothing. A smaller survey
        # is worse than an edge-heavy one, so the inset gives way.
        tiles = raster(n_cols=3, n_rows=3, pitch=300.0)
        picked = spread_survey_points(tiles, [], 6, 10000.0)
        assert len(picked) >= 5

    def test_the_inset_still_spreads_in_both_axes(self):
        tiles = raster()
        picked = np.asarray([tiles[i] for i in spread_survey_points(tiles, [], 9, 600.0)])
        assert np.ptp(picked[:, 0]) > 0
        assert np.ptp(picked[:, 1]) > 0

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

    def test_a_survey_with_autofocus_disabled_is_refused_not_crashed(self):
        # --af-disabled returns from _configure_autofocus before a strategy exists, so a
        # survey would hit an AttributeError on its first tissue check. The Java client
        # never sends both, but a direct socket client can.
        body = self.body()
        assert 'ctx.params.get("af_disabled") or ctx.af_strategy is None' in body
        assert "no focus to measure" in body

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

    def test_it_gets_the_objectives_edge_retry_budget(self):
        # The hardware default is 0 edge retries. Without the budget every survey point
        # whose focus lay outside one window around the seed failed: on the tilted
        # LC-PolScope slide of 2026-11-12, 5 of 9, each "after 0 retries".
        assert "edge_retries=ctx.af_edge_retries" in self.body()

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
        assert "prominence = _last_peak_prominence(hardware)" in body
        assert "surface.add(pos.x, pos.y, achieved, prominence)" in body

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


class TestSurveySeed:
    """A survey point starts from the tilt the points measured so far imply -- once that
    tilt has points to spare and they agree with it."""

    # LC-PolScope 2026-11-12, run 1: the plane its 54 autofocus results fit to 0.67 um RMS.
    @staticmethod
    def plane(x, y):
        return 73.76 - 5.84 * (x + 14488) / 1000 - 2.17 * (y - 2084) / 1000

    def measured(self, xys, noise=(0.4, -0.3, 0.5, -0.6, 0.2, -0.1)):
        return [(x, y, self.plane(x, y) + n) for (x, y), n in zip(xys, noise)]

    SPREAD = [(-22666, 3943), (-19188, 2949), (-20679, 2949), (-16000, 1500), (-13000, 3500)]

    def test_five_points_on_one_tilt_seed_a_far_point_near_its_focus(self):
        # Tile 66 started from 98.5 (the previous point) and accepted a false peak at
        # 98.6; the slide's plane puts it at about 58.
        z, used, rms = survey_seed_z(self.measured(self.SPREAD), -11737, 1956)
        assert used == 5 and rms < 1.0
        assert abs(z - self.plane(-11737, 1956)) < 1.5

    def test_three_points_are_not_enough_to_trust(self):
        # An exactly determined plane has no residual to check it against.
        assert survey_seed_z(self.measured(self.SPREAD[:3]), -11737, 1956) == (None, 0, None)

    def test_four_spread_points_are_not_enough_either(self):
        z, used, _ = survey_seed_z(self.measured(self.SPREAD[:4]), -11737, 1956)
        assert z is None and used == 0

    def test_a_false_peak_among_the_points_withdraws_the_fit(self):
        pts = self.measured(self.SPREAD)
        x, y, zz = pts[3]
        pts[3] = (x, y, zz + 40.0)  # 40 um off, like tile 66's false peak
        z, used, rms = survey_seed_z(pts, -11737, 1956)
        assert z is None and used == 0 and rms > 2.0

    def test_four_points_along_a_row_give_the_slope_along_it(self):
        row = [(0, 0, 10.0), (1000, 0, 16.1), (2000, 0, 21.9), (3000, 0, 28.0)]
        z, used, rms = survey_seed_z(row, 3500, 0)
        assert used == 4 and rms < 0.2
        assert z == pytest.approx(31.0, abs=0.3)

    def test_it_does_not_extrapolate_far_beyond_the_points(self):
        # Five points clustered in one corner say little about the far side of the slide.
        corner = [
            (-22000 + dx, 3500 + dy)
            for dx, dy in [(0, 0), (500, 0), (0, 500), (500, 500), (250, 250)]
        ]
        z, used, rms = survey_seed_z(self.measured(corner), -8000, 1500)
        assert z is None and used == 0

    def test_too_few_points_give_nothing(self):
        assert survey_seed_z([], 0, 0) == (None, 0, None)
        assert survey_seed_z([(0, 0, 1.0), (1, 1, 2.0), (2, 0, 3.0)], 0, 0) == (None, 0, None)

    def test_a_far_faint_peak_is_skipped_only_against_a_verified_fit(self):
        body = TestTheSurveyCannotHurtARun.body()
        assert "fit_points\n            and prominence is not None" in body
        assert "abs(achieved - seed_z) > far_um" in body
        assert body.index("skipping it as a false peak") < body.index(
            "ctx.completed_af_positions.append((pos.x, pos.y, achieved))"
        )
