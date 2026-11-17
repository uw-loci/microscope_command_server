"""The focus surface has to be right about when it is allowed to speak.

Every test here is about a refusal or an override, not about arithmetic. A surface
that predicts well but cannot tell when it is out of its depth is worse than no
surface, because it is confident and silent.

Numbers come from the measured corpus in
claude-reports/design/autofocus-empirical/summaries/focus_surface.md: ten PPM regions,
plane RMS 0.27-1.11 um, tilt 1.2-10.8 um/mm, wide-search autofocus landing >5 um off
the plane 25% of the time and up to 73 um off.
"""

import importlib.util
import logging
import pathlib
import sys

import numpy as np
import pytest

# Loaded by path, not as microscope_command_server.acquisition.focus_surface: that
# package's __init__ imports the hardware layer, which a bare dev checkout does not
# have. Keeping the import direct also pins a property worth keeping -- the fit
# depends on nothing but numpy, so it stays testable on any machine.
_SPEC = importlib.util.spec_from_file_location(
    "focus_surface",
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "acquisition"
    / "focus_surface.py",
)
focus_surface = importlib.util.module_from_spec(_SPEC)
# dataclasses resolves annotations through sys.modules, so register before exec.
sys.modules["focus_surface"] = focus_surface
_SPEC.loader.exec_module(focus_surface)

FocusSurface = focus_surface.FocusSurface
parse_mode = focus_surface.parse_mode
MODE_OFF = focus_surface.MODE_OFF
MODE_OBSERVE = focus_surface.MODE_OBSERVE
MODE_ENFORCE = focus_surface.MODE_ENFORCE

LOGGER = logging.getLogger("test")

# A real region: 14 mm across, 2.86 um/mm in X and 1.02 um/mm in Y, z0 -304.4 um.
# (2026-09-24 session, region at stage (35981, -7908).)
TILT_X = 2.86
TILT_Y = 1.02
Z0 = -304.4
CENTRE = (35981.0, -7908.0)


def real_plane(x, y):
    return Z0 + TILT_X * (x - CENTRE[0]) / 1000.0 + TILT_Y * (y - CENTRE[1]) / 1000.0


def tile_grid(n_side=6, pitch_um=2200.0):
    xs = CENTRE[0] + (np.arange(n_side) - n_side / 2) * pitch_um
    ys = CENTRE[1] + (np.arange(n_side) - n_side / 2) * pitch_um
    return [(float(x), float(y)) for x in xs for y in ys]


def surface(mode=MODE_ENFORCE, **kw):
    return FocusSurface(mode=mode, logger=LOGGER, **kw)


def feed(s, points, noise_um=0.5, seed=0):
    rng = np.random.default_rng(seed)
    for x, y in points:
        s.add(x, y, real_plane(x, y) + rng.normal(0, noise_um))
    return s


class TestWhenItRefusesToSpeak:
    def test_three_points_are_not_enough(self):
        # A plane has three parameters. Three points fit it exactly, the residual is
        # zero by construction, and one bad point moves the surface leaving no trace.
        # This is the same trap as a two-correspondence similarity transform.
        s = feed(surface(), tile_grid()[:3])
        assert s.predict(*CENTRE) is None
        assert not s.licensed

    def test_six_points_down_one_raster_column_are_still_a_line(self):
        # tile_grid() is built x-outer, y-inner, so the first six points in scan order
        # share an X. They fix the tilt ALONG the column and say nothing about the tilt
        # across it -- and least squares will still return a number and extrapolate it
        # over the whole slide. Measured here before the spread guard existed: it
        # predicted -323 um where the plane is -304 um, 19 um wrong, with every other
        # check passing. Scan order makes this the first thing a region's surface sees.
        s = feed(surface(), tile_grid()[:6])
        assert not s.licensed
        assert s.predict(*CENTRE) is None
        assert "do not constrain a tilt" in s.check(CENTRE[0], CENTRE[1], Z0).reason

    def test_it_speaks_once_the_points_spread_in_both_axes(self):
        # Two full columns, so there are distinct X values AND distinct Y values.
        # A diagonal selection would have been just as collinear as one column.
        spread = tile_grid(n_side=3, pitch_um=3000.0)[:6]
        s = feed(surface(), spread)
        assert s.licensed, s.describe()
        assert s.predict(*CENTRE) == pytest.approx(Z0, abs=1.5)

    def test_a_non_planar_sample_is_declined(self):
        # A folded or detached section, or a cytology smear. The surface must hand the
        # caller back to nearest-neighbour behaviour rather than average the fold away.
        #
        # The trap this caught: RANSAC over a scattered set always finds SOME subset
        # inside the inlier band, and the inlier RMS over that subset looks excellent.
        # Judging planarity on RMS alone licensed a surface fitted to a third of the
        # data with the confidence of a fit to all of it. What actually separates a
        # planar sample is the FRACTION that agrees -- 91-100% on every real region.
        s = surface()
        rng = np.random.default_rng(1)
        for x, y in tile_grid():
            s.add(x, y, real_plane(x, y) + rng.normal(0, 8.0))
        assert not s.licensed, s.describe()
        assert s.predict(*CENTRE) is None
        verdict = s.check(CENTRE[0], CENTRE[1], Z0)
        assert not verdict.would_reject
        assert "agree with the surface" in verdict.reason

    def test_off_mode_learns_nothing_and_says_nothing(self):
        s = feed(surface(MODE_OFF), tile_grid())
        assert s.predict(*CENTRE) is None
        assert not s.licensed
        assert s.summary()["points"] == 0


class TestCatchingTheWideSearchMisses:
    @pytest.mark.parametrize("miss_um", [21.0, 43.0, 73.0])
    def test_a_real_wide_search_miss_is_rejected(self, miss_um):
        # 21, 43 and 73 um are all measured standard-autofocus residuals from the
        # corpus; 73 um is the worst observed.
        s = feed(surface(), tile_grid())
        x, y = CENTRE
        verdict = s.check(x, y, real_plane(x, y) + miss_um)
        assert verdict.would_reject
        assert verdict.enforced
        assert verdict.residual_um == pytest.approx(miss_um, abs=1.5)

    def test_a_normal_sweep_result_is_accepted(self):
        # Narrow sweeps sit a median 0.40 um off the plane. None of that may be rejected.
        s = feed(surface(), tile_grid())
        rng = np.random.default_rng(7)
        for x, y in tile_grid(n_side=5, pitch_um=2600.0):
            z = real_plane(x, y) + rng.normal(0, 0.5)
            assert not s.check(x, y, z).would_reject

    def test_an_outlier_does_not_drag_the_surface(self):
        # The whole point of a robust fit: today a 70 um miss enters the seed list and
        # is handed to every tile near it.
        s = feed(surface(), tile_grid())
        clean = s.predict(*CENTRE)
        s.add(CENTRE[0], CENTRE[1], real_plane(*CENTRE) + 70.0)
        assert s.predict(*CENTRE) == pytest.approx(clean, abs=0.5)

    def test_the_gate_widens_with_a_noisier_fit(self):
        # A surface that fits poorly should be less willing to overrule a measurement.
        # The 5 um floor would mask the scaling at realistic noise levels (4 x 1.14 um
        # is still under it), so drop the floor here to isolate the sigma term.
        tight = feed(surface(reject_margin_um=0.5), tile_grid(), noise_um=0.2, seed=3)
        loose = feed(surface(reject_margin_um=0.5), tile_grid(), noise_um=2.0, seed=3)
        x, y = CENTRE
        assert (
            loose.check(x, y, real_plane(x, y) + 9.0).gate_um
            > tight.check(x, y, real_plane(x, y) + 9.0).gate_um
        )


class TestObserveChangesNothing:
    def test_observe_reports_but_does_not_enforce(self):
        s = feed(surface(MODE_OBSERVE), tile_grid())
        x, y = CENTRE
        verdict = s.check(x, y, real_plane(x, y) + 43.0)
        assert verdict.would_reject, "it must still say what it thinks"
        assert not verdict.enforced, "but nothing may act on it in observe mode"
        assert not s.enforcing

    def test_observe_and_enforce_learn_the_same_surface(self):
        # A dry run has to measure the surface enforcement would have used, or it proves
        # nothing. Driven through check()-then-add the way the acquisition does, and
        # including a result both modes would reject, so the two can only agree if the
        # rejected point is withheld from BOTH.
        points = tile_grid()
        obs = feed(surface(MODE_OBSERVE), points, seed=11)
        enf = feed(surface(MODE_ENFORCE), points, seed=11)
        for s in (obs, enf):
            x, y = CENTRE
            bad = real_plane(x, y) + 43.0
            verdict = s.check(x, y, bad)
            assert verdict.would_reject
            # _vet_autofocus_result returns learnable=False for this in either mode.
            if not verdict.would_reject:
                s.add(x, y, bad)
        assert obs.predict(*CENTRE) == pytest.approx(enf.predict(*CENTRE), abs=1e-9)
        assert obs.summary()["would_reject"] == enf.summary()["would_reject"]


class TestGatesThatRealLogsCorrected:
    """Three defects that only a replay against real runs could show.

    `focus_surface_replay.py` feeds the shipped class every region in every available
    server log, in acquisition order. Each of these was invisible to synthetic tests
    and obvious within one replay.
    """

    def test_more_points_can_never_license_less(self):
        # The spread measure divided by the point count, so adding points to a thin
        # cloud SHRANK the measured spread. Measured consequence: a 16-point survey was
        # refused on five regions where the same region's 9-point survey was licensed.
        tiles = tile_grid(n_side=4, pitch_um=600.0)
        small = feed(surface(), tiles[:9], seed=2)
        large = feed(surface(), tiles, seed=2)
        assert small.licensed
        assert large.licensed, large.describe()
        assert large.summary()["spread_minor_mm"] >= small.summary()["spread_minor_mm"]

    def test_a_thin_region_is_judged_against_its_own_width(self):
        # An absolute 0.5 mm floor refused five real regions at 0.38-0.49 mm -- thin
        # strips whose fits were excellent (0.22-0.41 um RMS, every point an inlier).
        # An unconstrained across-tilt can only do harm over the distance it is
        # extrapolated, and in a strip that is almost nothing.
        strip = []
        for i in range(12):
            strip.append((CENTRE[0] + i * 400.0, CENTRE[1]))
            strip.append((CENTRE[0] + i * 400.0, CENTRE[1] + 400.0))
        without = feed(surface(), strip, seed=4)
        assert not without.licensed, "0.4 mm of spread fails the absolute floor"
        aware = feed(surface(region_extent_mm=(4.4, 0.4)), strip, seed=4)
        assert aware.licensed, aware.describe()
        assert aware.required_spread_mm() < aware.min_spread_mm

    def test_three_identical_z_values_do_not_reset_a_good_surface(self):
        # The counter-example from the replay: three rejected results with a Z spread of
        # exactly 0.00 um reset a surface that 108 points defined to 0.27 um RMS. On a
        # tilted slide a MOVED sample's focus still changes across the field, so
        # identical Z at different XY is a stuck reading, not a translation -- and a
        # stuck reading is the one thing that must not dislodge a good surface.
        s = feed(surface(), tile_grid())
        stuck = -250.0
        for x, y in tile_grid(n_side=2, pitch_um=3000.0)[:3]:
            s.check(x, y, stuck)
        assert s.summary()["resurfaces"] == 0, s.describe()
        assert s.licensed

    def test_a_genuinely_translated_sample_still_resets_it(self):
        # The flip side: points that sit on this surface's own tilt at a new height are
        # what a re-seated slide looks like, and must still be conceded to.
        s = feed(surface(), tile_grid())
        shift = 60.0
        for x, y in tile_grid(n_side=2, pitch_um=3000.0)[:3]:
            s.check(x, y, real_plane(x, y) + shift)
        assert s.summary()["resurfaces"] == 1, s.describe()

    def test_the_summary_always_says_why_it_is_not_licensed(self):
        # A region with one usable measurement in ninety tiles reported only
        # "licensed: false", which says nothing about what to do next.
        s = surface()
        s.add(CENTRE[0], CENTRE[1], Z0)
        assert s.summary()["checking_reason"]
        assert "not enough focus points" in s.summary()["checking_reason"]
        assert s.summary()["seeding_reason"] == s.summary()["checking_reason"]


class TestConcedingToReality:
    def test_a_moved_sample_resets_the_surface(self):
        # The failure that matters most: the surface is confident and the sample has
        # moved. Three consecutive rejections that agree with EACH OTHER mean the
        # sample, not the autofocus, is where the surface is not.
        s = feed(surface(), tile_grid())
        shift = 60.0
        pts = tile_grid(n_side=2, pitch_um=2000.0)[:3]
        verdicts = [s.check(x, y, real_plane(x, y) + shift) for x, y in pts]
        assert verdicts[0].would_reject
        assert verdicts[-1].would_reject is False, "the third concedes"
        assert s.summary()["resurfaces"] == 1

    def test_scattered_failures_do_not_reset_it(self):
        # Three failed autofocus attempts in a row are not a moved sample. They
        # disagree with each other, so the surface must hold.
        s = feed(surface(), tile_grid())
        x, y = CENTRE
        for offset in (40.0, -55.0, 25.0):
            s.check(x, y, real_plane(x, y) + offset)
        assert s.summary()["resurfaces"] == 0
        assert s.licensed


class TestPlumbing:
    def test_unknown_mode_falls_back_to_off(self):
        assert parse_mode("enfroce", LOGGER) == MODE_OFF
        assert parse_mode(None) == MODE_OFF
        assert parse_mode("ENFORCE") == MODE_ENFORCE

    def test_describe_names_the_tilt_it_found(self):
        s = feed(surface(), tile_grid())
        text = s.describe()
        assert "tilt" in text and "RMS" in text and "mode=enforce" in text

    def test_summary_carries_the_fit(self):
        s = feed(surface(), tile_grid())
        summary = s.summary()
        assert summary["tilt_x_um_per_mm"] == pytest.approx(TILT_X, abs=0.3)
        assert summary["tilt_y_um_per_mm"] == pytest.approx(TILT_Y, abs=0.3)
        assert summary["inlier_rms_um"] < 1.0

    def test_a_single_row_annotation_declines_rather_than_inventing_a_tilt(self):
        # A one-tile-high strip is a real annotation shape. Every tile shares a Y, so
        # there is no evidence about the Y tilt at all. Declining costs a fallback to
        # nearest-neighbour; guessing costs an out-of-focus strip.
        s = surface()
        for i in range(8):
            x = CENTRE[0] + i * 300.0
            s.add(x, CENTRE[1], real_plane(x, CENTRE[1]))
        assert not s.licensed
        assert s.predict(CENTRE[0] + 1000.0, CENTRE[1]) is None

    def test_a_long_thin_region_is_still_usable(self):
        # Not every narrow region is degenerate: a few rows of tiles spread 0.9 mm in
        # the short axis is enough to constrain both tilts.
        s = surface()
        rng = np.random.default_rng(5)
        for i in range(10):
            for j in range(4):
                x = CENTRE[0] + i * 1200.0
                y = CENTRE[1] + j * 600.0
                s.add(x, y, real_plane(x, y) + rng.normal(0, 0.4))
        assert s.licensed, s.describe()


# A twisted slide, as measured on PPM 20x on 2026-11-07 (1208 autofocus results over
# 19 x 17 mm): plane terms 5.67 / -2.65 um/mm and an X*Y term of -0.094 um/mm^2.
TWIST = {"z0": -51.9, "ax": 5.67, "ay": -2.65, "qxx": -0.011, "qxy": -0.094, "qyy": -0.012}
TWIST_CENTRE = (6835.0, -5443.0)


def twisted(x, y):
    u = (x - TWIST_CENTRE[0]) / 1000.0
    v = (y - TWIST_CENTRE[1]) / 1000.0
    t = TWIST
    return (
        t["z0"] + t["ax"] * u + t["ay"] * v + t["qxx"] * u * u + t["qxy"] * u * v + t["qyy"] * v * v
    )


def twisted_grid(n_side=12, extent_um=18000.0):
    offs = (np.arange(n_side) / (n_side - 1) - 0.5) * extent_um
    return [(TWIST_CENTRE[0] + dx, TWIST_CENTRE[1] + dy) for dx in offs for dy in offs]


def feed_fn(s, points, fn, noise_um=0.5, seed=0):
    rng = np.random.default_rng(seed)
    for x, y in points:
        s.add(x, y, fn(x, y) + rng.normal(0, noise_um))
    return s


class TestCurvedSurface:
    def test_a_twisted_slide_gets_a_quadratic(self):
        s = feed_fn(surface(), twisted_grid(), twisted)
        assert s.summary()["model"] == "quadratic", s.describe()
        assert s.licensed
        # Every point agrees with it; the plane had dropped the row ends.
        assert s.summary()["inliers"] >= 0.97 * len(twisted_grid())

    def test_a_row_end_result_is_not_vetoed_on_a_twisted_slide(self):
        # 2026-11-07: the plane was about 5 um off at the row ends and enforcement replaced
        # correct results there. The true focus at a corner must agree with the surface.
        s = feed_fn(surface(), twisted_grid(), twisted)
        x, y = TWIST_CENTRE[0] + 9000.0, TWIST_CENTRE[1] + 9000.0
        v = s.check(x, y, twisted(x, y))
        assert not v.would_reject, v.reason

    def test_a_flat_slide_keeps_its_plane(self):
        grid = [
            (CENTRE[0] + dx, CENTRE[1] + dy)
            for dx in np.linspace(-7000, 7000, 8)
            for dy in np.linspace(-7000, 7000, 8)
        ]
        s = feed_fn(surface(), grid, real_plane)
        assert s.summary()["model"] == "plane", s.describe()

    def test_too_few_points_never_curve(self):
        pts = twisted_grid(n_side=4)  # 16 points, below QUAD_MIN_POINTS
        s = feed_fn(surface(), pts, twisted)
        assert s.summary()["model"] == "plane"

    def test_a_narrow_strip_never_curves(self):
        # 2 mm wide: curvature across it is not measurable.
        pts = [
            (TWIST_CENTRE[0] + dx, TWIST_CENTRE[1] + dy)
            for dx in np.linspace(-9000, 9000, 15)
            for dy in (-1000.0, 0.0, 1000.0)
        ]
        s = feed_fn(surface(), pts, twisted)
        assert s.summary()["model"] == "plane"

    def test_beyond_its_points_the_curve_continues_straight(self):
        # Fit only the left half; ask 10 mm past its right edge. The prediction must follow
        # the slope at the edge, not the parabola.
        left = [p for p in twisted_grid() if p[0] <= TWIST_CENTRE[0]]
        s = feed_fn(surface(), left, twisted, noise_um=0.2)
        fit = s._fit
        assert fit.quadratic
        xmax = fit.bounds[1]
        y = TWIST_CENTRE[1]
        edge = fit.predict(xmax, y)
        slope = (fit.predict(xmax, y) - fit.predict(xmax - 100.0, y)) / 0.1
        far = fit.predict(xmax + 10000.0, y)
        assert far == pytest.approx(edge + slope * 10.0, abs=0.05)

    def test_the_record_carries_the_curve(self):
        s = feed_fn(surface(), twisted_grid(), twisted)
        r = s.to_record()
        assert r["model"] == "quadratic" and r["licensed"]
        assert r["qxy_um_per_mm2"] == pytest.approx(-0.094, abs=0.01)
        assert len(r["bounds_um"]) == 4

    def test_no_record_when_off_or_unfitted(self):
        assert surface(MODE_OFF).to_record() is None
        assert surface().to_record() is None


# The 2026-11-10 slot-4 region (PPM 20x): eight clear standard-autofocus peaks
# (prominence 0.96-0.98) lie on z = -190.0 - 7.94*X_mm - 0.60*Y_mm to 0.79 um RMS. The
# survey and most standard results found faint peaks (median 0.48) about 120 um lower,
# and the surface licensed on those replaced all eight clear ones.
def slot4_tissue(x, y):
    return -190.0 - 7.94 * x / 1000.0 - 0.60 * y / 1000.0


def slot4_grid():
    xs = np.linspace(-55000.0, -42000.0, 5)
    ys = np.linspace(-31000.0, -26000.0, 4)
    return [(float(x), float(y)) for x in xs for y in ys]


def feed_quality(s, points, fn, qualities, noise_um=0.5, seed=0):
    rng = np.random.default_rng(seed)
    for (x, y), q in zip(points, qualities):
        s.add(x, y, fn(x, y) + rng.normal(0, noise_um), q)


class TestClearerEvidenceWins:
    STRONG = [(-49045.7, -30708.7), (-44213.1, -30215.8), (-45503.7, -29495.5)]

    def faint_surface(self, mode=MODE_ENFORCE):
        s = surface(mode=mode)
        rng = np.random.default_rng(3)
        feed_quality(
            s,
            slot4_grid(),
            lambda x, y: slot4_tissue(x, y) - 120.0,
            rng.uniform(0.37, 0.62, size=20),
        )
        assert s.licensed
        return s

    def test_two_agreeing_clear_peaks_reset_a_surface_built_from_faint_ones(self):
        s = self.faint_surface()
        (x1, y1), (x2, y2) = self.STRONG[:2]
        first = s.check(x1, y1, slot4_tissue(x1, y1), quality=0.97)
        assert first.would_reject and first.enforced
        assert "prominence 0.97" in first.reason

        second = s.check(x2, y2, slot4_tissue(x2, y2), quality=0.98)
        assert not second.would_reject
        assert "reset" in second.reason
        assert s.summary()["clearer_peak_resets"] == 1
        # The new surface starts from the first; the caller then learns the second.
        assert s.summary()["points"] == 1
        s.add(x2, y2, slot4_tissue(x2, y2), 0.98)
        assert s.summary()["points"] == 2

    def test_one_clear_peak_alone_does_not_reset_it(self):
        s = self.faint_surface()
        x, y = self.STRONG[0]
        v = s.check(x, y, slot4_tissue(x, y), quality=0.97)
        assert v.would_reject
        assert s.summary()["clearer_peak_resets"] == 0

    def test_clear_peaks_that_disagree_with_each_other_do_not_reset_it(self):
        # 2026-11-10: a 0.59 peak at a third height (145 um) followed the 0.45 start.
        s = self.faint_surface()
        (x1, y1), (x2, y2) = self.STRONG[:2]
        s.check(x1, y1, slot4_tissue(x1, y1) - 60.0, quality=0.97)
        v = s.check(x2, y2, slot4_tissue(x2, y2), quality=0.98)
        assert v.would_reject
        assert s.summary()["clearer_peak_resets"] == 0

    def test_the_same_tile_twice_is_not_two_results(self):
        s = self.faint_surface()
        x, y = self.STRONG[0]
        s.check(x, y, slot4_tissue(x, y), quality=0.97)
        v = s.check(x, y, slot4_tissue(x, y), quality=0.97)
        assert v.would_reject

    def test_faint_peaks_are_still_overruled_by_a_faint_surface(self):
        s = self.faint_surface()
        # Two results: a third consistent one would trip the moved-sample reset.
        for x, y in self.STRONG[:2]:
            v = s.check(x, y, slot4_tissue(x, y), quality=0.30)
            assert v.would_reject and v.enforced

    def test_clear_misses_are_still_overruled_by_a_clear_surface(self):
        # The protection the surface exists for: wide searches that land far off a
        # surface whose own peaks are at least as clear.
        s = surface()
        rng = np.random.default_rng(4)
        feed_quality(s, slot4_grid(), slot4_tissue, rng.uniform(0.95, 0.99, size=20))
        # Two results: a third consistent one would trip the moved-sample reset.
        for x, y in self.STRONG[:2]:
            v = s.check(x, y, slot4_tissue(x, y) - 120.0, quality=0.97)
            assert v.would_reject and v.enforced

    def test_a_surface_built_from_sweeps_is_not_weighed_by_prominence(self):
        # Drift sweeps carry no prominence, so standard results cannot claim to be
        # clearer than them: wide-search misses against a sweep surface (one region
        # had 12 of 12 a median 43 um off) must still be caught.
        s = surface()
        feed_quality(s, slot4_grid(), slot4_tissue, [None] * 20)
        # Two results: a third consistent one would trip the moved-sample reset.
        for x, y in self.STRONG[:2]:
            v = s.check(x, y, slot4_tissue(x, y) + 43.0, quality=0.99)
            assert v.would_reject and v.enforced

    def test_results_without_prominence_are_judged_as_before(self):
        s = self.faint_surface()
        # Two results: a third consistent one would trip the moved-sample reset.
        for x, y in self.STRONG[:2]:
            v = s.check(x, y, slot4_tissue(x, y))
            assert v.would_reject

    def test_observe_and_enforce_reset_alike(self):
        for mode in (MODE_OBSERVE, MODE_ENFORCE):
            s = self.faint_surface(mode)
            (x1, y1), (x2, y2) = self.STRONG[:2]
            s.check(x1, y1, slot4_tissue(x1, y1), quality=0.97)
            v = s.check(x2, y2, slot4_tissue(x2, y2), quality=0.98)
            assert not v.would_reject
            assert s.summary()["clearer_peak_resets"] == 1


# LC-PolScope 20x, 2026-11-15, region 35005_24510 (server_session_20261115_123621.log).
# A diagonal tissue strip, 6.7 x 0.9 mm along its own axes; the region's 53 tile centres
# span 7451 x 2980 um on the stage (the log's "Spatial grid AF ... over 7451x2980 um
# area", which is x_max - x_min by y_max - y_min of the tile positions). The bounds are
# inferred from that and the 496.74 um tile step: one step beyond the outermost AF tiles
# on each side, which also matches the 702 um (one diagonal step) the log reports from
# the first and last tiles to their nearest AF tile. Points are the
# autofocus results fed to the surface, in order: the pre-acquisition AF (Z logged to two
# decimals), the eight focus-survey points (all on ONE row, Y = 5310.67), then the 22
# results of the tile loop (drift sweeps that moved the stage, and one standard AF at
# tile 32; sweeps that held their Z were not recorded, as in workflow.py).
INCIDENT_EXTENT_MM = (7.451, 2.980)
INCIDENT_BOUNDS_UM = (2278.26, 9729.40, 3820.44, 6800.90)
INCIDENT_SURVEY = [
    (2775.005, 6304.155, 44.48),
    (4265.232, 5310.67, 45.20),
    (4761.975, 5310.67, 45.70),
    (5258.717, 5310.67, 45.50),
    (5755.459, 5310.67, 44.20),
    (6252.202, 5310.67, 44.60),
    (6748.944, 5310.67, 44.10),
    (7245.687, 5310.67, 44.30),
    (7742.429, 5310.67, 44.70),
]
INCIDENT_TILE_LOOP = [
    (2775.005, 6304.155, 44.786),
    (3271.747, 6304.155, 43.856),
    (4761.975, 5807.413, 45.832),
    (4265.232, 5807.413, 45.551),
    (3768.490, 5807.413, 45.898),
    (3271.747, 5807.413, 46.498),
    (2775.005, 5807.413, 46.969),
    (4265.232, 5310.670, 47.617),
    (4761.975, 5310.670, 47.261),
    (5258.717, 5310.670, 46.812),
    (5755.459, 5310.670, 46.486),
    (6252.202, 5310.670, 46.076),
    (6748.944, 5310.670, 45.494),
    (7245.687, 5310.670, 45.022),
    (8735.914, 4813.928, 45.745),
    (8239.171, 4813.928, 45.702),
    (7742.429, 4813.928, 46.408),
    (6748.944, 4813.928, 47.616),
    (6252.202, 4813.928, 48.123),
    (5755.459, 4813.928, 48.385),
    (8239.171, 4317.185, 48.532),
    (9232.656, 4317.185, 47.193),
]


class TestSeedingNeedsBothTiltsDetermined:
    """The 2026-11-15 LC-PolScope strip: it seeded on spread, not on evidence.

    After the survey the surface had 0.8 mm of spread (above the 0.5 mm floor) and 0.39
    um RMS, so it licensed and seeded every tile. But eight of its nine points sat on
    one row: the ninth alone set the tilt across the strip, which then wandered from
    -1.42 to -4.0 um/mm as the loop fed it results; seeds ran 1-1.5 um low and tile 21
    came out soft. Turning the surface off fixed it.

    Checking results (catching tens-of-um wide-search misses) keeps the previous
    licence; only seeding waits for both tilts.
    """

    def incident(self, **kw):
        return surface(
            region_extent_mm=INCIDENT_EXTENT_MM, region_bounds_um=INCIDENT_BOUNDS_UM, **kw
        )

    def test_the_replay_reproduces_the_incident_without_the_tilt_rule(self):
        s = self.incident(max_tilt_uncertainty_um=float("inf"))
        for x, y, z in INCIDENT_SURVEY:
            s.add(x, y, z)
        assert s.seeding, s.describe()
        summary = s.summary()
        # The log: "z = 44.75 -0.34*X_mm -1.42*Y_mm ... 9/9 inliers at 0.39 um RMS".
        assert summary["tilt_x_um_per_mm"] == pytest.approx(-0.34, abs=0.02)
        assert summary["tilt_y_um_per_mm"] == pytest.approx(-1.42, abs=0.02)
        assert summary["inlier_rms_um"] == pytest.approx(0.39, abs=0.02)

    def test_it_checks_but_does_not_seed_where_the_incident_seeded(self):
        s = self.incident()
        for x, y, z in INCIDENT_SURVEY:
            s.add(x, y, z)
        assert s.licensed and s.enforcing, "checking keeps the previous licence"
        assert s.predict(4265.232, 5807.413) is not None
        assert not s.seeding
        assert s.seed_z(4265.232, 5807.413) is None
        summary = s.summary()
        assert summary["checking"] and not summary["seeding"]
        assert summary["checking_reason"] is None
        reason = summary["seeding_reason"]
        assert "tilt along Y is not determined" in reason, reason
        # The fit's centre (5645, 5421) is 1.60 mm from the region's far edge in Y.
        assert "carried 1.60 mm to the region's far edge in Y" in reason, reason
        assert reason.isascii()
        assert "checking, NOT seeding" in s.describe()
        # Y, across the strip, is the undetermined axis, by a wide margin.
        assert summary["tilt_uncertainty_um"]["Y"] > 2 * s.max_tilt_uncertainty_um

    def test_it_never_seeds_on_that_strip_and_still_catches_a_miss(self):
        s = self.incident()
        for x, y, z in INCIDENT_SURVEY + INCIDENT_TILE_LOOP:
            s.check(x, y, z)
            assert s.seed_z(x, y) is None
            s.add(x, y, z)
        assert not s.seeding
        # A wide-search miss on that strip is still rejected and replaced.
        x, y = 6000.0, 5000.0
        v = s.check(x, y, s.predict(x, y) + 40.0)
        assert v.would_reject and v.enforced, v.reason

    def test_the_record_qpsc_seeds_from_is_not_licensed(self):
        s = self.incident()
        for x, y, z in INCIDENT_SURVEY:
            s.add(x, y, z)
        r = s.to_record()
        assert r["licensed"] is False and r["checking"] is True
        assert "tilt along Y" in r["unlicensed_reason"]

    def test_a_region_whose_tilts_are_both_measured_seeds(self):
        # About the incident's noise (0.4 um) on a grid that covers the region in both
        # axes determines both tilts well inside the limit.
        grid = np.asarray(tile_grid())
        bounds = (grid[:, 0].min(), grid[:, 0].max(), grid[:, 1].min(), grid[:, 1].max())
        s = feed(surface(region_bounds_um=bounds), tile_grid(), noise_um=0.4, seed=7)
        assert s.seeding, s.describe()
        assert s.seed_z(*CENTRE) == pytest.approx(s.predict(*CENTRE))
        assert "checking and seeding" in s.describe()
        assert max(s.summary()["tilt_uncertainty_um"].values()) <= s.max_tilt_uncertainty_um

    def test_without_a_region_extent_the_points_own_span_is_used(self):
        s = surface()
        for x, y, z in INCIDENT_SURVEY:
            s.add(x, y, z)
        reason = s.seeding_reason()
        # The points span 0.99 mm in Y (one point off the surveyed row).
        assert "carried 0.99 mm across the points in Y" in reason, reason

    def test_the_whole_extent_is_used_when_only_the_extent_is_known(self):
        s = surface(region_extent_mm=INCIDENT_EXTENT_MM)
        for x, y, z in INCIDENT_SURVEY:
            s.add(x, y, z)
        assert "carried 2.98 mm across the region in Y" in s.seeding_reason()

    def test_an_unlicensed_surface_neither_checks_nor_seeds(self):
        s = surface()
        for x, y, z in INCIDENT_SURVEY[:3]:
            s.add(x, y, z)
        assert not s.licensed and not s.seeding
        assert s.seeding_reason() == s.summary()["checking_reason"]

    def test_a_region_one_column_wide_does_not_need_the_tilt_across_it(self):
        # Tiles that all share an X are never predicted at another X, so the X tilt
        # does not matter there; the spread check alone speaks to that geometry.
        s = surface(region_extent_mm=(0.0, 6.0))
        for i in range(10):
            y = CENTRE[1] + i * 600.0
            s.add(CENTRE[0], y, real_plane(CENTRE[0], y))
        assert [axis for axis, *_ in s.tilt_uncertainty()] == ["Y"]
