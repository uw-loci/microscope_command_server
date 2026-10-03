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
        # Learning is unconditional in both modes, so a run in observe measures the
        # surface enforcement would have used. Otherwise the dry run proves nothing.
        points = tile_grid()
        obs = feed(surface(MODE_OBSERVE), points, seed=11)
        enf = feed(surface(MODE_ENFORCE), points, seed=11)
        assert obs.predict(*CENTRE) == pytest.approx(enf.predict(*CENTRE), abs=1e-9)


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
