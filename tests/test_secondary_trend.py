"""Reading the secondary metric when the primary has nothing to say.

The 2026-09-10 failure on PPM 20x at stage (-39155, -35245). Autofocus opened at
Z=-452.8 and refused after two attempts; the operator then focused by hand and a
re-run committed Z=-334.5, so true focus was 118 um above where the search started.
Neither scan window -- [-467.8, -437.8] then [-482.8, -422.8] -- came within 88 um
of it, and widening alone never could have: the 150 um cap around that centre reaches
only -377.8.

What the run DID have was a secondary metric doing exactly what a secondary metric is
for. ``brenner_gradient`` is a mean of squared gradients and is pinned to its noise
floor a hundred microns out (0.71% then 0.96% of peak). ``p98_p2`` is an intensity
spread, and defocused tissue still darkens the field, so it rose the whole way:
6.67% then 16.84%, with the gaussian R^2 climbing 0.96 -> 0.98. The abort gate read
only the primary, so the search was declared dead at its strongest moment.

The secondary was not peak-shaped, and that is the point. Its fitted sigma was 12.32 um
over a 22.97 um span, then 23.94 um over a 52.29 um span: the window grew 2.28x and the
fitted width grew 1.94x with it, staying at 0.54 then 0.46 of the span. A peak of fixed
width would have held sigma still while the window opened; a width that scales with the
window is a fitted ramp. Both readings therefore sat above the ``sigma < 0.45 * span``
shape test -- the second by 1.7% -- which suppressed the only log line carrying mu, and
with it the sign, the single quantity that says which way the sample is.
"""

import pytest

from microscope_command_server.server.focus_peaks import secondary_trend

TRUE_FOCUS_Z = -334.5
HALF_WIDTH_UM = 40.0
BASELINE = 0.5833  # relative to the contrast term; set so window 1 lands near 6.7%


def _trace(z_lo, z_hi, n=30, focus=TRUE_FOCUS_Z):
    """A scan window, with a flat-noise primary and a defocus-driven secondary.

    The secondary follows a Lorentzian in defocus distance -- contrast falling off
    either side of focus. Over a window entirely below focus that is a monotonic rise,
    which is the whole situation under test.
    """
    out = []
    for i in range(n):
        z = z_lo + (z_hi - z_lo) * i / (n - 1)
        contrast = 1.0 / (1.0 + ((z - focus) / HALF_WIDTH_UM) ** 2)
        out.append((i * 100.0, z, 12_960_000.0, BASELINE + contrast))
    return out


NEAR = _trace(-467.8, -437.8)  # attempt 1, range 30
WIDE = _trace(-482.8, -422.8)  # attempt 2, range 60 after widening


def test_a_window_entirely_below_focus_reads_as_a_rising_ramp():
    """Sign is the deliverable: rising with Z means focus is above the window."""
    t = secondary_trend(NEAR)
    assert t is not None
    assert t["pearson_r"] > 0.99, "monotonic rise toward focus"
    # Convention shared with the primary slope detector: r > 0 -> edge_high -> shift up.
    assert t["pearson_r"] > 0, "must steer toward less-negative Z, where focus was"


def test_widening_grows_the_secondary_past_the_bar_that_aborted_the_run():
    """The gate's own 1.5x test, applied to the metric that was actually moving."""
    near, wide = secondary_trend(NEAR), secondary_trend(WIDE)
    growth = wide["amplitude"] / near["amplitude"]
    assert growth > 1.5, f"secondary grew {growth:.2f}x -- the run must not abort"
    # For contrast, the primary in the real run grew 0.71% -> 0.96% = 1.36x, under the
    # bar. Defeat has to be unanimous or this exact run is lost again.
    assert 0.96 / 0.71 < 1.5


def test_flat_noise_offers_no_direction():
    """The abort must still be reachable -- this is what a genuinely dead field looks like."""
    flat = [
        (i * 100.0, -460.0 + 0.8 * i, 1.0, 1.0 + 0.0005 * ((i * 37) % 7 - 3)) for i in range(30)
    ]
    t = secondary_trend(flat)
    assert abs(t["pearson_r"]) < 0.5, "no ramp"
    assert t["amplitude"] < 0.005, "no amplitude either"


def test_an_interior_peak_is_not_mistaken_for_a_ramp():
    """A window straddling focus is symmetric, so it must not steer anywhere."""
    t = secondary_trend(_trace(TRUE_FOCUS_Z - 15.0, TRUE_FOCUS_Z + 15.0))
    assert abs(t["pearson_r"]) < 0.5


def test_too_short_a_trace_is_refused_rather_than_guessed():
    assert secondary_trend(_trace(-467.8, -437.8, n=3), min_frames=5) is None
    assert secondary_trend(None) is None
    assert secondary_trend([]) is None


def test_the_measured_sigma_tracked_the_window_which_is_why_the_sign_was_lost():
    """Pins the inference that the field data was a ramp, not a peak.

    Measured: sigma 12.32 um over a 22.97 um span, then 23.94 um over 52.29 um.
    """
    (s1, span1), (s2, span2) = (12.32, 22.97), (23.94, 52.29)

    # A fixed-width peak would not have moved; this tracked the window it was fitted in.
    assert s2 / s1 == pytest.approx(span2 / span1, rel=0.20), "fitted width follows the window"

    # Which is why both fell foul of the peak-shape gate, the second one barely.
    for sigma, span in ((s1, span1), (s2, span2)):
        assert sigma > 0.45 * span, "peak-shaped gate rejected it, discarding mu"
    assert s2 / (0.45 * span2) == pytest.approx(1.017, abs=0.005), "attempt 2 missed by 1.7%"
