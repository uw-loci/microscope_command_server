"""Where a focus sample's Z comes from.

Sample Z used to be modelled as ``wall_ms * slow_speed_um_per_s``. The model is an
extrapolation from a hand-calibrated constant; the Z-poll trace is a direct measurement
of the same quantity. They disagree, and the measurement is right.

Measured on PPM 20x, 2026-08-26, over a 267 um approach traverse: configured 11.5 um/s,
real ~11.96 um/s. Only 4% fast -- but the error INTEGRATES over the traverse, so by the
sample plane the modelled Z lagged the true Z by 8.6 um. The operator's own focus was
-237.0; the metric peak sat at z_polled = -236.65 and at z_model = -228.08. Committing
on the model would have driven to -228 and reported success.

This is an approach-mode hazard specifically. The same 4% error over a routine 20-40 um
sweep is sub-micron and invisible; the approach traverses ten times further from a
retracted start, which turns a negligible calibration error into a focus miss.
"""

import numpy as np
import pytest

from microscope_command_server.server.focus_geometry import build_z_from_poll


def _real_traverse_poll():
    """A 267 um descent at the real ~11.96 um/s, sampled every 50 ms like the poller."""
    return [(t * 1000.0, -11.96 * t) for t in np.arange(0.0, 22.4, 0.05)]


def test_sample_z_follows_the_measurement_not_the_model():
    z_at = build_z_from_poll(_real_traverse_poll())

    # The frame that peaked, at t = 19.833 s into the scan.
    peak_wall_ms = 19833.0
    measured = z_at(peak_wall_ms)
    modelled = -11.5 * (peak_wall_ms / 1000.0)

    assert measured == pytest.approx(-237.0, abs=1.0), "measurement should land on the true focus"
    assert modelled == pytest.approx(-228.1, abs=1.0), "the model is where the old code committed"
    # The gap is the whole point: far more than a 20x depth of field.
    assert abs(measured - modelled) > 5.0


def test_the_error_is_negligible_over_a_routine_sweep():
    """Why this survived so long: on a short sweep the same mis-calibration is invisible."""
    z_at = build_z_from_poll(_real_traverse_poll())

    # 30 um in, the distance a normal edge-retry sweep covers.
    wall_ms = 30.0 / 11.96 * 1000.0
    drift = abs(z_at(wall_ms) - (-11.5 * wall_ms / 1000.0))

    assert drift < 1.5, "a routine sweep hides the error, which is why it needed a traverse to find"


def test_interpolation_lands_between_polls():
    z_at = build_z_from_poll([(0.0, 0.0), (100.0, -10.0)])
    assert z_at(50.0) == pytest.approx(-5.0)
    assert z_at(0.0) == pytest.approx(0.0)
    assert z_at(100.0) == pytest.approx(-10.0)


def test_out_of_range_times_clamp_rather_than_extrapolate():
    # A frame retrieved after the last poll must not be projected past the end of travel.
    z_at = build_z_from_poll([(0.0, 0.0), (100.0, -10.0)])
    assert z_at(500.0) == pytest.approx(-10.0)
    assert z_at(-50.0) == pytest.approx(0.0)


def test_an_unsorted_poll_trace_is_still_usable():
    z_at = build_z_from_poll([(100.0, -10.0), (0.0, 0.0), (50.0, -5.0)])
    assert z_at(25.0) == pytest.approx(-2.5)


@pytest.mark.parametrize("trace", [[], [(0.0, 0.0)]])
def test_too_few_polls_returns_none_so_the_caller_falls_back(trace):
    # Not an error: with nothing to interpolate over, the model is all there is. The
    # caller warns and uses it rather than failing a scan that is otherwise fine.
    assert build_z_from_poll(trace) is None
