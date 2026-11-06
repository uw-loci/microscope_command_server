"""A narrow focus peak found at the edge of a streaming sweep.

PPM 20x, 2026-11-06 (server log 00:07:05-00:07:30): the third sweep, [-118.4, -38.4],
fit a clean peak pinned at its top edge (mu -38.70, sigma 2.35 um, R^2 1.00). The loop
then moved the next window a full range away (center +1.6), retries were used up, and
the Brent fallback searched [-118.4, +41.6] -- 160 um for a peak about 5 um wide. Its
snaps landed on baseline except the edge sample it already had, and it reported
"converged at -38.400". A narrow peak at the edge is a peak, not a slope: center on it.
"""

import logging

import numpy as np

from microscope_command_server.server.af_events import AutofocusEventLog
from microscope_command_server.server.handlers import streaming_focus as sf

MU, SIGMA = -38.701, 2.35  # the 2026-11-06 attempt-3 fit
HEAD_DISCARD_UM = 0.6 * 11.5  # 600 ms at the configured slow speed


def _edge(status="edge_high", best_z=MU, sigma=SIGMA):
    return sf._ScanAttemptResult(status, best_z, 81, 70.2, "pinned", peak_sigma=sigma)


# --- where the next retry sweep goes ---------------------------------------------


def test_a_narrow_edge_peak_is_centered_on_not_stepped_past():
    center, why = sf._next_center_after_edge(_edge(), -78.4, 80.0)
    assert center == MU  # was -78.4 + 80 = +1.6, which put the peak in the head discard
    assert "centering" in why


def test_a_slope_still_shifts_one_full_range_toward_the_edge():
    slope_high = sf._ScanAttemptResult("edge_high", None, 40, 30.0, "monotonic")
    slope_low = sf._ScanAttemptResult("edge_low", -20.0, 40, 30.0, "degenerate fit")
    assert abs(sf._next_center_after_edge(slope_high, -78.4, 80.0)[0] - 1.6) < 1e-9
    assert abs(sf._next_center_after_edge(slope_low, -78.4, 80.0)[0] + 158.4) < 1e-9


# --- the follow-up sweep after the retries run out ------------------------------


def test_the_follow_up_sweep_spans_the_peak_plus_the_discarded_head():
    center, rng = sf._edge_followup_window(_edge(), HEAD_DISCARD_UM)
    assert center == MU
    # +/- 4 sigma plus 6.9 um of head discard on each side: 18.8 + 13.8 = 32.6 um
    assert abs(rng - (8 * SIGMA + 2 * HEAD_DISCARD_UM)) < 1e-9
    half = rng / 2
    assert center - half + HEAD_DISCARD_UM < MU - 2 * SIGMA  # the peak is past the discard
    assert center + half > MU + 2 * SIGMA


def test_a_very_narrow_peak_still_gets_the_minimum_sweep():
    _, rng = sf._edge_followup_window(_edge(sigma=0.5), 0.0)
    assert rng == sf.EDGE_PEAK_MIN_RANGE_UM


def test_no_follow_up_for_a_slope_or_a_success():
    assert (
        sf._edge_followup_window(sf._ScanAttemptResult("edge_high", None, 1, 1, "s"), 6.9) is None
    )
    assert (
        sf._edge_followup_window(sf._ScanAttemptResult("success", -38.0, 1, 1, "ok"), 6.9) is None
    )


# --- the Brent fallback window ---------------------------------------------------


def test_the_brent_bracket_fits_a_narrow_edge_peak():
    lo, hi = sf._brent_bracket(_edge(), [(-38.4, 3e7)], 80.0, 3, -78.4, None, None)
    assert abs(lo - (MU - 4 * SIGMA)) < 1e-9 and abs(hi - (MU + 4 * SIGMA)) < 1e-9


def test_without_a_narrow_peak_the_bracket_is_one_range_around_the_best_sample():
    slope = sf._ScanAttemptResult("edge_high", None, 1, 1.0, "slope")
    assert sf._brent_bracket(slope, [(-38.4, 3e7), (-60.0, 1.5e7)], 80.0, 3, -78.4, None, None) == (
        -118.4,
        41.6,
    )
    # and it is clamped to the stage limits
    assert sf._brent_bracket(slope, [(-38.4, 3e7)], 80.0, 3, -78.4, -100.0, 0.0) == (-100.0, 0.0)


class _SpikeCore:
    """A focus stage whose frames are sharp only within a few um of Z = -36."""

    def __init__(self, peak=-36.0, sigma=SIGMA):
        self.z = 0.0
        self.peak = peak
        self.sigma = sigma

    def set_position(self, device, z):
        self.z = float(z)

    def get_position(self, device):
        return self.z

    def device_busy(self, device):
        return False

    def wait_for_device(self, device):
        pass

    def snap_image(self):
        pass

    def get_image(self):
        sharpness = 50.0 * np.exp(-0.5 * ((self.z - self.peak) / self.sigma) ** 2) + 1.0
        rng = np.random.default_rng(0)
        return (rng.standard_normal(32 * 32) * sharpness + 1000.0).astype(np.uint16)

    def get_image_width(self):
        return 32

    def get_image_height(self):
        return 32

    def get_number_of_components(self):
        return 1


def _brent(lo, hi, monkeypatch):
    log = AutofocusEventLog()
    monkeypatch.setattr(sf, "AF_EVENTS", log)
    log.af_started("streaming", {})
    return sf._brent_fallback_scan(_SpikeCore(), "Z", None, lo, hi, "brenner_gradient")


def test_brent_over_the_old_wide_bracket_says_it_found_nothing_better(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=sf.logger.name)
    _brent(-118.4, 41.6, monkeypatch)
    assert any("not a refined peak" in r.getMessage() for r in caplog.records)


def test_brent_over_the_peak_sized_bracket_finds_the_peak(monkeypatch):
    lo, hi = sf._brent_bracket(_edge(), [], 80.0, 3, -78.4, None, None)
    result = _brent(lo, hi, monkeypatch)
    assert result.status == "success"
    assert abs(result.best_z - (-36.0)) < 1.5
