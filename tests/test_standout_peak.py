"""The "extreme peak" short-circuit in the approach-from-safe-Z focus.

An approach normally walks every prominent peak in travel order, snapping at each to ask
the tissue gate about it. When the scan contains one unmistakable peak that is both far
above baseline and narrow, that is the sample and the walk is wasted work.

The bars must sit in the gap between three MEASURED populations (PPM, 2026-08-28):

    real tissue peak      2.44x baseline    FWHM   4.7 um
    spurious peak (20x)   1.17x baseline    FWHM ~125 um
    blank slide           1.006x baseline   no peak

so these tests are built from the recorded traces rather than from idealised curves.
"""

import math

from microscope_command_server.server.focus_peaks import (
    SHARP_PEAK_MAX_FWHM_UM,
    SHARP_PEAK_MIN_AMPLITUDE_RATIO,
    first_prominent_peaks_in_scan_order,
    peak_shape,
    standout_peak,
)

# Verbatim (z_actual_um, metric) around the peak of the real tissue scan.
TISSUE_PEAK_REGION = [
    (-211.8588, 12857876.1),
    (-213.4328, 12994238.0),
    (-215.4405, 13337417.8),
    (-217.6472, 14056054.4),
    (-218.8800, 14563951.1),
    (-219.7913, 14780715.6),
    (-219.9895, 15186911.1),
    (-220.3863, 16088536.9),
    (-221.1320, 16619095.2),
    (-221.3040, 17229676.7),
    (-221.4753, 18910499.6),
    (-222.3586, 25450257.6),
    (-223.8231, 30697980.8),
    (-225.7692, 25625561.2),
    (-227.2254, 18114096.2),
    (-229.2302, 14510964.1),
    (-231.2743, 13757867.6),
    (-232.8008, 13361307.3),
    (-233.7692, 13296567.1),
    (-233.9564, 13149914.8),
    (-234.9143, 13045667.1),
    (-235.8018, 13024599.6),
    (-235.9801, 12940551.8),
    (-236.6848, 12807754.6),
]


def _tissue_scan():
    """The real peak preceded by the flat run that actually precedes it in the CSV."""
    flat = [(-1.0 - 1.2 * i, 12.59e6 + (i % 5) * 12000.0) for i in range(170)]
    return flat + TISSUE_PEAK_REGION


def _blank_scan():
    """The blank-slide scan: no peak, 1.3% total variation over the whole traverse."""
    return [(-1.0 - 0.95 * i, 14.80e6 + ((i * 37) % 19 - 9) * 10000.0) for i in range(271)]


def _broad_spurious_scan():
    """The 20x peak that cost a slide: 17% above baseline, gaussian sigma 53 um."""
    base, amp, sigma, centre = 12.0e6, 0.167 * 12.0e6, 53.0, -160.0
    return [
        (z, base + amp * math.exp(-((z - centre) ** 2) / (2 * sigma**2)))
        for z in [-1.0 - 1.3 * i for i in range(275)]
    ]


def test_the_real_tissue_peak_is_a_standout():
    scan = _tissue_scan()
    peaks = first_prominent_peaks_in_scan_order(scan)
    result = standout_peak(scan, peaks)
    assert result is not None, "the measured tissue peak must short-circuit the walk"
    z, _m, amplitude, fwhm = result
    assert abs(z - (-223.8231)) < 1e-6
    assert amplitude > 2.4
    assert fwhm < 6.0


def test_the_broad_spurious_peak_is_not_a_standout():
    """The failure this guards: committing a tall-looking but broad peak as if it were focus."""
    scan = _broad_spurious_scan()
    peaks = first_prominent_peaks_in_scan_order(scan)
    assert peaks, "the spurious peak IS prominent -- that is why it fooled the relative test"
    assert standout_peak(scan, peaks) is None


def test_a_flat_blank_scan_produces_no_standout():
    """The degeneracy an absolute test exists to close.

    The relative prominence test cannot reject this: the blank scan's whole range is about
    1.3% of the signal, so ordinary noise clears any fraction of it, and the finder reports
    confident peaks on a slide with nothing on it.
    """
    scan = _blank_scan()
    peaks = first_prominent_peaks_in_scan_order(scan)
    assert peaks, "relative prominence finds 'peaks' in noise -- the reason for the absolute test"
    assert standout_peak(scan, peaks) is None


def test_two_comparable_peaks_fall_back_to_the_ordered_walk():
    """'Extreme' means one clear winner; two of a kind is exactly when order matters."""
    scan = _tissue_scan()
    # Graft a second peak of the same height earlier in travel order.
    doubled = scan[:60] + [(scan[60][0], 30.0e6), (scan[61][0], 20.0e6)] + scan[62:]
    peaks = first_prominent_peaks_in_scan_order(doubled)
    assert standout_peak(doubled, peaks) is None


def test_a_peak_at_the_scan_edge_is_disqualified():
    """No falling side means no measurable width, so it cannot be called sharp."""
    rising = [(-1.0 - 1.2 * i, 12.0e6 + i * 400000.0) for i in range(40)]
    amplitude, fwhm = peak_shape(rising, len(rising) - 1, 12.0e6)
    assert math.isinf(fwhm)
    assert amplitude > SHARP_PEAK_MIN_AMPLITUDE_RATIO  # tall, but still refused


def test_bars_sit_in_the_gap_rather_than_on_a_boundary():
    """Guards against a later 'tidy-up' tightening these onto the measured values."""
    assert 1.2 < SHARP_PEAK_MIN_AMPLITUDE_RATIO < 2.4, "must reject 1.17x and accept 2.44x"
    assert 6.0 < SHARP_PEAK_MAX_FWHM_UM < 125.0, "must accept 4.7 um and reject ~125 um"
