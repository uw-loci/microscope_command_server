"""Which peaks a focus scan contains, and whether one of them is unmistakable.

Pure analysis of a (z, metric) trace -- no hardware, no config. Lives outside
``server.handlers`` for the same reason ``server.focus_validity`` and
``server.focus_geometry`` do: importing the handlers package pulls in
``microscope_control`` -> ``pycromanager``, which a unit test has no business
needing in order to check how a curve is read.

Two tests live here and they are NOT interchangeable:

* ``first_prominent_peaks_in_scan_order`` is RELATIVE -- prominence against the
  scan's own range. It answers "what are the candidates", and it deliberately
  errs toward listing them.
* ``standout_peak`` is ABSOLUTE -- amplitude against the scan's baseline, and
  width in microns. It answers "is one of them beyond argument".

The relative test alone cannot fail, and that is a real failure mode rather than
a theoretical one: on a measured blank-slide scan the entire range is 1.3% of the
signal, so ordinary noise clears any fraction of it and the finder reports
confident peaks on a slide with nothing on it.
"""

import logging
from statistics import median
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def first_prominent_peaks_in_scan_order(
    samples_zm: List[Tuple[float, float]],
    prominence_fraction: float = 0.15,
) -> List[Tuple[float, float]]:
    """Prominent local maxima, in the order the scan met them.

    Order is what matters here: an approach commits to the FIRST peak it can
    justify, so the caller needs candidates in travel order, not by strength.
    Prominence is measured against the deeper adjacent valley so a shoulder on
    the way up is not mistaken for a separate focal plane.

    Returns a list of (z, metric).
    """
    if len(samples_zm) < 5:
        return []
    values = [m for _, m in samples_zm]
    lo = min(values)
    hi = max(values)
    rng = hi - lo
    if rng <= 0:
        return []

    peaks: List[Tuple[float, float]] = []
    for i in range(1, len(values) - 1):
        if values[i] <= values[i - 1] or values[i] < values[i + 1]:
            continue
        left_valley = values[i]
        for j in range(i, -1, -1):
            left_valley = min(left_valley, values[j])
            if values[j] > values[i]:
                break
        right_valley = values[i]
        for j in range(i, len(values)):
            right_valley = min(right_valley, values[j])
            if values[j] > values[i]:
                break
        prominence = values[i] - max(left_valley, right_valley)
        if prominence >= prominence_fraction * rng:
            peaks.append((samples_zm[i][0], values[i]))
    return peaks


#: A peak this far above the scan's baseline, and this narrow, is the sample and
#: nothing else -- so the approach goes straight to it instead of walking every
#: earlier candidate. Both bars must be cleared; either alone is not enough.
#:
#: Set from measured PPM 10x scans (2026-08-28), and the three populations are far
#: apart on BOTH axes:
#:
#:   real tissue peak      2.44x baseline    FWHM   4.7 um
#:   spurious peak (20x)   1.17x baseline    FWHM ~125 um   (sigma 53 um, R^2 0.78)
#:   blank slide           1.006x baseline   no peak at all
#:
#: That is 8.6x of separation on amplitude and 27x on width, so these bars sit in
#: a wide empty gap rather than being tuned to a boundary.
#:
#: Deliberately ABSOLUTE, unlike the prominence test in
#: first_prominent_peaks_in_scan_order, which measures against the scan's own
#: (max - min). A relative test cannot fail: on the blank scan the whole range is
#: 1.3% of the signal, so ordinary noise clears any fraction of it. That is how a
#: flat scan produces confident "peaks".
SHARP_PEAK_MIN_AMPLITUDE_RATIO = 1.6
SHARP_PEAK_MAX_FWHM_UM = 25.0


def peak_shape(
    samples_zm: List[Tuple[float, float]], index: int, baseline: float
) -> Tuple[float, float]:
    """(amplitude as a multiple of baseline, FWHM in um) for the peak at ``index``.

    FWHM is measured by walking out from the peak to where the metric falls to
    halfway between the baseline and the peak, then interpolating in Z. A peak whose
    half-max crossing is never reached on one side -- the scan ended, or it is a
    shoulder rather than a peak -- gets an infinite width, which disqualifies it.
    """
    values = [m for _, m in samples_zm]
    zs = [z for z, _ in samples_zm]
    peak = values[index]
    if baseline <= 0 or peak <= baseline:
        return (0.0, float("inf"))
    half = baseline + (peak - baseline) / 2.0

    def crossing(walk) -> Optional[float]:
        for j in walk:
            if values[j] < half:
                k = j + 1 if j < index else j - 1
                span = values[k] - values[j]
                if span == 0:
                    return zs[j]
                f = (half - values[j]) / span
                return zs[j] + f * (zs[k] - zs[j])
        return None

    left = crossing(range(index, -1, -1))
    right = crossing(range(index, len(values)))
    if left is None or right is None:
        return (peak / baseline, float("inf"))
    return (peak / baseline, abs(right - left))


def standout_peak(
    samples_zm: List[Tuple[float, float]], peaks: List[Tuple[float, float]]
) -> Optional[Tuple[float, float, float, float]]:
    """The one unmistakable focus peak in this scan, or None.

    "Unmistakable" means the strongest peak is both far above the scan's baseline and
    narrow -- the signature of the sample plane, which no surface or gradient in the
    measured data comes close to. When it is present the approach can go straight
    there instead of snapping at every earlier candidate in travel order.

    Only the STRONGEST peak is considered, and only when it is the clear winner. If a
    second peak is within reach of it, "extreme" is not the right word for either and
    the caller falls back to the ordered walk, which is the conservative reading.

    Returns (z, metric, amplitude_ratio, fwhm_um) or None.
    """
    if not peaks or len(samples_zm) < 5:
        return None
    values = [m for _, m in samples_zm]
    baseline = float(median(values))
    if baseline <= 0:
        return None

    by_z = {z: i for i, (z, _) in enumerate(samples_zm)}
    best_z, best_m = max(peaks, key=lambda p: p[1])
    index = by_z.get(best_z)
    if index is None:
        return None

    # Ambiguity guard: a runner-up of comparable height means this is not a standout.
    runners = [m for z, m in peaks if z != best_z]
    if runners:
        second = max(runners)
        if (second - baseline) > 0.5 * (best_m - baseline):
            logger.info(
                "STREAM_AF:approach strongest peak is not a standout (runner-up is %.0f%% of "
                "its height above baseline); walking peaks in travel order",
                100.0 * (second - baseline) / max(best_m - baseline, 1e-9),
            )
            return None

    amplitude, fwhm = peak_shape(samples_zm, index, baseline)
    if amplitude >= SHARP_PEAK_MIN_AMPLITUDE_RATIO and fwhm <= SHARP_PEAK_MAX_FWHM_UM:
        return (best_z, best_m, amplitude, fwhm)
    logger.info(
        "STREAM_AF:approach strongest peak at Z=%.3f is not sharp enough to short-circuit "
        "(amplitude %.2fx baseline, need %.2fx; FWHM %.1f um, need <= %.1f); walking peaks "
        "in travel order",
        best_z,
        amplitude,
        SHARP_PEAK_MIN_AMPLITUDE_RATIO,
        fwhm,
        SHARP_PEAK_MAX_FWHM_UM,
    )
    return None
