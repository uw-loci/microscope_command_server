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

#: How many samples must land across a peak's FWHM before its SHAPE is worth
#: believing. Below this the traverse is stepping over the peak rather than
#: resolving it, so the measured height and width are both accidents of where the
#: samples happened to fall.
#:
#: This is not hypothetical. In the measured 10x scan the local stride near the
#: peak was 0.93 um against a 4.7 um FWHM -- 5.0 samples, comfortable. The stride
#: does not change with magnification but the peak does: the same traverse gives
#: about 2.7 samples across a 2.5 um peak and 1.3 across a 1.2 um one, at which
#: point the apex can be stepped over completely and the recorded "peak" is a
#: shoulder. A blunter object in the same field -- a fibre lying across the slide,
#: which is tilted and so stays in focus over a much wider Z range -- is immune to
#: that and wins on score despite being the wrong thing to focus on.
#:
#: Note the existing pre-flight blur budget does NOT cover this. It bounds
#: exposure x velocity, i.e. smear within one frame; this bounds frame INTERVAL x
#: velocity, i.e. the gap between frames. The interval is dominated by readout, so
#: blur can be well inside budget while the stride is several times coarser.
MIN_SAMPLES_ACROSS_FWHM = 3.0


def local_sampling_gap_um(
    samples_zm: List[Tuple[float, float]], index: int, half_window: int = 10
) -> float:
    """Median Z stride between neighbouring samples around ``index``.

    Local rather than whole-scan because the stride is not uniform: streaming frames
    arrive irregularly, and in the measured trace the gap ranged from 0.74 um at the
    median to 4.23 um at the worst. What matters for a peak is the stride where that
    peak is, not the average over a traverse that is mostly empty.
    """
    lo = max(0, index - half_window)
    hi = min(len(samples_zm) - 1, index + half_window)
    gaps = [abs(samples_zm[i + 1][0] - samples_zm[i][0]) for i in range(lo, hi)]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return float("inf")
    return float(median(gaps))


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

    # Resolution guard: a peak the traverse stepped over has no trustworthy shape, so
    # "sharp" cannot be asserted about it either way. Refusing here is the conservative
    # branch -- it sends the caller to the ordered walk rather than committing to a
    # height and width that are artefacts of the sampling.
    stride = local_sampling_gap_um(samples_zm, index)
    samples_across = fwhm / stride if stride > 0 else 0.0
    if samples_across < MIN_SAMPLES_ACROSS_FWHM:
        logger.warning(
            "STREAM_AF:approach strongest peak at Z=%.3f is UNRESOLVED -- only %.1f samples "
            "across its %.1f um width at a %.2f um stride (need %.1f). The traverse is "
            "stepping over peaks this narrow, so both its height and its width are "
            "accidents of sampling, and a broader object in the field (a fibre, say) will "
            "out-score the sample. Slow the approach or raise the frame rate.",
            best_z,
            samples_across,
            fwhm,
            stride,
            MIN_SAMPLES_ACROSS_FWHM,
        )
        return None

    if amplitude >= SHARP_PEAK_MIN_AMPLITUDE_RATIO and fwhm <= SHARP_PEAK_MAX_FWHM_UM:
        logger.info(
            "STREAM_AF:approach standout peak at Z=%.3f -- %.2fx baseline, FWHM %.1f um, "
            "%.1f samples across it at a %.2f um stride",
            best_z,
            amplitude,
            fwhm,
            samples_across,
            stride,
        )
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
