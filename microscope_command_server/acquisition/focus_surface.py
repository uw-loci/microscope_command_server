"""A robust surface through the focus points an acquisition has measured.

Why this exists
---------------
The acquisition loop keeps a list of ``(x, y, z)`` focus results and, for every tile
that does not run its own autofocus, holds the Z of the spatially nearest one. That is
a zeroth-order focus map: no fitting, no smoothing, and -- the part that costs us --
no way to notice that one of those measurements is wrong.

Measured on ten regions across three PPM sessions (see
``claude-reports/design/autofocus-empirical/summaries/focus_surface.md``):

* A single tilted plane describes measured focus to 0.27-1.11 um RMS over regions up
  to 19 mm across, while focus itself travels 4-150 um over the same region. The
  sample is tilted, not curved -- a quadratic term buys about 0.1 um.
* Narrow sweep drift checks land more than 5 um off that plane 1.2% of the time.
  Wide standard autofocus does so **25%** of the time, and more than 15 um off 21% of
  the time, worst case 73 um. One region had 12 of 12 standard results a median 43 um
  off a plane its 111 sweeps defined to 0.27 um.
* The wide search runs at the first tissue tile of every region and after every jump,
  and its result is adopted with no validation and then handed to every tile around
  it. A region can start 70 um out of focus and stay there.

So this surface's first job is not to save time. It is to be the one thing in the
system that can tell a 43 um miss from a measurement, because hundreds of other
measurements of the same physical surface disagree with it.

How it stays honest
-------------------
* **It refuses to speak without redundancy.** A plane has three parameters, so three
  points determine it exactly and the residual is zero by construction -- a fit with
  no redundancy cannot report its own failure. Nothing is licensed below
  ``min_points`` (default 6).
* **It refuses to speak when it does not fit.** Above ``max_rms_um`` the sample is not
  planar -- a folded or detached section, a cytology smear -- and the surface returns
  None, which drops the caller back to today's nearest-neighbour behaviour.
* **The fit is robust.** RANSAC then refit on inliers: one 70 um miss drags a
  least-squares plane several microns, and that is exactly the input we expect.
* **It can be overruled by reality.** When several consecutive rejected measurements
  agree with each other and not with the surface, the sample moved (a re-seated slide,
  a stage frame shift) and the surface resets onto them rather than continuing to veto
  the sensor.

Modes: ``off`` (inert), ``observe`` (fit and report, change nothing), ``enforce``
(predict and reject). ``observe`` exists so a run can measure what enforcement would
have done before anything acts on it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

MODE_OFF = "off"
MODE_OBSERVE = "observe"
MODE_ENFORCE = "enforce"
VALID_MODES = (MODE_OFF, MODE_OBSERVE, MODE_ENFORCE)

# Points this far (um) from the hypothesis are not treated as describing the surface.
# 3.0 um admits real section topography on our samples (the measured plane RMS is
# 0.27-1.11 um) while excluding a focus that landed on the coverslip or on debris.
DEFAULT_INLIER_BAND_UM = 3.0

# Floor on the rejection gate. Below the depth of field there is nothing to reject.
DEFAULT_REJECT_MARGIN_UM = 5.0

# The gate also scales with the fit's own quality, so a noisier surface is less
# willing to overrule a measurement.
DEFAULT_REJECT_SIGMA = 4.0

# A plane needs three points; six is the smallest set that leaves enough redundancy
# that one bad point cannot quietly define the surface.
DEFAULT_MIN_POINTS = 6

# Above this the sample is not a plane and the surface declines to predict.
DEFAULT_MAX_RMS_UM = 3.0

# Fraction of measurements that must agree with the surface before it is believed.
# Without this, RANSAC can always find some flat-looking subset of a scattered set and
# report a small inlier RMS over it -- a surface fitted to a third of the data, with
# the confidence of a fit to all of it. Measured inlier fractions on real regions are
# 91-100%, so 0.6 is far below anything a planar sample produces.
DEFAULT_MIN_INLIER_FRACTION = 0.6

# The measurements must span at least this much (mm) in their WEAKER direction before
# the plane's two tilt terms are both constrained. Six points down one column of a
# raster are six points on a line: they fix the tilt along the column and say nothing
# about the tilt across it, yet least squares will happily return a number and
# extrapolate it across the whole slide. Scan order makes exactly that the first thing
# the surface sees, so this is not a hypothetical.
DEFAULT_MIN_SPREAD_MM = 0.5

# ...but a region only 0.5 mm wide cannot produce 0.5 mm of spread, and does not need
# to: an unconstrained across-tilt can only do damage over the distance it is
# extrapolated, which in a thin strip is almost nothing. Replaying 51 real regions, an
# absolute floor refused five of them at 0.38-0.49 mm -- thin strips where the fit was
# perfectly good (0.22-0.41 um RMS, every point an inlier). So the requirement is the
# floor OR this fraction of the region's own narrow dimension, whichever is smaller.
DEFAULT_SPREAD_COVERAGE = 0.6

# Consecutive rejections that agree with each other and not with the surface, after
# which the surface concedes that the sample moved.
RESURFACE_AFTER_REJECTIONS = 3

_RANSAC_ITERATIONS = 100
_RANSAC_SAMPLE = 4


@dataclass
class Verdict:
    """What the surface thinks of one measurement."""

    predicted_z: Optional[float] = None
    residual_um: Optional[float] = None
    gate_um: Optional[float] = None
    # True when the measurement disagrees with a licensed surface by more than the gate.
    would_reject: bool = False
    # True only when the surface is also enforcing, i.e. the caller should act on it.
    enforced: bool = False
    reason: str = "surface not licensed"


@dataclass
class _Fit:
    """A plane z = z0 + ax * X_mm + ay * Y_mm about a centre, with its own quality."""

    centre: Tuple[float, float]
    z0: float
    ax: float
    ay: float
    inlier_rms_um: float
    n_inliers: int
    n_points: int
    # Singular values (mm) of the centred inlier XY cloud. The smaller one is how well
    # the second tilt term is constrained; near zero means the points are a line.
    spread_mm: Tuple[float, float] = (0.0, 0.0)

    def predict(self, x: float, y: float) -> float:
        return (
            self.z0
            + self.ax * (x - self.centre[0]) / 1000.0
            + self.ay * (y - self.centre[1]) / 1000.0
        )

    def tilt_um_per_mm(self) -> float:
        return float(np.hypot(self.ax, self.ay))


@dataclass
class FocusSurface:
    """Accumulates focus measurements and fits a robust plane through them.

    Learning is unconditional: every measurement is added and the robust fit decides
    what describes the surface. The mode only governs whether anyone acts on the
    result, so ``observe`` and ``enforce`` learn exactly the same surface.
    """

    mode: str = MODE_OFF
    min_points: int = DEFAULT_MIN_POINTS
    max_rms_um: float = DEFAULT_MAX_RMS_UM
    min_inlier_fraction: float = DEFAULT_MIN_INLIER_FRACTION
    min_spread_mm: float = DEFAULT_MIN_SPREAD_MM
    spread_coverage: float = DEFAULT_SPREAD_COVERAGE
    # (dx, dy) extent of the tiles this surface will be asked about, in mm. Lets the
    # spread requirement scale to the region instead of assuming every region is wide.
    region_extent_mm: Optional[Tuple[float, float]] = None
    inlier_band_um: float = DEFAULT_INLIER_BAND_UM
    reject_margin_um: float = DEFAULT_REJECT_MARGIN_UM
    reject_sigma: float = DEFAULT_REJECT_SIGMA
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    _points: List[Tuple[float, float, float]] = field(default_factory=list)
    _fit: Optional[_Fit] = None
    _rejected_streak: List[Tuple[float, float, float]] = field(default_factory=list)
    _n_checks: int = 0
    _n_would_reject: int = 0
    _n_resurfaces: int = 0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(12345))

    # ---------- state ----------

    @property
    def active(self) -> bool:
        """Whether the surface is doing anything at all."""
        return self.mode in (MODE_OBSERVE, MODE_ENFORCE)

    @property
    def licensed(self) -> bool:
        """Whether the fit is good enough to be worth consulting."""
        return self._unlicensed_reason() is None

    def _unlicensed_reason(self) -> Optional[str]:
        """Why the surface may not be consulted, or None when it may be.

        Four separate ways a fit can look fine and not be:
        too few agreeing points, a residual too large to be a plane, a majority that
        does not agree with it, and points that do not spread far enough to constrain
        both tilts. The last two were each found by a test that the first two passed.
        """
        if not self.active:
            return "focus surface off"
        f = self._fit
        if f is None:
            return f"not enough focus points yet ({len(self._points)} of {self.min_points})"
        if f.n_inliers < self.min_points:
            return (
                f"only {f.n_inliers} of {f.n_points} focus points agree on a surface "
                f"(need {self.min_points})"
            )
        if f.inlier_rms_um > self.max_rms_um:
            return (
                f"sample is not planar here (fit RMS {f.inlier_rms_um:.2f} um "
                f"> {self.max_rms_um:.2f} um limit)"
            )
        if f.n_points > 0 and (f.n_inliers / f.n_points) < self.min_inlier_fraction:
            return (
                f"only {100.0 * f.n_inliers / f.n_points:.0f}% of focus points agree "
                f"with the surface (need {100.0 * self.min_inlier_fraction:.0f}%)"
            )
        required = self.required_spread_mm()
        if min(f.spread_mm) < required:
            scaled = ""
            if self.region_extent_mm is not None and required < self.min_spread_mm:
                scaled = (
                    f", scaled down from {self.min_spread_mm:.2f} mm because the region "
                    f"is only {min(self.region_extent_mm):.2f} mm across"
                )
            return (
                f"focus points span only {min(f.spread_mm):.2f} mm across "
                f"(need {required:.2f} mm{scaled}) -- they do not constrain a tilt "
                f"in both axes"
            )
        return None

    def required_spread_mm(self) -> float:
        """How much spread this region's geometry can reasonably be asked for."""
        if self.region_extent_mm is None:
            return self.min_spread_mm
        narrow = min(self.region_extent_mm)
        if narrow <= 0:
            return self.min_spread_mm
        return min(self.min_spread_mm, self.spread_coverage * narrow)

    @property
    def enforcing(self) -> bool:
        """Whether the caller should act on this surface's opinion."""
        return self.mode == MODE_ENFORCE and self.licensed

    # ---------- learning ----------

    def add(self, x: float, y: float, z: float) -> None:
        """Record a focus measurement and refit."""
        if not self.active:
            return
        self._points.append((float(x), float(y), float(z)))
        self._refit()

    def _refit(self) -> None:
        if len(self._points) < self.min_points:
            self._fit = None
            return
        pts = np.asarray(self._points, dtype=float)
        self._fit = _ransac_plane(pts, self.inlier_band_um, self._rng, self.logger)

    # ---------- using ----------

    def predict(self, x: float, y: float) -> Optional[float]:
        """Predicted focus Z, or None when the surface is not licensed to say."""
        if not self.licensed:
            return None
        return float(self._fit.predict(x, y))

    def check(self, x: float, y: float, z: float) -> Verdict:
        """Compare one measurement against the surface.

        Always computes, whatever the mode, so an ``observe`` run can report exactly
        what enforcement would have done. Only ``enforced`` says to act.
        """
        if not self.active:
            return Verdict(reason="focus surface off")
        self._n_checks += 1
        predicted = self.predict(x, y)
        if predicted is None:
            return Verdict(reason=self._unlicensed_reason() or "surface not licensed")

        residual = abs(z - predicted)
        gate = max(self.reject_margin_um, self.reject_sigma * self._fit.inlier_rms_um)
        if residual <= gate:
            self._rejected_streak.clear()
            return Verdict(
                predicted_z=predicted,
                residual_um=residual,
                gate_um=gate,
                reason="agrees with the focus surface",
            )

        self._n_would_reject += 1
        self._rejected_streak.append((float(x), float(y), float(z)))
        resurfaced = self._maybe_resurface()
        if resurfaced:
            # The sample moved; this measurement is now the truth, not an outlier.
            return Verdict(
                predicted_z=predicted,
                residual_um=residual,
                gate_um=gate,
                reason="focus surface reset onto the recent measurements",
            )
        return Verdict(
            predicted_z=predicted,
            residual_um=residual,
            gate_um=gate,
            would_reject=True,
            enforced=self.enforcing,
            reason=(
                f"{residual:.1f} um from a surface {self._fit.n_inliers} points define "
                f"to {self._fit.inlier_rms_um:.2f} um RMS (gate {gate:.1f} um)"
            ),
        )

    def _maybe_resurface(self) -> bool:
        """Concede to the sensor when the rejected measurements agree with each other.

        A surface that keeps vetoing is the worst outcome available: it is confident,
        wrong, and silent. The discriminator is whether the rejections are mutually
        consistent. Scattered rejections are failed autofocus attempts; rejections that
        themselves describe one surface mean the sample is no longer where this one
        thinks it is -- a re-seated slide, or one of the stage frame shifts this rig
        has had.

        "Consistent" has to mean consistent with a *translated version of this
        surface*, not merely close to each other in Z. Replaying 51 real regions threw
        up the counter-example: three rejected results with a Z spread of exactly
        0.00 um reset a surface that 108 points defined to 0.27 um RMS. Three identical
        Z values at different XY are not a moved sample -- on a tilted slide a moved
        sample's focus still changes across the field. They are a stage or a sweep
        returning a stale reading, which is the one thing that must not be allowed to
        dislodge a good surface.
        """
        if len(self._rejected_streak) < RESURFACE_AFTER_REJECTIONS:
            return False
        recent = np.asarray(self._rejected_streak[-RESURFACE_AFTER_REJECTIONS:], dtype=float)
        if self._fit is None:
            return False
        # Distinct places, or this is one tile being retried rather than a moved slide.
        if len(np.unique(np.round(recent[:, :2], 1), axis=0)) < RESURFACE_AFTER_REJECTIONS:
            return False
        # Keep this surface's tilt, re-fit the offset only, and see whether the rejected
        # points actually lie on the result. A translated slide does; a stuck reading
        # does not, because it ignores the tilt.
        tilted = np.array(
            [
                self._fit.ax * (x - self._fit.centre[0]) / 1000.0
                + self._fit.ay * (y - self._fit.centre[1]) / 1000.0
                for x, y, _ in recent
            ]
        )
        offsets = recent[:, 2] - tilted
        if float(np.std(offsets)) > self.max_rms_um:
            # Not telling a coherent story -- these are failures, not a moved sample.
            return False
        spread = float(np.std(offsets))
        old = self._fit
        self._points = [tuple(p) for p in recent]
        self._fit = None
        self._rejected_streak.clear()
        self._n_resurfaces += 1
        self.logger.warning(
            "FOCUS SURFACE RESET: the last %d autofocus results sit on this surface's "
            "own tilt at a different height (offset spread %.2f um) and disagree with "
            "it (which %d points defined to %.2f um RMS). That is what a moved sample "
            "looks like, so rebuilding from these points. If this repeats, suspect the "
            "stage frame or a re-seated slide.",
            RESURFACE_AFTER_REJECTIONS,
            spread,
            old.n_inliers if old else 0,
            old.inlier_rms_um if old else float("nan"),
        )
        return True

    # ---------- reporting ----------

    def describe(self) -> str:
        """One line for the log: what the surface currently believes."""
        if not self.active:
            return "focus surface: off"
        if self._fit is None:
            return (
                f"focus surface: {len(self._points)} points, "
                f"not fitted yet (needs {self.min_points})"
            )
        f = self._fit
        why = self._unlicensed_reason()
        verdict = "licensed" if why is None else f"NOT licensed: {why}"
        return (
            f"focus surface: z = {f.z0:.2f} {f.ax:+.2f}*X_mm {f.ay:+.2f}*Y_mm about "
            f"({f.centre[0]:.0f}, {f.centre[1]:.0f}), tilt {f.tilt_um_per_mm():.2f} um/mm, "
            f"{f.n_inliers}/{f.n_points} inliers at {f.inlier_rms_um:.2f} um RMS, "
            f"spread {f.spread_mm[0]:.1f} x {f.spread_mm[1]:.1f} mm, "
            f"mode={self.mode}, {verdict}"
        )

    def summary(self) -> dict:
        """End-of-region numbers, for the acquisition summary and for tests."""
        out = {
            "mode": self.mode,
            "points": len(self._points),
            "checks": self._n_checks,
            "would_reject": self._n_would_reject,
            "resurfaces": self._n_resurfaces,
            "licensed": self.licensed,
            # Always present, including when there is no fit at all -- a summary that
            # omits the reason precisely when the surface did nothing is the least
            # useful version of itself. Replaying real logs turned up regions with one
            # usable measurement in ninety tiles, and the summary said only "licensed:
            # false".
            "unlicensed_reason": self._unlicensed_reason(),
        }
        if self._fit is not None:
            out.update(
                {
                    "z0_um": round(self._fit.z0, 3),
                    "tilt_x_um_per_mm": round(self._fit.ax, 4),
                    "tilt_y_um_per_mm": round(self._fit.ay, 4),
                    "inlier_rms_um": round(self._fit.inlier_rms_um, 3),
                    "inliers": self._fit.n_inliers,
                    "spread_major_mm": round(self._fit.spread_mm[0], 3),
                    "spread_minor_mm": round(self._fit.spread_mm[1], 3),
                    "required_spread_mm": round(self.required_spread_mm(), 3),
                }
            )
        return out


def _design(xy: np.ndarray, centre: Tuple[float, float]) -> np.ndarray:
    """Plane design matrix in mm about ``centre``, which keeps it conditioned.

    Stage coordinates run to tens of thousands of microns, so fitting in raw um makes
    the normal equations badly scaled and the tilt terms numerically mushy.
    """
    x = (xy[:, 0] - centre[0]) / 1000.0
    y = (xy[:, 1] - centre[1]) / 1000.0
    return np.column_stack([np.ones_like(x), x, y])


def _spread_mm(xy: np.ndarray) -> Tuple[float, float]:
    """Peak-to-peak spread, in mm, along the points' two principal directions.

    The smaller value is the one that matters: it is how much evidence the fit has for
    the second tilt term. A single raster column returns zero.

    Peak-to-peak, not a standard deviation. An RMS-like measure divides by the point
    count, so adding points to a thin cloud makes the measured spread *shrink* -- which
    had the absurd consequence that a 16-point survey could be refused where the same
    region's 9-point survey was licensed. More evidence must never license less.
    """
    if len(xy) < 2:
        return (0.0, 0.0)
    centred = (xy - xy.mean(axis=0)) / 1000.0
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return (0.0, 0.0)
    projected = centred @ vt.T
    major = float(np.ptp(projected[:, 0]))
    minor = float(np.ptp(projected[:, 1])) if projected.shape[1] > 1 else 0.0
    return (major, minor)


def _ransac_plane(
    pts: np.ndarray,
    band_um: float,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> Optional[_Fit]:
    """Fit a plane that a minority of bad points cannot drag off."""
    centre = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    A = _design(pts[:, :2], centre)
    z = pts[:, 2]
    n = len(pts)
    sample_size = min(_RANSAC_SAMPLE, n)

    best_inliers = None
    for _ in range(_RANSAC_ITERATIONS):
        idx = rng.choice(n, sample_size, replace=False)
        try:
            coef, *_ = np.linalg.lstsq(A[idx], z[idx], rcond=None)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(coef)):
            continue
        inliers = np.abs(z - A @ coef) < band_um
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 3:
        # Every hypothesis was degenerate (collinear sample, or fewer than three
        # distinct points). Fall back to plain least squares so the caller at least
        # gets a surface, and let the RMS gate decide whether it is usable.
        try:
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        except np.linalg.LinAlgError:
            logger.debug("Focus surface: plane fit failed outright on %d points", n)
            return None
        resid = z - A @ coef
        return _Fit(
            centre,
            float(coef[0]),
            float(coef[1]),
            float(coef[2]),
            float(resid.std()),
            int(n),
            int(n),
        )

    coef, *_ = np.linalg.lstsq(A[best_inliers], z[best_inliers], rcond=None)
    resid = z - A @ coef
    inliers = np.abs(resid) < band_um
    if inliers.sum() < 3:
        inliers = best_inliers
    rms = float(np.sqrt(np.mean(resid[inliers] ** 2)))
    return _Fit(
        centre,
        float(coef[0]),
        float(coef[1]),
        float(coef[2]),
        rms,
        int(inliers.sum()),
        int(n),
        _spread_mm(pts[inliers, :2]),
    )


def parse_mode(raw: Optional[str], logger: Optional[logging.Logger] = None) -> str:
    """Normalise a mode string, defaulting to off rather than guessing."""
    if raw is None:
        return MODE_OFF
    mode = str(raw).strip().lower()
    if mode in VALID_MODES:
        return mode
    if logger:
        logger.warning(
            "Unknown focus-surface mode '%s'; valid modes are %s. Using '%s'.",
            raw,
            ", ".join(VALID_MODES),
            MODE_OFF,
        )
    return MODE_OFF
