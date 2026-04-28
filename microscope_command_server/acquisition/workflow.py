"""Acquisition workflow and microscope-side operations for the command server.

This module contains the acquisition logic and helpers that interact with the
microscope hardware, separated from the socket server/transport logic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional, Dict, Any
from pathlib import Path
import shutil
import logging
import yaml

import numpy as np
from scipy.spatial.distance import cdist as _cdist_scipy

from microscope_control.hardware import Position
from microscope_control.hardware.pycromanager import PycromanagerHardware
from microscope_control.autofocus.core import AutofocusUtils
from microscope_command_server.acquisition.tiles import TileConfigUtils
from microscope_command_server.modality import get_config as get_modality_config
from microscope_imageprocessing.io.writer import ome_tiff_writer
from microscope_imageprocessing.correction.background import BackgroundCorrectionUtils
from microscope_command_server.acquisition.timepoint_scheduler import TimepointScheduler
import shlex
import skimage.filters
from concurrent.futures import ThreadPoolExecutor, Future

logger = logging.getLogger(__name__)


# Modalities where every tile is expected to fill the detector's dynamic range.
# Under-exposure on these is a calibration symptom (stale WB, wrong detector
# profile, lamp moved); on other modalities (fluorescence, laser scanning)
# dim tiles are normal/desired and warning would be noise. Mirrors the Java
# ModalityHandler.expectsUniformBrightness() default-false contract.
_UNIFORM_BRIGHT_MODALITY_PREFIXES = ("ppm", "brightfield", "bf")


def _modality_expects_uniform_brightness(modality: str) -> bool:
    """Return True iff the modality expects every tile to fill the dynamic range.

    The match is prefix-based (case-insensitive) so config-side variants like
    ``ppm_10x``, ``ppm_20x``, ``Brightfield_20x``, ``bf_40x`` all resolve.
    Fluorescence, multiphoton, SHG, and laser-scanning modalities return
    False -- a dim tile on those is normal (rare cell types, sparse signal).
    """
    if not modality:
        return False
    lowered = modality.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _UNIFORM_BRIGHT_MODALITY_PREFIXES)


# Saturation role classification: which angles in a multi-angle modality are
# expected to be bright (saturation OK) vs faint (saturation is a real defect).
# Mirrors qupath.ext.qpsc.modality.ModalityHandler.classifyAngleSaturation()
# on the Java side -- both must use the same |angle-90|<2 tolerance for PPM
# uncrossed detection so the role labels agree.
SATURATION_ROLE_LOW = "signal_low"        # faint signal expected; saturation is bad
SATURATION_ROLE_HIGH = "signal_high"      # bright by design (PPM uncrossed); saturation OK
SATURATION_ROLE_NORMAL = "signal_normal"  # ordinary tile; saturation is bad

_PPM_UNCROSSED_TOLERANCE_DEG = 2.0


def _saturation_role_for(modality: str, angle_deg: float) -> str:
    """Classify how to interpret saturation on a tile captured at this angle.

    For PPM (and prefix variants), the uncrossed angle (~90 deg) is intentionally
    bright and saturation there is normal/expected; the crossed (0) and small
    polarisation angles (+/-7) are low-signal and saturation there is a real
    calibration defect. Other modalities return SATURATION_ROLE_NORMAL.

    Stays in lock-step with Java PPMModalityHandler.classifyAngleSaturation().
    """
    if not modality:
        return SATURATION_ROLE_NORMAL
    lowered = modality.strip().lower()
    if lowered.startswith("ppm"):
        if abs(abs(angle_deg) - 90.0) < _PPM_UNCROSSED_TOLERANCE_DEG:
            return SATURATION_ROLE_HIGH
        return SATURATION_ROLE_LOW
    return SATURATION_ROLE_NORMAL


# Underexposure thresholds (p99 below these flags the tile as underexposed).
# uint8 / 255 ratio, applied to whichever dtype max is in use. Uniform-bright
# modalities (PPM, brightfield) use a stricter threshold because every tile
# is supposed to fill the dynamic range; sparse modalities (fluorescence,
# laser scanning) tolerate dim tiles.
_UNDEREXPOSED_P99_RATIO_UNIFORM = 60.0 / 255.0   # PPM, brightfield: <p99=60/255
_UNDEREXPOSED_P99_RATIO_SPARSE = 30.0 / 255.0    # fluorescence, LSM: <p99=30/255


def _compute_tile_stats(image) -> dict:
    """Per-channel percentile + mean + std stats for a tile image.

    Returns a flat dict with keys like ``p1_R``, ``p99_R``, ``mean_R``,
    ``std_R``, ``dynamic_range_R`` (and similarly for G/B; or ``_gray`` for
    monochrome). Empty dict on None or unexpected shape. Computed once per
    snapped image and accumulated across angles by the caller.
    """
    if image is None:
        return {}
    out = {}
    try:
        if image.ndim == 2:
            ch_keys = [("gray", image)]
        elif image.ndim == 3 and image.shape[2] >= 3:
            ch_keys = [("R", image[:, :, 0]), ("G", image[:, :, 1]), ("B", image[:, :, 2])]
        else:
            return {}
        for label, ch in ch_keys:
            p1 = float(np.percentile(ch, 1))
            p99 = float(np.percentile(ch, 99))
            out[f"p1_{label}"] = round(p1, 1)
            out[f"p99_{label}"] = round(p99, 1)
            out[f"mean_{label}"] = round(float(ch.mean()), 1)
            out[f"std_{label}"] = round(float(ch.std()), 1)
            out[f"dynamic_range_{label}"] = round(p99 - p1, 1)
    except Exception:
        return {}
    return out


def _accumulate_tile_stats(accum: dict, sample: dict) -> None:
    """Aggregate per-image stats into a per-tile worst/best/mean accumulator.

    For p1 keep the smallest seen, for p99 keep the largest, for mean and
    std keep a running average across angles. Accumulator dict is mutated
    in place.
    """
    if not sample:
        return
    counts = accum.setdefault("__counts__", {})
    for key, val in sample.items():
        if key.startswith("p1_"):
            accum[key] = min(accum.get(key, val), val) if key in accum else val
        elif key.startswith("p99_") or key.startswith("dynamic_range_"):
            accum[key] = max(accum.get(key, val), val) if key in accum else val
        elif key.startswith("mean_") or key.startswith("std_"):
            n = counts.get(key, 0)
            prev = accum.get(key, 0.0)
            accum[key] = round((prev * n + val) / (n + 1), 1)
            counts[key] = n + 1


def _stats_underexposed(stats: dict, modality: str) -> bool:
    """Return True when the tile's p99 sits below the modality's underexposure threshold.

    Uses the strict 60/255 threshold for modalities that expect every tile
    to fill the dynamic range (PPM, brightfield) and the relaxed 30/255
    threshold otherwise. Detects the data range from p99 magnitude (>4096
    = uint16, scaled to 16-bit max).
    """
    if not stats:
        return False
    # Pick the worst (lowest) p99 across channels as the canonical brightness signal.
    p99_keys = [k for k in stats.keys() if k.startswith("p99_")]
    if not p99_keys:
        return False
    p99_min = min(stats[k] for k in p99_keys)
    # Scale threshold to data range
    is_uint16 = p99_min > 4096 or any(stats[k] > 4096 for k in p99_keys)
    if _modality_expects_uniform_brightness(modality):
        ratio = _UNDEREXPOSED_P99_RATIO_UNIFORM
    else:
        ratio = _UNDEREXPOSED_P99_RATIO_SPARSE
    threshold = ratio * (65535 if is_uint16 else 255)
    return p99_min < threshold


def _saturation_threshold(image) -> int:
    """Derive near-saturation threshold from image dtype.

    Returns a value just below the dtype maximum to catch near-saturation
    where the channel mean is pulled down by a few non-clipped pixels.
    """
    if image.dtype == np.uint16:
        return 64000
    return 250  # uint8


def _get_worst_channel_saturation(image) -> float:
    """Return the worst per-channel saturation percentage.

    Works for both RGB (H,W,3) and monochrome (H,W) images.
    Uses dtype-aware saturation threshold.
    Returns 0.0 for None images.
    """
    if image is None:
        return 0.0
    sat_thresh = _saturation_threshold(image)
    total_px = image.shape[0] * image.shape[1]
    worst = 0.0
    if image.ndim == 2:
        # Monochrome
        sat_pct = 100.0 * int(np.sum(image >= sat_thresh)) / total_px
        worst = sat_pct
    elif image.ndim == 3 and image.shape[2] >= 3:
        # RGB
        for c in range(min(3, image.shape[2])):
            sat_pct = 100.0 * int(np.sum(image[:, :, c] >= sat_thresh)) / total_px
            worst = max(worst, sat_pct)
    return worst


def _check_saturation(image, angle_name, log, threshold_pct=1.0):
    """Check per-channel saturation and warn if above threshold.

    Supports both RGB (H,W,3) and monochrome (H,W) images with
    dtype-aware saturation thresholds (250 for uint8, 64000 for uint16).

    Args:
        image: Image array (H,W) for monochrome or (H,W,3) for RGB, or None
        angle_name: Label for the image (e.g., angle or "background")
        log: Logger instance
        threshold_pct: Saturation warning threshold as percentage (default 1.0%)

    Returns:
        Dict with per-channel saturation percentages.
        RGB: {'R': float, 'G': float, 'B': float}
        Mono: {'Gray': float}
        None if the image is invalid.
    """
    if image is None:
        return None
    sat_thresh = _saturation_threshold(image)
    total_px = image.shape[0] * image.shape[1]
    result = {}

    if image.ndim == 2:
        # Monochrome image
        sat_count = int(np.sum(image >= sat_thresh))
        sat_pct = 100.0 * sat_count / total_px
        result["Gray"] = sat_pct
        if sat_pct > threshold_pct:
            log.warning(
                f"SATURATION WARNING [{angle_name}]: {sat_pct:.1f}% saturated "
                f"pixels ({sat_count}/{total_px}, threshold >= {sat_thresh})"
            )
    elif image.ndim == 3 and image.shape[2] >= 3:
        # RGB image
        for i, ch in enumerate(["R", "G", "B"]):
            sat_count = int(np.sum(image[:, :, i] >= sat_thresh))
            sat_pct = 100.0 * sat_count / total_px
            result[ch] = sat_pct
            if sat_pct > threshold_pct:
                log.warning(
                    f"SATURATION WARNING [{angle_name}]: {ch} channel has "
                    f"{sat_pct:.1f}% saturated pixels ({sat_count}/{total_px}, "
                    f"threshold >= {sat_thresh})"
                )
    else:
        return None

    return result


class SaturationMonitor:
    """Tracks per-angle saturation across tiles and enforces limits.

    Birefringence angles (small off-crossed angles like +/-7 deg) should have
    zero saturation -- saturated tissue images cannot be corrected and produce
    tiling artifacts. If saturation is detected on the initial tiles, the
    acquisition should abort early rather than waste hours producing unusable data.

    The uncrossed angle (90 deg) inherently has high brightness and some
    background saturation is acceptable. Warnings at this angle are rate-limited
    to avoid flooding the log.
    """

    # Angles within this range of 90 are considered uncrossed
    UNCROSSED_TOLERANCE = 2.0
    # Default: abort if worst channel exceeds this on birefringence angles.
    # Configurable via acquisition_settings.saturation_abort_threshold_pct in YAML.
    BIREF_ABORT_THRESHOLD_PCT = 10.0
    # Number of initial tiles to monitor before deciding to abort
    MONITORING_WINDOW = 3
    # For uncrossed angle, only log every Nth saturated tile
    UNCROSSED_LOG_INTERVAL = 50

    def __init__(self, angles, logger=None,
                 biref_abort_threshold_pct=None,
                 monitoring_window=None):
        """Initialize the saturation monitor.

        Args:
            angles: List of acquisition angles in degrees
            logger: Logger instance
            biref_abort_threshold_pct: Override default abort threshold for
                birefringence angles (percentage of pixels saturated)
            monitoring_window: Override default number of initial tiles to check
        """
        self._log = logger or logging.getLogger(__name__)
        self._angles = list(angles)

        self._biref_threshold = (
            biref_abort_threshold_pct
            if biref_abort_threshold_pct is not None
            else self.BIREF_ABORT_THRESHOLD_PCT
        )
        self._window = (
            monitoring_window
            if monitoring_window is not None
            else self.MONITORING_WINDOW
        )

        # Per-angle tracking
        self._tile_count = {a: 0 for a in self._angles}
        self._saturated_tile_count = {a: 0 for a in self._angles}
        self._worst_seen = {a: 0.0 for a in self._angles}
        self._total_warnings_suppressed = {a: 0 for a in self._angles}
        self._aborted = False
        self._abort_reason = ""

        # Per-tile detail records for saturated tiles
        # Each entry: {filename, angle, pos_idx, r_pct, g_pct, b_pct, stage_x, stage_y, stage_z}
        self._saturated_tiles = []

    def _is_uncrossed(self, angle: float) -> bool:
        """Check if angle is the uncrossed position (near 90 deg)."""
        return abs(abs(angle) - 90.0) < self.UNCROSSED_TOLERANCE

    def check_tile(self, sat_result, angle, tile_idx, filename,
                   stage_x=None, stage_y=None, stage_z=None):
        """Record saturation for a tile and determine if acquisition should abort.

        Args:
            sat_result: Dict from _check_saturation ({'R': pct, 'G': pct, 'B': pct})
                       or None if image was invalid
            angle: Acquisition angle in degrees
            tile_idx: Position index (0-based)
            filename: Tile filename for logging
            stage_x: Optional stage X position in microns (for saturation report)
            stage_y: Optional stage Y position in microns (for saturation report)
            stage_z: Optional stage Z position in microns (for saturation report)

        Returns:
            True if acquisition should abort due to excessive saturation,
            False to continue.
        """
        if sat_result is None or self._aborted:
            return self._aborted

        worst = max(sat_result.values())
        self._tile_count[angle] = self._tile_count.get(angle, 0) + 1
        if worst > 1.0:
            self._saturated_tile_count[angle] = (
                self._saturated_tile_count.get(angle, 0) + 1
            )
            # Record detail for saturated tile
            self._saturated_tiles.append({
                "filename": filename,
                "angle": angle,
                "pos_idx": tile_idx,
                "r_pct": round(sat_result.get("R", 0.0), 1),
                "g_pct": round(sat_result.get("G", 0.0), 1),
                "b_pct": round(sat_result.get("B", 0.0), 1),
                "worst_pct": round(worst, 1),
                "stage_x": round(stage_x, 2) if stage_x is not None else None,
                "stage_y": round(stage_y, 2) if stage_y is not None else None,
                "stage_z": round(stage_z, 2) if stage_z is not None else None,
            })
        self._worst_seen[angle] = max(self._worst_seen.get(angle, 0.0), worst)

        if self._is_uncrossed(angle):
            return self._handle_uncrossed(sat_result, angle, tile_idx, filename)
        else:
            return self._handle_birefringence(sat_result, angle, tile_idx, filename)

    def _handle_birefringence(self, sat_result, angle, tile_idx, filename):
        """Handle saturation for birefringence angles -- abort on excessive saturation."""
        worst = max(sat_result.values())
        tile_num = self._tile_count[angle]

        if worst > self._biref_threshold and tile_num <= self._window:
            self._log.error(
                f"SATURATION ABORT: tile {filename} at {angle} deg has "
                f"{worst:.1f}% saturation (threshold: {self._biref_threshold:.1f}%). "
                f"Tile {tile_num}/{self._window} in monitoring window. "
                f"Per-channel: R={sat_result['R']:.1f}%, "
                f"G={sat_result['G']:.1f}%, B={sat_result['B']:.1f}%"
            )
            # Check if ALL tiles in the window so far have exceeded threshold
            if self._saturated_tile_count[angle] >= tile_num:
                if tile_num >= self._window:
                    self._aborted = True
                    self._abort_reason = (
                        f"All {self._window} initial tiles at {angle} deg exceeded "
                        f"{self._biref_threshold:.1f}% saturation. "
                        f"Worst: {self._worst_seen[angle]:.1f}%. "
                        f"The white balance target intensity is too high for this tissue. "
                        f"To fix: open White Balance Calibration and lower the Target "
                        f"Intensity setting, then recalibrate. "
                        f"Collect new background images before re-acquiring."
                    )
                    self._log.error(f"=== ACQUISITION ABORTED: {self._abort_reason} ===")
                    return True
                else:
                    self._log.warning(
                        f"Saturation at {angle} deg: {tile_num}/{self._window} "
                        f"monitoring tiles saturated -- will abort if all "
                        f"{self._window} initial tiles are saturated"
                    )
        return False

    def _handle_uncrossed(self, sat_result, angle, tile_idx, filename):
        """Handle saturation for uncrossed angle -- rate-limit warnings."""
        worst = max(sat_result.values())
        if worst <= 1.0:
            return False

        count = self._saturated_tile_count[angle]
        # Log first occurrence, then every Nth
        if count == 1:
            self._log.info(
                f"Saturation at {angle} deg (uncrossed) is expected for bright "
                f"backgrounds. Further warnings will be logged every "
                f"{self.UNCROSSED_LOG_INTERVAL} tiles."
            )
        elif count % self.UNCROSSED_LOG_INTERVAL != 0:
            self._total_warnings_suppressed[angle] = (
                self._total_warnings_suppressed.get(angle, 0) + 1
            )
            return False
        # Log periodic update
        self._log.warning(
            f"Saturation at {angle} deg (uncrossed): {count} of "
            f"{self._tile_count[angle]} tiles saturated so far "
            f"(worst: {self._worst_seen[angle]:.1f}%, "
            f"current: R={sat_result['R']:.1f}%, "
            f"G={sat_result['G']:.1f}%, B={sat_result['B']:.1f}%)"
        )
        return False

    def log_summary(self):
        """Log a final summary of saturation across all angles."""
        any_saturation = False
        for angle in self._angles:
            total = self._tile_count.get(angle, 0)
            saturated = self._saturated_tile_count.get(angle, 0)
            if saturated > 0:
                any_saturation = True
                suppressed = self._total_warnings_suppressed.get(angle, 0)
                label = "uncrossed" if self._is_uncrossed(angle) else "birefringence"
                msg = (
                    f"Saturation summary for {angle} deg ({label}): "
                    f"{saturated}/{total} tiles had saturation >1%, "
                    f"worst channel: {self._worst_seen[angle]:.1f}%"
                )
                if suppressed > 0:
                    msg += f" ({suppressed} warnings suppressed)"
                self._log.info(msg)
        if not any_saturation:
            self._log.info("Saturation summary: no saturation detected at any angle")

    def should_suppress_warnings(self, angle: float) -> bool:
        """Return True if per-tile SATURATION WARNING logs should be suppressed.

        For uncrossed angles: suppress after the first tile (monitor handles
        rate-limited logging instead). For birefringence angles: never suppress.
        """
        if not self._is_uncrossed(angle):
            return False
        return self._tile_count.get(angle, 0) >= 1

    def write_saturation_report(self, output_path) -> str:
        """Write per-tile saturation details to a JSON file.

        Args:
            output_path: Acquisition output directory (Path or str)

        Returns:
            Path to the written report file, or None if no saturated tiles
        """
        import json

        if not self._saturated_tiles:
            return None

        output_path = Path(output_path) if isinstance(output_path, str) else output_path
        report_path = output_path / "saturation_report.json"

        report = {
            "summary": {
                angle: {
                    "total_tiles": self._tile_count.get(angle, 0),
                    "saturated_tiles": self._saturated_tile_count.get(angle, 0),
                    "worst_pct": round(self._worst_seen.get(angle, 0.0), 1),
                    "is_uncrossed": self._is_uncrossed(angle),
                }
                for angle in self._angles
            },
            "saturated_tiles": self._saturated_tiles,
        }

        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            self._log.info(
                f"Wrote saturation report with {len(self._saturated_tiles)} "
                f"saturated tile entries to {report_path}"
            )
            return str(report_path)
        except Exception as e:
            self._log.error(f"Failed to write saturation report: {e}")
            return None

    def get_summary_string(self) -> str:
        """Return a compact saturation summary for protocol transmission.

        Format: semicolon-separated entries per angle:
          angle:saturated/total:worst_pct;angle:saturated/total:worst_pct
        Example: "7.0:3/2404:21.6;-7.0:108/2404:49.6;90.0:0/2404:0.0"

        Returns empty string if no saturation data available.
        """
        if not self._angles:
            return ""
        parts = []
        for angle in self._angles:
            total = self._tile_count.get(angle, 0)
            saturated = self._saturated_tile_count.get(angle, 0)
            worst = self._worst_seen.get(angle, 0.0)
            parts.append(f"{angle}:{saturated}/{total}:{worst:.1f}")
        return ";".join(parts)

    @property
    def has_saturation(self) -> bool:
        """Return True if any angle had saturation >1%."""
        return any(self._saturated_tile_count.get(a, 0) > 0 for a in self._angles)

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def abort_reason(self) -> str:
        return self._abort_reason


def load_jai_calibration_from_imageprocessing(
    config_path: Path,
    per_angle: bool = False,
    modality: str = "ppm",
    objective: str = None,
    detector: str = None,
    logger=None,
) -> Optional[Dict[str, Any]]:
    """
    Load JAI white balance calibration from imageprocessing YAML.

    The calibration data is stored in the imaging_profiles section:
    imaging_profiles.{modality}.{objective}.{detector}.exposures_ms.{angle}.{r,g,b}

    Args:
        config_path: Path to the main config file (config_PPM.yml)
                    - imageprocessing file is derived from this
        per_angle: If True, load PPM per-angle calibration with R,G,B values
                  If False, load simple calibration (single exposure)
        modality: Modality name (e.g., "ppm", "brightfield")
        objective: Objective ID (e.g., "LOCI_OBJECTIVE_OLYMPUS_20X_POL_001")
        detector: Detector ID (e.g., "LOCI_DETECTOR_JAI_001")
        logger: Optional logger instance

    Returns:
        Dictionary with calibration data or None if not found.
        For PPM mode: {'angles': {'positive': {'exposures_ms': {'r': x, 'g': y, 'b': z}}, ...}}
    """
    config_path = Path(config_path)

    # Derive imageprocessing file path from the config file name.
    # Naming convention: config_PPM.yml -> imageprocessing_PPM.yml
    # The imageprocessing file stores calibrated per-channel exposure/gain
    # values that were computed by JAIWhiteBalanceCalibrator. This separation
    # keeps hardware config (config_*.yml) independent from calibration
    # results (imageprocessing_*.yml) so recalibration doesn't modify the
    # main config and calibration data can be updated independently.
    config_name = config_path.stem  # e.g., "config_PPM"
    if config_name.startswith("config_"):
        microscope_name = config_name[7:]  # e.g., "PPM"
        imageprocessing_name = f"imageprocessing_{microscope_name}.yml"
    else:
        imageprocessing_name = f"imageprocessing_{config_name}.yml"

    imageprocessing_path = config_path.parent / imageprocessing_name

    if not imageprocessing_path.exists():
        if logger:
            logger.info(f"No imageprocessing config found at {imageprocessing_path}")
        return None

    if not objective or not detector:
        if logger:
            logger.warning("Objective or detector not specified for calibration lookup")
        return None

    try:
        with open(imageprocessing_path, "r") as f:
            ip_data = yaml.safe_load(f) or {}

        # Navigate to imaging_profiles.{modality}.{objective}.{detector}
        imaging_profiles = ip_data.get("imaging_profiles", {})
        modality_profiles = imaging_profiles.get(modality, {})
        objective_profiles = modality_profiles.get(objective, {})
        detector_profile = objective_profiles.get(detector, {})

        if not detector_profile:
            if logger:
                logger.info(f"No profile found for {modality}/{objective}/{detector}")
            return None

        # Freshness check: if wb_last_modified is more than 14 days old, warn.
        # The 2026-04-27 silent-first-detector incident left a 12-day-stale JAI
        # 10x calibration in place while a fresh WB run wrote to the wrong
        # detector profile -- without this warning, the only signal was very
        # dim acquired tiles. Warn loudly so the user can recalibrate (or
        # confirm the slot is intentionally frozen).
        wb_last_modified = detector_profile.get("wb_last_modified")
        simple_wb_section = detector_profile.get("simple_wb", {}) or {}
        simple_last = simple_wb_section.get("last_calibrated") if isinstance(simple_wb_section, dict) else None
        if logger:
            try:
                from datetime import datetime, timedelta
                now = datetime.now()
                if isinstance(wb_last_modified, str) and wb_last_modified:
                    try:
                        ts = datetime.fromisoformat(wb_last_modified)
                        age = now - ts
                        if age > timedelta(days=14):
                            logger.warning(
                                "WB calibration for %s/%s/%s is %d days old "
                                "(wb_last_modified=%s). Acquired images may not match "
                                "calibration targets; consider recalibrating.",
                                modality, objective, detector, age.days, wb_last_modified,
                            )
                    except ValueError:
                        logger.debug("Could not parse wb_last_modified=%r", wb_last_modified)
                # Drift check between the per-angle wb_last_modified and the
                # simple_wb.last_calibrated. If they disagree by more than 7
                # days, the user has likely run one calibration mode but not
                # the other -- BG correction with the older mode will silently
                # use stale data.
                if (
                    isinstance(wb_last_modified, str)
                    and isinstance(simple_last, str)
                    and wb_last_modified and simple_last
                ):
                    try:
                        ts_pa = datetime.fromisoformat(wb_last_modified)
                        ts_simple = datetime.fromisoformat(simple_last)
                        drift = abs(ts_pa - ts_simple)
                        if drift > timedelta(days=7):
                            logger.warning(
                                "WB mode drift on %s/%s/%s: per-angle wb_last_modified=%s "
                                "vs simple_wb.last_calibrated=%s (%d-day drift). "
                                "BG correction with the older mode may use stale exposures.",
                                modality, objective, detector,
                                wb_last_modified, simple_last, drift.days,
                            )
                    except ValueError:
                        pass
            except Exception:
                # Freshness logging is best-effort -- never block the load
                pass

        exposures_ms = detector_profile.get("exposures_ms", {})
        gains = detector_profile.get("gains", {})

        if not exposures_ms:
            if logger:
                logger.info(f"No exposures_ms found in profile for {modality}/{objective}/{detector}")
            return None

        if per_angle:
            # PPM requires per-angle calibration because each polarizer angle
            # produces dramatically different light levels (crossed is very dim,
            # uncrossed is very bright). A single set of per-channel exposures
            # cannot achieve white balance across all angles, so each angle
            # gets its own independently calibrated R,G,B exposure and gain set.
            #
            # Build per-angle calibration structure from exposures_ms
            # Expected format in YAML:
            #   exposures_ms:
            #     positive: {all: 800, r: 750, g: 800, b: 850}
            #     negative: {all: 800, r: 750, g: 800, b: 850}
            #     ...
            angles_data = {}
            for angle_name, exp_data in exposures_ms.items():
                if isinstance(exp_data, dict) and 'r' in exp_data and 'g' in exp_data and 'b' in exp_data:
                    angles_data[angle_name] = {
                        'exposures_ms': {
                            'r': exp_data.get('r', 50.0),
                            'g': exp_data.get('g', 50.0),
                            'b': exp_data.get('b', 50.0),
                        }
                    }
                    # Add gains if available
                    if gains and angle_name in gains:
                        angle_gains = gains[angle_name]
                        # Support both new format (analog_red/analog_blue) and
                        # old format (r/g/b) for backward compatibility
                        if 'analog_red' in angle_gains:
                            angles_data[angle_name]['gains'] = angle_gains
                        elif 'r' in angle_gains:
                            # Old format: map r -> analog_red, b -> analog_blue
                            angles_data[angle_name]['gains'] = {
                                'unified_gain': angle_gains.get('unified_gain', 1.0),
                                'analog_red': angle_gains.get('r', 1.0),
                                'analog_blue': angle_gains.get('b', 1.0),
                                'wb_method': angle_gains.get('wb_method', 'unknown'),
                            }
                            if logger:
                                logger.info(
                                    f"Mapped old gain format (r/g/b) to new "
                                    f"(analog_red/analog_blue) for angle {angle_name}"
                                )
                        else:
                            angles_data[angle_name]['gains'] = angle_gains

            if angles_data:
                if logger:
                    logger.info(f"Loaded JAI PPM calibration for angles: {list(angles_data.keys())}")
                return {'angles': angles_data}
            else:
                if logger:
                    logger.info("No per-channel (r,g,b) exposure data found in exposures_ms")
                return None
        else:
            # Simple mode - return first available exposure settings
            if logger:
                logger.info(f"Loaded JAI simple calibration from {modality}/{objective}/{detector}")
            return {'exposures_ms': exposures_ms, 'gains': gains}

    except Exception as e:
        if logger:
            logger.warning(f"Failed to load JAI calibration from {imageprocessing_path}: {e}")
        return None


def load_simple_wb_from_imageprocessing(
    config_path: Path,
    modality: str = "ppm",
    objective: str = None,
    detector: str = None,
    logger=None,
) -> Optional[Dict[str, Any]]:
    """
    Load simple WB (Mode 2) pre-computed per-angle scaled exposures from YAML.

    The simple_wb section stores per-angle exposures that preserve the uncrossed
    R:G:B ratio while scaling intensity for each angle. This data is written
    by background collection in simple WB mode.

    Args:
        config_path: Path to the main config file (config_PPM.yml)
        modality: Modality name (e.g., "ppm")
        objective: Objective ID
        detector: Detector ID
        logger: Optional logger instance

    Returns:
        Dictionary with simple_wb data or None if not found.
        Format: {
            'base_angle': 'uncrossed',
            'base_exposures_ms': {'r': x, 'g': y, 'b': z},
            'angles': {
                'uncrossed': {'scale': 1.0, 'unified_gain': 1.0, 'r': x, 'g': y, 'b': z},
                'positive': {'scale': s, 'unified_gain': g, 'r': x*s, 'g': y*s, 'b': z*s},
                ...
            }
        }
    """
    config_path = Path(config_path)

    config_name = config_path.stem
    if config_name.startswith("config_"):
        microscope_name = config_name[7:]
        imageprocessing_name = f"imageprocessing_{microscope_name}.yml"
    else:
        imageprocessing_name = f"imageprocessing_{config_name}.yml"

    imageprocessing_path = config_path.parent / imageprocessing_name

    if not imageprocessing_path.exists():
        if logger:
            logger.debug(f"No imageprocessing config for simple_wb at {imageprocessing_path}")
        return None

    if not objective or not detector:
        if logger:
            logger.debug("Objective or detector not specified for simple_wb lookup")
        return None

    try:
        with open(imageprocessing_path, "r") as f:
            ip_data = yaml.safe_load(f) or {}

        detector_profile = (
            ip_data
            .get("imaging_profiles", {})
            .get(modality, {})
            .get(objective, {})
            .get(detector, {})
        )

        simple_wb = detector_profile.get("simple_wb")
        if simple_wb and "angles" in simple_wb:
            if logger:
                logger.info(
                    f"Loaded simple_wb data: base_angle={simple_wb.get('base_angle')}, "
                    f"angles={list(simple_wb['angles'].keys())}"
                )
            return simple_wb
        else:
            if logger:
                logger.debug("No simple_wb section found in detector profile")
            return None

    except Exception as e:
        if logger:
            logger.warning(f"Failed to load simple_wb from {imageprocessing_path}: {e}")
        return None


def get_interpolated_calibration_for_angle(
    angle: float,
    angles_cal: Dict[str, Dict],
    logger=None,
) -> Optional[Dict[str, Any]]:
    """
    Get calibration data for an angle, interpolating unified gain if necessary.

    For PPM birefringence sweep angles (-10 to +10 degrees), this function:
    1. Returns exact calibration for angles matching 0, +/-7, or 90 degrees (within 1 deg)
    2. For other angles, interpolates the unified gain between calibrated angles
    3. Uses per-channel exposures from the nearest birefringence angle (+/-7 deg)
       since color balance characteristics are similar across the sweep range

    The interpolation is based on the relationship between polarizer angle and
    light transmission. For angles between 0 (crossed) and +/-7 (birefringence),
    the unified gain is linearly interpolated.

    Args:
        angle: Rotation angle in degrees
        angles_cal: Dictionary of calibration data keyed by angle name
                   ('crossed', 'uncrossed', 'positive', 'negative')
        logger: Optional logger instance

    Returns:
        Dictionary with 'exposures_ms' and 'gains' keys, or None if interpolation fails.
        If interpolation was used, includes 'interpolated': True flag.
    """
    # Calibrated angle reference points
    ANGLE_TO_NAME = {
        0.0: "crossed",
        7.0: "positive",
        -7.0: "negative",
        90.0: "uncrossed",
    }

    # Check for exact match (within 0.2 degree tolerance)
    # Reduced from 1.0 to minimize discontinuity at interpolation boundary
    for cal_angle, name in ANGLE_TO_NAME.items():
        if abs(angle - cal_angle) < 0.2:
            cal_data = angles_cal.get(name)
            if cal_data:
                if logger:
                    logger.debug(f"Using exact calibration '{name}' for angle {angle:.2f} deg")
                return cal_data
            else:
                if logger:
                    logger.warning(f"Calibration '{name}' not found for angle {angle:.2f} deg")
                return None

    # Interpolate for birefringence sweep range (-10 to +10 degrees)
    if -15.0 <= angle <= 15.0:
        # Determine interpolation endpoints based on angle sign
        if angle > 0:
            # Positive angles: interpolate between crossed (0) and positive (7)
            low_name, high_name = "crossed", "positive"
            low_angle, high_angle = 0.0, 7.0
        else:
            # Negative angles: interpolate between crossed (0) and negative (-7)
            low_name, high_name = "crossed", "negative"
            low_angle, high_angle = 0.0, -7.0

        low_cal = angles_cal.get(low_name)
        high_cal = angles_cal.get(high_name)

        if not low_cal or not high_cal:
            if logger:
                logger.warning(
                    f"Cannot interpolate for {angle:.2f} deg - "
                    f"missing '{low_name}' or '{high_name}' calibration"
                )
            return None

        # Calculate interpolation factor based on absolute angle
        # factor = 0 at crossed (0 deg), factor = 1 at birefringence (+/-7 deg)
        abs_angle = abs(angle)
        if abs_angle <= 7.0:
            factor = abs_angle / 7.0
        else:
            # Beyond +/-7 deg, extrapolate linearly but cap the factor
            # to avoid extreme values. Use factor > 1 for extrapolation.
            factor = abs_angle / 7.0
            if factor > 1.5:
                factor = 1.5  # Cap extrapolation at 1.5x

        # Interpolate unified gain
        low_gains = low_cal.get("gains", {})
        high_gains = high_cal.get("gains", {})

        low_unified = low_gains.get("unified_gain", 1.0)
        high_unified = high_gains.get("unified_gain", 1.0)
        interp_unified = low_unified + factor * (high_unified - low_unified)

        # Interpolate per-channel exposures for smooth color transition
        # Previously this used high_cal directly, causing a sharp discontinuity
        # at the boundary between exact match (0 deg) and interpolation (>1 deg)
        low_exp = low_cal.get("exposures_ms", {})
        high_exp = high_cal.get("exposures_ms", {})

        interp_exposures = {
            "r": low_exp.get("r", 50.0) + factor * (high_exp.get("r", 50.0) - low_exp.get("r", 50.0)),
            "g": low_exp.get("g", 50.0) + factor * (high_exp.get("g", 50.0) - low_exp.get("g", 50.0)),
            "b": low_exp.get("b", 50.0) + factor * (high_exp.get("b", 50.0) - low_exp.get("b", 50.0)),
        }

        # Interpolate analog R/B gains as well for consistency
        low_analog_red = low_gains.get("analog_red", 1.0)
        low_analog_blue = low_gains.get("analog_blue", 1.0)
        high_analog_red = high_gains.get("analog_red", 1.0)
        high_analog_blue = high_gains.get("analog_blue", 1.0)

        interp_analog_red = low_analog_red + factor * (high_analog_red - low_analog_red)
        interp_analog_blue = low_analog_blue + factor * (high_analog_blue - low_analog_blue)

        # Build interpolated calibration result
        result = {
            "exposures_ms": interp_exposures,
            "gains": {
                "unified_gain": interp_unified,
                "analog_red": interp_analog_red,
                "analog_blue": interp_analog_blue,
            },
            "interpolated": True,
            "interpolation_factor": factor,
        }

        if logger:
            logger.info(
                f"Interpolated calibration for {angle:.2f} deg: "
                f"unified_gain={interp_unified:.3f} "
                f"(factor={factor:.3f}, between {low_name}={low_unified:.3f} and {high_name}={high_unified:.3f})"
            )

        return result

    # Angle outside supported range
    if logger:
        logger.warning(f"Angle {angle:.2f} deg outside calibration range (-15 to +15 or 90)")
    return None


def apply_jai_calibration_for_angle(
    hardware: "PycromanagerHardware",
    jai_calibration: Dict[str, Any],
    angle: float,
    per_angle: bool = False,
    logger=None,
    exposure_scale: float = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Apply JAI white balance calibration settings before image capture.

    This enables individual exposure mode and sets per-channel exposures
    based on the calibration data. Optionally scales exposures while
    preserving color balance ratios.

    Args:
        hardware: PycromanagerHardware instance
        jai_calibration: Calibration data from load_jai_calibration_from_imageprocessing()
        angle: Current rotation angle (used for PPM mode to select angle-specific settings)
        per_angle: If True, use angle-specific settings from jai_ppm
                  If False, use single settings from jai_simple
        logger: Optional logger instance
        exposure_scale: If provided, scale all per-channel exposures by this factor.
                       This preserves color balance ratios while adjusting intensity.
                       Example: if calibrated exposures are R=21ms, G=25ms, B=19ms
                       and exposure_scale=2.0, applied exposures will be
                       R=42ms, G=50ms, B=38ms.

    Returns:
        Tuple of (success, exposure_info):
        - success: True if settings were applied, False otherwise
        - exposure_info: Dict with 'exposures_ms' (r,g,b values actually applied),
                        'base_exposure' (mean of calibrated values before scaling),
                        'scale_applied' (the scale factor used, or 1.0 if none).
                        None if application failed.
    """
    # Only applies to cameras with per-channel exposure control (e.g. JAI 3-CCD)
    try:
        camera_name = hardware.get_camera_name()
        if not hardware.camera.supports_per_channel_exposure():
            if logger:
                logger.debug(f"Per-channel calibration skipped - camera is {camera_name}")
            return False, None
    except Exception:
        return False, None

    try:
        # Get calibration settings for this angle
        if per_angle:
            # Use interpolation-aware calibration lookup for PPM angles.
            # This supports exact matches at 0, +/-7, 90 degrees, and interpolates
            # the unified gain for intermediate angles in the birefringence sweep range.
            if "angles" not in jai_calibration:
                if logger:
                    logger.warning(f"No PPM calibration angles found")
                return False, None

            angle_cal = get_interpolated_calibration_for_angle(
                angle=angle,
                angles_cal=jai_calibration["angles"],
                logger=logger,
            )
            if not angle_cal:
                # Interpolation failed - no calibration available for this angle
                return False, None

            exposures = angle_cal.get("exposures_ms", {})
            gains = angle_cal.get("gains", {})
        else:
            # Simple mode - use same settings for all angles
            exposures = jai_calibration.get("exposures_ms", {})
            gains = jai_calibration.get("gains", {})

        if not exposures:
            if logger:
                logger.debug("No exposure data in JAI calibration")
            return False, None

        # Get base per-channel exposures from calibration
        base_exp_r = exposures.get("r", 50.0)
        base_exp_g = exposures.get("g", 50.0)
        base_exp_b = exposures.get("b", 50.0)

        # Calculate base exposure (mean of calibrated values)
        base_exposure = (base_exp_r + base_exp_g + base_exp_b) / 3.0

        # Apply scaling if provided (preserves color balance ratios)
        scale_applied = exposure_scale if exposure_scale is not None else 1.0
        exp_r = base_exp_r * scale_applied
        exp_g = base_exp_g * scale_applied
        exp_b = base_exp_b * scale_applied

        # Build exposure info for return value
        exposure_info = {
            "exposures_ms": {"r": exp_r, "g": exp_g, "b": exp_b},
            "base_exposure": base_exposure,
            "scale_applied": scale_applied,
        }

        # Extract gains (new format: unified_gain, analog_red, analog_blue)
        unified_gain = gains.get("unified_gain", 1.0)
        analog_red = gains.get("analog_red", None)
        analog_blue = gains.get("analog_blue", None)

        # Backward compatibility: if old r/g/b keys found, map them
        if analog_red is None and "r" in gains:
            analog_red = gains.get("r", 1.0)
            analog_blue = gains.get("b", 1.0)

        if analog_red is None:
            analog_red = 1.0
        if analog_blue is None:
            analog_blue = 1.0

        # Apply mode + exposures + gains atomically via Camera.apply_settings().
        # This stops streaming once (if needed), applies all settings, and avoids
        # partial-state windows between individual set_*() calls.
        hardware.camera.apply_settings(
            exposures={"r": exp_r, "g": exp_g, "b": exp_b},
            unified_gain=unified_gain,
            analog_red=analog_red,
            analog_blue=analog_blue,
            individual_exposure=True,
        )

        if logger:
            if scale_applied != 1.0:
                logger.info(
                    "Applied JAI calibration for angle %s: "
                    "R=%.1fms, G=%.1fms, B=%.1fms (scale=%.2fx) "
                    "| Gain: %.3f, aR=%.3f, aB=%.3f",
                    angle, exp_r, exp_g, exp_b, scale_applied,
                    unified_gain, analog_red, analog_blue,
                )
            else:
                logger.info(
                    "Applied JAI calibration for angle %s: "
                    "R=%.1fms, G=%.1fms, B=%.1fms "
                    "| Gain: %.3f, aR=%.3f, aB=%.3f",
                    angle, exp_r, exp_g, exp_b,
                    unified_gain, analog_red, analog_blue,
                )

        return True, exposure_info

    except Exception as e:
        if logger:
            logger.warning(f"Failed to apply JAI calibration: {e}")
        return False, None


def load_and_apply_white_balance_settings(
    hardware: PycromanagerHardware,
    calibration_folder: str,
    detector: str,
    modality: str,
    objective: str,
    logger=None,
) -> bool:
    """
    Load and apply white balance calibration settings for JAI camera.

    Looks for white_balance_settings.yml in the calibration folder structure:
    {calibration_folder}/{detector}/{modality}/{objective}/white_balance_settings.yml

    Args:
        hardware: PycromanagerHardware instance
        calibration_folder: Base path for calibration data
        detector: Detector ID (e.g., "JAI")
        modality: Modality name (e.g., "ppm", "brightfield")
        objective: Objective ID (e.g., "20x")
        logger: Optional logger instance

    Returns:
        True if settings were loaded and applied, False otherwise
    """
    # Strict camera name check: only applies to exactly "JAICamera".
    # Unlike the other WB functions that check for "JAI" substring (which
    # would match hypothetical "JAI_Fusion" etc.), this function uses an
    # exact match because the white_balance_settings.yml file format is
    # tightly coupled to JAICameraProperties.apply_white_balance_settings()
    # Only applies to cameras with per-channel exposure (e.g. JAI 3-CCD).
    camera_name = hardware.get_camera_name()
    if not hardware.camera.supports_per_channel_exposure():
        if logger:
            logger.debug(f"White balance loading skipped - camera {camera_name} has no per-channel exposure")
        return False

    try:
        # Build path to settings file
        wb_settings_path = (
            Path(calibration_folder)
            / detector
            / modality
            / objective
            / "white_balance_settings.yml"
        )

        if not wb_settings_path.exists():
            if logger:
                logger.info(f"No white balance settings found at {wb_settings_path}")
            return False

        # Load and apply settings via camera properties
        success = hardware.camera.properties.apply_white_balance_settings(str(wb_settings_path))

        if success and logger:
            logger.info(f"Applied white balance settings from {wb_settings_path}")

        return success

    except Exception as e:
        if logger:
            logger.warning(f"Failed to load white balance settings: {e}")
        return False


class _TileWritePool:
    """Bounded background thread pool for overlapping TIFF writes with hardware ops.

    Pending writes are drained at each autofocus check position, which runs every
    N tiles (N >= 1, enforced by af_n_tiles validation). This bounds the write
    queue to at most N tiles worth of images between drain points.

    During autofocus (1-5s of in-memory hardware ops), pending writes execute in
    parallel with full disk bandwidth, so drain() after AF is usually instant.
    """

    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tile_io"
        )
        self._pending: List[Future] = []
        self._failed_count = 0

    def submit(self, fn, *args, **kwargs):
        """Submit a write operation to the background pool."""
        future = self._executor.submit(fn, *args, **kwargs)
        self._pending.append(future)

    def drain(self, timeout_per_write: float = 30.0):
        """Wait for all pending writes to complete. Log errors but don't raise.

        Args:
            timeout_per_write: Seconds to wait per individual write before warning.

        Returns:
            Number of failed writes since last drain.
        """
        failed = 0
        for future in self._pending:
            try:
                future.result(timeout=timeout_per_write)
            except Exception as e:
                failed += 1
                self._failed_count += 1
                logger.error(f"Background TIFF write failed: {e}")
        self._pending.clear()
        return failed

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def total_failed(self) -> int:
        return self._failed_count

    def shutdown(self):
        """Drain remaining writes and shut down the pool."""
        if self._pending:
            logger.info(
                f"Shutting down write pool, draining {len(self._pending)} pending writes..."
            )
            self.drain(timeout_per_write=60.0)
        self._executor.shutdown(wait=True)


def log_timing(logger, operation_name, start_time):
    """Log elapsed time for an operation in milliseconds.

    Args:
        logger: Logger instance
        operation_name: Description of the operation
        start_time: Start time from time.perf_counter()

    Returns:
        Current time for use as next start_time
    """
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(f"  [TIMING] {operation_name}: {elapsed_ms:.1f}ms")
    return time.perf_counter()


def write_acquisition_metadata(output_path, params, tile_measurements,
                               sat_monitor, camera_settings=None):
    """Write acquisition_metadata.json alongside tile data.

    Captures acquisition parameters, camera settings, autofocus config,
    timing summary, and software versions in a single structured file
    for FAIR-compliant metadata export.

    This is Phase 1 of the 4DN-BINA-OME metadata integration -- sidecar
    JSON alongside the stitched output. Failure here never blocks acquisition.

    Args:
        output_path: Path to the acquisition output directory
        params: The parsed acquisition parameters dict
        tile_measurements: List of per-tile measurement dicts
        sat_monitor: SaturationMonitor instance for summary stats
        camera_settings: Optional dict of per-angle camera settings
                        (exposures, gains) captured during acquisition
    """
    _log = logging.getLogger(__name__)
    try:
        from datetime import datetime, timezone
        import platform

        # --- Acquisition parameters ---
        acq_params = {
            "scan_type": params.get("scan_type"),
            "objective": params.get("objective"),
            "detector": params.get("detector"),
            "pixel_size_um": params.get("pixel_size"),
            "modality": params.get("modality"),
            "region_name": params.get("region_name"),
            "sample_label": params.get("sample_label"),
        }

        # Angles / channels
        if params.get("angles"):
            acq_params["angles_deg"] = params["angles"]
        if params.get("exposures"):
            acq_params["exposures_ms"] = params["exposures"]
        if params.get("channels_str"):
            acq_params["channels"] = params["channels_str"]
        if params.get("channel_exposures_str"):
            acq_params["channel_exposures_ms"] = params["channel_exposures_str"]

        # White balance
        acq_params["wb_mode"] = params.get("wb_mode", "off")
        acq_params["wb_enabled"] = params.get("white_balance_enabled", False)
        acq_params["wb_per_angle"] = params.get("white_balance_per_angle", False)

        # Background correction
        acq_params["bg_correction_enabled"] = params.get(
            "background_correction_enabled", False)
        acq_params["bg_correction_method"] = params.get(
            "background_correction_method")

        # Z-stack
        if params.get("z_stack"):
            acq_params["z_stack"] = {
                "enabled": True,
                "z_start_um": params.get("z_start"),
                "z_end_um": params.get("z_end"),
                "z_step_um": params.get("z_step"),
                "z_pixel_size_um": params.get("z_pixel_size_um"),
                "projection": params.get("z_projection"),
            }

        # Processing pipeline
        acq_params["processing_pipeline"] = params.get("processing_pipeline")
        acq_params["save_raw"] = params.get("save_raw", False)

        # --- Autofocus config ---
        af_config = {
            "strategy": params.get("af_strategy"),
            "n_tiles": params.get("autofocus_tiles"),
            "n_steps": params.get("autofocus_steps"),
            "range_um": params.get("autofocus_range"),
        }

        # --- Camera settings (per-angle exposures/gains if available) ---
        camera_section = camera_settings if camera_settings else {}

        # --- Timing summary ---
        timing = {}
        if tile_measurements:
            all_times = [m["tile_time_ms"] for m in tile_measurements]
            af_tiles = [m for m in tile_measurements if m.get("af_performed")]
            non_af = [m for m in tile_measurements if not m.get("af_performed")]
            total_wall_s = sum(all_times) / 1000
            timing = {
                "total_tiles": len(all_times),
                "total_wall_time_s": round(total_wall_s, 1),
                "avg_tile_ms": round(sum(all_times) / len(all_times), 1),
                "avg_af_tile_ms": round(
                    sum(m["tile_time_ms"] for m in af_tiles) / len(af_tiles), 1
                ) if af_tiles else None,
                "avg_non_af_tile_ms": round(
                    sum(m["tile_time_ms"] for m in non_af) / len(non_af), 1
                ) if non_af else None,
                "af_tile_count": len(af_tiles),
                "tiles_per_hour": round(
                    len(all_times) / (total_wall_s / 3600), 0
                ) if total_wall_s > 0 else None,
            }

        # --- AF quality summary ---
        af_summary = {}
        if tile_measurements:
            af_performed = [m for m in tile_measurements if m.get("af_performed")]
            af_failed = [m for m in tile_measurements if m.get("af_failed")]
            af_types = {}
            for m in tile_measurements:
                t = m.get("af_type", "none")
                af_types[t] = af_types.get(t, 0) + 1
            af_summary = {
                "total_af_events": len(af_performed),
                "af_failures": len(af_failed),
                "af_type_counts": af_types,
            }
            # Drift statistics from sweep drift checks
            drifts = [m.get("af_drift_um") for m in tile_measurements
                      if m.get("af_drift_um") is not None]
            if drifts:
                af_summary["drift_um_min"] = round(min(drifts), 2)
                af_summary["drift_um_max"] = round(max(drifts), 2)
                af_summary["drift_um_mean"] = round(sum(drifts) / len(drifts), 2)

        # --- Saturation summary ---
        sat_section = {}
        try:
            sat_section = {
                "summary": sat_monitor.get_summary_string(),
            }
        except Exception:
            pass

        # --- Software versions ---
        versions = {}
        try:
            import microscope_command_server
            versions["microscope_command_server"] = getattr(
                microscope_command_server, "__version__", "unknown")
        except Exception:
            pass
        try:
            import microscope_control
            versions["microscope_control"] = getattr(
                microscope_control, "__version__", "unknown")
        except Exception:
            pass
        try:
            import microscope_imageprocessing
            versions["microscope_imageprocessing"] = getattr(
                microscope_imageprocessing, "__version__", "unknown")
        except Exception:
            pass
        versions["python"] = platform.python_version()

        # --- Assemble ---
        metadata = {
            "schema_version": "1.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "acquisition": acq_params,
            "autofocus": af_config,
            "autofocus_quality": af_summary,
            "camera_settings": camera_section,
            "timing": timing,
            "saturation": sat_section,
            "software_versions": versions,
        }

        # Strip None values for cleaner output
        def strip_none(d):
            if isinstance(d, dict):
                return {k: strip_none(v) for k, v in d.items() if v is not None}
            if isinstance(d, list):
                return [strip_none(i) for i in d]
            return d

        metadata = strip_none(metadata)

        meta_path = output_path / "acquisition_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        _log.info("Wrote acquisition metadata to %s", meta_path)

    except Exception as e:
        _log.warning("Failed to write acquisition metadata (non-fatal): %s", e)


def autofocus_with_manual_fallback(
    hardware: PycromanagerHardware,
    logger,
    request_manual_focus: Optional[Callable[[int], str]] = None,
    max_retries: int = 3,
    fallback_z: Optional[float] = None,
    **autofocus_kwargs
):
    """
    Perform autofocus with manual focus fallback on failure.

    If autofocus fails (returns failure dict), prompts user for manual focus
    and retries. Shows dialog even on last attempt with retry button disabled.

    After manual focus completes, the stage XY position is restored to the
    original tile position (user may have moved XY to find tissue for focusing).

    Args:
        hardware: PycromanagerHardware instance
        logger: Logger instance
        request_manual_focus: Optional callback to request manual focus from user.
                             Callback receives retries_remaining (int) and returns
                             user choice: "retry", "skip", or "cancel".
                             If None, will raise exception on autofocus failure.
        max_retries: Maximum number of retry attempts after manual focus
        fallback_z: Z position to use when user skips (e.g., the Z-hint from
                    the tilt model). If None, uses the failed AF's attempted_z.
        **autofocus_kwargs: Arguments to pass to hardware.autofocus(). Pass
                            ``edge_retries=N`` to enable transparent
                            peak-at-edge widen-and-shift retries inside the
                            hardware layer before this wrapper sees a failure.

    Returns:
        float: Best focus Z position on success

    Raises:
        RuntimeError: If user cancels acquisition or no callback provided
    """
    # Capture original XY position before any autofocus attempts
    # User may move XY during manual focus dialog to find tissue
    original_pos = hardware.get_current_position()
    original_x, original_y = original_pos.x, original_pos.y

    for attempt in range(max_retries):
        result = hardware.autofocus(**autofocus_kwargs)

        # Check if autofocus succeeded (returns float) or failed (returns dict)
        if isinstance(result, float):
            # Success!
            return result
        elif isinstance(result, dict) and result.get('success') == False:
            # Autofocus failed with primary metric
            logger.warning(f"Autofocus failed (attempt {attempt + 1}/{max_retries}): {result['message']}")
            logger.warning(f"  Quality score: {result['quality_score']:.2f}, "
                          f"Prominence: {result['peak_prominence']:.2f}")

            # Note: p98_p2 fallback and (optionally) peak-at-edge widen-shift
            # retries are handled inside hardware.autofocus() itself.

            if request_manual_focus is not None:
                # Always show dialog, even on last attempt
                retries_remaining = max_retries - attempt - 1
                logger.info(f"Requesting manual focus from user (retries remaining: {retries_remaining})...")
                user_choice = request_manual_focus(retries_remaining)  # Pass retries info

                if user_choice == "skip":
                    # User chose to skip -- use fallback_z (hint from tilt model)
                    # if available, otherwise use the failed AF's attempted_z.
                    # The tilt model hint is usually much more accurate than a
                    # failed autofocus result.
                    if fallback_z is not None:
                        skip_z = fallback_z
                        logger.info(f"Using fallback Z (hint): {skip_z:.2f} um "
                                   f"(failed AF attempted: {result['attempted_z']:.2f} um)")
                        hardware.move_to_position(Position(z=skip_z))
                    else:
                        skip_z = result['attempted_z']
                        logger.info(f"Using failed AF position: {skip_z:.2f} um (no fallback hint)")
                    current_pos = hardware.get_current_position()
                    if abs(current_pos.x - original_x) > 1.0 or abs(current_pos.y - original_y) > 1.0:
                        logger.info(f"Restoring XY position after manual focus: "
                                   f"({current_pos.x:.1f}, {current_pos.y:.1f}) -> ({original_x:.1f}, {original_y:.1f})")
                        restore_pos = Position(original_x, original_y, skip_z)
                        hardware.move_to_position(restore_pos)
                    logger.info("User chose to skip autofocus")
                    return skip_z
                elif user_choice == "cancel":
                    # User chose to cancel acquisition. Raise the sentinel the
                    # outer acquisition loop already catches so state transitions
                    # to CANCELLED (not FAILED). A plain RuntimeError would be
                    # caught by the generic `except Exception` and mapped to
                    # FAILED, which is semantically wrong and doesn't close the
                    # QuPath progress dialog the same way.
                    logger.warning("User cancelled acquisition during manual focus")
                    raise _AcquisitionCancelled("User cancelled acquisition during manual focus")
                elif user_choice == "retry":
                    if retries_remaining > 0:
                        # IMPORTANT: Run autofocus at CURRENT position (where user found tissue)
                        # BEFORE restoring XY. This ensures autofocus runs where there's tissue.
                        logger.info(f"Running autofocus at current position (where user found tissue)...")
                        retry_result = hardware.autofocus(**autofocus_kwargs)

                        if isinstance(retry_result, float):
                            # Autofocus succeeded at current position - restore XY with new Z
                            logger.info(f"Autofocus succeeded at current position: Z={retry_result:.2f} um")
                            current_pos = hardware.get_current_position()
                            if abs(current_pos.x - original_x) > 1.0 or abs(current_pos.y - original_y) > 1.0:
                                logger.info(f"Restoring XY position: "
                                           f"({current_pos.x:.1f}, {current_pos.y:.1f}) -> ({original_x:.1f}, {original_y:.1f})")
                                # Position already imported at top of file (line 18)
                                restore_pos = Position(original_x, original_y, retry_result)
                                hardware.move_to_position(restore_pos)
                            return retry_result
                        else:
                            # Autofocus failed again - continue to next attempt
                            logger.warning(f"Autofocus retry failed: {retry_result.get('message', 'unknown error')}")
                            continue
                    else:
                        # No retries left - shouldn't happen since button should be disabled
                        logger.warning("User chose retry but no retries remaining - using fallback Z")
                        return fallback_z if fallback_z is not None else result['attempted_z']
                else:
                    # Unknown choice - default to skip
                    logger.warning(f"Unknown user choice '{user_choice}' - using fallback Z")
                    return fallback_z if fallback_z is not None else result['attempted_z']
            else:
                # No callback provided, raise exception
                raise RuntimeError(
                    f"Autofocus failed: {result['message']}. "
                    f"Quality score: {result['quality_score']:.2f}, "
                    f"Prominence: {result['peak_prominence']:.2f}"
                )
        else:
            # Unexpected return type
            raise RuntimeError(f"Unexpected autofocus return value: {result}")

    # Should never reach here
    raise RuntimeError("Autofocus retry loop exited unexpectedly")


def calculate_luminance_gain(r, g, b):
    """Calculate luminance-based gain from RGB values."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def _merge_device_property_overrides(
    library_props: List[Dict[str, Any]],
    override_props: Any,
) -> List[Dict[str, Any]]:
    """Merges profile-level device_properties overrides into a channel's library
    device_properties list. Match semantics parallel the Java side
    (MicroscopeConfigManager.mergeDevicePropertyOverrides):

    - Match by (device, property) tuple
    - If match found: replace value in place (preserving list order)
    - If no match: append to the end

    Returns a new list if any overrides were applied, or the original list if
    the override list is empty / malformed.
    """
    if not isinstance(override_props, list) or not override_props:
        return list(library_props)
    merged: List[Dict[str, Any]] = [dict(p) for p in library_props]
    for entry in override_props:
        if not isinstance(entry, dict):
            continue
        device = entry.get("device")
        prop = entry.get("property")
        value = entry.get("value")
        if device is None or prop is None or value is None:
            logger.warning(
                "device_properties override has missing device/property/value; skipping: %s",
                entry,
            )
            continue
        match_idx = None
        for i, existing in enumerate(merged):
            if existing.get("device") == device and existing.get("property") == prop:
                match_idx = i
                break
        if match_idx is not None:
            merged[match_idx] = {"device": device, "property": prop, "value": value}
        else:
            merged.append({"device": device, "property": prop, "value": value})
    return merged


def resolve_channel_plan(
    ppm_settings: Dict[str, Any],
    scan_type: str,
    channel_ids: List[str],
    channel_exposures: List[float],
    channel_intensity_overrides: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Resolve the per-channel acquisition plan for a widefield IF tile.

    For each requested channel id, looks up the channel definition in
    ``modalities.<modality>.channels`` (keyed off the acquisition profile's
    ``modality`` field) and returns a plan dict containing the id, exposure,
    mm_setup_presets, and device_properties. Channels not found in the YAML
    are skipped with a warning.

    Profile-level ``channel_overrides.<id>.device_properties`` are merged into
    the library entry here so that per-objective tuning (e.g. BF_IF_10x
    overriding DiaLamp intensity) lands before the hardware state is applied.
    Exposure overrides are typically passed in already-resolved via the
    ``channel_exposures`` argument (Java-side merge), but the channel library
    default is also used as a fallback when the list is shorter than ``channel_ids``.

    The design is intentionally vendor-agnostic: everything is driven by
    Micro-Manager primitives (ConfigGroup presets, device property writes),
    so the same code path supports any multi-channel illumination hardware.
    """
    if not channel_ids:
        return []

    # Resolve modality from scan_type via acquisition_profiles.
    profile_key = None
    acq_profiles = (ppm_settings or {}).get("acquisition_profiles", {}) or {}
    if scan_type in acq_profiles:
        profile_key = scan_type
    else:
        # The Java side may append a counter suffix like "_1"; try stripping it.
        base = "_".join(scan_type.split("_")[:-1]) if "_" in scan_type else scan_type
        if base in acq_profiles:
            profile_key = base
    modality_name = None
    profile_overrides: Dict[str, Dict[str, Any]] = {}
    if profile_key is not None:
        profile_cfg = acq_profiles[profile_key] or {}
        modality_name = profile_cfg.get("modality")
        raw_overrides = profile_cfg.get("channel_overrides") or {}
        if isinstance(raw_overrides, dict):
            profile_overrides = raw_overrides
    if not modality_name:
        return []

    modalities = (ppm_settings or {}).get("modalities", {}) or {}
    modality_cfg = modalities.get(modality_name, {}) or {}
    library = modality_cfg.get("channels", []) or []
    by_id: Dict[str, Dict[str, Any]] = {}
    for entry in library:
        if isinstance(entry, dict) and entry.get("id"):
            by_id[entry["id"]] = entry

    # Zip ids with exposures, falling back to the library default when the
    # --channel-exposures list is shorter than --channels.
    plan: List[Dict[str, Any]] = []
    runtime_overrides = channel_intensity_overrides or {}
    for i, cid in enumerate(channel_ids):
        entry = by_id.get(cid)
        if entry is None:
            continue  # unknown channel id
        exposure_ms = (
            channel_exposures[i]
            if i < len(channel_exposures) and channel_exposures[i] > 0
            else float(entry.get("exposure_ms", 0) or 0)
        )
        # Apply profile-level device_properties overrides for this channel.
        library_props = entry.get("device_properties", []) or []
        channel_override = profile_overrides.get(cid, {}) or {}
        merged_props = _merge_device_property_overrides(
            library_props, channel_override.get("device_properties"))

        # Runtime per-channel intensity override from --channel-intensities.
        # Looks up the channel's intensity_property pointer in the YAML and
        # replaces the matching (device, property) entry in merged_props with
        # the override value. Falls back to append if the library didn't
        # already include that pair. Missing intensity_property => warn and
        # ignore the override (the channel has no declared intensity knob).
        if cid in runtime_overrides:
            override_value = runtime_overrides[cid]
            intensity_ref = entry.get("intensity_property") or {}
            ip_device = intensity_ref.get("device") if isinstance(intensity_ref, dict) else None
            ip_property = intensity_ref.get("property") if isinstance(intensity_ref, dict) else None
            if ip_device and ip_property:
                merged_props = list(merged_props)  # defensive copy
                replaced = False
                for idx, prop in enumerate(merged_props):
                    if (
                        isinstance(prop, dict)
                        and prop.get("device") == ip_device
                        and prop.get("property") == ip_property
                    ):
                        merged_props[idx] = {
                            "device": ip_device,
                            "property": ip_property,
                            "value": str(override_value),
                        }
                        replaced = True
                        break
                if not replaced:
                    merged_props.append(
                        {
                            "device": ip_device,
                            "property": ip_property,
                            "value": str(override_value),
                        }
                    )
            else:
                logger.warning(
                    "Channel '%s' received runtime intensity override %s but has no "
                    "intensity_property declared in the YAML; ignoring override",
                    cid,
                    override_value,
                )

        plan.append(
            {
                "id": cid,
                "display_name": entry.get("display_name", cid),
                "exposure_ms": exposure_ms,
                "mm_setup_presets": entry.get("mm_setup_presets", []) or [],
                "device_properties": merged_props,
                "settle_ms": entry.get("settle_ms"),
            }
        )
    return plan


def apply_channel_hardware_state(
    hardware,
    channel_plan_entry: Dict[str, Any],
    logger_: logging.Logger,
    preset_cache: Optional[Dict[str, str]] = None,
) -> None:
    """Apply one channel's Micro-Manager state before snapping.

    Applies ``mm_setup_presets`` via ``core.set_config(group, preset)`` then
    ``device_properties`` via ``core.set_property(device, property, value)``.
    After each batch, waits for the affected devices to report "not busy" so
    rapid back-to-back channel transitions can't race the snap. An optional
    ``settle_ms`` field on the channel entry adds a dumb sleep fallback for
    hardware whose ``isBusy()`` reports complete too early (some filter
    turrets, reflector wheels, serial LED controllers).

    When ``preset_cache`` is a dict, it is used as a memoization table keyed
    by ConfigGroup name. If a preset request matches the last-applied value
    for that group, the ``set_config`` + ``wait_for_config`` pair is skipped
    entirely. On OWS3 every channel targets the same Filter Turret preset, so
    caching saves ~300-600 ms per channel per tile. The caller owns the cache
    lifetime -- reset the dict between acquisitions.

    Both primitives are generic Micro-Manager -- no vendor-specific knowledge.
    """
    core = getattr(hardware, "core", None)
    if core is None:
        logger_.warning("Hardware has no .core attribute; cannot apply channel state")
        return

    # Pycromanager's MMCore wrapper exposes Python-idiomatic snake_case method
    # names (set_config, set_property, wait_for_config, wait_for_device), NOT
    # the Java camelCase (setConfig, setProperty, waitForConfig, waitForDevice).
    # The first dryrun used camelCase and every preset / property write silently
    # raised AttributeError and got caught by the warning handler, so the DLED
    # was never actually switched between channels -- every tile captured the
    # same hardware state and looked like "only one channel was collected".
    for preset in channel_plan_entry.get("mm_setup_presets", []) or []:
        if not isinstance(preset, dict):
            continue
        group = preset.get("group")
        preset_name = preset.get("preset")
        if not (group and preset_name):
            continue
        group_str = str(group)
        preset_str = str(preset_name)
        # Skip the MMCore roundtrip if we just applied this exact preset
        # for the same group. Cache lifetime is owned by the caller.
        if preset_cache is not None and preset_cache.get(group_str) == preset_str:
            continue
        try:
            core.set_config(group_str, preset_str)
            core.wait_for_config(group_str, preset_str)
            if preset_cache is not None:
                preset_cache[group_str] = preset_str
        except Exception as e:
            logger_.warning(
                "Failed to apply channel preset %s=%s: %s", group, preset_name, e
            )
            continue

    property_devices: set = set()
    for prop in channel_plan_entry.get("device_properties", []) or []:
        if not isinstance(prop, dict):
            continue
        device = prop.get("device")
        property_name = prop.get("property")
        value = prop.get("value")
        if not (device and property_name and value is not None):
            continue
        try:
            core.set_property(str(device), str(property_name), str(value))
            property_devices.add(str(device))
        except Exception as e:
            logger_.warning(
                "Failed to set channel property %s.%s=%s: %s",
                device,
                property_name,
                value,
                e,
            )

    # Wait for each touched device to idle. Back-to-back set_property calls on
    # CoolLED/DLED/Lumencor can outrun the serial-command settle so the camera
    # integrates before the intensity is actually applied. wait_for_device is
    # always safe -- it no-ops on devices that don't implement isBusy().
    for dev in property_devices:
        try:
            core.wait_for_device(dev)
        except Exception as e:
            logger_.debug("wait_for_device(%s) raised (non-fatal): %s", dev, e)

    # Optional dumb-sleep fallback for hardware whose isBusy() reports early.
    settle_ms = channel_plan_entry.get("settle_ms")
    if isinstance(settle_ms, (int, float)) and settle_ms > 0:
        time.sleep(float(settle_ms) / 1000.0)


def parse_angles_exposures(angles_str, exposures_str=None) -> Tuple[List[float], List[int]]:
    """Parse angle and exposure strings from various formats."""
    angles: List[float] = []
    exposures: List[int] = []

    # Parse angles
    if isinstance(angles_str, list):
        angles = angles_str
    elif isinstance(angles_str, str):
        angles_str = angles_str.strip("()")
        if "," in angles_str:
            angles = [float(x.strip()) for x in angles_str.split(",")]
        elif angles_str:
            angles = [float(x) for x in angles_str.split()]

    # Parse exposures if provided
    if exposures_str:
        if isinstance(exposures_str, list):
            exposures = exposures_str
        elif isinstance(exposures_str, str):
            exposures_str = exposures_str.strip("()")
            if "," in exposures_str:
                exposures = [float(x.strip()) for x in exposures_str.split(",")]
            elif exposures_str:
                exposures = [float(x) for x in exposures_str.split()]

    # Default exposures if not provided
    if not exposures and angles:
        for angle in angles:
            if angle == 90.0:
                exposures.append(10.0)
            elif angle == 0.0:
                exposures.append(800.0)
            else:
                exposures.append(500.0)

    return angles, exposures


def parse_acquisition_message(message: str) -> dict:
    """Parse acquisition message in flag-based format."""
    # Remove END_MARKER if present
    message = message.replace(" END_MARKER", "").replace("END_MARKER", "").strip()

    # Parse flag-based format
    if "--" in message:
        # Parse flag-based format
        params = {}

        # Split by spaces but preserve quoted strings
        try:
            # For Windows compatibility, temporarily replace backslashes
            temp_message = message.replace("\\", "|||BACKSLASH|||")
            parts = shlex.split(temp_message)
            # Restore backslashes
            parts = [part.replace("|||BACKSLASH|||", "\\") for part in parts]
        except Exception:
            # Fallback to simple split if shlex fails
            parts = message.split()

        i = 0
        while i < len(parts):
            if parts[i] == "--yaml" and i + 1 < len(parts):
                params["yaml_file_path"] = parts[i + 1]
                i += 2
            elif parts[i] == "--projects" and i + 1 < len(parts):
                params["projects_folder_path"] = parts[i + 1]
                i += 2
            elif parts[i] == "--sample" and i + 1 < len(parts):
                params["sample_label"] = parts[i + 1]
                i += 2
            elif parts[i] == "--scan-type" and i + 1 < len(parts):
                params["scan_type"] = parts[i + 1]
                i += 2
            elif parts[i] == "--region" and i + 1 < len(parts):
                params["region_name"] = parts[i + 1]
                i += 2
            elif parts[i] == "--angles" and i + 1 < len(parts):
                params["angles_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--exposures" and i + 1 < len(parts):
                params["exposures_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--channels" and i + 1 < len(parts):
                params["channels_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--channel-exposures" and i + 1 < len(parts):
                params["channel_exposures_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--channel-intensities" and i + 1 < len(parts):
                params["channel_intensities_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--focus-channel" and i + 1 < len(parts):
                params["focus_channel"] = parts[i + 1]
                i += 2
            elif parts[i] == "--af-strategy" and i + 1 < len(parts):
                params["af_strategy"] = parts[i + 1]
                i += 2
            elif parts[i] == "--bg-correction" and i + 1 < len(parts):
                params["background_correction_enabled"] = parts[i + 1].lower() == "true"
                i += 2
            elif parts[i] == "--bg-method" and i + 1 < len(parts):
                params["background_correction_method"] = parts[i + 1]
                i += 2
            elif parts[i] == "--bg-folder" and i + 1 < len(parts):
                params["background_folder"] = parts[i + 1]
                i += 2
            elif parts[i] == "--bg-disabled-angles" and i + 1 < len(parts):
                params["background_disabled_angles_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--wb-mode" and i + 1 < len(parts):
                params["wb_mode"] = parts[i + 1].lower()
                i += 2
            elif parts[i] == "--white-balance" and i + 1 < len(parts):
                params["white_balance_enabled"] = parts[i + 1].lower() == "true"
                i += 2
            elif parts[i] == "--wb-per-angle" and i + 1 < len(parts):
                params["white_balance_per_angle"] = parts[i + 1].lower() == "true"
                i += 2
            elif parts[i] == "--objective" and i + 1 < len(parts):
                params["objective"] = parts[i + 1]
                i += 2
            elif parts[i] == "--detector" and i + 1 < len(parts):
                params["detector"] = parts[i + 1]
                i += 2
            elif parts[i] == "--pixel-size" and i + 1 < len(parts):
                params["pixel_size"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--af-tiles" and i + 1 < len(parts):
                params["autofocus_tiles"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--af-steps" and i + 1 < len(parts):
                params["autofocus_steps"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--af-range" and i + 1 < len(parts):
                params["autofocus_range"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--af-disabled":
                params["af_disabled"] = True
                i += 1
            elif parts[i] == "--save-raw" and i + 1 < len(parts):
                params["save_raw"] = parts[i + 1].lower() == "true"
                i += 2
            elif parts[i] == "--processing" and i + 1 < len(parts):
                params["processing_pipeline"] = parts[i + 1]
                i += 2
            elif parts[i] == "--hint-z" and i + 1 < len(parts):
                params["hint_z"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--preferred-af-tile" and i + 1 < len(parts):
                params["preferred_af_tile"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--z-stack":
                params["z_stack"] = True
                i += 1
            elif parts[i] == "--z-start" and i + 1 < len(parts):
                params["z_start"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--z-end" and i + 1 < len(parts):
                params["z_end"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--z-step" and i + 1 < len(parts):
                params["z_step"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--z-pixel-size" and i + 1 < len(parts):
                params["z_pixel_size_um"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--z-projection" and i + 1 < len(parts):
                params["z_projection"] = parts[i + 1]
                i += 2
            # LSM / multiphoton flags
            elif parts[i] == "--laser-power" and i + 1 < len(parts):
                params["laser_power"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--laser-wavelength" and i + 1 < len(parts):
                params["laser_wavelength"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--dwell-time" and i + 1 < len(parts):
                params["dwell_time"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--averaging" and i + 1 < len(parts):
                params["averaging"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--biref-min-intensity" and i + 1 < len(parts):
                params["biref_min_intensity"] = int(parts[i + 1])
                i += 2
            # Time-lapse + output-format flags (Z-stack + time-lapse refactor).
            # Defaults (timepoints=1, interval=0.0, output_format='ome-per-t')
            # preserve pre-refactor single-pass behavior when the Java side
            # omits these flags.
            elif parts[i] == "--timepoints" and i + 1 < len(parts):
                params["n_timepoints"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--interval" and i + 1 < len(parts):
                params["interval_seconds"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--output-format" and i + 1 < len(parts):
                params["output_format"] = parts[i + 1]
                i += 2
            else:
                i += 1

        # Parse angles and exposures if present
        angles, exposures = parse_angles_exposures(
            params.get("angles_str", "()"), params.get("exposures_str", None)
        )
        params["angles"] = angles
        params["exposures"] = exposures

        # Parse per-channel sequence (widefield IF / multi-channel fluorescence).
        # Channel ids are strings, exposures are floats (ms). Mutually exclusive
        # with --angles on the Java side; here we just pass both through so the
        # acquisition loop can choose based on presence.
        channel_ids: List[str] = []
        channel_exposures: List[float] = []
        channels_str = params.get("channels_str", "()")
        if channels_str and channels_str != "()":
            cs = channels_str.strip("()")
            if "," in cs:
                channel_ids = [x.strip() for x in cs.split(",") if x.strip()]
            elif cs:
                channel_ids = [cs.strip()]
        channel_exposures_str = params.get("channel_exposures_str", "()")
        if channel_exposures_str and channel_exposures_str != "()":
            ces = channel_exposures_str.strip("()")
            if "," in ces:
                channel_exposures = [float(x.strip()) for x in ces.split(",") if x.strip()]
            elif ces:
                channel_exposures = [float(ces.strip())]
        params["channels"] = channel_ids
        params["channel_exposures"] = channel_exposures

        # Per-channel intensity overrides: --channel-intensities "(DAPI=25,FITC=30)".
        # Map channel id -> float. Only channels that the user changed away from
        # the YAML default appear here; the acquisition loop falls back to the
        # YAML value for anything missing.
        channel_intensities: Dict[str, float] = {}
        channel_intensities_str = params.get("channel_intensities_str", "()")
        if channel_intensities_str and channel_intensities_str != "()":
            cis = channel_intensities_str.strip("()")
            for pair in cis.split(","):
                pair = pair.strip()
                if not pair or "=" not in pair:
                    continue
                key, _, raw_val = pair.partition("=")
                key = key.strip()
                raw_val = raw_val.strip()
                if not key or not raw_val:
                    continue
                try:
                    channel_intensities[key] = float(raw_val)
                except ValueError:
                    logger.warning(
                        "Ignoring non-numeric --channel-intensities value for %s: %r",
                        key,
                        raw_val,
                    )
        params["channel_intensities"] = channel_intensities

        # Defensive: channel-based acquisition is mutually exclusive with angle-based
        # in the Java emitter, but stale angle fields from an old command can still
        # arrive if a caller drifts. Force the angle branch off when channels are
        # present so the channel path is the single source of truth.
        if channel_ids:
            if params.get("angles"):
                logger.warning(
                    "Both --channels and --angles were supplied; ignoring angles "
                    "(%s) and proceeding with %d channels",
                    params.get("angles"),
                    len(channel_ids),
                )
            params["angles"] = []
            # Keep params["exposures"] alone -- not used by the channel branch.

        # Parse disabled angles for background correction
        disabled_angles = []
        disabled_angles_str = params.get("background_disabled_angles_str", "()")
        if disabled_angles_str and disabled_angles_str != "()":
            disabled_angles_str = disabled_angles_str.strip("()")
            if "," in disabled_angles_str:
                disabled_angles = [float(x.strip()) for x in disabled_angles_str.split(",")]
            elif disabled_angles_str:
                disabled_angles = [float(x) for x in disabled_angles_str.split()]
        params["background_disabled_angles"] = disabled_angles

        # Time-lapse + output-format defaults. Keeping these outside the
        # flag loop so callers that omit --timepoints / --interval / --output-format
        # still get concrete values downstream (single pass, 'ome-per-t').
        params.setdefault("n_timepoints", 1)
        params.setdefault("interval_seconds", 0.0)
        params.setdefault("output_format", "ome-per-t")

        # Validate required parameters
        required = [
            "yaml_file_path",
            "projects_folder_path",
            "sample_label",
            "scan_type",
            "region_name",
        ]
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        return params

    raise ValueError("Invalid acquisition message format - must use flag-based format with '--' parameters")


class _AcquisitionCancelled(Exception):
    """Raised inside extracted functions when is_cancelled() returns True."""


@dataclass
class AcquisitionContext:
    """Bundles all state needed across the acquisition workflow phases.

    Created by _prepare_acquisition(), then passed to all subsequent functions.
    Autofocus and infrastructure fields are populated by _configure_autofocus()
    and _initialize_loop_infrastructure() respectively.
    """

    # -- Core --
    params: Dict[str, Any]
    hardware: PycromanagerHardware
    config_manager: Any
    logger: Any  # logging.Logger
    client_addr: Any
    output_path: Path
    modality: str
    mod_config: Any  # ModalityConfig from get_modality_config()
    ppm_settings: dict

    # -- Tile positions --
    positions: List[Tuple[Any, str]]  # [(Position, filename), ...]
    xy_positions: List[Tuple[float, float]]
    total_images: int = 0

    # -- Background correction --
    background_correction_enabled: bool = False
    background_correction_method: str = "divide"
    background_disabled_angles: list = field(default_factory=list)
    background_images: dict = field(default_factory=dict)
    background_scaling_factors: dict = field(default_factory=dict)
    channel_background_images: dict = field(default_factory=dict)

    # -- White balance --
    white_balance_enabled: bool = True
    white_balance_per_angle: bool = False
    wb_mode: str = "off"
    angles_wb: dict = field(default_factory=dict)
    jai_calibration: Optional[dict] = None
    simple_wb_data: Optional[dict] = None
    camera_awb_gains: dict = field(default_factory=dict)
    is_jai_camera: bool = False
    simple_wb_analog_red: float = 1.0
    simple_wb_analog_blue: float = 1.0

    # -- Z-stack --
    z_stack_enabled: bool = False
    z_offsets: list = field(default_factory=lambda: [0.0])
    projection_fn: Optional[Callable] = None
    save_raw_tiles: bool = False

    # -- Time-lapse + output format --
    # Defaults preserve the pre-refactor single-pass behavior: n_timepoints=1
    # collapses the T-outer loop in _acquisition_workflow to one pass and
    # makes the position loop behave identically to before the refactor.
    n_timepoints: int = 1
    interval_seconds: float = 0.0
    output_format: str = "ome-per-t"
    t0: float = 0.0  # set at loop start (time.monotonic()); 0.0 means "not yet started"

    # -- Autofocus (populated by _configure_autofocus) --
    af_n_tiles: int = 5
    af_n_steps: int = 11
    af_search_range: float = 50.0
    af_interp_strength: float = 100.0
    af_interp_kind: str = "quadratic"
    af_score_metric: Optional[Callable] = None
    af_score_metric_name: str = "normalized_variance"
    af_sweep_range_um: float = 10.0
    af_sweep_n_steps: int = 5
    af_edge_retries: int = 2
    af_gap_index_multiplier: int = 3
    af_gap_spatial_multiplier: float = 2.0
    af_strategy: Any = None
    af_strategy_name: Optional[str] = None
    af_focus_channel: Optional[str] = None
    af_positions: list = field(default_factory=list)
    af_min_distance: float = 0.0
    exposure_90: float = 0.0  # mutable: doubled during brightness checks
    hint_z: Optional[float] = None
    metadata_txt_for_positions: Optional[Path] = None

    # -- Infrastructure (populated by _initialize_loop_infrastructure) --
    sat_monitor: Any = None  # SaturationMonitor
    write_pool: Any = None  # _TileWritePool
    channel_preset_cache: dict = field(default_factory=dict)

    # -- Callbacks --
    update_progress: Optional[Callable] = None
    set_state: Optional[Callable] = None
    is_cancelled: Optional[Callable] = None
    request_manual_focus: Optional[Callable] = None
    request_hardware_error_recovery: Optional[Callable] = None

    # -- Mutable loop state --
    image_count: int = 0
    dynamic_af_positions: set = field(default_factory=set)
    deferred_af_positions: set = field(default_factory=set)
    completed_af_positions: list = field(default_factory=list)
    first_tissue_autofocus_done: bool = False
    last_af_pos_idx: int = -1
    tile_measurements: list = field(default_factory=list)
    tile_measurements_stream: Any = None  # file handle
    stage_positions_collected: list = field(default_factory=list)
    starting_position: Any = None  # Position or None
    channel_consecutive_saturated: dict = field(default_factory=dict)
    progress_warning_fired: bool = False


def _acquisition_workflow(
    message: str,
    client_addr,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
    update_progress: Callable[[int, int], None],
    set_state: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    request_manual_focus: Optional[Callable[[], None]] = None,
    request_hardware_error_recovery: Optional[Callable[[str], str]] = None,
    connection_config_path: Optional[str] = None,
):
    """Execute the main image acquisition workflow with progress and cancellation.

    Args:
        message: Acquisition command message
        client_addr: Client address for logging
        hardware: Hardware interface
        config_manager: Configuration manager
        logger: Logger instance
        update_progress: Callback to update progress (current, total)
        set_state: Callback to set acquisition state
        is_cancelled: Callback to check if cancelled
        request_manual_focus: Optional callback to request manual focus from user
                             when autofocus fails. If None, autofocus failures will
                             raise exceptions as before.
        request_hardware_error_recovery: Optional callback to request hardware error
                             recovery from user. Receives error message string and
                             returns user choice: "retry", "skip", or "cancel".
                             If None, hardware errors will fail the acquisition.
        connection_config_path: Optional path to config from initial CONFIG command,
                               used to warn if ACQUIRE uses different config.
    """
    ctx = None
    try:
        # Phase 1-6: Parse message, load configs, setup BG/WB/Z-stack, create dirs
        ctx = _prepare_acquisition(
            message=message,
            client_addr=client_addr,
            hardware=hardware,
            config_manager=config_manager,
            logger=logger,
            update_progress=update_progress,
            set_state=set_state,
            is_cancelled=is_cancelled,
            request_manual_focus=request_manual_focus,
            request_hardware_error_recovery=request_hardware_error_recovery,
            connection_config_path=connection_config_path,
        )

        # Phase 7: Load AF settings, resolve strategy, compute positions
        _configure_autofocus(ctx)

        # Phase 8: Initial autofocus at first tissue position
        _run_pre_acquisition_autofocus(ctx)

        # Phase 9: Create saturation monitor, write pool, NDJSON stream
        _initialize_loop_infrastructure(ctx)

        # Phase 10: Main acquisition loop (T-outer / position-middle).
        # When n_timepoints == 1 (default) the scheduler is a no-op and
        # the loop collapses to the pre-refactor single-pass behavior.
        ctx.t0 = time.monotonic()
        scheduler = TimepointScheduler(
            t0_monotonic=ctx.t0,
            interval_seconds=ctx.interval_seconds,
            logger=logger,
        )
        for t_idx in range(ctx.n_timepoints):
            if t_idx > 0:
                # Time-lapse pacing. wait_until() returns 0 if overdue
                # (acq_time > interval) with a warning, or after the
                # cancel poll fires.
                if ctx.is_cancelled():
                    logger.warning(f"Acquisition cancelled by client {client_addr}")
                    ctx.set_state("CANCELLED")
                    return
                scheduler.wait_until(t_idx, cancel_event=ctx.is_cancelled)
                if ctx.is_cancelled():
                    logger.warning(f"Acquisition cancelled by client {client_addr}")
                    ctx.set_state("CANCELLED")
                    return
                logger.info(
                    "=== TIMEPOINT %d/%d starting at t0+%.1fs ===",
                    t_idx + 1,
                    ctx.n_timepoints,
                    time.monotonic() - ctx.t0,
                )

            for pos_idx, (pos, filename) in enumerate(ctx.positions):
                if ctx.is_cancelled():
                    logger.warning(f"Acquisition cancelled by client {client_addr}")
                    ctx.set_state("CANCELLED")
                    return

                logger.info(f"Position {pos_idx + 1}/{len(ctx.positions)}: {filename}")
                tile_start = time.perf_counter()

                # AF decision + stage move + autofocus execution
                needs_af, af_type, drift, af_failed, xy_move_pending = (
                    _handle_tile_autofocus(ctx, pos_idx, pos, filename)
                )

                # User chose to skip this tile during hardware error recovery
                if af_type == "skipped":
                    continue

                # Collect stage position for this tile
                current_stage_pos = hardware.get_current_position()
                ctx.stage_positions_collected.append((
                    filename,
                    current_stage_pos.x,
                    current_stage_pos.y,
                    current_stage_pos.z,
                ))

                # Dispatch by modality
                tile_worst_sat = {"R": 0.0, "G": 0.0, "B": 0.0}
                tile_role_sat = {SATURATION_ROLE_LOW: 0.0, SATURATION_ROLE_HIGH: 0.0, SATURATION_ROLE_NORMAL: 0.0}
                tile_stats: dict = {}
                if ctx.params["angles"]:
                    tile_worst_sat, tile_role_sat, tile_stats, xy_move_pending = _acquire_tile_angles(
                        ctx, pos_idx, pos, filename, current_stage_pos, xy_move_pending
                    )
                else:
                    # Wait for non-blocking XY move before single-image/channel snap
                    if xy_move_pending:
                        ctx.hardware.wait_for_xy()

                    # Try channel-based acquisition first
                    channel_plan = resolve_channel_plan(
                        ctx.ppm_settings,
                        ctx.params.get("scan_type", ""),
                        ctx.params.get("channels", []) or [],
                        ctx.params.get("channel_exposures", []) or [],
                        channel_intensity_overrides=ctx.params.get("channel_intensities") or None,
                    )
                    if channel_plan:
                        tile_worst_sat, tile_role_sat, tile_stats = _acquire_tile_channels(ctx, pos, filename, current_stage_pos)
                    else:
                        tile_worst_sat, tile_role_sat, tile_stats = _acquire_tile_single(ctx, pos, filename, current_stage_pos)

                # Record per-tile measurements
                _record_tile_measurement(
                    ctx, pos_idx, filename, tile_start,
                    needs_af, af_type, drift, af_failed,
                    tile_worst_sat, current_stage_pos,
                    tile_role_sat=tile_role_sat,
                    tile_stats=tile_stats,
                )

        # Phase 11: Post-acquisition finalization
        _finalize_acquisition(ctx)

    except _AcquisitionCancelled:
        logger.warning(f"Acquisition cancelled by client {client_addr}")
        set_state("CANCELLED")
    except Exception as e:
        logger.error("=== ACQUISITION FAILED ===")
        logger.error(f"Error: {str(e)}", exc_info=True)
        set_state("FAILED", str(e))
    finally:
        if ctx is not None:
            _cleanup_acquisition(ctx)


##############################################################################
# Extracted acquisition workflow functions
# These were split from _acquisition_workflow() to improve readability and
# testability. Each function receives an AcquisitionContext and operates on it.
##############################################################################


def _finalize_acquisition(ctx: AcquisitionContext) -> None:
    """Post-acquisition: close streams, write metadata files, set COMPLETED state."""
    logger = ctx.logger

    # Close NDJSON stream now that the loop has finished normally
    if ctx.tile_measurements_stream is not None:
        try:
            ctx.tile_measurements_stream.close()
        except Exception as e:
            logger.debug("Error closing tile measurements NDJSON stream: %s", e)
        ctx.tile_measurements_stream = None

    # Drain any remaining background writes before finalizing
    if ctx.write_pool.pending_count > 0:
        t_drain = time.perf_counter()
        n_remaining = ctx.write_pool.pending_count
        logger.info(f"Draining {n_remaining} remaining background writes...")
        failed = ctx.write_pool.drain()
        log_timing(logger, f"Final drain of {n_remaining} writes ({failed} failed)", t_drain)
    if ctx.write_pool.total_failed > 0:
        logger.warning(
            f"Total background write failures during acquisition: {ctx.write_pool.total_failed}"
        )

    # Save device properties
    current_props = ctx.hardware.get_device_properties()
    props_path = ctx.output_path / "MMproperties.txt"
    with open(props_path, "w") as fid:
        from pprint import pprint as dict_printer
        dict_printer(current_props, stream=fid)

    # Write TileConfiguration with stage coordinates including Z
    if ctx.stage_positions_collected:
        TileConfigUtils.write_tileconfig_stage(ctx.output_path, ctx.stage_positions_collected)

    # Write consolidated tile manifest (intended vs actual positions)
    try:
        manifest_path = ctx.output_path / "tile_manifest.csv"
        with open(manifest_path, "w") as mf:
            mf.write("filename,intended_x_um,intended_y_um,actual_x_um,actual_y_um,actual_z_um,dx_um,dy_um\n")
            intended = {fn: (pos.x, pos.y) for pos, fn in ctx.positions}
            for entry in ctx.stage_positions_collected:
                fn, ax, ay, az = entry
                ix, iy = intended.get(fn, (0.0, 0.0))
                mf.write(f"{fn},{ix:.2f},{iy:.2f},{ax:.2f},{ay:.2f},{az:.2f},{ax - ix:.2f},{ay - iy:.2f}\n")
        logger.info(f"Wrote tile manifest: {manifest_path}")
    except Exception as e:
        logger.warning(f"Failed to write tile manifest: {e}")

    # Write saturation report
    ctx.sat_monitor.write_saturation_report(ctx.output_path)

    # Write per-tile measurements JSON (authoritative copy)
    try:
        measurements_path = ctx.output_path / "tile_measurements.json"
        with open(measurements_path, "w") as f:
            json.dump(ctx.tile_measurements, f, indent=2)
        logger.info(f"Wrote tile measurements: {measurements_path}")
    except Exception as e:
        logger.warning(f"Failed to write tile measurements: {e}")

    # Write FAIR/4DN-BINA-OME acquisition metadata
    try:
        write_acquisition_metadata(
            output_path=ctx.output_path,
            params=ctx.params,
            tile_measurements=ctx.tile_measurements,
            sat_monitor=ctx.sat_monitor,
        )
    except Exception as e:
        logger.warning(f"Failed to write acquisition metadata: {e}")

    # Post-acquisition timing report
    all_times = [m["tile_time_ms"] for m in ctx.tile_measurements]
    if all_times:
        af_tiles = [m for m in ctx.tile_measurements if m["af_performed"]]
        non_af_tiles = [m for m in ctx.tile_measurements if not m["af_performed"]]
        total_wall_s = sum(all_times) / 1000
        avg_all = sum(all_times) / len(all_times)
        avg_af = sum(m["tile_time_ms"] for m in af_tiles) / len(af_tiles) if af_tiles else 0
        avg_non_af = sum(m["tile_time_ms"] for m in non_af_tiles) / len(non_af_tiles) if non_af_tiles else 0
        throughput = len(all_times) / (total_wall_s / 3600) if total_wall_s > 0 else 0
        logger.info("=== ACQUISITION TIMING REPORT ===")
        logger.info("  Total tiles: %d, wall time: %.1fh", len(all_times), total_wall_s / 3600)
        logger.info("  Average: %.1fms/tile (%.1fs)", avg_all, avg_all / 1000)
        logger.info(
            "  AF tiles: %d (avg %.1fms), non-AF: %d (avg %.1fms)",
            len(af_tiles), avg_af, len(non_af_tiles), avg_non_af,
        )
        logger.info("  Throughput: %.0f tiles/hr", throughput)

    # Get final Z position for tilt correction model
    final_z = ctx.hardware.get_current_position().z
    sat_summary = ctx.sat_monitor.get_summary_string()
    ctx.set_state("COMPLETED", final_z=final_z, saturation_summary=sat_summary)
    logger.info("=== ACQUISITION COMPLETED SUCCESSFULLY ===")
    ctx.sat_monitor.log_summary()
    logger.info(f"Final Z position: {final_z:.2f} um")
    logger.info(f"Total images saved: {ctx.image_count}/{ctx.total_images}")
    logger.info(f"Output directory: {ctx.output_path}")

    # Report autofocus activity
    if ctx.deferred_af_positions:
        logger.info(
            f"Autofocus deferred at {len(ctx.deferred_af_positions)} positions "
            f"due to insufficient tissue: {sorted(ctx.deferred_af_positions)}"
        )


def _cleanup_acquisition(ctx: AcquisitionContext) -> None:
    """Guaranteed cleanup: shutdown write pool, return stage to start."""
    # Shut down background write pool (drains any remaining writes)
    if ctx.write_pool is not None:
        try:
            ctx.write_pool.shutdown()
        except Exception as e:
            ctx.logger.warning(f"Error shutting down write pool: {e}")

    # Close NDJSON stream if still open (error path)
    if ctx.tile_measurements_stream is not None:
        try:
            ctx.tile_measurements_stream.close()
        except Exception:
            pass
        ctx.tile_measurements_stream = None

    # Return XY to starting position (preserve Z from last autofocus so the
    # next annotation's Z-hint starts near the actual focal plane rather than
    # resetting to the user's initial Z).
    if ctx.starting_position is not None:
        try:
            ctx.logger.info("Returning to starting XY position (preserving Z)")
            ctx.hardware.move_to_position(
                Position(x=ctx.starting_position.x, y=ctx.starting_position.y)
            )
        except Exception as e:
            ctx.logger.warning(f"Failed to return to starting position: {e}")


def _initialize_loop_infrastructure(ctx: AcquisitionContext) -> None:
    """Create saturation monitor, write pool, NDJSON stream, and tracking vars."""
    logger = ctx.logger

    # Initialize saturation monitor for adaptive abort/rate-limiting
    sat_threshold = None
    try:
        sat_threshold = ctx.config_manager.get(
            "acquisition_settings", "saturation_abort_threshold_pct"
        )
        if sat_threshold is not None:
            sat_threshold = float(sat_threshold)
            logger.info(f"Saturation abort threshold from config: {sat_threshold}%")
    except Exception:
        pass
    ctx.sat_monitor = SaturationMonitor(
        angles=ctx.params.get("angles", []),
        logger=logger,
        biref_abort_threshold_pct=sat_threshold,
    )

    # Initialize background write pool for overlapped I/O.
    ctx.write_pool = _TileWritePool(max_workers=2)

    # Reset AF position tracking (may have been set during pre-acquisition AF)
    if ctx.last_af_pos_idx < 0:
        ctx.last_af_pos_idx = -1

    # Open NDJSON stream for live Java-side tile updates.
    tile_measurements_ndjson_path = ctx.output_path / "tile_measurements.ndjson"
    try:
        ctx.tile_measurements_stream = open(
            tile_measurements_ndjson_path, "w", encoding="utf-8"
        )
        logger.info("Streaming per-tile measurements to %s", tile_measurements_ndjson_path)
    except Exception as e:
        logger.warning("Could not open tile measurements NDJSON stream: %s", e)
        ctx.tile_measurements_stream = None


def _prepare_acquisition(
    message: str,
    client_addr,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
    update_progress: Callable,
    set_state: Callable,
    is_cancelled: Callable,
    request_manual_focus: Optional[Callable] = None,
    request_hardware_error_recovery: Optional[Callable] = None,
    connection_config_path: Optional[str] = None,
) -> AcquisitionContext:
    """Parse message, load configs, setup BG/WB/Z-stack, create output dirs.

    Returns a populated AcquisitionContext. Raises on any setup failure.
    """
    logger.info(f"=== ACQUISITION WORKFLOW STARTED for client {client_addr} ===")

    # Stop live mode if running - JAI camera properties cannot be changed during live streaming
    try:
        hardware.camera.stop_if_streaming()
        logger.info("Ensured camera not streaming before acquisition")
    except Exception as e:
        logger.warning(f"Could not stop live/sequence mode: {e}")

    # Invalidate camera settings state so first apply_settings() writes
    # all properties to hardware (defensive -- state may be stale from
    # live mode or manual adjustments).
    hardware.camera.invalidate_settings_state()

    # Parse the acquisition parameters
    params = parse_acquisition_message(message)

    logger.info("Acquisition parameters:")
    logger.info(f"  Client: {client_addr}")
    logger.info(f"  Sample label: {params['sample_label']}")
    logger.info(f"  Scan type: {params['scan_type']}")
    logger.info(f"  Region: {params['region_name']}")
    if params.get("channels"):
        logger.info(f"  Channels: {params['channels']}")
        logger.info(f"  Channel exposures: {params['channel_exposures']} ms")
        if params.get("channel_intensities"):
            logger.info(f"  Channel intensity overrides: {params['channel_intensities']}")
    else:
        logger.info(f"  Angles: {params['angles']} degrees")
        logger.info(f"  Exposures: {params['exposures']} ms")

    # Load the yaml file
    if not params["yaml_file_path"]:
        raise ValueError("YAML file path is required")
    if not Path(params["yaml_file_path"]).exists():
        raise FileNotFoundError(f"YAML file {params['yaml_file_path']} does not exist")

    # Load configuration using the config manager
    ppm_settings = config_manager.load_config_file(params["yaml_file_path"])
    loci_rsc_file = str(
        Path(params["yaml_file_path"]).parent / "resources" / "resources_LOCI.yml"
    )
    loci_resources = config_manager.load_config_file(loci_rsc_file)
    ppm_settings.update(loci_resources)
    hardware.settings = ppm_settings

    # SAFETY WARNING: Check if ACQUIRE yaml differs from CONFIG
    if connection_config_path:
        acquire_yaml = Path(params["yaml_file_path"]).resolve()
        connection_yaml = Path(connection_config_path).resolve()
        if acquire_yaml != connection_yaml:
            logger.warning("=" * 80)
            logger.warning("CONFIG MISMATCH WARNING")
            logger.warning(f"Connection CONFIG:  {connection_yaml}")
            logger.warning(f"ACQUIRE --yaml:     {acquire_yaml}")
            logger.warning("ACQUIRE yaml has overridden connection config for this acquisition")
            logger.warning("This may cause unexpected behavior or hardware misconfiguration!")
            logger.warning("=" * 80)

    # Re-initialize microscope-specific methods with updated settings
    if hasattr(hardware, "_initialize_microscope_methods"):
        hardware._initialize_microscope_methods()
        logger.info("Re-initialized hardware methods with updated settings")

    # Apply acquisition profile mode setup
    scan_type = params.get("scan_type", "")
    if hasattr(hardware, "apply_mode_setup"):
        hardware.apply_mode_setup(scan_type)

    # Try to load and apply JAI white balance settings if available
    wb_calibration_folder = params.get("white_balance_calibration_folder")
    if wb_calibration_folder:
        mod_cfg = get_modality_config(params.get("scan_type", ""))
        wb_modality = mod_cfg.wb_settings_key or params.get("scan_type", "").split("_")[0].lower()
        wb_objective = params.get("objective", "default")
        wb_detector = params.get("detector", "default")
        load_and_apply_white_balance_settings(
            hardware=hardware,
            calibration_folder=wb_calibration_folder,
            detector=wb_detector,
            modality=wb_modality,
            objective=wb_objective,
            logger=logger,
        )

    # Extract modality from scan type
    modality = BackgroundCorrectionUtils.get_modality_from_scan_type(params["scan_type"])
    mod_config = get_modality_config(params.get("scan_type", ""))
    logger.info(f"Using modality: {modality}")

    # Get processing settings from parameters
    background_correction_enabled = params.get("background_correction_enabled", False)
    background_correction_method = params.get("background_correction_method", "divide")
    background_disabled_angles = params.get("background_disabled_angles", [])
    white_balance_enabled = params.get("white_balance_enabled", True)
    save_raw_tiles = params.get("save_raw", False)
    logger.info(f"Save raw tiles: {save_raw_tiles}")

    # Z-stack parameters
    z_stack_enabled = params.get("z_stack", False)
    z_offsets = [0.0]
    projection_fn = None
    logger.info("Z-stack check: z_stack=%s, z_start=%s, z_end=%s, z_step=%s, z_projection=%s",
                z_stack_enabled, params.get("z_start"), params.get("z_end"),
                params.get("z_step"), params.get("z_projection"))
    if z_stack_enabled:
        z_start_abs = params.get("z_start")
        z_end_abs = params.get("z_end")
        z_step = params.get("z_step")

        if z_start_abs is not None and z_end_abs is not None and z_step is not None:
            z_total_range = abs(z_end_abs - z_start_abs)
        elif z_step is not None:
            z_total_range = 0
        else:
            z_total_range = 0

        if z_step is None or z_step <= 0:
            logger.warning(
                "Z-stack enabled but z_step is missing or invalid (step=%s). "
                "Continuing in 2D mode.", z_step
            )
            z_stack_enabled = False
        elif z_total_range <= 0:
            logger.warning(
                "Z-stack range is zero or negative (start=%s, end=%s). "
                "Continuing in 2D mode.", z_start_abs, z_end_abs
            )
            z_stack_enabled = False
        else:
            from microscope_command_server.acquisition.projections import (
                generate_z_offsets, get_projection,
            )
            z_offsets = generate_z_offsets(z_total_range, z_step)
            projection_name = params.get("z_projection", "max")
            try:
                projection_fn = get_projection(projection_name)
            except KeyError as e:
                logger.error("Invalid z_projection: %s. Falling back to 'max'.", e)
                projection_fn = get_projection("max")
                projection_name = "max"
            logger.info(
                "Z-stack: %d planes over +/-%.1f um (step=%.1f), projection=%s",
                len(z_offsets), z_total_range / 2, z_step, projection_name
            )

    # Log background correction configuration
    if background_correction_enabled:
        logger.info(f"Background correction enabled with method: {background_correction_method}")
        if background_disabled_angles:
            logger.info(f"Background correction will be disabled for angles: {background_disabled_angles}")
    else:
        logger.info("Background correction disabled")

    if background_correction_enabled and white_balance_enabled:
        logger.info(
            "Both background correction and white balance enabled "
            "(backgrounds must be captured with matching WB settings)"
        )

    # ======= BACKGROUND CORRECTION SETUP =======
    background_images = {}
    background_scaling_factors = {}
    background_wb_coeffs = {}
    channel_background_images: Dict[str, Any] = {}

    if background_correction_enabled:
        background_dir = None

        # Priority 1: Message parameter
        if "background_folder" in params:
            background_dir = Path(params["background_folder"])
            logger.info(f"Using background folder from message: {background_dir}")
        else:
            # Priority 2: YAML configuration from imageprocessing config file
            config_path = Path(params["yaml_file_path"])
            imageprocessing_path = config_path.parent / f"imageprocessing_{config_path.stem.replace('config_', '')}.yml"

            bc_settings = None
            if imageprocessing_path.exists():
                try:
                    imageprocessing_config = config_manager.load_config_file(str(imageprocessing_path))
                    bc_config = imageprocessing_config.get("background_correction", {})
                    bc_settings = bc_config.get(modality, {})
                    logger.info(f"Loaded background correction settings from: {imageprocessing_path}")
                except Exception as e:
                    logger.warning(f"Failed to load imageprocessing config: {e}")
            else:
                logger.warning(f"Imageprocessing config not found at: {imageprocessing_path}")

            if bc_settings and bc_settings.get("enabled") and bc_settings.get("base_folder"):
                background_dir = Path(bc_settings["base_folder"]) / modality
                logger.info(f"Using background folder from YAML config: {background_dir}")

        # Load background images if directory is valid
        if background_dir and background_dir.exists():
            logger.info(f"Loading background images from: {background_dir}")
            bg_load_angles = params["angles"] if params["angles"] else [0.0]
            background_images, background_scaling_factors, background_wb_coeffs = (
                BackgroundCorrectionUtils.load_background_images(
                    background_dir, bg_load_angles, logger
                )
            )

            if background_images:
                logger.info(f"Loaded {len(background_images)} background images")
            else:
                logger.warning("No background images found - disabling background correction")
                background_correction_enabled = False

            # Opt-in per-channel background loading for multi-channel widefield IF
            if params.get("channels"):
                try:
                    import skimage.io as _skio
                    for cid in params["channels"]:
                        candidates = [
                            background_dir / cid / "background.tif",
                            background_dir / f"{cid}.tif",
                            background_dir / f"{cid}.tiff",
                        ]
                        for candidate in candidates:
                            if candidate.exists():
                                try:
                                    channel_background_images[cid] = _skio.imread(str(candidate))
                                    logger.info("Loaded channel background for %s from %s", cid, candidate)
                                    break
                                except Exception as load_e:
                                    logger.warning("Failed to load channel background %s: %s", candidate, load_e)
                    if channel_background_images:
                        logger.info(
                            "Loaded %d per-channel backgrounds for widefield IF "
                            "(missing channels will acquire uncorrected)",
                            len(channel_background_images),
                        )
                    else:
                        logger.info(
                            "No per-channel backgrounds found under %s -- "
                            "channel acquisition will run without flat-field correction",
                            background_dir,
                        )
                except Exception as e:
                    logger.warning("Per-channel background load failed: %s", e)
        else:
            logger.warning(f"Background directory not found: {background_dir}")
            logger.warning("Disabling background correction")
            background_correction_enabled = False

    # ======= WHITE BALANCE SETUP =======
    angles_wb = {}

    wb_mode = params.get("wb_mode")
    if wb_mode is None:
        wb_enabled = params.get("white_balance_enabled", True)
        wb_per_angle = params.get("white_balance_per_angle", False)
        if not wb_enabled:
            wb_mode = "off"
        elif wb_per_angle:
            wb_mode = "per_angle"
        else:
            raise ValueError(
                "No --wb-mode specified in acquisition request. "
                "White balance mode must be explicitly chosen by the user: "
                "camera_awb, simple, or per_angle. "
                "Update the client to always send --wb-mode."
            )
    logger.info(f"White balance mode: {wb_mode}")

    _wb_mod_check = get_modality_config(params.get("scan_type", ""))
    if wb_mode != "off" and _wb_mod_check.wb_settings_key is None:
        logger.info(
            "White balance mode '%s' requested but modality has no WB settings key "
            "(monochrome or single-channel modality). Forcing wb_mode='off'.",
            wb_mode,
        )
        wb_mode = "off"

    white_balance_enabled = wb_mode != "off"
    white_balance_per_angle = wb_mode == "per_angle"

    # Auto-detect JAI camera
    is_jai_camera = False
    try:
        camera_name = hardware.get_camera_name()
        is_jai_camera = hardware.camera.supports_per_channel_exposure()
        if is_jai_camera:
            logger.info(f"Per-channel camera detected: {camera_name}")
    except Exception as e:
        logger.debug(f"Could not detect camera type: {e}")

    # Load JAI hardware white balance calibration
    jai_calibration = None
    simple_wb_data = None
    simple_wb_analog_red = 1.0
    simple_wb_analog_blue = 1.0
    camera_awb_gains = {}
    _wb_mod_config = get_modality_config(params.get("scan_type", ""))
    base_modality = _wb_mod_config.wb_settings_key or params["scan_type"].split("_")[0].lower()

    if wb_mode == "camera_awb":
        logger.info(
            "Camera AWB mode: ensure AWB was configured in MicroManager's "
            "Device Property Browser before starting acquisition."
        )
        jai_calibration = load_jai_calibration_from_imageprocessing(
            config_path=Path(params["yaml_file_path"]),
            per_angle=True,
            modality=base_modality,
            objective=params.get("objective"),
            detector=params.get("detector"),
            logger=logger,
        )
        if is_jai_camera:
            try:
                hardware.camera.disable_individual_exposure()
                hardware.camera.disable_individual_gain()
                logger.info("Camera AWB mode: disabled per-channel exposure/gain (preserving AWB analog gains)")
            except Exception as e:
                logger.warning(f"Could not configure camera AWB mode: {e}")
        if jai_calibration:
            logger.info("Camera AWB: loaded unified gains from calibration for brightness control")
        else:
            logger.info("Camera AWB: no calibration data found, using client exposures only")
        if jai_calibration and "angles" in jai_calibration:
            for angle_name, angle_data in jai_calibration["angles"].items():
                gains = angle_data.get("gains", {})
                camera_awb_gains[angle_name] = gains.get("unified_gain", 1.0)
            logger.info(f"Camera AWB unified gains: {camera_awb_gains}")
        jai_calibration = None  # Don't use per-channel calibration in acquisition loop

    elif wb_mode == "simple":
        if is_jai_camera:
            try:
                hardware.camera.clear_awb_corrections()
                hardware.camera.disable_individual_exposure()
                hardware.camera.disable_individual_gain()
                logger.info("Simple WB: cleared AWB + disabled individual mode")
            except Exception as e:
                logger.warning(f"Could not clear AWB corrections before simple WB: {e}")
        jai_calibration = load_jai_calibration_from_imageprocessing(
            config_path=Path(params["yaml_file_path"]),
            per_angle=True,
            modality=base_modality,
            objective=params.get("objective"),
            detector=params.get("detector"),
            logger=logger,
        )
        if jai_calibration:
            logger.info("Simple WB: loaded per-angle calibration as base for ratio-scaling")
            uncrossed_gains = (
                jai_calibration.get("angles", {})
                .get("uncrossed", {})
                .get("gains", {})
            )
            simple_wb_analog_red = uncrossed_gains.get("analog_red", 1.0)
            simple_wb_analog_blue = uncrossed_gains.get("analog_blue", 1.0)
            logger.info(
                f"Simple WB: analog gains from uncrossed calibration: "
                f"R={simple_wb_analog_red:.3f}, B={simple_wb_analog_blue:.3f}"
            )
            simple_wb_data = load_simple_wb_from_imageprocessing(
                config_path=Path(params["yaml_file_path"]),
                modality=base_modality,
                objective=params.get("objective"),
                detector=params.get("detector"),
                logger=logger,
            )
            if simple_wb_data:
                logger.info(f"Simple WB: loaded pre-computed scales for {len(simple_wb_data.get('angles', {}))} angles")
            else:
                logger.info("Simple WB: no pre-computed data, will use uncrossed ratios with exposure_scale")
        else:
            logger.warning(
                "Simple WB mode requested but no Mode 3 calibration found! "
                "Run 'PPM White Balance Calibration' first."
            )

    elif wb_mode == "per_angle":
        if is_jai_camera:
            try:
                hardware.camera.clear_awb_corrections()
                hardware.camera.disable_individual_exposure()
                hardware.camera.disable_individual_gain()
                logger.info("Per-angle WB: cleared AWB + disabled individual mode")
            except Exception as e:
                logger.warning(f"Could not clear AWB corrections before per-angle WB: {e}")
        jai_calibration = load_jai_calibration_from_imageprocessing(
            config_path=Path(params["yaml_file_path"]),
            per_angle=True,
            modality=base_modality,
            objective=params.get("objective"),
            detector=params.get("detector"),
            logger=logger,
        )
        if jai_calibration:
            logger.info(
                f"Per-angle WB: loaded calibration "
                f"for {base_modality}/{params.get('objective')}/{params.get('detector')}"
            )
        else:
            if is_jai_camera:
                logger.warning(
                    "JAI camera detected but no calibration found! "
                    "Run 'White Balance Calibration' for proper color balance."
                )
            logger.info("No JAI calibration found - using software white balance")
    else:
        logger.info("White balance disabled (wb_mode=%s)", wb_mode)

    if white_balance_enabled:
        if background_wb_coeffs:
            angles_wb = {angle: list(coeffs) for angle, coeffs in background_wb_coeffs.items()}
            logger.info(
                "Software WB: using background-derived coefficients for %d angles",
                len(angles_wb),
            )
        else:
            mod_config = get_modality_config(modality)
            angle_mapping = mod_config.name_to_angle if mod_config.name_to_angle else {}
            if angle_mapping:
                angles_wb = {v: [1.0, 1.0, 1.0] for v in angle_mapping.values()}
            else:
                angles_wb = {0.0: [1.0, 1.0, 1.0]}
            logger.info(
                "Software WB: no background-derived coefficients; using neutral [1,1,1] for %d angles",
                len(angles_wb),
            )

        if white_balance_per_angle:
            logger.info(f"Using per-angle white balance for {len(angles_wb)} angles")
        else:
            uncrossed_profile = angles_wb.get(90.0, [1.0, 1.0, 1.0])
            logger.info(f"Using single white balance profile for all angles: {uncrossed_profile}")
            for angle in params.get("angles", []):
                angles_wb[angle] = uncrossed_profile

    # Save starting position
    starting_position = hardware.get_current_position()

    # Set up output paths
    project_path = Path(params["projects_folder_path"]) / params["sample_label"]
    output_path = project_path / params["scan_type"] / params["region_name"]
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_path}")

    # Read tile positions
    tile_config_path = output_path / "TileConfiguration.txt"
    positions = TileConfigUtils.read_tile_config(tile_config_path, hardware.core)

    if not positions:
        raise RuntimeError(f"No tile positions found in {tile_config_path}")

    xy_positions = [(pos.x, pos.y) for pos, filename in positions]

    # Create angle subdirectories
    if params["angles"]:
        for angle in params["angles"]:
            angle_dir = output_path / str(angle)
            angle_dir.mkdir(exist_ok=True)
            shutil.copy2(tile_config_path, angle_dir / "TileConfiguration.txt")

    # Create channel subdirectories
    if params.get("channels"):
        for cid in params["channels"]:
            channel_dir = output_path / str(cid)
            channel_dir.mkdir(exist_ok=True)
            shutil.copy2(tile_config_path, channel_dir / "TileConfiguration.txt")

    # Calculate total images and update progress
    n_z_planes = len(z_offsets)
    n_channels = len(params.get("channels", []) or [])
    if n_channels > 0:
        n_steps_per_tile = n_channels
    elif params.get("angles"):
        n_steps_per_tile = len(params["angles"])
    else:
        n_steps_per_tile = 1
    n_angles = len(params["angles"]) if params["angles"] else 1
    # Time-lapse multiplier. For single-pass acquisitions (the default),
    # n_timepoints=1 and total_images / log line are unchanged.
    n_timepoints = int(params.get("n_timepoints", 1))
    if n_timepoints < 1:
        logger.warning(
            "n_timepoints=%d is invalid; clamping to 1", n_timepoints
        )
        n_timepoints = 1
    total_images = len(positions) * n_z_planes * n_steps_per_tile * n_timepoints

    update_progress(0, total_images)
    if n_timepoints > 1:
        interval_seconds = float(params.get("interval_seconds", 0.0))
        logger.info(
            f"Starting time-lapse acquisition of {total_images} total images "
            f"({n_timepoints} timepoints x {len(positions)} positions x "
            f"{n_z_planes} Z-planes x {n_angles} angles, "
            f"interval={interval_seconds:.1f}s)"
        )
    else:
        logger.info(
            f"Starting acquisition of {total_images} total images "
            f"({len(positions)} positions x {n_z_planes} Z-planes x {n_angles} angles)"
        )

    metadata_txt_for_positions = output_path / "image_positions_metadata.txt"

    # Build and return context
    return AcquisitionContext(
        params=params,
        hardware=hardware,
        config_manager=config_manager,
        logger=logger,
        client_addr=client_addr,
        output_path=output_path,
        modality=modality,
        mod_config=mod_config,
        ppm_settings=ppm_settings,
        positions=positions,
        xy_positions=xy_positions,
        total_images=total_images,
        background_correction_enabled=background_correction_enabled,
        background_correction_method=background_correction_method,
        background_disabled_angles=background_disabled_angles,
        background_images=background_images,
        background_scaling_factors=background_scaling_factors,
        channel_background_images=channel_background_images,
        white_balance_enabled=white_balance_enabled,
        white_balance_per_angle=white_balance_per_angle,
        wb_mode=wb_mode,
        angles_wb=angles_wb,
        jai_calibration=jai_calibration,
        simple_wb_data=simple_wb_data,
        camera_awb_gains=camera_awb_gains,
        is_jai_camera=is_jai_camera,
        simple_wb_analog_red=simple_wb_analog_red,
        simple_wb_analog_blue=simple_wb_analog_blue,
        z_stack_enabled=z_stack_enabled,
        z_offsets=z_offsets,
        projection_fn=projection_fn,
        save_raw_tiles=save_raw_tiles,
        n_timepoints=n_timepoints,
        interval_seconds=float(params.get("interval_seconds", 0.0)),
        output_format=str(params.get("output_format", "ome-per-t")),
        hint_z=params.get("hint_z"),
        metadata_txt_for_positions=metadata_txt_for_positions,
        update_progress=update_progress,
        set_state=set_state,
        is_cancelled=is_cancelled,
        request_manual_focus=request_manual_focus,
        request_hardware_error_recovery=request_hardware_error_recovery,
        starting_position=starting_position,
    )


def _configure_autofocus(ctx: AcquisitionContext) -> None:
    """Load AF settings from YAML, resolve strategy, compute AF positions.

    Mutates ctx autofocus fields in-place.

    When --af-disabled was sent on the wire, this is a no-op: the YAML
    is not loaded, no AF positions are scheduled, and the per-tile
    dispatcher's `pos_idx in ctx.dynamic_af_positions` test will be
    False for every tile, killing pre-acquisition AF, drift checks, and
    the manual-focus fallback.
    """
    logger = ctx.logger
    params = ctx.params

    if params.get("af_disabled"):
        logger.warning(
            "Autofocus DISABLED for this acquisition (--af-disabled). "
            "No pre-acquisition AF, no per-tile drift checks, no manual-focus prompts. "
            "Z drift will not be corrected."
        )
        ctx.af_positions = []
        ctx.dynamic_af_positions = set()
        ctx.deferred_af_positions = set()
        ctx.completed_af_positions = []
        return

    fov = ctx.hardware.get_fov()

    # Load autofocus settings from separate autofocus_{microscope}.yml file
    af_n_tiles = 5
    af_search_range = 50
    af_n_steps = 11
    af_interp_strength = 100
    af_interp_kind = "quadratic"
    af_score_metric_name = "normalized_variance"
    af_texture_threshold = 0.005
    af_tissue_area_threshold = 0.2
    af_rgb_brightness_threshold = 240.0
    af_sweep_range_um = 10.0
    af_sweep_n_steps = 5
    af_edge_retries = 2
    af_gap_index_multiplier = 3
    af_gap_spatial_multiplier = 2.0

    current_objective = params.get("objective", "")
    af_settings_found = False

    # Derive autofocus config path from main config path
    config_path = Path(params["yaml_file_path"])
    config_name = config_path.stem
    microscope_name = config_name.replace("config_", "")
    autofocus_file = config_path.parent / f"autofocus_{microscope_name}.yml"

    if not autofocus_file.exists():
        raise RuntimeError(
            f"Autofocus configuration file not found: {autofocus_file}\n"
            f"Cannot proceed with acquisition - autofocus settings are required for objective '{current_objective}'.\n"
            f"Please create the autofocus configuration file with settings for your objectives."
        )

    with open(autofocus_file, "r") as f:
        autofocus_config = yaml.safe_load(f)

    # Find settings for current objective
    af_settings_list = autofocus_config.get("autofocus_settings", [])
    af_setting = None
    for _af_setting in af_settings_list:
        if _af_setting.get("objective") == current_objective:
            af_setting = _af_setting
            af_n_tiles = af_setting.get("n_tiles", af_n_tiles)
            af_search_range = af_setting.get("search_range_um", af_search_range)
            af_n_steps = af_setting.get("n_steps", af_n_steps)
            af_interp_strength = af_setting.get("interp_strength", af_interp_strength)
            af_interp_kind = af_setting.get("interp_kind", af_interp_kind)
            af_score_metric_name = af_setting.get("score_metric", af_score_metric_name)
            af_texture_threshold = af_setting.get("texture_threshold", af_texture_threshold)
            af_tissue_area_threshold = af_setting.get("tissue_area_threshold", af_tissue_area_threshold)
            af_rgb_brightness_threshold = af_setting.get("rgb_brightness_threshold", af_rgb_brightness_threshold)
            af_sweep_range_um = af_setting.get("sweep_range_um", af_sweep_range_um)
            af_sweep_n_steps = af_setting.get("sweep_n_steps", af_sweep_n_steps)
            af_edge_retries = af_setting.get("edge_retries", af_edge_retries)
            af_gap_index_multiplier = af_setting.get("gap_index_multiplier", af_gap_index_multiplier)
            af_gap_spatial_multiplier = af_setting.get("gap_spatial_multiplier", af_gap_spatial_multiplier)
            # Legacy support: old adaptive_initial_step_um -> sweep_range_um
            if "adaptive_initial_step_um" in af_setting and "sweep_range_um" not in af_setting:
                af_sweep_range_um = af_setting["adaptive_initial_step_um"] * 2
            logger.info(
                f"Loaded autofocus settings for {current_objective}: "
                f"n_steps={af_n_steps}, search_range={af_search_range}um, n_tiles={af_n_tiles}, "
                f"interp_strength={af_interp_strength}, interp_kind={af_interp_kind}, "
                f"score_metric={af_score_metric_name}, "
                f"texture_threshold={af_texture_threshold}, tissue_area_threshold={af_tissue_area_threshold}, "
                f"rgb_brightness_threshold={af_rgb_brightness_threshold}, "
                f"sweep: range={af_sweep_range_um}um, n_steps={af_sweep_n_steps}, "
                f"edge_retries={af_edge_retries}, "
                f"gap_index_mult={af_gap_index_multiplier}, gap_spatial_mult={af_gap_spatial_multiplier}"
            )
            af_settings_found = True
            break

    if not af_settings_found:
        available_objectives = [s.get("objective", "unknown") for s in af_settings_list]
        raise RuntimeError(
            f"No autofocus settings found for objective '{current_objective}' in {autofocus_file}\n"
            f"Available objectives in config: {available_objectives}\n"
            f"Cannot proceed with acquisition - please add autofocus settings for '{current_objective}' "
            f"or verify the objective name matches the configuration."
        )

    # Check that autofocus settings have been explicitly calibrated.
    af_calibrated = af_setting.get("calibrated", True) if af_setting else True
    if af_calibrated is False:
        raise RuntimeError(
            f"Autofocus settings for objective '{current_objective}' have not been calibrated!\n"
            f"The autofocus configuration file was generated with default placeholder values "
            f"that may not be safe for your hardware.\n"
            f"\n"
            f"To fix this:\n"
            f"  1. Open {autofocus_file}\n"
            f"  2. Adjust search_range_um and n_steps for your objective\n"
            f"  3. Set 'calibrated: true' for this objective\n"
            f"  4. Or run the Autofocus Benchmark utility from the QP Scope menu\n"
            f"\n"
            f"CRITICAL: An incorrect search_range_um can crash the objective into the sample!"
        )

    # -------- Schema v2 strategy resolution --------
    from microscope_control.autofocus.strategies import (
        build_strategy,
        StrategyFailureMode,
    )

    af_strategy = None
    af_strategy_name = None
    af_focus_channel = params.get("focus_channel")

    try:
        schema_version = (
            autofocus_config.get("schema_version", 1) if isinstance(autofocus_config, dict) else 1
        )
        strategies_library = autofocus_config.get("strategies", {}) if isinstance(autofocus_config, dict) else {}
        modality_bindings = autofocus_config.get("modalities", {}) if isinstance(autofocus_config, dict) else {}

        if schema_version >= 2 and strategies_library:
            current_modality = ctx.modality or ""
            current_modality_lower = current_modality.lower()
            best_match = None
            best_len = 0
            for mod_key in modality_bindings.keys():
                mod_key_str = str(mod_key).lower()
                if current_modality_lower.startswith(mod_key_str) and len(mod_key_str) > best_len:
                    best_match = mod_key
                    best_len = len(mod_key_str)

            if best_match is not None:
                binding = modality_bindings[best_match]
                strategy_name = binding.get("strategy", "dense_texture")
                library_entry = strategies_library.get(strategy_name, {})
                resolved_params = dict(library_entry)
                overrides_block = binding.get("overrides", {}) or {}
                if "validity_params" in overrides_block and isinstance(
                    overrides_block["validity_params"], dict
                ):
                    merged_vp = dict(library_entry.get("validity_params", {}) or {})
                    merged_vp.update(overrides_block["validity_params"])
                    resolved_params = dict(library_entry)
                    resolved_params["validity_params"] = merged_vp
                if "on_failure" in overrides_block:
                    resolved_params["on_failure"] = overrides_block["on_failure"]

                af_strategy_name = strategy_name
                af_strategy = build_strategy(strategy_name, resolved_params)
                logger.info(
                    "Autofocus strategy resolved: modality='%s' -> binding '%s' -> strategy '%s' (on_failure=%s)",
                    current_modality, best_match, strategy_name, af_strategy.on_failure.value,
                )
            else:
                logger.info(
                    "No v2 modality binding found for '%s'; using v1 dense_texture compatibility",
                    current_modality,
                )

        # Per-acquisition override: --af-strategy CLI flag wins over YAML.
        cli_strategy_override = params.get("af_strategy")
        if cli_strategy_override:
            library_entry = strategies_library.get(cli_strategy_override, {}) if strategies_library else {}
            af_strategy = build_strategy(cli_strategy_override, library_entry)
            af_strategy_name = cli_strategy_override
            logger.info(
                "Autofocus strategy overridden by --af-strategy CLI flag: '%s' (on_failure=%s)",
                cli_strategy_override, af_strategy.on_failure.value,
            )

        # Fallback: build a DenseTextureStrategy from flat v1 fields
        if af_strategy is None:
            af_strategy = build_strategy(
                "dense_texture",
                {
                    "validity_params": {
                        "texture_threshold": af_texture_threshold,
                        "tissue_area_threshold": af_tissue_area_threshold,
                        "rgb_brightness_threshold": af_rgb_brightness_threshold,
                    },
                },
            )
            af_strategy_name = "dense_texture (v1 compat)"
            logger.info("Autofocus strategy: v1 compatibility dense_texture built from flat fields")
    except Exception as strat_err:
        logger.warning(
            "Autofocus strategy resolution failed (%s); falling back to dense_texture default",
            strat_err, exc_info=True,
        )
        af_strategy = build_strategy("dense_texture", {})
        af_strategy_name = "dense_texture (fallback)"

    if af_focus_channel:
        logger.info("Autofocus focus channel: %s", af_focus_channel)

    if af_n_tiles < 1:
        logger.warning(
            f"af_n_tiles={af_n_tiles} is below minimum (1), clamping to 1. "
            f"At least 1 AF position is required for overlapped I/O drain."
        )
        af_n_tiles = max(1, af_n_tiles)

    # Map score metric name to function
    score_metric_map = {
        "laplacian_variance": AutofocusUtils.autofocus_profile_laplacian_variance,
        "sobel": AutofocusUtils.autofocus_profile_sobel,
        "brenner_gradient": AutofocusUtils.autofocus_profile_brenner_gradient,
        "robust_sharpness": AutofocusUtils.autofocus_profile_robust_sharpness_metric,
        "hybrid_sharpness": AutofocusUtils.autofocus_profile_hybrid_sharpness_metric,
    }
    af_score_metric = score_metric_map.get(
        af_score_metric_name, AutofocusUtils.autofocus_profile_laplacian_variance
    )

    timing_window_size = max(10, 3 * af_n_tiles)
    logger.info(
        f"Timing window size for progress estimation: {timing_window_size} tiles "
        f"(3 x {af_n_tiles} AF positions, min 10)"
    )

    preferred_af_tile = params.get("preferred_af_tile")
    af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
        fov, ctx.xy_positions, n_tiles=af_n_tiles,
        preferred_first_af=preferred_af_tile,
    )

    small_grid_override = (len(ctx.xy_positions) <= 9)
    if small_grid_override:
        logger.info(
            f"Small grid override: {len(ctx.xy_positions)} tiles <= 9, "
            f"autofocus at ALL positions: {af_positions} (min_distance={af_min_distance})"
        )
    else:
        pref_msg = f" (preferred tile {preferred_af_tile} from WSI)" if preferred_af_tile is not None else ""
        logger.info(
            f"Autofocus positions ({len(af_positions)}/{len(ctx.xy_positions)} tiles): "
            f"{af_positions} (min_distance={af_min_distance:.1f}){pref_msg}"
        )

    # Write timing metadata
    timing_metadata_path = ctx.output_path / "acquisition_metadata.txt"
    with open(timing_metadata_path, "w") as f:
        f.write(f"timing_window_size={timing_window_size}\n")
        f.write(f"af_n_tiles={len(af_positions)}\n")
        f.write(f"total_tiles={ctx.total_images}\n")
        f.write(f"af_n_steps={af_n_steps}\n")
        f.write(f"objective={current_objective}\n")
    logger.debug(
        f"Wrote timing metadata to {timing_metadata_path}: "
        f"window={timing_window_size}, af_positions={len(af_positions)}, tiles={ctx.total_images}"
    )

    # Populate ctx with all autofocus fields
    ctx.af_n_tiles = af_n_tiles
    ctx.af_n_steps = af_n_steps
    ctx.af_search_range = af_search_range
    ctx.af_interp_strength = af_interp_strength
    ctx.af_interp_kind = af_interp_kind
    ctx.af_score_metric = af_score_metric
    ctx.af_score_metric_name = af_score_metric_name
    ctx.af_sweep_range_um = af_sweep_range_um
    ctx.af_sweep_n_steps = af_sweep_n_steps
    ctx.af_edge_retries = af_edge_retries
    ctx.af_gap_index_multiplier = af_gap_index_multiplier
    ctx.af_gap_spatial_multiplier = af_gap_spatial_multiplier
    ctx.af_strategy = af_strategy
    ctx.af_strategy_name = af_strategy_name
    ctx.af_focus_channel = af_focus_channel
    ctx.af_positions = af_positions
    ctx.af_min_distance = af_min_distance
    ctx.dynamic_af_positions = set(af_positions)
    ctx.deferred_af_positions = set()
    ctx.first_tissue_autofocus_done = False
    ctx.completed_af_positions = []


def _run_pre_acquisition_autofocus(ctx: AcquisitionContext) -> None:
    """Run initial autofocus at first tissue position before main loop.

    Mutates ctx: exposure_90, first_tissue_autofocus_done, completed_af_positions,
    dynamic_af_positions, last_af_pos_idx.
    """
    from microscope_control.autofocus.strategies import StrategyFailureMode

    logger = ctx.logger
    params = ctx.params
    hardware = ctx.hardware

    if len(ctx.positions) == 0 or len(ctx.af_positions) == 0:
        return

    # Apply Z-focus hint if provided (predicted from tilt correction model)
    if ctx.hint_z is not None:
        current_z = hardware.get_current_position().z
        logger.info(f"Z-focus hint received: {ctx.hint_z:.2f} um (current Z: {current_z:.2f} um)")
        logger.info("Moving to predicted Z position before acquisition...")
        hardware.move_to_position(Position(z=ctx.hint_z))
        logger.info(f"Moved to predicted Z: {ctx.hint_z:.2f} um")

    first_af_idx = ctx.af_positions[0]
    first_af_pos, first_af_filename = ctx.positions[first_af_idx]
    logger.info(f"=== PRE-ACQUISITION AUTOFOCUS at position {first_af_idx} ===")
    logger.info(f"Using diagonal autofocus position: X={first_af_pos.x}, Y={first_af_pos.y}")

    # For rotation modalities, set rotation to autofocus angle
    if ctx.mod_config.autofocus_angle is not None and hasattr(hardware, "set_psg_ticks"):
        af_angle = ctx.mod_config.autofocus_angle
        hardware.set_psg_ticks(af_angle)
        logger.info("Set rotation to %.0f deg for initial autofocus", af_angle)

        # Get autofocus-angle exposure. The user is NOT required to include the
        # AF angle (e.g. 90 deg uncrossed for PPM) in their acquisition angles,
        # so fall back to the calibrated uncrossed exposure when the acquisition
        # does not cover the AF angle. AF brightness check will adapt further.
        af_exposure: Optional[float] = None
        if af_angle in params["angles"]:
            angle_idx = params["angles"].index(af_angle)
            if angle_idx < len(params["exposures"]):
                af_exposure = params["exposures"][angle_idx]
                logger.info(
                    f"AF exposure {af_exposure}ms from acquisition params "
                    f"(angle {af_angle} deg)"
                )

        if af_exposure is None and ctx.jai_calibration is not None:
            uncrossed_exp = (
                ctx.jai_calibration.get("angles", {})
                .get("uncrossed", {})
                .get("exposures_ms", {})
            )
            r = uncrossed_exp.get("r")
            g = uncrossed_exp.get("g")
            b = uncrossed_exp.get("b")
            if r is not None and g is not None and b is not None:
                af_exposure = (float(r) + float(g) + float(b)) / 3.0
                logger.info(
                    f"AF angle {af_angle} deg not in acquisition angles; "
                    f"using calibrated uncrossed exposure (R={r:.2f}, G={g:.2f}, "
                    f"B={b:.2f} -> mean={af_exposure:.2f}ms)"
                )

        if af_exposure is None:
            if params.get("exposures"):
                # Uncrossed is brighter than birefringence/crossed angles, so
                # halve the first acquisition exposure as a conservative start.
                # AF brightness check will double if too dim.
                af_exposure = params["exposures"][0] / 2.0
                logger.warning(
                    f"AF angle {af_angle} deg not in acquisition angles and no "
                    f"uncrossed calibration exposure available; falling back to "
                    f"half the first acquisition exposure ({af_exposure:.2f}ms). "
                    f"Run WB calibration for reliable AF exposure."
                )
            else:
                raise ValueError(
                    f"Cannot determine AF exposure for angle {af_angle} deg: "
                    f"no acquisition exposures and no uncrossed calibration data."
                )

        ctx.exposure_90 = af_exposure

        # Disable per-channel mode before AF, apply analog gains
        if ctx.is_jai_camera:
            try:
                hardware.camera.disable_individual_exposure()
                hardware.camera.disable_individual_gain()
                if ctx.jai_calibration is not None:
                    uncrossed_gains = (
                        ctx.jai_calibration.get("angles", {})
                        .get("uncrossed", {})
                        .get("gains", {})
                    )
                    af_unified_gain = uncrossed_gains.get("unified_gain", 1.0)
                    hardware.camera.set_unified_gain(af_unified_gain)
                    af_analog_red = uncrossed_gains.get("analog_red", 1.0)
                    af_analog_blue = uncrossed_gains.get("analog_blue", 1.0)
                    hardware.camera.set_rb_analog_gains(
                        analog_red=af_analog_red, analog_blue=af_analog_blue
                    )
                    logger.info(
                        "Applied uncrossed calibration for AF: "
                        f"gain={af_unified_gain:.2f}x, "
                        f"R={af_analog_red:.3f}, B={af_analog_blue:.3f}"
                    )
            except Exception as e:
                logger.warning(f"Could not configure camera for AF: {e}")

        hardware.set_exposure(ctx.exposure_90)
        logger.info(f"Set exposure to {ctx.exposure_90}ms for initial autofocus")

    # Calculate direction toward center for tissue search loop
    start_pos = np.array([first_af_pos.x, first_af_pos.y])
    center_pos = np.mean(ctx.xy_positions, axis=0)
    direction = center_pos - start_pos
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)

    # Tissue detection loop: try current position, then move 1 FOV toward center
    max_tissue_search_attempts = 3
    tissue_found = False
    search_pos = Position(first_af_pos.x, first_af_pos.y, hardware.get_current_position().z)
    fov = hardware.get_fov()
    fov_diagonal = np.sqrt(fov[0]**2 + fov[1]**2)

    for attempt in range(max_tissue_search_attempts):
        hardware.move_to_position(search_pos)
        logger.info(
            f"Tissue search attempt {attempt + 1}/{max_tissue_search_attempts}: "
            f"X={search_pos.x:.1f}, Y={search_pos.y:.1f}"
        )

        test_img, _ = hardware.snap_image()

        # Ensure consistent format for tissue detection
        if test_img.dtype in [np.float32, np.float64]:
            if test_img.max() <= 1.0 and test_img.min() >= 0.0:
                test_img = (test_img * 255).astype(np.uint8)
            else:
                test_img = np.clip(test_img, 0, 255).astype(np.uint8)

        # Brightness safety check (strategy-aware)
        af_brightness_attempts = 0
        while af_brightness_attempts < 4:
            bright_ok, bright_stats = ctx.af_strategy.brightness_acceptable(test_img)
            if bright_ok:
                break
            ctx.exposure_90 *= 2.0
            hardware.set_exposure(ctx.exposure_90)
            logger.warning(
                f"AF test image brightness_check failed ({bright_stats}), "
                f"doubling exposure to {ctx.exposure_90:.2f}ms"
            )
            test_img, _ = hardware.snap_image()
            if test_img.dtype in [np.float32, np.float64]:
                if test_img.max() <= 1.0 and test_img.min() >= 0.0:
                    test_img = (test_img * 255).astype(np.uint8)
                else:
                    test_img = np.clip(test_img, 0, 255).astype(np.uint8)
            af_brightness_attempts += 1
        if af_brightness_attempts > 0:
            logger.info(
                f"AF exposure adjusted to {ctx.exposure_90:.2f}ms "
                f"(strategy={ctx.af_strategy_name})"
            )

        # Strategy-aware validity check
        signal_valid, strategy_stats = ctx.af_strategy.is_valid(test_img, logger_=logger)

        if signal_valid:
            logger.info(f"Signal valid at attempt {attempt + 1} (strategy={ctx.af_strategy_name})")
            tissue_found = True
            break

        failure_mode = ctx.af_strategy.on_failure
        logger.warning(
            f"Signal check failed at attempt {attempt + 1} "
            f"(strategy={ctx.af_strategy_name}, on_failure={failure_mode.value}, "
            f"stats={strategy_stats})"
        )
        if failure_mode is StrategyFailureMode.PROCEED:
            logger.info("Strategy failure_mode=PROCEED: breaking search loop and running AF anyway")
            tissue_found = True
            break
        if failure_mode is StrategyFailureMode.MANUAL:
            logger.info("Strategy failure_mode=MANUAL: breaking search loop to pop manual dialog")
            break

        # DEFER: move toward center for next attempt
        if attempt < max_tissue_search_attempts - 1:
            new_xy = np.array([search_pos.x, search_pos.y]) + direction * fov_diagonal
            search_pos = Position(new_xy[0], new_xy[1], search_pos.z)
            logger.info("Moving 1 FOV diagonal toward center for next attempt")

    # Run autofocus (with manual fallback if no tissue found)
    try:
        if tissue_found:
            logger.info("Tissue found - running autofocus with manual fallback")
            initial_z = autofocus_with_manual_fallback(
                hardware=hardware,
                request_manual_focus=ctx.request_manual_focus,
                max_retries=3,
                fallback_z=ctx.hint_z,
                n_steps=ctx.af_n_steps,
                edge_retries=ctx.af_edge_retries,
                search_range=ctx.af_search_range,
                score_metric=ctx.af_score_metric,
                diagnostic_output_path=str(ctx.output_path),
                logger=logger,
            )
        else:
            logger.warning(f"No tissue found after {max_tissue_search_attempts} search attempts")
            logger.warning("Attempting autofocus anyway - will go to manual dialog if it fails")
            initial_z = autofocus_with_manual_fallback(
                hardware=hardware,
                request_manual_focus=ctx.request_manual_focus,
                max_retries=0,
                fallback_z=ctx.hint_z,
                n_steps=ctx.af_n_steps,
                edge_retries=ctx.af_edge_retries,
                search_range=ctx.af_search_range,
                score_metric=ctx.af_score_metric,
                diagnostic_output_path=str(ctx.output_path),
                logger=logger,
            )

        logger.info(f"Initial autofocus completed: Z={initial_z:.2f} um")
        ctx.first_tissue_autofocus_done = True
        ctx.last_af_pos_idx = first_af_idx
        ctx.completed_af_positions.append((first_af_pos.x, first_af_pos.y, initial_z))
        ctx.dynamic_af_positions.discard(first_af_idx)

    except RuntimeError as e:
        logger.error(f"Initial autofocus failed: {e}")
        if "cancelled" in str(e).lower():
            raise _AcquisitionCancelled() from e

    logger.info("=== Starting main acquisition loop ===")


def _handle_tile_autofocus(
    ctx: AcquisitionContext, pos_idx: int, pos, filename: str,
) -> Tuple[bool, str, float, bool, bool]:
    """Decide if AF is needed, move stage, and run AF.

    Returns (needs_af, af_type, drift, af_failed, xy_move_pending).

    Mutates ctx: last_af_pos_idx, completed_af_positions, dynamic_af_positions,
    deferred_af_positions, first_tissue_autofocus_done, exposure_90.
    """
    from microscope_control.autofocus.strategies import StrategyFailureMode

    logger = ctx.logger
    hardware = ctx.hardware
    params = ctx.params
    af_type_for_this_tile = "none"
    drift_for_this_tile = 0.0
    af_failed_for_this_tile = False

    # Determine the best Z for this tile position.
    needs_af = pos_idx in ctx.dynamic_af_positions

    # Skip AF on the last tile -- the corrected Z has no downstream
    # snap to benefit from it, and even a successful drift update
    # would never be applied. Saves ~3-5s per acquisition for free.
    if needs_af and pos_idx == len(ctx.positions) - 1:
        logger.info(
            "  Skipping AF at final position %d/%d (no downstream tiles to use the result)",
            pos_idx, len(ctx.positions) - 1,
        )
        needs_af = False

    # Safety net 1: index gap check
    index_gap_threshold = ctx.af_gap_index_multiplier * ctx.af_n_tiles
    if not needs_af and ctx.last_af_pos_idx >= 0:
        gap = pos_idx - ctx.last_af_pos_idx
        if gap > index_gap_threshold:
            needs_af = True
            logger.info(
                "  Forcing AF: index gap of %d positions since last AF "
                "(threshold: %d = %dx%d)",
                gap, index_gap_threshold,
                ctx.af_gap_index_multiplier, ctx.af_n_tiles,
            )

    # Safety net 2: spatial gap check
    spatial_gap_threshold = ctx.af_gap_spatial_multiplier * ctx.af_min_distance
    if not needs_af and ctx.completed_af_positions:
        tile_xy = np.array([[pos.x, pos.y]])
        af_xy = np.array([(ax, ay) for ax, ay, _ in ctx.completed_af_positions])
        nearest_af_dist = float(np.min(_cdist_scipy(tile_xy, af_xy)))
        if nearest_af_dist > spatial_gap_threshold:
            needs_af = True
            logger.info(
                "  Forcing AF: spatial distance %.0f um to nearest AF "
                "exceeds threshold %.0f um (%.1fx af_min_distance)",
                nearest_af_dist, spatial_gap_threshold,
                ctx.af_gap_spatial_multiplier,
            )

    # For non-AF tiles, move Z to the spatially nearest AF's Z
    if not needs_af and ctx.completed_af_positions:
        tile_xy = np.array([[pos.x, pos.y]])
        af_xy = np.array([(ax, ay) for ax, ay, _ in ctx.completed_af_positions])
        nearest_idx = int(np.argmin(_cdist_scipy(tile_xy, af_xy)[0]))
        nearest_z = ctx.completed_af_positions[nearest_idx][2]
        current_z = hardware.get_current_position().z
        if abs(nearest_z - current_z) > 0.1:
            hardware.move_to_position(Position(z=nearest_z))
            logger.debug(
                "  Nearest-AF Z correction: %.2f -> %.2f um "
                "(nearest AF at X=%.0f, Y=%.0f, dist=%.0f um)",
                current_z, nearest_z,
                ctx.completed_af_positions[nearest_idx][0],
                ctx.completed_af_positions[nearest_idx][1],
                float(_cdist_scipy(tile_xy, [af_xy[nearest_idx]])[0][0]),
            )
        pos.z = nearest_z
    else:
        pos.z = hardware.get_current_position().z

    # Stage move with retry
    logger.debug(f"Moving to position: X={pos.x}, Y={pos.y}, Z={pos.z}")
    t0 = time.perf_counter()
    move_succeeded = False
    last_move_error = None
    for move_attempt in range(3):
        try:
            if needs_af:
                hardware.move_to_position(pos)
            else:
                hardware.move_xy_no_wait(pos.x, pos.y)
            move_succeeded = True
            break
        except Exception as move_err:
            last_move_error = move_err
            if move_attempt < 2:
                logger.warning(
                    f"Stage move failed (attempt {move_attempt + 1}/3): {move_err} "
                    f"-- retrying in 2s"
                )
                time.sleep(2.0)
            else:
                logger.error(
                    f"Stage move failed after 3 attempts: {move_err} "
                    f"-- requesting user intervention"
                )

    if not move_succeeded:
        if ctx.request_hardware_error_recovery is not None:
            error_detail = str(last_move_error) if last_move_error else "Stage move failed after 3 attempts"
            logger.info("Pausing acquisition for user to resolve stage error")
            user_choice = ctx.request_hardware_error_recovery(error_detail)
            if user_choice == "cancel":
                logger.warning("User cancelled acquisition during stage error recovery")
                raise _AcquisitionCancelled()
            elif user_choice == "skip":
                logger.info(f"User chose to skip position {pos_idx}")
                # Return with af_type="skipped" so caller knows to skip
                return needs_af, "skipped", drift_for_this_tile, True, False
            # "retry" -- try one more time after user intervention
            try:
                if needs_af:
                    hardware.move_to_position(pos)
                else:
                    hardware.move_xy_no_wait(pos.x, pos.y)
            except Exception as final_err:
                raise RuntimeError(f"Stage move still failing after user intervention: {final_err}")
        else:
            raise RuntimeError("Stage move failed after 3 attempts")

    xy_move_pending = not needs_af
    log_timing(logger, "Stage XY movement command", t0)

    if not needs_af:
        return needs_af, af_type_for_this_tile, drift_for_this_tile, af_failed_for_this_tile, xy_move_pending

    # Perform autofocus
    logger.info(f"Checking for autofocus at position {pos_idx}: X={pos.x}, Y={pos.y}, Z={pos.z}")

    # For rotation modalities, always autofocus at the configured angle
    if ctx.mod_config.autofocus_angle is not None and hasattr(hardware, "set_psg_ticks"):
        af_angle = ctx.mod_config.autofocus_angle
        t_rot = time.perf_counter()
        hardware.set_psg_ticks(af_angle)
        t_rot = log_timing(logger, "Rotation to %.0fdeg for autofocus" % af_angle, t_rot)
        logger.info("Set rotation to %.0f deg for autofocus", af_angle)
        if af_angle in params["angles"]:
            angle_idx = params["angles"].index(af_angle)
            if angle_idx < len(params["exposures"]):
                ctx.exposure_90 = params["exposures"][angle_idx]

        # Disable per-channel mode and apply analog gains
        if ctx.is_jai_camera:
            try:
                hardware.camera.disable_individual_exposure()
                hardware.camera.disable_individual_gain()
                if ctx.jai_calibration is not None:
                    uncrossed_gains = (
                        ctx.jai_calibration.get("angles", {})
                        .get("uncrossed", {})
                        .get("gains", {})
                    )
                    af_unified_gain = uncrossed_gains.get("unified_gain", 1.0)
                    hardware.camera.set_unified_gain(af_unified_gain)
                    hardware.camera.set_rb_analog_gains(
                        analog_red=uncrossed_gains.get("analog_red", 1.0),
                        analog_blue=uncrossed_gains.get("analog_blue", 1.0),
                    )
            except Exception as e:
                logger.warning(f"Could not configure camera for AF: {e}")

        t_exp = time.perf_counter()
        hardware.set_exposure(ctx.exposure_90)
        t_exp = log_timing(logger, "Set exposure for tissue detection", t_exp)
        logger.info(f"Set exposure to {ctx.exposure_90}ms for 90 deg tissue detection")

    # Take a quick image to assess tissue content
    t_snap = time.perf_counter()
    test_img, _ = hardware.snap_image()
    t_snap = log_timing(logger, "Snap test image for tissue detection", t_snap)

    if test_img.dtype in [np.float32, np.float64]:
        if test_img.max() <= 1.0 and test_img.min() >= 0.0:
            test_img = (test_img * 255).astype(np.uint8)
        else:
            test_img = np.clip(test_img, 0, 255).astype(np.uint8)

    # Brightness safety check (strategy-aware)
    af_brightness_attempts = 0
    while af_brightness_attempts < 4:
        bright_ok, bright_stats = ctx.af_strategy.brightness_acceptable(test_img)
        if bright_ok:
            break
        ctx.exposure_90 *= 2.0
        hardware.set_exposure(ctx.exposure_90)
        logger.warning(
            f"AF drift-check image brightness_check failed ({bright_stats}), "
            f"doubling exposure to {ctx.exposure_90:.2f}ms"
        )
        test_img, _ = hardware.snap_image()
        if test_img.dtype in [np.float32, np.float64]:
            if test_img.max() <= 1.0 and test_img.min() >= 0.0:
                test_img = (test_img * 255).astype(np.uint8)
            else:
                test_img = np.clip(test_img, 0, 255).astype(np.uint8)
        af_brightness_attempts += 1
    if af_brightness_attempts > 0:
        logger.info(
            f"AF drift-check exposure adjusted to {ctx.exposure_90:.2f}ms "
            f"(strategy={ctx.af_strategy_name})"
        )

    # Strategy-aware validity check
    signal_valid, strategy_stats = ctx.af_strategy.is_valid(test_img, logger_=logger)

    failure_mode = ctx.af_strategy.on_failure
    should_run_af = signal_valid or failure_mode is StrategyFailureMode.PROCEED

    if should_run_af:
        logger.info(
            f"Running drift-check AF (strategy={ctx.af_strategy_name}, "
            f"valid={signal_valid}, stats={strategy_stats})"
        )

        if not ctx.first_tissue_autofocus_done:
            logger.info("  First tissue position - using STANDARD autofocus for accuracy")
            t_af = time.perf_counter()
            new_z = autofocus_with_manual_fallback(
                hardware=hardware,
                logger=logger,
                request_manual_focus=ctx.request_manual_focus,
                max_retries=3,
                fallback_z=ctx.hint_z,
                move_stage_to_estimate=True,
                n_steps=ctx.af_n_steps,
                search_range=ctx.af_search_range,
                interp_strength=ctx.af_interp_strength,
                interp_kind=ctx.af_interp_kind,
                score_metric=ctx.af_score_metric,
                diagnostic_output_path=ctx.output_path,
                position_index=pos_idx,
            )
            t_af = log_timing(logger, "STANDARD autofocus", t_af)
            ctx.first_tissue_autofocus_done = True
            af_type_for_this_tile = "standard"
            logger.info(f"  Standard autofocus :: New Z {new_z}")
        else:
            SMALL_GRID_SKIP_DRIFT_MAX_TILES = 9
            if len(ctx.positions) <= SMALL_GRID_SKIP_DRIFT_MAX_TILES:
                z_before_adaptive = hardware.get_current_position().z
                logger.info(
                    "  Small grid (%d tiles <= %d) - skipping sweep drift check, "
                    "reusing Z=%.2f um from initial AF",
                    len(ctx.positions), SMALL_GRID_SKIP_DRIFT_MAX_TILES, z_before_adaptive,
                )
                new_z = z_before_adaptive
                af_type_for_this_tile = "skip_small_grid"
                drift_for_this_tile = 0.0
            else:
                z_before_adaptive = hardware.get_current_position().z
                logger.info("  Subsequent tissue position - using SWEEP drift check for speed")
                t_af = time.perf_counter()
                new_z = hardware.autofocus_sweep_drift_check(
                    range_um=ctx.af_sweep_range_um,
                    n_steps=ctx.af_sweep_n_steps,
                    score_metric=ctx.af_score_metric_name,
                    max_retries=ctx.af_edge_retries,
                )
                t_af = log_timing(logger, "SWEEP drift check", t_af)
                drift = new_z - z_before_adaptive
                af_type_for_this_tile = "sweep"
                drift_for_this_tile = drift
                logger.info(f"  Sweep drift check :: New Z {new_z} (drift: {drift:+.2f} um)")

        # Track this position as the last AF position
        ctx.last_af_pos_idx = pos_idx
        af_z = hardware.get_current_position().z
        if af_type_for_this_tile == "sweep" and abs(drift_for_this_tile) < 0.05:
            logger.info(
                "  Sweep produced no drift -- not recording "
                "in AF map (prevents stale Z propagation)"
            )
        else:
            ctx.completed_af_positions.append((pos.x, pos.y, af_z))
    else:
        # Strategy rejected this tile
        af_failed_for_this_tile = True
        logger.warning(
            f"Strategy {ctx.af_strategy_name} rejected tile {pos_idx} "
            f"(on_failure={failure_mode.value}, stats={strategy_stats}) - deferring autofocus"
        )

        ctx.dynamic_af_positions.discard(pos_idx)
        ctx.deferred_af_positions.add(pos_idx)

        next_af_pos = AutofocusUtils.defer_autofocus_to_next_tile(
            current_pos_idx=pos_idx,
            original_af_positions=ctx.af_positions,
            total_positions=len(ctx.positions),
            af_min_distance=ctx.af_min_distance,
            positions=ctx.xy_positions,
            logger=logger,
        )
        if next_af_pos is not None and next_af_pos < len(ctx.positions):
            ctx.dynamic_af_positions.add(next_af_pos)
            logger.info(f"Added position {next_af_pos} to autofocus queue")
        else:
            logger.warning("Could not find suitable position to defer autofocus to")

    # Drain pending background TIFF writes
    if ctx.write_pool is not None and ctx.write_pool.pending_count > 0:
        t_drain = time.perf_counter()
        n_drained = ctx.write_pool.pending_count
        failed = ctx.write_pool.drain()
        log_timing(logger, f"Drain {n_drained} pending writes ({failed} failed)", t_drain)

    # NOTE: No WB restore needed here. The angle loop's first iteration
    # calls apply_jai_calibration_for_angle (per_angle mode) or
    # camera.apply_settings (simple mode), which re-enables per-channel
    # mode as part of its operation. The previous explicit restore was
    # redundant and cost ~1.3s per AF tile (614 duplicate calls = 13 min
    # wasted in a 2,399-tile acquisition, April 2026 log analysis).

    return needs_af, af_type_for_this_tile, drift_for_this_tile, af_failed_for_this_tile, xy_move_pending


def _acquire_tile_angles(ctx: AcquisitionContext, pos_idx: int, pos, filename: str,
                         current_stage_pos, xy_move_pending: bool) -> Tuple[dict, dict, dict, bool]:
    """Acquire all angles (and Z-planes) for a single tile position.

    Returns (tile_worst_sat, tile_role_sat, tile_stats, xy_move_pending updated).
    tile_role_sat collects worst-pct broken down by SaturationRole so the
    QuPath measurement table can filter PPM tiles where small-angle (low
    signal) channels saturated, ignoring uncrossed (90 deg) which is
    intentionally bright. tile_stats collects p1/p99/mean/std/dynamic_range
    per channel, aggregated across angles -- p1 is min seen, p99 + dynamic
    range are max seen, mean and std are angle-averaged.
    """
    logger = ctx.logger
    hardware = ctx.hardware
    params = ctx.params
    tile_worst_sat = {"R": 0.0, "G": 0.0, "B": 0.0}
    tile_role_sat = {SATURATION_ROLE_LOW: 0.0, SATURATION_ROLE_HIGH: 0.0, SATURATION_ROLE_NORMAL: 0.0}
    tile_stats: dict = {}

    center_z = current_stage_pos.z
    z_stack_images = {}
    angle_images = {}

    for z_idx, z_offset in enumerate(ctx.z_offsets):
        if ctx.z_stack_enabled and z_offset != 0.0:
            target_z = center_z + z_offset
            hardware.move_to_position(Position(z=target_z))
            logger.debug(
                "Z-stack: plane %d/%d, Z=%.2f (offset=%+.1f)",
                z_idx + 1, len(ctx.z_offsets), target_z, z_offset,
            )

        for angle_idx, angle in enumerate(params["angles"]):
            if ctx.is_cancelled():
                raise _AcquisitionCancelled()

            angle_start = time.perf_counter()
            t_rot = time.perf_counter()
            hardware.set_psg_ticks_no_wait(angle)
            t_exp = time.perf_counter()

            if ctx.wb_mode == "camera_awb":
                if angle_idx < len(params["exposures"]):
                    exposure_ms = params["exposures"][angle_idx]
                    hardware.set_exposure(exposure_ms)
                angle_name = angle_to_name(angle, modality=ctx.modality)
                if ctx.camera_awb_gains and angle_name in ctx.camera_awb_gains:
                    try:
                        gain_val = ctx.camera_awb_gains[angle_name]
                        hardware.camera.set_unified_gain(gain_val)
                        logger.info(f"  Camera AWB: unified gain={gain_val:.2f} for {angle_name}")
                    except Exception as e:
                        logger.debug(f"Could not set unified gain: {e}")

            elif ctx.wb_mode == "simple" and ctx.simple_wb_data:
                angle_name = angle_to_name(angle, modality=ctx.modality)
                sw_angles = ctx.simple_wb_data.get("angles", {})
                if angle_name in sw_angles:
                    angle_sw = sw_angles[angle_name]
                    try:
                        exp_r = angle_sw["r"]
                        exp_g = angle_sw["g"]
                        exp_b = angle_sw["b"]
                        is_unified = abs(exp_r - exp_g) < 0.01 and abs(exp_g - exp_b) < 0.01
                        sw_gain = angle_sw.get("unified_gain", 1.0)
                        hardware.camera.apply_settings(
                            exposures=({"all": exp_g} if is_unified
                                       else {"r": exp_r, "g": exp_g, "b": exp_b}),
                            unified_gain=sw_gain,
                            analog_red=ctx.simple_wb_analog_red,
                            analog_blue=ctx.simple_wb_analog_blue,
                            individual_exposure=not is_unified,
                        )
                        logger.debug(
                            "  Simple WB: R=%.1fms, G=%.1fms, B=%.1fms "
                            "(scale=%sx, gain=%.2f, aR=%.3f, aB=%.3f)",
                            exp_r, exp_g, exp_b,
                            angle_sw.get("scale", "?"), sw_gain,
                            ctx.simple_wb_analog_red, ctx.simple_wb_analog_blue,
                        )
                    except Exception as e:
                        logger.warning(f"Simple WB failed for {angle_name}: {e}")
                        if angle_idx < len(params["exposures"]):
                            hardware.set_exposure(params["exposures"][angle_idx])
                else:
                    logger.info(f"  Simple WB: no data for {angle_name}, using calibration with scale")
                    if ctx.jai_calibration is not None:
                        applied, _ = apply_jai_calibration_for_angle(
                            hardware=hardware,
                            jai_calibration=ctx.jai_calibration,
                            angle=angle,
                            per_angle=False,
                            logger=logger,
                        )
                        if not applied and angle_idx < len(params["exposures"]):
                            hardware.set_exposure(params["exposures"][angle_idx])
                    elif angle_idx < len(params["exposures"]):
                        hardware.set_exposure(params["exposures"][angle_idx])

            elif ctx.wb_mode == "simple" and ctx.jai_calibration is not None:
                applied, _ = apply_jai_calibration_for_angle(
                    hardware=hardware,
                    jai_calibration=ctx.jai_calibration,
                    angle=angle,
                    per_angle=False,
                    logger=logger,
                )
                if not applied and angle_idx < len(params["exposures"]):
                    hardware.set_exposure(params["exposures"][angle_idx])

            elif ctx.jai_calibration is not None:
                applied, _ = apply_jai_calibration_for_angle(
                    hardware=hardware,
                    jai_calibration=ctx.jai_calibration,
                    angle=angle,
                    per_angle=ctx.white_balance_per_angle,
                    logger=logger,
                )
                if not applied and angle_idx < len(params["exposures"]):
                    try:
                        hardware.camera.disable_individual_exposure()
                        hardware.camera.disable_individual_gain()
                        hardware.camera.set_rb_analog_gains(analog_red=1.0, analog_blue=1.0)
                    except Exception:
                        pass
                    exposure_ms = params["exposures"][angle_idx]
                    hardware.set_exposure(exposure_ms)
                    logger.info(f"  JAI calibration failed, using single exposure: {exposure_ms}ms")
            elif angle_idx < len(params["exposures"]):
                exposure_ms = params["exposures"][angle_idx]
                hardware.set_exposure(exposure_ms)
            t_exp = log_timing(logger, f"Set exposure for angle {angle}deg", t_exp)

            hardware.wait_for_rotation()
            t_rot = log_timing(logger, f"Rotation to {angle}deg", t_rot)

            if xy_move_pending:
                hardware.wait_for_xy()
                xy_move_pending = False

            t_snap = time.perf_counter()
            image, metadata = hardware.snap_image(debayering=False)
            t_snap = log_timing(logger, f"Snap image at {angle}deg (includes camera+USB+internal processing)", t_snap)

            if image is None:
                logger.error(f"Failed to acquire image at angle {angle}")
                continue

            t_stats = time.perf_counter()
            img_mean = image.mean((0, 1))
            t_stats = log_timing(logger, f"Calculate image stats at {angle}deg", t_stats)
            logger.debug(f"  Image shape: {image.shape}, mean: {img_mean}")

            sat_warn_threshold = (
                101.0 if ctx.sat_monitor.should_suppress_warnings(angle) else 1.0
            )
            sat_result = _check_saturation(
                image, f"tile {filename} at {angle}deg", logger,
                threshold_pct=sat_warn_threshold,
            )
            if sat_result:
                for ch in ("R", "G", "B"):
                    if sat_result.get(ch, 0) > tile_worst_sat[ch]:
                        tile_worst_sat[ch] = sat_result[ch]
                # Per-role saturation aggregation: track the worst per-channel
                # pct seen at any angle that classifies into each role. PPM
                # uncrossed (~90 deg) hits role_high; crossed/positive/negative
                # hit role_low. Non-PPM modalities collapse all into role_normal.
                role = _saturation_role_for(ctx.modality, angle)
                worst_this_angle = max(sat_result.values()) if sat_result else 0.0
                if worst_this_angle > tile_role_sat.get(role, 0.0):
                    tile_role_sat[role] = worst_this_angle

            # Per-tile percentile / mean / std stats, aggregated across angles.
            # P1 catches under-exposure (low p99) -- the same QuPath measurement
            # table now surfaces both saturation and dim-tile failure modes.
            try:
                _accumulate_tile_stats(tile_stats, _compute_tile_stats(image))
            except Exception as stats_err:
                logger.debug(f"  Stats compute failed at {angle}deg: {stats_err}")

            if ctx.sat_monitor.check_tile(
                sat_result, angle, pos_idx, filename,
                stage_x=current_stage_pos.x,
                stage_y=current_stage_pos.y,
                stage_z=current_stage_pos.z,
            ):
                ctx.sat_monitor.log_summary()
                raise RuntimeError(ctx.sat_monitor.abort_reason)

            # Save raw image
            if ctx.save_raw_tiles:
                raw_output_path = ctx.output_path.parent / "Raw" / ctx.output_path.name
                raw_image_path = raw_output_path / str(angle) / filename
                t_mkdir = time.perf_counter()
                if not raw_image_path.parent.exists():
                    raw_image_path.parent.mkdir(parents=True, exist_ok=True)
                t_mkdir = log_timing(logger, f"Create directories at {angle}deg", t_mkdir)
                try:
                    write_position_metadata(
                        ctx.metadata_txt_for_positions, raw_image_path, hardware, ctx.modality
                    )
                    raw_pixel_size = hardware.get_pixel_size_um()
                    ctx.write_pool.submit(
                        ome_tiff_writer,
                        filename=str(raw_image_path),
                        pixel_size_um=raw_pixel_size,
                        data=image,
                    )
                    logger.info(f"  Queued raw image write: {raw_image_path}")
                except Exception as e:
                    logger.warning(f"  Failed to queue raw image: {e}")

            # Apply background correction
            if (
                ctx.background_correction_enabled
                and angle in ctx.background_images
                and angle not in ctx.background_disabled_angles
            ):
                bg_img = ctx.background_images[angle]
                logger.debug(f"  Applying background correction for {angle} degrees")
                logger.debug(f"    Background stats: mean={bg_img.mean():.1f}, std={bg_img.std():.1f}")
                t_bg = time.perf_counter()
                image = BackgroundCorrectionUtils.apply_flat_field_correction(
                    image,
                    ctx.background_images[angle],
                    ctx.background_scaling_factors[angle],
                    method=ctx.background_correction_method,
                )
                t_bg = log_timing(logger, f"Background correction at {angle}deg", t_bg)
                logger.debug(f"    Correction applied with method: {ctx.background_correction_method}")
                logger.debug(f"    Post-correction RGB means: {image.mean(axis=(0,1))}")
                if not ctx.sat_monitor._is_uncrossed(angle):
                    _check_saturation(image, f"post-correction tile {filename} at {angle}deg", logger)
            elif ctx.background_correction_enabled and angle in ctx.background_disabled_angles:
                logger.info(
                    f"  Background correction SKIPPED for {angle} deg "
                    "(disabled by acquisition parameters - exposure mismatch or missing background)"
                )
            elif ctx.background_correction_enabled and angle not in ctx.background_images:
                logger.info(
                    f"  Background correction SKIPPED for {angle} deg "
                    "(no background image available)"
                )

            # Apply white balance (software-only, skip when hardware WB active)
            if ctx.white_balance_enabled and ctx.jai_calibration is None and ctx.wb_mode not in ("camera_awb", "simple"):
                if angle in ctx.angles_wb:
                    wb_profile = ctx.angles_wb[angle]
                else:
                    wb_profile = [1.0, 1.0, 1.0]
                    logger.warning(f"    No white balance profile for {angle} deg, using neutral")
                t_wb = time.perf_counter()
                gain = calculate_luminance_gain(*wb_profile)
                image = hardware.white_balance(image, white_balance_profile=wb_profile, gain=gain)
                t_wb = log_timing(logger, f"White balance at {angle}deg", t_wb)
                logger.info(
                    f"  Applied software white balance: R={wb_profile[0]:.2f}, G={wb_profile[1]:.2f}, B={wb_profile[2]:.2f}"
                )
            elif ctx.white_balance_enabled and (ctx.jai_calibration is not None or ctx.wb_mode in ("camera_awb", "simple")):
                logger.debug(f"  Software WB skipped (hardware WB active for {angle} deg, mode={ctx.wb_mode})")

            # Save or accumulate image depending on Z-stack mode
            if not ctx.z_stack_enabled:
                image_path = ctx.output_path / str(angle) / filename
                if image_path.parent.exists():
                    proc_pixel_size = hardware.get_pixel_size_um()
                    ctx.write_pool.submit(
                        ome_tiff_writer,
                        filename=str(image_path),
                        pixel_size_um=proc_pixel_size,
                        data=image,
                    )
                    angle_images[angle] = image
                else:
                    logger.error(f"Failed to save {image_path} - parent directory missing")
            else:
                z_stack_images.setdefault(angle, []).append(image)
                if ctx.save_raw_tiles:
                    z_plane_path = ctx.output_path / str(angle) / f"z{z_idx:03d}" / filename
                    z_plane_path.parent.mkdir(parents=True, exist_ok=True)
                    ctx.write_pool.submit(
                        ome_tiff_writer,
                        filename=str(z_plane_path),
                        pixel_size_um=hardware.get_pixel_size_um(),
                        data=image,
                    )

            ctx.image_count += 1
            ctx.update_progress(ctx.image_count, ctx.total_images)

            angle_elapsed_ms = (time.perf_counter() - angle_start) * 1000
            logger.debug(f"  [TIMING] Total for angle {angle}deg: {angle_elapsed_ms:.1f}ms")

    # Z-stack projection
    if ctx.z_stack_enabled and ctx.projection_fn is not None:
        for angle in params["angles"]:
            if angle in z_stack_images and len(z_stack_images[angle]) > 0:
                projected = ctx.projection_fn(z_stack_images[angle])
                angle_images[angle] = projected
                image_path = ctx.output_path / str(angle) / filename
                if image_path.parent.exists():
                    ctx.write_pool.submit(
                        ome_tiff_writer,
                        filename=str(image_path),
                        pixel_size_um=hardware.get_pixel_size_um(),
                        data=projected,
                    )
        logger.info(
            "Z-stack projection (%s) computed for %d angles",
            params.get("z_projection", "max"), len(z_stack_images),
        )
        hardware.move_to_position(Position(z=center_z))

    # Create birefringence image
    positive_angles = [a for a in angle_images.keys() if a > 0 and a != 90]
    negative_angles = [a for a in angle_images.keys() if a < 0]
    logger.debug(
        f"Biref check: angle_images keys={list(angle_images.keys())}, "
        f"positive={positive_angles}, negative={negative_angles}"
    )

    if positive_angles and negative_angles:
        pos_angle = min(positive_angles)
        neg_angle = max(negative_angles)
        biref_dir = ctx.output_path / f"{pos_angle}.biref"
        tile_config_source = ctx.output_path / str(pos_angle) / "TileConfiguration.txt"

        from ppm_library.imaging.writer import TifWriterUtils as PpmWriterUtils
        biref_pixel_size = hardware.get_pixel_size_um()
        biref_pos_img = angle_images[pos_angle]
        biref_neg_img = angle_images[neg_angle]
        ctx.write_pool.submit(
            PpmWriterUtils.create_normalized_birefringence_tile,
            pos_image=biref_pos_img,
            neg_image=biref_neg_img,
            output_dir=biref_dir,
            filename=filename,
            pixel_size_um=biref_pixel_size,
            tile_config_source=tile_config_source,
            logger=logger,
            min_intensity=params.get("biref_min_intensity", 0),
        )
    else:
        logger.warning(
            f"Skipping birefringence for tile {filename}: "
            f"need both positive (>0, !=90) and negative (<0) angles "
            f"but got angles={list(angle_images.keys())}"
        )

    tile_stats.pop("__counts__", None)
    return tile_worst_sat, tile_role_sat, tile_stats, xy_move_pending


def _acquire_tile_channels(ctx: AcquisitionContext, pos, filename: str,
                           current_stage_pos) -> Tuple[dict, dict, dict]:
    """Acquire all channels for a single tile position (widefield IF).

    Returns (tile_worst_sat, tile_role_sat, tile_stats).
    Channel-based modalities have no angle concept, so all saturation is
    reported under SaturationRole.SIGNAL_NORMAL.
    """
    logger = ctx.logger
    hardware = ctx.hardware
    tile_worst_sat = {}
    tile_role_sat = {SATURATION_ROLE_LOW: 0.0, SATURATION_ROLE_HIGH: 0.0, SATURATION_ROLE_NORMAL: 0.0}
    tile_stats: dict = {}

    CHANNEL_SAT_RUNAWAY_N = 3
    CHANNEL_SAT_PCT_THRESHOLD = 5.0

    channel_plan = resolve_channel_plan(
        ctx.ppm_settings,
        ctx.params.get("scan_type", ""),
        ctx.params.get("channels", []) or [],
        ctx.params.get("channel_exposures", []) or [],
        channel_intensity_overrides=ctx.params.get("channel_intensities") or None,
    )

    center_z = current_stage_pos.z

    for ch_entry in channel_plan:
        apply_channel_hardware_state(
            hardware, ch_entry, logger, preset_cache=ctx.channel_preset_cache
        )
        exposure_ms = float(ch_entry.get("exposure_ms") or 0)
        if exposure_ms > 0:
            hardware.set_exposure(exposure_ms)
            logger.debug("Channel %s: set exposure to %.2f ms", ch_entry["id"], exposure_ms)

        ch_id = ch_entry["id"]
        z_stack_planes = []  # accumulate Z planes when z_stack_enabled
        worst_sat_for_channel = {}

        for z_idx, z_offset in enumerate(ctx.z_offsets):
            # Move Z if doing Z-stack (skip for single-plane 2D)
            if ctx.z_stack_enabled and z_offset != 0.0:
                target_z = center_z + z_offset
                hardware.move_to_position(Position(z=target_z))
                logger.debug(
                    "Channel %s Z-stack: plane %d/%d, Z=%.2f (offset=%+.1f)",
                    ch_id, z_idx + 1, len(ctx.z_offsets), target_z, z_offset,
                )

            image, metadata = hardware.snap_image()

            # Per-channel flat-field correction
            if ctx.channel_background_images and ch_id in ctx.channel_background_images:
                try:
                    image = BackgroundCorrectionUtils.apply_flat_field_correction(
                        image,
                        ctx.channel_background_images[ch_id],
                        scaling_factor=1.0,
                        method=ctx.background_correction_method or "divide",
                    )
                    logger.debug(
                        "  Applied %s background for channel %s",
                        ctx.background_correction_method or "divide",
                        ch_id,
                    )
                except Exception as bg_e:
                    logger.warning("  Channel %s background correction failed: %s", ch_id, bg_e)

            sat_result = _check_saturation(image, f"tile[{ch_id}]", logger)
            if sat_result:
                for ch_key, pct in sat_result.items():
                    if pct > worst_sat_for_channel.get(ch_key, 0):
                        worst_sat_for_channel[ch_key] = pct
                # Channel-based: no per-angle distinction, accumulate into normal role
                worst_this_channel = max(sat_result.values()) if sat_result else 0.0
                if worst_this_channel > tile_role_sat[SATURATION_ROLE_NORMAL]:
                    tile_role_sat[SATURATION_ROLE_NORMAL] = worst_this_channel

            try:
                _accumulate_tile_stats(tile_stats, _compute_tile_stats(image))
            except Exception as stats_err:
                logger.debug(f"  Stats compute failed for {ch_id}: {stats_err}")

            if not ctx.z_stack_enabled:
                # 2D mode: save directly
                image_path = ctx.output_path / str(ch_id) / filename
                if image_path.parent.exists():
                    bf_pixel_size = hardware.get_pixel_size_um()
                    ctx.write_pool.submit(
                        ome_tiff_writer,
                        filename=str(image_path),
                        pixel_size_um=bf_pixel_size,
                        data=image,
                    )
                    ctx.image_count += 1
                    ctx.update_progress(ctx.image_count, ctx.total_images)
                try:
                    write_position_metadata(ctx.metadata_txt_for_positions, image_path, hardware, ctx.modality)
                except Exception as e:
                    logger.warning(f"  Failed to write position text {ctx.metadata_txt_for_positions}: {e}")
            else:
                # Z-stack mode: accumulate for projection
                z_stack_planes.append(image)
                if ctx.save_raw_tiles:
                    z_plane_path = ctx.output_path / str(ch_id) / f"z{z_idx:03d}" / filename
                    z_plane_path.parent.mkdir(parents=True, exist_ok=True)
                    ctx.write_pool.submit(
                        ome_tiff_writer,
                        filename=str(z_plane_path),
                        pixel_size_um=hardware.get_pixel_size_um(),
                        data=image,
                    )
                ctx.image_count += 1
                ctx.update_progress(ctx.image_count, ctx.total_images)

        # Z-stack projection per channel: reduce planes to single 2D image
        if ctx.z_stack_enabled and ctx.projection_fn is not None and z_stack_planes:
            projected = ctx.projection_fn(z_stack_planes)
            image_path = ctx.output_path / str(ch_id) / filename
            if image_path.parent.exists():
                ctx.write_pool.submit(
                    ome_tiff_writer,
                    filename=str(image_path),
                    pixel_size_um=hardware.get_pixel_size_um(),
                    data=projected,
                )
            logger.info(
                "Channel %s Z-stack projection (%s) computed for %d planes",
                ch_id, ctx.params.get("z_projection", "max"), len(z_stack_planes),
            )
            try:
                write_position_metadata(ctx.metadata_txt_for_positions, image_path, hardware, ctx.modality)
            except Exception as e:
                logger.warning(f"  Failed to write position text {ctx.metadata_txt_for_positions}: {e}")

        # Restore Z to center for next channel
        if ctx.z_stack_enabled:
            hardware.move_to_position(Position(z=center_z))

        # Aggregate per-channel saturation into tile_worst_sat + runaway detection
        if worst_sat_for_channel:
            for ch_key, pct in worst_sat_for_channel.items():
                key = f"{ch_id}/{ch_key}"
                if pct > tile_worst_sat.get(key, 0):
                    tile_worst_sat[key] = pct
            worst_channel_sat = max(worst_sat_for_channel.values(), default=0.0)
            if worst_channel_sat > CHANNEL_SAT_PCT_THRESHOLD:
                ctx.channel_consecutive_saturated[ch_id] = (
                    ctx.channel_consecutive_saturated.get(ch_id, 0) + 1
                )
                if ctx.channel_consecutive_saturated[ch_id] == CHANNEL_SAT_RUNAWAY_N:
                    logger.error(
                        "CHANNEL SATURATION RUNAWAY: channel %s has "
                        "exceeded %.1f%% worst-channel saturation on "
                        "%d consecutive tiles (current: %.1f%%). "
                        "Consider cancelling and lowering the %s "
                        "intensity before more imaging time is wasted.",
                        ch_id, CHANNEL_SAT_PCT_THRESHOLD,
                        CHANNEL_SAT_RUNAWAY_N, worst_channel_sat, ch_id,
                    )
            else:
                ctx.channel_consecutive_saturated[ch_id] = 0
        else:
            ctx.channel_consecutive_saturated[ch_id] = 0

    tile_stats.pop("__counts__", None)
    return tile_worst_sat, tile_role_sat, tile_stats


def _acquire_tile_single(ctx: AcquisitionContext, pos, filename: str,
                         current_stage_pos) -> Tuple[dict, dict, dict]:
    """Acquire image(s) for a non-rotation tile (BF, fluorescence).

    Supports Z-stack: when z_stack_enabled, iterates Z-offsets around
    center_z, accumulates planes, and saves a projected 2D image.

    Returns (tile_worst_sat, tile_role_sat, tile_stats). Single-acquisition
    modalities have no angle concept, so all saturation lands in
    SaturationRole.SIGNAL_NORMAL.
    """
    logger = ctx.logger
    hardware = ctx.hardware
    params = ctx.params
    tile_worst_sat = {"R": 0.0, "G": 0.0, "B": 0.0}
    tile_role_sat = {SATURATION_ROLE_LOW: 0.0, SATURATION_ROLE_HIGH: 0.0, SATURATION_ROLE_NORMAL: 0.0}
    tile_stats: dict = {}

    # Set exposure explicitly from params
    if params.get("exposures"):
        exposure_ms = float(params["exposures"][0])
        hardware.set_exposure(exposure_ms)
        logger.debug("Single-image path: set exposure to %.2f ms", exposure_ms)

    center_z = current_stage_pos.z
    z_stack_planes = []  # accumulate Z planes when z_stack_enabled

    for z_idx, z_offset in enumerate(ctx.z_offsets):
        # Move Z if doing Z-stack (skip for single-plane 2D)
        if ctx.z_stack_enabled and z_offset != 0.0:
            target_z = center_z + z_offset
            hardware.move_to_position(Position(z=target_z))
            logger.debug(
                "Z-stack: plane %d/%d, Z=%.2f (offset=%+.1f)",
                z_idx + 1, len(ctx.z_offsets), target_z, z_offset,
            )

        image, metadata = hardware.snap_image()

        # Background correction
        if ctx.background_correction_enabled and ctx.background_images:
            bg_key = 0.0
            if bg_key in ctx.background_images:
                try:
                    bg_scale = ctx.background_scaling_factors.get(bg_key, 1.0) \
                        if ctx.background_scaling_factors else 1.0
                    image = BackgroundCorrectionUtils.apply_flat_field_correction(
                        image, ctx.background_images[bg_key],
                        scaling_factor=bg_scale,
                        method=ctx.background_correction_method)
                except Exception as e:
                    logger.warning("  Background correction failed: %s", e)

        # Saturation check
        sat_result = _check_saturation(image, "tile", logger)
        if sat_result:
            for ch_key, pct in sat_result.items():
                if pct > tile_worst_sat.get(ch_key, 0):
                    tile_worst_sat[ch_key] = pct
            # Single-image modalities: no angle context, treat as normal
            worst_this_image = max(sat_result.values()) if sat_result else 0.0
            if worst_this_image > tile_role_sat[SATURATION_ROLE_NORMAL]:
                tile_role_sat[SATURATION_ROLE_NORMAL] = worst_this_image

        try:
            _accumulate_tile_stats(tile_stats, _compute_tile_stats(image))
        except Exception as stats_err:
            logger.debug(f"  Stats compute failed: {stats_err}")

        if not ctx.z_stack_enabled:
            # 2D mode: save directly
            image_path = ctx.output_path / filename
            if image_path.parent.exists():
                bf_pixel_size = hardware.get_pixel_size_um()
                ctx.write_pool.submit(
                    ome_tiff_writer,
                    filename=str(image_path),
                    pixel_size_um=bf_pixel_size,
                    data=image,
                )
        else:
            # Z-stack mode: accumulate for projection
            z_stack_planes.append(image)
            if ctx.save_raw_tiles:
                z_plane_path = ctx.output_path / f"z{z_idx:03d}" / filename
                z_plane_path.parent.mkdir(parents=True, exist_ok=True)
                ctx.write_pool.submit(
                    ome_tiff_writer,
                    filename=str(z_plane_path),
                    pixel_size_um=hardware.get_pixel_size_um(),
                    data=image,
                )

        ctx.image_count += 1
        ctx.update_progress(ctx.image_count, ctx.total_images)

    # Z-stack projection: reduce planes to single 2D image
    if ctx.z_stack_enabled and ctx.projection_fn is not None and z_stack_planes:
        projected = ctx.projection_fn(z_stack_planes)
        image_path = ctx.output_path / filename
        if image_path.parent.exists():
            ctx.write_pool.submit(
                ome_tiff_writer,
                filename=str(image_path),
                pixel_size_um=hardware.get_pixel_size_um(),
                data=projected,
            )
        logger.info(
            "Z-stack projection (%s) computed for %d planes",
            ctx.params.get("z_projection", "max"), len(z_stack_planes),
        )
        # Return Z to center position for next tile
        hardware.move_to_position(Position(z=center_z))

    try:
        image_path = ctx.output_path / filename
        write_position_metadata(ctx.metadata_txt_for_positions, image_path, hardware, ctx.modality)
    except Exception as e:
        logger.warning(f"  Failed to write position text {ctx.metadata_txt_for_positions}: {e}")

    tile_stats.pop("__counts__", None)
    return tile_worst_sat, tile_role_sat, tile_stats


def _record_tile_measurement(
    ctx: AcquisitionContext, pos_idx: int, filename: str, tile_start: float,
    needs_af: bool, af_type: str, drift: float, af_failed: bool,
    tile_worst_sat: dict, current_stage_pos,
    tile_role_sat: Optional[dict] = None,
    tile_stats: Optional[dict] = None,
) -> None:
    """Record per-tile measurement data and stream to NDJSON."""
    logger = ctx.logger
    tile_elapsed_ms = (time.perf_counter() - tile_start) * 1000
    logger.info(
        "Tile %d/%d: %.1fs (%s)",
        pos_idx + 1, len(ctx.positions), tile_elapsed_ms / 1000,
        "AF" if needs_af else "no-AF",
    )

    # Sparse-sample-aware progress sanity check
    if len(ctx.tile_measurements) >= 3 and not ctx.progress_warning_fired:
        completed_tile_times_ms = [m["tile_time_ms"] for m in ctx.tile_measurements[-10:]]
        avg_recent_ms = sum(completed_tile_times_ms) / len(completed_tile_times_ms)
        baseline_ms = ctx.tile_measurements[0]["tile_time_ms"]
        if baseline_ms > 0 and avg_recent_ms > 3.0 * baseline_ms:
            logger.error(
                "PROGRESS SANITY CHECK: average tile time over last %d tiles is "
                "%.1fs, which is %.1fx the first tile's %.1fs. This usually means "
                "the autofocus strategy is a poor match for the sample (e.g. "
                "dense_texture applied to sparse IF). Current strategy: %s. "
                "Consider cancelling and picking a different strategy from the "
                "Advanced panel dropdown.",
                len(completed_tile_times_ms),
                avg_recent_ms / 1000.0,
                avg_recent_ms / baseline_ms,
                baseline_ms / 1000.0,
                ctx.af_strategy_name,
            )
            ctx.progress_warning_fired = True

    # Periodic progress summary
    if (pos_idx + 1) % 100 == 0 and pos_idx > 0:
        completed = pos_idx + 1
        elapsed_total_s = sum(
            m["tile_time_ms"] for m in ctx.tile_measurements
        ) / 1000 + tile_elapsed_ms / 1000
        avg_s = elapsed_total_s / completed
        remaining = len(ctx.positions) - completed
        eta_s = remaining * avg_s
        eta_h = eta_s / 3600
        throughput = completed / (elapsed_total_s / 3600) if elapsed_total_s > 0 else 0
        logger.info(
            "[PROGRESS] %d/%d (%.1f%%) | avg %.1fs/tile | ETA %.1fh | %.0f tiles/hr",
            completed, len(ctx.positions),
            100 * completed / len(ctx.positions),
            avg_s, eta_h, throughput,
        )

    tile_role_sat = tile_role_sat or {}
    role_low_pct = round(tile_role_sat.get(SATURATION_ROLE_LOW, 0.0), 1)
    role_high_pct = round(tile_role_sat.get(SATURATION_ROLE_HIGH, 0.0), 1)
    role_normal_pct = round(tile_role_sat.get(SATURATION_ROLE_NORMAL, 0.0), 1)

    # The "primary" role for this tile is whichever non-zero role has the
    # worst saturation, biased toward the role that actually matters for
    # filtering (LOW > NORMAL > HIGH). PPM tiles always carry both LOW (small
    # angles) and HIGH (uncrossed) roles; we expose both as separate fields
    # below and tag the dominant concerning role here. role_label is the
    # filtering-friendly tag QuPath measurements / dialogs use.
    if role_low_pct > 0:
        role_label = SATURATION_ROLE_LOW
    elif role_normal_pct > 0:
        role_label = SATURATION_ROLE_NORMAL
    elif role_high_pct > 0:
        role_label = SATURATION_ROLE_HIGH
    else:
        # No saturation at all; tag with what role the modality would assign
        # to a default (0 deg) tile so downstream filters still see a label.
        role_label = _saturation_role_for(ctx.modality, 0.0)

    tile_measurement_entry = {
        "position_index": pos_idx,
        "filename": filename,
        "z_um": round(current_stage_pos.z, 2),
        "af_performed": needs_af,
        "af_type": af_type,
        "af_strategy": ctx.af_strategy_name,
        "af_drift_um": round(drift, 2),
        "af_failed": af_failed,
        "tile_time_ms": round(tile_elapsed_ms, 0),
        "saturation_R_pct": round(tile_worst_sat.get("R", tile_worst_sat.get("Gray", 0)), 1),
        "saturation_G_pct": round(tile_worst_sat.get("G", tile_worst_sat.get("Gray", 0)), 1),
        "saturation_B_pct": round(tile_worst_sat.get("B", tile_worst_sat.get("Gray", 0)), 1),
        "saturation_worst_pct": round(max(tile_worst_sat.values()) if tile_worst_sat else 0, 1),
        # Role-aggregate saturation: LOW = small-angle PPM (saturation is bad);
        # HIGH = uncrossed (~90 deg, intentionally bright, saturation OK);
        # NORMAL = single-image/channel modalities. role_label is the filter
        # tag for the QuPath measurement table.
        "saturation_role": role_label,
        "saturation_role_low_pct": role_low_pct,
        "saturation_role_high_pct": role_high_pct,
        "saturation_role_normal_pct": role_normal_pct,
        "acq_order_index": pos_idx,
        "acq_timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(tile_start)),
    }
    # Per-channel percentile / mean / std stats (P1 metrics). Surfaces dim-tile
    # under-exposure in the same QuPath measurement table as saturation.
    if tile_stats:
        for k, v in tile_stats.items():
            tile_measurement_entry[k] = v
        tile_measurement_entry["underexposed"] = bool(_stats_underexposed(tile_stats, ctx.modality))
    ctx.tile_measurements.append(tile_measurement_entry)

    if ctx.tile_measurements_stream is not None:
        try:
            ctx.tile_measurements_stream.write(json.dumps(tile_measurement_entry) + "\n")
            ctx.tile_measurements_stream.flush()
        except Exception as e:
            logger.debug("Failed to stream NDJSON entry for tile %d: %s", pos_idx, e)


def write_position_metadata(metadata_txt_for_positions, raw_image_path, hardware, modality):
    pos_read = hardware.get_current_position()
    line = (
        f"filename = {raw_image_path} ; "
        f"(x,y,z) = ({pos_read.x},{pos_read.y},{round(pos_read.z, 3)}); "
    )

    mod_config = get_modality_config(modality)
    if mod_config.has_rotation and hasattr(hardware, "get_psg_ticks"):
        angle = hardware.get_psg_ticks()
        line += f"r = {angle} ; "

    line += f"exposure (ms) = {hardware.get_exposure()}\n"

    with open(metadata_txt_for_positions, "a") as f:
        f.write(line)


def angle_to_name(angle: float, modality: Optional[str] = None) -> str:
    """Convert numeric angle to canonical name using modality config.

    Looks up the angle in the modality's angle_names mapping.
    Falls back to fuzzy matching (within 2 degrees) and then a generic label.

    Args:
        angle: Rotation angle in degrees
        modality: Modality identifier for config lookup (optional)

    Returns:
        Angle name (e.g., 'uncrossed', 'crossed', 'positive', 'negative')
    """
    mod_config = get_modality_config(modality)

    # Exact match first
    if angle in mod_config.angle_names:
        return mod_config.angle_names[angle]

    # Fuzzy match: find closest named angle within 2 degrees
    best_name = None
    best_dist = float("inf")
    for named_angle, name in mod_config.angle_names.items():
        dist = abs(angle - named_angle)
        if dist < best_dist and dist <= 2.0:
            best_dist = dist
            best_name = name

    if best_name is not None:
        return best_name

    return f"angle_{angle}"


def get_default_target_intensity(modality: str, angle: float) -> float:
    """Get default target intensity for background acquisition.

    Uses the modality config's angle_intensity_targets for angle-specific
    targets, falling back to the modality's default_target_intensity.

    Args:
        modality: Modality identifier (e.g., "ppm", "brightfield")
        angle: Rotation angle in degrees

    Returns:
        Target grayscale intensity (0-255)
    """
    mod_config = get_modality_config(modality)
    return mod_config.get_target_intensity(angle)


def load_calibration_targets_from_yaml(config_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load calibration targets from imageprocessing YAML file.

    Looks for the calibration_targets section which contains:
    - target_intensities: Per-angle default targets
    - background_exposures: Achieved intensities from prior background collection

    Args:
        config_path: Path to the main config file (config_PPM.yml)

    Returns:
        Dictionary with calibration_targets data or None if not found
    """
    config_path = Path(config_path)

    # Derive imageprocessing file path
    config_name = config_path.stem
    if config_name.startswith("config_"):
        microscope_name = config_name[7:]
        imageprocessing_name = f"imageprocessing_{microscope_name}.yml"
    else:
        imageprocessing_name = f"imageprocessing_{config_name}.yml"

    imageprocessing_path = config_path.parent / imageprocessing_name

    if not imageprocessing_path.exists():
        return None

    try:
        with open(imageprocessing_path, "r") as f:
            ip_data = yaml.safe_load(f) or {}
        return ip_data.get("calibration_targets")
    except Exception as e:
        logger.warning(f"Failed to load calibration targets from {imageprocessing_path}: {e}")
        return None


def get_target_intensity_for_angle(
    angle: float,
    modality: str = "ppm",
    config_path: Optional[Path] = None,
) -> Tuple[float, str]:
    """Get target intensity for a specific angle with YAML priority logic.

    Priority order:
    1. background_exposures.angles.{name}.achieved_intensity (from prior BG collection)
    2. calibration_targets.target_intensities.{name} (YAML configured)
    3. Modality config defaults (based on optical properties)

    This ensures white balance calibration uses the same target intensity
    as background collection, so white-balanced images match backgrounds.

    Args:
        angle: Rotation angle in degrees
        modality: Modality identifier (default: "ppm")
        config_path: Path to config file (optional, enables YAML lookup)

    Returns:
        Tuple of (target_intensity, source) where source describes where
        the value came from (e.g., "background_exposures", "yaml_config", "default")
    """
    angle_name = angle_to_name(angle, modality=modality)

    # Try YAML lookup if config_path provided
    if config_path is not None:
        cal_targets = load_calibration_targets_from_yaml(config_path)
        if cal_targets is not None:
            # Priority 1: Check background_exposures (achieved intensity from BG collection)
            bg_exposures = cal_targets.get("background_exposures", {})
            if bg_exposures and "angles" in bg_exposures:
                angle_data = bg_exposures["angles"].get(angle_name)
                if angle_data and "achieved_intensity" in angle_data:
                    return float(angle_data["achieved_intensity"]), "background_exposures"

            # Priority 2: Check configured target_intensities
            target_intensities = cal_targets.get("target_intensities", {})
            if angle_name in target_intensities:
                return float(target_intensities[angle_name]), "yaml_config"
            # Also check for 'default' key
            if "default" in target_intensities:
                return float(target_intensities["default"]), "yaml_config_default"

    # Priority 3: Hardcoded defaults
    return get_default_target_intensity(modality, angle), "default"


def get_target_intensity_for_background(modality: str, angle: float) -> float:
    """
    Get target intensity for background acquisition based on modality and angle.

    This is a convenience wrapper around get_target_intensity_for_angle() that
    only returns the intensity value (not the source). Use this for backward
    compatibility with existing code.

    For new code that needs to know where the value came from (e.g., to log
    whether YAML or defaults are being used), use get_target_intensity_for_angle().

    Args:
        modality: Modality identifier (e.g., "ppm", "brightfield")
        angle: Rotation angle in degrees (for PPM)

    Returns:
        Target grayscale intensity (0-255)

    Examples:
        >>> get_target_intensity_for_background("brightfield", 0)
        250.0
        >>> get_target_intensity_for_background("ppm", 90)
        245.0
        >>> get_target_intensity_for_background("ppm", 5)
        160.0
        >>> get_target_intensity_for_background("ppm", -5)
        160.0
        >>> get_target_intensity_for_background("ppm", 0)
        125.0
    """
    # For backward compatibility, use defaults only (no YAML lookup)
    # Callers that want YAML lookup should use get_target_intensity_for_angle()
    return get_default_target_intensity(modality, angle)


def save_background_exposures_to_yaml(
    config_path: Path,
    final_exposures: Dict[float, float],
    achieved_intensities: Dict[float, float],
    modality: str = "ppm",
    objective: Optional[str] = None,
    detector: Optional[str] = None,
) -> bool:
    """
    Save background collection exposures and achieved intensities to YAML.

    Updates the calibration_targets.background_exposures section in the
    imageprocessing YAML file. This data becomes the source of truth for
    target intensities in white balance calibration.

    Args:
        config_path: Path to the main config file (config_PPM.yml)
        final_exposures: Dictionary mapping angles to final exposure times (ms)
        achieved_intensities: Dictionary mapping angles to achieved median intensity
        modality: Modality name (e.g., "ppm")
        objective: Objective LOCI ID (optional)
        detector: Detector LOCI ID (optional)

    Returns:
        True if successfully saved, False otherwise
    """
    from datetime import datetime

    config_path = Path(config_path)

    # Derive imageprocessing file path
    config_name = config_path.stem
    if config_name.startswith("config_"):
        microscope_name = config_name[7:]
        imageprocessing_name = f"imageprocessing_{microscope_name}.yml"
    else:
        imageprocessing_name = f"imageprocessing_{config_name}.yml"

    imageprocessing_path = config_path.parent / imageprocessing_name

    try:
        # Load existing file or create empty dict
        if imageprocessing_path.exists():
            with open(imageprocessing_path, "r") as f:
                ip_data = yaml.safe_load(f) or {}
        else:
            ip_data = {}

        # Ensure calibration_targets section exists
        if "calibration_targets" not in ip_data:
            ip_data["calibration_targets"] = {}

        # Build background_exposures data
        angles_data = {}
        for angle, exposure_ms in final_exposures.items():
            angle_name = angle_to_name(angle, modality=modality)
            angles_data[angle_name] = {
                "angle_degrees": angle,
                "exposure_ms": round(exposure_ms, 2),
                "achieved_intensity": round(achieved_intensities.get(angle, 0.0), 1),
            }

        ip_data["calibration_targets"]["background_exposures"] = {
            "last_calibrated": datetime.now().isoformat(),
            "modality": modality,
            "objective": objective,
            "detector": detector,
            "angles": angles_data,
        }

        # Also ensure target_intensities has defaults if not present
        if "target_intensities" not in ip_data["calibration_targets"]:
            ip_data["calibration_targets"]["target_intensities"] = {
                "uncrossed": 245.0,
                "positive": 160.0,
                "negative": 160.0,
                "crossed": 125.0,
                "default": 180.0,
            }

        # Save updated file
        with open(imageprocessing_path, "w") as f:
            yaml.dump(ip_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved background exposures to {imageprocessing_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save background exposures to YAML: {e}")
        return False


def save_simple_wb_to_yaml(
    config_path: Path,
    simple_wb_results: Dict[str, Dict[str, float]],
    base_exposures: Dict[str, float],
    modality: str = "ppm",
    objective: Optional[str] = None,
    detector: Optional[str] = None,
    logger=None,
) -> bool:
    """
    Save simple WB (Mode 2) per-angle scaled exposures to imageprocessing YAML.

    Writes the simple_wb section under the detector profile so that acquisition
    can load pre-computed ratio-preserving per-angle exposures.

    Args:
        config_path: Path to main config file (config_PPM.yml)
        simple_wb_results: Dict mapping angle names to per-angle data, e.g.:
            {'uncrossed': {'scale': 1.0, 'unified_gain': 1.0, 'r': 0.66, 'g': 0.97, 'b': 4.76}, ...}
        base_exposures: Dict with base uncrossed R:G:B values ('r', 'g', 'b' keys)
        modality: Modality name
        objective: Objective LOCI ID
        detector: Detector LOCI ID
        logger: Optional logger

    Returns:
        True if saved successfully, False otherwise
    """
    from datetime import datetime

    config_path = Path(config_path)

    config_name = config_path.stem
    if config_name.startswith("config_"):
        microscope_name = config_name[7:]
        imageprocessing_name = f"imageprocessing_{microscope_name}.yml"
    else:
        imageprocessing_name = f"imageprocessing_{config_name}.yml"

    imageprocessing_path = config_path.parent / imageprocessing_name

    if not objective or not detector:
        if logger:
            logger.warning("Cannot save simple_wb: objective or detector not specified")
        return False

    try:
        if imageprocessing_path.exists():
            with open(imageprocessing_path, "r") as f:
                ip_data = yaml.safe_load(f) or {}
        else:
            ip_data = {}

        # Navigate/create to imaging_profiles.{modality}.{objective}.{detector}
        ip_data.setdefault("imaging_profiles", {})
        ip_data["imaging_profiles"].setdefault(modality, {})
        ip_data["imaging_profiles"][modality].setdefault(objective, {})
        ip_data["imaging_profiles"][modality][objective].setdefault(detector, {})

        detector_profile = ip_data["imaging_profiles"][modality][objective][detector]

        # Build simple_wb section
        detector_profile["simple_wb"] = {
            "last_calibrated": datetime.now().isoformat(),
            "base_angle": "uncrossed",
            "base_exposures_ms": {
                "r": round(base_exposures.get("r", 0), 2),
                "g": round(base_exposures.get("g", 0), 2),
                "b": round(base_exposures.get("b", 0), 2),
            },
            "base_gains": base_exposures.get("gains", {}),
            "angles": simple_wb_results,
        }

        # NOTE: Do NOT write wb_last_modified here. This function is called both
        # during WB calibration AND during background collection. Writing it during
        # BG collection would make the timestamp newer than the backgrounds, causing
        # the Java validator to report them as stale. wb_last_modified is written
        # by update_imageprocessing_config() in calibration.py during actual WB calibration.

        with open(imageprocessing_path, "w") as f:
            yaml.dump(ip_data, f, default_flow_style=False, sort_keys=False)

        if logger:
            logger.info(
                f"Saved simple_wb to {imageprocessing_path}: "
                f"{len(simple_wb_results)} angles"
            )
        return True

    except Exception as e:
        if logger:
            logger.error(f"Failed to save simple_wb to YAML: {e}")
        return False


def acquire_background_with_target_intensity(
    hardware: PycromanagerHardware,
    target_intensity: float,
    tolerance: float = 2.5,
    initial_exposure_ms: float = 100.0,
    max_iterations: int = 10,
    logger=None,
    preserve_analog_gains: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Acquire background image with adaptive exposure to reach target intensity.

    Uses proportional control to iteratively adjust exposure time until the
    median image intensity is within tolerance of the target value. Median is
    used instead of mean as it is more robust to outliers and hot pixels.

    Args:
        hardware: Microscope hardware interface
        target_intensity: Target median grayscale value (0-255)
        tolerance: Acceptable deviation from target (default +/-2.5)
        initial_exposure_ms: Starting exposure time in milliseconds
        max_iterations: Maximum adjustment iterations
        logger: Logger instance for tracking convergence
        preserve_analog_gains: If True, do not reset analog R/B gains to 1.0.
            Used by camera_awb mode where AWB corrections are stored in
            Gain_AnalogRed/Blue and must be preserved for correct color balance.

    Returns:
        Tuple of (image, final_exposure_ms)
            image: Acquired image at target intensity
            final_exposure_ms: Final exposure time used

    Raises:
        RuntimeError: If image acquisition fails
    """
    # Exposure bounds to prevent extreme values
    MIN_EXPOSURE_MS = 0.0001
    MAX_EXPOSURE_MS = 5000.0

    # Ensure per-channel mode is disabled before using unified set_exposure().
    # If per-channel mode is active (from calibration or previous operations),
    # hardware.set_exposure() would be silently ignored.
    try:
        hardware.camera.disable_individual_exposure()
        hardware.camera.disable_individual_gain()
        if preserve_analog_gains:
            # Camera AWB mode: preserve AWB corrections in Gain_AnalogRed/Blue.
            # These were set by one-shot AWB calibration and must remain active
            # so backgrounds match the WB state of tissue tiles.
            if logger:
                try:
                    cur_gains = hardware.camera.get_rb_analog_gains()
                    logger.info(
                        f"Preserving AWB analog gains: "
                        f"R={cur_gains['analog_red']:.3f}, B={cur_gains['analog_blue']:.3f}"
                    )
                except Exception:
                    logger.info("Preserving analog gains (could not read current values)")
        else:
            hardware.camera.set_rb_analog_gains(analog_red=1.0, analog_blue=1.0)
    except Exception:
        pass  # Not a JAI camera or module not available

    # Set initial exposure
    current_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, initial_exposure_ms))
    hardware.set_exposure(current_exposure)

    # Verify exposure was accepted by reading it back
    try:
        readback_exp = float(hardware.get_exposure())
        if logger:
            logger.info(
                f"Starting adaptive exposure: target={target_intensity:.1f}, "
                f"tolerance={tolerance:.1f}, initial_exposure={current_exposure:.1f}ms "
                f"(camera readback: {readback_exp:.1f}ms)"
            )
            if abs(readback_exp - current_exposure) > 1.0:
                logger.error(
                    f"EXPOSURE MISMATCH: set {current_exposure:.1f}ms "
                    f"but camera reports {readback_exp:.1f}ms! "
                    f"Exposure control may not be working."
                )
    except Exception:
        if logger:
            logger.info(
                f"Starting adaptive exposure: target={target_intensity:.1f}, "
                f"tolerance={tolerance:.1f}, initial_exposure={current_exposure:.1f}ms"
            )

    last_image = None
    last_exposure = current_exposure

    # Saturation limits are derived from the first image's dtype to support
    # both 8-bit (JAI, 0-255) and 16-bit (Hamamatsu, 0-65535) cameras.
    # Initialized on first snap, then fixed for all subsequent iterations.
    channel_sat_limit = None
    overall_sat_limit = None

    for iteration in range(max_iterations):
        # Snap image (debayering auto-detected based on camera type)
        image, metadata = hardware.snap_image()

        if image is None:
            raise RuntimeError(f"Failed to acquire image at iteration {iteration}")

        # Derive saturation limits from image dtype on first iteration.
        # Also scale target_intensity if it appears to be an 8-bit value on
        # a 16-bit camera (e.g., target=200 on uint16 -> target=51200).
        if channel_sat_limit is None:
            if image.dtype == np.uint16:
                channel_sat_limit = 64000
                overall_sat_limit = 64000
                if target_intensity <= 255:
                    old_target = target_intensity
                    target_intensity = target_intensity * 256.0
                    tolerance = tolerance * 256.0
                    if logger:
                        logger.info(
                            f"  16-bit camera detected: scaling target "
                            f"{old_target:.0f} -> {target_intensity:.0f}, "
                            f"tolerance -> {tolerance:.0f}"
                        )
            else:
                channel_sat_limit = 245
                overall_sat_limit = 254
            if logger:
                logger.info(
                    f"  Image dtype: {image.dtype}, saturation limits: "
                    f"channel={channel_sat_limit}, overall={overall_sat_limit}, "
                    f"target={target_intensity:.0f}"
                )

        # Calculate median intensity across all channels (more robust than mean)
        mean_intensity = float(np.median(image))

        # Check per-channel medians for saturation (RGB image expected)
        channel_saturated = False
        max_ch_median = mean_intensity
        max_ch_name = "all"
        if image.ndim == 3 and image.shape[2] >= 3:
            ch_names = ["R", "G", "B"]
            ch_medians = [float(np.median(image[:, :, c])) for c in range(3)]
            max_ch_idx = int(np.argmax(ch_medians))
            max_ch_median = ch_medians[max_ch_idx]
            max_ch_name = ch_names[max_ch_idx]
            channel_saturated = max_ch_median >= channel_sat_limit

        # Store for potential use if we don't converge
        last_image = image
        last_exposure = current_exposure

        if logger:
            ch_info = ""
            if image.ndim == 3 and image.shape[2] >= 3:
                ch_info = (
                    f", channels=[R={ch_medians[0]:.0f}, "
                    f"G={ch_medians[1]:.0f}, B={ch_medians[2]:.0f}]"
                )
            logger.info(
                f"  Iteration {iteration + 1}/{max_iterations}: "
                f"median={mean_intensity:.1f}, exposure={current_exposure:.1f}ms"
                f"{ch_info}"
            )

        # Per-channel saturation takes priority over convergence.
        # Even if overall median is on target, a saturated channel means
        # the exposure is too high and must be reduced.
        if channel_saturated:
            if max_ch_median >= (overall_sat_limit * 0.99):
                # Fully clipped -- no info about how far over, so halve.
                # Gets from 10ms to 0.6ms in ~4 iterations.
                reduction = 0.5
            else:
                # Partially saturated -- proportional reduction
                reduction = (channel_sat_limit * 0.90) / max_ch_median
            new_exposure = max(current_exposure * reduction, MIN_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    {max_ch_name} channel saturated "
                    f"(median={max_ch_median:.0f} >= {channel_sat_limit}), "
                    f"reducing exposure {current_exposure:.1f}ms "
                    f"-> {new_exposure:.1f}ms"
                )
            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)
            continue

        # Check convergence (only when no channel is saturated)
        intensity_error = abs(mean_intensity - target_intensity)
        if intensity_error <= tolerance:
            if logger:
                logger.info(
                    f"Converged! Final: median={mean_intensity:.1f}, "
                    f"exposure={current_exposure:.1f}ms, iterations={iteration + 1}"
                )
            _check_saturation(image, "background", logger or logging.getLogger(__name__))
            return image, current_exposure

        # Calculate proportional adjustment
        # If image is too dark, increase exposure; if too bright, decrease
        if mean_intensity >= overall_sat_limit:
            # Image is saturated - decrease exposure aggressively
            # Proportional control alone is too slow when saturated
            new_exposure = max(current_exposure * 0.5, MIN_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image saturated (median={mean_intensity:.1f}), halving exposure to {new_exposure:.1f}ms"
                )
            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)
        elif mean_intensity > 0:
            adjustment_ratio = target_intensity / mean_intensity
            # Clamp adjustment to prevent overshooting into channel saturation.
            # If max channel is already close to the limit, cap the increase.
            if max_ch_median > 0 and adjustment_ratio > 1.0:
                max_safe_ratio = channel_sat_limit / max_ch_median
                if adjustment_ratio > max_safe_ratio:
                    adjustment_ratio = max_safe_ratio * 0.90
                    if logger:
                        logger.info(
                            f"    Capping exposure increase to avoid "
                            f"{max_ch_name} saturation "
                            f"(capped ratio={adjustment_ratio:.2f})"
                        )
            new_exposure = current_exposure * adjustment_ratio

            # Clamp to bounds
            new_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, new_exposure))

            if logger:
                logger.info(
                    f"    Adjusting exposure: {current_exposure:.1f}ms -> {new_exposure:.1f}ms "
                    f"(ratio={adjustment_ratio:.2f})"
                )

            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)
        else:
            # Image is completely black, increase exposure significantly
            new_exposure = min(current_exposure * 2.0, MAX_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image completely black, doubling exposure to {new_exposure:.1f}ms"
                )
            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)

    # Max iterations reached without convergence
    if logger:
        logger.warning(
            f"Did not converge after {max_iterations} iterations. "
            f"Using last image: median={float(np.median(last_image)):.1f}, exposure={last_exposure:.1f}ms"
        )
    _check_saturation(last_image, "background", logger or logging.getLogger(__name__))

    return last_image, last_exposure


def acquire_background_with_biref_matching(
    hardware: PycromanagerHardware,
    reference_image: np.ndarray,
    tolerance: float = 5.0,
    initial_exposure_ms: float = 100.0,
    max_iterations: int = 10,
    logger=None,
) -> Tuple[np.ndarray, float, float]:
    """
    Acquire background image optimized to minimize birefringence against reference.

    Instead of matching overall intensity, this directly minimizes the
    birefringence metric (sum of absolute channel differences) against
    a reference image (typically the positive angle background).

    This ensures that when birefringence is calculated as:
        |R_pos - R_neg| + |G_pos - G_neg| + |B_pos - B_neg|
    the result is minimized for background regions.

    Args:
        hardware: Microscope hardware interface
        reference_image: Reference image to match against (e.g., +7 deg background)
        tolerance: Target mean birefringence value (default 5.0, ideal is 0)
        initial_exposure_ms: Starting exposure time in milliseconds
        max_iterations: Maximum adjustment iterations
        logger: Logger instance for tracking convergence

    Returns:
        Tuple of (image, final_exposure_ms, mean_biref)
            image: Acquired image that minimizes birefringence
            final_exposure_ms: Final exposure time used
            mean_biref: Achieved mean birefringence value

    Raises:
        RuntimeError: If image acquisition fails
    """
    MIN_EXPOSURE_MS = 0.0001
    MAX_EXPOSURE_MS = 5000.0

    # Ensure per-channel mode is disabled before using unified set_exposure()
    try:
        hardware.camera.disable_individual_exposure()
        hardware.camera.disable_individual_gain()
        hardware.camera.set_rb_analog_gains(analog_red=1.0, analog_blue=1.0)
    except Exception:
        pass  # Not a JAI camera or module not available

    current_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, initial_exposure_ms))
    hardware.set_exposure(current_exposure)

    # Convert reference to int16 for signed arithmetic
    ref_i16 = reference_image.astype(np.int16)
    ref_mean = float(np.mean(reference_image))

    if logger:
        logger.info(
            f"Starting biref-matched exposure: target_biref<={tolerance:.1f}, "
            f"ref_mean={ref_mean:.1f}, initial_exposure={current_exposure:.1f}ms"
        )

    best_biref = float('inf')
    best_image = None
    best_exposure = current_exposure

    for iteration in range(max_iterations):
        image, metadata = hardware.snap_image()

        if image is None:
            raise RuntimeError(f"Failed to acquire image at iteration {iteration}")

        img_i16 = image.astype(np.int16)

        # Calculate birefringence metric (same as ppm_angle_difference)
        # This is: |R_ref - R_img| + |G_ref - G_img| + |B_ref - B_img| per pixel
        abs_diff = np.abs(ref_i16 - img_i16)
        biref_per_pixel = np.sum(abs_diff, axis=2)
        mean_biref = float(np.mean(biref_per_pixel))

        # Calculate signed mean difference to determine adjustment direction
        img_mean = float(np.mean(image))
        signed_diff = ref_mean - img_mean

        # Track best result
        if mean_biref < best_biref:
            best_biref = mean_biref
            best_image = image.copy()
            best_exposure = current_exposure

        if logger:
            logger.info(
                f"  Iteration {iteration + 1}/{max_iterations}: "
                f"biref={mean_biref:.1f}, img_mean={img_mean:.1f}, "
                f"signed_diff={signed_diff:+.1f}, exposure={current_exposure:.1f}ms"
            )

        # Check convergence
        if mean_biref <= tolerance:
            if logger:
                logger.info(
                    f"Converged! Final biref={mean_biref:.1f}, "
                    f"exposure={current_exposure:.1f}ms, iterations={iteration + 1}"
                )
            _check_saturation(image, "biref-background", logger or logging.getLogger(__name__))
            return image, current_exposure, mean_biref

        # Check if we can improve further with exposure adjustment
        # If images have similar overall intensity but high biref, it means
        # per-channel ratios differ - exposure alone cannot fix this
        if abs(signed_diff) < 2.0 and iteration > 0:
            if logger:
                logger.warning(
                    f"    Images have similar intensity (diff={signed_diff:+.1f}) "
                    f"but biref={mean_biref:.1f}. Per-channel differences may not be "
                    f"correctable by exposure adjustment alone."
                )
            # Continue trying a few more iterations in case we can improve
            if iteration >= 3:
                break

        # Proportional adjustment based on mean intensity difference
        if img_mean >= 254.0:
            # Image saturated - decrease aggressively
            new_exposure = max(current_exposure * 0.5, MIN_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image saturated, halving exposure to {new_exposure:.1f}ms"
                )
        elif img_mean > 0:
            # Adjust to match reference intensity
            adjustment_ratio = ref_mean / img_mean
            new_exposure = current_exposure * adjustment_ratio
            new_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, new_exposure))

            if logger:
                logger.info(
                    f"    Adjusting: {current_exposure:.1f}ms -> {new_exposure:.1f}ms "
                    f"(ratio={adjustment_ratio:.3f})"
                )
        else:
            # Image completely black
            new_exposure = min(current_exposure * 2.0, MAX_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image black, doubling exposure to {new_exposure:.1f}ms"
                )

        current_exposure = new_exposure
        hardware.set_exposure(current_exposure)

    # Return best result found
    if logger:
        logger.info(
            f"Max iterations reached. Using best result: "
            f"biref={best_biref:.1f}, exposure={best_exposure:.1f}ms"
        )
    _check_saturation(best_image, "biref-background", logger or logging.getLogger(__name__))

    return best_image, best_exposure, best_biref


def acquire_background_with_per_channel_adaptive(
    hardware: PycromanagerHardware,
    initial_exposures: Dict[str, float],
    target_intensity: float = 200.0,
    tolerance: float = 2.5,
    max_iterations: int = 10,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Acquire background image using per-channel exposure mode with adaptive scaling.

    Unlike acquire_background_with_target_intensity which uses a single exposure,
    this function maintains per-channel exposure ratios (for white balance) while
    scaling all channels proportionally to reach the target intensity.

    Args:
        hardware: Microscope hardware interface
        initial_exposures: Dict with 'r', 'g', 'b' exposure values in ms
                          e.g., {'r': 45.0, 'g': 50.0, 'b': 55.0}
        target_intensity: Target median grayscale value (0-255)
        tolerance: Acceptable deviation from target (default +/-2.5)
        max_iterations: Maximum adjustment iterations
        logger: Logger instance for tracking convergence

    Returns:
        Tuple of (image, final_exposures)
            image: Acquired image at target intensity
            final_exposures: Dict with final per-channel exposures {'r': x, 'g': y, 'b': z}

    Raises:
        RuntimeError: If image acquisition fails
    """
    if not hardware.camera.supports_per_channel_exposure():
        if logger:
            logger.warning("Camera does not support per-channel exposure - falling back to single exposure")
        # Fall back to regular adaptive exposure
        image, final_exp = acquire_background_with_target_intensity(
            hardware=hardware,
            target_intensity=target_intensity,
            tolerance=tolerance,
            initial_exposure_ms=initial_exposures.get('g', 100.0),
            max_iterations=max_iterations,
            logger=logger,
        )
        return image, {'r': final_exp, 'g': final_exp, 'b': final_exp}

    # Exposure bounds
    MIN_EXPOSURE_MS = 0.01
    MAX_EXPOSURE_MS = 5000.0

    # Get initial per-channel exposures
    exp_r = max(MIN_EXPOSURE_MS, initial_exposures.get('r', 50.0))
    exp_g = max(MIN_EXPOSURE_MS, initial_exposures.get('g', 50.0))
    exp_b = max(MIN_EXPOSURE_MS, initial_exposures.get('b', 50.0))

    # Calculate ratios relative to green (reference channel)
    ratio_r = exp_r / exp_g
    ratio_b = exp_b / exp_g

    if logger:
        logger.info(
            f"Starting per-channel adaptive exposure: target={target_intensity:.1f}, "
            f"initial R={exp_r:.1f}ms, G={exp_g:.1f}ms, B={exp_b:.1f}ms"
        )
        logger.info(f"  Channel ratios (R:G:B) = {ratio_r:.3f}:1.000:{ratio_b:.3f}")

    # Apply initial per-channel exposures
    hardware.camera.set_channel_exposures(red=exp_r, green=exp_g, blue=exp_b, auto_enable=True)

    last_image = None
    last_exposures = {'r': exp_r, 'g': exp_g, 'b': exp_b}

    for iteration in range(max_iterations):
        # Snap image
        image, metadata = hardware.snap_image()

        if image is None:
            raise RuntimeError(f"Failed to acquire image at iteration {iteration}")

        # Calculate median intensity
        median_intensity = float(np.median(image))

        last_image = image
        last_exposures = {'r': exp_r, 'g': exp_g, 'b': exp_b}

        if logger:
            logger.info(
                f"  Iteration {iteration + 1}/{max_iterations}: "
                f"median={median_intensity:.1f}, G_exp={exp_g:.1f}ms"
            )

        # Check convergence
        if abs(median_intensity - target_intensity) <= tolerance:
            if logger:
                logger.info(
                    f"Converged! Final: median={median_intensity:.1f}, "
                    f"R={exp_r:.1f}ms, G={exp_g:.1f}ms, B={exp_b:.1f}ms"
                )
            return image, {'r': exp_r, 'g': exp_g, 'b': exp_b}

        # Calculate scale factor
        if median_intensity >= 254.0:
            # Saturated - reduce aggressively
            scale = 0.5
            if logger:
                logger.warning(f"    Image saturated, halving exposures")
        elif median_intensity > 0:
            scale = target_intensity / median_intensity
        else:
            # Black image - increase
            scale = 2.0
            if logger:
                logger.warning(f"    Image black, doubling exposures")

        # Scale all channel exposures proportionally (maintaining ratios)
        exp_g = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, exp_g * scale))
        exp_r = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, exp_g * ratio_r))
        exp_b = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, exp_g * ratio_b))

        if logger:
            logger.info(
                f"    Scaled exposures: R={exp_r:.1f}ms, G={exp_g:.1f}ms, B={exp_b:.1f}ms"
            )

        # Apply scaled per-channel exposures
        hardware.camera.set_channel_exposures(red=exp_r, green=exp_g, blue=exp_b, auto_enable=True)

    # Max iterations reached
    if logger:
        logger.warning(
            f"Did not converge after {max_iterations} iterations. "
            f"Using last image: median={float(np.median(last_image)):.1f}"
        )

    return last_image, last_exposures


def _resolve_background_profile_key(
    modality: str,
    objective: str,
    hardware: PycromanagerHardware,
    logger,
) -> "str | None":
    """Resolve a background-collection modality + objective pair to a full
    acquisition_profiles key.

    Java sends --modality as the bare modality name ("Brightfield") and
    --objective as the full objective id ("0.5NA_AIR_10x" on OWS3, or
    "LOCI_OBJECTIVE_OLYMPUS_10X_POL_001" on PPM), but acquisition_profiles
    in the YAML are keyed by <modality>_<objective-suffix>
    ("Brightfield_10x"). We walk the profiles dict and pick the first key
    that starts with "<modality>_" and whose suffix is a substring of the
    objective id (case-insensitive). This is intentionally scope-agnostic
    so every scope's objective-id convention works without a lookup table.

    Returns the matched profile key, or None if no match is found (in
    which case the caller falls back to using the bare modality and lets
    apply_profile_illumination log its own "profile not found" warning).
    """
    if not modality:
        return None
    try:
        profiles = hardware.settings.get("acquisition_profiles", {}) or {}
    except Exception:
        return None
    if not profiles:
        return None

    modality_prefix = modality + "_"
    # Strategy 1: exact match on modality alone (unusual but possible if
    # someone keyed a profile by modality-only).
    if modality in profiles:
        return modality

    obj_lower = (objective or "").lower()

    # Strategy 2: look for "<modality>_<suffix>" where <suffix> appears in
    # the objective id. Prefer the longest matching suffix so e.g. "10x"
    # beats a hypothetical "x" on the same scope.
    candidates = []
    for key in profiles:
        if not key.startswith(modality_prefix):
            continue
        suffix = key[len(modality_prefix):]
        if suffix and suffix.lower() in obj_lower:
            candidates.append((len(suffix), key))
    if candidates:
        candidates.sort(reverse=True)
        matched = candidates[0][1]
        logger.info(
            "Resolved background profile: modality='%s' objective='%s' -> '%s'",
            modality, objective, matched,
        )
        return matched

    # Strategy 3: case-insensitive fallback for strategy 2, in case the
    # profile key and the modality arg differ in capitalization.
    modality_prefix_lower = modality_prefix.lower()
    for key in profiles:
        if not key.lower().startswith(modality_prefix_lower):
            continue
        suffix = key[len(modality_prefix):]
        if suffix and suffix.lower() in obj_lower:
            logger.info(
                "Resolved background profile (case-insensitive): modality='%s' objective='%s' -> '%s'",
                modality, objective, key,
            )
            return key

    logger.warning(
        "Could not resolve background profile for modality='%s' objective='%s'. "
        "Available profiles: %s",
        modality, objective, sorted(profiles.keys()),
    )
    return None


def simple_background_collection(
    yaml_file_path: str,
    output_folder_path: str,
    modality: str,
    angles_str: str,
    exposures_str: str,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
    update_progress: Callable[[int, int], None],
    use_per_angle_wb: bool = False,
    wb_mode: str = None,
    objective: str = None,
    detector: str = None,
    target_intensity_override: float = None,
):
    """
    Simplified background collection for BackgroundCollectionWorkflow.

    Acquires background images at current position using adaptive exposure
    to reach target intensities. Saves directly to correct folder structure
    for flat field correction.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds
        modality: Modality identifier (e.g., "ppm")
        angles_str: String of angles like "(0,90,5,-5)"
        exposures_str: String of initial exposure times like "(1.5,100,50,50)".
                      These are used as starting points for adaptive exposure.
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance
        update_progress: Progress callback function
        use_per_angle_wb: Whether to apply per-angle white balance calibration
                         before acquiring each background image (legacy param)
        wb_mode: White balance mode string: "camera_awb", "simple", "per_angle", "off".
                 If None, derived from use_per_angle_wb for backward compatibility.
        objective: Objective ID for calibration lookup (e.g., "LOCI_OBJECTIVE_OLYMPUS_20X_POL_001").
                  Required for WB calibration data to be loaded from imageprocessing YAML.
        detector: Detector ID for calibration lookup (e.g., "LOCI_DETECTOR_JAI_001").
                 Required for WB calibration data to be loaded from imageprocessing YAML.

    Returns:
        Dict[float, float]: Dictionary mapping angles to final exposure times (ms)
                           e.g., {90.0: 1.2, 5.0: 45.8, ...}
    """
    # Resolve wb_mode: require explicit choice, never silently pick a mode
    if wb_mode is None:
        if use_per_angle_wb:
            wb_mode = "per_angle"
        else:
            raise ValueError(
                "No wb_mode specified for background collection. "
                "White balance mode must be explicitly chosen by the user: "
                "camera_awb, simple, or per_angle. "
                "Update the client to always send --wb-mode."
            )
    # Keep use_per_angle_wb in sync for downstream code
    use_per_angle_wb = wb_mode == "per_angle"
    logger.info(f"Background collection wb_mode: {wb_mode}")
    logger.info("=== SIMPLE BACKGROUND COLLECTION STARTED ===")

    try:
        # Stop live mode if running - JAI camera properties cannot be changed during live streaming
        # This is the same pattern used in calibration.py
        try:
            hardware.camera.stop_if_streaming()
            logger.info("Ensured camera not streaming before background collection")
        except Exception as e:
            logger.warning(f"Could not stop live/sequence mode: {e}")

        # Align lamp intensity with the acquisition profile before collecting
        # backgrounds. Without this, background collection uses whatever lamp
        # level the user left in the hardware (e.g. from the Live Viewer
        # Camera tab), while the subsequent tiled acquisition calls
        # apply_mode_setup() and overwrites the lamp with the profile's
        # illumination_intensity -- producing flat-fields that silently
        # correct for the wrong illumination pattern.
        #
        # This only touches lamp intensity -- it does NOT move stages,
        # switch detectors, or apply MM presets, so the user's manually
        # positioned blank area is preserved.
        # The Java client sends --modality as the bare modality name (e.g.
        # "Brightfield"), but acquisition_profiles in the YAML are keyed by
        # the enhanced form <modality>_<objective-suffix> (e.g. "Brightfield_10x").
        # Calling apply_profile_illumination("Brightfield") would fail the
        # lookup and silently leave whatever lamp level the Live Viewer last
        # used, producing flat-fields against the wrong intensity. Resolve
        # the full profile key here using both modality and objective.
        resolved_profile = _resolve_background_profile_key(
            modality, objective, hardware, logger
        )
        profile_to_apply = resolved_profile or modality
        try:
            applied_intensity = hardware.apply_profile_illumination(profile_to_apply)
            if applied_intensity is not None:
                logger.info(
                    f"Background collection aligned to profile '{profile_to_apply}' "
                    f"illumination intensity: {applied_intensity}"
                )
            else:
                logger.info(
                    f"Background collection: no profile illumination applied for '{profile_to_apply}' "
                    "(profile missing, no illumination_intensity, or no active illumination device)"
                )
        except Exception as e:
            logger.warning(f"Could not apply profile illumination for background collection: {e}")

        # Parse angles and exposures from client
        # Use client's exposures as initial values for adaptive exposure
        angles, exposures = parse_angles_exposures(angles_str, exposures_str)

        # Brightfield and other single-angle modalities send empty angles [].
        # Treat this as a single background at angle 0 (no rotation).
        # Track whether this was a non-rotation collection so we can save the
        # output file as background.tif instead of 0.0.tif -- the 0.0 is a
        # placeholder, not a real polarization angle, and saving it as a
        # numeric filename is confusing for monochrome / brightfield users.
        is_non_rotation_background = not angles
        if not angles:
            angles = [0.0]
            exposures = [exposures[0] if exposures else 50.0]
            logger.info("No angles specified -- collecting single brightfield background (angle=0)")

        logger.info(f"Collecting backgrounds for angles: {angles} using adaptive exposure")
        logger.info(f"Initial exposures from client: {exposures}")

        # Load microscope configuration
        if not Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        # Load main configuration file
        settings = config_manager.load_config_file(yaml_file_path)

        # Load and merge LOCI resources (derive path from config file, not package)
        loci_rsc_file = str(
            Path(yaml_file_path).parent / "resources" / "resources_LOCI.yml"
        )
        try:
            loci_resources = config_manager.load_config_file(loci_rsc_file)
            settings.update(loci_resources)
            logger.info("Loaded and merged LOCI resources for background collection")
        except FileNotFoundError:
            logger.warning(
                f"LOCI resources file not found at {loci_rsc_file}, continuing without device mappings"
            )

        hardware.settings = settings

        # Re-initialize microscope-specific methods with updated settings
        # This is critical for PPM rotation to work correctly
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Auto-detect JAI camera and load calibration automatically
        # For JAI cameras, per-channel white balance is REQUIRED for correct flat-field correction
        # because JAI uses different per-channel exposures for each angle during acquisition.
        # If backgrounds are captured without matching these exposures, flat-field correction
        # will over/under-correct the images.
        jai_calibration = None
        is_jai_camera = False
        try:
            camera_name = hardware.get_camera_name()
            is_jai_camera = hardware.camera.supports_per_channel_exposure()
            logger.info(f"Camera detected: {camera_name} (per_channel={is_jai_camera})")
        except Exception as e:
            logger.debug(f"Could not detect camera type: {e}")

        # Use target intensity override from Java UI if provided (non-RGB cameras only)
        if target_intensity_override is not None and target_intensity_override > 0:
            logger.info(
                "Using target intensity override from client: %.0f",
                target_intensity_override,
            )

        # Load calibration based on wb_mode
        # All modes except "off" need Mode 3 calibration data for reference
        should_load_calibration = wb_mode != "off" and (is_jai_camera or use_per_angle_wb or wb_mode in ("camera_awb", "simple"))

        # Resolve objective/detector for calibration lookup.
        # Prefer explicitly passed values (from BGACQUIRE --objective/--detector flags),
        # fall back to settings from config YAML (which may be null).
        if not objective:
            objective = settings.get("objective_in_use") or settings.get("objective")
        if not detector:
            detector = settings.get("detector_in_use") or settings.get("detector")

        if should_load_calibration:
            # Get objective and detector from settings or parse from output path
            # Output path structure: {base}/{detector}/{modality}/{magnification}
            # e.g., D:\data\background_tiles\LOCI_DETECTOR_JAI_001\ppm\20x

            # If not in settings, try to extract from output path
            if not detector or not objective:
                path_parts = Path(output_folder_path).parts
                # Look for detector pattern (LOCI_DETECTOR_*)
                for i, part in enumerate(path_parts):
                    if part.startswith("LOCI_DETECTOR_"):
                        detector = part
                        # Magnification is typically 2 parts after detector (detector/modality/mag)
                        if i + 2 < len(path_parts):
                            magnification = path_parts[i + 2]  # e.g., "20x"
                            # Find matching objective in imaging_profiles
                            try:
                                with open(Path(yaml_file_path).parent / f"imageprocessing_{Path(yaml_file_path).stem.replace('config_', '')}.yml", "r") as f:
                                    ip_data = yaml.safe_load(f) or {}
                                modality_profiles = ip_data.get("imaging_profiles", {}).get(modality, {})
                                for obj_name in modality_profiles.keys():
                                    # Match magnification in objective name (e.g., "20X" in "LOCI_OBJECTIVE_OLYMPUS_20X_POL_001")
                                    if magnification.upper().replace("X", "") in obj_name.upper():
                                        objective = obj_name
                                        logger.info(f"Matched objective {objective} from magnification {magnification}")
                                        break
                            except Exception as e:
                                logger.warning(f"Failed to find objective from path: {e}")
                        break

            logger.info(f"Looking up calibration for modality={modality}, objective={objective}, detector={detector}")

            jai_calibration = load_jai_calibration_from_imageprocessing(
                config_path=Path(yaml_file_path),
                per_angle=True,
                modality=modality,
                objective=objective,
                detector=detector,
                logger=logger,
            )
            if jai_calibration:
                logger.info(f"Per-angle calibration loaded for background collection (wb_mode={wb_mode})")
                if wb_mode == "per_angle" and is_jai_camera:
                    if not use_per_angle_wb:
                        logger.info("JAI camera detected: automatically enabling per-channel WB for background collection")
                    use_per_angle_wb = True
            else:
                if is_jai_camera:
                    logger.warning(
                        "JAI camera detected but no calibration found! "
                        "Backgrounds may not match acquisition conditions. "
                        "Run 'White Balance Calibration' first for best results."
                    )
                else:
                    logger.warning("Per-angle white balance requested but no calibration found")

        # Clear any lingering AWB corrections for simple/per_angle modes.
        # These modes set explicit per-channel exposures + analog gains per tile;
        # residual AWB analog gain corrections would compound incorrectly.
        if wb_mode in ("simple", "per_angle") and is_jai_camera:
            try:
                hardware.camera.clear_awb_corrections()
            except Exception as e:
                logger.warning(f"Could not clear AWB corrections before {wb_mode} WB: {e}")

        # Camera AWB mode: disable per-channel exposure/gain, use unified controls.
        # The camera's built-in AWB applies corrections through an INTERNAL processing
        # pipeline -- NOT through Gain_AnalogRed/Gain_AnalogBlue registers (those
        # always read 1.0 regardless of AWB state). The AWB Continuous mode was run
        # during calibration and the internal corrections persist after setting Off.
        # Both backgrounds and tissue tiles see the same internal corrections.
        camera_awb_gains = {}
        if wb_mode == "camera_awb" and is_jai_camera:
            try:
                hardware.camera.disable_individual_exposure()
                hardware.camera.disable_individual_gain()
                logger.info(
                    "Camera AWB: using unified exposure/gain. "
                    "Internal AWB corrections applied via camera pipeline."
                )
            except Exception as e:
                logger.warning(f"Could not configure camera AWB mode: {e}")
            # Extract unified gains from Mode 3 calibration for brightness
            if jai_calibration and "angles" in jai_calibration:
                for angle_name, angle_data in jai_calibration["angles"].items():
                    gains = angle_data.get("gains", {})
                    camera_awb_gains[angle_name] = gains.get("unified_gain", 1.0)
                logger.info(f"Camera AWB unified gains for background: {camera_awb_gains}")
            # Don't use per-channel calibration in the per-angle loop
            jai_calibration = None

        # Simple WB mode: extract uncrossed R:G:B base ratios from Mode 3 calibration
        simple_wb_base = None
        if wb_mode == "simple" and jai_calibration and "angles" in jai_calibration:
            uncrossed_cal = jai_calibration["angles"].get("uncrossed")
            if uncrossed_cal and "exposures_ms" in uncrossed_cal:
                simple_wb_base = {
                    "r": uncrossed_cal["exposures_ms"]["r"],
                    "g": uncrossed_cal["exposures_ms"]["g"],
                    "b": uncrossed_cal["exposures_ms"]["b"],
                    "gains": uncrossed_cal.get("gains", {}),
                }
                logger.info(
                    f"Simple WB base ratios from uncrossed: "
                    f"R={simple_wb_base['r']:.2f}ms, "
                    f"G={simple_wb_base['g']:.2f}ms, "
                    f"B={simple_wb_base['b']:.2f}ms"
                )
            else:
                raise ValueError(
                    "Simple WB mode requires uncrossed (90 deg) calibration data "
                    "but none was found. Run 'PPM White Balance Calibration' first, "
                    "or select a different WB mode."
                )

        # Get current position for reference
        current_pos = hardware.get_current_position()
        logger.info(
            f"Acquiring backgrounds at position: X={current_pos.x:.1f}, Y={current_pos.y:.1f}, Z={current_pos.z:.1f}"
        )

        # Create output directory structure
        output_path = Path(output_folder_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving backgrounds to: {output_path}")

        # Initialize progress
        total_images = len(angles)
        update_progress(0, total_images)

        # Track final exposures and achieved intensities for each angle
        final_exposures = {}
        achieved_intensities = {}

        # Load saved exposures from prior background collection (if available).
        # These are much better starting points than the client's 10ms defaults,
        # especially for 90-deg uncrossed which needs ~0.5ms at 20x.
        saved_exposures = {}
        try:
            cal_targets = load_calibration_targets_from_yaml(Path(yaml_file_path))
            if cal_targets:
                bg_exp = cal_targets.get("background_exposures", {})
                if bg_exp and "angles" in bg_exp:
                    for aname, adata in bg_exp["angles"].items():
                        if "angle_degrees" in adata and "exposure_ms" in adata:
                            saved_exposures[adata["angle_degrees"]] = adata["exposure_ms"]
                    if saved_exposures:
                        logger.info(f"Loaded prior exposures: {saved_exposures}")
        except Exception:
            pass  # No saved data, will use client defaults

        # Track reference images for birefringence pair matching
        # When acquiring paired polarization angles (+7/-7 or +5/-5), the negative
        # angle should minimize birefringence against the positive angle's IMAGE,
        # not just match intensity. This uses the same metric as biref calculation:
        # sum(|R_pos - R_neg| + |G_pos - G_neg| + |B_pos - B_neg|)
        biref_pair_references = {}  # Maps positive angle -> reference image

        # Acquire background for each angle
        for angle_idx, angle in enumerate(angles):
            logger.info(f"Acquiring background {angle_idx + 1}/{total_images} for angle {angle}")

            # Set rotation angle if rotation stage is present
            if hardware.rotation_stage is not None:
                hardware.set_psg_ticks(angle)
                logger.info(f"Set angle to {angle}")

            # Use saved exposure from prior run if available, else client default
            if angle in saved_exposures:
                initial_exposure_ms = saved_exposures[angle]
                logger.info(f"Initial exposure from prior run: {initial_exposure_ms:.2f}ms")
            else:
                initial_exposure_ms = exposures[angle_idx] if angle_idx < len(exposures) else 100.0
                logger.info(f"Initial exposure from client: {initial_exposure_ms:.2f}ms")

            # Choose acquisition method based on white balance mode
            image = None  # Will be set by whichever branch acquires

            if wb_mode == "camera_awb":
                # Camera AWB mode: unified exposure only, adaptive intensity matching
                # Apply unified gain for brightness at dim angles
                angle_name = angle_to_name(angle, modality=modality)
                if camera_awb_gains and angle_name in camera_awb_gains:
                    try:
                        gain_val = camera_awb_gains[angle_name]
                        hardware.camera.set_unified_gain(gain_val)
                        logger.info(f"  Camera AWB: unified gain={gain_val:.2f} for {angle_name}")
                    except Exception as e:
                        logger.debug(f"Could not set unified gain: {e}")

                # Use standard adaptive intensity matching (single unified exposure)
                target_intensity = (target_intensity_override
                                    if target_intensity_override and target_intensity_override > 0
                                    else get_target_intensity_for_background(modality, angle))
                # AWB equalizes all channels, so the target applies to EVERY
                # channel equally. Without AWB, only the dominant channel
                # (red, ~3.5x bias) reaches 245 while green/blue are much
                # lower. With AWB, a target of 245 puts ALL channels at ~245
                # leaving no headroom. Cap to keep channels well below clipping.
                AWB_MAX_TARGET = 210.0
                if target_intensity > AWB_MAX_TARGET:
                    logger.info(
                        f"  Camera AWB: capping target {target_intensity:.0f} "
                        f"-> {AWB_MAX_TARGET:.0f} (AWB equalizes all channels)"
                    )
                    target_intensity = AWB_MAX_TARGET
                logger.info(f"Camera AWB target intensity: {target_intensity:.1f}")
                try:
                    image, final_exposure = acquire_background_with_target_intensity(
                        hardware=hardware,
                        target_intensity=target_intensity,
                        tolerance=2.5,
                        initial_exposure_ms=initial_exposure_ms,
                        max_iterations=10,
                        logger=logger,
                        preserve_analog_gains=True,  # Keep AWB corrections active
                    )
                    actual_intensity = float(np.median(image))

                    _check_saturation(image, f"AWB angle={angle}", logger)
                    final_exposures[angle] = final_exposure
                    achieved_intensities[angle] = actual_intensity
                except RuntimeError as e:
                    logger.error(f"Failed to acquire camera AWB background at angle {angle}: {e}")
                    continue

            elif wb_mode == "simple" and simple_wb_base is not None:
                # Simple WB mode: ratio-preserving adaptive scaling
                # Use uncrossed R:G:B ratios, uniformly scale per angle to hit target
                angle_name = angle_to_name(angle, modality=modality)
                try:
                    # Get unified gain from Mode 3 calibration for this angle
                    unified_gain = 1.0
                    if jai_calibration and "angles" in jai_calibration:
                        angle_cal = jai_calibration["angles"].get(angle_name, {})
                        unified_gain = angle_cal.get("gains", {}).get("unified_gain", 1.0)
                    hardware.camera.set_unified_gain(unified_gain)

                    # Apply calibrated R/B analog gains from uncrossed calibration.
                    # Phase 2 of calibration fine-tunes color balance via analog gains;
                    # without these, per-channel exposures alone are insufficient and
                    # the image appears yellow (red-dominant).
                    wb_gains = simple_wb_base.get("gains", {})
                    analog_red = wb_gains.get("analog_red", 1.0)
                    analog_blue = wb_gains.get("analog_blue", 1.0)
                    hardware.camera.set_rb_analog_gains(analog_red=analog_red, analog_blue=analog_blue)
                    logger.info(
                        f"  Simple WB: analog gains R={analog_red:.3f}, B={analog_blue:.3f}"
                    )

                    # Start with base uncrossed R:G:B (scale=1.0)
                    base_r = simple_wb_base["r"]
                    base_g = simple_wb_base["g"]
                    base_b = simple_wb_base["b"]
                    target_intensity = (target_intensity_override
                                    if target_intensity_override and target_intensity_override > 0
                                    else get_target_intensity_for_background(modality, angle))
                    logger.info(
                        f"Simple WB adaptive: base R={base_r:.1f}, G={base_g:.1f}, B={base_b:.1f}, "
                        f"gain={unified_gain:.2f}, target={target_intensity:.1f}"
                    )

                    # Iterative scaling: snap image, measure, scale all channels uniformly.
                    # When pixels are clipped (median >= 254), ratio scaling is useless
                    # because the measured value is a lower bound on the real signal.
                    # Use exponential backoff (halve) until out of saturation, then
                    # switch to ratio scaling for fine convergence.
                    scale = 1.0
                    tolerance = 2.5
                    max_iter = 8
                    for iteration in range(max_iter):
                        exp_r = base_r * scale
                        exp_g = base_g * scale
                        exp_b = base_b * scale
                        hardware.camera.set_channel_exposures(
                            red=exp_r, green=exp_g, blue=exp_b, auto_enable=True,
                        )
                        image, metadata = hardware.snap_image()
                        if image is None:
                            raise RuntimeError("Failed to acquire image")
                        measured = float(np.median(image))
                        # Check clipped pixel fraction for saturation detection
                        clipped_frac = float(np.mean(image >= 254)) if image.dtype == np.uint8 else 0.0
                        logger.info(
                            f"  Iteration {iteration}: scale={scale:.3f}, "
                            f"R={exp_r:.1f}ms G={exp_g:.1f}ms B={exp_b:.1f}ms, "
                            f"median={measured:.1f} (target={target_intensity:.1f}), "
                            f"clipped={clipped_frac:.1%}"
                        )
                        if abs(measured - target_intensity) <= tolerance:
                            logger.info(f"  Converged at iteration {iteration}")
                            break
                        if measured < 1.0:
                            scale *= 5.0  # Very dark, big jump
                        elif clipped_frac > 0.05 or measured >= 254:
                            # Saturated: ratio scaling is meaningless. Halve exposure.
                            scale *= 0.5
                            logger.info(f"  Clipped -- exponential backoff to scale={scale:.3f}")
                        else:
                            scale *= target_intensity / measured

                    actual_intensity = float(np.median(image))
                    final_exposures[angle] = exp_g  # Store green as reference
                    achieved_intensities[angle] = actual_intensity

                    # NOTE: simple_wb.angles is now written during WB calibration
                    # (WBSIMPLE handler), NOT during background collection.
                    # BG collection uses per-channel scaling for flat-field acquisition,
                    # but the acquisition itself uses unified values from calibration.

                    logger.info(
                        f"Simple WB background: shape={image.shape}, "
                        f"median={actual_intensity:.1f}, scale={scale:.3f}"
                    )
                    # Warn if background is still saturated after adaptive loop
                    sat_frac = float(np.mean(image >= 254)) if image.dtype == np.uint8 else 0.0
                    if sat_frac > 0.05:
                        logger.warning(
                            "SATURATION WARNING: simple WB angle=%s has %.1f%% clipped pixels",
                            angle, sat_frac * 100)

                    # Store reference for biref pair matching
                    if angle > 0 and angle != 90:
                        biref_pair_references[angle] = image.copy()
                except RuntimeError as e:
                    logger.error(f"Failed to acquire simple WB background at angle {angle}: {e}")
                    continue

            elif jai_calibration and use_per_angle_wb:
                # Per-angle white balance mode: use per-channel adaptive exposure
                # This maintains the R:G:B ratio while scaling to target intensity
                angle_mapping = {90.0: "uncrossed", 0.0: "crossed", 7.0: "positive", -7.0: "negative"}
                angle_name = angle_mapping.get(angle)
                if not angle_name:
                    for a, name in angle_mapping.items():
                        if abs(a - angle) < 1.0:
                            angle_name = name
                            break

                if angle_name and "angles" in jai_calibration:
                    angle_cal = jai_calibration["angles"].get(angle_name)
                    if angle_cal and "exposures_ms" in angle_cal:
                        per_channel_exp = angle_cal["exposures_ms"]
                        # Per-angle white balance: use calibrated exposures DIRECTLY
                        # No adaptive adjustment - the calibration already determined the
                        # correct per-channel exposures for white balance at a specific intensity
                        logger.info(f"Applying calibrated per-channel exposures for angle {angle}")
                        logger.info(f"  R={per_channel_exp.get('r', 50):.1f}ms, "
                                   f"G={per_channel_exp.get('g', 50):.1f}ms, B={per_channel_exp.get('b', 50):.1f}ms")
                        try:
                            # Apply the calibrated per-channel exposures
                            hardware.camera.set_channel_exposures(
                                red=per_channel_exp.get('r', 50.0),
                                green=per_channel_exp.get('g', 50.0),
                                blue=per_channel_exp.get('b', 50.0),
                                auto_enable=True,
                            )

                            # Apply calibrated gain settings
                            # Read from gains sub-dict (new calibration format)
                            per_channel_gains = angle_cal.get("gains", {})
                            unified_gain = per_channel_gains.get("unified_gain", 1.0)
                            analog_red = per_channel_gains.get("analog_red", 1.0)
                            analog_blue = per_channel_gains.get("analog_blue", 1.0)

                            # ALWAYS apply unified gain and R/B analog gains.
                            # During PPM acquisition the camera cycles through angles
                            # with different gains. Skipping unity-gain angles leaves
                            # the previous angle's gain active -> saturated images.
                            hardware.camera.set_unified_gain(unified_gain)
                            hardware.camera.set_rb_analog_gains(
                                analog_red=analog_red, analog_blue=analog_blue
                            )
                            logger.info(
                                f"  Applied gains: unified={unified_gain:.2f}x, "
                                f"analog R={analog_red:.3f}, B={analog_blue:.3f}"
                            )

                            # Capture single image with calibrated settings
                            image, metadata = hardware.snap_image()
                            if image is None:
                                raise RuntimeError("Failed to acquire image")

                            actual_intensity = float(np.median(image))
                            # Store green channel as reference exposure for compatibility
                            final_exposures[angle] = per_channel_exp.get('g', 100.0)
                            achieved_intensities[angle] = actual_intensity

                            # Log per-channel intensities for WB verification
                            if image.ndim == 3 and image.shape[2] >= 3:
                                r_mean = float(np.mean(image[:, :, 0]))
                                g_mean = float(np.mean(image[:, :, 1]))
                                b_mean = float(np.mean(image[:, :, 2]))
                                logger.info(
                                    f"Acquired background: shape={image.shape}, "
                                    f"R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f} "
                                    f"(median={actual_intensity:.1f})"
                                )
                            else:
                                logger.info(
                                    f"Acquired background: shape={image.shape}, median={actual_intensity:.1f}"
                                )

                            # Per-angle mode uses calibrated exposures verbatim --
                            # if calibration is stale (wrong lamp, wrong objective),
                            # the image can be fully saturated. Detect and abort
                            # rather than silently saving a useless background.
                            sat_frac = float(np.mean(image >= 254)) if image.dtype == np.uint8 else 0.0
                            if sat_frac > 0.50:
                                raise RuntimeError(
                                    f"Per-angle background at {angle} deg is {sat_frac:.0%} saturated. "
                                    f"Calibration may be stale (captured at different lamp/objective). "
                                    f"Recalibrate white balance before collecting backgrounds."
                                )
                            elif sat_frac > 0.05:
                                logger.warning(
                                    "SATURATION WARNING: per-angle WB angle=%s has %.1f%% clipped pixels",
                                    angle, sat_frac * 100)

                            # Under-exposure guard. Only enabled for modalities
                            # where every tile is expected to fill the dynamic
                            # range (PPM, Brightfield) -- on fluorescence /
                            # laser-scanning channels a dim tile is normal
                            # (rare cell types, sparse signals) and warning
                            # would be noise. The 2026-04-27 silent-first-
                            # detector incident produced exactly this dim
                            # pattern and would have been caught here.
                            if _modality_expects_uniform_brightness(modality):
                                if image.dtype == np.uint8:
                                    median = float(np.median(image))
                                    # 20/255 ~= 8% of full scale; well below any
                                    # reasonable calibration target (125-245).
                                    if median < 20.0:
                                        logger.warning(
                                            "UNDER-EXPOSURE WARNING: per-angle WB angle=%s has "
                                            "median=%.1f (uniform-bright modality '%s' expects ~target). "
                                            "Likely stale calibration or wrong detector profile.",
                                            angle, median, modality,
                                        )

                            # Store reference for biref pair matching
                            if angle > 0 and angle != 90:
                                biref_pair_references[angle] = image.copy()
                                logger.info(f"Stored +{angle} image as reference for birefringence pair matching")
                        except RuntimeError as e:
                            logger.error(f"Failed to acquire background at angle {angle}: {e}")
                            continue
                    else:
                        logger.warning(f"No calibration for angle {angle_name}, falling back to standard mode")
                        # Fall through to standard acquisition below
                        jai_calibration = None  # Disable for this angle
                else:
                    logger.warning(f"Unknown angle {angle}, falling back to standard mode")
                    jai_calibration = None  # Disable for this angle

            # Standard acquisition mode (no per-angle white balance or fallback)
            # Skip if image was already acquired by camera_awb or simple mode above
            if image is not None and wb_mode in ("camera_awb", "simple"):
                pass  # Image already acquired above
            elif not (jai_calibration and use_per_angle_wb):
                # For negative polarization angles, use biref-matching against positive angle
                paired_positive = abs(angle)  # e.g., -7 pairs with 7
                if angle < 0 and angle != -90:  # Negative polarization angle (not -90 brightfield)
                    if paired_positive in biref_pair_references:
                        # Use biref-matching: minimize sum of abs channel differences
                        reference_image = biref_pair_references[paired_positive]
                        logger.info(
                            f"Biref pair matching: minimizing biref metric against +{paired_positive} reference"
                        )
                        try:
                            image, final_exposure, achieved_biref = acquire_background_with_biref_matching(
                                hardware=hardware,
                                reference_image=reference_image,
                                tolerance=5.0,  # Target mean biref <= 5
                                initial_exposure_ms=initial_exposure_ms,
                                max_iterations=10,
                                logger=logger,
                            )
                            actual_intensity = float(np.median(image))
                            logger.info(
                                f"Acquired background: shape={image.shape}, "
                                f"achieved_biref={achieved_biref:.1f}, median={actual_intensity:.1f}, "
                                f"final_exposure={final_exposure:.1f}ms"
                            )
                            _check_saturation(image, f"biref-match angle={angle}", logger)
                            final_exposures[angle] = final_exposure
                            achieved_intensities[angle] = actual_intensity
                        except RuntimeError as e:
                            logger.error(f"Failed to acquire background at angle {angle}: {e}")
                            continue
                    else:
                        # Positive angle hasn't been acquired yet - fall back to intensity matching
                        logger.warning(
                            f"Biref pair matching: +{paired_positive} not yet acquired. "
                            f"For best results, acquire positive angles before negative. "
                            f"Falling back to intensity-based matching."
                        )
                        target_intensity = (target_intensity_override
                                    if target_intensity_override and target_intensity_override > 0
                                    else get_target_intensity_for_background(modality, angle))
                        try:
                            image, final_exposure = acquire_background_with_target_intensity(
                                hardware=hardware,
                                target_intensity=target_intensity,
                                tolerance=2.5,
                                initial_exposure_ms=initial_exposure_ms,
                                max_iterations=10,
                                logger=logger,
                            )
                            actual_intensity = float(np.median(image))
                            logger.info(
                                f"Acquired background: shape={image.shape}, "
                                f"median={actual_intensity:.1f}, "
                                f"final_exposure={final_exposure:.1f}ms"
                            )
                            _check_saturation(image, f"fallback angle={angle}", logger)
                            final_exposures[angle] = final_exposure
                            achieved_intensities[angle] = actual_intensity
                        except RuntimeError as e:
                            logger.error(f"Failed to acquire background at angle {angle}: {e}")
                            continue
                else:
                    # Non-biref angles (0, 90, positive angles): use standard intensity matching
                    target_intensity = (target_intensity_override
                                    if target_intensity_override and target_intensity_override > 0
                                    else get_target_intensity_for_background(modality, angle))
                    logger.info(f"Target intensity: {target_intensity:.1f}")

                    try:
                        image, final_exposure = acquire_background_with_target_intensity(
                            hardware=hardware,
                            target_intensity=target_intensity,
                            tolerance=2.5,
                            initial_exposure_ms=initial_exposure_ms,
                            max_iterations=10,
                            logger=logger,
                        )
                        actual_intensity = float(np.median(image))
                        logger.info(
                            f"Acquired background: shape={image.shape}, median={actual_intensity:.1f}, "
                            f"final_exposure={final_exposure:.1f}ms"
                        )
                        _check_saturation(image, f"standard angle={angle}", logger)
                        final_exposures[angle] = final_exposure
                        achieved_intensities[angle] = actual_intensity

                        # Store reference image for positive polarization angles
                        # This will be used by paired negative angles for biref matching
                        if angle > 0 and angle != 90:  # Positive polarization angles (not brightfield)
                            biref_pair_references[angle] = image.copy()
                            logger.info(
                                f"Stored +{angle} image as reference for birefringence pair matching"
                            )
                    except RuntimeError as e:
                        logger.error(f"Failed to acquire background at angle {angle}: {e}")

            # Save background image.
            # Rotation modalities (PPM): save as {angle}.tif (e.g. 7.0.tif).
            # Non-rotation modalities (brightfield, fluorescence, monochrome):
            # save as background.tif -- a numeric 0.0.tif name is confusing
            # when there is no real polarization angle.
            if is_non_rotation_background:
                background_path = output_path / "background.tif"
            else:
                background_path = output_path / f"{angle}.tif"
            if image is not None and image.ndim == 3 and image.shape[2] >= 3:
                ch_means = image.mean(axis=(0, 1))
                logger.info(
                    f"Background {angle} deg pre-save: shape={image.shape}, "
                    f"R={ch_means[0]:.1f}, G={ch_means[1]:.1f}, B={ch_means[2]:.1f}, "
                    f"median={float(np.median(image)):.1f}"
                )
            ome_tiff_writer(  # background -single
                filename=str(background_path),
                pixel_size_um=hardware.get_pixel_size_um(),
                data=image,
            )

            logger.info(f"Saved background for {angle} deg to {background_path}")

            # Update progress
            update_progress(angle_idx + 1, total_images)

        # Reset per-channel mode and gains after background collection
        # so subsequent operations don't inherit angle-specific settings.
        # CRITICAL: Must disable per-channel mode entirely, not just reset values.
        # If per-channel exposure mode stays active, subsequent set_exposure() calls
        # (e.g., autofocus) will be silently ignored.
        if wb_mode in ("per_angle", "simple", "camera_awb") or (jai_calibration and use_per_angle_wb):
            try:
                hardware.camera.disable_individual_exposure()
                hardware.camera.disable_individual_gain()
                if wb_mode != "camera_awb":
                    # For simple/per_angle: clear analog gains to neutral state
                    hardware.camera.set_rb_analog_gains(analog_red=1.0, analog_blue=1.0)
                else:
                    # For camera_awb: preserve AWB analog gain corrections --
                    # tissue acquisition needs the same gains for flat-field match
                    logger.info("Camera AWB: preserving analog gain corrections through cleanup")
                hardware.camera.set_unified_gain(1.0)
                logger.info("Disabled per-channel mode and reset unified gain after background collection")
            except Exception as e:
                logger.warning(f"Could not reset per-channel mode: {e}")

        logger.info("=== SIMPLE BACKGROUND COLLECTION COMPLETE ===")
        logger.info(f"Successfully collected {len(angles)} background images (wb_mode={wb_mode})")

        # Save background exposures and achieved intensities to imageprocessing YAML
        # This data becomes the source of truth for white balance target intensities
        try:
            save_background_exposures_to_yaml(
                config_path=Path(yaml_file_path),
                final_exposures=final_exposures,
                achieved_intensities=achieved_intensities,
                modality=modality,
                objective=settings.get("objective") or objective,
                detector=settings.get("detector") or detector,
            )
            logger.info("Background exposures saved to imageprocessing YAML")
        except Exception as e:
            logger.warning(f"Failed to save background exposures to YAML: {e}")
            # Non-fatal - continue returning the exposures

        # NOTE: simple_wb.angles is now written during WB calibration (WBSIMPLE),
        # not during background collection. This ensures the acquisition uses the
        # correct unified exposure values from calibration, even if PPM WB later
        # overwrites the shared exposures_ms section.

        # Return final exposures for metadata writing
        return final_exposures

    except Exception as e:
        logger.error(f"Simple background collection failed: {str(e)}", exc_info=True)
        raise


def background_acquisition_workflow(
    yaml_file_path: str,
    output_folder_path: str,
    modality: str,
    angles_str: str,
    exposures_str: Optional[str],
    hardware: PycromanagerHardware,
    config_manager,
    logger,
):
    """
    Acquire background images for flat-field correction.

    IMPORTANT: Position the microscope at a blank area before calling this function.
    The system will acquire images at the current position using adaptive exposure
    to reach target intensities.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds (will create modality subfolder)
        modality: Modality identifier (e.g., "PPM_20x")
        angles_str: String of angles like "(0,90,5,-5)"
        exposures_str: String of initial exposure times like "(1.5,100,50,50)".
                      These are used as starting points for adaptive exposure.
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance

    Returns:
        Tuple[str, Dict[float, float]]: (output_path, final_exposures)
            output_path: Path where backgrounds were saved
            final_exposures: Dictionary mapping angles to final exposure times (ms)
    """
    logger.info("=== BACKGROUND ACQUISITION WORKFLOW STARTED ===")
    logger.warning("Ensure microscope is positioned at a clean, blank area!")

    # Get and log current position for reference
    current_pos = hardware.get_current_position()
    logger.info(
        f"Acquiring backgrounds at position: X={current_pos.x:.1f}, "
        f"Y={current_pos.y:.1f}, Z={current_pos.z:.1f}"
    )

    try:
        # Parse angles and exposures from client
        # Use client's exposures as initial values for adaptive exposure
        angles, exposures = parse_angles_exposures(angles_str, exposures_str)
        logger.info(f"Initial exposures from client: {exposures}")

        # Load the microscope configuration
        if not Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        settings = config_manager.load_config_file(yaml_file_path)
        hardware.settings = settings

        # Re-initialize microscope-specific methods with updated settings
        # This is critical for PPM rotation to work correctly
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Create output directory structure with modality
        output_path = Path(output_folder_path) / "backgrounds" / modality
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving backgrounds to: {output_path}")

        # Track final exposures and achieved intensities for each angle
        final_exposures = {}
        achieved_intensities = {}

        # Track reference images for birefringence pair matching
        # Uses the same metric as biref calculation: sum(|R_pos - R_neg| + ...)
        biref_pair_references = {}  # Maps positive angle -> reference image

        # Acquire background for each angle
        for angle_idx, angle in enumerate(angles):
            # Create angle subdirectory
            angle_dir = output_path / str(angle)
            angle_dir.mkdir(exist_ok=True)

            # Set rotation angle if rotation stage is present
            if hardware.rotation_stage is not None:
                hardware.set_psg_ticks(angle)
                logger.info(f"Set angle to {angle}")

            # Use exposure from client as initial value for adaptive exposure
            initial_exposure_ms = exposures[angle_idx] if angle_idx < len(exposures) else 100.0
            logger.info(f"Initial exposure from client: {initial_exposure_ms:.2f}ms")

            # For negative polarization angles, use biref-matching against positive angle
            paired_positive = abs(angle)  # e.g., -7 pairs with 7
            if angle < 0 and angle != -90:  # Negative polarization angle
                if paired_positive in biref_pair_references:
                    # Use biref-matching: minimize sum of abs channel differences
                    reference_image = biref_pair_references[paired_positive]
                    logger.info(
                        f"Biref pair matching: minimizing biref metric against +{paired_positive} reference"
                    )
                    try:
                        image, final_exposure, achieved_biref = acquire_background_with_biref_matching(
                            hardware=hardware,
                            reference_image=reference_image,
                            tolerance=5.0,
                            initial_exposure_ms=initial_exposure_ms,
                            max_iterations=10,
                            logger=logger,
                        )
                        actual_intensity = float(np.median(image))
                        logger.info(
                            f"Acquired background: achieved_biref={achieved_biref:.1f}, "
                            f"median={actual_intensity:.1f}, final_exposure={final_exposure:.1f}ms"
                        )
                        final_exposures[angle] = final_exposure
                        achieved_intensities[angle] = actual_intensity
                    except RuntimeError as e:
                        logger.error(f"Failed to acquire background at angle {angle}: {e}")
                        continue
                else:
                    # Positive angle not yet acquired - fall back to intensity matching
                    logger.warning(
                        f"Biref pair matching: +{paired_positive} not yet acquired. "
                        f"Falling back to intensity-based matching."
                    )
                    target_intensity = (target_intensity_override
                                    if target_intensity_override and target_intensity_override > 0
                                    else get_target_intensity_for_background(modality, angle))
                    try:
                        image, final_exposure = acquire_background_with_target_intensity(
                            hardware=hardware,
                            target_intensity=target_intensity,
                            tolerance=2.5,
                            initial_exposure_ms=initial_exposure_ms,
                            max_iterations=10,
                            logger=logger,
                        )
                        actual_intensity = float(np.median(image))
                        logger.info(
                            f"Acquired background: median={actual_intensity:.1f}, "
                            f"final_exposure={final_exposure:.1f}ms"
                        )
                        final_exposures[angle] = final_exposure
                        achieved_intensities[angle] = actual_intensity
                    except RuntimeError as e:
                        logger.error(f"Failed to acquire background at angle {angle}: {e}")
                        continue
            else:
                # Non-biref angles: use standard intensity matching
                target_intensity = (target_intensity_override
                                    if target_intensity_override and target_intensity_override > 0
                                    else get_target_intensity_for_background(modality, angle))
                logger.info(f"Target intensity: {target_intensity:.1f}")

                try:
                    image, final_exposure = acquire_background_with_target_intensity(
                        hardware=hardware,
                        target_intensity=target_intensity,
                        tolerance=2.5,
                        initial_exposure_ms=initial_exposure_ms,
                        max_iterations=10,
                        logger=logger,
                    )
                    actual_intensity = float(np.median(image))
                    logger.info(
                        f"Acquired background: median={actual_intensity:.1f}, "
                        f"final_exposure={final_exposure:.1f}ms"
                    )
                    final_exposures[angle] = final_exposure
                    achieved_intensities[angle] = actual_intensity

                    # Store reference for positive polarization angles
                    if angle > 0 and angle != 90:
                        biref_pair_references[angle] = image.copy()
                        logger.info(
                            f"Stored +{angle} image as reference for birefringence pair matching"
                        )
                except RuntimeError as e:
                    logger.error(f"Failed to acquire background at angle {angle}: {e}")
                    continue

            # Save background image
            background_path = angle_dir / "background.tif"
            ome_tiff_writer(  # background 2 with bkg-workflow
                filename=str(background_path),
                pixel_size_um=hardware.get_pixel_size_um(),
                data=image,
            )

            logger.info(f"Saved background for {angle} deg to {background_path}")

        logger.info("=== BACKGROUND ACQUISITION COMPLETE ===")

        # Save background exposures and achieved intensities to imageprocessing YAML
        try:
            save_background_exposures_to_yaml(
                config_path=Path(yaml_file_path),
                final_exposures=final_exposures,
                achieved_intensities=achieved_intensities,
                modality=modality,
                objective=settings.get("objective"),
                detector=settings.get("detector"),
            )
            logger.info("Background exposures saved to imageprocessing YAML")
        except Exception as e:
            logger.warning(f"Failed to save background exposures to YAML: {e}")

        return str(output_path), final_exposures

    except Exception as e:
        logger.error(f"Background acquisition failed: {str(e)}", exc_info=True)
        raise


def polarizer_calibration_workflow(
    yaml_file_path: str,
    output_folder_path: str,
    start_angle: float,
    end_angle: float,
    step_size: float,
    exposure_ms: float,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
    progress_callback=None,
) -> str:
    """
    Calibrate PPM polarizer rotation stage to find crossed polarizer positions.

    IMPORTANT: Position microscope at uniform, bright background before calling.
    This workflow sweeps the rotation stage through angles, measures intensity,
    and determines optimal crossed polarizer positions for config_PPM.yml.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds (will write report at top level)
        start_angle: Starting angle for sweep (degrees)
        end_angle: Ending angle for sweep (degrees)
        step_size: Step size for sweep (degrees)
        exposure_ms: Exposure time (milliseconds)
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance
        progress_callback: Optional callback(current, total, stage, message) for socket keepalive

    Returns:
        str: Path to the calibration report text file
    """
    logger.info("=== POLARIZER CALIBRATION WORKFLOW STARTED ===")
    logger.warning("Ensure microscope is positioned at a uniform, bright background!")

    # Get and log current position for reference
    current_pos = hardware.get_current_position()
    logger.info(
        f"Running calibration at position: X={current_pos.x:.1f}, "
        f"Y={current_pos.y:.1f}, Z={current_pos.z:.1f}"
    )

    try:
        # Load the microscope configuration
        if not Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        settings = config_manager.load_config_file(yaml_file_path)

        # Load and merge LOCI resources (required for rotation stage device lookup)
        loci_rsc_file = str(
            Path(yaml_file_path).parent / "resources" / "resources_LOCI.yml"
        )
        if Path(loci_rsc_file).exists():
            loci_resources = config_manager.load_config_file(loci_rsc_file)
            settings.update(loci_resources)
            logger.info("Loaded and merged LOCI resources")
        else:
            logger.warning(f"LOCI resources file not found: {loci_rsc_file}")

        hardware.settings = settings

        # Re-initialize microscope-specific methods
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Verify PPM rotation stage is available
        if hardware.rotation_stage is None:
            raise RuntimeError(
                "No rotation stage configured. PPM calibration requires a "
                "rotation stage. Check modality and ppm_optics settings."
            )

        # Import the calibration utility
        from ppm_library.ppm.polarizer_calibration import PolarizerCalibrationUtils

        # Run two-stage calibration to determine exact hardware offset
        logger.info(
            f"Starting two-stage hardware calibration: "
            f"Coarse: 0-360 deg in {step_size} deg steps, "
            f"Fine: +/-{step_size} deg in 0.1 deg steps"
        )
        logger.info(f"Exposure: {exposure_ms} ms")

        result = PolarizerCalibrationUtils.calibrate_hardware_offset_with_stability_check(
            hardware=hardware,
            num_runs=3,  # Run 3 times to validate stability
            stability_threshold_counts=50.0,  # Warn if variation > 0.05 deg
            coarse_range_deg=360.0,  # Full rotation
            coarse_step_deg=step_size,  # Use user-specified step size for coarse
            fine_range_deg=10.0,  # Increased from 5.0 for better safety margin
            fine_step_deg=0.1,  # Fine step for precise positioning
            exposure_ms=exposure_ms,
            channel=1,  # Green channel
            logger_instance=logger,
            progress_callback=progress_callback,  # Keep socket alive during long calibration
        )

        # Write calibration report
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"polarizer_calibration_{timestamp}.txt"
        report_path = Path(output_folder_path) / report_filename

        # Ensure output directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            # Get the first run's results for displaying exact positions
            primary_result = result.get('all_runs', [result])[0] if 'all_runs' in result else result
            hw_per_deg = result.get('hw_per_deg', primary_result.get('hw_per_deg', 1000.0))

            f.write("=" * 80 + "\n")
            f.write("PPM POLARIZER CALIBRATION RESULTS\n")
            f.write("=" * 80 + "\n\n")

            # ===== RESULTS FIRST - THE KEY VALUES =====
            f.write("CROSSED POLARIZER POSITIONS (use these values in config_PPM.yml):\n\n")

            f.write(f"  >>> ppm_pizstage_offset: {result['recommended_offset']:.1f} <<<\n\n")

            f.write(f"  Found {len(primary_result['exact_minima'])} crossed polarizer positions:\n\n")

            for i, (hw_pos, opt_angle) in enumerate(
                zip(primary_result["exact_minima"], primary_result["optical_angles"])
            ):
                # Find corresponding intensity
                intensity_str = ""
                for fine_result in primary_result["fine_results"]:
                    if abs(fine_result["exact_position"] - hw_pos) < 0.1:
                        intensity_str = f", intensity={fine_result['exact_intensity']:.1f}"
                        break
                f.write(f"    Position {i+1}: {hw_pos:.1f} counts ({opt_angle:.1f} deg optical){intensity_str}\n")

            f.write("\n")

            # Separation check
            if len(primary_result["exact_minima"]) >= 2:
                separation = abs(primary_result["exact_minima"][1] - primary_result["exact_minima"][0])
                separation_deg = separation / hw_per_deg
                f.write(f"  Separation: {separation_deg:.1f} deg (expected: 180.0 deg)\n")

            # Stability summary
            if 'offset_std' in result:
                stability_deg = result['offset_range'] / hw_per_deg
                f.write(f"  Stability: {'PASS' if result['is_stable'] else 'FAIL'} ")
                f.write(f"(variation: {stability_deg:.4f} deg across {len(result.get('all_runs', [result]))} runs)\n")

            f.write("\n")

            # ===== CONFIG RECOMMENDATIONS =====
            f.write("=" * 80 + "\n")
            f.write("CONFIG_PPM.YML UPDATE\n")
            f.write("=" * 80 + "\n\n")

            f.write("Update your config_PPM.yml with:\n\n")
            f.write(f"ppm_pizstage_offset: {result['recommended_offset']:.1f}\n\n")

            f.write("rotation_angles:\n")
            f.write("  - name: 'crossed'\n")
            f.write(f"    tick: 0   # Reference position (hardware: {result['recommended_offset']:.1f})\n")

            if len(primary_result["exact_minima"]) >= 2:
                other_angle = primary_result["optical_angles"][1]
                other_hw = primary_result["exact_minima"][1]
                f.write(f"    # OR tick: {other_angle:.0f}   # Alternate crossed (hardware: {other_hw:.1f})\n")

            f.write("  - name: 'uncrossed'\n")
            f.write("    tick: 90  # 90 deg from crossed (perpendicular)\n\n")

            # ===== CALIBRATION DETAILS =====
            f.write("=" * 80 + "\n")
            f.write("CALIBRATION DETAILS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {yaml_file_path}\n")
            f.write(f"Position: X={current_pos.x:.1f}, Y={current_pos.y:.1f}, Z={current_pos.z:.1f}\n\n")

            f.write("Parameters:\n")
            f.write(f"  Coarse: 0-360 deg in {step_size} deg steps\n")
            f.write(f"  Fine: +/-10 deg in 0.1 deg steps around each minimum\n")
            f.write(f"  Exposure: {exposure_ms} ms, Channel: Green\n")
            f.write(f"  Stability runs: {len(result.get('all_runs', [result]))}\n\n")

            coarse_intensities = primary_result["coarse_intensities"]
            f.write("Intensity Statistics:\n")
            f.write(f"  Range: {coarse_intensities.min():.1f} to {coarse_intensities.max():.1f}\n")
            f.write(f"  Dynamic Range: {coarse_intensities.max() / coarse_intensities.min():.1f}x\n\n")

            # ===== STABILITY CHECK DETAILS =====
            if 'offset_std' in result:
                f.write("=" * 80 + "\n")
                f.write("STABILITY CHECK\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Runs: {len(result.get('all_runs', [result]))}\n")
                f.write(f"Raw offsets: {result['individual_offsets']}\n")
                f.write(f"Normalized (mod 180 deg): {result['normalized_offsets']}\n")
                f.write(f"  Note: Crossed polarizers repeat every 180 deg, so positions\n")
                f.write(f"        differing by 180 deg are equivalent.\n\n")
                f.write(f"Std deviation: {result['offset_std']:.2f} counts ({result['offset_std']/hw_per_deg:.4f} deg)\n")
                f.write(f"Range: {result['offset_range']:.1f} counts ({result['offset_range']/hw_per_deg:.4f} deg)\n")
                f.write(f"Threshold: 50.0 counts (0.05 deg)\n")
                f.write(f"Result: {'PASS - Stable' if result['is_stable'] else 'FAIL - Unstable'}\n")
                if not result['is_stable']:
                    f.write(f"\nWARNING: Check polarizer/analyzer mounts for mechanical issues.\n")
                f.write("\n")

            # ===== RAW DATA =====
            f.write("=" * 80 + "\n")
            f.write("RAW DATA - COARSE SWEEP (RUN 1)\n")
            f.write("=" * 80 + "\n\n")

            f.write("Hardware Position (counts), Intensity\n")
            for hw_pos, intensity in zip(
                primary_result["coarse_hardware_positions"], primary_result["coarse_intensities"]
            ):
                f.write(f"{hw_pos:.1f}, {intensity:.2f}\n")

            # Write raw data for all runs if stability check was performed
            all_runs = result.get('all_runs', [primary_result])
            for run_idx, run_result in enumerate(all_runs, 1):
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"RAW DATA - FINE SWEEPS (RUN {run_idx})\n")
                f.write("=" * 80 + "\n\n")

                for i, fine_result in enumerate(run_result["fine_results"]):
                    f.write(
                        f"\nFine Sweep {i+1} (centered on {fine_result['approximate_position']:.1f}):\n"
                    )
                    f.write("Hardware Position (counts), Intensity\n")
                    for hw_pos, intensity in zip(
                        fine_result["fine_hw_positions"], fine_result["fine_intensities"]
                    ):
                        f.write(f"{hw_pos:.1f}, {intensity:.2f}\n")

        logger.info(f"Calibration report saved to: {report_path}")
        logger.info("=== POLARIZER CALIBRATION WORKFLOW COMPLETE ===")

        return str(report_path)

    except Exception as e:
        logger.error(f"Polarizer calibration failed: {str(e)}", exc_info=True)
        raise
