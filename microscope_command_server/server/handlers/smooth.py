"""Smooth (streaming) focus autofocus handler.

Continuous-Z autofocus built on top of the camera's continuous
sequence acquisition path:

1. Save original stage speed property and Z.
2. Pre-flight: verify exposure * min_velocity is within a motion-blur
   budget, and verify the live image is not saturated. If either
   check fails, respond UNAVAILABLE with a reason -- the Java caller
   falls back to the stepped Sweep Focus path.
3. Seed-move to z_start at FULL speed (positioning move).
4. Drop stage speed property to slow_value.
5. Start continuous sequence acquisition.
6. Fire non-blocking move to z_end.
7. Pop frames from the circular buffer as they arrive, compute a
   focus metric on each, record (t_ms, z_at_pop, metric).
8. Wait for stage done via tight device_busy polling.
9. Stop sequence acquisition, restore speed property.
10. Parabolic fit on the motion-phase samples -> peak Z.
11. Move to peak Z (blocking, busy-polled).

Reads per-objective sweep_range_um from autofocus_<scope>.yml. The
objective is resolved in this order:
    1. --objective <id> from the client
    2. Auto-match by current pixel size (query get_pixel_size_um,
       scan config.hardware.objectives for a pixel_size_xy_um
       entry within 0.01 um)
    3. First entry in autofocus_<scope>.yml as a safe default

Protocol (reuses the existing "--flag value" text payload pattern):

    Command: SMOOTHZ (8 bytes)
    Payload: variable-length string terminated by END_MARKER
             --yaml <path>           (required; path to the active config yaml)
             --objective <id>        (optional; preferred source of truth)
             --range <um>            (optional override of sweep_range_um)

    Response: SUCCESS:<initial>:<final>:<shift>:<n_samples>:<span>
              UNAVAILABLE:<reason>
              FAILED:<reason>

where UNAVAILABLE means a pre-flight check refused to run (caller
should fall back gracefully) and FAILED means a mid-scan error
(caller should report but the stage state is still restored).
"""

import logging
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from microscope_command_server.server.handlers.utils import (
    read_message_string,
    parse_flags,
)

logger = logging.getLogger(__name__)


# ----- Tunables -----

# Stage speed property to search for. First match on the focus
# device's property list wins.
SPEED_PROPERTY_CANDIDATES = ("MaxSpeed", "Velocity", "Speed", "MaxVelocity")

# Slow value used during the scan. Prior ProScan MaxSpeed is a
# 1-100 percent scale; "1" is the slowest usable. Other hardware
# may need per-rig tuning -- eventually this moves to YAML.
SLOW_SPEED_VALUE = "1"

# Normal value to restore after the scan.
NORMAL_SPEED_VALUE = "100"

# Motion blur budget (um). If expected blur per frame exceeds this,
# Smooth is not feasible. Derived from 25% of a representative 20X
# DOF (~2 um).
BLUR_BUDGET_UM = 0.5

# Per-modality saturation refusal thresholds. A uniform 5% check is
# wrong for both extremes:
#
#   - In brightfield, the bright background saturates easily
#     (specular highlights, bare glass, even the illumination field
#     itself) but the tissue itself stays dark and retains focus
#     information. 5% would refuse a perfectly usable scene.
#
#   - In fluorescence or laser-scanning modalities the image is
#     mostly black and the signal is confined to a small fraction of
#     pixels. If 5% of pixels are saturated and 5% of pixels are
#     signal, it's likely that ALL the signal pixels are clipped and
#     focus discrimination is gone. A 5% threshold is way too loose
#     for these modalities -- we need 1-2%.
#
# Map modality names (normalized to lower case) to the max saturation
# fraction allowed before SMOOTH refuses with UNAVAILABLE.
# Values are chosen to be defensible defaults per modality class,
# not per-rig calibrated. A future follow-up may move these into
# config_<scope>.yml per modality.
SATURATION_THRESHOLD_BY_MODALITY = {
    "brightfield": 0.30,  # bright background with dark tissue -- tolerant
    "bf": 0.30,
    "ppm": 0.05,           # polarized: both channels contribute -- moderate
    "polarized": 0.05,
    "fluorescence": 0.02,  # widefield fluorescence -- strict
    "fluorescent": 0.02,
    "widefield": 0.02,
    "wf": 0.02,
    "laser_scanning": 0.01,  # 1P/2P/SHG -- sparse signal, very strict
    "lsm": 0.01,
    "shg": 0.01,
    "multiphoton": 0.01,
    "1p": 0.01,
    "2p": 0.01,
}
# Default when no modality is provided or the provided name is unknown.
# Matches the old blanket behavior.
DEFAULT_SATURATION_REFUSE_FRACTION = 0.05

# Parabolic fit uses this many samples on either side of the argmax
# metric. Keeps the fit robust to flat-top regions.
FIT_NEIGHBORHOOD = 3

# Polling interval inside the scan loop. Tight enough to keep the
# pop loop responsive without burning CPU.
SCAN_POLL_SLEEP_S = 0.002

# Minimum in-motion frames required for a reliable parabolic fit.
# Fewer than this and we refuse to commit -- caller falls back.
MIN_FRAMES_FOR_FIT = 6

# Maximum number of edge-retry attempts beyond the first scan. Each
# retry shifts the scan window one full range in the direction of
# the previously-detected peak. With 2 retries (MAX_EDGE_RETRIES=2)
# the total Z coverage is 3 * range centered on the original initial
# Z -- e.g. a 6 um range scan covers [-9, +9] um around start, a
# 10 um range scan covers [-15, +15] um. Stops early if any
# attempted scan window would step outside the stage z limits from
# config.stage.limits.z_um.
MAX_EDGE_RETRIES = 2

# Default ROI crop factor (fraction of full sensor width/height
# used during the scan). Smaller value = smaller per-frame transfer
# = faster pop loop = denser sampling. Inspired by MM OughtaFocus's
# cropFactor parameter. 0.5 means the center 50% width x 50% height
# = 25% of pixels, reducing ZMQ transfer time by ~4x on cameras
# where transfer is the bottleneck (JAI at 2064x1544 drops from
# ~50-100ms per pop to ~15-30ms).
DEFAULT_CROP_FACTOR = 0.5

# (Drain-based flushing was retired in favor of
# core.clear_circular_buffer() at the top of _run_smooth_scan.
# See the block comment there for why.)

# Hard deadline multiplier. Scan deadline = range_um * HARD_DEADLINE_SEC_PER_UM + 2.0s.
# At SLOW_SPEED_VALUE=1 on Prior (~11.5 um/s) we need ~0.09 s/um, so
# 0.15 gives enough headroom for other stage hardware without being
# absurd.
HARD_DEADLINE_SEC_PER_UM = 0.15

# Tail after expected motion end (motion_duration_ms) before we
# exit the scan loop. Catches frames in flight from the camera
# pipeline plus a small margin for velocity-model error. Kept
# tight (100 ms) because the data showed Prior's device_busy
# lingers for ~2 seconds after physical motion ends -- if we
# waited for device_busy we'd waste that whole window popping
# stable frames at z_end. The retry loop in _attempt_one_scan
# handles the case where velocity_um_s is so wrong that we exit
# before useful samples arrive (it widens the range and re-runs).
SCAN_TAIL_MS = 100.0

# Default fallback range if yaml lookup completely fails.
FALLBACK_RANGE_UM = 6.0


# ----- Small pixel helpers (duplicated from probez for isolation) -----


def _pop_image_as_numpy(core) -> Optional[np.ndarray]:
    """Pop one frame from the circular buffer as a numpy array.

    Handles both monochrome (ndim=2) and multi-component (ndim=3)
    cameras. Returns None if the pop failed or the buffer was empty.
    """
    try:
        pixels = core.pop_next_image()
    except Exception as e:
        logger.debug("pop_next_image failed: %s", e)
        return None
    if pixels is None:
        return None
    try:
        w = core.get_image_width()
        h = core.get_image_height()
        nch = core.get_number_of_components()
    except Exception as e:
        logger.debug("image geometry query failed: %s", e)
        return None
    arr = np.asarray(pixels)
    try:
        if nch == 1:
            return arr.reshape(h, w)
        return arr.reshape(h, w, nch)
    except Exception:
        return arr


def _pop_tagged_frame(core) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Pop one frame from the circular buffer with its camera-native
    elapsed-time metadata.

    Returns (img_numpy, elapsed_time_ms) where:
      - img_numpy is the reshaped pixel array (H, W) or (H, W, C)
      - elapsed_time_ms is the camera's 'ElapsedTime-ms' tag value,
        or None if the metadata key wasn't present

    Falls back to _pop_image_as_numpy() when pop_next_tagged_image()
    isn't available or raises. In that case elapsed_time_ms is None
    and callers should use a wall-clock estimate instead.

    Why this function exists: during a Smooth scan we need to know
    when each popped frame was CAPTURED, not when we POPPED it. The
    buffer can queue frames faster than we pop them (pop takes
    ~100 ms over the ZMQ bridge, camera produces every ~33 ms), so
    by the time we pop sample N the frame is already (N-1)*33 ms
    old and the stage has moved since. Using live stage-position
    reads at pop time labels every sample with the wrong Z.
    """
    try:
        tagged = core.pop_next_tagged_image()
    except Exception as e:
        logger.debug("pop_next_tagged_image unavailable: %s", e)
        return _pop_image_as_numpy(core), None

    if tagged is None:
        return None, None

    try:
        tags = dict(tagged.tags) if hasattr(tagged, "tags") else {}
    except Exception:
        tags = {}
    try:
        pixels = tagged.pix
    except Exception:
        return None, None
    if pixels is None:
        return None, None

    # Extract elapsed time. Key spelling varies across MM/pycromanager
    # versions -- try the common ones.
    elapsed_ms: Optional[float] = None
    for key in ("ElapsedTime-ms", "ElapsedTimeMs", "ElapsedTime", "Elapsed-Time-ms"):
        if key in tags:
            try:
                elapsed_ms = float(tags[key])
                break
            except (TypeError, ValueError):
                pass

    # Reshape pixels. Prefer tag-provided geometry for correctness;
    # fall back to core queries if tags don't have it.
    try:
        h = int(tags.get("Height") or core.get_image_height())
        w = int(tags.get("Width") or core.get_image_width())
    except Exception:
        return None, elapsed_ms
    arr = np.asarray(pixels)
    try:
        total = arr.size
        nch = total // (h * w) if (h * w) > 0 else 1
        if nch <= 1:
            img = arr.reshape(h, w)
        else:
            img = arr.reshape(h, w, nch)
        return img, elapsed_ms
    except Exception:
        return arr, elapsed_ms


def _snap_image_as_numpy(core) -> Optional[np.ndarray]:
    """Snap one image (blocking) and return as numpy array."""
    try:
        core.snap_image()
        pixels = core.get_image()
    except Exception as e:
        logger.debug("snap_image failed: %s", e)
        return None
    if pixels is None:
        return None
    w = core.get_image_width()
    h = core.get_image_height()
    nch = core.get_number_of_components()
    arr = np.asarray(pixels)
    try:
        if nch == 1:
            return arr.reshape(h, w)
        return arr.reshape(h, w, nch)
    except Exception:
        return arr


def _focus_metric_normalized_variance(gray: np.ndarray) -> float:
    """Variance / mean. Robust, cheap, works well for texture-rich
    scenes (brightfield, PPM). Can be misleading for sparse-signal
    fluorescence where a few bright pixels dominate the variance
    regardless of focus."""
    mean = gray.mean()
    if mean <= 1e-9:
        return 0.0
    return float(gray.var() / mean)


def _focus_metric_volath5(gray: np.ndarray) -> float:
    """Volath's F5 metric (autocorrelation at lag 1 minus N*mean^2).

    From OughtaFocus / ImgSharpnessAnalysis in the MicroManager
    source. The comment in the MM code describes this as 'smooths
    out high-frequency (suppresses noise)' which is the key win for
    noisy sparse-signal modalities -- widefield fluorescence and
    laser-scanning microscopy both tend to have mostly-dark
    backgrounds with signal confined to a small fraction of pixels,
    and normalized variance in that regime is dominated by shot
    noise in the background rather than the focus of the signal
    pixels. Volath5's autocorrelation form effectively ignores
    uncorrelated noise.

    Math: F5 = sum_{x, y} I(x, y) * I(x+1, y) - M * N * mean(I)^2

    Equivalent numpy:  (I[:, :-1] * I[:, 1:]).sum() - N * mean(I)^2
    """
    if gray.ndim != 2 or gray.shape[1] < 2:
        return 0.0
    shifted_product = float((gray[:, :-1] * gray[:, 1:]).sum())
    n = float(gray.size)
    mean = float(gray.mean())
    return shifted_product - n * mean * mean


def _focus_metric_tenengrad(gray: np.ndarray) -> float:
    """Sum of squared Sobel gradient magnitudes. Robust sharpness
    metric; the 2016 light-sheet paper cited in MM's
    ImgSharpnessAnalysis calls it 'best non-spectral metric' for
    their application. Alternative to normalized_variance for
    texture-rich tissue imaging.

    Implemented with plain numpy (no scipy dep) via first
    differences in X and Y, squared and summed. This is the
    discrete Sobel without the 2x center weighting; close enough
    for focus ranking purposes.
    """
    if gray.ndim != 2 or gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float((gx * gx).sum() + (gy * gy).sum())


# Registry of available focus metrics. New implementations drop in
# here and are automatically available to the modality dispatcher.
_FOCUS_METRICS = {
    "normalized_variance": _focus_metric_normalized_variance,
    "volath5": _focus_metric_volath5,
    "tenengrad": _focus_metric_tenengrad,
}


# Per-modality default metric. The mapping is defensible rather
# than rig-calibrated: normalized_variance for texture-rich
# modalities, volath5 for sparse-signal modalities where noise
# suppression matters more than dynamic range.
METRIC_BY_MODALITY = {
    "brightfield": "normalized_variance",
    "bf": "normalized_variance",
    "ppm": "normalized_variance",
    "polarized": "normalized_variance",
    "fluorescence": "volath5",
    "fluorescent": "volath5",
    "widefield": "volath5",
    "wf": "volath5",
    "laser_scanning": "volath5",
    "lsm": "volath5",
    "shg": "volath5",
    "multiphoton": "volath5",
    "1p": "volath5",
    "2p": "volath5",
}
DEFAULT_METRIC_NAME = "normalized_variance"


def _resolve_metric_name(modality: Optional[str]) -> str:
    """Pick a focus metric for the given modality, falling back to
    the default when modality is None or unknown."""
    if not modality:
        return DEFAULT_METRIC_NAME
    return METRIC_BY_MODALITY.get(modality.strip().lower(), DEFAULT_METRIC_NAME)


def _focus_metric(img, metric_name: str = DEFAULT_METRIC_NAME) -> float:
    """Compute a focus metric on the given image.

    Dispatches on metric_name. Always extracts the green/first
    channel for multi-component images, then delegates to the
    chosen metric implementation. Returns 0.0 on empty/bad input
    so callers can sort/argmax without special-casing None.
    """
    if img is None:
        return 0.0
    a = np.asarray(img)
    if a.size == 0:
        return 0.0
    if a.ndim == 3:
        ch = 1 if a.shape[2] >= 2 else 0
        gray = a[:, :, ch]
    else:
        gray = a
    g = gray.astype(np.float64, copy=False)
    fn = _FOCUS_METRICS.get(metric_name, _FOCUS_METRICS[DEFAULT_METRIC_NAME])
    try:
        return fn(g)
    except Exception as e:
        logger.debug("focus metric '%s' raised: %s", metric_name, e)
        return 0.0


def _saturation_fraction(img) -> float:
    """Fraction of pixels at/near the dtype maximum."""
    if img is None:
        return 0.0
    a = np.asarray(img)
    if a.dtype == np.uint16:
        threshold = 65000
    else:
        threshold = 250
    if a.ndim == 3:
        a = a[..., 1] if a.shape[-1] >= 2 else a[..., 0]
    sat = (a >= threshold).sum()
    total = a.size
    if total == 0:
        return 0.0
    return float(sat) / float(total)


# ----- YAML loader -----


def _load_autofocus_yaml_for_objective(yaml_path: str, objective: Optional[str]) -> Dict[str, Any]:
    """Load autofocus_<scope>.yml and return the settings dict for the
    given objective. Derives the autofocus file path from the main
    config path (config_<scope>.yml -> autofocus_<scope>.yml).

    Returns an empty dict if the file doesn't exist, yaml parsing
    fails, or the objective isn't found. Callers should treat a
    missing value as "use defaults".
    """
    try:
        import yaml
    except Exception as e:
        logger.warning("PyYAML not available: %s", e)
        return {}

    try:
        config_path = Path(yaml_path)
        config_stem = config_path.stem  # e.g. "config_PPM"
        scope_name = config_stem.replace("config_", "")
        autofocus_file = config_path.parent / f"autofocus_{scope_name}.yml"
    except Exception as e:
        logger.warning("Failed to derive autofocus path from %s: %s", yaml_path, e)
        return {}

    if not autofocus_file.exists():
        logger.warning("Autofocus yaml not found: %s", autofocus_file)
        return {}

    try:
        with open(autofocus_file, "r") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to parse %s: %s", autofocus_file, e)
        return {}

    entries = doc.get("autofocus_settings", []) or []
    if not isinstance(entries, list):
        return {}

    if objective:
        for entry in entries:
            if isinstance(entry, dict) and entry.get("objective") == objective:
                return entry
    # Fall back to the first entry (caller logs this).
    if entries and isinstance(entries[0], dict):
        return entries[0]
    return {}


def _resolve_objective(core, settings, client_objective: Optional[str], pixel_tol: float = 0.01) -> Tuple[Optional[str], str]:
    """Pick an objective id for this Smooth run.

    Returns (objective_id, source_string). source_string is one of
    'client', 'pixel-match', 'fallback', or 'unknown', for logging.
    """
    if client_objective:
        return client_objective, "client"

    try:
        current_px = float(core.get_pixel_size_um())
    except Exception as e:
        logger.debug("get_pixel_size_um failed: %s", e)
        current_px = None

    if current_px and current_px > 0 and settings:
        try:
            hardware_objectives = settings.get("hardware", {}).get("objectives", [])
            for obj in hardware_objectives:
                obj_id = obj.get("id") if isinstance(obj, dict) else None
                if not obj_id:
                    continue
                px_dict = obj.get("pixel_size_xy_um") or {}
                if isinstance(px_dict, dict):
                    for _, px_val in px_dict.items():
                        try:
                            if abs(float(px_val) - current_px) <= pixel_tol:
                                return obj_id, "pixel-match"
                        except Exception:
                            continue
                else:
                    try:
                        if abs(float(px_dict) - current_px) <= pixel_tol:
                            return obj_id, "pixel-match"
                    except Exception:
                        continue
        except Exception as e:
            logger.debug("Objective pixel-match scan failed: %s", e)

    return None, "unknown"


# ----- Property helpers -----


def _str_vector_to_list(vec) -> list:
    if vec is None:
        return []
    try:
        return list(vec)
    except TypeError:
        pass
    try:
        return [vec.get(i) for i in range(int(vec.size()))]
    except Exception:
        return []


def _find_speed_property(core, device: str) -> Optional[str]:
    """Return the first writable speed-like property on `device`
    whose name is in SPEED_PROPERTY_CANDIDATES, or None."""
    try:
        props = _str_vector_to_list(core.get_device_property_names(device))
    except Exception:
        return None
    for name in props:
        if name in SPEED_PROPERTY_CANDIDATES:
            try:
                if core.is_property_read_only(device, name):
                    continue
            except Exception:
                pass
            return name
    return None


def _try_set(core, device: str, prop: str, value: str) -> bool:
    try:
        core.set_property(device, prop, value)
        return True
    except Exception as e:
        logger.debug("set_property(%s.%s=%s) failed: %s", device, prop, value, e)
        return False


def _read_roi(core) -> Optional[Tuple[int, int, int, int]]:
    """Return the current camera ROI as (x, y, w, h).

    pycromanager's get_roi() returns either a 4-tuple OR a
    java_awt_Rectangle object depending on MM version and the
    camera adapter. Try unpacking as a tuple first, fall back to
    the Rectangle attribute accessors. Returns None if neither
    works.
    """
    try:
        roi = core.get_roi()
    except Exception as e:
        logger.warning("SMOOTH: core.get_roi() raised: %s", e)
        return None
    if roi is None:
        return None
    # First try: 4-element iterable (older pycromanager / wrapped tuple).
    try:
        x, y, w, h = roi
        return (int(x), int(y), int(w), int(h))
    except (TypeError, ValueError):
        pass
    # Second try: java.awt.Rectangle with .x/.y/.width/.height attributes.
    try:
        return (int(roi.x), int(roi.y), int(roi.width), int(roi.height))
    except Exception as e:
        logger.warning("SMOOTH: get_roi() returned %r which is neither "
                        "iterable nor Rectangle-shaped: %s", type(roi).__name__, e)
        return None


def _apply_crop_roi(
    core, crop_factor: float
) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
    """Save the current camera ROI and install a centered crop.

    Returns (saved_roi, sequence_was_running_when_called) where:
      - saved_roi is (x, y, w, h) tuple of the ORIGINAL ROI for
        later restoration, or None if the crop didn't apply
      - sequence_was_running_when_called is True if we had to
        stop+restart a running sequence to set the ROI (callers
        of _restore_roi must pass this back)

    JAI / GenAPI cameras lock the Width and Height properties as
    "not writable" while a sequence acquisition is running. So
    the only way to install a new ROI is:

      1. Stop the sequence
      2. Set the ROI
      3. Restart the sequence (with the new ROI in effect)

    This costs ~150 ms of camera warmup vs. the unstop-able path,
    but the per-frame transfer savings dwarf that overhead -- a
    50% crop is 4x fewer pixels per frame, dropping per-pop time
    from ~150 ms to ~40 ms on the JAI. For a 20-sample scan that
    saves ~2 seconds, well over the 300 ms warmup penalty.

    crop_factor=1.0 (no crop) is a no-op that returns (None, False)
    with no camera state changes.

    Restoration is symmetric: caller's finally block invokes
    _restore_roi() which stops the sequence again, sets the
    original ROI, and restarts. The Live Viewer's frame poller
    sees a brief gap in frames and recovers automatically.
    """
    if crop_factor <= 0.0 or crop_factor >= 1.0:
        return (None, False)
    saved = _read_roi(core)
    if saved is None:
        logger.warning("SMOOTH: could not query camera ROI for crop "
                        "(see prior warning); skipping crop")
        return (None, False)
    x0, y0, w0, h0 = saved

    new_w = max(1, int(round(w0 * crop_factor)))
    new_h = max(1, int(round(h0 * crop_factor)))
    new_x = x0 + (w0 - new_w) // 2
    new_y = y0 + (h0 - new_h) // 2

    # JAI / GenAPI requires the sequence to be stopped before ROI
    # changes. Stop, set, restart.
    seq_running = False
    try:
        seq_running = bool(core.is_sequence_running())
    except Exception:
        pass

    if seq_running:
        try:
            core.stop_sequence_acquisition()
        except Exception as e:
            logger.warning("SMOOTH: could not stop sequence for ROI crop: %s", e)
            return (None, False)

    try:
        core.set_roi(new_x, new_y, new_w, new_h)
    except Exception as e:
        logger.warning("SMOOTH: could not install centered crop ROI "
                        "(%d, %d, %d, %d): %s", new_x, new_y, new_w, new_h, e)
        # Try to restart the sequence we stopped before bailing.
        if seq_running:
            try:
                core.start_continuous_sequence_acquisition(0)
            except Exception:
                pass
        return (None, seq_running)

    if seq_running:
        try:
            core.clear_circular_buffer()
            core.start_continuous_sequence_acquisition(0)
            # Brief warmup to let the camera deliver its first
            # post-ROI-change frame before the scan starts popping.
            time.sleep(0.15)
        except Exception as e:
            logger.warning("SMOOTH: could not restart sequence after "
                            "ROI crop: %s", e)
            # Best-effort restore and bail.
            try:
                core.set_roi(int(x0), int(y0), int(w0), int(h0))
            except Exception:
                pass
            return (None, seq_running)

    logger.info("SMOOTH: cropped camera ROI (%d, %d, %dx%d) -> (%d, %d, %dx%d) "
                "(factor=%.2f, pixel area %.0f%% of original)",
                x0, y0, w0, h0, new_x, new_y, new_w, new_h,
                crop_factor, (crop_factor * crop_factor) * 100.0)
    return ((x0, y0, w0, h0), seq_running)


def _restore_roi(
    core,
    saved_roi: Optional[Tuple[int, int, int, int]],
    sequence_was_running: bool,
) -> None:
    """Put the camera ROI back to what _apply_crop_roi() saved.

    Symmetric inverse of _apply_crop_roi: stops the sequence (if
    we had stopped one to install the crop), restores the ROI,
    and restarts. No-op if saved_roi is None.
    """
    if saved_roi is None:
        return
    x0, y0, w0, h0 = saved_roi

    # Stop the sequence again to allow the ROI change. Use the
    # current state, not the saved sequence_was_running flag,
    # because the sequence may have been stopped/restarted in
    # the interim.
    stopped_for_restore = False
    try:
        if core.is_sequence_running():
            core.stop_sequence_acquisition()
            stopped_for_restore = True
    except Exception:
        pass

    try:
        core.set_roi(int(x0), int(y0), int(w0), int(h0))
        logger.info("SMOOTH: restored camera ROI to (%d, %d, %dx%d)",
                    x0, y0, w0, h0)
    except Exception as e:
        # JAI / GenAPI failure mode: when the camera is currently in
        # a cropped state, the GenAPI Width and Height nodes report a
        # Max equal to the *current* (cropped) extent, not the full
        # sensor. set_roi() back to the full-sensor original then
        # fails with "Value=2064 must be <= Max=1552" or similar.
        # Recovery: clear_roi() resets the camera to its full sensor
        # (which restores Width/Height Max to the absolute maximum),
        # then we re-apply the original ROI only if it wasn't already
        # equal to the full sensor.
        logger.debug(
            "SMOOTH: set_roi(%d, %d, %dx%d) failed (%s); "
            "trying clear_roi + retry",
            x0, y0, w0, h0, e,
        )
        try:
            core.clear_roi()
            full_roi = _read_roi(core)
            if full_roi != (x0, y0, w0, h0):
                core.set_roi(int(x0), int(y0), int(w0), int(h0))
            logger.info(
                "SMOOTH: restored camera ROI to (%d, %d, %dx%d) via clear_roi",
                x0, y0, w0, h0,
            )
        except Exception as e2:
            logger.warning(
                "SMOOTH: failed to restore camera ROI even after "
                "clear_roi (%s -> %s)", e, e2,
            )

    # Restart the sequence iff:
    #   (a) we just stopped it for the restore, OR
    #   (b) the caller had a sequence on entry to _apply_crop_roi
    # We never want to restart a sequence that wasn't running
    # before, but we always want to leave it running if it was.
    if stopped_for_restore or sequence_was_running:
        try:
            core.clear_circular_buffer()
            core.start_continuous_sequence_acquisition(0)
        except Exception as e:
            logger.warning("SMOOTH: could not restart sequence after "
                            "ROI restore: %s", e)


def _try_get(core, device: str, prop: str) -> Optional[str]:
    try:
        return core.get_property(device, prop)
    except Exception:
        return None


def _wait_via_busy(core, device: str, timeout_s: float = 10.0) -> None:
    """Tight busy-poll wait for the focus device. Same correctness
    safeguards as microscope_control.hardware.stage._wait_z_via_busy:
    requires 2 consecutive not-busy reads before returning; falls
    back to core.wait_for_device on exception or timeout.
    """
    try:
        deadline = time.perf_counter() + timeout_s
        clear = 0
        while time.perf_counter() < deadline:
            try:
                if not core.device_busy(device):
                    clear += 1
                    if clear >= 2:
                        return
                else:
                    clear = 0
            except Exception:
                break
            time.sleep(0.003)
    except Exception:
        pass
    try:
        core.wait_for_device(device)
    except Exception:
        pass


# ----- Parabolic peak fit -----


def _get_z_limits(settings: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Return (z_low, z_high) from config.stage.limits.z_um, or
    (None, None) if either limit is missing. The scan retry loop
    uses these to refuse attempts that would move the stage past
    the user's configured safety envelope."""
    try:
        z_um = settings.get("stage", {}).get("limits", {}).get("z_um", {})
        low = z_um.get("low")
        high = z_um.get("high")
        return (
            float(low) if low is not None else None,
            float(high) if high is not None else None,
        )
    except Exception:
        return (None, None)


def _scan_window_within_limits(
    z_center: float, range_um: float,
    z_low: Optional[float], z_high: Optional[float],
) -> bool:
    """Check that a proposed scan window centered on `z_center` with
    total span `range_um` fits inside [z_low, z_high]. Missing limits
    (None) count as 'no limit on that side'."""
    half = range_um / 2.0
    z_start = z_center - half
    z_end = z_center + half
    if z_low is not None and z_start < z_low:
        return False
    if z_high is not None and z_end > z_high:
        return False
    return True


def _gaussian_peak(zs: List[float], ms: List[float]) -> Optional[float]:
    """Fit a Gaussian A*exp(-(z-mu)^2 / 2 sigma^2) + C to all samples
    and return mu, the peak location. Returns None on any failure
    (insufficient samples, degenerate z values, fitter non-convergence,
    out-of-bracket mu).

    Motivated by MM's ZStackFocusOptimizer using a full-sample
    Gaussian fit instead of a 3-point parabola. Uses more of the
    scan data than _parabolic_peak does -- the parabolic fit only
    considers the 3 samples around the argmax, discarding all the
    rest. The Gaussian fit averages over all N samples, so a single
    noisy point near the peak doesn't distort the result, and we
    get a natural sigma estimate we could use later to reject flat
    curves.

    Falls back to the parabolic fit path when scipy is unavailable
    or when the Gaussian doesn't converge within reasonable bounds.
    """
    n = len(zs)
    if n < 4:
        return None
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return None

    z_arr = np.asarray(zs, dtype=np.float64)
    m_arr = np.asarray(ms, dtype=np.float64)

    # Initial guesses: mu at argmax, A as max-min, sigma from the
    # z range over 4 (tunable), C at min.
    argmax_idx = int(np.argmax(m_arr))
    mu_init = float(z_arr[argmax_idx])
    m_max = float(m_arr.max())
    m_min = float(m_arr.min())
    A_init = max(m_max - m_min, 1e-6)
    z_range = float(z_arr.max() - z_arr.min())
    if z_range < 1e-6:
        return None
    sigma_init = z_range / 4.0
    C_init = m_min

    def gaussian(z, A, mu, sigma, C):
        return A * np.exp(-0.5 * ((z - mu) / sigma) ** 2) + C

    # Bounds: mu within the sampled range, sigma positive and
    # bounded to prevent degenerate spikes, A positive, C
    # within a reasonable range.
    lo_bounds = [0.0, float(z_arr.min()), 1e-6, -np.inf]
    hi_bounds = [np.inf, float(z_arr.max()), z_range, np.inf]
    try:
        popt, _ = curve_fit(
            gaussian, z_arr, m_arr,
            p0=[A_init, mu_init, sigma_init, C_init],
            bounds=(lo_bounds, hi_bounds),
            maxfev=500,
        )
    except Exception as e:
        logger.debug("gaussian curve_fit failed: %s", e)
        return None

    mu_fit = float(popt[1])
    if not math.isfinite(mu_fit):
        return None
    if mu_fit < z_arr.min() or mu_fit > z_arr.max():
        return None
    return mu_fit


def _parabolic_peak(zs: List[float], ms: List[float]) -> Optional[float]:
    """3-point parabolic fit around the argmax of ms.

    Returns the interpolated z at the peak, or None if the fit is
    degenerate (duplicate z values, wrong curvature, peak outside
    the triplet). Caller should fall back to the raw argmax.
    """
    n = len(zs)
    if n < 3:
        return None
    best_idx = int(np.argmax(ms))
    if best_idx <= 0 or best_idx >= n - 1:
        return None

    # Build triplets around the peak. Use FIT_NEIGHBORHOOD only if
    # they're available; otherwise use the immediate neighbors.
    lo = max(0, best_idx - 1)
    hi = min(n - 1, best_idx + 1)
    z0, z1, z2 = zs[lo], zs[best_idx], zs[hi]
    m0, m1, m2 = ms[lo], ms[best_idx], ms[hi]

    # Degenerate: duplicate z values would divide by zero.
    if abs(z2 - z0) < 1e-6 or abs(z1 - z0) < 1e-6 or abs(z2 - z1) < 1e-6:
        return None

    denom = (z0 - z1) * (z0 - z2) * (z1 - z2)
    if abs(denom) < 1e-12:
        return None
    a = (z2 * (m1 - m0) + z1 * (m0 - m2) + z0 * (m2 - m1)) / denom
    b = (z2 ** 2 * (m0 - m1) + z1 ** 2 * (m2 - m0) + z0 ** 2 * (m1 - m2)) / denom
    if a >= 0:
        # Wrong curvature -- not a maximum.
        return None
    z_peak = -b / (2 * a)
    if z_peak < min(z0, z1, z2) or z_peak > max(z0, z1, z2):
        return None
    return float(z_peak)


def _fit_union_samples(
    samples_zm: List[Tuple[float, float]],
    n_attempts_so_far: int,
) -> Optional["_ScanAttemptResult"]:
    """Fit a peak across the union of (z, metric) samples from
    multiple Smooth attempts.

    Used to short-circuit edge oscillation: when two adjacent scan
    windows each find their peak at the boundary between them, the
    true peak is provably inside the gap and a fit on the union
    finds it directly.

    Accepts the union only when the raw-argmax of the combined data
    is INTERIOR (not at either end of the merged Z range). Otherwise
    we'd be doing the same edge-detection that already failed at the
    per-attempt level. Returns None to signal "no clean union peak,
    fall back to the next strategy".

    Returns a _ScanAttemptResult shaped like _attempt_one_scan's
    success result so the caller can drop it into final_result and
    commit through the same code path.
    """
    if len(samples_zm) < 4:
        return None

    # Sort by z so the argmax-at-edge check has meaning.
    sorted_zm = sorted(samples_zm, key=lambda zm: zm[0])
    zs = [zm[0] for zm in sorted_zm]
    ms = [zm[1] for zm in sorted_zm]
    n = len(zs)

    raw_max_idx = int(np.argmax(ms))
    # Refuse if the union argmax is also at the edge -- we have no
    # evidence the true peak is inside the data we collected.
    if raw_max_idx == 0 or raw_max_idx == n - 1:
        return None

    # Try Gaussian first (uses all samples), fall back to parabolic
    # around the argmax, fall back to raw argmax.
    fit_z: Optional[float] = _gaussian_peak(zs, ms)
    fit_kind = "gaussian"
    if fit_z is None or fit_z < zs[0] or fit_z > zs[-1]:
        fit_z = _parabolic_peak(zs, ms)
        fit_kind = "parabolic"
    if fit_z is None:
        fit_z = zs[raw_max_idx]
        fit_kind = "raw-argmax"

    z_span = zs[-1] - zs[0]
    logger.info(
        "SMOOTH: union-fit across %d samples from %d attempts -- "
        "interior argmax at Z=%.3f (idx %d/%d), fit=%s best_z=%.3f, span=%.2f",
        n, n_attempts_so_far, zs[raw_max_idx], raw_max_idx, n,
        fit_kind, fit_z, z_span,
    )
    return _ScanAttemptResult(
        "success", float(fit_z), n, float(z_span),
        f"union-fit {fit_kind} peak at Z={fit_z:.3f} from {n} samples",
    )


# ----- The scan -----


def _run_smooth_scan(
    core,
    focus_device: str,
    speed_prop: str,
    z_start: float,
    z_end: float,
    hard_deadline_s: float,
    velocity_um_s: float = 11.5,
    metric_name: str = DEFAULT_METRIC_NAME,
) -> List[Tuple[float, float, float]]:
    """Execute the streaming-sample scan and return a list of
    (t_capture_ms, z_interp, metric) triples. Leaves the camera in
    whatever streaming state it was in on entry (caller is
    responsible for starting / stopping the sequence) and does NOT
    restore the speed property; caller is responsible for that too.

    Sample labeling:

    Each sample's Z is computed from a LINEAR MOTION MODEL using the
    frame's CAPTURE timestamp (from camera metadata), not a live
    stage-position query at pop time. This is the fix for the
    pop-time-vs-capture-time bug that corrupted early Smooth runs:
    the buffer can accumulate frames faster than we pop (camera at
    ~30 fps, pop over ZMQ at ~10 fps), so by the time we retrieve a
    frame the stage has moved well past where it was when the sensor
    captured the image. Using core.get_position() at pop time labels
    every sample with the wrong Z and produces (z, metric) pairs
    that are physically impossible (e.g. two samples 5 um apart
    showing identical metric to 0.001 because one of them was
    actually from much earlier in the motion).

    Motion model: z_at_capture = z_start + (t_capture - t_move) *
    velocity_um_s, clamped to [z_start, z_end] for the forward
    direction. The first post-clear frame is assumed captured at
    ~one camera period after move fire (small fixed offset ~33 ms).

    Falls back to pop-time wall clock when the camera doesn't expose
    'ElapsedTime-ms' in the pop metadata (no-metadata path prints a
    one-time warning and uses uniform 30 fps spacing).

    Args:
        velocity_um_s: expected stage velocity during the scan at
            the currently-set slow speed. Used to interpolate Z
            from capture time. 11.5 um/s is the measured Prior
            MaxSpeed=1 forward rate from PROBEZ step 3.
    """
    # Atomic buffer flush just before firing the move. See the
    # earlier block comment for why the loop-based drain didn't
    # work (buffer refills faster than we pop).
    try:
        core.clear_circular_buffer()
        logger.info("SMOOTH: flushed circular buffer before firing move")
    except Exception as e:
        logger.warning("SMOOTH: clear_circular_buffer failed "
                        "(continuing with whatever's queued): %s", e)

    direction = 1.0 if z_end >= z_start else -1.0
    motion_um = abs(z_end - z_start)
    motion_duration_ms = (motion_um / max(velocity_um_s, 0.01)) * 1000.0
    # Time-based scan exit. Replaces the old device_busy poll loop:
    # on Prior's serial-driven adapter, device_busy lingers ~2 s
    # after physical motion ends, so the loop wasted that whole
    # window popping stable frames at z_end. The motion model
    # (velocity_um_s, calibrated by PROBEZ) is a much tighter
    # predictor of when the stage is actually done.
    scan_exit_at_ms = motion_duration_ms + SCAN_TAIL_MS

    # Per-pop record: (pop_wall_ms, metric, raw_pop_index)
    # Z is computed AFTER the scan from the pop order, not during,
    # because the only reliable timing info we have is the FIFO
    # ordering guarantee + the wall-clock duration of the whole
    # scan. ElapsedTime-ms metadata varies in semantics across MM
    # camera drivers (the JAI build saw 17/45 ms alternating
    # values that aren't a real elapsed-since-start clock), so we
    # ignore it for now and rely on the pop ordering.
    pop_records: List[Tuple[float, float, int]] = []

    t0 = time.perf_counter()
    try:
        core.set_position(focus_device, z_end)
    except Exception as e:
        logger.error("SMOOTH: non-blocking move to z_end failed: %s", e)
        return []

    deadline = time.perf_counter() + hard_deadline_s
    pop_index = 0
    tags_logged = False

    while time.perf_counter() < deadline:
        t_now_ms = (time.perf_counter() - t0) * 1000.0
        if t_now_ms > scan_exit_at_ms:
            break

        # One get_remaining_image_count RPC per outer iteration to
        # decide whether to enter the pop loop or sleep. When there
        # are frames, pop ALL of them in one batch (no popped >= 4
        # cap) -- the cap was capping us at ~13 fps when the camera
        # produces ~35 fps, so we were missing more than half the
        # frames. Inside the batch we don't re-check the count;
        # _pop_tagged_frame returns (None, None) when the buffer
        # empties, which terminates the inner loop naturally.
        try:
            remaining = core.get_remaining_image_count()
        except Exception:
            remaining = 0

        if remaining > 0:
            try:
                while True:
                    img, elapsed_ms = _pop_tagged_frame(core)
                    if img is None:
                        break

                    # One-time debug: log the elapsed_ms tag of the
                    # first popped frame for cross-checking against
                    # the wall-clock pop time.
                    if not tags_logged:
                        tags_logged = True
                        try:
                            if elapsed_ms is not None:
                                logger.info(
                                    "SMOOTH: first frame elapsed_ms tag "
                                    "= %.3f (only diagnostic; not used "
                                    "for Z labeling)", elapsed_ms,
                                )
                        except Exception:
                            pass

                    pop_wall_ms = (time.perf_counter() - t0) * 1000.0
                    metric = _focus_metric(img, metric_name)
                    pop_records.append((pop_wall_ms, metric, pop_index))
                    pop_index += 1
            except Exception as e:
                logger.warning("SMOOTH: pop loop failed: %s", e)
        else:
            # Buffer empty -- sleep briefly so we don't burn CPU
            # spinning on get_remaining_image_count RPCs while the
            # camera is producing the next frame.
            time.sleep(SCAN_POLL_SLEEP_S)

    total_scan_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "SMOOTH: scan exit at t=%.0fms (motion_end=%.0fms + tail=%.0fms) "
        "total_pops=%d", total_scan_ms, motion_duration_ms, SCAN_TAIL_MS,
        pop_index,
    )

    # --- Post-scan Z assignment ---
    # FIFO guarantee: pop N retrieves the N-th frame the camera
    # produced after the buffer was cleared. With a constant camera
    # frame rate the N-th frame was captured at N * camera_period_ms
    # (relative to the start of streaming, which is approximately
    # t0 since we cleared the buffer just before firing the move).
    #
    # Estimate camera_period_ms by total_frames / scan_duration.
    # Now that the scan loop exits at motion_duration_ms + tail
    # (not on device_busy), we use the same value here -- it's the
    # actual wall-clock window during which frames were captured.
    if not pop_records:
        return []

    scan_duration_ms = max(motion_duration_ms + SCAN_TAIL_MS, 1.0)

    # Camera period: total elapsed / number of frames captured.
    # This is a good estimator only when the buffer-flush + move
    # timing is tight enough that every produced frame got popped.
    # In practice we drain the buffer after the move is done, so
    # this is approximately correct.
    if pop_index >= 2:
        camera_period_ms = scan_duration_ms / pop_index
    else:
        camera_period_ms = 33.0  # fallback for 30 fps

    # Sanity check: clamp camera_period to [10, 500] ms. If it's
    # way outside that, the scan was unusual (very fast or very
    # slow camera) and we fall back to 33 ms.
    if camera_period_ms < 10.0 or camera_period_ms > 500.0:
        logger.warning("SMOOTH: implausible camera period %.1f ms "
                        "(scan=%dms n=%d); falling back to 33ms",
                        camera_period_ms, int(scan_duration_ms), pop_index)
        camera_period_ms = 33.0

    logger.info("SMOOTH: estimated camera period %.1f ms "
                "(scan duration %dms / %d frames)",
                camera_period_ms, int(scan_duration_ms), pop_index)

    # Now assign each popped frame's capture time + interpolated Z.
    # The N-th popped frame was captured at (N + 0.5) * camera_period
    # -- the +0.5 puts the timestamp at the midpoint of the frame's
    # exposure window rather than its start, which slightly improves
    # accuracy when the exposure is a meaningful fraction of the
    # period.
    samples: List[Tuple[float, float, float]] = []
    for (_, metric, idx) in pop_records:
        capture_offset_ms = (idx + 0.5) * camera_period_ms
        if capture_offset_ms <= 0:
            z_interp = z_start
        elif capture_offset_ms >= motion_duration_ms:
            z_interp = z_end
        else:
            progress_um = (capture_offset_ms / 1000.0) * velocity_um_s * direction
            z_interp = z_start + progress_um
        samples.append((capture_offset_ms, float(z_interp), metric))

    return samples


# ----- Handler entry point -----


class _ScanAttemptResult:
    """Result of one _attempt_one_scan call.

    status is one of:
        'success'             -- peak found, best_z set
        'edge_low'            -- argmax at first usable sample; shift down
        'edge_high'           -- argmax at last usable sample; shift up
        'insufficient_samples' -- not enough samples for a fit
        'error'               -- hardware or protocol error mid-scan
    """
    def __init__(self, status: str, best_z: Optional[float],
                 n_samples: int, z_span: float, reason: str,
                 samples_trace: Optional[list] = None):
        self.status = status
        self.best_z = best_z
        self.n_samples = n_samples
        self.z_span = z_span
        self.reason = reason
        self.samples_trace = samples_trace or []


def _attempt_one_scan(
    core,
    focus_device: str,
    speed_prop: str,
    z_center: float,
    range_um: float,
    sequence_was_running_on_entry: bool,
    attempt_label: str = "",
    velocity_um_s: float = 11.5,
    metric_name: str = "normalized_variance",
) -> _ScanAttemptResult:
    """Run one Smooth scan centered on z_center with the given range.

    Returns an _ScanAttemptResult describing the outcome. Does NOT
    commit the peak (caller decides whether to retry or commit) and
    does NOT restore the stage Z (caller handles cleanup).

    The `attempt_label` is prepended to log lines so multi-attempt
    runs are easy to follow (e.g. 'attempt 2/3: ').

    Args:
        velocity_um_s: expected slow-speed stage velocity; used by
            _run_smooth_scan to interpolate Z at frame capture time.
        metric_name: which focus metric to compute per frame.
    """
    tag_prefix = f"{attempt_label}: " if attempt_label else ""
    z_start = z_center - range_um / 2.0
    z_end = z_center + range_um / 2.0
    logger.info("SMOOTH: %sscan window [%.3f -> %.3f] (center %.3f, range %.2f)",
                tag_prefix, z_start, z_end, z_center, range_um)

    try:
        # Positioning seed at full speed.
        _try_set(core, focus_device, speed_prop, NORMAL_SPEED_VALUE)
        core.set_position(focus_device, z_start)
        _wait_via_busy(core, focus_device)

        # Drop to slow speed for the scan motion only.
        if not _try_set(core, focus_device, speed_prop, SLOW_SPEED_VALUE):
            return _ScanAttemptResult(
                "error", None, 0, 0.0,
                f"could not set {speed_prop}={SLOW_SPEED_VALUE}",
            )

        if sequence_was_running_on_entry:
            logger.info("SMOOTH: %sreusing already-running sequence", tag_prefix)
        else:
            logger.info("SMOOTH: %sno active sequence; starting one for the scan",
                        tag_prefix)
            core.clear_circular_buffer()
            core.start_continuous_sequence_acquisition(0)
            time.sleep(0.15)

        hard_deadline_s = max(1.0, range_um * HARD_DEADLINE_SEC_PER_UM + 2.0)
        samples = _run_smooth_scan(core, focus_device, speed_prop,
                                    z_start, z_end, hard_deadline_s,
                                    velocity_um_s=velocity_um_s)

        if not sequence_was_running_on_entry:
            try:
                core.stop_sequence_acquisition()
            except Exception:
                pass
            try:
                core.clear_circular_buffer()
            except Exception:
                pass

        _try_set(core, focus_device, speed_prop, NORMAL_SPEED_VALUE)

        # --- Sample filtering and fit ---
        clean = [(t, z, m) for (t, z, m) in samples
                 if z == z and m == m and math.isfinite(z) and math.isfinite(m)]
        in_motion = []
        stable_run = 0
        last_z = None
        for (t, z, m) in clean:
            if last_z is not None and abs(z - last_z) < 0.05:
                stable_run += 1
                if stable_run >= 3:
                    break
            else:
                stable_run = 0
                in_motion.append((t, z, m))
                last_z = z
        if len(in_motion) < MIN_FRAMES_FOR_FIT and len(clean) >= MIN_FRAMES_FOR_FIT:
            in_motion = clean[:max(MIN_FRAMES_FOR_FIT, len(in_motion))]

        n_motion_samples = len(in_motion)
        if n_motion_samples >= 2:
            zs = [p[1] for p in in_motion]
            ms = [p[2] for p in in_motion]
            z_span = float(max(zs) - min(zs))
            raw_peak_idx = int(np.argmax(ms))
            raw_peak_z = zs[raw_peak_idx]
            # Prefer a full-sample Gaussian fit (uses all N samples,
            # robust to a single noisy point), fall back to 3-point
            # parabolic (uses only the argmax neighborhood), fall
            # back to raw argmax.
            gaussian_fit = _gaussian_peak(zs, ms) if n_motion_samples >= 4 else None
            parabolic = _parabolic_peak(zs, ms) if n_motion_samples >= 3 else None
            if gaussian_fit is not None:
                best_z = gaussian_fit
                fit_kind = "gaussian"
            elif parabolic is not None:
                best_z = parabolic
                fit_kind = "parabolic"
            else:
                best_z = raw_peak_z
                fit_kind = "raw-argmax"
            logger.info("SMOOTH: %s%d in-motion samples  raw peak Z=%.3f  "
                        "fit=%s best_z=%.3f  z_span=%.3f",
                        tag_prefix, n_motion_samples, raw_peak_z,
                        fit_kind, best_z, z_span)
        else:
            logger.warning("SMOOTH: %sonly %d in-motion samples -- cannot fit",
                           tag_prefix, n_motion_samples)
            return _ScanAttemptResult(
                "insufficient_samples", None, n_motion_samples, 0.0,
                f"only {n_motion_samples} usable samples, need {MIN_FRAMES_FOR_FIT}",
                samples_trace=list(in_motion),
            )

        for i, (t, z, m) in enumerate(in_motion):
            logger.info("SMOOTH: %ssample %3d  t=%7.1f ms  z=%.3f  metric=%.4f",
                        tag_prefix, i, t, z, m)

        if n_motion_samples < MIN_FRAMES_FOR_FIT or best_z is None:
            return _ScanAttemptResult(
                "insufficient_samples", None, n_motion_samples, z_span,
                f"only {n_motion_samples} usable samples, need {MIN_FRAMES_FOR_FIT}",
                samples_trace=list(in_motion),
            )

        # Edge-of-window detection.
        if n_motion_samples >= 3 and raw_peak_idx in (0, n_motion_samples - 1):
            if raw_peak_idx == 0:
                status = "edge_low"
                direction = "more negative Z (below z_start)"
            else:
                status = "edge_high"
                direction = "more positive Z (above z_end)"
            reason = (
                f"peak at edge of scan window (sample {raw_peak_idx} of "
                f"{n_motion_samples}, z={zs[raw_peak_idx]:.3f}, "
                f"metric={ms[raw_peak_idx]:.3f}). True focus is likely "
                f"at {direction}"
            )
            return _ScanAttemptResult(
                status, None, n_motion_samples, z_span, reason,
                samples_trace=list(in_motion),
            )

        return _ScanAttemptResult(
            "success", best_z, n_motion_samples, z_span,
            f"peak at Z={best_z:.3f}",
            samples_trace=list(in_motion),
        )

    except Exception as e:
        logger.error("SMOOTH: %sunhandled error during scan: %s",
                     tag_prefix, e, exc_info=True)
        return _ScanAttemptResult(
            "error", None, 0, 0.0, str(e),
        )


def _brent_fallback_scan(
    core,
    focus_device: str,
    speed_prop: str,
    z_lo: float,
    z_hi: float,
    metric_name: str,
    max_evals: int = 8,
    abs_tolerance_um: float = 0.5,
) -> _ScanAttemptResult:
    """Stop-and-snap Brent's method search over [z_lo, z_hi].

    Used as a final fallback when the streaming retry loop exhausts
    all attempts without finding a peak. Brent's method is much
    smarter about WHERE to sample than the streaming scan's
    uniform coverage, converging in 6-8 evaluations even when the
    peak location is uncertain. On PPM with ~250 ms per snap that's
    ~2 s for a 6 um-wide search, which is competitive with a single
    streaming attempt while being more robust to partial failures.

    This is a stop-and-snap path: each evaluation does a blocking
    move at FULL speed (NORMAL_SPEED_VALUE) + snap + metric. It
    intentionally does NOT use streaming / slow-speed motion
    because Brent only does single Z queries; there's no advantage
    to running the stage slowly for a stationary snap.

    The caller is responsible for:
      - Leaving the camera in a state where snap_image() works
      - Restoring stage speed property on exit
      - Committing the returned best_z (or not)

    Returns a _ScanAttemptResult shaped like _attempt_one_scan's
    success/error results so callers can dispatch uniformly.
    """
    tag = "brent-fallback"
    logger.info("SMOOTH: %s: Brent search over [%.3f, %.3f] metric=%s",
                tag, z_lo, z_hi, metric_name)

    try:
        from scipy.optimize import minimize_scalar
    except Exception as e:
        return _ScanAttemptResult(
            "error", None, 0, 0.0, f"scipy not available for Brent: {e}",
        )

    if z_hi <= z_lo:
        return _ScanAttemptResult(
            "error", None, 0, 0.0, f"empty Brent bracket [{z_lo}, {z_hi}]",
        )

    # Brent's method needs a 3-point bracket where the middle has a
    # lower function value than both ends (we're MINIMIZING negative
    # metric, i.e. maximizing metric). Start with a center at
    # midpoint of the bracket.
    z_mid = (z_lo + z_hi) / 2.0

    # Use full stage speed for Brent evaluations -- each one is a
    # stationary snap, no benefit to running slowly.
    _try_set(core, focus_device, speed_prop, NORMAL_SPEED_VALUE)

    # Track every evaluation for the eventual result.
    evals: List[Tuple[float, float]] = []  # (z, metric)

    def neg_metric(z: float) -> float:
        try:
            core.set_position(focus_device, float(z))
            _wait_via_busy(core, focus_device)
            core.snap_image()
            img = _snap_image_as_numpy(core)
            z_actual = float(core.get_position(focus_device))
        except Exception as e:
            logger.warning("SMOOTH: %s eval at z=%.3f failed: %s", tag, z, e)
            return 0.0
        m = _focus_metric(img, metric_name)
        evals.append((z_actual, m))
        logger.info("SMOOTH: %s eval %2d  z=%.3f  metric=%.4f",
                    tag, len(evals), z_actual, m)
        # minimize_scalar expects a MINIMIZATION objective, so flip
        # the sign: better focus -> lower (more negative) value.
        return -m

    try:
        result = minimize_scalar(
            neg_metric,
            bracket=(z_lo, z_mid, z_hi),
            method="brent",
            options={"xtol": abs_tolerance_um, "maxiter": max_evals},
        )
    except Exception as e:
        logger.warning("SMOOTH: %s minimize_scalar raised: %s", tag, e)
        # Fall back to argmax of what we collected.
        if evals:
            best_z, best_m = max(evals, key=lambda p: p[1])
            z_span = max(z for z, _ in evals) - min(z for z, _ in evals)
            return _ScanAttemptResult(
                "success" if best_m > 0 else "error",
                best_z, len(evals), z_span,
                f"Brent raised ({e}); argmax of {len(evals)} evals",
                samples_trace=list(evals),
            )
        return _ScanAttemptResult(
            "error", None, 0, 0.0, f"Brent failed with no evals: {e}",
        )

    best_z = float(result.x)
    # Clamp to bracket (scipy Brent can sometimes report just outside)
    best_z = max(z_lo, min(z_hi, best_z))
    z_span = (max(z for z, _ in evals) - min(z for z, _ in evals)) if evals else 0.0

    logger.info("SMOOTH: %s converged at z=%.3f after %d evals",
                tag, best_z, len(evals))
    return _ScanAttemptResult(
        "success", best_z, len(evals), z_span,
        f"Brent converged at z={best_z:.3f} after {len(evals)} evals",
        samples_trace=list(evals),
    )


def handle_smoothz(conn, client, hardware, settings, **kwargs):
    """Entry point for the SMOOTHZ command."""
    addr = getattr(client, "addr", client)

    # Read the text payload (same framing as other flag-based handlers).
    try:
        message = read_message_string(conn)
    except Exception as e:
        logger.error("SMOOTH: failed to read payload from %s: %s", addr, e)
        try:
            conn.sendall(f"FAILED:payload-read-error: {e}".encode())
        except Exception:
            pass
        return

    params = parse_flags(message,
                          ["--yaml", "--objective", "--range", "--modality",
                           "--crop-factor"])
    yaml_path = params.get("yaml")
    client_objective = params.get("objective")
    range_override_str = params.get("range")
    client_modality = params.get("modality")
    crop_factor_str = params.get("crop_factor")
    range_override_um: Optional[float] = None
    if range_override_str:
        try:
            range_override_um = float(range_override_str)
        except ValueError:
            logger.warning("SMOOTH: ignoring non-numeric --range: %r", range_override_str)

    crop_factor = DEFAULT_CROP_FACTOR
    if crop_factor_str:
        try:
            cf = float(crop_factor_str)
            if 0.0 < cf <= 1.0:
                crop_factor = cf
            else:
                logger.warning("SMOOTH: --crop-factor=%r out of (0, 1]; "
                                "using default %.2f", crop_factor_str, DEFAULT_CROP_FACTOR)
        except ValueError:
            logger.warning("SMOOTH: ignoring non-numeric --crop-factor: %r",
                            crop_factor_str)

    if not yaml_path:
        try:
            conn.sendall(b"FAILED:missing --yaml")
        except Exception:
            pass
        return

    logger.info("SMOOTH: request from %s yaml=%s objective=%s modality=%s "
                "range_override=%s crop_factor=%.2f",
                addr, yaml_path, client_objective, client_modality,
                range_override_um, crop_factor)

    # Resolve focus metric from modality. Pattern matches the
    # saturation threshold dispatch just below -- brightfield/PPM
    # use normalized_variance, fluorescence and laser-scanning use
    # Volath5 for noise robustness.
    metric_name = _resolve_metric_name(client_modality)
    logger.info("SMOOTH: focus metric for modality '%s' = '%s'",
                client_modality or "unknown", metric_name)

    # Resolve the saturation threshold from the client-provided
    # modality. Normalize to lower case for dict lookup; unknown or
    # missing modalities fall back to the conservative default.
    if client_modality:
        sat_threshold = SATURATION_THRESHOLD_BY_MODALITY.get(
            client_modality.strip().lower(),
            DEFAULT_SATURATION_REFUSE_FRACTION,
        )
        logger.info("SMOOTH: saturation threshold for modality '%s' = %.2f",
                    client_modality, sat_threshold)
    else:
        sat_threshold = DEFAULT_SATURATION_REFUSE_FRACTION
        logger.info("SMOOTH: no modality given, using default saturation threshold %.2f",
                    sat_threshold)

    core = hardware.core
    try:
        focus_device = core.get_focus_device()
    except Exception as e:
        logger.error("SMOOTH: get_focus_device failed: %s", e)
        conn.sendall(f"FAILED:no-focus-device: {e}".encode())
        return
    logger.info("SMOOTH: focus device = %s", focus_device)

    # --- Objective resolution ---
    objective, source = _resolve_objective(core, settings, client_objective)
    if objective:
        logger.info("SMOOTH: resolved objective '%s' via %s", objective, source)
    else:
        logger.warning("SMOOTH: could not resolve objective; using first yaml entry")

    af_entry = _load_autofocus_yaml_for_objective(yaml_path, objective)
    if not af_entry:
        logger.warning("SMOOTH: no autofocus yaml entry -- using fallback range %s um",
                       FALLBACK_RANGE_UM)

    if range_override_um is not None:
        range_um = max(1.0, float(range_override_um))
        logger.info("SMOOTH: using range override = %.2f um", range_um)
    else:
        range_um = float(af_entry.get("sweep_range_um", FALLBACK_RANGE_UM))
        logger.info("SMOOTH: using sweep_range_um from yaml = %.2f um", range_um)

    # --- Speed property discovery ---
    speed_prop = _find_speed_property(core, focus_device)
    if speed_prop is None:
        reason = (f"focus device '{focus_device}' has no speed property "
                  f"(MaxSpeed/Velocity/Speed/MaxVelocity)")
        logger.warning("SMOOTH: UNAVAILABLE -- %s", reason)
        conn.sendall(f"UNAVAILABLE:{reason}".encode())
        return
    logger.info("SMOOTH: stage speed property = '%s'", speed_prop)

    original_speed = _try_get(core, focus_device, speed_prop)
    try:
        initial_z = float(core.get_position(focus_device))
    except Exception as e:
        logger.error("SMOOTH: get_position failed: %s", e)
        conn.sendall(f"FAILED:get-position: {e}".encode())
        return

    # Apply the central ROI crop before preflight. This reduces the
    # per-frame pixel transfer for both the saturation-check frame
    # AND every frame popped during the scan, which is the single
    # biggest speedup available on cameras where the ZMQ transfer
    # dominates per-pop cost.
    #
    # On JAI / GenAPI cameras the Width property is not writable
    # while a sequence acquisition is running, so _apply_crop_roi
    # may need to stop+restart an in-progress stream. The second
    # return value tells _restore_roi whether to put the sequence
    # back at the end. Both saved values are passed to the helper
    # in the handler's finally block.
    saved_roi, roi_seq_was_running = _apply_crop_roi(core, crop_factor)

    # --- JAI FrameRateHz sanity check + force-to-max ---
    # The JAI device adapter has a 'FrameRateHz' property hardware-
    # coupled to the Exposure property. The two are normally kept in
    # sync by JAICamera.set_exposure(), which sets BOTH properties.
    # But if anything has set the JAI's Exposure directly (Device
    # Property Browser, a stale preset, or any code path that bypasses
    # the camera class) FrameRateHz can be left at a value that
    # bottlenecks streaming production rate, regardless of how short
    # the actual exposure is. Observed on PPM 2026-04-14: a 0.5 ms
    # Exposure with FrameRateHz=1 produced ~1.2 fps during a Smooth
    # scan, giving 1-2 samples instead of 15+.
    #
    # Fix: read FrameRateHz, force it to FRAME_RATE_MAX (38 Hz) for
    # the scan if it's below a usable threshold, and restore on exit.
    # The warning tells the operator that their camera preset is
    # misconfigured and Live Viewer streaming will also be running
    # slow until they re-apply a SETCAM-driven preset.
    saved_frame_rate_hz: Optional[float] = None
    try:
        active_cam = core.get_camera_device()
    except Exception:
        active_cam = None
    if active_cam == "JAICamera":
        try:
            saved_frame_rate_hz = float(
                core.get_property("JAICamera", "FrameRateHz")
            )
        except Exception as e:
            logger.warning("SMOOTH: could not read JAICamera FrameRateHz: %s", e)
            saved_frame_rate_hz = None
        if saved_frame_rate_hz is not None and saved_frame_rate_hz < 30.0:
            logger.warning(
                "SMOOTH: JAICamera FrameRateHz=%.2f Hz is too low for "
                "streaming focus; temporarily forcing to 38 Hz. The Live "
                "Viewer was also producing frames at this rate -- re-apply "
                "your camera preset to fix it permanently.",
                saved_frame_rate_hz,
            )
            try:
                core.set_property("JAICamera", "FrameRateHz", 38.0)
                logger.info(
                    "SMOOTH: bumped JAICamera FrameRateHz from %.2f to 38.0",
                    saved_frame_rate_hz,
                )
            except Exception as e:
                logger.warning(
                    "SMOOTH: could not set JAICamera FrameRateHz=38 mid-stream "
                    "(%s); scan may still be starved", e,
                )
        elif saved_frame_rate_hz is not None:
            logger.info(
                "SMOOTH: JAICamera FrameRateHz=%.2f Hz (above threshold, leaving alone)",
                saved_frame_rate_hz,
            )

    # --- Pre-flight: exposure * velocity blur budget ---
    try:
        exposure_ms = float(core.get_exposure())
    except Exception as e:
        logger.warning("SMOOTH: get_exposure failed: %s", e)
        exposure_ms = 0.0

    # Use a conservative min velocity estimate of 11.5 um/s (Prior
    # MaxSpeed=1 forward) unless we have a better source. Eventually
    # this comes from per-rig calibration; for v1 the fallback
    # matches the only rig we've measured.
    min_velocity_um_s = 11.5
    expected_blur_um = min_velocity_um_s * (exposure_ms / 1000.0) if exposure_ms else 0.0
    logger.info("SMOOTH: exposure=%.2fms  est min velocity=%.2f um/s  "
                "expected blur=%.3f um  budget=%.3f um",
                exposure_ms, min_velocity_um_s, expected_blur_um, BLUR_BUDGET_UM)
    if expected_blur_um > BLUR_BUDGET_UM:
        reason = (f"exposure {exposure_ms:.1f} ms x min velocity {min_velocity_um_s:.1f} "
                  f"um/s = {expected_blur_um:.2f} um motion blur, exceeds "
                  f"{BLUR_BUDGET_UM:.2f} um budget. Reduce exposure to "
                  f"<={BLUR_BUDGET_UM / min_velocity_um_s * 1000:.1f} ms "
                  f"or use a faster stage")
        logger.warning("SMOOTH: UNAVAILABLE -- %s", reason)
        conn.sendall(f"UNAVAILABLE:{reason}".encode())
        return

    # --- Pre-flight: saturation check ---
    # If the Live Viewer (or any caller) has a sequence running, pop
    # one frame from its buffer instead of calling snap_image(). A
    # blocking snap on the JAI costs ~400 ms (exposure + readout +
    # driver overhead) and is the single biggest fixed cost in the
    # Smooth handler -- nearly 20% of the total scan time. Stream
    # frames are already arriving at ~30 fps so a pop-with-timeout
    # gets us a fresh frame in <50 ms.
    preflight_sequence_running = False
    try:
        preflight_sequence_running = bool(core.is_sequence_running())
    except Exception:
        pass

    preflight_img = None
    if preflight_sequence_running:
        # Wait briefly for a fresh frame from the existing stream.
        # 100 ms is plenty at any realistic camera frame rate.
        deadline = time.perf_counter() + 0.1
        while time.perf_counter() < deadline:
            try:
                if int(core.get_remaining_image_count()) > 0:
                    preflight_img = _pop_image_as_numpy(core)
                    if preflight_img is not None:
                        break
            except Exception:
                break
            time.sleep(0.003)
        if preflight_img is not None:
            logger.info("SMOOTH: pre-flight frame via stream pop (no snap)")
        else:
            logger.info("SMOOTH: stream pop failed, falling back to snap_image")
    if preflight_img is None:
        preflight_img = _snap_image_as_numpy(core)
        logger.info("SMOOTH: pre-flight frame via snap_image")

    sat_frac = _saturation_fraction(preflight_img)
    logger.info("SMOOTH: pre-flight saturation fraction = %.3f (threshold %.2f)",
                sat_frac, sat_threshold)
    if sat_frac > sat_threshold:
        reason = (f"{sat_frac * 100:.1f}% of pixels saturated (threshold for "
                  f"'{client_modality or 'unknown'}' modality is "
                  f"{sat_threshold * 100:.1f}%); focus metric will not "
                  f"discriminate. Reduce exposure/gain before using Smooth")
        logger.warning("SMOOTH: UNAVAILABLE -- %s", reason)
        conn.sendall(f"UNAVAILABLE:{reason}".encode())
        return

    # --- Execute scan with edge-retry loop ---
    # Up to (MAX_EDGE_RETRIES + 1) attempts. Each attempt runs one
    # scan centered on a candidate Z with the current range. On
    # edge_low we shift the next attempt's center down by one full
    # range (covering new ground further in the -Z direction); on
    # edge_high we shift up. The shift never crosses outside the
    # stage Z limits from config.
    z_low, z_high = _get_z_limits(settings)
    logger.info("SMOOTH: stage Z limits from config: low=%s high=%s",
                f"{z_low:.3f}" if z_low is not None else "None",
                f"{z_high:.3f}" if z_high is not None else "None")

    # Check whether the Live Viewer already has a sequence running.
    # Computed once -- attempts share this state since we don't stop
    # the caller's stream between attempts.
    try:
        sequence_was_running = bool(core.is_sequence_running())
    except Exception:
        sequence_was_running = False

    attempts_log: List[str] = []
    final_result: Optional[_ScanAttemptResult] = None
    current_center = initial_z

    # Track every (z, metric) sample produced across all attempts so
    # that on edge oscillation we can fit the union and on Brent
    # bracket failure we can pick from the global argmax instead of
    # losing the streaming data.
    all_attempt_samples_zm: List[Tuple[float, float]] = []
    prev_attempt_status: Optional[str] = None

    try:
        for attempt_idx in range(MAX_EDGE_RETRIES + 1):
            attempt_num = attempt_idx + 1
            label = f"attempt {attempt_num}/{MAX_EDGE_RETRIES + 1}"

            # Check Z limits before each attempt. Refuse if the
            # proposed window would step outside the configured stage
            # limits; the current attempt's center came from a
            # previous edge detection, so this is where we stop
            # walking.
            if not _scan_window_within_limits(current_center, range_um,
                                               z_low, z_high):
                reason = (f"proposed scan window [{current_center - range_um/2:.3f} "
                          f"-> {current_center + range_um/2:.3f}] on "
                          f"{label} would exit stage z limits "
                          f"[{z_low}, {z_high}]")
                logger.warning("SMOOTH: %s", reason)
                attempts_log.append(f"{label}: out-of-range")
                final_result = _ScanAttemptResult(
                    "error", None, 0, 0.0, reason,
                )
                break

            # Run one attempt.
            result = _attempt_one_scan(
                core, focus_device, speed_prop,
                current_center, range_um,
                sequence_was_running,
                attempt_label=label,
                velocity_um_s=min_velocity_um_s,
                metric_name=metric_name,
            )
            attempts_log.append(
                f"{label}: center={current_center:.3f} "
                f"range={range_um:.2f} status={result.status} "
                f"n={result.n_samples} reason='{result.reason}'"
            )

            # Accumulate samples (z, metric) from this attempt.
            # _attempt_one_scan stores in_motion as (t, z, m) tuples.
            for s in result.samples_trace:
                if len(s) >= 3:
                    all_attempt_samples_zm.append((float(s[1]), float(s[2])))

            if result.status == "success":
                final_result = result
                break

            # Edge-oscillation short-circuit: if this attempt is
            # an edge in the OPPOSITE direction from the previous
            # attempt, the true peak is provably between the two
            # windows. Don't retry again -- combine all samples
            # collected so far and fit the union. This catches the
            # case (observed on PPM 10x 23:06) where the peak sits
            # right at the boundary of two adjacent windows and the
            # alternating shifts ping-pong forever, never landing
            # on the actual peak.
            opposite_edge = (
                (prev_attempt_status == "edge_low" and result.status == "edge_high")
                or
                (prev_attempt_status == "edge_high" and result.status == "edge_low")
            )
            if opposite_edge and len(all_attempt_samples_zm) >= 4:
                union_result = _fit_union_samples(
                    all_attempt_samples_zm, len(attempts_log),
                )
                if union_result is not None:
                    final_result = union_result
                    attempts_log.append(
                        f"union-fit: status={union_result.status} "
                        f"n={union_result.n_samples} "
                        f"reason='{union_result.reason}'"
                    )
                    break

            prev_attempt_status = result.status

            if result.status == "edge_low":
                # Shift down by one full range so the next window's
                # upper edge equals this one's lower edge -- we cover
                # new ground without overlap.
                current_center = current_center - range_um
                logger.info("SMOOTH: edge_low -- next attempt center will be %.3f",
                            current_center)
                continue

            if result.status == "edge_high":
                current_center = current_center + range_um
                logger.info("SMOOTH: edge_high -- next attempt center will be %.3f",
                            current_center)
                continue

            # Any other status (insufficient_samples, error) aborts
            # the retry loop -- shifting won't help those.
            final_result = result
            break
        else:
            # Ran out of retries without a success or early exit. The
            # last result is stored in `result` (still in scope).
            final_result = result  # noqa: F821  -- result is bound by the for-loop

        # --- Dispatch based on final result ---
        if final_result is None:
            # Should not happen, but defensive fallback.
            final_result = _ScanAttemptResult(
                "error", None, 0, 0.0, "unknown failure, no attempt completed",
            )

        # --- Union-fit pre-Brent escalation ---
        # Before going to Brent, give the union of all collected
        # samples one more chance. The retry loop may have exited
        # without triggering the in-loop opposite-edge short-circuit
        # (e.g. ran out of attempts on three same-direction edges,
        # or hit the stage limit). If the union has a clean
        # interior maximum we can commit it directly.
        if final_result.status in ("edge_low", "edge_high"):
            union_result = _fit_union_samples(
                all_attempt_samples_zm, len(attempts_log),
            )
            if union_result is not None:
                final_result = union_result
                attempts_log.append(
                    f"union-fit (post-retry): status={union_result.status} "
                    f"n={union_result.n_samples} "
                    f"reason='{union_result.reason}'"
                )

        # --- Brent fallback ---
        # If the union fit also failed (no interior peak in the
        # combined data), fall back to a Brent search. Brent uses
        # smart point placement and typically converges in 6-8
        # evaluations even when the peak location is unknown, so it
        # rescues cases where the streaming+shift approach misses
        # the peak due to sample density, metric noise, or awkward
        # initial offset. We seed the bracket from the metric peak
        # of all collected samples (when available) instead of the
        # full coverage span -- a tight bracket converges faster
        # and avoids Brent landing on irrelevant Z far from any
        # actual sample.
        if final_result.status in ("edge_low", "edge_high"):
            if all_attempt_samples_zm:
                # Anchor on the best sample we've already got, then
                # widen to one full range either side.
                best_z_so_far = max(all_attempt_samples_zm, key=lambda zm: zm[1])[0]
                brent_lo = best_z_so_far - range_um
                brent_hi = best_z_so_far + range_um
            else:
                total_span = range_um * (MAX_EDGE_RETRIES + 1)
                brent_lo = initial_z - total_span / 2.0
                brent_hi = initial_z + total_span / 2.0
            if z_low is not None:
                brent_lo = max(brent_lo, z_low)
            if z_high is not None:
                brent_hi = min(brent_hi, z_high)
            if brent_hi - brent_lo >= 2.0:  # need at least 2 um bracket
                logger.info("SMOOTH: streaming retries exhausted with edge; "
                            "escalating to Brent fallback over [%.3f, %.3f]",
                            brent_lo, brent_hi)
                try:
                    # Stop the caller's sequence temporarily because
                    # Brent's snap_image conflicts with a running stream.
                    resume_sequence = False
                    if sequence_was_running:
                        try:
                            if core.is_sequence_running():
                                core.stop_sequence_acquisition()
                                resume_sequence = True
                        except Exception:
                            pass
                    brent_result = _brent_fallback_scan(
                        core, focus_device, speed_prop,
                        brent_lo, brent_hi, metric_name,
                    )
                    # Restart the sequence if the Live Viewer was
                    # depending on it when we arrived.
                    if resume_sequence:
                        try:
                            core.clear_circular_buffer()
                            core.start_continuous_sequence_acquisition(0)
                        except Exception as e:
                            logger.warning("SMOOTH: could not resume sequence "
                                            "after Brent: %s", e)
                    attempts_log.append(
                        f"brent-fallback: bracket=[{brent_lo:.3f}, "
                        f"{brent_hi:.3f}] status={brent_result.status} "
                        f"n={brent_result.n_samples} "
                        f"reason='{brent_result.reason}'"
                    )
                    # Merge Brent's evals into our global sample
                    # pool. Brent's samples_trace is (z, metric)
                    # 2-tuples; streaming samples are (t, z, m)
                    # but we already projected those to (z, m).
                    for s in brent_result.samples_trace:
                        if len(s) >= 2:
                            all_attempt_samples_zm.append(
                                (float(s[0]), float(s[1]))
                            )
                    # Use Brent's best_z if it converged AND its
                    # metric beats our running global best. This
                    # protects against the failure mode where
                    # Brent's bracketing fails and minimize_scalar
                    # picks a far-edge eval (-9 um catastrophe on
                    # 23:06): instead we commit whichever sample
                    # across all attempts (streaming + Brent) had
                    # the highest metric.
                    global_best = max(
                        all_attempt_samples_zm, key=lambda zm: zm[1],
                        default=None,
                    )
                    if brent_result.status == "success" and brent_result.best_z is not None:
                        # Brent converged. Trust it unless our
                        # global pool has something dramatically
                        # better at a Z that Brent never visited
                        # near. (We don't second-guess a converged
                        # Brent on small differences.)
                        final_result = brent_result
                    elif global_best is not None:
                        gz, gm = global_best
                        z_span = (
                            max(zm[0] for zm in all_attempt_samples_zm)
                            - min(zm[0] for zm in all_attempt_samples_zm)
                        )
                        logger.info(
                            "SMOOTH: Brent did not converge; committing "
                            "global argmax across %d collected samples "
                            "at Z=%.3f (metric=%.4f)",
                            len(all_attempt_samples_zm), gz, gm,
                        )
                        final_result = _ScanAttemptResult(
                            "success", gz, len(all_attempt_samples_zm),
                            z_span,
                            f"global argmax across {len(all_attempt_samples_zm)} "
                            f"samples at Z={gz:.3f}",
                        )
                except Exception as e:
                    logger.error("SMOOTH: Brent fallback raised: %s", e, exc_info=True)

        if final_result.status == "success":
            # Commit the peak Z.
            best_z = final_result.best_z
            core.set_position(focus_device, best_z)
            _wait_via_busy(core, focus_device)
            try:
                final_z = float(core.get_position(focus_device))
            except Exception:
                final_z = best_z

            z_shift = final_z - initial_z
            logger.info("SMOOTH: committed final Z=%.3f  shift=%+.3f  n=%d  span=%.2f  "
                        "after %d attempt(s)",
                        final_z, z_shift, final_result.n_samples,
                        final_result.z_span, len(attempts_log))
            for entry in attempts_log:
                logger.info("SMOOTH: attempt log -- %s", entry)

            response = (f"SUCCESS:{initial_z:.3f}:{final_z:.3f}:{z_shift:+.3f}:"
                        f"{final_result.n_samples}:{final_result.z_span:.3f}")
            try:
                conn.sendall(response.encode())
            except Exception as e:
                logger.error("SMOOTH: reply send failed: %s", e)
        else:
            # Every attempt failed or refused. Restore original Z and
            # respond UNAVAILABLE with a consolidated reason.
            try:
                core.set_position(focus_device, initial_z)
                _wait_via_busy(core, focus_device)
            except Exception:
                pass

            if final_result.status in ("edge_low", "edge_high"):
                summary = (f"could not find peak after {len(attempts_log)} "
                           f"attempts ({MAX_EDGE_RETRIES + 1} max). Last attempt: "
                           f"{final_result.reason}. Try moving Z closer to "
                           f"focus manually or picking a wider scan range")
            elif final_result.status == "insufficient_samples":
                summary = (f"{final_result.reason}; scan too short or "
                           f"stage/camera timing off")
            else:
                summary = final_result.reason

            logger.warning("SMOOTH: UNAVAILABLE -- %s", summary)
            for entry in attempts_log:
                logger.warning("SMOOTH: attempt log -- %s", entry)
            try:
                conn.sendall(f"UNAVAILABLE:{summary}".encode())
            except Exception as e:
                logger.error("SMOOTH: reply send failed: %s", e)

    except Exception as e:
        logger.error("SMOOTH: unhandled error in retry loop: %s", e, exc_info=True)
        try:
            conn.sendall(f"FAILED:{e}".encode())
        except Exception:
            pass
    finally:
        # Safety restore: speed property. We intentionally do NOT
        # restore Z in the success path because we want to leave the
        # stage at the new focus. In error paths the except block
        # above already tried to put it back.
        #
        # Sequence acquisition state: we only stop it if WE started
        # it. If the caller (typically the Live Viewer) already had
        # a stream running when we arrived, we want to leave it
        # running so they keep receiving frames afterwards. Calling
        # stop_sequence_acquisition here would break the Live
        # Viewer's frame poller until it auto-recovers (10+ seconds
        # of dead time).
        if not sequence_was_running:
            try:
                if core.is_sequence_running():
                    core.stop_sequence_acquisition()
            except Exception:
                pass
        if original_speed is not None:
            _try_set(core, focus_device, speed_prop, str(original_speed))
        else:
            _try_set(core, focus_device, speed_prop, NORMAL_SPEED_VALUE)
        # Always restore the camera ROI -- a no-op if crop wasn't
        # applied. Leaving a cropped ROI would affect every
        # subsequent live-viewer frame and every acquisition snap
        # until the user manually reconfigured the camera. The
        # roi_seq_was_running flag tells _restore_roi to bring
        # the original streaming state back up.
        _restore_roi(core, saved_roi, roi_seq_was_running)
        # JAI FrameRateHz: deliberately do NOT restore if the saved
        # value was below the streaming threshold. A low FrameRateHz
        # is almost always a stale misconfiguration (the JAI device
        # adapter persists this property across MM sessions, and the
        # only code path that keeps it in sync with Exposure is
        # JAICamera.set_exposure -- anything that writes Exposure
        # directly leaves FrameRateHz dangling). "Restoring" to a
        # broken value would just perpetuate the bug AND keep the
        # Live Viewer running at the same slow rate. Leaving it at
        # 38 Hz fixes both. If the saved value was already healthy
        # (>= 30 Hz), we restore it so we don't silently change
        # whatever the operator had configured.
        if saved_frame_rate_hz is not None:
            if saved_frame_rate_hz >= 30.0:
                try:
                    core.set_property(
                        "JAICamera", "FrameRateHz", saved_frame_rate_hz,
                    )
                    logger.info(
                        "SMOOTH: restored JAICamera FrameRateHz to %.2f",
                        saved_frame_rate_hz,
                    )
                except Exception as e:
                    logger.warning(
                        "SMOOTH: could not restore JAICamera FrameRateHz: %s",
                        e,
                    )
            else:
                logger.info(
                    "SMOOTH: leaving JAICamera FrameRateHz at 38.0 Hz "
                    "(saved %.2f Hz was a stale misconfiguration; "
                    "Live Viewer will now stream at full rate)",
                    saved_frame_rate_hz,
                )
