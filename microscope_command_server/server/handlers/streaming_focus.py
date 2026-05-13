"""Streaming autofocus handler.

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

    Command: STRMAFZ (8 bytes)
    Payload: variable-length string terminated by END_MARKER
             --yaml <path>           (required; path to the active config yaml)
             --objective <id>        (optional; preferred source of truth)
             --range <um>            (optional override of sweep_range_um)
             --modality <name>       (optional; selects metric/threshold profile)
             --crop-factor <0..1]    (optional override of DEFAULT_CROP_FACTOR)
             --dump 1                (optional; per-sample TIF + CSV diagnostic)
             --max-attempts <N>      (optional; caps the edge-retry walk to N
                                      scans. Default MAX_EDGE_RETRIES+1=3. Pass
                                      1 from acquisition tile-AF so a tight
                                      single scan replaces the multi-attempt
                                      walk.)

    Response: SUCCESS:<initial>:<final>:<shift>:<n_samples>:<span>
              UNAVAILABLE:<reason>
              FAILED:<reason>

where UNAVAILABLE means a pre-flight check refused to run (caller
should fall back gracefully) and FAILED means a mid-scan error
(caller should report but the stage state is still restored).

----------------------------------------------------------------
Attribution / Prior art
----------------------------------------------------------------

Several pieces of this handler are adapted (re-implemented in
Python) from the Micro-Manager open-source project:

    https://github.com/micro-manager/micro-manager
    License: LGPL-2.0

Specifically:

- The ROI crop-factor optimization (``_apply_crop_roi`` /
  ``_restore_roi`` / ``DEFAULT_CROP_FACTOR``) is inspired by the
  ``cropFactor`` parameter in ``OughtaFocus.java`` (MM plugin
  ``plugins/AutofocusFunctions``). MM uses it to cut per-frame
  transfer cost during a scan; we use it the same way for the
  continuous-stream case. Our implementation anchors absolutely on
  the full sensor (clear_roi -> crop -> clear_roi to restore),
  unlike MM's which preserves the entry ROI -- this prevents a
  cropped state from persisting across runs when an exit path is
  unexpectedly bypassed. See ``_apply_crop_roi`` docstring for the
  2026-05-11 rationale.

- ``_focus_metric_volath5`` (Volath's F5 autocorrelation metric)
  and ``_focus_metric_tenengrad`` (Sobel-squared sum) are
  Python re-implementations of the corresponding methods in
  MM's ``ImgSharpnessAnalysis.java``
  (``mmstudio/src/main/java/org/micromanager/internal/utils/imageanalysis``).
  The MM source comments describing these metrics ("smooths out
  high-frequency noise", "best non-spectral metric") guided our
  per-modality metric dispatch.

- ``_gaussian_peak`` fits a full-sample Gaussian to the scan data
  rather than just a 3-point parabolic triplet around the argmax.
  This approach is motivated by MM's ``ZStackFocusOptimizer.java``,
  which uses a full-sample fit via its internal Fitter class.

- ``_brent_fallback_scan`` is a Python port of the Brent's-method
  search pattern from MM's ``BrentFocusOptimizer.java``. We use
  ``scipy.optimize.minimize_scalar`` with the same bracket-then-
  refine structure as the Java version.

No code is copied verbatim; every piece was re-written against
the same algorithmic description. The attribution is here so the
origin of the ideas is traceable and so any reader who wants to
cross-check the implementation can find the upstream source.
"""

import logging
import math
import threading
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

# Default expected slow-speed Z velocity (um/s). Used for sample-time
# Z interpolation in _run_streaming_scan and as the blur-budget gate in
# the saturation pre-flight. Empirically measured on Prior MaxSpeed=1.
# Overridden per-rig by stage.streaming_af.slow_speed_um_per_s in the
# main config YAML.
MIN_VELOCITY_UM_S = 11.5

# Motion blur budget (um). If expected blur per frame exceeds this,
# Streaming autofocus is not feasible. Derived from 25% of a representative 20X
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
# fraction allowed before streaming AF refuses with UNAVAILABLE.
# Values are chosen to be defensible defaults per modality class,
# not per-rig calibrated. A future follow-up may move these into
# config_<scope>.yml per modality.
SATURATION_THRESHOLD_BY_MODALITY = {
    # Brightfield: very tolerant. Bare glass / illumination field /
    # specular highlights routinely saturate >30% of pixels with the
    # tissue features still dark and informative for focus. The
    # operator's reaction to repeated cancellations at modest
    # saturation has been "why are we cancelling again?" -- so the
    # threshold sits at 0.50 to only refuse when saturation is so
    # heavy the metric really has no discrimination left.
    "brightfield": 0.50,
    "bf": 0.50,
    "ppm": 0.05,  # polarized: both channels contribute -- moderate
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

# Minimum metric range (as a fraction of the metric peak value)
# required to trust the fit. Below this the entire scan sits in
# what is effectively one depth-of-field and the metric variation
# is indistinguishable from noise -- the argmax becomes random.
# Dominant failure mode at 10x NA 0.25 when sweep_range_um is
# 10 um or less: the whole scan stays inside the ~10 um DOF, the
# metric varies by <1%, and any committed shift is coin-flip
# random. Empirically seen as 0.6-0.7% at PPM 10x; a real peak
# at 20x gives ~80%+.
#
# 2026-05-04: bumped 5% -> 8% after a PPM 40x scan with 4% range
# committed Z=89.3 when truth was 90 (0.7 um = ~1.2 DOFs off). At
# 4-5% range the gaussian fit minimum-quality is below useful;
# refusing the commit and asking the user to widen the scan or
# focus manually is more honest than landing 1+ DOFs off.
#
# 2026-05-06: dropped back to 4% as a hard floor, but added a
# gaussian-fit shape check (FLAT_METRIC_GAUSSIAN_R2) that lets a
# clean peak through even if amplitude is low. At 10x with
# low-contrast tissue the metric range is routinely 5-8% of peak
# but the curve is a clean Gaussian; rejecting on amplitude alone
# refused real focus repeatedly. The new logic is: refuse only if
# the metric range is below the hard floor AND the gaussian fit
# quality is poor (or the fit didn't converge at all).
FLAT_METRIC_FRACTION = 0.04

# Minimum gaussian-fit R^2 required to trust a low-amplitude scan.
# When the metric range is between FLAT_METRIC_FRACTION and
# FLAT_METRIC_AMPLITUDE_TRUSTED, we additionally require that the
# gaussian fit explains at least this fraction of the variance and
# that sigma is well below the scan range (i.e. it's actually
# peak-shaped, not a flat baseline that happens to satisfy R^2 by
# coincidence). 0.7 is conservative -- a clean low-contrast peak
# routinely fits at 0.85-0.95 in 10x test data.
FLAT_METRIC_GAUSSIAN_R2 = 0.70

# Above this metric range fraction we always trust the fit, no R^2
# check needed -- the peak is unambiguous.
FLAT_METRIC_AMPLITUDE_TRUSTED = 0.15

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
# = faster pop loop = denser sampling. 0.5 means the center 50%
# width x 50% height = 25% of pixels, reducing ZMQ transfer time
# by ~4x on cameras where transfer is the bottleneck (JAI at
# 2064x1544 drops from ~50-100ms per pop to ~15-30ms).
#
# Adapted from: Micro-Manager's OughtaFocus.java ``cropFactor``
# parameter. See the "Attribution / Prior art" section in the
# module docstring above for the full citation.
DEFAULT_CROP_FACTOR = 0.5

# (Drain-based flushing was retired in favor of
# core.clear_circular_buffer() at the top of _run_streaming_scan.
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

    Why this function exists: during a streaming AF scan we need to know
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

    # Diagnostic 2026-05-05: detect the stale-trailing-row bug
    # (TODO_LIST.md). If MM Core's allocated dimensions differ from the
    # camera's reported per-frame dimensions in the tags, the buffer
    # has more rows than the camera writes, and the trailing rows are
    # stale content from prior frames. WARNING fires once per frame
    # with the diff so the contamination band can be sized exactly.
    try:
        tag_h = tags.get("Height")
        tag_w = tags.get("Width")
        if tag_h is not None and tag_w is not None:
            tag_h_int = int(tag_h)
            tag_w_int = int(tag_w)
            core_h = int(core.get_image_height())
            core_w = int(core.get_image_width())
            if tag_h_int != core_h or tag_w_int != core_w:
                if not getattr(_pop_tagged_frame, "_dim_warn_logged", False):
                    logger.warning(
                        "STREAM_AF:DIMENSION MISMATCH detected -- core says "
                        "%dx%d, frame tags say %dx%d (delta_rows=%d, "
                        "delta_cols=%d). This is the stale-trailing-row bug "
                        "from TODO_LIST.md; trailing rows of every popped "
                        "frame contain stale content from prior frames. "
                        "Logged once per session.",
                        core_w,
                        core_h,
                        tag_w_int,
                        tag_h_int,
                        core_h - tag_h_int,
                        core_w - tag_w_int,
                    )
                    _pop_tagged_frame._dim_warn_logged = True
    except Exception:
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


# Focus-metric implementations and the per-modality default lookup
# moved to microscope_imageprocessing.focus on 2026-05-01. The streaming
# AF code path is one of several call sites; consolidating means
# autofocus_<scope>.yml's score_metric field uses the same vocabulary
# everywhere and a typo can't drift between code paths.
#
# The dispatcher (resolve_metric) raises on unknown names. The streaming
# AF wrapper below catches that and falls back to the modality default
# with a logged warning so a stale YAML cannot crash an acquisition --
# the user gets a clear "renamed to X" message in the log instead.
from microscope_imageprocessing.focus import (
    UnknownMetricError,
    modality_default_metric,
    resolve_metric,
)

DEFAULT_METRIC_NAME = "tenengrad"


def _resolve_metric_name(
    modality: Optional[str],
    af_entry: Optional[Dict[str, Any]] = None,
) -> str:
    """Pick a focus metric for the streaming AF run.

    Resolution order:
      1. Per-objective ``score_metric`` from autofocus_<scope>.yml
         (passed in via af_entry). Lets a user override the metric
         per objective without code changes.
      2. Modality default (from focus_metrics_manifest.yml's
         modality_defaults).
      3. ``DEFAULT_METRIC_NAME`` (tenengrad).

    A YAML-named metric not known to the manifest (typo, removed
    alias, or a metric that does not support the streaming path) is
    logged with a clear migration hint and resolution falls through
    to the modality default. The sentinel ``"none"`` is treated as
    "skip YAML override" -- used by the manual_only strategy where
    streaming AF is itself bypassed.
    """
    yaml_metric: Optional[str] = None
    if af_entry:
        raw = af_entry.get("score_metric")
        if isinstance(raw, str):
            yaml_metric = raw.strip().lower()

    if yaml_metric and yaml_metric != "none":
        try:
            resolve_metric(yaml_metric)
            return yaml_metric
        except UnknownMetricError as e:
            logger.warning(
                "STREAM_AF: autofocus yaml score_metric=%r is invalid "
                "(%s); falling back to modality default.",
                yaml_metric,
                e,
            )

    return modality_default_metric(modality, fallback=DEFAULT_METRIC_NAME)


def _focus_metric(img, metric_name: str = DEFAULT_METRIC_NAME) -> float:
    """Compute a focus metric on the given image.

    Dispatches via ``resolve_metric``. The dispatcher accepts 2D and
    3D input directly (multi-channel reduces to the green/index-1
    channel) and returns 0.0 on empty/bad input. If ``metric_name``
    is unknown the call is logged once and falls back to the default.
    """
    if img is None:
        return 0.0
    try:
        fn = resolve_metric(metric_name)
    except UnknownMetricError as e:
        logger.debug(
            "focus metric '%s' not in manifest (%s); using %s",
            metric_name,
            e,
            DEFAULT_METRIC_NAME,
        )
        fn = resolve_metric(DEFAULT_METRIC_NAME)
    try:
        return float(fn(img))
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


def _load_streaming_af_config(yaml_path: str) -> Dict[str, Any]:
    """Load `stage.streaming_af.*` from the main config_<scope>.yml.

    Returns a dict with whatever keys are present, or an empty dict if
    the file is missing / unreadable / lacks the block. Callers treat
    each missing key as "use the legacy hardcoded default" so a
    pre-migration config still works.

    Expected keys (all optional from this loader's perspective; the
    Java schema may require some of them at v3):
        enabled            -- bool
        speed_property     -- str or None
        slow_speed_value   -- str (raw stage value, e.g. '1' or '0.50mm/sec')
        slow_speed_um_per_s -- float (actual velocity, for blur calc)
        normal_speed_value -- str (raw stage value to restore)
    """
    if not yaml_path:
        return {}
    try:
        import yaml
    except Exception as e:
        logger.warning("PyYAML not available: %s", e)
        return {}
    try:
        with open(yaml_path, "r") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to parse %s: %s", yaml_path, e)
        return {}
    block = ((doc.get("stage") or {}).get("streaming_af")) or {}
    if not isinstance(block, dict):
        return {}
    return block


def _resolve_objective(
    core, settings, client_objective: Optional[str], pixel_tol: float = 0.01
) -> Tuple[Optional[str], str]:
    """Pick an objective id for this streaming AF run.

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
        logger.warning("STREAM_AF:core.get_roi() raised: %s", e)
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
        logger.warning(
            "STREAM_AF:get_roi() returned %r which is neither " "iterable nor Rectangle-shaped: %s",
            type(roi).__name__,
            e,
        )
        return None


def _apply_crop_roi(core, crop_factor: float) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
    """Anchor on full sensor, then install a centered crop.

    Returns (saved_roi, sequence_was_running_when_called) where:
      - saved_roi is the FULL-SENSOR (x, y, w, h) tuple, used by
        _restore_roi as a sanity check; None if we could not
        establish a clean full-sensor baseline.
      - sequence_was_running_when_called is True if we had to
        stop+restart a running sequence to set the ROI (callers
        of _restore_roi must pass this back).

    Absolute-anchoring rationale (2026-05-11):
        The previous version stored "whatever ROI is current" as
        saved_roi and cropped 50% relative to it. If the camera
        entered streaming AF in an already-cropped state (e.g. from
        a pre-2026-05-08 UNAVAILABLE leak that never restored, a
        manual MM Property Browser action, or a hypothetical bug in
        another code path), every "restore" preserved the entry
        crop instead of going back to full sensor. The Live Viewer
        was then stuck on a cropped view indefinitely, and downstream
        workflows acquired against the cropped frame -- a bug that
        cost ~4 hours of acquisition on PPM on 2026-05-10.

        The fix: always ``clear_roi()`` first to reset GenAPI Width
        and Height Max nodes to the absolute sensor maximum, then
        crop relative to THAT known-absolute baseline. The contract
        is now: streaming AF temporarily crops to the center 50% of
        the full sensor and unconditionally returns to full sensor
        on every exit path, regardless of entry state.

        Operator-set custom MM ROI (set via Property Browser before
        Live Viewer) is intentionally not preserved -- the workflow-
        level ``QPScopeChecks.validateCameraRoi`` gate
        (qupath-extension-qpsc commit ``a7dce28``) catches custom
        ROIs at workflow start and prompts the operator to clear
        them. See the TROUBLESHOOTING entry for the rationale.

    JAI / GenAPI cameras lock the Width and Height properties as
    "not writable" while a sequence acquisition is running. So the
    only way to install a new ROI is:

      1. Stop the sequence
      2. clear_roi() + set_roi() the centered crop
      3. Restart the sequence (with the new ROI in effect)

    This costs ~150 ms of camera warmup vs. the unstop-able path,
    but the per-frame transfer savings dwarf that overhead -- a
    50% crop is 4x fewer pixels per frame, dropping per-pop time
    from ~150 ms to ~40 ms on the JAI. For a 20-sample scan that
    saves ~2 seconds, well over the 300 ms warmup penalty.

    crop_factor=1.0 (no crop) is a no-op that returns (None, False)
    with no camera state changes.

    Adapted from: Micro-Manager's OughtaFocus.java ``cropFactor``
    parameter and the surrounding save/restore pattern. See the
    "Attribution / Prior art" section in the module docstring for
    the full citation.
    """
    # Always log the entry ROI -- diagnostic smoking gun if the
    # full-sensor anchoring is ever defeated by a future regression.
    entry_roi = _read_roi(core)
    if entry_roi is not None:
        logger.info(
            "STREAM_AF:entry camera ROI = (%d, %d, %dx%d)",
            entry_roi[0],
            entry_roi[1],
            entry_roi[2],
            entry_roi[3],
        )

    if crop_factor <= 0.0 or crop_factor >= 1.0:
        return (None, False)

    # JAI / GenAPI requires the sequence to be stopped before ROI
    # changes. Stop first.
    seq_running = False
    try:
        seq_running = bool(core.is_sequence_running())
    except Exception:
        pass

    if seq_running:
        try:
            core.stop_sequence_acquisition()
        except Exception as e:
            logger.warning("STREAM_AF:could not stop sequence for ROI crop: %s", e)
            return (None, False)

    # Clear to full sensor first -- this is the absolute anchor.
    # If clear_roi() raises (hypothetical adapter that doesn't
    # support it), fall back to the legacy relative-crop behavior
    # by treating the pre-clear ROI as the baseline. That's strictly
    # not worse than the pre-2026-05-11 code.
    cleared_ok = False
    try:
        core.clear_roi()
        cleared_ok = True
    except Exception as e:
        logger.warning("STREAM_AF:clear_roi() failed (%s); falling back to relative crop", e)

    if cleared_ok:
        full = _read_roi(core)
        if full is None:
            logger.warning(
                "STREAM_AF:could not query camera ROI after clear_roi; "
                "skipping crop and leaving camera at full sensor"
            )
            # Restart sequence (if we stopped one) at full sensor.
            if seq_running:
                try:
                    core.clear_circular_buffer()
                    core.start_continuous_sequence_acquisition(0)
                    time.sleep(0.15)
                except Exception:
                    pass
            return (None, seq_running)
        x0, y0, w0, h0 = full
    else:
        # clear_roi failed -- fall back to current ROI as baseline.
        # This is the pre-2026-05-11 behavior and inherits its
        # known weakness (stuck-crop preservation), but it's the
        # safest we can do without a working clear path.
        baseline = entry_roi if entry_roi is not None else _read_roi(core)
        if baseline is None:
            logger.warning("STREAM_AF:no ROI baseline available; skipping crop")
            if seq_running:
                try:
                    core.start_continuous_sequence_acquisition(0)
                except Exception:
                    pass
            return (None, seq_running)
        x0, y0, w0, h0 = baseline

    new_w = max(1, int(round(w0 * crop_factor)))
    new_h = max(1, int(round(h0 * crop_factor)))
    new_x = x0 + (w0 - new_w) // 2
    new_y = y0 + (h0 - new_h) // 2

    try:
        core.set_roi(new_x, new_y, new_w, new_h)
    except Exception as e:
        logger.warning(
            "STREAM_AF:could not install centered crop ROI " "(%d, %d, %d, %d): %s",
            new_x,
            new_y,
            new_w,
            new_h,
            e,
        )
        # Try to restart the sequence we stopped before bailing.
        # Camera is at full sensor (post clear_roi); _restore_roi
        # called with saved_roi=full is a no-op so signal that with
        # None and stash the full ROI in our return for diagnostic
        # purposes (caller doesn't use it).
        if seq_running:
            try:
                core.clear_circular_buffer()
                core.start_continuous_sequence_acquisition(0)
                time.sleep(0.15)
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
            logger.warning("STREAM_AF:could not restart sequence after " "ROI crop: %s", e)
            # Best-effort restore to full sensor and bail.
            try:
                core.clear_roi()
            except Exception:
                pass
            return (None, seq_running)

    logger.info(
        "STREAM_AF:cropped camera ROI (%d, %d, %dx%d) -> (%d, %d, %dx%d) "
        "(factor=%.2f, pixel area %.0f%% of full sensor)",
        x0,
        y0,
        w0,
        h0,
        new_x,
        new_y,
        new_w,
        new_h,
        crop_factor,
        (crop_factor * crop_factor) * 100.0,
    )
    return ((x0, y0, w0, h0), seq_running)


def _restore_roi(
    core,
    saved_roi: Optional[Tuple[int, int, int, int]],
    sequence_was_running: bool,
) -> None:
    """Unconditionally return the camera to full sensor.

    The symmetric inverse of the 2026-05-11 _apply_crop_roi: stops the
    sequence (if one is currently running), clears the ROI to full
    sensor, optionally re-applies a non-full saved_roi (currently
    unused -- _apply_crop_roi always anchors on full sensor, so
    saved_roi here is always either None or the full-sensor extent),
    and restarts.

    Always clearing first (rather than only on set_roi failure as the
    pre-2026-05-11 code did) ensures the camera ends at full sensor
    regardless of the entry state of the AF run. This is the
    proactive complement to the qupath-extension-qpsc workflow gate
    `QPScopeChecks.validateCameraRoi` (commit ``a7dce28``): the gate
    refuses workflows if the camera is cropped at start, this fix
    makes that condition unreachable from the streaming-AF code path.
    """
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

    # Primary path: clear_roi() resets to full sensor and resets the
    # GenAPI Width/Height Max nodes to their absolute maximum. This
    # is the new normal exit -- the camera is at full sensor after
    # this call regardless of where it was on entry.
    cleared_ok = False
    try:
        core.clear_roi()
        cleared_ok = True
    except Exception as e:
        logger.warning(
            "STREAM_AF:clear_roi() failed during restore (%s); falling back to set_roi", e
        )

    if cleared_ok:
        # Verify we landed at the expected dimensions (diagnostic only;
        # do not abort on mismatch since the camera is in *some* known
        # state regardless).
        post = _read_roi(core)
        if post is not None:
            logger.info(
                "STREAM_AF:restored camera ROI to (%d, %d, %dx%d) [full sensor]",
                post[0],
                post[1],
                post[2],
                post[3],
            )
        # If saved_roi was a non-full extent (currently unused -- see
        # docstring), re-apply it. Today this branch never fires
        # because _apply_crop_roi always stores the full-sensor ROI.
        if saved_roi is not None and post is not None and tuple(saved_roi) != tuple(post):
            try:
                core.set_roi(
                    int(saved_roi[0]), int(saved_roi[1]), int(saved_roi[2]), int(saved_roi[3])
                )
                logger.info(
                    "STREAM_AF:re-applied non-full saved ROI (%d, %d, %dx%d)",
                    saved_roi[0],
                    saved_roi[1],
                    saved_roi[2],
                    saved_roi[3],
                )
            except Exception as e:
                logger.warning(
                    "STREAM_AF:could not re-apply non-full saved ROI (%s); "
                    "camera left at full sensor",
                    e,
                )
    else:
        # Fallback: clear_roi failed (hypothetical adapter that
        # doesn't support it). Best-effort set_roi to the saved
        # extent. This is the pre-2026-05-11 behavior and inherits
        # its known weakness (stuck crops not recovered), but it's
        # the safest action when clear_roi is unavailable.
        if saved_roi is not None:
            try:
                core.set_roi(
                    int(saved_roi[0]), int(saved_roi[1]), int(saved_roi[2]), int(saved_roi[3])
                )
                logger.info(
                    "STREAM_AF:restored camera ROI to (%d, %d, %dx%d) "
                    "via set_roi fallback (clear_roi unavailable)",
                    saved_roi[0],
                    saved_roi[1],
                    saved_roi[2],
                    saved_roi[3],
                )
            except Exception as e2:
                logger.warning(
                    "STREAM_AF:failed to restore camera ROI via " "set_roi fallback either: %s",
                    e2,
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
            # Wait briefly for the first frame at the restored ROI to
            # arrive in the circular buffer. Without this, the next
            # get_live_frame call can pull a stale frame whose pixel
            # count doesn't match the restored dimensions, causing a
            # reshape crash (observed on PPM 2026-04-16).
            time.sleep(0.15)
        except Exception as e:
            logger.warning("STREAM_AF:could not restart sequence after " "ROI restore: %s", e)


def _try_get(core, device: str, prop: str) -> Optional[str]:
    try:
        return core.get_property(device, prop)
    except Exception:
        return None


def _wait_via_busy(
    core,
    device: str,
    timeout_s: float = 10.0,
    target_z: float | None = None,
    tolerance_um: float = 0.5,
) -> None:
    """Tight busy-poll wait for the focus device. Mirrors
    microscope_control.hardware.stage._wait_z_via_busy:
      - 5 consecutive not-busy reads required (was 2; bumped 2026-05-05
        to be more resistant to firmware reporting transient idle states
        before the trajectory has settled).
      - Optional ``target_z`` arrival verification with WARNING log when
        the stage reports not-busy but is far from the commanded Z.
        Catches the "stage control silently broken" failure mode the
        user flagged on 2026-05-05 with sweep + streaming AF.
      - Falls back to core.wait_for_device on exception, timeout, or
        arrival-check failure.
    """
    try:
        deadline = time.perf_counter() + timeout_s
        clear = 0
        clear_required = 5
        while time.perf_counter() < deadline:
            try:
                if not core.device_busy(device):
                    clear += 1
                    if clear >= clear_required:
                        break
                else:
                    clear = 0
            except Exception:
                break
            time.sleep(0.003)
        else:
            try:
                core.wait_for_device(device)
            except Exception:
                pass
    except Exception:
        try:
            core.wait_for_device(device)
        except Exception:
            pass

    # Arrival verification when target_z is known.
    if target_z is not None:
        try:
            actual_z = core.get_position()
            err = abs(actual_z - target_z)
            if err > tolerance_um:
                logger.warning(
                    "streaming wait_via_busy arrival FAILED: target=%.3f um, "
                    "actual=%.3f um, err=%.3f um (tol=%.3f um) on '%s'. "
                    "Stage reported not-busy but did not arrive. Falling "
                    "back to wait_for_device.",
                    target_z,
                    actual_z,
                    err,
                    tolerance_um,
                    device,
                )
                try:
                    core.wait_for_device(device)
                except Exception as e:
                    logger.warning("wait_for_device fallback failed: %s", e)
                try:
                    actual_z2 = core.get_position()
                    err2 = abs(actual_z2 - target_z)
                    if err2 > tolerance_um:
                        logger.error(
                            "streaming wait_via_busy STILL off-target: "
                            "target=%.3f, actual=%.3f, err=%.3f um. "
                            "Stage controller may be in a bad state.",
                            target_z,
                            actual_z2,
                            err2,
                        )
                except Exception:
                    pass
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
    z_center: float,
    range_um: float,
    z_low: Optional[float],
    z_high: Optional[float],
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


def _gaussian_peak(zs: List[float], ms: List[float]) -> Optional[Tuple[float, float, float]]:
    """Fit a Gaussian A*exp(-(z-mu)^2 / 2 sigma^2) + C to all samples
    and return (mu, r_squared, sigma) where mu is the peak location,
    r_squared is the coefficient of determination of the fit (0..1,
    1 = perfect), and sigma is the fitted Gaussian width in microns.
    Returns None on any failure (insufficient samples, degenerate z
    values, fitter non-convergence, out-of-bracket mu).

    R^2 is exposed so the caller can distinguish a low-amplitude but
    well-shaped peak (real focus, weak texture) from a flat noise
    field with no recoverable peak. At 10x with low-contrast tissue
    the metric range can be 5-8% of peak yet the curve is still a
    clean Gaussian -- rejecting on amplitude alone refuses real
    focus. Sigma is also returned so the caller can sanity-check
    that the fit is actually peak-shaped (sigma << z_range) rather
    than a degenerate baseline-fit (sigma == z_range bound).

    Full-sample fit (uses all N samples) instead of a 3-point
    parabola. Uses more of the scan data than _parabolic_peak
    does -- the parabolic fit only considers the 3 samples around
    the argmax, discarding all the rest.

    Falls back to the parabolic fit path when scipy is unavailable
    or when the Gaussian doesn't converge within reasonable bounds.

    Adapted from: Micro-Manager's ZStackFocusOptimizer.java, which
    uses a full-sample Gaussian fit via its internal Fitter class.
    See the "Attribution / Prior art" section in the module
    docstring for the full citation.
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
            gaussian,
            z_arr,
            m_arr,
            p0=[A_init, mu_init, sigma_init, C_init],
            bounds=(lo_bounds, hi_bounds),
            maxfev=500,
        )
    except Exception as e:
        logger.debug("gaussian curve_fit failed: %s", e)
        return None

    A_fit = float(popt[0])
    mu_fit = float(popt[1])
    sigma_fit = float(popt[2])
    C_fit = float(popt[3])
    if not (math.isfinite(mu_fit) and math.isfinite(sigma_fit) and math.isfinite(A_fit)):
        return None
    if mu_fit < z_arr.min() or mu_fit > z_arr.max():
        return None

    # R^2 = 1 - SS_res / SS_tot. A perfect fit gives 1; pure noise
    # gives ~0 (or negative, clamped to 0). SS_tot can be ~0 for a
    # totally flat metric -- guard with a small floor.
    predicted = gaussian(z_arr, A_fit, mu_fit, sigma_fit, C_fit)
    ss_res = float(np.sum((m_arr - predicted) ** 2))
    ss_tot = float(np.sum((m_arr - m_arr.mean()) ** 2))
    if ss_tot < 1e-12:
        r_squared = 0.0
    else:
        r_squared = max(0.0, 1.0 - ss_res / ss_tot)
    return mu_fit, r_squared, sigma_fit


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
    b = (z2**2 * (m0 - m1) + z1**2 * (m2 - m0) + z0**2 * (m1 - m2)) / denom
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
    multiple streaming AF attempts.

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
    gauss_result = _gaussian_peak(zs, ms)
    fit_z: Optional[float] = None
    if gauss_result is not None:
        fit_z = gauss_result[0]
    fit_kind = "gaussian"
    if fit_z is None or fit_z < zs[0] or fit_z > zs[-1]:
        fit_z = _parabolic_peak(zs, ms)
        fit_kind = "parabolic"
    if fit_z is None:
        fit_z = zs[raw_max_idx]
        fit_kind = "raw-argmax"

    z_span = zs[-1] - zs[0]
    logger.info(
        "STREAM_AF:union-fit across %d samples from %d attempts -- "
        "interior argmax at Z=%.3f (idx %d/%d), fit=%s best_z=%.3f, span=%.2f",
        n,
        n_attempts_so_far,
        zs[raw_max_idx],
        raw_max_idx,
        n,
        fit_kind,
        fit_z,
        z_span,
    )
    return _ScanAttemptResult(
        "success",
        float(fit_z),
        n,
        float(z_span),
        f"union-fit {fit_kind} peak at Z={fit_z:.3f} from {n} samples",
    )


# ----- The scan -----


def _run_streaming_scan(
    core,
    focus_device: str,
    speed_prop: str,
    z_start: float,
    z_end: float,
    hard_deadline_s: float,
    velocity_um_s: float = 11.5,
    metric_name: str = DEFAULT_METRIC_NAME,
    dump_dir: Optional[Path] = None,
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
    pop-time-vs-capture-time bug that corrupted early streaming AF runs:
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
    # Peek-based sampling. The old FIFO pop_next_image approach
    # was fundamentally bottlenecked at ~10 fps because every pop
    # issued a ZMQ image transfer that serialized on the bridge,
    # while the camera produces at ~38 fps. We were draining only
    # the first ~10-20% of frames the camera generated during the
    # scan, so a 20 um scan that saw focus pass through the middle
    # would only sample the first ~4 um of Z -- the focus
    # transition was there, we just never retrieved the frames.
    # (Observed 00:18 PPM 10x: user *visually* saw focus pass
    # through the middle of a 20 um scan, but the metric showed
    # only 1.17% variation because sampled frames were all from
    # the first few microns of travel.)
    #
    # Fix: use core.get_last_image() -- the same non-consuming
    # PEEK that the Live Viewer already uses to display the
    # running stream. It returns whatever frame the camera
    # currently has most-recent, without removing it from the
    # circular buffer. We poll it at a fixed cadence (every
    # SCAN_POLL_SLEEP_S seconds), tagging each sample with the
    # wall time since the move fired. Z is computed directly
    # from wall_time * velocity -- no back-fill, no camera_period
    # inference.
    #
    # Duplicate detection: multiple consecutive calls to
    # get_last_image can return the same underlying frame if we
    # poll faster than the camera produces. We track the camera's
    # remaining-image count as a monotonic "new frame arrived"
    # signal -- when it increases, the latest frame is
    # guaranteed-new. When it doesn't, we skip the compute to
    # avoid wasted metric evaluations on duplicates.
    try:
        core.clear_circular_buffer()
        logger.info("STREAM_AF:flushed circular buffer before firing move")
    except Exception as e:
        logger.warning(
            "STREAM_AF:clear_circular_buffer failed " "(continuing with whatever's queued): %s", e
        )

    direction = 1.0 if z_end >= z_start else -1.0
    motion_um = abs(z_end - z_start)
    motion_duration_ms = (motion_um / max(velocity_um_s, 0.01)) * 1000.0
    scan_exit_at_ms = motion_duration_ms + SCAN_TAIL_MS

    # Cache image geometry once. get_last_image returns a flat
    # pixel buffer; we need width/height/channels to reshape.
    try:
        img_w = int(core.get_image_width())
        img_h = int(core.get_image_height())
        img_nch = int(core.get_number_of_components())
    except Exception as e:
        logger.warning("STREAM_AF:could not query image geometry: %s", e)
        img_w = img_h = 0
        img_nch = 1

    # Raw captures during the scan. Each entry is
    # (wall_ms, image_array). Metric is computed AFTER the loop
    # so we don't block the sampling cadence on per-frame CPU
    # work (normalized_variance on a 1024x772x3 image is ~5-15ms,
    # enough to skew a 20-50ms poll cadence if done inline).
    raw_captures: List[Tuple[float, np.ndarray]] = []
    # Content fingerprint of the last captured frame. We compare
    # bytes from a CENTRAL slice of get_last_image() to detect new
    # frames instead of using core.get_remaining_image_count().
    #
    # PRIOR DESIGN #1 (failed PPM 40x 2026-05-04): the loop used
    # `remaining > last_remaining` to detect new frames. That
    # works while the MM circular buffer is filling monotonically,
    # but Micro-Manager's default behaviour when the buffer
    # saturates is to OVERWRITE the oldest frame in place --
    # `remaining` plateaus at the buffer capacity instead of
    # continuing to grow. Once that happens, the new-frame check
    # never fires again and the loop captures zero frames for the
    # rest of the scan.
    #
    # PRIOR DESIGN #2 (mitigated, 2026-05-04): fingerprint sampled
    # the FIRST 32 bytes of the flat pixel buffer. For a row-major
    # H*W*C image that is the top-left corner -- exactly where the
    # JAI prism's optical vignette darkens pixels to near-zero. With
    # near-uniform dark bytes, fingerprints can collide across
    # different frames whose centres differ wildly. The current
    # design samples from FOUR disjoint regions spread across the
    # buffer (20%/40%/60%/80%) so at least one block lands inside
    # the in-tissue, well-exposed centre.
    #
    # Observed (PRIOR #1): a 4.2 s PPM scan at 38 fps captured the
    # first 39 frames over t=0-2070ms (filling the buffer), then
    # ZERO frames for the remaining 2.3 s (t=2070-4319ms). Gaussian
    # fit on the first-half-only data picked Z=86.4 and committed a
    # -1.0 um shift AWAY from focus.
    #
    # Pollrate (2 ms) is much faster than frame period (~26 ms at
    # 38 fps), so dedupe is reliable. CPU cost of slicing & comparing
    # 128 bytes is negligible.
    last_fingerprint: Optional[bytes] = None
    FINGERPRINT_BYTES = 128
    FINGERPRINT_SAMPLE_FRACTIONS = (0.2, 0.4, 0.6, 0.8)

    t0 = time.perf_counter()

    # 2026-05-06: parallel stage-Z polling thread for diagnostics.
    # Sole purpose: confirm the stage is actually moving at the
    # configured slow velocity. PPM bug observed today: caller
    # configures velocity_um_s=1.42 (slow speed), but the user sees
    # the scan complete "almost instantly" through the eyepiece,
    # meaning the stage is running at normal speed (~100 um/s) and
    # finishes the 20 um sweep in ~0.2 s. The remaining ~14 s of
    # samples are stationary frames at z_end, but the linear
    # time-to-Z map labels them with intermediate Z values, putting
    # the metric peak at the wrong end of the scan.
    #
    # Polls core.get_position() every Z_POLL_INTERVAL_S during the
    # scan and stores (wall_ms, z_actual). After the loop, compares
    # observed velocity to configured velocity_um_s and WARNs loudly
    # if they diverge, which is the diagnostic signature for
    # "speed property accepted but had no effect" or "speed property
    # is in different units than YAML thinks".
    Z_POLL_INTERVAL_S = 0.05  # 50 ms; fast enough for sub-second motion
    z_poll_samples: List[Tuple[float, float]] = []
    z_poll_stop = threading.Event()

    def _z_poll_loop():
        while not z_poll_stop.is_set():
            try:
                z = float(core.get_position(focus_device))
                t_ms = (time.perf_counter() - t0) * 1000.0
                z_poll_samples.append((t_ms, z))
            except Exception:
                pass
            if z_poll_stop.wait(Z_POLL_INTERVAL_S):
                break

    z_poll_thread = threading.Thread(target=_z_poll_loop, daemon=True)
    z_poll_thread.start()

    try:
        core.set_position(focus_device, z_end)
    except Exception as e:
        logger.error("STREAM_AF:non-blocking move to z_end failed: %s", e)
        z_poll_stop.set()
        z_poll_thread.join(timeout=0.5)
        return []

    deadline = time.perf_counter() + hard_deadline_s

    while time.perf_counter() < deadline:
        t_now_ms = (time.perf_counter() - t0) * 1000.0
        if t_now_ms > scan_exit_at_ms:
            break

        try:
            pixels = core.get_last_image()
        except Exception:
            pixels = None

        if pixels is not None:
            try:
                # Cheap fingerprint: 4 disjoint 32-byte blocks from
                # 20%/40%/60%/80% through the flat buffer. At least
                # one block lands inside the in-tissue centre, even
                # when corners and edges sit in vignette darkness
                # whose bytes don't change frame-to-frame. ~128
                # bytes out of ~2.4 MB; CPU cost negligible.
                arr = np.asarray(pixels)
                flat = arr.reshape(-1).view(np.uint8)
                n_bytes = flat.size
                block_size = FINGERPRINT_BYTES // len(FINGERPRINT_SAMPLE_FRACTIONS)
                if n_bytes >= FINGERPRINT_BYTES:
                    parts = []
                    for frac in FINGERPRINT_SAMPLE_FRACTIONS:
                        off = int(n_bytes * frac) - (block_size // 2)
                        if off < 0:
                            off = 0
                        if off + block_size > n_bytes:
                            off = n_bytes - block_size
                        parts.append(flat[off : off + block_size].tobytes())
                    fp = b"".join(parts)
                elif n_bytes > 0:
                    fp = flat.tobytes()
                else:
                    fp = None
            except Exception:
                fp = None
                arr = None
            if fp is not None and fp != last_fingerprint:
                raw_captures.append((t_now_ms, arr))
                last_fingerprint = fp

        time.sleep(SCAN_POLL_SLEEP_S)

    total_scan_ms = (time.perf_counter() - t0) * 1000.0

    # Stop the Z-polling thread and analyse the actual stage motion
    # against the configured slow velocity. See the comment block at
    # thread launch for the bug this catches. Cheap; runs once per
    # scan and only logs a few lines.
    z_poll_stop.set()
    try:
        z_poll_thread.join(timeout=0.5)
    except Exception:
        pass

    if len(z_poll_samples) >= 2:
        first_t, first_z = z_poll_samples[0]
        last_t, last_z = z_poll_samples[-1]
        observed_total_um = abs(last_z - first_z)
        observed_dur_s = max((last_t - first_t) / 1000.0, 1e-3)
        observed_avg_velocity_um_s = observed_total_um / observed_dur_s

        # Find when the stage first reached z_end (within 0.25 um). If
        # this happens long before scan exit, the configured slow speed
        # didn't take effect and the rest of the scan was stationary.
        Z_END_REACHED_TOLERANCE_UM = 0.25
        t_reached_z_end_ms: Optional[float] = None
        for t_ms, z in z_poll_samples:
            if abs(z - z_end) <= Z_END_REACHED_TOLERANCE_UM:
                t_reached_z_end_ms = t_ms
                break

        # Sample the start to confirm we actually started near z_start.
        z_start_actual = first_z

        logger.info(
            "STREAM_AF:Z-poll trace: n=%d, first(t=%.0fms, Z=%.3f), "
            "last(t=%.0fms, Z=%.3f), observed_avg_velocity=%.2f um/s "
            "(configured %.2f um/s), reached_z_end at t=%s",
            len(z_poll_samples),
            first_t,
            z_start_actual,
            last_t,
            last_z,
            observed_avg_velocity_um_s,
            velocity_um_s,
            (
                f"{t_reached_z_end_ms:.0f}ms"
                if t_reached_z_end_ms is not None
                else "(not reached during poll window)"
            ),
        )

        # Smoking-gun warning. Two signals must agree before firing:
        # (a) the Z-poll trace shows the stage *appeared* to reach
        # z_end in less than half the expected time, AND
        # (b) the average velocity across the full poll window is
        # also at least 2x the configured slow speed.
        #
        # Both signals are required because some stage adapters
        # occasionally return the commanded destination Z from
        # get_position() during the move, which produces a single
        # spurious sample at z_end early in the trace. In that case
        # the avg velocity over the whole trace still matches
        # configured -- the early reached_z_end timestamp is noise,
        # not a real fast-move. (Observed on the ASI ZDrive 2026-05-13:
        # reached_z_end at t=306ms but avg_velocity = 6.03 um/s vs
        # configured 6.08 um/s, with a clean Pearson r=-0.927 metric
        # slope confirming the stage really did move slowly.)
        reached_z_end_early = (
            t_reached_z_end_ms is not None and t_reached_z_end_ms < motion_duration_ms * 0.5
        )
        avg_velocity_high = observed_avg_velocity_um_s > velocity_um_s * 2.0
        if reached_z_end_early and not avg_velocity_high:
            logger.debug(
                "STREAM_AF:Z-poll glitch suppressed -- reached_z_end at "
                "t=%.0fms looks early vs motion_duration_ms=%.0f, but "
                "observed_avg_velocity=%.2f um/s matches configured "
                "%.2f um/s. Likely a single spurious Z reading; ignoring.",
                t_reached_z_end_ms,
                motion_duration_ms,
                observed_avg_velocity_um_s,
                velocity_um_s,
            )
        if reached_z_end_early and avg_velocity_high:
            # Compute the in-motion velocity (during the actual move
            # only, NOT averaged over the full poll window). This is
            # the number you actually want to put into
            # slow_speed_um_per_s YAML, since the average-over-window
            # number gets diluted by stationary post-motion samples.
            in_motion_velocity_um_s = abs(z_end - z_start) / max(t_reached_z_end_ms / 1000.0, 1e-3)
            logger.warning(
                "STREAM_AF:STAGE SPEED MISMATCH -- expected slow scan to "
                "take ~%.0fms (velocity_um_s=%.2f, range=%.2fum), but "
                "stage reached z_end in only %.0fms. Actual in-motion "
                "velocity = %.2f um/s. Slow-speed property is not slowing "
                "the stage enough; the scan was %.0f%% stationary frames "
                "post-motion, so only ~%.0f real in-motion samples were "
                "captured. To find focus reliably we need the stage to "
                "move at ~1-3 um/s for ~14s (giving ~70-150 in-motion "
                "samples). Two options: (1) update YAML "
                "stage.streaming_af.slow_speed_um_per_s to %.2f so the "
                "loop duration matches reality (algorithm self-corrects "
                "via Z-poll labelling but loop still over-samples post-"
                "motion); (2) find a stage property setting that "
                "actually slows the move -- try setting Acceleration "
                "and SCurve properties to low values on the focus "
                "device, or use a fractional MaxSpeed value if the "
                "adapter accepts it.",
                motion_duration_ms,
                velocity_um_s,
                abs(z_end - z_start),
                t_reached_z_end_ms,
                in_motion_velocity_um_s,
                100.0 * (1.0 - t_reached_z_end_ms / max(observed_dur_s * 1000.0, 1e-3)),
                max(1, int(round(t_reached_z_end_ms / 1000.0 * 38.0))),
                in_motion_velocity_um_s,
            )

            # One-shot diagnostic: dump allowed values for every speed-
            # related property the focus device exposes so the operator
            # can see which knob to try next. Logged at WARNING because
            # if the user is looking at the speed-mismatch warning,
            # they're already in the middle of fixing this and want
            # the data right next to the warning.
            for prop_name in (
                "MaxSpeed",
                "Velocity",
                "Speed",
                "MaxVelocity",
                "Acceleration",
                "SCurve",
            ):
                try:
                    cur = core.get_property(focus_device, prop_name)
                except Exception:
                    continue  # property doesn't exist
                allowed = "(no enum)"
                try:
                    raw = core.get_allowed_property_values(focus_device, prop_name)
                    allowed_list = _str_vector_to_list(raw) if raw else None
                    if allowed_list:
                        allowed = ", ".join(allowed_list[:20]) + (
                            "..." if len(allowed_list) > 20 else ""
                        )
                except Exception:
                    pass
                # Probe whether fractional values are accepted (vendor
                # adapters often accept arbitrary numbers in a numeric
                # range even when the GUI shows just "1" through "100").
                accepts_fractional = "?"
                try:
                    saved_val = cur
                    test_val = "0.5"
                    core.set_property(focus_device, prop_name, test_val)
                    after = core.get_property(focus_device, prop_name)
                    accepts_fractional = "yes" if after == test_val else f"no (clamped to {after})"
                    core.set_property(focus_device, prop_name, saved_val)
                except Exception:
                    accepts_fractional = "no (rejected)"

                # Probe a range of integer values so we can see the
                # actual accepted range without relying on the device
                # adapter's enum metadata (which Prior ProScan does
                # not expose). For each candidate we try set_property
                # and read back; values that round-trip identically
                # are accepted, values that get clamped land at the
                # boundary, and values that throw are rejected.
                int_probe_values = (
                    "0",
                    "1",
                    "2",
                    "3",
                    "5",
                    "10",
                    "20",
                    "50",
                    "100",
                    "200",
                    "500",
                    "1000",
                )
                int_results: list = []
                for tv in int_probe_values:
                    try:
                        core.set_property(focus_device, prop_name, tv)
                        after = core.get_property(focus_device, prop_name)
                        if after == tv:
                            int_results.append(f"{tv}")
                        else:
                            int_results.append(f"{tv}->({after})")
                    except Exception:
                        int_results.append(f"{tv}!")
                # Always restore original
                try:
                    core.set_property(focus_device, prop_name, cur)
                except Exception:
                    pass
                logger.warning(
                    "STREAM_AF:property survey -- %s.%s = %r (allowed=[%s], "
                    "accepts 0.5? %s, integer probes: %s) "
                    "[N=accepted, N->(M)=clamped to M, N!=rejected]",
                    focus_device,
                    prop_name,
                    cur,
                    allowed,
                    accepts_fractional,
                    ", ".join(int_results),
                )
        elif observed_avg_velocity_um_s > velocity_um_s * 3.0:
            logger.warning(
                "STREAM_AF:STAGE SPEED MISMATCH -- observed avg velocity "
                "%.2f um/s > 3x configured %.2f um/s. Slow-speed property "
                "may not be in expected units; verify stage YAML.",
                observed_avg_velocity_um_s,
                velocity_um_s,
            )
    else:
        logger.debug(
            "STREAM_AF:Z-poll trace: only %d samples collected; " "skipping velocity analysis.",
            len(z_poll_samples),
        )

    # --- Post-scan: reshape + metric computation ---
    samples: List[Tuple[float, float, float]] = []
    # Parallel list: per-accepted-sample (wall_ms, image_2D_or_3D, metric)
    # only populated when dump_dir is set, since each entry holds a full
    # frame and we don't want to keep them in the hot path.
    dump_records: Optional[List[Tuple[float, np.ndarray, float, float]]] = (
        [] if dump_dir is not None else None
    )
    for wall_ms, arr in raw_captures:
        try:
            if img_nch <= 1:
                img = arr.reshape(img_h, img_w)
            else:
                img = arr.reshape(img_h, img_w, img_nch)
        except Exception:
            img = arr
        try:
            metric = _focus_metric(img, metric_name)
        except Exception as e:
            logger.debug("STREAM_AF:metric compute failed: %s", e)
            continue

        # Z from wall time * velocity. This is now directly
        # accurate -- no camera_period back-fill, no inference.
        if wall_ms <= 0:
            z_interp = z_start
        elif wall_ms >= motion_duration_ms:
            z_interp = z_end
        else:
            progress_um = (wall_ms / 1000.0) * velocity_um_s * direction
            z_interp = z_start + progress_um
        samples.append((wall_ms, float(z_interp), metric))
        if dump_records is not None:
            dump_records.append((wall_ms, img, float(z_interp), metric))

    logger.info(
        "STREAM_AF:scan exit at t=%.0fms (motion_end=%.0fms + tail=%.0fms) "
        "captures=%d samples=%d",
        total_scan_ms,
        motion_duration_ms,
        SCAN_TAIL_MS,
        len(raw_captures),
        len(samples),
    )

    if dump_dir is not None and dump_records is not None:
        try:
            _dump_streaming_scan(
                dump_dir=dump_dir,
                dump_records=dump_records,
                z_poll_samples=z_poll_samples,
                z_start=z_start,
                z_end=z_end,
                velocity_um_s=velocity_um_s,
                motion_duration_ms=motion_duration_ms,
                metric_name=metric_name,
            )
        except Exception as e:
            logger.warning(
                "STREAM_AF:dump_streaming_scan failed (non-fatal): %s",
                e,
            )

    return samples


def _dump_streaming_scan(
    dump_dir: Path,
    dump_records: List[Tuple[float, np.ndarray, float, float]],
    z_poll_samples: List[Tuple[float, float]],
    z_start: float,
    z_end: float,
    velocity_um_s: float,
    motion_duration_ms: float,
    metric_name: str,
) -> None:
    """Write per-sample TIFs + a CSV trace + a manifest into dump_dir.

    Layout:
        dump_dir/
          frames/frame_NNNN_t<ms>_zassumed<um>.tif    -- one per kept sample
          samples.csv                                  -- (idx, wall_ms, z_assumed_um, z_actual_um, metric)
          z_poll.csv                                   -- (wall_ms, z_actual_um) raw poll
          manifest.json                                -- scan params

    z_actual is the polled stage Z linearly interpolated to each sample's
    wall_ms timestamp from the z_poll_samples trace, so the CSV directly
    shows the time->space mapping the algorithm assumed vs. what the
    stage actually did.
    """
    import csv
    import json

    try:
        import tifffile
    except Exception as e:
        logger.warning("STREAM_AF:dump skipped, tifffile unavailable: %s", e)
        return

    dump_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = dump_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build the sorted z_poll arrays for O(log n) interpolation.
    poll_t = np.asarray([t for (t, _) in z_poll_samples], dtype=np.float64)
    poll_z = np.asarray([z for (_, z) in z_poll_samples], dtype=np.float64)

    def _z_actual_at(wall_ms: float) -> Optional[float]:
        if poll_t.size == 0:
            return None
        if wall_ms <= poll_t[0]:
            return float(poll_z[0])
        if wall_ms >= poll_t[-1]:
            return float(poll_z[-1])
        return float(np.interp(wall_ms, poll_t, poll_z))

    samples_csv_path = dump_dir / "samples.csv"
    with open(samples_csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "wall_ms", "z_assumed_um", "z_actual_um", "metric"])
        for idx, (wall_ms, img, z_assumed, metric) in enumerate(dump_records):
            z_actual = _z_actual_at(wall_ms)
            w.writerow(
                [
                    idx,
                    f"{wall_ms:.2f}",
                    f"{z_assumed:.4f}",
                    "" if z_actual is None else f"{z_actual:.4f}",
                    f"{metric:.6f}",
                ]
            )
            tif_name = (
                f"frame_{idx:04d}_t{int(round(wall_ms)):06d}ms_" f"zass{z_assumed:+09.3f}.tif"
            ).replace(" ", "0")
            try:
                tifffile.imwrite(
                    str(frames_dir / tif_name),
                    img,
                    photometric="minisblack" if img.ndim == 2 else "rgb",
                )
            except Exception as e:
                logger.debug(
                    "STREAM_AF:dump frame %d write failed: %s",
                    idx,
                    e,
                )

    z_poll_csv_path = dump_dir / "z_poll.csv"
    with open(z_poll_csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wall_ms", "z_actual_um"])
        for t_ms, z in z_poll_samples:
            w.writerow([f"{t_ms:.2f}", f"{z:.4f}"])

    manifest = {
        "z_start": z_start,
        "z_end": z_end,
        "velocity_um_s_configured": velocity_um_s,
        "motion_duration_ms": motion_duration_ms,
        "metric_name": metric_name,
        "n_kept_samples": len(dump_records),
        "n_z_poll_samples": len(z_poll_samples),
    }
    if z_poll_samples:
        first_t, first_z = z_poll_samples[0]
        last_t, last_z = z_poll_samples[-1]
        observed_dur_s = max((last_t - first_t) / 1000.0, 1e-3)
        manifest["z_poll_first_t_ms"] = first_t
        manifest["z_poll_first_z"] = first_z
        manifest["z_poll_last_t_ms"] = last_t
        manifest["z_poll_last_z"] = last_z
        manifest["observed_avg_velocity_um_s"] = abs(last_z - first_z) / observed_dur_s
    with open(dump_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(
        "STREAM_AF:dump written: %d frames + samples.csv + z_poll.csv " "+ manifest.json under %s",
        len(dump_records),
        dump_dir,
    )


# ----- Handler entry point -----


class _ScanAttemptResult:
    """Result of one _attempt_one_scan call.

    status is one of:
        'success'             -- peak found, best_z set
        'edge_low'            -- argmax at first usable sample; shift down
        'edge_high'           -- argmax at last usable sample; shift up
        'insufficient_samples' -- not enough samples for a fit
        'metric_flat'         -- metric variation across the scan is
                                 within noise (scan window inside one
                                 depth-of-field); retrying with the
                                 same range will not help
        'no_slow_speed'       -- the stage will not accept a slow speed
                                 for streaming (property absent or value
                                 rejected); caller should route this
                                 acquisition to the Brent fallback
        'error'               -- hardware or protocol error mid-scan
    """

    def __init__(
        self,
        status: str,
        best_z: Optional[float],
        n_samples: int,
        z_span: float,
        reason: str,
        samples_trace: Optional[list] = None,
    ):
        self.status = status
        self.best_z = best_z
        self.n_samples = n_samples
        self.z_span = z_span
        self.reason = reason
        self.samples_trace = samples_trace or []


def _attempt_one_scan(
    core,
    focus_device: str,
    speed_prop: Optional[str],
    z_center: float,
    range_um: float,
    sequence_was_running_on_entry: bool,
    attempt_label: str = "",
    velocity_um_s: float = MIN_VELOCITY_UM_S,
    metric_name: str = "normalized_variance",
    slow_value: str = SLOW_SPEED_VALUE,
    normal_value: str = NORMAL_SPEED_VALUE,
    dump_dir: Optional[Path] = None,
) -> _ScanAttemptResult:
    """Run one streaming AF scan centered on z_center with the given range.

    Returns an _ScanAttemptResult describing the outcome. Does NOT
    commit the peak (caller decides whether to retry or commit) and
    does NOT restore the stage Z (caller handles cleanup).

    The `attempt_label` is prepended to log lines so multi-attempt
    runs are easy to follow (e.g. 'attempt 2/3: ').

    `speed_prop` may be None when the focus device has no writable
    speed-like property -- in that case streaming is not feasible and
    this returns 'no_slow_speed' immediately so the caller can route
    to Brent's snap-and-stop fallback.

    Args:
        velocity_um_s: expected slow-speed stage velocity; used by
            _run_streaming_scan to interpolate Z at frame capture time.
        metric_name: which focus metric to compute per frame.
        slow_value: value to set on speed_prop during the slow scan.
            Defaults to SLOW_SPEED_VALUE; per-rig override comes
            from stage.streaming_af.slow_speed_value YAML.
        normal_value: value to restore for positioning moves and
            after the scan. Defaults to NORMAL_SPEED_VALUE; per-rig
            override comes from stage.streaming_af.normal_speed_value.
    """
    tag_prefix = f"{attempt_label}: " if attempt_label else ""
    z_start = z_center - range_um / 2.0
    z_end = z_center + range_um / 2.0
    logger.info(
        "STREAM_AF:%sscan window [%.3f -> %.3f] (center %.3f, range %.2f)",
        tag_prefix,
        z_start,
        z_end,
        z_center,
        range_um,
    )

    if speed_prop is None:
        return _ScanAttemptResult(
            "no_slow_speed",
            None,
            0,
            0.0,
            f"focus device '{focus_device}' has no writable speed property",
        )

    try:
        # Positioning seed at full speed.
        _try_set(core, focus_device, speed_prop, normal_value)
        core.set_position(focus_device, z_start)
        _wait_via_busy(core, focus_device, target_z=z_start)

        # Drop to slow speed for the scan motion only.
        if not _try_set(core, focus_device, speed_prop, slow_value):
            # Diagnostic: log the property's allowed values so we can
            # tell from a log alone what slow value this stage would
            # accept. Different vendors use different units / scales
            # (Prior: 1-100 percent; ASI: um/s; some adapters: enum).
            try:
                allowed = _str_vector_to_list(
                    core.get_allowed_property_values(focus_device, speed_prop)
                )
                logger.debug(
                    "STREAM_AF:%sslow-speed set rejected; %s.%s allowed values = %s",
                    tag_prefix,
                    focus_device,
                    speed_prop,
                    allowed or "(none reported)",
                )
            except Exception:
                pass
            return _ScanAttemptResult(
                "no_slow_speed",
                None,
                0,
                0.0,
                f"stage rejected {speed_prop}={slow_value}",
            )

        sequence_started_here = False
        if sequence_was_running_on_entry:
            logger.info("STREAM_AF:%sreusing already-running sequence", tag_prefix)
        else:
            logger.info("STREAM_AF:%sno active sequence; starting one for the scan", tag_prefix)
            core.clear_circular_buffer()
            core.start_continuous_sequence_acquisition(0)
            sequence_started_here = True
            time.sleep(0.15)

        try:
            # Hard deadline must comfortably exceed motion_duration_ms or
            # the loop exits before the stage reaches z_end. The original
            # formula (range_um * 0.15 + 2.0s) was tuned for a fast
            # stage (~11.5 um/s SLOW_SPEED) and trips early at PPM speeds
            # (1.42 um/s, motion_duration ~4.2s for a 6 um scan, deadline
            # 2.9s -> scan truncated to ~70% of the planned range).
            # Floor the deadline at motion_duration_ms + 2s plus the old
            # multiplier as a fallback.
            motion_duration_s = abs(z_end - z_start) / max(velocity_um_s, 0.01)
            hard_deadline_s = max(
                1.0,
                range_um * HARD_DEADLINE_SEC_PER_UM + 2.0,
                motion_duration_s + 2.0,
            )
            samples = _run_streaming_scan(
                core,
                focus_device,
                speed_prop,
                z_start,
                z_end,
                hard_deadline_s,
                velocity_um_s=velocity_um_s,
                metric_name=metric_name,
                dump_dir=dump_dir,
            )
        finally:
            if sequence_started_here:
                try:
                    core.stop_sequence_acquisition()
                except Exception:
                    pass
                try:
                    core.clear_circular_buffer()
                except Exception:
                    pass

        # Diagnostic: read actual stage position immediately after the
        # scan loop returns. Compare to expected z_end. If the actual Z
        # is far from z_end, the stage either never accepted the slow
        # speed (so it ran much faster, finished early, and the "scan"
        # was static frames at z_end) OR the slow speed value is mis-
        # calibrated (so the scan didn't traverse the planned range).
        # Either case explains a flat-metric refusal even when the scan
        # window is well within DOF.
        try:
            actual_z_after_scan = float(core.get_position(focus_device))
            expected_motion_um = abs(z_end - z_start)
            actual_motion_um = abs(actual_z_after_scan - z_start)
            motion_ratio = actual_motion_um / max(expected_motion_um, 1e-3)
            logger.info(
                "STREAM_AF:%spost-scan stage Z=%.3f (expected z_end=%.3f); "
                "achieved %.2f um of planned %.2f um = %.0f%% (motion_duration_s=%.2f, "
                "velocity_um_s=%.2f, hard_deadline_s=%.2f)",
                tag_prefix,
                actual_z_after_scan,
                z_end,
                actual_motion_um,
                expected_motion_um,
                motion_ratio * 100.0,
                motion_duration_s,
                velocity_um_s,
                hard_deadline_s,
            )
            if motion_ratio < 0.5 or motion_ratio > 1.5:
                logger.warning(
                    "STREAM_AF:%sstage motion mismatch: achieved %.0f%% of planned "
                    "range. Slow-speed value '%s' on %s.%s may be misconfigured "
                    "for this rig. Verify stage.streaming_af.slow_speed_value and "
                    "stage.streaming_af.slow_speed_um_per_s in YAML.",
                    tag_prefix,
                    motion_ratio * 100.0,
                    slow_value,
                    focus_device,
                    speed_prop,
                )
        except Exception as e:
            logger.debug(
                "STREAM_AF:%scould not read post-scan stage Z: %s",
                tag_prefix,
                e,
            )

        _try_set(core, focus_device, speed_prop, normal_value)

        # --- Sample filtering and fit ---
        # Truncation: time-based ONLY. wall_ms is authoritative;
        # interpolated Z = z_start + wall_ms * velocity_um_s is a
        # linear function of wall_ms by construction. Anything past
        # motion_end_ms is post-motion tail (stage at z_end with
        # static frames) -- discard. A small grace window
        # (POST_MOTION_GRACE_MS) keeps samples whose timestamps land
        # just after motion_end due to poll jitter.
        #
        # PRIOR HISTORY: a "Z-stagnation safety net" lived here to
        # detect motor stalls. It was deleted on 2026-05-04 after a
        # PPM session showed it false-tripping on every scan: because
        # z_interp is derived from wall_ms via a constant velocity,
        # the ratio z_span_w / t_span_w is identically velocity_um_s
        # for any window. The check (z_span < STALL_Z_UM AND t_span >
        # STALL_MIN_DT_MS) reduced to "5 samples span 50..352 ms" --
        # which fires on normal frame timing, not on stalls. Real
        # stall detection would require querying actual stage Z
        # during the scan, not interpolating from time. With the
        # interpolated-Z model, time-based cutoff already handles the
        # only failure mode worth catching: stage finishes early and
        # subsequent frames are at z_end. The post-motion grace
        # window does the right thing in that case (samples after
        # motion_end_ms get dropped, leaving the in-motion samples).
        POST_MOTION_GRACE_MS = 100.0
        motion_end_ms = abs(z_end - z_start) / max(velocity_um_s, 0.01) * 1000.0
        time_cutoff_ms = motion_end_ms + POST_MOTION_GRACE_MS

        # PRE-MOTION HEAD DISCARD
        # Stage acceleration ramp + stale-buffer frames captured before
        # constant-velocity motion is reached produce samples whose
        # interpolated Z (z_start + wall_ms * velocity_um_s) does NOT
        # match the actual stage position. On PPM 40x 2026-05-04 the
        # first ~290ms of every scan showed a metric "peak" 50%+ above
        # the rest of the scan baseline, the gaussian fit latched onto
        # it, and the AF moved the stage 2-3 um AWAY from focus.
        #
        # 2026-05-04 follow-up: 300ms wasn't enough. After the buffer-
        # saturation fix admitted the full 4 s scan, we still saw 3-5%
        # elevated metric at samples just past the 300ms cutoff
        # (t=484-565 ms). With the rest of the scan flat at <1%, those
        # 2-3 head samples biased the gaussian fit toward the LOW-Z
        # edge of the scan window: Z_truth=90, fit committed Z=89.3.
        # 600ms cleanly clears the contamination zone while still
        # leaving 3.6 s of the 4.2 s motion_duration on PPM 40x.
        #
        # Velocity-aware: the alternative head = "first 5% of
        # motion_end_ms" would shrink to ~100ms on a 2um/1s scan and
        # still admit accel artifacts; a fixed floor is safer.
        HEAD_DISCARD_MS = 600.0

        clean = [
            (t, z, m)
            for (t, z, m) in samples
            if z == z and m == m and math.isfinite(z) and math.isfinite(m)
        ]
        in_motion = [(t, z, m) for (t, z, m) in clean if HEAD_DISCARD_MS <= t <= time_cutoff_ms]
        logger.info(
            "STREAM_AF:%sin_motion filter kept %d/%d samples "
            "(head_discard=%.0fms time_cutoff=%.0fms motion_end=%.0fms)",
            tag_prefix,
            len(in_motion),
            len(clean),
            HEAD_DISCARD_MS,
            time_cutoff_ms,
            motion_end_ms,
        )

        n_motion_samples = len(in_motion)
        if n_motion_samples >= 2:
            zs = [p[1] for p in in_motion]
            ms = [p[2] for p in in_motion]
            z_span = float(max(zs) - min(zs))
            raw_peak_idx = int(np.argmax(ms))
            raw_peak_z = zs[raw_peak_idx]

            # Run the gaussian fit early so the flat-metric refusal
            # can use the fit quality (R^2 and sigma) as a sanity
            # check on low-amplitude scans.
            gaussian_fit = _gaussian_peak(zs, ms) if n_motion_samples >= 4 else None
            parabolic = _parabolic_peak(zs, ms) if n_motion_samples >= 3 else None

            # --- Flat-metric refusal ---
            # If the metric range across all samples is within noise
            # of the metric peak AND the gaussian fit is poor, there
            # is no findable peak and any "argmax" is whichever
            # sample won a coin flip. Two-condition check because at
            # 10x with low-contrast tissue the metric range can be
            # 5-8% of peak yet the curve is a clean Gaussian -- a
            # pure-amplitude refusal rejected real focus repeatedly.
            metric_peak = float(max(ms))
            metric_trough = float(min(ms))
            metric_range = metric_peak - metric_trough
            metric_range_frac = metric_range / max(abs(metric_peak), 1e-6)

            shape_ok = False
            r2 = 0.0
            sigma_fit = 0.0
            if gaussian_fit is not None:
                _mu, r2, sigma_fit = gaussian_fit
                # Peak-shaped if R^2 is high enough AND sigma is
                # narrow vs the scan window (fitted sigma at the
                # upper bound = degenerate baseline fit).
                shape_ok = r2 >= FLAT_METRIC_GAUSSIAN_R2 and sigma_fit < 0.45 * max(z_span, 1e-6)

            amplitude_trusted = metric_range_frac >= FLAT_METRIC_AMPLITUDE_TRUSTED
            amplitude_above_floor = metric_range_frac >= FLAT_METRIC_FRACTION

            if not amplitude_trusted and not (amplitude_above_floor and shape_ok):
                # 2026-05-12: Before refusing as metric_flat, check whether
                # the gaussian fit's mu landed pinned at a sample boundary
                # with a peak-shaped fit. _gaussian_peak constrains mu to
                # [z_arr.min(), z_arr.max()]; when the true peak sits
                # OUTSIDE the sampled range, curve_fit pushes mu to the
                # nearest boundary. So "shape_ok + mu within sigma of a
                # boundary + low amplitude" is the signature of a peak
                # just past the scan edge -- exactly the case the edge-
                # retry loop is designed to recover (shift the next
                # window's center by one full range in that direction).
                #
                # Repeatable test case (PPM 10x, 2026-05-12): focus at
                # Z=-29.4, scan from Z=0 with range=50 -> window [-25,
                # +25]. Head-discard eats the first ~7 um so the first
                # in-motion sample is at z=-18. Gaussian fit pins mu near
                # -18 with R^2=0.93, sigma~3 um. The legacy raw-argmax-
                # position edge check (line ~2517) misses this because
                # peak_z=-18.1 is 6.9 um from commanded z_lo=-25, outside
                # the 5 um tolerance. mu-at-boundary correctly flags it
                # as edge_low; retry shifts to center=-50, finds focus.
                if shape_ok and not amplitude_above_floor:
                    z_min_sampled = float(min(zs))
                    z_max_sampled = float(max(zs))
                    mu_fit = float(gaussian_fit[0]) if gaussian_fit is not None else 0.0
                    boundary_tol = max(sigma_fit, 0.5)
                    mu_at_low = abs(mu_fit - z_min_sampled) <= boundary_tol
                    mu_at_high = abs(mu_fit - z_max_sampled) <= boundary_tol
                    if mu_at_low != mu_at_high:
                        edge_status = "edge_low" if mu_at_low else "edge_high"
                        direction = (
                            "more negative Z (below z_start)"
                            if mu_at_low
                            else "more positive Z (above z_end)"
                        )
                        logger.info(
                            "STREAM_AF:%sgaussian mu=%.3f pinned within %.2f um "
                            "of sampled %s boundary [%.3f, %.3f]; amplitude "
                            "low (%.2f%%) but shape clean (R^2=%.2f sigma=%.2f "
                            "um). Classifying as %s -- retry will shift toward "
                            "%s.",
                            tag_prefix,
                            mu_fit,
                            boundary_tol,
                            "low" if mu_at_low else "high",
                            z_min_sampled,
                            z_max_sampled,
                            metric_range_frac * 100.0,
                            r2,
                            sigma_fit,
                            edge_status,
                            direction,
                        )
                        return _ScanAttemptResult(
                            edge_status,
                            None,
                            n_motion_samples,
                            z_span,
                            f"gaussian mu={mu_fit:.3f} pinned at sampled "
                            f"{'low' if mu_at_low else 'high'} boundary "
                            f"(R^2={r2:.2f}, sigma={sigma_fit:.2f}um, "
                            f"amplitude {metric_range_frac:.2%}). True focus "
                            f"is likely at {direction}",
                            samples_trace=list(in_motion),
                        )

                # 2026-05-12 follow-up: monotonic-slope detector.
                #
                # The mu-at-boundary check above only fires when the
                # gaussian converges to a peak-shaped fit (shape_ok).
                # When the true peak is FAR outside the scan window the
                # metric across the scan is a clean monotonic slope, the
                # gaussian fit goes degenerate (sigma >= 0.45 * z_span,
                # fitter saturated at the upper bound), shape_ok rejects
                # it, and we fall through to refusal -- losing the
                # directional information the slope is screaming at us.
                #
                # Repeatable test cases (PPM 10x, 2026-05-12):
                #  - 13:50 / 14:26: focus ~14 um past Z=+25, scan from
                #    Z=0 with range=50. Amplitude 3.92-4.05%, sigma~42um
                #    (== z_span -> degenerate).
                #  - 14:30: started at Z=10, focus far above Z=+35.
                #    Amplitude only 2.39% but the metric rises cleanly
                #    from 176 at z=-7 to 180 at z=+35 with R^2 ~0.85
                #    on a linear fit.
                #
                # Criterion: Pearson correlation between z and metric.
                # Captures monotonicity DIRECTLY, independent of raw
                # amplitude. ~0 for noise, ~0 for centered interior
                # peaks (rises then falls cancel out), ~+/-1 for clean
                # monotonic slopes regardless of slope magnitude.
                #
                # Gating: sigma_degenerate (sigma >= 0.45*z_span OR
                # gaussian failed) screens out interior peaks -- those
                # have narrow sigma so the slope detector never runs
                # on them. That preserves yesterday's z=-16.9 case
                # (sigma=3.19 << 19.2 -> not degenerate -> slope check
                # skipped -> ambiguous interior peak refused, correct)
                # and the 2026-05-06 regression case.
                #
                # Floor: SLOPE_MIN_AMPLITUDE = 0.005 (0.5% of peak).
                # Just enough to reject "metric is constant to 4
                # decimal places but happens to land sigma_degenerate
                # because gaussian failed to converge on flat noise."
                #
                # Existing safety nets keep the retry walk bounded:
                # MAX_EDGE_RETRIES caps to 3 attempts total; stage
                # z-limit gate refuses windows past configured bounds;
                # opposite-edge oscillation short-circuit catches
                # ping-pong; union-fit recovers if a real interior peak
                # was visible in any attempt.
                SLOPE_PEARSON_R_THRESHOLD = 0.5
                SLOPE_MIN_AMPLITUDE = 0.005
                sigma_degenerate = gaussian_fit is None or sigma_fit >= 0.45 * max(z_span, 1e-6)
                if sigma_degenerate and n_motion_samples >= 8:
                    zs_arr = np.asarray([s[1] for s in in_motion])
                    ms_arr = np.asarray([s[2] for s in in_motion])
                    if float(np.std(zs_arr)) > 1e-6 and float(np.std(ms_arr)) > 1e-12:
                        pearson_r = float(np.corrcoef(zs_arr, ms_arr)[0, 1])
                    else:
                        pearson_r = 0.0
                    amplitude_above_floor_for_slope = metric_range_frac >= SLOPE_MIN_AMPLITUDE
                    if (
                        abs(pearson_r) >= SLOPE_PEARSON_R_THRESHOLD
                        and amplitude_above_floor_for_slope
                    ):
                        if pearson_r > 0:
                            slope_status = "edge_high"
                            slope_direction = "more positive Z (above z_end)"
                        else:
                            slope_status = "edge_low"
                            slope_direction = "more negative Z (below z_start)"
                        logger.info(
                            "STREAM_AF:%smonotonic slope detected: Pearson "
                            "r=%+.3f across %d samples (amplitude %.2f%% of "
                            "peak). Gaussian fit degenerate (sigma=%.2f um vs "
                            "0.45*span=%.2f um) but head-to-tail trend is "
                            "clear. Classifying as %s -- retry will shift "
                            "toward %s.",
                            tag_prefix,
                            pearson_r,
                            n_motion_samples,
                            metric_range_frac * 100.0,
                            sigma_fit,
                            0.45 * z_span,
                            slope_status,
                            slope_direction,
                        )
                        return _ScanAttemptResult(
                            slope_status,
                            None,
                            n_motion_samples,
                            z_span,
                            f"monotonic slope across scan "
                            f"(Pearson r={pearson_r:+.3f}, amplitude "
                            f"{metric_range_frac:.2%}). True focus is "
                            f"likely at {slope_direction}",
                            samples_trace=list(in_motion),
                        )

                logger.warning(
                    "STREAM_AF:%smetric range %.3f (%.2f%% of peak %.3f) "
                    "is within noise -- gaussian R^2=%.2f sigma=%.2f um "
                    "(span %.2f um) does not show a clear peak. "
                    "Widen --range or use a higher-mag objective.",
                    tag_prefix,
                    metric_range,
                    metric_range_frac * 100.0,
                    metric_peak,
                    r2,
                    sigma_fit,
                    z_span,
                )
                # Dump per-sample trace on the refusal path so a
                # developer can confirm whether the metric is genuinely
                # flat across the swept Z range, or whether all samples
                # collapsed to the same Z (stage not actually moving).
                # Logged at DEBUG so production operator logs are not
                # flooded with 100s of lines per refusal; the WARNING
                # summary line above is enough for operator visibility.
                for i, (t, z, m) in enumerate(in_motion):
                    logger.debug(
                        "STREAM_AF:%sFLAT sample %3d  t=%7.1f ms  z=%.3f  metric=%.4f",
                        tag_prefix,
                        i,
                        t,
                        z,
                        m,
                    )
                return _ScanAttemptResult(
                    "metric_flat",
                    None,
                    n_motion_samples,
                    z_span,
                    f"metric range {metric_range_frac:.2%} of peak is "
                    f"within noise (gaussian R^2={r2:.2f}); scan window "
                    f"{z_span:.2f} um is likely inside one depth-of-field. "
                    f"Widen --range or switch to Sweep Focus.",
                    samples_trace=list(in_motion),
                )

            # Prefer a full-sample Gaussian fit (uses all N samples,
            # robust to a single noisy point), fall back to 3-point
            # parabolic (uses only the argmax neighborhood), fall
            # back to raw argmax.
            if gaussian_fit is not None:
                best_z = gaussian_fit[0]
                fit_kind = "gaussian"
            elif parabolic is not None:
                best_z = parabolic
                fit_kind = "parabolic"
            else:
                best_z = raw_peak_z
                fit_kind = "raw-argmax"
            logger.info(
                "STREAM_AF:%s%d in-motion samples  raw peak Z=%.3f  "
                "fit=%s best_z=%.3f  z_span=%.3f  range_frac=%.2f%%  R^2=%.2f  sigma=%.2f",
                tag_prefix,
                n_motion_samples,
                raw_peak_z,
                fit_kind,
                best_z,
                z_span,
                metric_range_frac * 100.0,
                r2,
                sigma_fit,
            )
        else:
            logger.warning(
                "STREAM_AF:%sonly %d in-motion samples -- cannot fit", tag_prefix, n_motion_samples
            )
            return _ScanAttemptResult(
                "insufficient_samples",
                None,
                n_motion_samples,
                0.0,
                f"only {n_motion_samples} usable samples, need {MIN_FRAMES_FOR_FIT}",
                samples_trace=list(in_motion),
            )

        # Per-sample trace at DEBUG only -- 268 INFO lines per AF run was
        # the "spam" the user flagged on 2026-05-05. The summary line
        # above already names raw_peak/best_z/z_span at INFO; the
        # detailed trace is for the FLAT-refusal branch (still INFO --
        # see ~50 lines earlier) and for log-level=DEBUG triage.
        if logger.isEnabledFor(logging.DEBUG):
            for i, (t, z, m) in enumerate(in_motion):
                logger.debug(
                    "STREAM_AF:%ssample %3d  t=%7.1f ms  z=%.3f  metric=%.4f",
                    tag_prefix,
                    i,
                    t,
                    z,
                    m,
                )

        # Concise diagnostic at INFO when the peak looks suspicious so
        # the operator notices without having to enable DEBUG. "Peak in
        # first 10% of the sweep with metric flat across the rest" is
        # the textbook coverslip / stale-buffer signature.
        try:
            metrics_arr = [m for (_, _, m) in in_motion]
            if metrics_arr and raw_peak_idx is not None:
                head_frac = (raw_peak_idx + 1) / max(len(metrics_arr), 1)
                tail_metrics = metrics_arr[max(raw_peak_idx + 5, 0) :]
                if tail_metrics:
                    tail_med = float(np.median(tail_metrics))
                    tail_range = float(max(tail_metrics) - min(tail_metrics))
                    tail_var_pct = (tail_range / max(tail_med, 1.0)) * 100.0
                    peak_metric = float(metrics_arr[raw_peak_idx])
                    peak_over_tail = peak_metric / max(tail_med, 1.0)
                    if head_frac < 0.15 and tail_var_pct < 2.0 and peak_over_tail > 1.3:
                        logger.warning(
                            "STREAM_AF:%speak suspicious -- raw peak in first "
                            "%.0f%% of sweep (idx=%d/%d, Z=%.3f), then metric "
                            "flat at %.2g (%.1f%% range) across rest. "
                            "Suggests coverslip / stale-buffer false peak "
                            "rather than tissue focus.",
                            tag_prefix,
                            head_frac * 100,
                            raw_peak_idx,
                            len(metrics_arr),
                            in_motion[raw_peak_idx][1],
                            tail_med,
                            tail_var_pct,
                        )
        except Exception:
            pass

        if n_motion_samples < MIN_FRAMES_FOR_FIT or best_z is None:
            return _ScanAttemptResult(
                "insufficient_samples",
                None,
                n_motion_samples,
                z_span,
                f"only {n_motion_samples} usable samples, need {MIN_FRAMES_FOR_FIT}",
                samples_trace=list(in_motion),
            )

        # 2026-05-12 follow-up #2: sampled-boundary edge detection.
        #
        # The legacy edge check below uses raw_peak_z vs the COMMANDED
        # z_start/z_end with a fixed 10%-of-range tolerance. After
        # HEAD_DISCARD_MS removes the first ~7 um at v=11.5 um/s, the
        # first in-motion sample sits ~7 um inside z_start -- so a
        # fitted peak whose mu lands at the actual sampled boundary
        # gets classified as "interior by ~7 um" against the commanded
        # boundary and falsely commits.
        #
        # _gaussian_peak constrains mu to [z_arr.min(), z_arr.max()];
        # when the true peak extends past the sampled range, curve_fit
        # pushes mu to the nearest boundary. So "mu within sigma_fit of
        # a sampled boundary" is the signature of a peak whose actual
        # center is past the boundary -- regardless of whether the
        # observed amplitude is high or low.
        #
        # Repeatable test cases (PPM 10x with brenner_gradient, 2026-
        # 05-12 23:00):
        #  - Start Z=0, scan [-25, 25]: fits mu=-16.7, sigma=4.99,
        #    amplitude 34%, R^2=0.97. First in-motion sample ~ z=-18.
        #    |mu - z_min_sampled| = 1.4 um <= 4.99 -> mu pinned at low
        #    boundary. Legacy check used commanded z_lo=-25 -> distance
        #    8.3 um > 5 um tolerance -> classified as interior ->
        #    committed -16.7. True focus at -26.
        #  - Start Z=10, scan [-15, 35]: fits mu=-6.9, sigma=8.62,
        #    amplitude 11.3%, R^2=0.96. First in-motion ~ z=-8.1.
        #    |mu - z_min_sampled| = 1.2 um <= 8.62. Same failure.
        #
        # Returns best_z = mu_fit so the retry loop's post-loop
        # fallback can commit to this peak if the walk doesn't find
        # anything better. Yesterday's z=-16.9 in [-58.5, -8.5] case
        # still passes through: sigma=3.19, distance from mu to
        # nearest sampled boundary >= 8.4 um >> 3.19 -> NOT at boundary.
        if gaussian_fit is not None and shape_ok and n_motion_samples >= 3:
            z_min_sampled = float(min(zs))
            z_max_sampled = float(max(zs))
            mu_fit = float(gaussian_fit[0])
            boundary_tol = max(sigma_fit, 0.5)
            mu_at_low_sampled = abs(mu_fit - z_min_sampled) <= boundary_tol
            mu_at_high_sampled = abs(mu_fit - z_max_sampled) <= boundary_tol
            if mu_at_low_sampled != mu_at_high_sampled:
                edge_status = "edge_low" if mu_at_low_sampled else "edge_high"
                direction = (
                    "more negative Z (below z_start)"
                    if mu_at_low_sampled
                    else "more positive Z (above z_end)"
                )
                logger.info(
                    "STREAM_AF:%sgaussian mu=%.3f pinned within %.2f um of "
                    "sampled %s boundary [%.3f, %.3f] (sigma=%.2f, R^2=%.2f, "
                    "amplitude=%.2f%%). Peak likely extends past sampled "
                    "range. Classifying as %s with best_z=%.3f as fallback "
                    "if retry finds nothing better.",
                    tag_prefix,
                    mu_fit,
                    boundary_tol,
                    "low" if mu_at_low_sampled else "high",
                    z_min_sampled,
                    z_max_sampled,
                    sigma_fit,
                    r2,
                    metric_range_frac * 100.0,
                    edge_status,
                    mu_fit,
                )
                return _ScanAttemptResult(
                    edge_status,
                    mu_fit,
                    n_motion_samples,
                    z_span,
                    f"gaussian mu={mu_fit:.3f} pinned at sampled "
                    f"{'low' if mu_at_low_sampled else 'high'} boundary "
                    f"(sigma={sigma_fit:.2f}um, amplitude="
                    f"{metric_range_frac:.2%}, R^2={r2:.2f}). True focus "
                    f"is likely at {direction}",
                    samples_trace=list(in_motion),
                )

        # Edge-of-window detection.
        # 2026-05-06: was `raw_peak_idx in (0, n_motion_samples - 1)`,
        # which checked SAMPLE INDEX. That worked when samples were
        # densely uniform across the scan range, but on PPM today with
        # ~20 in-motion samples and HEAD_DISCARD_MS=600 eating 35% of
        # the 1.7 s scan, sample 0 is no longer at the scan edge -- it's
        # 35% of the way through the move. Falsely flagged true-interior
        # peaks at Z=-65.97 (6.9 um from the low edge of [-72.9, -52.9])
        # as edge_low and triggered an unnecessary retry that walked off
        # focus. Switch to checking the peak's actual Z relative to the
        # commanded z_start / z_end with a tolerance of max(1 um, 10% of
        # range).
        edge_tolerance_um = max(1.0, range_um * 0.10)
        peak_z = zs[raw_peak_idx]
        z_lo, z_hi = (min(z_start, z_end), max(z_start, z_end))
        at_low_edge = peak_z <= z_lo + edge_tolerance_um
        at_high_edge = peak_z >= z_hi - edge_tolerance_um
        if n_motion_samples >= 3 and (at_low_edge or at_high_edge):
            if at_low_edge:
                status = "edge_low"
                direction = "more negative Z (below z_start)"
            else:
                status = "edge_high"
                direction = "more positive Z (above z_end)"
            reason = (
                f"peak Z={peak_z:.3f} within {edge_tolerance_um:.1f} um of "
                f"scan edge [{z_lo:.3f}, {z_hi:.3f}] (sample idx "
                f"{raw_peak_idx} of {n_motion_samples}). True focus is "
                f"likely at {direction}"
            )
            return _ScanAttemptResult(
                status,
                None,
                n_motion_samples,
                z_span,
                reason,
                samples_trace=list(in_motion),
            )

        return _ScanAttemptResult(
            "success",
            best_z,
            n_motion_samples,
            z_span,
            f"peak at Z={best_z:.3f}",
            samples_trace=list(in_motion),
        )

    except Exception as e:
        logger.error("STREAM_AF:%sunhandled error during scan: %s", tag_prefix, e, exc_info=True)
        return _ScanAttemptResult(
            "error",
            None,
            0,
            0.0,
            str(e),
        )


def _brent_fallback_scan(
    core,
    focus_device: str,
    speed_prop: Optional[str],
    z_lo: float,
    z_hi: float,
    metric_name: str,
    max_evals: int = 8,
    abs_tolerance_um: float = 0.5,
    normal_value: str = NORMAL_SPEED_VALUE,
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

    Adapted from: Micro-Manager's BrentFocusOptimizer.java. We
    use scipy.optimize.minimize_scalar in place of MM's Java
    Brent implementation, with the same bracket-then-refine
    structure. See the "Attribution / Prior art" section in the
    module docstring for the full citation.
    """
    tag = "brent-fallback"
    logger.info(
        "STREAM_AF:%s: Brent search over [%.3f, %.3f] metric=%s", tag, z_lo, z_hi, metric_name
    )

    try:
        from scipy.optimize import minimize_scalar
    except Exception as e:
        return _ScanAttemptResult(
            "error",
            None,
            0,
            0.0,
            f"scipy not available for Brent: {e}",
        )

    if z_hi <= z_lo:
        return _ScanAttemptResult(
            "error",
            None,
            0,
            0.0,
            f"empty Brent bracket [{z_lo}, {z_hi}]",
        )

    # Brent's method needs a 3-point bracket where the middle has a
    # lower function value than both ends (we're MINIMIZING negative
    # metric, i.e. maximizing metric). Start with a center at
    # midpoint of the bracket.
    z_mid = (z_lo + z_hi) / 2.0

    # Use full stage speed for Brent evaluations -- each one is a
    # stationary snap, no benefit to running slowly. Skip when the
    # stage has no writable speed property; Brent doesn't need one.
    if speed_prop is not None:
        _try_set(core, focus_device, speed_prop, normal_value)

    # Track every evaluation for the eventual result.
    evals: List[Tuple[float, float]] = []  # (z, metric)

    def neg_metric(z: float) -> float:
        try:
            core.set_position(focus_device, float(z))
            _wait_via_busy(core, focus_device, target_z=float(z))
            core.snap_image()
            img = _snap_image_as_numpy(core)
            z_actual = float(core.get_position(focus_device))
        except Exception as e:
            logger.warning("STREAM_AF:%s eval at z=%.3f failed: %s", tag, z, e)
            return 0.0
        m = _focus_metric(img, metric_name)
        evals.append((z_actual, m))
        logger.info("STREAM_AF:%s eval %2d  z=%.3f  metric=%.4f", tag, len(evals), z_actual, m)
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
        logger.warning("STREAM_AF:%s minimize_scalar raised: %s", tag, e)
        # Fall back to argmax of what we collected.
        if evals:
            best_z, best_m = max(evals, key=lambda p: p[1])
            z_span = max(z for z, _ in evals) - min(z for z, _ in evals)
            return _ScanAttemptResult(
                "success" if best_m > 0 else "error",
                best_z,
                len(evals),
                z_span,
                f"Brent raised ({e}); argmax of {len(evals)} evals",
                samples_trace=list(evals),
            )
        return _ScanAttemptResult(
            "error",
            None,
            0,
            0.0,
            f"Brent failed with no evals: {e}",
        )

    best_z = float(result.x)
    # Clamp to bracket (scipy Brent can sometimes report just outside)
    best_z = max(z_lo, min(z_hi, best_z))
    z_span = (max(z for z, _ in evals) - min(z for z, _ in evals)) if evals else 0.0

    logger.info("STREAM_AF:%s converged at z=%.3f after %d evals", tag, best_z, len(evals))
    return _ScanAttemptResult(
        "success",
        best_z,
        len(evals),
        z_span,
        f"Brent converged at z={best_z:.3f} after {len(evals)} evals",
        samples_trace=list(evals),
    )


def handle_streaming_focus(conn, client, hardware, settings, **kwargs):
    """Entry point for the STRMAFZ (streaming autofocus) command."""
    addr = getattr(client, "addr", client)

    # Read the text payload (same framing as other flag-based handlers).
    try:
        message = read_message_string(conn)
    except Exception as e:
        logger.error("STREAM_AF:failed to read payload from %s: %s", addr, e)
        try:
            conn.sendall(f"FAILED:payload-read-error: {e}".encode())
        except Exception:
            pass
        return

    params = parse_flags(
        message,
        [
            "--yaml",
            "--objective",
            "--range",
            "--modality",
            "--crop-factor",
            "--dump",
            "--max-attempts",
        ],
    )
    yaml_path = params.get("yaml")
    client_objective = params.get("objective")
    range_override_str = params.get("range")
    client_modality = params.get("modality")
    crop_factor_str = params.get("crop_factor")
    dump_flag = params.get("dump")
    max_attempts_str = params.get("max_attempts")
    range_override_um: Optional[float] = None
    if range_override_str:
        try:
            range_override_um = float(range_override_str)
        except ValueError:
            logger.warning("STREAM_AF:ignoring non-numeric --range: %r", range_override_str)

    # --max-attempts overrides MAX_EDGE_RETRIES+1 for callers that want a
    # tighter retry budget. Acquisition tile-AF passes 1 to skip the walk
    # entirely (the previous tile's Z is a tight seed; the peak should fit
    # in one scan). Live Viewer leaves it unset to keep the default 3.
    max_attempts = MAX_EDGE_RETRIES + 1
    if max_attempts_str:
        try:
            requested = int(max_attempts_str)
            if requested >= 1:
                max_attempts = requested
            else:
                logger.warning("STREAM_AF:ignoring --max-attempts < 1: %r", max_attempts_str)
        except ValueError:
            logger.warning("STREAM_AF:ignoring non-integer --max-attempts: %r", max_attempts_str)

    crop_factor = DEFAULT_CROP_FACTOR
    if crop_factor_str:
        try:
            cf = float(crop_factor_str)
            if 0.0 < cf <= 1.0:
                crop_factor = cf
            else:
                logger.warning(
                    "STREAM_AF:--crop-factor=%r out of (0, 1]; " "using default %.2f",
                    crop_factor_str,
                    DEFAULT_CROP_FACTOR,
                )
        except ValueError:
            logger.warning("STREAM_AF:ignoring non-numeric --crop-factor: %r", crop_factor_str)

    if not yaml_path:
        try:
            conn.sendall(b"FAILED:missing --yaml")
        except Exception:
            pass
        return

    # Dump mode: when --dump is set (any truthy string), the streaming
    # scan saves per-sample TIFs + a CSV trace + a manifest under the
    # config directory's logs subdir. The Test Streaming AF button
    # in the autofocus editor is the primary caller; the path is sent
    # back in the SUCCESS response so the UI can surface it.
    dump_root: Optional[Path] = None
    dump_enabled = bool(dump_flag) and str(dump_flag).strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if dump_enabled:
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dump_root = (
                Path(yaml_path).parent / "logs" / "streaming_af_dumps" / f"streaming_af_{timestamp}"
            )
            dump_root.mkdir(parents=True, exist_ok=True)
            logger.info("STREAM_AF:dump enabled, writing to %s", dump_root)
        except Exception as e:
            logger.warning(
                "STREAM_AF:could not create dump dir (%s); " "continuing without dump",
                e,
            )
            dump_root = None

    logger.info(
        "STREAM_AF:request from %s yaml=%s objective=%s modality=%s "
        "range_override=%s crop_factor=%.2f dump=%s",
        addr,
        yaml_path,
        client_objective,
        client_modality,
        range_override_um,
        crop_factor,
        dump_root if dump_root else "off",
    )

    # Note: focus-metric resolution is deferred until after the
    # autofocus yaml entry is loaded, so the per-objective
    # `score_metric` field (if any) can override the modality
    # default. See _resolve_metric_name() and the call site below.

    # Resolve the saturation threshold from the client-provided
    # modality. Normalize to lower case for dict lookup; unknown or
    # missing modalities fall back to the conservative default.
    if client_modality:
        sat_threshold = SATURATION_THRESHOLD_BY_MODALITY.get(
            client_modality.strip().lower(),
            DEFAULT_SATURATION_REFUSE_FRACTION,
        )
        logger.info(
            "STREAM_AF:saturation threshold for modality '%s' = %.2f",
            client_modality,
            sat_threshold,
        )
    else:
        sat_threshold = DEFAULT_SATURATION_REFUSE_FRACTION
        logger.info(
            "STREAM_AF:no modality given, using default saturation threshold %.2f", sat_threshold
        )

    core = hardware.core
    try:
        focus_device = core.get_focus_device()
    except Exception as e:
        logger.error("STREAM_AF:get_focus_device failed: %s", e)
        conn.sendall(f"FAILED:no-focus-device: {e}".encode())
        return
    logger.info("STREAM_AF:focus device = %s", focus_device)

    # --- Objective resolution ---
    objective, source = _resolve_objective(core, settings, client_objective)
    if objective:
        logger.info("STREAM_AF:resolved objective '%s' via %s", objective, source)
    else:
        logger.warning("STREAM_AF:could not resolve objective; using first yaml entry")

    af_entry = _load_autofocus_yaml_for_objective(yaml_path, objective)
    if not af_entry:
        logger.warning(
            "STREAM_AF:no autofocus yaml entry -- using fallback range %s um", FALLBACK_RANGE_UM
        )

    # Now that af_entry is known, resolve the focus metric. The
    # yaml's per-objective `score_metric` wins over the modality
    # default; unknown names fall through with a warning.
    metric_name = _resolve_metric_name(client_modality, af_entry)
    yaml_score_metric = af_entry.get("score_metric") if af_entry else None
    if yaml_score_metric:
        logger.info(
            "STREAM_AF:focus metric for modality '%s' = '%s' " "(yaml score_metric=%r)",
            client_modality or "unknown",
            metric_name,
            yaml_score_metric,
        )
    else:
        logger.info(
            "STREAM_AF:focus metric for modality '%s' = '%s' (modality default)",
            client_modality or "unknown",
            metric_name,
        )

    if range_override_um is not None:
        range_um = max(1.0, float(range_override_um))
        logger.info("STREAM_AF:using range override = %.2f um", range_um)
    else:
        range_um = float(af_entry.get("sweep_range_um", FALLBACK_RANGE_UM))
        logger.info("STREAM_AF:using sweep_range_um from yaml = %.2f um", range_um)

    # --- streaming_af YAML config (populated by setup-wizard probe) ---
    # The block at stage.streaming_af in config_<scope>.yml drives the
    # speed values used during the sweep. Each key falls back to the
    # legacy hardcoded constant when absent so pre-migration configs
    # keep working until they're re-probed.
    sa_cfg = _load_streaming_af_config(yaml_path)
    yaml_enabled = sa_cfg.get("enabled")
    yaml_speed_prop = sa_cfg.get("speed_property")
    yaml_slow_value = sa_cfg.get("slow_speed_value")
    yaml_slow_um_s = sa_cfg.get("slow_speed_um_per_s")
    yaml_normal_value = sa_cfg.get("normal_speed_value")

    # The handler's per-call effective values. Start from legacy
    # constants; YAML overrides any populated key.
    eff_slow_value = str(yaml_slow_value) if yaml_slow_value is not None else SLOW_SPEED_VALUE
    eff_normal_value = (
        str(yaml_normal_value) if yaml_normal_value is not None else NORMAL_SPEED_VALUE
    )
    eff_slow_um_s = float(yaml_slow_um_s) if yaml_slow_um_s is not None else MIN_VELOCITY_UM_S
    if sa_cfg:
        logger.info(
            "STREAM_AF:streaming_af config: enabled=%s slow=%r normal=%r um/s=%.2f",
            yaml_enabled,
            eff_slow_value,
            eff_normal_value,
            eff_slow_um_s,
        )
    else:
        logger.info(
            "STREAM_AF:no stage.streaming_af YAML block; using legacy "
            "constants (slow=%r normal=%r). Run 'Re-probe Stage AF' to "
            "calibrate for this hardware.",
            eff_slow_value,
            eff_normal_value,
        )

    # --- Speed property discovery ---
    # speed_prop is None when:
    #   (a) the stage exposes no writable speed-like property, OR
    #   (b) the YAML explicitly disables streaming for this rig.
    # In both cases the per-attempt _try_set short-circuits with
    # 'no_slow_speed' and the existing Brent escalation handles the
    # acquisition. No additional branches.
    if yaml_enabled is False:
        speed_prop = None
        original_speed = None
        logger.info(
            "STREAM_AF:streaming disabled in YAML (stage.streaming_af.enabled=false); "
            "routing this acquisition to Brent snap-and-stop fallback",
        )
    else:
        # If the wizard recorded a specific property name, prefer it
        # so we don't re-scan candidates every call. Validate it's
        # writable; fall back to auto-detect on miss.
        speed_prop = None
        if yaml_speed_prop:
            try:
                if not core.is_property_read_only(focus_device, yaml_speed_prop):
                    speed_prop = yaml_speed_prop
            except Exception:
                pass
        if speed_prop is None:
            speed_prop = _find_speed_property(core, focus_device)
        if speed_prop is None:
            logger.info(
                "STREAM_AF:focus device '%s' has no writable speed property "
                "(searched %s); streaming disabled, using Brent snap-and-stop "
                "fallback for this acquisition",
                focus_device,
                list(SPEED_PROPERTY_CANDIDATES),
            )
            original_speed = None
        else:
            logger.info("STREAM_AF:stage speed property = '%s'", speed_prop)
            original_speed = _try_get(core, focus_device, speed_prop)
    try:
        initial_z = float(core.get_position(focus_device))
    except Exception as e:
        logger.error("STREAM_AF:get_position failed: %s", e)
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
    # Exposure with FrameRateHz=1 produced ~1.2 fps during a streaming AF
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
            saved_frame_rate_hz = float(core.get_property("JAICamera", "FrameRateHz"))
        except Exception as e:
            logger.warning("STREAM_AF:could not read JAICamera FrameRateHz: %s", e)
            saved_frame_rate_hz = None
        if saved_frame_rate_hz is not None and saved_frame_rate_hz < 30.0:
            logger.warning(
                "STREAM_AF:JAICamera FrameRateHz=%.2f Hz is too low for "
                "streaming focus; temporarily forcing to 38 Hz. The Live "
                "Viewer was also producing frames at this rate -- re-apply "
                "your camera preset to fix it permanently.",
                saved_frame_rate_hz,
            )
            try:
                core.set_property("JAICamera", "FrameRateHz", 38.0)
                logger.info(
                    "STREAM_AF:bumped JAICamera FrameRateHz from %.2f to 38.0",
                    saved_frame_rate_hz,
                )
            except Exception as e:
                logger.warning(
                    "STREAM_AF:could not set JAICamera FrameRateHz=38 mid-stream "
                    "(%s); scan may still be starved",
                    e,
                )
        elif saved_frame_rate_hz is not None:
            logger.info(
                "STREAM_AF:JAICamera FrameRateHz=%.2f Hz (above threshold, leaving alone)",
                saved_frame_rate_hz,
            )

    # --- Pre-flight: exposure * velocity blur budget ---
    try:
        exposure_ms = float(core.get_exposure())
    except Exception as e:
        logger.warning("STREAM_AF:get_exposure failed: %s", e)
        exposure_ms = 0.0

    # Per-rig slow velocity estimate. Sourced from
    # stage.streaming_af.slow_speed_um_per_s in the YAML when the
    # wizard has probed this rig; otherwise falls back to
    # MIN_VELOCITY_UM_S (Prior MaxSpeed=1 measurement).
    min_velocity_um_s = eff_slow_um_s
    expected_blur_um = min_velocity_um_s * (exposure_ms / 1000.0) if exposure_ms else 0.0
    logger.info(
        "STREAM_AF:exposure=%.2fms  est min velocity=%.2f um/s  "
        "expected blur=%.3f um  budget=%.3f um",
        exposure_ms,
        min_velocity_um_s,
        expected_blur_um,
        BLUR_BUDGET_UM,
    )
    if expected_blur_um > BLUR_BUDGET_UM:
        reason = (
            f"exposure {exposure_ms:.1f} ms x min velocity {min_velocity_um_s:.1f} "
            f"um/s = {expected_blur_um:.2f} um motion blur, exceeds "
            f"{BLUR_BUDGET_UM:.2f} um budget. Reduce exposure to "
            f"<={BLUR_BUDGET_UM / min_velocity_um_s * 1000:.1f} ms "
            f"or use a faster stage"
        )
        logger.warning("STREAM_AF:UNAVAILABLE -- %s", reason)
        conn.sendall(f"UNAVAILABLE:{reason}".encode())
        # The crop ROI was applied above; UNAVAILABLE returns are
        # outside the main try/finally that calls _restore_roi, so
        # restore here or the Live Viewer keeps showing the cropped
        # area on subsequent frame polls.
        _restore_roi(core, saved_roi, roi_seq_was_running)
        return

    # --- Pre-flight: saturation check ---
    # If the Live Viewer (or any caller) has a sequence running, pop
    # one frame from its buffer instead of calling snap_image(). A
    # blocking snap on the JAI costs ~400 ms (exposure + readout +
    # driver overhead) and is the single biggest fixed cost in the
    # Streaming AF handler -- nearly 20% of the total scan time. Stream
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
            logger.info("STREAM_AF:pre-flight frame via stream pop (no snap)")
        else:
            logger.info("STREAM_AF:stream pop failed, falling back to snap_image")
    if preflight_img is None:
        preflight_img = _snap_image_as_numpy(core)
        logger.info("STREAM_AF:pre-flight frame via snap_image")

    sat_frac = _saturation_fraction(preflight_img)
    logger.info(
        "STREAM_AF:pre-flight saturation fraction = %.3f (threshold %.2f)", sat_frac, sat_threshold
    )
    if sat_frac > sat_threshold:
        reason = (
            f"{sat_frac * 100:.1f}% of pixels saturated (threshold for "
            f"'{client_modality or 'unknown'}' modality is "
            f"{sat_threshold * 100:.1f}%); focus metric will not "
            f"discriminate. Reduce exposure/gain before using streaming autofocus"
        )
        logger.warning("STREAM_AF:UNAVAILABLE -- %s", reason)
        conn.sendall(f"UNAVAILABLE:{reason}".encode())
        # Same caveat as the blur-budget early return above: this
        # bypasses the main try/finally block, so restore the camera
        # ROI explicitly or the Live Viewer keeps showing the cropped
        # area indefinitely.
        _restore_roi(core, saved_roi, roi_seq_was_running)
        return

    # --- Execute scan with edge-retry loop ---
    # Up to max_attempts attempts (default MAX_EDGE_RETRIES + 1, or
    # whatever the caller passed via --max-attempts). Each attempt
    # runs one scan centered on a candidate Z with the current range.
    # On edge_low we shift the next attempt's center down by one full
    # range (covering new ground further in the -Z direction); on
    # edge_high we shift up. The shift never crosses outside the
    # stage Z limits from config.
    z_low, z_high = _get_z_limits(settings)
    logger.info(
        "STREAM_AF:stage Z limits from config: low=%s high=%s",
        f"{z_low:.3f}" if z_low is not None else "None",
        f"{z_high:.3f}" if z_high is not None else "None",
    )

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
    # 2026-05-12: track the best (peak_metric, peak_z) seen across all
    # attempts so far -- includes mu values from edge_low/edge_high
    # results when the mu-at-sampled-boundary check fires with a clean
    # gaussian fit. If the retry loop terminates without committing a
    # success, we fall back to this peak instead of refusing.
    fallback_peak_z: Optional[float] = None
    fallback_peak_metric: float = -float("inf")

    try:
        for attempt_idx in range(max_attempts):
            attempt_num = attempt_idx + 1
            label = f"attempt {attempt_num}/{max_attempts}"

            # Check Z limits before each attempt. Refuse if the
            # proposed window would step outside the configured stage
            # limits; the current attempt's center came from a
            # previous edge detection, so this is where we stop
            # walking.
            if not _scan_window_within_limits(current_center, range_um, z_low, z_high):
                reason = (
                    f"proposed scan window [{current_center - range_um/2:.3f} "
                    f"-> {current_center + range_um/2:.3f}] on "
                    f"{label} would exit stage z limits "
                    f"[{z_low}, {z_high}]"
                )
                logger.warning("STREAM_AF:%s", reason)
                attempts_log.append(f"{label}: out-of-range")
                final_result = _ScanAttemptResult(
                    "error",
                    None,
                    0,
                    0.0,
                    reason,
                )
                break

            # Run one attempt.
            attempt_dump_dir = (
                dump_root / f"attempt_{attempt_idx + 1}" if dump_root is not None else None
            )
            result = _attempt_one_scan(
                core,
                focus_device,
                speed_prop,
                current_center,
                range_um,
                sequence_was_running,
                attempt_label=label,
                velocity_um_s=min_velocity_um_s,
                metric_name=metric_name,
                slow_value=eff_slow_value,
                normal_value=eff_normal_value,
                dump_dir=attempt_dump_dir,
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

            # 2026-05-12: track the best peak across attempts. When mu-at-
            # sampled-boundary fires, _attempt_one_scan sets best_z=mu_fit
            # even though it returns edge_low/edge_high. If retries don't
            # find anything better, the post-loop dispatch commits to the
            # peak with the highest metric value observed across all
            # attempts (samples_trace contains all in-motion samples).
            if result.best_z is not None and result.samples_trace:
                metric_at_best = max(
                    (float(s[2]) for s in result.samples_trace if len(s) >= 3),
                    default=-float("inf"),
                )
                if metric_at_best > fallback_peak_metric:
                    fallback_peak_metric = metric_at_best
                    fallback_peak_z = float(result.best_z)

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
                prev_attempt_status == "edge_low" and result.status == "edge_high"
            ) or (prev_attempt_status == "edge_high" and result.status == "edge_low")
            if opposite_edge and len(all_attempt_samples_zm) >= 4:
                union_result = _fit_union_samples(
                    all_attempt_samples_zm,
                    len(attempts_log),
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
                logger.info(
                    "STREAM_AF:edge_low -- next attempt center will be %.3f", current_center
                )
                continue

            if result.status == "edge_high":
                current_center = current_center + range_um
                logger.info(
                    "STREAM_AF:edge_high -- next attempt center will be %.3f", current_center
                )
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
                "error",
                None,
                0,
                0.0,
                "unknown failure, no attempt completed",
            )

        # --- Union-fit pre-Brent escalation ---
        # Before going to Brent, give the union of all collected
        # samples one more chance. The retry loop may have exited
        # without triggering the in-loop opposite-edge short-circuit
        # (e.g. ran out of attempts on three same-direction edges,
        # or hit the stage limit). If the union has a clean
        # interior maximum we can commit it directly.
        #
        # metric_flat is included because a typical "walked past
        # focus" sequence is edge_low / edge_low / metric_flat:
        # the first two attempts catch the slope via Pearson, the
        # third lands in the flat region beyond focus and looks
        # noise-bounded on its own. But the *union* of all 3 still
        # has a clean interior maximum near the boundary between
        # the slope and the flat region. _fit_union_samples returns
        # None when the union argmax sits at an edge, so this is
        # safe -- worst case we fall through to Brent or UNAVAILABLE
        # exactly as before.
        if final_result.status in ("edge_low", "edge_high", "metric_flat"):
            union_result = _fit_union_samples(
                all_attempt_samples_zm,
                len(attempts_log),
            )
            if union_result is not None:
                final_result = union_result
                attempts_log.append(
                    f"union-fit (post-retry): status={union_result.status} "
                    f"n={union_result.n_samples} "
                    f"reason='{union_result.reason}'"
                )

        # --- Brent fallback ---
        # Three escalation paths land here:
        #   - edge_low / edge_high: streaming saw a slope but the
        #     peak is outside the window. The union fit also failed
        #     to find an interior peak in the combined data.
        #   - no_slow_speed: the stage refused the slow-speed value
        #     for streaming OR has no writable speed property at all
        #     (OWS3 'Speed' value rejected, or hardware that doesn't
        #     expose any of the candidate properties). Streaming is
        #     impossible but Brent's snap-and-stop search needs no
        #     speed manipulation, so we degrade to it directly.
        # Brent uses smart point placement and typically converges
        # in 6-8 evaluations even when the peak location is unknown,
        # so it rescues cases where the streaming+shift approach
        # misses the peak due to sample density, metric noise, or
        # awkward initial offset. We seed the bracket from the
        # metric peak of all collected samples (when available)
        # instead of the full coverage span -- a tight bracket
        # converges faster and avoids Brent landing on irrelevant Z
        # far from any actual sample. With no samples (no_slow_speed
        # on first attempt) we search a wider window.
        if final_result.status in ("edge_low", "edge_high", "no_slow_speed"):
            if all_attempt_samples_zm:
                # Anchor on the best sample we've already got, then
                # widen to one full range either side.
                best_z_so_far = max(all_attempt_samples_zm, key=lambda zm: zm[1])[0]
                brent_lo = best_z_so_far - range_um
                brent_hi = best_z_so_far + range_um
            else:
                total_span = range_um * max_attempts
                brent_lo = initial_z - total_span / 2.0
                brent_hi = initial_z + total_span / 2.0
            if z_low is not None:
                brent_lo = max(brent_lo, z_low)
            if z_high is not None:
                brent_hi = min(brent_hi, z_high)
            if brent_hi - brent_lo >= 2.0:  # need at least 2 um bracket
                logger.info(
                    "STREAM_AF:streaming retries exhausted with edge; "
                    "escalating to Brent fallback over [%.3f, %.3f]",
                    brent_lo,
                    brent_hi,
                )
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
                        core,
                        focus_device,
                        speed_prop,
                        brent_lo,
                        brent_hi,
                        metric_name,
                        normal_value=eff_normal_value,
                    )
                    # Restart the sequence if the Live Viewer was
                    # depending on it when we arrived.
                    if resume_sequence:
                        try:
                            core.clear_circular_buffer()
                            core.start_continuous_sequence_acquisition(0)
                        except Exception as e:
                            logger.warning(
                                "STREAM_AF:could not resume sequence " "after Brent: %s", e
                            )
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
                            all_attempt_samples_zm.append((float(s[0]), float(s[1])))
                    # Use Brent's best_z if it converged AND its
                    # metric beats our running global best. This
                    # protects against the failure mode where
                    # Brent's bracketing fails and minimize_scalar
                    # picks a far-edge eval (-9 um catastrophe on
                    # 23:06): instead we commit whichever sample
                    # across all attempts (streaming + Brent) had
                    # the highest metric.
                    global_best = max(
                        all_attempt_samples_zm,
                        key=lambda zm: zm[1],
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
                        z_span = max(zm[0] for zm in all_attempt_samples_zm) - min(
                            zm[0] for zm in all_attempt_samples_zm
                        )
                        logger.info(
                            "STREAM_AF:Brent did not converge; committing "
                            "global argmax across %d collected samples "
                            "at Z=%.3f (metric=%.4f)",
                            len(all_attempt_samples_zm),
                            gz,
                            gm,
                        )
                        final_result = _ScanAttemptResult(
                            "success",
                            gz,
                            len(all_attempt_samples_zm),
                            z_span,
                            f"global argmax across {len(all_attempt_samples_zm)} "
                            f"samples at Z={gz:.3f}",
                        )
                except Exception as e:
                    logger.error("STREAM_AF:Brent fallback raised: %s", e, exc_info=True)

        # 2026-05-12: mu-at-sampled-boundary fallback.
        #
        # If we still don't have a success and any earlier attempt had a
        # gaussian-fit peak (recorded in fallback_peak_z when the mu-at-
        # boundary check fired), commit to that peak instead of refusing.
        # The walk found nothing better, so the original peak -- even if
        # pinned to a boundary -- is the best information we have.
        #
        # Concretely: attempt 1 fits a clean peak at mu=-16.7 (pinned at
        # low boundary) and returns edge_low with best_z=-16.7. Retry
        # shifts the window and either finds a better peak (we commit
        # there) or returns metric_flat / nothing. In the latter case
        # the original -16.7 is still a real focus point in the FOV --
        # just not necessarily the global tissue focus the operator
        # wanted. Better than UNAVAILABLE.
        if final_result.status != "success" and fallback_peak_z is not None:
            # Only commit if the fallback peak's metric beats anything
            # else we've seen across all attempts. Otherwise the union-
            # fit / global-argmax path above already handled it.
            global_best_metric = max((m for _, m in all_attempt_samples_zm), default=-float("inf"))
            if fallback_peak_metric >= global_best_metric * 0.95:
                logger.info(
                    "STREAM_AF:no better peak found across %d attempts; "
                    "falling back to earlier mu-at-boundary peak at "
                    "Z=%.3f (metric=%.4f, global best metric=%.4f)",
                    len(attempts_log),
                    fallback_peak_z,
                    fallback_peak_metric,
                    global_best_metric,
                )
                final_result = _ScanAttemptResult(
                    "success",
                    fallback_peak_z,
                    final_result.n_samples,
                    final_result.z_span,
                    f"mu-at-boundary fallback peak at Z={fallback_peak_z:.3f} "
                    f"(no better peak found in {len(attempts_log)} attempts)",
                )

        if final_result.status == "success":
            # Commit the peak Z.
            best_z = final_result.best_z
            core.set_position(focus_device, best_z)
            _wait_via_busy(core, focus_device, target_z=best_z)
            try:
                final_z = float(core.get_position(focus_device))
            except Exception:
                final_z = best_z

            z_shift = final_z - initial_z
            logger.info(
                "STREAM_AF:committed final Z=%.3f  shift=%+.3f  n=%d  span=%.2f  "
                "after %d attempt(s)",
                final_z,
                z_shift,
                final_result.n_samples,
                final_result.z_span,
                len(attempts_log),
            )
            for entry in attempts_log:
                logger.info("STREAM_AF:attempt log -- %s", entry)

            response = (
                f"SUCCESS:{initial_z:.3f}:{final_z:.3f}:{z_shift:+.3f}:"
                f"{final_result.n_samples}:{final_result.z_span:.3f}"
            )
            if dump_root is not None:
                # Tack on the dump directory so the Test button in the
                # autofocus editor can render the curves and link the
                # TIF folder. Path uses the server-local FS layout
                # (Windows backslashes when the server is on Windows).
                response += f":dump={dump_root}"
            try:
                conn.sendall(response.encode())
            except Exception as e:
                logger.error("STREAM_AF:reply send failed: %s", e)
        else:
            # Every attempt failed to find an interior peak.
            # If we have collected samples with a focus slope, move to the
            # global argmax -- it's better than returning to initial_z.
            best_slope_z = None
            if final_result.status in ("edge_low", "edge_high") and all_attempt_samples_zm:
                global_best = max(
                    all_attempt_samples_zm,
                    key=lambda zm: zm[1],
                    default=None,
                )
                if global_best is not None:
                    best_slope_z = global_best[0]
                    try:
                        core.set_position(focus_device, best_slope_z)
                        _wait_via_busy(core, focus_device, target_z=best_slope_z)
                        logger.info(
                            "STREAM_AF:no peak found but moving to best Z=%.3f "
                            "(slope argmax across %d samples, shift %+.3f)",
                            best_slope_z,
                            len(all_attempt_samples_zm),
                            best_slope_z - initial_z,
                        )
                    except Exception:
                        best_slope_z = None

            if best_slope_z is None:
                try:
                    core.set_position(focus_device, initial_z)
                    _wait_via_busy(core, focus_device, target_z=initial_z)
                except Exception:
                    pass

            if final_result.status in ("edge_low", "edge_high"):
                if best_slope_z is not None:
                    summary = (
                        f"no peak found after {len(attempts_log)} "
                        f"attempts ({max_attempts} max), "
                        f"moved to best Z={best_slope_z:.3f} "
                        f"(shift {best_slope_z - initial_z:+.3f}um)"
                    )
                else:
                    summary = (
                        f"could not find peak after {len(attempts_log)} "
                        f"attempts ({max_attempts} max). Last attempt: "
                        f"{final_result.reason}. Try moving Z closer to "
                        f"focus manually or picking a wider scan range"
                    )
            elif final_result.status == "insufficient_samples":
                summary = f"{final_result.reason}; scan too short or " f"stage/camera timing off"
            else:
                summary = final_result.reason

            logger.warning("STREAM_AF:UNAVAILABLE -- %s", summary)
            for entry in attempts_log:
                logger.warning("STREAM_AF:attempt log -- %s", entry)
            unavailable_msg = f"UNAVAILABLE:{summary}"
            if dump_root is not None:
                unavailable_msg += f":dump={dump_root}"
            try:
                conn.sendall(unavailable_msg.encode())
            except Exception as e:
                logger.error("STREAM_AF:reply send failed: %s", e)

    except Exception as e:
        logger.error("STREAM_AF:unhandled error in retry loop: %s", e, exc_info=True)
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
        # Skip the restore entirely when the stage has no writable
        # speed property (speed_prop is None) -- there's nothing to
        # restore and _try_set with prop=None would fail noisily.
        if speed_prop is not None:
            if original_speed is not None:
                _try_set(core, focus_device, speed_prop, str(original_speed))
            else:
                _try_set(core, focus_device, speed_prop, eff_normal_value)
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
                        "JAICamera",
                        "FrameRateHz",
                        saved_frame_rate_hz,
                    )
                    logger.info(
                        "STREAM_AF:restored JAICamera FrameRateHz to %.2f",
                        saved_frame_rate_hz,
                    )
                except Exception as e:
                    logger.warning(
                        "STREAM_AF:could not restore JAICamera FrameRateHz: %s",
                        e,
                    )
            else:
                logger.info(
                    "STREAM_AF:leaving JAICamera FrameRateHz at 38.0 Hz "
                    "(saved %.2f Hz was a stale misconfiguration; "
                    "Live Viewer will now stream at full rate)",
                    saved_frame_rate_hz,
                )
