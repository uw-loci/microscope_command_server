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

# Polling intervals inside the scan loop.
SCAN_POLL_SLEEP_S = 0.002
BUSY_CHECK_EVERY_N = 6

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

# (Drain-based flushing was retired in favor of
# core.clear_circular_buffer() at the top of _run_smooth_scan.
# See the block comment there for why.)

# Hard deadline multiplier. Scan deadline = range_um * HARD_DEADLINE_SEC_PER_UM + 2.0s.
# At SLOW_SPEED_VALUE=1 on Prior (~11.5 um/s) we need ~0.09 s/um, so
# 0.15 gives enough headroom for other stage hardware without being
# absurd.
HARD_DEADLINE_SEC_PER_UM = 0.15

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


def _focus_metric(img) -> float:
    """Normalized variance on the green/first channel.

    Matches the metric family used by the acquisition-path sweep
    drift check (normalized_variance). Implemented inline so Smooth
    has zero runtime coupling to the autofocus module beyond what
    the yaml schema names.
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
    mean = g.mean()
    if mean <= 1e-9:
        return 0.0
    var = g.var()
    return float(var / mean)


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


# ----- The scan -----


def _run_smooth_scan(
    core,
    focus_device: str,
    speed_prop: str,
    z_start: float,
    z_end: float,
    hard_deadline_s: float,
) -> List[Tuple[float, float, float]]:
    """Execute the streaming-sample scan and return a list of
    (t_ms, z_at_pop, metric) triples. Leaves the camera in
    whatever streaming state it was in on entry (the caller is
    responsible for starting / stopping the sequence) and does NOT
    restore the speed property; caller is responsible for that too.
    """
    # Flush the buffer with a single atomic call right before firing
    # the non-blocking move.
    #
    # Pop-to-drain was the wrong primitive. The camera refills the
    # circular buffer faster than we can pop (camera at ~30 fps =
    # ~33 ms per frame, each pop over the ZMQ bridge is ~100 ms),
    # so any loop-based drain is losing ground. In the reuse-Live-
    # Viewer case the seed move to z_start takes ~200 ms, and a
    # drain of up to 200 ms more gives the camera ~400 ms of
    # stream time at z_start. That's ~12 frames of pre-motion
    # content queued in the buffer. Popping and draining only
    # removes maybe 3-4 of them, leaving 8 still queued when we
    # start the scan loop. Those stale frames get popped early,
    # labeled with mid-motion Z values (because z_at_pop reads the
    # LIVE stage position), and the metric/position pairs become
    # physical nonsense (two samples at very different labeled Z
    # showing identical metric to 0.001).
    #
    # clear_circular_buffer() is a single pycromanager RPC that
    # empties the FIFO atomically. It's safe to call on a running
    # sequence -- the camera keeps producing new frames into the
    # now-empty buffer as usual. Any frame that appears AFTER the
    # clear is from the camera's ongoing post-clear stream, which
    # means the first one arrives ~33 ms later, by which point
    # our non-blocking move has also been issued and the stage is
    # already accelerating toward z_end. Capture-vs-label Z skew
    # drops from ~3-5 um (drain approach) to ~0.2-0.4 um (below
    # stage quantization).
    try:
        core.clear_circular_buffer()
        logger.info("SMOOTH: flushed circular buffer before firing move")
    except Exception as e:
        logger.warning("SMOOTH: clear_circular_buffer failed "
                        "(continuing with whatever's queued): %s", e)

    samples: List[Tuple[float, float, float]] = []
    t0 = time.perf_counter()
    try:
        core.set_position(focus_device, z_end)
    except Exception as e:
        logger.error("SMOOTH: non-blocking move to z_end failed: %s", e)
        return samples

    stage_done_at: Optional[float] = None
    loop_count = 0
    deadline = time.perf_counter() + hard_deadline_s

    while time.perf_counter() < deadline:
        popped = 0
        try:
            while core.get_remaining_image_count() > 0:
                img = _pop_image_as_numpy(core)
                t_pop = (time.perf_counter() - t0) * 1000.0
                try:
                    z_at_pop = float(core.get_position(focus_device))
                except Exception:
                    z_at_pop = float("nan")
                samples.append((t_pop, z_at_pop, _focus_metric(img)))
                popped += 1
                if popped >= 4:
                    break
        except Exception as e:
            logger.warning("SMOOTH: pop loop failed: %s", e)

        loop_count += 1
        if loop_count % BUSY_CHECK_EVERY_N == 0:
            try:
                busy = core.device_busy(focus_device)
            except Exception:
                busy = None
            if busy is False and stage_done_at is None:
                stage_done_at = (time.perf_counter() - t0) * 1000.0

        if stage_done_at is not None:
            # Short tail so we catch any frames arriving between
            # stage-done and the command handshake settling.
            if (time.perf_counter() - t0) * 1000.0 - stage_done_at > 200.0:
                break

        time.sleep(SCAN_POLL_SLEEP_S)

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
) -> _ScanAttemptResult:
    """Run one Smooth scan centered on z_center with the given range.

    Returns an _ScanAttemptResult describing the outcome. Does NOT
    commit the peak (caller decides whether to retry or commit) and
    does NOT restore the stage Z (caller handles cleanup).

    The `attempt_label` is prepended to log lines so multi-attempt
    runs are easy to follow (e.g. 'attempt 2/3: ').
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
                                    z_start, z_end, hard_deadline_s)

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
            parabolic = _parabolic_peak(zs, ms) if n_motion_samples >= 3 else None
            best_z = parabolic if parabolic is not None else raw_peak_z
            logger.info("SMOOTH: %s%d in-motion samples  raw peak Z=%.3f  "
                        "parabolic peak=%s  z_span=%.3f",
                        tag_prefix, n_motion_samples, raw_peak_z,
                        f"{parabolic:.3f}" if parabolic is not None else "None",
                        z_span)
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

    params = parse_flags(message, ["--yaml", "--objective", "--range", "--modality"])
    yaml_path = params.get("yaml")
    client_objective = params.get("objective")
    range_override_str = params.get("range")
    client_modality = params.get("modality")
    range_override_um: Optional[float] = None
    if range_override_str:
        try:
            range_override_um = float(range_override_str)
        except ValueError:
            logger.warning("SMOOTH: ignoring non-numeric --range: %r", range_override_str)

    if not yaml_path:
        try:
            conn.sendall(b"FAILED:missing --yaml")
        except Exception:
            pass
        return

    logger.info("SMOOTH: request from %s yaml=%s objective=%s modality=%s range_override=%s",
                addr, yaml_path, client_objective, client_modality, range_override_um)

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
            )
            attempts_log.append(
                f"{label}: center={current_center:.3f} "
                f"range={range_um:.2f} status={result.status} "
                f"n={result.n_samples} reason='{result.reason}'"
            )

            if result.status == "success":
                final_result = result
                break

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
