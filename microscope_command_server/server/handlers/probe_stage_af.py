"""Setup-wizard probe for streaming-autofocus stage parameters.

Handles PRBSAFZ -- the wizard sends this once per rig (during initial
setup or via a "Re-probe Stage AF" menu item) to discover the focus
stage's writable speed property, parse its allowed values, time-verify
the result with a 1-um round-trip, and recommend the values that
should be written into config_<scope>.yml under stage.streaming_af.

The streaming-autofocus handler (streaming_focus.py) reads those YAML
values at runtime instead of relying on the Prior-1-100-percent
constants that used to be hardcoded.

Probe flow per attempt:

    Step 1 -- discover the speed property (reuses
              streaming_focus._find_speed_property).
    Step 2 -- nameplate parse:
              - '<X>mm/sec' enum -> pick slowest, convert to um/s
              - numeric allowed-values list -> pick min
              - empty allowed list + numeric default -> Prior-style
                1-100 percent: write slow="1", normal="100"
    Step 3 -- live verify: set the chosen slow value, time a 1-um
              round-trip, derive measured um/s. If measured matches
              parsed within 30 percent, accept parsed; otherwise
              record the measured value and flag the disagreement
              for the wizard UI.
    Step 4 -- viability: at slow um/s, does sweep_range_um /
              velocity * camera_fps >= MIN_FRAMES_FOR_FIT * 1.5?
              Sets the recommended `enabled` flag.
    Step 5 -- assemble + return JSON.

Wire format:

    request:  PRBSAFZ + flag string + END_MARKER
              flags: [--yaml <path>]
                     [--device <focus_device>]   (testing override)
                     [--sweep-range <um>]        (default 12.0)
                     [--camera-fps <hz>]         (default 30.0)
    response: SUCCESS:<json> on probe complete
              FAILED:<reason> on hardware error / safety abort
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from microscope_command_server.server.handlers.utils import (
    parse_flags,
    read_message_string,
)
from microscope_command_server.server.handlers.streaming_focus import (
    SPEED_PROPERTY_CANDIDATES,
    _find_speed_property,
    _str_vector_to_list,
    _try_get,
    _try_set,
    _wait_via_busy,
)
from microscope_command_server.server.probe_parsers import (
    classify_allowed_values,
    pick_recommended_values,
)

logger = logging.getLogger(__name__)


# Default sweep range used by the viability check. Mirrors the typical
# autofocus_<scope>.yml sweep_range_um for a representative objective.
# The wizard can override per-call via --sweep-range.
DEFAULT_SWEEP_RANGE_UM = 12.0

# Default camera frame rate for viability. Hamamatsu C11440 ~30 fps,
# JAI ~38 fps. Wizard can override via --camera-fps.
DEFAULT_CAMERA_FPS = 30.0

# Minimum frames we want a streaming sweep to capture for a usable
# parabolic fit. Mirrors streaming_focus.MIN_FRAMES_FOR_FIT * 1.5
# margin so the viability verdict is conservative.
VIABILITY_MIN_FRAMES = 8

# Distance (um) for the round-trip verify move.
#
# 2026-05-16: bumped from 1.0 to 20.0. At 1 um, the round-trip time is
# dominated by busy-poll overhead (5 consecutive not-busy reads + any
# debounce in the stage's reporting) rather than actual motion time --
# observed on the OWS3 Prior Z, the verify reported 6.079 um/s when the
# stage actually moves at 169 um/s for a non-blocking move (busy-poll
# floor ~150ms, 1 um @ 169 um/s = 6 ms, computed velocity = 1/0.156 =
# 6.4 um/s, falsely matching the nameplate "0.50mm/sec" interpretation).
# A 20 um move at 169 um/s takes 118 ms while busy-poll overhead stays
# at ~150 ms; that ratio (and the new peak-interval-velocity check
# below) makes the failure mode detectable.
VERIFY_MOVE_UM = 20.0

# Refuse to verify when the focus device is closer than this to either
# Z limit -- the round-trip move could cross the limit otherwise.
Z_LIMIT_BUFFER_UM = 2.0

# Tolerance for accepting nameplate vs measured velocity (fraction).
# 30 percent is generous: nameplate "0.50mm/sec" usually means a
# nominal max linear velocity, but observed um/s during a non-blocking
# 1-um move can be slowed by acceleration ramps.
VELOCITY_AGREE_TOL = 0.30

# Per-interval safeguard. If a Z-poll thread sees the stage moving at
# > RAPID_JUMP_RATIO x the endpoint-derived velocity at any point
# during the verify move, the endpoint reading is overhead-dominated
# (the stage finished quickly then sat) and the measured number is a
# lie. Mark the verify as INVALID and refuse to write the number to
# YAML. 10x is conservative -- legitimate accel/decel ramps usually
# stay within 3-4x of average; only rapid-jump behaviour produces
# 10x+ discrepancy.
VERIFY_RAPID_JUMP_RATIO = 10.0

# Polling interval for the verify-move Z-poll thread. 25ms gives
# ~20-40 samples on a typical verify move duration of 0.5-1.0 s --
# enough granularity to spot the fast-jump pattern without flooding
# the MM core.
VERIFY_POLL_INTERVAL_S = 0.025


# ----- Live verify ---------------------------------------------------


def _poll_z_during_move(
    core,
    focus_device: str,
    stop_event: threading.Event,
    samples_out: List[Tuple[float, float]],
    t_start_monotonic: float,
) -> None:
    """Background poller: append (elapsed_ms, z_um) every
    VERIFY_POLL_INTERVAL_S until stop_event is set. Best effort; an
    exception in get_position aborts the thread silently so the main
    verify path still completes."""
    while not stop_event.is_set():
        try:
            z = float(core.get_position(focus_device))
            t_ms = (time.monotonic() - t_start_monotonic) * 1000.0
            samples_out.append((t_ms, z))
        except Exception:
            pass
        if stop_event.wait(VERIFY_POLL_INTERVAL_S):
            break


def _peak_interval_velocity(samples: List[Tuple[float, float]]) -> float:
    """Compute max |dz/dt| across consecutive poll samples (um/s).

    Returns 0.0 if fewer than 2 samples (cannot derive velocity).
    Robust against single-sample glitches because we take the MAX
    across all intervals -- a single spurious commanded-Z reading
    early in the trace produces one fast interval (which is what we
    want to detect; the rapid-jump pattern looks exactly like this
    but sustained).
    """
    if len(samples) < 2:
        return 0.0
    peak = 0.0
    for i in range(1, len(samples)):
        t_prev, z_prev = samples[i - 1]
        t_now, z_now = samples[i]
        dt_s = (t_now - t_prev) / 1000.0
        if dt_s <= 0:
            continue
        v = abs(z_now - z_prev) / dt_s
        if v > peak:
            peak = v
    return peak


def _measure_velocity_um_s(
    core,
    focus_device: str,
    z_lo: Optional[float],
    z_hi: Optional[float],
    verify_move_um: float = VERIFY_MOVE_UM,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Time a round-trip move at the currently-set speed.

    Returns (endpoint_velocity_um_per_s, info). The endpoint velocity
    is move_distance / (forward + back) / 2, the historic measurement.
    `info` is a dict with:
        note               -- short timing summary string
        endpoint_um_per_s  -- same as the returned tuple [0] (or None)
        peak_interval_um_per_s -- max per-poll |dz/dt| during the move
        valid              -- bool: True if endpoint reading is
                              trusted, False if peak_interval >>
                              endpoint (overhead-dominated reading)
        invalid_reason     -- str explaining why valid=False (empty
                              when valid=True)
        verify_move_um     -- the move distance used (for the log)

    Returns endpoint_velocity = None when the move was refused for
    safety or timing was unreliable. Restores the original Z position
    before returning.
    """
    info: Dict[str, Any] = {
        "note": "",
        "endpoint_um_per_s": None,
        "peak_interval_um_per_s": None,
        "valid": False,
        "invalid_reason": "",
        "verify_move_um": float(verify_move_um),
    }

    try:
        z0 = float(core.get_position(focus_device))
    except Exception as e:
        info["note"] = f"get_position failed: {e}"
        info["invalid_reason"] = info["note"]
        return None, info

    # Z-limit safety: refuse if either direction would exit the
    # configured stage z limits.
    if z_lo is not None and (z0 - verify_move_um) < (z_lo + Z_LIMIT_BUFFER_UM):
        info["note"] = f"too close to z low limit (z={z0:.3f}, low={z_lo})"
        info["invalid_reason"] = info["note"] + "; skipped live verify"
        return None, info
    if z_hi is not None and (z0 + verify_move_um) > (z_hi - Z_LIMIT_BUFFER_UM):
        info["note"] = f"too close to z high limit (z={z0:.3f}, high={z_hi})"
        info["invalid_reason"] = info["note"] + "; skipped live verify"
        return None, info

    # Per-interval velocity sampling. A background thread polls Z
    # every VERIFY_POLL_INTERVAL_S so we can compute the peak
    # in-motion velocity, not just the endpoint slope. This is the
    # safeguard against overhead-dominated readings -- if the stage
    # actually moves at 169 um/s for a 20 um trip, the polls catch
    # it; if it really does move at 6 um/s, no interval exceeds that
    # significantly.
    poll_samples: List[Tuple[float, float]] = []
    stop_event = threading.Event()

    try:
        t0 = time.monotonic()
        poll_thread = threading.Thread(
            target=_poll_z_during_move,
            args=(core, focus_device, stop_event, poll_samples, t0),
            daemon=True,
        )
        poll_thread.start()

        # Move +verify_move_um, time it.
        core.set_position(focus_device, z0 + verify_move_um)
        _wait_via_busy(core, focus_device, target_z=z0 + verify_move_um)
        forward_s = time.monotonic() - t0
        # Move back -verify_move_um, time it.
        t1 = time.monotonic()
        core.set_position(focus_device, z0)
        _wait_via_busy(core, focus_device, target_z=z0)
        back_s = time.monotonic() - t1
    except Exception as e:
        stop_event.set()
        # Best effort to put it back.
        try:
            core.set_position(focus_device, z0)
            _wait_via_busy(core, focus_device, target_z=z0)
        except Exception:
            pass
        info["note"] = f"round-trip move failed: {e}"
        info["invalid_reason"] = info["note"]
        return None, info
    finally:
        stop_event.set()
        try:
            poll_thread.join(timeout=0.5)
        except Exception:
            pass

    # Average the two halves; reject obviously-wrong readings (e.g.
    # zero or sub-millisecond -- the busy loop has its own poll
    # interval so anything below ~10 ms is suspect).
    half = 0.5 * (forward_s + back_s)
    if half < 0.01:
        info["note"] = f"round-trip too fast ({half*1000:.1f}ms half) for reliable timing"
        info["invalid_reason"] = info["note"]
        return None, info
    endpoint_um_s = verify_move_um / half
    peak_interval_um_s = _peak_interval_velocity(poll_samples)
    info["note"] = (
        f"verify_move={verify_move_um:.1f}um, "
        f"forward {forward_s*1000:.1f}ms, back {back_s*1000:.1f}ms, "
        f"n_polls={len(poll_samples)}, "
        f"peak_interval={peak_interval_um_s:.1f}um/s"
    )
    info["endpoint_um_per_s"] = endpoint_um_s
    info["peak_interval_um_per_s"] = peak_interval_um_s

    # Safeguard: if the peak per-interval velocity is dramatically
    # larger than the endpoint reading, the stage finished fast and
    # sat for the rest of the timing window. The endpoint is the
    # busy-poll floor, not the real motion speed. Refuse to trust
    # this number for YAML.
    if (
        peak_interval_um_s > 0
        and endpoint_um_s > 0
        and peak_interval_um_s > endpoint_um_s * VERIFY_RAPID_JUMP_RATIO
    ):
        info["valid"] = False
        info["invalid_reason"] = (
            f"endpoint {endpoint_um_s:.1f}um/s but peak per-interval "
            f"{peak_interval_um_s:.1f}um/s (ratio {peak_interval_um_s/endpoint_um_s:.0f}x "
            f">= {VERIFY_RAPID_JUMP_RATIO:.0f}x). Stage finished the "
            f"{verify_move_um:.0f}um move quickly and the timing window "
            f"is dominated by busy-poll overhead, not motion. The slow_speed "
            f"property is not slowing this stage's non-blocking moves; the "
            f"endpoint reading is misleading."
        )
    else:
        info["valid"] = True

    return endpoint_um_s, info


# ----- YAML helpers (Z limits) --------------------------------------


def _read_z_limits_from_yaml(yaml_path: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Pull stage.limits.z_um.{low, high} from the active config.

    Returns (None, None) if the yaml path is missing, unreadable,
    or doesn't declare z limits. Used purely as a safety guard for
    the round-trip verify -- callers degrade gracefully on miss.
    """
    if not yaml_path:
        return None, None
    try:
        import yaml as _yaml

        with open(yaml_path, "r") as fh:
            doc = _yaml.safe_load(fh) or {}
    except Exception as e:
        logger.debug("PRBSAF: could not read z limits from %s: %s", yaml_path, e)
        return None, None
    z = ((doc.get("stage") or {}).get("limits") or {}).get("z_um") or {}
    lo = z.get("low")
    hi = z.get("high")
    return (
        float(lo) if isinstance(lo, (int, float)) else None,
        float(hi) if isinstance(hi, (int, float)) else None,
    )


# ----- Handler entry point ------------------------------------------


def handle_probe_stage_af(conn, client, hardware, settings, **kwargs):
    """Entry point for the PRBSAFZ command (wizard stage probe)."""
    addr = getattr(client, "addr", client)

    try:
        message = read_message_string(conn)
    except Exception as e:
        logger.error("PRBSAF: read_message failed: %s", e)
        try:
            conn.sendall(f"FAILED:read-message: {e}".encode())
        except Exception:
            pass
        return
    if message is None:
        message = ""

    params = parse_flags(
        message,
        [
            "--yaml",
            "--device",
            "--sweep-range",
            "--camera-fps",
            "--verify-um",
        ],
    )
    yaml_path = params.get("yaml")
    override_device = params.get("device")
    try:
        sweep_range_um = float(params.get("sweep_range", DEFAULT_SWEEP_RANGE_UM))
    except (TypeError, ValueError):
        sweep_range_um = DEFAULT_SWEEP_RANGE_UM
    try:
        camera_fps = float(params.get("camera_fps", DEFAULT_CAMERA_FPS))
    except (TypeError, ValueError):
        camera_fps = DEFAULT_CAMERA_FPS
    try:
        verify_move_um = float(params.get("verify_um", VERIFY_MOVE_UM))
    except (TypeError, ValueError):
        verify_move_um = VERIFY_MOVE_UM

    logger.info(
        "PRBSAF:request from %s yaml=%s device=%s sweep=%.1fum fps=%.1f verify=%.1fum",
        addr,
        yaml_path,
        override_device,
        sweep_range_um,
        camera_fps,
        verify_move_um,
    )

    core = hardware.core
    if override_device:
        focus_device = override_device
    else:
        try:
            focus_device = core.get_focus_device()
        except Exception as e:
            logger.error("PRBSAF: get_focus_device failed: %s", e)
            conn.sendall(f"FAILED:no-focus-device: {e}".encode())
            return
    logger.info("PRBSAF:focus device = %s", focus_device)

    result: Dict[str, Any] = {
        "focus_device": focus_device,
        "speed_property": None,
        "current_value": None,
        "allowed_values": [],
        "classification": "unknown",
        "slow_speed_value": None,
        "normal_speed_value": None,
        "slow_speed_um_per_s": None,
        "slow_speed_um_per_s_measured": None,
        # New 2026-05-16 safeguard fields. peak_interval is the
        # max per-poll velocity observed during the verify move;
        # verify_valid is False when the endpoint reading is
        # overhead-dominated (peak >> endpoint, see
        # VERIFY_RAPID_JUMP_RATIO). Wizard should refuse to write
        # the slow_speed_um_per_s into YAML when verify_valid is
        # False -- the number would lie about the stage's real
        # behaviour and break streaming AF downstream.
        "slow_speed_um_per_s_peak_interval": None,
        "verify_valid": True,
        "verify_invalid_reason": "",
        "verify_move_um": float(verify_move_um),
        "enabled": False,
        "viability_reason": "",
        "verify_note": "",
        "warnings": [],
    }

    speed_prop = _find_speed_property(core, focus_device)
    if speed_prop is None:
        # No writable speed property at all -- streaming AF cannot
        # slow the stage. The wizard should write enabled: false.
        result["viability_reason"] = (
            f"focus device '{focus_device}' has no writable speed property "
            f"(searched {list(SPEED_PROPERTY_CANDIDATES)})"
        )
        logger.info("PRBSAF:%s", result["viability_reason"])
        conn.sendall(f"SUCCESS:{json.dumps(result)}".encode())
        return

    result["speed_property"] = speed_prop
    logger.info("PRBSAF:speed property = '%s'", speed_prop)

    # Read current setting + allowed values.
    current_value = _try_get(core, focus_device, speed_prop)
    result["current_value"] = current_value
    try:
        allowed = _str_vector_to_list(core.get_allowed_property_values(focus_device, speed_prop))
    except Exception as e:
        logger.warning("PRBSAF:get_allowed_property_values failed: %s", e)
        allowed = []
    result["allowed_values"] = allowed
    logger.info("PRBSAF:current=%r allowed=%s", current_value, allowed or "(none)")

    # Classify + recommend.
    classification, parsed = classify_allowed_values(allowed)
    result["classification"] = classification
    slow_v, normal_v, slow_ums_estimate, reason = pick_recommended_values(
        classification,
        parsed,
        current_value,
    )
    result["slow_speed_value"] = slow_v
    result["normal_speed_value"] = normal_v
    result["slow_speed_um_per_s"] = slow_ums_estimate
    logger.info(
        "PRBSAF:%s -> slow=%r normal=%r ums=%s", reason, slow_v, normal_v, slow_ums_estimate
    )

    # If we couldn't pick a slow value, return early -- live verify
    # has nothing to set.
    if slow_v is None:
        result["viability_reason"] = "no recommended slow value; manual override required"
        result["warnings"].append(reason)
        conn.sendall(f"SUCCESS:{json.dumps(result)}".encode())
        return

    # ---- Live verify ----
    # Save original speed so we can restore on exit. Then set the
    # recommended slow value and time a 1-um round-trip.
    z_lo, z_hi = _read_z_limits_from_yaml(yaml_path)
    set_ok = _try_set(core, focus_device, speed_prop, slow_v)
    if not set_ok:
        result["warnings"].append(f"could not set {speed_prop}={slow_v!r} during live verify")
        result["viability_reason"] = "slow value rejected by stage"
        conn.sendall(f"SUCCESS:{json.dumps(result)}".encode())
        return

    measured_um_s, verify_info = _measure_velocity_um_s(
        core,
        focus_device,
        z_lo,
        z_hi,
        verify_move_um=verify_move_um,
    )
    result["slow_speed_um_per_s_measured"] = measured_um_s
    result["slow_speed_um_per_s_peak_interval"] = verify_info.get("peak_interval_um_per_s")
    result["verify_valid"] = bool(verify_info.get("valid", True))
    result["verify_invalid_reason"] = verify_info.get("invalid_reason", "")
    result["verify_note"] = verify_info.get("note", "")
    logger.info(
        "PRBSAF:live verify -- endpoint=%s um/s peak_interval=%s um/s valid=%s (%s)",
        measured_um_s,
        verify_info.get("peak_interval_um_per_s"),
        result["verify_valid"],
        result["verify_note"],
    )
    if not result["verify_valid"]:
        logger.warning("PRBSAF:verify INVALID -- %s", result["verify_invalid_reason"])
        result["warnings"].append(f"verify invalid: {result['verify_invalid_reason']}")

    # Restore speed to the recommended normal value (or the original
    # if we never derived a normal). _try_set is best-effort.
    restore_value = normal_v if normal_v is not None else current_value
    if restore_value is not None:
        _try_set(core, focus_device, speed_prop, str(restore_value))

    # ---- Decide final slow_speed_um_per_s ----
    # When the verify is INVALID (peak per-interval >> endpoint), the
    # endpoint number is a lie (overhead-dominated). Do NOT update
    # the recommended slow_speed_um_per_s -- it would land the rig
    # in the streaming AF rapid-jump failure mode silently. Keep the
    # nameplate estimate (if any) for the wizard UI but flag the
    # whole probe as not-viable so the wizard writes enabled: false.
    if measured_um_s is not None and result["verify_valid"]:
        if slow_ums_estimate is None:
            # We had no nameplate estimate -- trust measured.
            result["slow_speed_um_per_s"] = measured_um_s
        else:
            # Compare. If they disagree by more than tolerance,
            # prefer measured but keep both for the wizard UI.
            if slow_ums_estimate <= 0:
                result["slow_speed_um_per_s"] = measured_um_s
            else:
                ratio = abs(measured_um_s - slow_ums_estimate) / slow_ums_estimate
                if ratio > VELOCITY_AGREE_TOL:
                    result["warnings"].append(
                        f"nameplate {slow_ums_estimate:.1f} um/s vs measured "
                        f"{measured_um_s:.1f} um/s differ by {ratio*100:.0f}%; "
                        f"using measured"
                    )
                    result["slow_speed_um_per_s"] = measured_um_s
                # else: keep the nameplate value; measured agrees.
    elif measured_um_s is not None and not result["verify_valid"]:
        # The endpoint reading is overhead-dominated; the peak per-
        # interval velocity is the real stage speed and it's way
        # higher than configured. Do not feed the lie into YAML;
        # surface both numbers to the operator and disable streaming.
        peak = verify_info.get("peak_interval_um_per_s") or 0.0
        result["warnings"].append(
            f"endpoint verify {measured_um_s:.1f} um/s is overhead-dominated; "
            f"actual stage motion peaked at {peak:.1f} um/s during the move. "
            f"Slow-speed property is not slowing this stage's non-blocking "
            f"moves; streaming AF will rapid-jump. Switch to Sweep AF on "
            f"this rig or find a stage property that actually controls "
            f"continuous-motion velocity."
        )
        result["slow_speed_um_per_s"] = None  # do not write to YAML

    # ---- Viability verdict ----
    final_velocity = result["slow_speed_um_per_s"]
    if final_velocity is None or final_velocity <= 0:
        result["enabled"] = False
        result["viability_reason"] = "no usable velocity estimate"
    else:
        sweep_time_s = sweep_range_um / final_velocity
        expected_frames = sweep_time_s * camera_fps
        result["enabled"] = expected_frames >= VIABILITY_MIN_FRAMES
        result["viability_reason"] = (
            f"sweep {sweep_range_um:.1f}um at {final_velocity:.1f}um/s "
            f"= {sweep_time_s*1000:.0f}ms; ~{expected_frames:.1f} frames "
            f"at {camera_fps:.0f}fps "
            f"(need >= {VIABILITY_MIN_FRAMES})"
        )
    logger.info(
        "PRBSAF:viability -> enabled=%s (%s)", result["enabled"], result["viability_reason"]
    )

    try:
        conn.sendall(f"SUCCESS:{json.dumps(result)}".encode())
    except Exception as e:
        logger.error("PRBSAF: reply send failed: %s", e)
