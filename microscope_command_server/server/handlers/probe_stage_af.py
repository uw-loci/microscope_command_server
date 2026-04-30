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
import time
from typing import Any, Dict, Optional, Tuple

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
    PRIOR_FALLBACK_NORMAL,
    PRIOR_FALLBACK_SLOW,
    PRIOR_FALLBACK_SLOW_UM_S,
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
VERIFY_MOVE_UM = 1.0

# Refuse to verify when the focus device is closer than this to either
# Z limit -- the round-trip move could cross the limit otherwise.
Z_LIMIT_BUFFER_UM = 2.0

# Tolerance for accepting nameplate vs measured velocity (fraction).
# 30 percent is generous: nameplate "0.50mm/sec" usually means a
# nominal max linear velocity, but observed um/s during a non-blocking
# 1-um move can be slowed by acceleration ramps.
VELOCITY_AGREE_TOL = 0.30


# ----- Live verify ---------------------------------------------------


def _measure_velocity_um_s(
    core,
    focus_device: str,
    z_lo: Optional[float],
    z_hi: Optional[float],
) -> Tuple[Optional[float], str]:
    """Time a 1-um round-trip move at the currently-set speed.

    Returns (measured_um_per_s, note). `measured_um_per_s` is None if
    the move was refused for safety or the timing was unreliable.
    Restores the original Z position before returning.
    """
    try:
        z0 = float(core.get_position(focus_device))
    except Exception as e:
        return None, f"get_position failed: {e}"

    # Z-limit safety: refuse if either direction would exit the
    # configured stage z limits.
    if z_lo is not None and (z0 - VERIFY_MOVE_UM) < (z_lo + Z_LIMIT_BUFFER_UM):
        return None, (f"too close to z low limit (z={z0:.3f}, "
                      f"low={z_lo}); skipped live verify")
    if z_hi is not None and (z0 + VERIFY_MOVE_UM) > (z_hi - Z_LIMIT_BUFFER_UM):
        return None, (f"too close to z high limit (z={z0:.3f}, "
                      f"high={z_hi}); skipped live verify")

    try:
        # Move +1 um, time it.
        t0 = time.monotonic()
        core.set_position(focus_device, z0 + VERIFY_MOVE_UM)
        _wait_via_busy(core, focus_device)
        forward_s = time.monotonic() - t0
        # Move back -1 um, time it.
        t1 = time.monotonic()
        core.set_position(focus_device, z0)
        _wait_via_busy(core, focus_device)
        back_s = time.monotonic() - t1
    except Exception as e:
        # Best effort to put it back.
        try:
            core.set_position(focus_device, z0)
            _wait_via_busy(core, focus_device)
        except Exception:
            pass
        return None, f"round-trip move failed: {e}"

    # Average the two halves; reject obviously-wrong readings (e.g.
    # zero or sub-millisecond -- the busy loop has its own poll
    # interval so anything below ~10 ms is suspect).
    half = 0.5 * (forward_s + back_s)
    if half < 0.01:
        return None, (f"round-trip too fast ({half*1000:.1f}ms half) "
                      f"for reliable timing")
    measured_um_s = VERIFY_MOVE_UM / half
    return measured_um_s, (f"forward {forward_s*1000:.1f}ms, "
                           f"back {back_s*1000:.1f}ms")


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
    z = (((doc.get("stage") or {}).get("limits") or {}).get("z_um") or {})
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

    params = parse_flags(message, [
        "--yaml", "--device", "--sweep-range", "--camera-fps",
    ])
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

    logger.info("PRBSAF:request from %s yaml=%s device=%s sweep=%.1fum fps=%.1f",
                addr, yaml_path, override_device, sweep_range_um, camera_fps)

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
        allowed = _str_vector_to_list(
            core.get_allowed_property_values(focus_device, speed_prop)
        )
    except Exception as e:
        logger.warning("PRBSAF:get_allowed_property_values failed: %s", e)
        allowed = []
    result["allowed_values"] = allowed
    logger.info("PRBSAF:current=%r allowed=%s", current_value, allowed or "(none)")

    # Classify + recommend.
    classification, parsed = classify_allowed_values(allowed)
    result["classification"] = classification
    slow_v, normal_v, slow_ums_estimate, reason = pick_recommended_values(
        classification, parsed, current_value,
    )
    result["slow_speed_value"] = slow_v
    result["normal_speed_value"] = normal_v
    result["slow_speed_um_per_s"] = slow_ums_estimate
    logger.info("PRBSAF:%s -> slow=%r normal=%r ums=%s",
                reason, slow_v, normal_v, slow_ums_estimate)

    # If we couldn't pick a slow value, return early -- live verify
    # has nothing to set.
    if slow_v is None:
        result["viability_reason"] = (
            "no recommended slow value; manual override required"
        )
        result["warnings"].append(reason)
        conn.sendall(f"SUCCESS:{json.dumps(result)}".encode())
        return

    # ---- Live verify ----
    # Save original speed so we can restore on exit. Then set the
    # recommended slow value and time a 1-um round-trip.
    z_lo, z_hi = _read_z_limits_from_yaml(yaml_path)
    set_ok = _try_set(core, focus_device, speed_prop, slow_v)
    if not set_ok:
        result["warnings"].append(
            f"could not set {speed_prop}={slow_v!r} during live verify"
        )
        result["viability_reason"] = "slow value rejected by stage"
        conn.sendall(f"SUCCESS:{json.dumps(result)}".encode())
        return

    measured_um_s, verify_note = _measure_velocity_um_s(
        core, focus_device, z_lo, z_hi,
    )
    result["slow_speed_um_per_s_measured"] = measured_um_s
    result["verify_note"] = verify_note
    logger.info("PRBSAF:live verify -- measured=%s um/s (%s)",
                measured_um_s, verify_note)

    # Restore speed to the recommended normal value (or the original
    # if we never derived a normal). _try_set is best-effort.
    restore_value = normal_v if normal_v is not None else current_value
    if restore_value is not None:
        _try_set(core, focus_device, speed_prop, str(restore_value))

    # ---- Decide final slow_speed_um_per_s ----
    if measured_um_s is not None:
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
    logger.info("PRBSAF:viability -> enabled=%s (%s)",
                result["enabled"], result["viability_reason"])

    try:
        conn.sendall(f"SUCCESS:{json.dumps(result)}".encode())
    except Exception as e:
        logger.error("PRBSAF: reply send failed: %s", e)
