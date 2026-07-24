"""Camera control command handlers.

Handles camera property queries and settings:
GETCAM, GETMODE, SETMODE, GETEXP, SETEXP, GETGAIN, SETGAIN, SETCAM,
GETBIN, SETBIN, GETCAP

These commands use the Camera ABC's per-channel capability methods.
Cameras that support per-channel control (e.g. JAI 3-CCD) return True
from supports_per_channel_exposure(); all others use unified defaults.
"""

import json
import re
import struct
import time
import logging

logger = logging.getLogger(__name__)


def handle_getcam(conn, client, hardware, settings, **kwargs):
    """Return camera name as a 32-byte null-padded string.

    Response: 32 bytes (camera name UTF-8 padded with null bytes),
    or 'ERROR:<msg>' on failure.
    """
    logger.debug("Client %s requested camera name", client.addr)
    try:
        camera_name = hardware.get_camera_name()
        # Pad or truncate to 32 bytes
        camera_name_bytes = camera_name.encode("utf-8")[:32].ljust(32, b"\x00")
        conn.sendall(camera_name_bytes)
        logger.info("Sent camera name to %s: %s", client.addr, camera_name)
    except Exception as e:
        logger.error("Failed to get camera name: %s", e)
        # Send error response (32 bytes, starts with ERROR)
        error_msg = f"ERROR:{str(e)[:23]}"
        conn.sendall(error_msg.encode("utf-8").ljust(32, b"\x00"))


def handle_getmode(conn, client, hardware, settings, **kwargs):
    """Return exposure/gain mode flags (individual vs unified).

    Gain is always reported as unified (0) since individual gain mode
    is no longer used. R/B analog gains work in unified mode.

    Response: 16-byte padded string.
    - Per-channel camera: 'JAI_EXP:<0|1|U>_GAIN:0'
    - Non-per-channel / generic: 'UNIFIED_________'
    - Error: 'ERROR:<msg>' (16 bytes padded)
    """
    logger.debug("Client %s requested camera mode flags", client.addr)
    try:
        cam = hardware.camera
        if cam.supports_per_channel_exposure():
            exp_individual = cam.is_individual_exposure_enabled()
            mode_str = f"JAI_EXP:{1 if exp_individual else 0}_GAIN:0"
            conn.sendall(mode_str.encode("utf-8").ljust(16, b"\x00"))
            logger.info(
                "Sent per-channel mode flags: exp_ind=%s, gain_ind=false",
                exp_individual,
            )
        else:
            conn.sendall(b"UNIFIED_________")
            logger.info("Camera does not support per-channel mode - sent UNIFIED")
    except Exception as e:
        logger.error("Failed to get camera mode: %s", e)
        error_msg = f"ERROR:{str(e)[:8]}"
        conn.sendall(error_msg.encode("utf-8").ljust(16, b"\x00"))


def handle_setmode(conn, client, hardware, settings, **kwargs):
    """Set exposure/gain mode flags.

    Sets exposure mode (individual or unified) via Camera ABC methods.
    Gain mode byte is accepted but ignored - gain is always unified.
    R/B analog gains work in unified mode via set_rb_analog_gains().

    Protocol: 2 bytes [exp_mode, gain_mode]
      exp_mode:  1 = individual (R,G,B separate), 0 = unified
      gain_mode: ignored (always unified), logged if True requested

    Response: 'ACK_____' on success, 'ERR_NSUP' if individual mode not
      supported, 'ERR_MODE' on failure.
    """
    logger.debug("Client %s requested to set camera mode", client.addr)
    try:
        # Read 2 bytes: [exp_mode, gain_mode]
        mode_bytes = conn.recv(2)
        if len(mode_bytes) != 2:
            raise ValueError("Expected 2 bytes for mode flags")

        exp_individual = mode_bytes[0] == 1
        gain_individual = mode_bytes[1] == 1

        if gain_individual:
            logger.warning(
                "Individual gain mode requested but ignored - "
                "gain is always unified. Use R/B analog gains instead."
            )

        logger.info(
            "Setting mode: exp_individual=%s, gain_individual=false(forced)",
            exp_individual,
        )

        cam = hardware.camera

        if exp_individual and not cam.supports_per_channel_exposure():
            conn.sendall(b"ERR_NSUP")
            logger.error(
                "Individual exposure mode requested but camera %s "
                "does not support per-channel exposure.",
                cam.get_name(),
            )
            return

        # Safety net: stop any active streaming before changing camera properties
        stopped = False
        try:
            if hardware.core.is_sequence_running():
                logger.warning("Core sequence running during SETMODE - auto-stopping")
                hardware.core.stop_sequence_acquisition()
                stopped = True
                time.sleep(0.2)
        except Exception as seq_err:
            logger.debug("Could not check/stop sequence: %s", seq_err)
        try:
            if hardware.studio is not None and hardware.studio.live().is_live_mode_on():
                logger.warning("MM Studio live mode on during SETMODE - auto-stopping")
                hardware.studio.live().set_live_mode(False)
                stopped = True
                time.sleep(0.2)
        except Exception as live_err:
            logger.debug("Could not check/stop studio live: %s", live_err)

        if exp_individual:
            cam.enable_individual_exposure()
        else:
            cam.disable_individual_exposure()

        # Always ensure gain is unified
        cam.disable_individual_gain()

        conn.sendall(b"ACK_____")
        if stopped:
            logger.info("Camera mode set successfully (auto-stopped streaming first)")
        else:
            logger.info("Camera mode set successfully")
    except Exception as e:
        logger.error("Failed to set camera mode: %s", e)
        conn.sendall(b"ERR_MODE")


def handle_getexp(conn, client, hardware, settings, **kwargs):
    """Return exposure values (unified or per-channel RGB).

    Per-channel camera with individual exposures: 4 floats (all, R, G, B) = 16 bytes.
    Unified / generic: 1 float = 4 bytes.
    Error: 1 float = -1.0.
    """
    logger.debug("Client %s requested exposure values", client.addr)
    try:
        cam = hardware.camera

        if cam.supports_per_channel_exposure() and cam.is_individual_exposure_enabled():
            exposures = cam.get_channel_exposures()
            all_exp = hardware.get_exposure()
            response = struct.pack(
                "!ffff",
                float(all_exp),
                float(exposures["red"]),
                float(exposures["green"]),
                float(exposures["blue"]),
            )
            conn.sendall(response)
            logger.info(
                "Sent per-channel exposures: all=%s, R=%s, G=%s, B=%s",
                all_exp,
                exposures["red"],
                exposures["green"],
                exposures["blue"],
            )
        else:
            exposure = hardware.get_exposure()
            response = struct.pack("!f", float(exposure))
            conn.sendall(response)
            logger.info("Sent unified exposure: %s", exposure)
    except Exception as e:
        logger.error("Failed to get exposure: %s", e)
        conn.sendall(struct.pack("!f", -1.0))


def handle_setexp(conn, client, hardware, settings, **kwargs):
    """Set exposure values.

    count=1: Sets unified exposure (any camera).
    count>=3: Sets per-channel R,G,B exposures via Camera ABC.
              Falls back to unified (green channel) if not supported.

    Protocol: 1 count byte + (count * 4) bytes of big-endian floats.
    Response: 'ACK_____' on success, 'ERR_EXPO' on failure.
    """
    logger.debug("Client %s requested to set exposure", client.addr)
    try:
        count_byte = conn.recv(1)
        count = count_byte[0]
        logger.debug("SETEXP: expecting %d exposure values", count)

        float_data = conn.recv(count * 4)
        if len(float_data) != count * 4:
            raise ValueError(f"Expected {count * 4} bytes, got {len(float_data)}")

        exposures = struct.unpack(f"!{'f' * count}", float_data)
        logger.info("Setting exposures: %s", exposures)

        cam = hardware.camera

        if count == 1:
            hardware.set_exposure(exposures[0])
            logger.info("Set unified exposure to %s ms", exposures[0])
        elif count >= 3:
            if cam.supports_per_channel_exposure():
                r, g, b = exposures[0], exposures[1], exposures[2]
                # SETEXP is the interactive Live Viewer path (Camera-tab exposure
                # spinner and PPM angle clicks), which streams in CONTINUOUS mode.
                # Snap BLUE out of the JAI contamination trigger window HERE:
                # unlike apply_settings, the thin set_channel_exposures does not,
                # so a per-angle exposure whose blue lands in (4.0, 5.6) ms would
                # otherwise show the half-frame contamination bar / freeze in live
                # view. Snap-mode acquisition is unaffected (it uses apply_settings
                # and is clean at any value), so this does not change acquired data.
                snapper = getattr(cam, "snap_exposures_out_of_trigger_window", None)
                if snapper is not None:
                    r, g, b, snapped = snapper(r, g, b)
                    if snapped:
                        logger.warning(
                            "SETEXP: blue exposure landed in the JAI contamination "
                            "trigger window; snapped to a safe edge (R=%.4f G=%.4f "
                            "B=%.4f ms) to keep live view clean; WB ratio preserved. "
                            "See claude-reports/2026-07-24_jai-live-exposure-snap-gap.md.",
                            r,
                            g,
                            b,
                        )
                cam.set_channel_exposures(
                    red=r,
                    green=g,
                    blue=b,
                    auto_enable=True,
                )
                logger.info(
                    "Set per-channel exposures: R=%s, G=%s, B=%s",
                    r,
                    g,
                    b,
                )
            else:
                # Fall back to unified using green channel value
                hardware.set_exposure(exposures[1])
                logger.warning(
                    "Per-channel exposures requested but camera %s does not "
                    "support per-channel mode - using green (%.2f ms) as unified",
                    cam.get_name(),
                    exposures[1],
                )

        conn.sendall(b"ACK_____")
    except Exception as e:
        logger.error("Failed to set exposure: %s", e)
        conn.sendall(b"ERR_EXPO")


def handle_getgain(conn, client, hardware, settings, **kwargs):
    """Return gain values.

    Always returns 3 floats: [unified_gain, analog_red, analog_blue] = 12 bytes.
    Non-per-channel cameras return (1.0, 1.0, 1.0) via Camera ABC defaults.
    Error: returns (-1.0, -1.0, -1.0).
    """
    logger.debug("Client %s requested gain values", client.addr)
    try:
        cam = hardware.camera
        unified = cam.get_unified_gain()
        rb_gains = cam.get_rb_analog_gains()
        response = struct.pack(
            "!fff",
            float(unified),
            float(rb_gains.get("analog_red", 1.0)),
            float(rb_gains.get("analog_blue", 1.0)),
        )
        conn.sendall(response)
        logger.info(
            "Sent gains: unified=%s, analog_red=%s, analog_blue=%s",
            unified,
            rb_gains.get("analog_red", 1.0),
            rb_gains.get("analog_blue", 1.0),
        )
    except Exception as e:
        logger.error("Failed to get gain: %s", e)
        conn.sendall(struct.pack("!fff", -1.0, -1.0, -1.0))


def handle_setgain(conn, client, hardware, settings, **kwargs):
    """Set gain values.

    count=1: Sets unified gain (range depends on camera).
    count=3: Sets [unified_gain, analog_red, analog_blue].
             Does NOT enable individual gain mode.

    Protocol: 1 count byte + (count * 4) bytes floats.
    Response: 'ACK_____' or 'ERR_GAIN'.
    """
    logger.debug("Client %s requested to set gain", client.addr)
    try:
        count_byte = conn.recv(1)
        count = count_byte[0]
        logger.debug("SETGAIN: expecting %d gain values", count)

        float_data = conn.recv(count * 4)
        if len(float_data) != count * 4:
            raise ValueError(f"Expected {count * 4} bytes, got {len(float_data)}")

        gains = struct.unpack(f"!{'f' * count}", float_data)
        logger.info("Setting gains: %s", gains)

        cam = hardware.camera

        # Stop any active streaming before changing gain properties
        try:
            if hardware.core.is_sequence_running():
                hardware.core.stop_sequence_acquisition()
                time.sleep(0.2)
        except Exception:
            pass

        if count == 1:
            cam.set_unified_gain(gains[0])
            logger.info("Set unified gain: %s", gains[0])
        elif count >= 3:
            cam.set_unified_gain(gains[0])
            cam.set_rb_analog_gains(analog_red=gains[1], analog_blue=gains[2])
            logger.info(
                "Set gains: unified=%s, analog_red=%s, analog_blue=%s",
                gains[0],
                gains[1],
                gains[2],
            )

        conn.sendall(b"ACK_____")
    except Exception as e:
        logger.error("Failed to set gain: %s", e)
        conn.sendall(b"ERR_GAIN")


def handle_setcam(conn, client, hardware, settings, **kwargs):
    """Set camera mode, exposures, and gains atomically in one command.

    Replaces the sequence SETMODE -> SETEXP -> SETGAIN with a single
    round-trip. Stops streaming once, applies all settings, responds once.

    Protocol:
        1 byte:  exp_mode (1=individual, 0=unified)
        1 byte:  exposure_count (1=unified, 3=per-channel)
        N*4 bytes: exposures (big-endian floats)
        1 byte:  gain_count (1=unified only, 3=unified+analog_r+analog_b)
        N*4 bytes: gains (big-endian floats)

    Response: 'ACK_____' on success, 'ERR_SETC' on failure.
    """
    logger.debug("Client %s requested SETCAM (atomic camera settings)", client.addr)
    try:
        # Read header: exp_mode (1) + exp_count (1)
        header = conn.recv(2)
        if len(header) != 2:
            raise ValueError("Expected 2-byte SETCAM header")

        exp_individual = header[0] == 1
        exp_count = header[1]

        # Read exposures
        exp_data = conn.recv(exp_count * 4)
        if len(exp_data) != exp_count * 4:
            raise ValueError(f"Expected {exp_count * 4} exposure bytes, got {len(exp_data)}")
        exposures = struct.unpack(f"!{'f' * exp_count}", exp_data)

        # Read gain count + gains
        gain_count_byte = conn.recv(1)
        gain_count = gain_count_byte[0]
        gain_data = conn.recv(gain_count * 4)
        if len(gain_data) != gain_count * 4:
            raise ValueError(f"Expected {gain_count * 4} gain bytes, got {len(gain_data)}")
        gains = struct.unpack(f"!{'f' * gain_count}", gain_data)

        logger.info(
            "SETCAM: mode=%s, exposures=%s, gains=%s",
            "individual" if exp_individual else "unified",
            exposures,
            gains,
        )

        cam = hardware.camera

        # Stop streaming ONCE before all changes
        stopped = False
        try:
            if hardware.core.is_sequence_running():
                hardware.core.stop_sequence_acquisition()
                stopped = True
                time.sleep(0.1)
        except Exception:
            pass
        try:
            if hardware.studio is not None and hardware.studio.live().is_live_mode_on():
                hardware.studio.live().set_live_mode(False)
                stopped = True
                time.sleep(0.1)
        except Exception:
            pass

        if exp_individual and not cam.supports_per_channel_exposure():
            conn.sendall(b"ERR_NSUP")
            logger.error("Individual exposure not supported by %s", cam.get_name())
            return

        # Route through JAICamera.apply_settings when available so the
        # blue-window contamination mitigation (jai_camera.py 2026-05-06)
        # and the equal-channel fall-through both fire on EVERY calibration
        # application path -- not just direct apply_settings callers. Prior
        # versions of this handler called the property setters one-by-one,
        # which silently bypassed the mitigations and left users seeing the
        # bottom-of-frame contamination band after applying a WB preset
        # whose blue exposure landed in the trigger window (e.g. 4.32 ms).
        # See claude-reports/2026-05-06_jai-contamination-bar-internal.md.
        if hasattr(cam, "apply_settings"):
            if exp_count >= 3:
                exposures_dict = {
                    "r": exposures[0],
                    "g": exposures[1],
                    "b": exposures[2],
                }
            else:
                exposures_dict = {"all": exposures[0]}

            unified_gain = gains[0] if gain_count >= 1 else 1.0
            analog_red = gains[1] if gain_count >= 3 else 1.0
            analog_blue = gains[2] if gain_count >= 3 else 1.0

            cam.apply_settings(
                exposures=exposures_dict,
                unified_gain=unified_gain,
                analog_red=analog_red,
                analog_blue=analog_blue,
                individual_exposure=exp_individual,
            )
            logger.info(
                "SETCAM via apply_settings: mode=%s, exposures=%s, "
                "unified_gain=%.2f, aR=%.3f, aB=%.3f",
                "individual" if exp_individual else "unified",
                exposures_dict,
                unified_gain,
                analog_red,
                analog_blue,
            )
        else:
            # Non-JAI camera fallback: original direct-property path. No
            # contamination mitigation needed (mitigation is JAI-specific).
            if exp_individual:
                cam.enable_individual_exposure()
            else:
                cam.disable_individual_exposure()
            cam.disable_individual_gain()

            if exp_count == 1:
                hardware.set_exposure(exposures[0])
                logger.info("SETCAM: unified exposure %.3fms", exposures[0])
            elif exp_count >= 3:
                if cam.supports_per_channel_exposure():
                    cam.set_channel_exposures(
                        red=exposures[0],
                        green=exposures[1],
                        blue=exposures[2],
                        auto_enable=False,
                    )
                    logger.info(
                        "SETCAM: per-channel R=%.3f G=%.3f B=%.3fms",
                        exposures[0],
                        exposures[1],
                        exposures[2],
                    )
                else:
                    hardware.set_exposure(exposures[1])
                    logger.info("SETCAM: fallback unified %.3fms (green)", exposures[1])

            if gain_count == 1:
                cam.set_unified_gain(gains[0])
                logger.info("SETCAM: unified gain %.2f", gains[0])
            elif gain_count >= 3:
                cam.set_unified_gain(gains[0])
                # NB: JAICamera wrapper uses analog_red/analog_blue kwargs
                # (matching the inner set_rb_analog_gains in jai/properties.py
                # renamed to analog_* to avoid ambiguity with camera-role
                # 'red channel' vs analog gain register). Earlier versions
                # of this handler accidentally used red=/blue= which broke
                # all per-channel WB presets with ERR_SETC.
                cam.set_rb_analog_gains(analog_red=gains[1], analog_blue=gains[2])
                logger.info(
                    "SETCAM: unified=%.2f, aR=%.3f, aB=%.3f",
                    gains[0],
                    gains[1],
                    gains[2],
                )

        conn.sendall(b"ACK_____")
        logger.info(
            "SETCAM complete (streaming was %s)",
            "stopped" if stopped else "not running",
        )

    except Exception as e:
        logger.error("SETCAM failed: %s", e)
        conn.sendall(b"ERR_SETC")


# --- Binning (Camera Control v2 phase 1) -------------------------------


def handle_getbin(conn, client, hardware, settings, **kwargs):
    """Return available binning factors and the current factor.

    Response shape (variable length):
        1 byte  : count N of available factors
        N bytes : each factor as an unsigned byte (1, 2, 4, 8, ...)
        1 byte  : current binning factor

    Cameras with no binning support report N=1, factors=[1], current=1.

    Errors return a single 'ERROR:<msg>' length-padded to 16 bytes so the
    Java client can distinguish from the success path by reading the
    leading byte (which would be a valid count for binning, but cameras
    with N>16 are extremely rare; the Java client checks for the ASCII
    prefix 'E' too).
    """
    logger.debug("Client %s requested binning", client.addr)
    try:
        cam = hardware.camera
        available = cam.get_available_binnings() or [1]
        # Clamp to byte range -- nobody bins by more than 255 in practice.
        available = [int(v) & 0xFF for v in available if 1 <= int(v) <= 255]
        if not available:
            available = [1]
        current = int(cam.get_binning()) & 0xFF
        if current < 1:
            current = 1
        payload = bytes([len(available)]) + bytes(available) + bytes([current])
        conn.sendall(payload)
        logger.info("GETBIN: available=%s current=%d", available, current)
    except Exception as e:
        logger.error("GETBIN failed: %s", e)
        msg = f"ERROR:{str(e)[:9]}".encode("utf-8").ljust(16, b"\x00")
        conn.sendall(msg)


def handle_setbin(conn, client, hardware, settings, **kwargs):
    """Set the camera binning factor.

    Payload: 1 byte (unsigned) -- the binning factor to apply.
    Response: 'ACK_____' (8 bytes) on success, 'ERR_SETB' on failure.

    Wrapped here in a stop-if-streaming pattern matching SETCAM, since
    binning changes are typically rejected by drivers while a sequence
    acquisition is running.
    """
    logger.debug("Client %s requested set-binning", client.addr)
    try:
        data = conn.recv(1)
        if len(data) != 1:
            raise ValueError(f"Expected 1-byte payload, got {len(data)}")
        value = int.from_bytes(data, "big")
        if value < 1:
            raise ValueError(f"Binning value must be >= 1, got {value}")

        core = hardware.core
        stopped = False
        try:
            if core.is_sequence_running():
                core.stop_sequence_acquisition()
                stopped = True
        except Exception as e:
            logger.debug("SETBIN stop-streaming probe failed: %s", e)

        hardware.camera.set_binning(value)

        conn.sendall(b"ACK_____")
        logger.info(
            "SETBIN: applied binning=%d (streaming was %s)",
            value,
            "stopped" if stopped else "not running",
        )
    except Exception as e:
        logger.error("SETBIN failed: %s", e)
        conn.sendall(b"ERR_SETB")


# --- Camera Control v2 phase 2: capability query ---


def _resolve_profile(profile_name, profiles):
    """Resolve a profile name to its config dict.

    Accepts trailing "_<counter>" suffixes (e.g. Brightfield_10x_3) the
    same way apply_mode_setup does, so the Java client can pass exactly
    what it would pass to APPLYPR.
    """
    if not profile_name:
        return None
    if profile_name in profiles:
        return profiles[profile_name]
    stripped = re.sub(r"_(\d+)$", "", profile_name)
    return profiles.get(stripped)


def _modality_section_for_profile(profile_cfg, modalities):
    """Find the modality section a profile points at."""
    if not isinstance(profile_cfg, dict):
        return None, None
    mod_name = profile_cfg.get("modality")
    if not mod_name:
        return None, None
    return mod_name, modalities.get(mod_name)


def _build_illumination_descriptors(hardware, modalities):
    """Walk every modality and yield {label, device, power_range, current_power, is_on}.

    Dedups by device name. Each illumination object is built via the same
    _build_illumination_from_config helper apply_mode_setup uses, so the
    Java side sees exactly what the workflow would activate.
    """
    descriptors = []
    seen = set()

    def _emit(source):
        if source is None:
            return
        device = getattr(source, "_device", None) or getattr(source, "_label", None) or "<unknown>"
        if device in seen:
            return
        seen.add(device)
        try:
            power_range = list(source.get_power_range())
        except Exception:
            power_range = [0.0, 0.0]
        try:
            current_power = float(source.get_power())
        except Exception:
            current_power = 0.0
        try:
            is_on = bool(source.is_on())
        except Exception:
            is_on = False

        # value_type tells the UI which widget to render. "binary" means
        # the source has only the State property (state_prop == intensity_prop)
        # and the only valid powers are 0 and max_intensity -- a checkbox /
        # toggle is appropriate, NOT a free-form spinner. "continuous" means
        # any value in power_range is valid (a spinner is right). "discrete"
        # is reserved for future use when an MM device exposes a small
        # enumerated intensity property (radio button list).
        try:
            is_binary = bool(getattr(source, "_is_binary", lambda: False)())
        except Exception:
            is_binary = False
        value_type = "binary" if is_binary else "continuous"

        descriptors.append(
            {
                "label": getattr(source, "_label", device),
                "device": device,
                "power_range": power_range,
                "current_power": current_power,
                "is_on": is_on,
                "value_type": value_type,
            }
        )

    if hardware._illumination is not None:
        _emit(hardware._illumination)

    for mod_name, mod_config in (modalities or {}).items():
        if not isinstance(mod_config, dict):
            continue
        try:
            source = hardware._build_illumination_from_config(mod_config)
        except Exception as e:
            logger.debug("GETCAP: skipping illumination for modality '%s': %s", mod_name, e)
            continue
        _emit(source)

    return descriptors


def _detect_camera_type(hardware):
    """Coarse classification used by Camera Control v2 to pick UI branches."""
    cam = hardware.camera
    cls_name = type(cam).__name__.lower()
    if "jai" in cls_name:
        return "jai"
    if "laserscanning" in cls_name or "lsm" in cls_name:
        return "laser_scanning"
    name = getattr(cam, "_name", "") or ""
    if "hamamatsu" in name.lower():
        return "hamamatsu"
    return "generic"


def _modality_channels(mod_config, modality_name):
    """Return a list of channel descriptors, or None for non-channel modalities.

    A channel descriptor is the dict the modality config already stores
    (channel id, exposure_ms, intensity device-property hints) plus an
    "id" field copied from the channel key when missing. Different
    configs structure channels two ways:

    1. mod_config['channels'] is a dict keyed by channel id (BF+IF,
       Fluorescence with full library)
    2. mod_config['channels'] is a list of strings (channel ids only,
       library lives elsewhere -- e.g. profile-keyed)

    For (2) the client can fall back to the modality library lookup it
    already does today; this function just returns the IDs.
    """
    raw = mod_config.get("channels") if isinstance(mod_config, dict) else None
    if raw is None:
        return None
    if isinstance(raw, dict):
        out = []
        for cid, cfg in raw.items():
            entry = {"id": cid}
            if isinstance(cfg, dict):
                if "exposure_ms" in cfg:
                    entry["exposure_ms"] = cfg["exposure_ms"]
                if "intensity_property" in cfg:
                    entry["intensity_property"] = cfg["intensity_property"]
                if "default_intensity" in cfg:
                    entry["default_intensity"] = cfg["default_intensity"]
            out.append(entry)
        return out
    if isinstance(raw, list):
        return [{"id": str(cid)} for cid in raw]
    return None


def _modality_rotation_angles(mod_config):
    """Return a list of angle ticks, or None when modality is single-angle."""
    raw = mod_config.get("rotation_angles") if isinstance(mod_config, dict) else None
    if not isinstance(raw, list) or not raw:
        return None
    angles = []
    for entry in raw:
        if isinstance(entry, dict) and "tick" in entry:
            try:
                angles.append(float(entry["tick"]))
            except (TypeError, ValueError):
                continue
        else:
            try:
                angles.append(float(entry))
            except (TypeError, ValueError):
                continue
    return angles or None


def handle_getcap(conn, client, hardware, settings, **kwargs):
    """Return a JSON capability descriptor for Camera Control v2.

    Payload: 32-byte null-padded UTF-8 profile name (or empty).
        - empty / blank   -> describe current state, using
                             hardware._active_profile if known
        - non-empty       -> describe the profile (without applying it),
                             so the dialog can render the controls that
                             would be relevant after Apply.

    Response: 4-byte big-endian length + UTF-8 JSON. On error returns
    a 4-byte 0 length so the Java reader doesn't hang.
    """
    logger.debug("Client %s requested GETCAP", client.addr)
    try:
        data = conn.recv(32)
        requested = data.rstrip(b"\x00").decode("utf-8", errors="replace").strip() if data else ""

        profiles = settings.get("acquisition_profiles", {}) or {}
        modalities = settings.get("modalities", {}) or {}

        if requested:
            profile_name = requested
        else:
            profile_name = hardware._active_profile or ""

        profile_cfg = _resolve_profile(profile_name, profiles)
        modality_name, mod_config = _modality_section_for_profile(profile_cfg, modalities)

        cam = hardware.camera
        try:
            available_binnings = list(cam.get_available_binnings())
        except Exception:
            available_binnings = [1]
        try:
            current_binning = int(cam.get_binning())
        except Exception:
            current_binning = 1
        try:
            exp_min = float(cam.get_min_exposure_ms())
            exp_max = float(cam.get_max_exposure_ms())
        except Exception:
            exp_min, exp_max = 0.01, 10000.0
        try:
            gain_range = cam.get_gain_range()
            gain_range = list(gain_range) if gain_range is not None else None
        except Exception:
            gain_range = None

        camera_block = {
            "name": (
                cam.get_name() if hasattr(cam, "get_name") else getattr(cam, "_name", "<unknown>")
            ),
            "type": _detect_camera_type(hardware),
            "supports_per_channel_exposure": bool(cam.supports_per_channel_exposure()),
            "supports_hardware_white_balance": bool(cam.supports_hardware_white_balance()),
            "available_binnings": available_binnings,
            "current_binning": current_binning,
            "exposure_range_ms": [exp_min, exp_max],
            "gain_range": gain_range,
        }

        if mod_config is not None:
            channels = _modality_channels(mod_config, modality_name)
            angles = _modality_rotation_angles(mod_config)
            modality_block = {
                "name": modality_name,
                "default_wb_mode": mod_config.get("default_wb_mode", "off"),
                "is_multi_angle": angles is not None and len(angles) > 1,
                "channels": channels,
                "rotation_angles": angles,
            }
        else:
            modality_block = {
                "name": modality_name,
                "default_wb_mode": "off",
                "is_multi_angle": False,
                "channels": None,
                "rotation_angles": None,
            }

        capabilities = {
            "camera": camera_block,
            "illumination": _build_illumination_descriptors(hardware, modalities),
            "modality": modality_block,
            "active_profile": hardware._active_profile,
        }

        payload = json.dumps(capabilities).encode("utf-8")
        conn.sendall(struct.pack(">I", len(payload)))
        conn.sendall(payload)
        logger.info(
            "GETCAP: profile='%s' (active='%s') -> %d-byte JSON, %d illumination(s), modality=%s",
            requested or "(current)",
            hardware._active_profile,
            len(payload),
            len(capabilities["illumination"]),
            modality_name,
        )
    except Exception as e:
        logger.error("GETCAP failed: %s", e, exc_info=True)
        try:
            conn.sendall(struct.pack(">I", 0))
        except Exception:
            pass
