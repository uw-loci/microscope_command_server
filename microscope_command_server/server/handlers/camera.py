"""Camera control command handlers.

Handles camera property queries and settings:
GETCAM, GETMODE, SETMODE, GETEXP, SETEXP, GETGAIN, SETGAIN, SETCAM

These commands use the Camera ABC's per-channel capability methods.
Cameras that support per-channel control (e.g. JAI 3-CCD) return True
from supports_per_channel_exposure(); all others use unified defaults.
"""

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
                all_exp, exposures["red"], exposures["green"], exposures["blue"],
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
            raise ValueError(
                f"Expected {count * 4} bytes, got {len(float_data)}"
            )

        exposures = struct.unpack(f"!{'f' * count}", float_data)
        logger.info("Setting exposures: %s", exposures)

        cam = hardware.camera

        if count == 1:
            hardware.set_exposure(exposures[0])
            logger.info("Set unified exposure to %s ms", exposures[0])
        elif count >= 3:
            if cam.supports_per_channel_exposure():
                cam.set_channel_exposures(
                    red=exposures[0],
                    green=exposures[1],
                    blue=exposures[2],
                    auto_enable=True,
                )
                logger.info(
                    "Set per-channel exposures: R=%s, G=%s, B=%s",
                    exposures[0], exposures[1], exposures[2],
                )
            else:
                # Fall back to unified using green channel value
                hardware.set_exposure(exposures[1])
                logger.warning(
                    "Per-channel exposures requested but camera %s does not "
                    "support per-channel mode - using green (%.2f ms) as unified",
                    cam.get_name(), exposures[1],
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
            unified, rb_gains.get("analog_red", 1.0),
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
            raise ValueError(
                f"Expected {count * 4} bytes, got {len(float_data)}"
            )

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
                gains[0], gains[1], gains[2],
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
            exposures, gains,
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

        # 1. Set mode
        if exp_individual:
            if not cam.supports_per_channel_exposure():
                conn.sendall(b"ERR_NSUP")
                logger.error("Individual exposure not supported by %s", cam.get_name())
                return
            cam.enable_individual_exposure()
        else:
            cam.disable_individual_exposure()
        cam.disable_individual_gain()

        # 2. Set exposures
        if exp_count == 1:
            hardware.set_exposure(exposures[0])
            logger.info("SETCAM: unified exposure %.3fms", exposures[0])
        elif exp_count >= 3:
            if cam.supports_per_channel_exposure():
                cam.set_channel_exposures(
                    red=exposures[0], green=exposures[1], blue=exposures[2],
                    auto_enable=False,
                )
                logger.info("SETCAM: per-channel R=%.3f G=%.3f B=%.3fms",
                            exposures[0], exposures[1], exposures[2])
            else:
                hardware.set_exposure(exposures[1])
                logger.info("SETCAM: fallback unified %.3fms (green)", exposures[1])

        # 3. Set gains
        if gain_count == 1:
            cam.set_unified_gain(gains[0])
            logger.info("SETCAM: unified gain %.2f", gains[0])
        elif gain_count >= 3:
            cam.set_unified_gain(gains[0])
            cam.set_rb_analog_gains(red=gains[1], blue=gains[2])
            logger.info("SETCAM: unified=%.2f, aR=%.3f, aB=%.3f",
                        gains[0], gains[1], gains[2])

        conn.sendall(b"ACK_____")
        logger.info("SETCAM complete (streaming was %s)",
                     "stopped" if stopped else "not running")

    except Exception as e:
        logger.error("SETCAM failed: %s", e)
        conn.sendall(b"ERR_SETC")
