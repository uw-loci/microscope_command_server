"""Camera control command handlers.

Handles camera property queries and settings:
GETCAM, GETMODE, SETMODE, GETEXP, SETEXP, GETGAIN, SETGAIN

These commands interact with JAI camera properties extensively.
For non-JAI cameras, reasonable defaults or generic hardware calls are used.
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
    - JAI camera: 'JAI_EXP:<0|1|U>_GAIN:0'
    - Non-JAI / missing module: 'UNIFIED_________'
    - Error: 'ERROR:<msg>' (16 bytes padded)
    """
    # TODO: migrate to hardware.camera.properties
    logger.debug("Client %s requested camera mode flags", client.addr)
    try:
        from microscope_control.jai import JAICameraProperties
        jai_props = JAICameraProperties(hardware.core)

        if jai_props.validate_camera():
            if not jai_props.supports_individual_exposure():
                # JAI camera but ExposureIsIndividual property not available
                # (e.g. MM adapter without PR #781 support)
                mode_str = "JAI_EXP:U_GAIN:0"
                conn.sendall(mode_str.encode("utf-8").ljust(16, b"\x00"))
                logger.info("JAI camera without individual exposure support - sent EXP:U")
            else:
                exp_individual = jai_props.is_individual_exposure_enabled()
                # Gain is always unified in new model
                mode_str = f"JAI_EXP:{1 if exp_individual else 0}_GAIN:0"
                conn.sendall(mode_str.encode("utf-8").ljust(16, b"\x00"))
                logger.info(
                    "Sent JAI mode flags: exp_ind=%s, gain_ind=false",
                    exp_individual,
                )
        else:
            conn.sendall(b"UNIFIED_________")
            logger.info("Non-JAI camera - sent UNIFIED mode")
    except ImportError:
        conn.sendall(b"UNIFIED_________")
        logger.info("JAI module not available - sent UNIFIED mode")
    except Exception as e:
        logger.error("Failed to get camera mode: %s", e)
        error_msg = f"ERROR:{str(e)[:8]}"
        conn.sendall(error_msg.encode("utf-8").ljust(16, b"\x00"))


def handle_setmode(conn, client, hardware, settings, **kwargs):
    """Set exposure/gain mode flags.

    JAI-SPECIFIC: Sets exposure mode (individual or unified).
    Gain mode byte is accepted but ignored - gain is always unified.
    R/B analog gains work in unified mode via set_rb_analog_gains().

    Protocol: 2 bytes [exp_mode, gain_mode]
      exp_mode:  1 = individual (R,G,B separate), 0 = unified
      gain_mode: ignored (always unified), logged if True requested

    Response: 'ACK_____' on success, 'ERR_NJAI' if not JAI,
      'ERR_NSUP' if individual mode not supported, 'ERR_MODE' on failure.
    """
    # TODO: migrate to hardware.camera.properties
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

        # Safety net: stop any active streaming before changing camera properties.
        # JAI cameras cannot change ExposureIsIndividual while hardware is busy.
        stopped_sequence = False
        stopped_studio_live = False
        try:
            if hardware.core.is_sequence_running():
                logger.warning("Core sequence running during SETMODE - auto-stopping")
                hardware.core.stop_sequence_acquisition()
                stopped_sequence = True
                time.sleep(0.2)
        except Exception as seq_err:
            logger.debug("Could not check/stop sequence: %s", seq_err)
        try:
            if hardware.studio is not None and hardware.studio.live().is_live_mode_on():
                logger.warning("MM Studio live mode on during SETMODE - auto-stopping")
                hardware.studio.live().set_live_mode(False)
                stopped_studio_live = True
                time.sleep(0.2)
        except Exception as live_err:
            logger.debug("Could not check/stop studio live: %s", live_err)

        from microscope_control.jai import JAICameraProperties
        jai_props = JAICameraProperties(hardware.core)

        if not jai_props.validate_camera():
            raise RuntimeError("JAI camera not active - cannot set individual mode")

        if exp_individual and not jai_props.supports_individual_exposure():
            conn.sendall(b"ERR_NSUP")
            logger.error(
                "Individual exposure mode requested but ExposureIsIndividual "
                "property not available. Check MicroManager device adapter "
                "version (requires PR #781)."
            )
            return

        if exp_individual:
            jai_props.enable_individual_exposure()
        else:
            jai_props.disable_individual_exposure()

        # Always ensure gain is unified
        jai_props.disable_individual_gain()

        conn.sendall(b"ACK_____")
        if stopped_sequence or stopped_studio_live:
            logger.info("Camera mode set successfully (auto-stopped streaming first)")
        else:
            logger.info("Camera mode set successfully")
    except ImportError:
        conn.sendall(b"ERR_NJAI")
        logger.error("JAI module not available")
    except Exception as e:
        logger.error("Failed to set camera mode: %s", e)
        conn.sendall(b"ERR_MODE")


def handle_getexp(conn, client, hardware, settings, **kwargs):
    """Return exposure values (unified or per-channel RGB).

    JAI with individual exposures: 4 floats (all, R, G, B) = 16 bytes.
    Unified / non-JAI: 1 float = 4 bytes.
    Error: 1 float = -1.0.
    """
    # TODO: migrate to hardware.camera.properties
    logger.debug("Client %s requested exposure values", client.addr)
    try:
        from microscope_control.jai import JAICameraProperties
        jai_props = JAICameraProperties(hardware.core)

        if jai_props.validate_camera() and jai_props.is_individual_exposure_enabled():
            # JAI with individual exposures - return 4 floats (all, R, G, B)
            exposures = jai_props.get_channel_exposures()
            # Get unified exposure as well for "all" value
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
            # Unified exposure - return 1 float
            exposure = hardware.get_exposure()
            response = struct.pack("!f", float(exposure))
            conn.sendall(response)
            logger.info("Sent unified exposure: %s", exposure)
    except ImportError:
        # JAI module not available - get unified exposure
        exposure = hardware.get_exposure()
        response = struct.pack("!f", float(exposure))
        conn.sendall(response)
        logger.info("Sent unified exposure (no JAI): %s", exposure)
    except Exception as e:
        logger.error("Failed to get exposure: %s", e)
        # Send error as negative value
        conn.sendall(struct.pack("!f", -1.0))


def handle_setexp(conn, client, hardware, settings, **kwargs):
    """Set exposure values.

    MIXED: count=1 is GENERIC (calls hardware.set_exposure for any camera),
    count>=3 is JAI-SPECIFIC (sets per-channel R,G,B exposures via
    JAICameraProperties.set_channel_exposures with auto_enable=True,
    which implicitly enables individual exposure mode).

    Protocol: 1 count byte + (count * 4) bytes of big-endian floats
    (exposure values in ms).

    Response: 'ACK_____' on success, 'ERR_NJAI' if JAI module unavailable
    for per-channel, 'ERR_EXPO' on other failure.
    """
    # TODO: migrate to hardware.camera.properties
    logger.debug("Client %s requested to set exposure", client.addr)
    try:
        # Read count byte first
        count_byte = conn.recv(1)
        count = count_byte[0]
        logger.debug("SETEXP: expecting %d exposure values", count)

        # Read float values
        float_data = conn.recv(count * 4)
        if len(float_data) != count * 4:
            raise ValueError(
                f"Expected {count * 4} bytes, got {len(float_data)}"
            )

        exposures = struct.unpack(f"!{'f' * count}", float_data)
        logger.info("Setting exposures: %s", exposures)

        if count == 1:
            # Unified exposure
            hardware.set_exposure(exposures[0])
            logger.info("Set unified exposure to %s ms", exposures[0])
        elif count >= 3:
            # Per-channel exposures (R, G, B)
            from microscope_control.jai import JAICameraProperties
            jai_props = JAICameraProperties(hardware.core)
            if not jai_props.supports_individual_exposure():
                # Fall back to unified using green channel value
                hardware.set_exposure(exposures[1])
                logger.warning(
                    "Per-channel exposures requested but individual mode not "
                    "supported - using green channel value (%.2f ms) as unified",
                    exposures[1],
                )
            else:
                jai_props.set_channel_exposures(
                    red=exposures[0],
                    green=exposures[1],
                    blue=exposures[2],
                    auto_enable=True,
                )
                logger.info(
                    "Set per-channel exposures: R=%s, G=%s, B=%s",
                    exposures[0], exposures[1], exposures[2],
                )

        conn.sendall(b"ACK_____")
    except ImportError:
        conn.sendall(b"ERR_NJAI")
        logger.error("JAI module not available for per-channel exposure")
    except Exception as e:
        logger.error("Failed to set exposure: %s", e)
        conn.sendall(b"ERR_EXPO")


def handle_getgain(conn, client, hardware, settings, **kwargs):
    """Return gain values.

    Always returns 3 floats: [unified_gain, analog_red, analog_blue] = 12 bytes.
    Non-JAI or missing module: returns (1.0, 1.0, 1.0).
    Error: returns (-1.0, -1.0, -1.0).
    """
    # TODO: migrate to hardware.camera.properties
    logger.debug("Client %s requested gain values", client.addr)
    try:
        from microscope_control.jai import JAICameraProperties
        jai_props = JAICameraProperties(hardware.core)

        if jai_props.validate_camera():
            unified = jai_props.get_unified_gain()
            rb_gains = jai_props.get_rb_analog_gains()
            response = struct.pack(
                "!fff",
                float(unified),
                float(rb_gains["red"]),
                float(rb_gains["blue"]),
            )
            conn.sendall(response)
            logger.info(
                "Sent gains: unified=%s, analog_red=%s, analog_blue=%s",
                unified, rb_gains["red"], rb_gains["blue"],
            )
        else:
            # Not JAI - return defaults
            response = struct.pack("!fff", 1.0, 1.0, 1.0)
            conn.sendall(response)
            logger.info("Non-JAI camera - sent default gains (1.0, 1.0, 1.0)")
    except ImportError:
        response = struct.pack("!fff", 1.0, 1.0, 1.0)
        conn.sendall(response)
        logger.info("JAI module not available - sent default gains")
    except Exception as e:
        logger.error("Failed to get gain: %s", e)
        conn.sendall(struct.pack("!fff", -1.0, -1.0, -1.0))


def handle_setgain(conn, client, hardware, settings, **kwargs):
    """Set gain values.

    JAI-SPECIFIC (both paths):
      count=1: Sets unified gain via set_unified_gain (range 1.0-8.0)
      count=3: Sets [unified_gain, analog_red, analog_blue]
               - unified gain applied to all channels
               - analog_red/blue applied via set_rb_analog_gains (0.47-4.0)
               - Does NOT enable individual gain mode

    Protocol: 1 count byte + (count * 4) bytes floats.
    Response: 'ACK_____', 'ERR_NJAI', or 'ERR_GAIN'.
    """
    # TODO: migrate to hardware.camera.properties
    logger.debug("Client %s requested to set gain", client.addr)
    try:
        # Read count byte first
        count_byte = conn.recv(1)
        count = count_byte[0]
        logger.debug("SETGAIN: expecting %d gain values", count)

        # Read float values
        float_data = conn.recv(count * 4)
        if len(float_data) != count * 4:
            raise ValueError(
                f"Expected {count * 4} bytes, got {len(float_data)}"
            )

        gains = struct.unpack(f"!{'f' * count}", float_data)
        logger.info("Setting gains: %s", gains)

        from microscope_control.jai import JAICameraProperties
        jai_props = JAICameraProperties(hardware.core)

        # Stop any active streaming before changing gain properties
        # (same pattern as SETMODE handler)
        try:
            if hardware.core.is_sequence_running():
                hardware.core.stop_sequence_acquisition()
                time.sleep(0.2)
        except Exception:
            pass

        if count == 1:
            # Unified gain only
            jai_props.set_unified_gain(gains[0])
            logger.info("Set unified gain: %s", gains[0])
        elif count >= 3:
            # New semantics: [unified_gain, analog_red, analog_blue]
            jai_props.set_unified_gain(gains[0])
            jai_props.set_rb_analog_gains(red=gains[1], blue=gains[2])
            logger.info(
                "Set gains: unified=%s, analog_red=%s, analog_blue=%s",
                gains[0], gains[1], gains[2],
            )

        conn.sendall(b"ACK_____")
    except ImportError:
        conn.sendall(b"ERR_NJAI")
        logger.error("JAI module not available for gain control")
    except Exception as e:
        logger.error("Failed to set gain: %s", e)
        conn.sendall(b"ERR_GAIN")
