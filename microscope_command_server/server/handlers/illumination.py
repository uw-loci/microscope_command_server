"""Illumination and acquisition profile command handlers.

Handles real-time illumination control and acquisition profile switching:
GETILLM, SETILLM, APPLYPR

These commands wrap the Illumination ABC (get_power, set_power, etc.)
and the hardware.apply_mode_setup() method for profile switching.
"""

import struct
import logging

logger = logging.getLogger(__name__)


def handle_getillm(conn, client, hardware, settings, **kwargs):
    """Return current illumination state (power, range, on/off).

    Protocol response:
    - If no illumination configured: 1 byte 0x00
    - If available: 1 byte 0x01 + 3 floats (power, min, max) + 1 byte is_on
      Total: 14 bytes

    The power/min/max units are source-specific (voltage for analog,
    intensity for DeviceProperty, etc.).
    """
    logger.debug("Client %s requested illumination state", client.addr)
    try:
        illum = getattr(hardware, '_illumination', None)
        if illum is None:
            # Always send 14 bytes (Java client reads exactly 14)
            response = struct.pack(">BfffB", 0x00, 0.0, 0.0, 0.0, 0)
            conn.sendall(response)
            logger.info("No illumination configured - sent unavailable")
            return

        power = illum.get_power()
        power_range = illum.get_power_range()
        is_on = illum.is_on()

        response = struct.pack(
            ">BfffB",
            0x01,
            float(power),
            float(power_range[0]),
            float(power_range[1]),
            1 if is_on else 0,
        )
        conn.sendall(response)
        logger.info(
            "Sent illumination state: power=%.1f, range=(%.1f, %.1f), on=%s",
            power, power_range[0], power_range[1], is_on,
        )
    except Exception as e:
        logger.error("Failed to get illumination state: %s", e)
        response = struct.pack(">BfffB", 0x00, 0.0, 0.0, 0.0, 0)
        conn.sendall(response)


def handle_setillm(conn, client, hardware, settings, **kwargs):
    """Set illumination power.

    Protocol: 4-byte big-endian float (desired power level).
    - power == 0: turns illumination off
    - power > 0: sets power (auto-enables source via set_power())

    Response: 'ACK_____' (8 bytes) on success, 'ERR_ILLM' if no illumination.
    """
    logger.debug("Client %s requesting illumination power change", client.addr)
    try:
        # Read 4-byte float payload
        data = conn.recv(4)
        if len(data) < 4:
            logger.error("Incomplete SETILLM payload: %d bytes", len(data))
            conn.sendall(b"ERR_ILLM")
            return

        power = struct.unpack(">f", data)[0]

        illum = getattr(hardware, '_illumination', None)
        if illum is None:
            logger.warning("SETILLM: no illumination configured")
            conn.sendall(b"ERR_ILLM")
            return

        if power <= 0:
            illum.off()
            logger.info("Illumination turned off")
        else:
            illum.set_power(power)
            logger.info("Illumination power set to %.1f", power)

        conn.sendall(b"ACK_____")
    except Exception as e:
        logger.error("Failed to set illumination: %s", e)
        conn.sendall(b"ERR_ILLM")


def handle_applypr(conn, client, hardware, settings, **kwargs):
    """Apply an acquisition profile (calls apply_mode_setup).

    Protocol: 32-byte null-padded UTF-8 string (profile name).
    The profile name is a key in the 'acquisition_profiles' section
    of the microscope config (e.g., 'bf_20x', 'fl_20x').

    Response: 'ACK_____' (8 bytes) on success, 'ERR_PROF' on failure.
    """
    logger.debug("Client %s requesting profile application", client.addr)
    try:
        # Read 32-byte profile name
        data = conn.recv(32)
        if len(data) < 1:
            logger.error("Empty APPLYPR payload")
            conn.sendall(b"ERR_PROF")
            return

        profile_name = data.rstrip(b"\x00").decode("utf-8").strip()
        if not profile_name:
            logger.error("APPLYPR: empty profile name after decoding")
            conn.sendall(b"ERR_PROF")
            return

        logger.info("Applying acquisition profile: '%s'", profile_name)
        hardware.apply_mode_setup(profile_name)
        logger.info("Profile '%s' applied successfully", profile_name)
        conn.sendall(b"ACK_____")
    except Exception as e:
        logger.error("Failed to apply profile '%s': %s",
                     profile_name if 'profile_name' in dir() else '?', e)
        conn.sendall(b"ERR_PROF")
