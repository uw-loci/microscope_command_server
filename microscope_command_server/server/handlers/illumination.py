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


def handle_applych(conn, client, hardware, settings, **kwargs):
    """Apply a single channel from a profile's channel library.

    Drives the same hardware state the acquisition workflow uses for that
    channel: ConfigGroup presets (cube turret, shutter, etc.) plus
    device_property writes (per-channel light source + intensity) plus
    exposure. Empty channel id deactivates all illumination sources
    declared in the profile's modality (used by the Live Viewer's "None"
    radio to fully unset).

    Protocol: 64-byte payload = 32-byte profile name + 32-byte channel id
    (both null-padded UTF-8).

    Response: 'ACK_____' on success, 'ERR_CHAN' on any failure.

    The channel switch is intentionally vendor-agnostic -- everything is
    driven by the channel's mm_setup_presets + device_properties from the
    YAML, so any multi-channel illumination hardware works without code
    changes here.
    """
    logger.debug("Client %s requesting channel application", client.addr)
    profile_name = "?"
    channel_id = "?"
    try:
        data = conn.recv(64)
        if len(data) < 64:
            logger.error("APPLYCH: short payload (%d bytes, want 64)", len(data))
            conn.sendall(b"ERR_CHAN")
            return

        profile_name = data[:32].rstrip(b"\x00").decode("utf-8").strip()
        channel_id = data[32:].rstrip(b"\x00").decode("utf-8").strip()

        # Resolve profile -> modality -> channel library
        profiles = (settings or {}).get("acquisition_profiles", {}) or {}
        profile_cfg = profiles.get(profile_name)
        if profile_cfg is None and "_" in profile_name:
            # Strip a trailing "_<counter>" the way apply_mode_setup does.
            stem = "_".join(profile_name.split("_")[:-1])
            profile_cfg = profiles.get(stem)
        if not isinstance(profile_cfg, dict):
            logger.error("APPLYCH: unknown profile '%s'", profile_name)
            conn.sendall(b"ERR_CHAN")
            return

        modality_name = profile_cfg.get("modality")
        modalities = (settings or {}).get("modalities", {}) or {}
        modality_cfg = modalities.get(modality_name) if modality_name else None
        if not isinstance(modality_cfg, dict):
            logger.error(
                "APPLYCH: profile '%s' has no resolvable modality",
                profile_name,
            )
            conn.sendall(b"ERR_CHAN")
            return

        # "None" / empty channel id: deactivate every illumination source
        # the modality declares. Same helper apply_mode_setup uses for
        # symmetric teardown.
        if not channel_id:
            try:
                hardware._disable_all_modality_illuminations()
                logger.info(
                    "APPLYCH: deactivated all illumination for profile '%s' (modality '%s')",
                    profile_name, modality_name,
                )
                conn.sendall(b"ACK_____")
            except Exception as e:
                logger.error("APPLYCH: deactivation failed: %s", e)
                conn.sendall(b"ERR_CHAN")
            return

        # Find the channel entry by id in the modality's channel library.
        library = modality_cfg.get("channels", []) or []
        ch_entry = None
        for entry in library:
            if isinstance(entry, dict) and entry.get("id") == channel_id:
                ch_entry = dict(entry)  # shallow copy for override merge
                break
        if ch_entry is None:
            logger.error(
                "APPLYCH: channel '%s' not found in modality '%s' library",
                channel_id, modality_name,
            )
            conn.sendall(b"ERR_CHAN")
            return

        # Merge profile-level channel_overrides (per-objective tuning).
        overrides = profile_cfg.get("channel_overrides") or {}
        if isinstance(overrides, dict) and isinstance(overrides.get(channel_id), dict):
            ch_override = overrides[channel_id]
            for k, v in ch_override.items():
                if k == "device_properties" and isinstance(v, list):
                    base_props = list(ch_entry.get("device_properties") or [])
                    base_props.extend(v)
                    ch_entry["device_properties"] = base_props
                else:
                    ch_entry[k] = v

        # Lazy import to avoid a server -> control circular import when
        # this module is loaded at startup.
        from microscope_command_server.acquisition.workflow import (
            apply_channel_hardware_state,
        )
        apply_channel_hardware_state(hardware, ch_entry, logger)

        # Apply exposure if declared (skip silently if not).
        exposure_ms = ch_entry.get("exposure_ms")
        if isinstance(exposure_ms, (int, float)) and exposure_ms > 0:
            try:
                hardware.set_exposure(float(exposure_ms))
            except Exception as e:
                logger.warning("APPLYCH: exposure write failed: %s", e)

        logger.info(
            "APPLYCH: applied channel '%s' from profile '%s' (modality '%s')",
            channel_id, profile_name, modality_name,
        )
        conn.sendall(b"ACK_____")
    except Exception as e:
        logger.error(
            "APPLYCH failed for profile='%s' channel='%s': %s",
            profile_name, channel_id, e,
        )
        conn.sendall(b"ERR_CHAN")
