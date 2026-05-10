"""Live mode and snapshot command handlers.

Handles live viewing, sequence control, and snapshot acquisition:
GETLIVE, SETLIVE, GETFRAME, STRTSEQ, STOPSEQ, SNAP

SNAP is a complex handler that supports per-channel exposure control,
white balance calibration lookup, rotation angle setting, and TIFF output.
"""

import socket
import struct
import time
import logging

import numpy as np

logger = logging.getLogger(__name__)


def handle_getlive(conn, client, hardware, settings, **kwargs):
    """Check if live mode is currently running.

    Response: 1 byte (0 = not live, 1 = live).
    """
    logger.debug("Client %s requested live mode status", client.addr)
    try:
        is_live = False
        # Check if sequence is running (indicates live mode)
        if hardware.core.is_sequence_running():
            is_live = True
        # Also check via studio if available
        elif hardware.studio is not None:
            try:
                is_live = hardware.studio.live().is_live_mode_on()
            except Exception:
                pass  # Fall back to is_sequence_running result

        # Response: 1 byte (0 = not live, 1 = live)
        conn.sendall(bytes([1 if is_live else 0]))
        logger.info("Live mode status: %s", "ON" if is_live else "OFF")
    except Exception as e:
        logger.error("Failed to get live mode status: %s", e)
        conn.sendall(bytes([0]))  # Default to not live on error


def handle_setlive(conn, client, hardware, settings, **kwargs):
    """Set live mode on or off.

    When turning OFF, also stops core-level sequence acquisition so that
    SETLIVE OFF is comprehensive (matches what GETLIVE reports).

    Protocol: 1 byte (0 = off, 1 = on).
    Response: 'ACK_____', 'ERR_NSTD' if no studio, 'ERR_LIVE' on failure.
    """
    logger.debug("Client %s requested to set live mode", client.addr)
    try:
        # Read 1 byte: 0 = off, 1 = on
        enable_byte = conn.recv(1)
        if len(enable_byte) != 1:
            raise ValueError("Expected 1 byte for live mode flag")

        enable_live = enable_byte[0] == 1
        logger.info("Setting live mode: %s", "ON" if enable_live else "OFF")

        if not enable_live:
            # Stop core-level sequence acquisition (QPSC Live Viewer uses this).
            # Verify the stop actually took effect to avoid stuck-live state
            # where is_sequence_running() stays True after a workflow error.
            try:
                if hardware.core.is_sequence_running():
                    hardware.core.stop_sequence_acquisition()
                    import time

                    time.sleep(0.05)
                    if hardware.core.is_sequence_running():
                        logger.warning("Sequence still running after stop -- retrying")
                        hardware.core.stop_sequence_acquisition()
                        time.sleep(0.1)
                    logger.info("Stopped core sequence acquisition via SETLIVE OFF")
                    # Clear stale frames from the circular buffer
                    try:
                        hardware.core.clear_circular_buffer()
                    except Exception:
                        pass
            except Exception as seq_err:
                logger.warning("Could not stop core sequence: %s", seq_err)

        if hardware.studio is not None:
            hardware.studio.live().set_live_mode(enable_live)
            conn.sendall(b"ACK_____")
            logger.info("Live mode set to %s", "ON" if enable_live else "OFF")
        else:
            # No studio available - cannot control live mode
            conn.sendall(b"ERR_NSTD")
            logger.warning("No studio available to control live mode")
    except Exception as e:
        logger.error("Failed to set live mode: %s", e)
        conn.sendall(b"ERR_LIVE")


def handle_getframe(conn, client, hardware, settings, **kwargs):
    """Get latest frame from MM circular buffer (for live viewer).

    Response: 20-byte header (5 big-endian ints: w, h, channels, bpp, data_len)
    followed by raw pixel bytes. On error or no frame, sends zero header.
    """
    try:
        image, meta = hardware.get_live_frame()
        if image is None:
            # No frame available - send zero header
            conn.sendall(struct.pack(">5i", 0, 0, 0, 0, 0))
            return

        h, w = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        bpp = image.dtype.itemsize

        # Convert uint16 to big-endian for wire transfer
        if image.dtype == np.uint16:
            image = image.astype(">u2")

        raw_bytes = np.ascontiguousarray(image).tobytes()
        header = struct.pack(">5i", w, h, channels, bpp, len(raw_bytes))
        conn.sendall(header + raw_bytes)
    except Exception as e:
        logger.error("GETFRAME failed: %s", e)
        try:
            conn.sendall(struct.pack(">5i", 0, 0, 0, 0, 0))
        except Exception:
            pass


def handle_correctframe(conn, client, hardware, settings, **kwargs):
    """Get latest frame with server-side flat-field correction applied.

    Same wire format as GETFRAME on success: 20-byte header
    (5 big-endian ints: w, h, channels, bpp, data_len) followed by raw
    pixel bytes. On any configuration failure (no background folder,
    missing per-angle file, shape mismatch) the server replies with a
    textual ``FAILED:<reason>`` payload instead so the Java client can
    fall back to an uncorrected snap with a clear status message.

    The Java side is expected to have already validated settings-match
    (modality / objective / detector / WB mode / current angle) before
    issuing this command -- those are the same checks ACQUIRE runs in
    AcquisitionConfigurationBuilder. This handler is the second line of
    defense for racy YAML edits during a session.

    Args (via kwargs):
        active_connection_config_path: Path to the active config_<scope>.yml
            from which the imageprocessing_<scope>.yml peer file is derived.
    """
    from pathlib import Path

    addr = client.addr
    logger.info("Client %s requested corrected frame", addr)

    def _fail(reason):
        try:
            conn.sendall(f"FAILED:{reason}".encode("ascii", errors="replace"))
        except Exception as send_err:
            logger.error("CORRECTFRAME: could not send FAILED response: %s", send_err)

    try:
        # Step 1: pull current frame (identical to GETFRAME).
        image, _ = hardware.get_live_frame()
        if image is None:
            # No frame available -- behave like GETFRAME and send a zero
            # header. The Java client treats this as "skip the snap".
            conn.sendall(struct.pack(">5i", 0, 0, 0, 0, 0))
            return

        # Step 2: resolve modality, objective, detector from the cached
        # CONFIG settings.
        if not hasattr(hardware, "settings") or not hardware.settings:
            _fail("Server has no CONFIG settings loaded")
            return
        mscope = hardware.settings.get("microscope", {}) or {}
        modality = mscope.get("modality_in_use") or mscope.get("default_modality")
        objective = mscope.get("objective_in_use") or mscope.get("objective")
        detector = mscope.get("detector_in_use") or mscope.get("detector")
        if not modality:
            _fail("No active modality (microscope.modality_in_use)")
            return
        if not objective:
            _fail("No active objective (microscope.objective_in_use)")
            return
        if not detector:
            _fail("No active detector (microscope.detector_in_use)")
            return

        # Step 3: locate the background folder via the imageprocessing config.
        config_path_str = kwargs.get("active_connection_config_path")
        if not config_path_str:
            _fail("No active CONFIG path on the server")
            return
        config_path = Path(config_path_str)
        ip_path = (
            config_path.parent / f"imageprocessing_{config_path.stem.replace('config_', '')}.yml"
        )
        if not ip_path.exists():
            _fail(f"No imageprocessing config at {ip_path}")
            return
        try:
            from microscope_command_server.utils import config_manager

            ip_config = config_manager.load_config_file(str(ip_path))
        except Exception as e:
            _fail(f"Could not load imageprocessing config: {e}")
            return
        bc_root = (ip_config or {}).get("background_correction", {})
        bc_settings = bc_root.get(modality) or bc_root.get(_strip_mag_suffix(modality)) or {}
        if not bc_settings.get("enabled"):
            _fail(f"Background correction not enabled for modality '{modality}'")
            return
        bg_base = bc_settings.get("base_folder")
        if not bg_base:
            _fail(f"No base_folder configured for modality '{modality}'")
            return

        # Step 4: determine current rotation angle (0 for non-rotation modalities).
        try:
            angle = float(hardware.get_rotation_position())
        except Exception:
            angle = 0.0
        method = bc_settings.get("method", "divide")

        # Step 5: load the matching background image.
        try:
            from microscope_imageprocessing.correction.background import (
                BackgroundCorrectionUtils,
            )
        except Exception as e:
            _fail(f"BackgroundCorrectionUtils unavailable: {e}")
            return
        bg_root = Path(bg_base)
        backgrounds, scaling_factors, _ = BackgroundCorrectionUtils.load_background_images(
            bg_root, [angle], logger
        )
        if not backgrounds or angle not in backgrounds:
            _fail(f"No background image for angle {angle} in {bg_root} " f"(modality={modality})")
            return

        # Step 6: apply correction.
        try:
            corrected = BackgroundCorrectionUtils.apply_flat_field_correction(
                image,
                backgrounds[angle],
                scaling_factors.get(angle, 1.0),
                method=method,
            )
        except Exception as e:
            _fail(f"apply_flat_field_correction failed: {e}")
            return

        # Step 7: send back same wire format as GETFRAME.
        h, w = corrected.shape[:2]
        channels = 1 if corrected.ndim == 2 else corrected.shape[2]
        bpp = corrected.dtype.itemsize

        if corrected.dtype == np.uint16:
            corrected = corrected.astype(">u2")
        raw_bytes = np.ascontiguousarray(corrected).tobytes()
        header = struct.pack(">5i", w, h, channels, bpp, len(raw_bytes))
        conn.sendall(header + raw_bytes)
        logger.info(
            "CORRECTFRAME OK: %dx%d, %d ch, %d-bit (modality=%s, angle=%s)",
            w,
            h,
            channels,
            bpp * 8,
            modality,
            angle,
        )
    except Exception as e:
        logger.exception("CORRECTFRAME failed: %s", e)
        try:
            _fail(f"Internal error: {e}")
        except Exception:
            pass


def _strip_mag_suffix(modality):
    """Strip a trailing _<n>x magnification suffix from a modality id.

    e.g. ``ppm_20x`` -> ``ppm``. Mirrors ``HardwareKey.stripMagnificationSuffix``
    in the Java side so the same modality lookup keys work in both halves.
    """
    if not modality:
        return modality
    import re

    return re.sub(r"_\d+x$", "", modality, flags=re.IGNORECASE)


def handle_strtseq(conn, client, hardware, settings, **kwargs):
    """Start continuous sequence acquisition (core-level, bypasses MM live window).

    Response: 'ACK_____' on success, 'ERR_SEQ_' on failure.
    """
    logger.info("Client %s requested start continuous acquisition", client.addr)
    try:
        hardware.start_continuous_acquisition()
        conn.sendall(b"ACK_____")
        logger.info("Continuous sequence acquisition started")
    except Exception as e:
        logger.error("Failed to start continuous acquisition: %s", e)
        conn.sendall(b"ERR_SEQ_")


def handle_stopseq(conn, client, hardware, settings, **kwargs):
    """Stop continuous sequence acquisition (core-level).

    Response: 'ACK_____' on success, 'ERR_SEQ_' on failure.

    Holds ``sequence_op_lock`` for the duration of the MM-core stop call so
    a concurrent CONFIG-takeover orphan-stop from a same-IP reconnect does
    not re-enter the core simultaneously (deadlock observed 2026-05-02
    OWS3). Entry / mid / exit are all logged so a hang inside the core call
    is visible from the server log instead of looking like a silent failure.
    """
    logger.info("Client %s requested stop continuous acquisition", client.addr)
    sequence_op_lock = kwargs.get("sequence_op_lock")
    if sequence_op_lock is not None:
        sequence_op_lock.acquire()
    try:
        logger.info("STOPSEQ: calling hardware.stop_continuous_acquisition() for %s", client.addr)
        hardware.stop_continuous_acquisition()
        logger.info("STOPSEQ: hardware.stop_continuous_acquisition() returned for %s", client.addr)
        conn.sendall(b"ACK_____")
        logger.info("Continuous sequence acquisition stopped")
    except Exception as e:
        logger.error("Failed to stop continuous acquisition: %s", e)
        conn.sendall(b"ERR_SEQ_")
    finally:
        if sequence_op_lock is not None:
            sequence_op_lock.release()


def handle_snap(conn, client, hardware, settings, **kwargs):
    """Snapshot with fixed exposure, optional white balance and rotation.

    Reads a variable-length message terminated by END_MARKER containing
    flags: --angle, --exposure, --output, --debayer, --white_balance,
    --yaml, --objective, --detector, --exp_r, --exp_g, --exp_b.

    Supports three exposure modes (priority order):
    1. Direct per-channel exposures (--exp_r, --exp_g, --exp_b) for calibration
    2. WB calibration lookup from YAML (--white_balance + --yaml)
    3. Unified exposure (--exposure)

    Response: 'SUCCESS:<output_path>' or 'FAILED:<error>'.

    Args (via kwargs):
        end_marker: The END_MARKER bytes to detect message boundary.
    """
    from microscope_command_server.server.protocol import END_MARKER

    addr = client.addr
    logger.info("Client %s requested simple snap (fixed exposure)", addr)
    snap_start_time = time.time()

    # Read the message with parameters
    message_parts = []
    total_bytes = 0
    start_time = time.time()

    conn.settimeout(5.0)

    try:
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                logger.error("Connection closed while reading snap message")
                conn.sendall(b"FAILED:Connection closed")
                break

            message_parts.append(chunk.decode("utf-8"))
            total_bytes += len(chunk)
            logger.debug("SNAP: received %d bytes so far", total_bytes)

            full_message = "".join(message_parts)

            if END_MARKER in full_message:
                message = full_message.replace(END_MARKER, "").strip()

                # Parse the message
                params = {}

                # Parse flags
                flags = [
                    "--angle",
                    "--exposure",
                    "--output",
                    "--debayer",
                    "--white_balance",
                    "--yaml",
                    "--objective",
                    "--detector",
                    "--exp_r",
                    "--exp_g",
                    "--exp_b",
                ]

                for i, flag in enumerate(flags):
                    if flag in message:
                        start_idx = message.index(flag) + len(flag)
                        end_idx = len(message)
                        for next_flag in flags[i + 1 :]:
                            if next_flag in message[start_idx:]:
                                next_pos = message.index(next_flag, start_idx)
                                if next_pos < end_idx:
                                    end_idx = next_pos
                                    break

                        value = message[start_idx:end_idx].strip()

                        if flag == "--angle":
                            params["angle"] = float(value)
                        elif flag == "--exposure":
                            params["exposure_ms"] = float(value)
                        elif flag == "--output":
                            params["output_path"] = value
                        elif flag == "--debayer":
                            # Support "auto", "true"/"1"/"yes", "false"/"0"/"no"
                            val = value.lower()
                            if val == "auto":
                                params["debayer"] = "auto"
                            else:
                                params["debayer"] = val in ("true", "1", "yes")
                        elif flag == "--white_balance":
                            params["white_balance"] = value.lower() in ("true", "1", "yes")
                        elif flag == "--yaml":
                            params["yaml_path"] = value
                        elif flag == "--objective":
                            params["objective"] = value
                        elif flag == "--detector":
                            params["detector"] = value
                        elif flag == "--exp_r":
                            params["exp_r"] = float(value)
                        elif flag == "--exp_g":
                            params["exp_g"] = float(value)
                        elif flag == "--exp_b":
                            params["exp_b"] = float(value)

                # Validate required parameters
                required = ["angle", "exposure_ms", "output_path"]
                missing = [key for key in required if key not in params]
                if missing:
                    error_msg = f"Missing required parameters: {missing}"
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                    break

                try:
                    import tifffile
                    from pathlib import Path

                    angle = params["angle"]
                    exposure_ms = params["exposure_ms"]
                    output_path = Path(params["output_path"])
                    debayer = params.get("debayer", "auto")
                    use_white_balance = params.get("white_balance", False)
                    yaml_path = params.get("yaml_path")

                    # Per-channel exposures for direct control (e.g., WB calibration loops)
                    exp_r = params.get("exp_r")
                    exp_g = params.get("exp_g")
                    exp_b = params.get("exp_b")

                    # Create output directory if needed
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Priority 1: Direct per-channel exposures (for calibration loops)
                    # Priority 2: WB calibration lookup from YAML
                    # Priority 3: Unified exposure
                    wb_applied = False
                    cam = hardware.camera
                    if exp_r is not None and exp_g is not None and exp_b is not None:
                        # Direct per-channel control - used for WB calibration loops
                        if cam.supports_per_channel_exposure():
                            try:
                                cam.set_channel_exposures(
                                    red=exp_r,
                                    green=exp_g,
                                    blue=exp_b,
                                    auto_enable=True,
                                )
                                wb_applied = True
                                logger.info(
                                    "SNAP: Applied direct per-channel exposures: "
                                    "R=%.2fms, G=%.2fms, B=%.2fms",
                                    exp_r,
                                    exp_g,
                                    exp_b,
                                )
                            except Exception as e:
                                logger.warning("SNAP: Failed to set per-channel exposures: %s", e)
                        else:
                            logger.debug(
                                "SNAP: Per-channel exposures provided but camera "
                                "does not support per-channel mode"
                            )

                    elif use_white_balance and yaml_path:
                        wb_applied = _apply_snap_white_balance(
                            hardware, params, angle, exposure_ms, yaml_path, logger
                        )

                    # If white balance was not applied, use the default behavior:
                    # disable per-channel mode and use unified exposure
                    if not wb_applied:
                        cam.disable_individual_exposure()
                        cam.disable_individual_gain()

                        # Set unified exposure (fixed - no adaptive adjustment!)
                        hardware.set_exposure(exposure_ms)
                        logger.info("Set exposure to %.2f ms (FIXED)", exposure_ms)
                    else:
                        # White balance was applied - per-channel exposures are set,
                        # so we don't call hardware.set_exposure() which would be ignored
                        # (or potentially interfere with per-channel mode)
                        logger.debug(
                            "SNAP: Using per-channel exposures from WB calibration "
                            "(exposure_ms=%.2f ignored)",
                            exposure_ms,
                        )

                    # Set rotation angle
                    if hasattr(hardware, "set_psg_ticks"):
                        hardware.set_psg_ticks(angle)
                        logger.info("Set rotation angle to %.2f deg", angle)

                    # Snap image with simple acquisition
                    image, metadata = hardware.snap_image(debayering=debayer)

                    if image is None:
                        raise RuntimeError("snap_image returned None")

                    # Save the image
                    tifffile.imwrite(
                        str(output_path),
                        image,
                        compression="zlib",
                        compressionargs={"level": 6},
                    )

                    elapsed = time.time() - snap_start_time
                    logger.info(
                        "SNAP complete: %s, "
                        "angle=%.2fdeg, exposure=%.2fms, "
                        "shape=%s, median=%.1f, "
                        "total_time=%.2fs",
                        output_path.name,
                        angle,
                        exposure_ms,
                        image.shape,
                        float(image.mean()),
                        elapsed,
                    )

                    # Send success response
                    response = f"SUCCESS:{output_path}".encode()
                    conn.sendall(response)
                    logger.debug("SNAP: sent SUCCESS response")

                except Exception as e:
                    logger.error("SNAP failed: %s", str(e), exc_info=True)
                    response = f"FAILED:{str(e)}".encode()
                    conn.sendall(response)

                break

            # Safety checks
            if total_bytes > 10000:
                logger.error("SNAP message too large: %d bytes", total_bytes)
                conn.sendall(b"FAILED:Message too large")
                break

            if time.time() - start_time > 10:
                logger.error("Timeout reading SNAP message")
                conn.sendall(b"FAILED:Timeout waiting for complete message")
                break

    except socket.timeout:
        logger.error("Timeout reading SNAP message from %s", addr)
        conn.sendall(b"FAILED:Timeout reading message")
    except Exception as e:
        logger.error("Error in SNAP: %s", str(e), exc_info=True)
        conn.sendall(f"FAILED:{str(e)}".encode())
    finally:
        conn.settimeout(None)


def _apply_snap_white_balance(hardware, params, angle, exposure_ms, yaml_path, logger):
    """Apply white balance calibration for SNAP command.

    Attempts to load JAI calibration data from the YAML config and apply
    per-angle white balance with optional exposure scaling.

    Returns:
        True if white balance was successfully applied, False otherwise.
    """
    try:
        from pathlib import Path
        from microscope_command_server.acquisition.workflow import (
            load_jai_calibration_from_imageprocessing,
            apply_jai_calibration_for_angle,
            get_interpolated_calibration_for_angle,
        )

        # Get objective/detector from params or hardware.settings
        wb_objective = params.get("objective")
        wb_detector = params.get("detector")
        if not wb_objective or not wb_detector:
            if hasattr(hardware, "settings") and hardware.settings:
                wb_objective = (
                    wb_objective
                    or hardware.settings.get("objective_in_use")
                    or hardware.settings.get("objective")
                )
                wb_detector = (
                    wb_detector
                    or hardware.settings.get("detector_in_use")
                    or hardware.settings.get("detector")
                )

        if not wb_objective or not wb_detector:
            logger.warning(
                "SNAP: Cannot apply WB - missing objective (%s) or detector (%s)",
                wb_objective,
                wb_detector,
            )
            return False

        # Derive modality from YAML config path
        # (e.g. config_PPM.yml -> "PPM")
        snap_modality = params.get("modality", "ppm")
        jai_cal = load_jai_calibration_from_imageprocessing(
            config_path=Path(yaml_path),
            per_angle=True,
            modality=snap_modality,
            objective=wb_objective,
            detector=wb_detector,
            logger=logger,
        )
        if not jai_cal:
            logger.warning("SNAP: No JAI calibration found in %s", yaml_path)
            return False

        # Calculate exposure scale factor to allow adaptive
        # exposure control while preserving WB color ratios.
        # The calibration provides per-channel exposures for
        # color balance; we scale them by the ratio of the
        # adaptive exposure_ms to the calibration base exposure.
        exposure_scale = None
        if "angles" in jai_cal:
            angle_cal = get_interpolated_calibration_for_angle(
                angle=angle,
                angles_cal=jai_cal["angles"],
                logger=logger,
            )
            if angle_cal:
                cal_exposures = angle_cal.get("exposures_ms", {})
                base_exp = (
                    cal_exposures.get("r", 50.0)
                    + cal_exposures.get("g", 50.0)
                    + cal_exposures.get("b", 50.0)
                ) / 3.0
                if base_exp > 0:
                    exposure_scale = exposure_ms / base_exp
                    logger.debug(
                        "SNAP: WB exposure scale=%.2fx " "(adaptive=%.1fms / base=%.1fms)",
                        exposure_scale,
                        exposure_ms,
                        base_exp,
                    )

        wb_applied, exp_info = apply_jai_calibration_for_angle(
            hardware=hardware,
            jai_calibration=jai_cal,
            angle=angle,
            per_angle=True,
            logger=logger,
            exposure_scale=exposure_scale,
        )
        if wb_applied:
            if exposure_scale is not None and exposure_scale != 1.0:
                logger.info(
                    "SNAP: Applied WB with intensity scaling for %.2f deg " "(scale=%.2fx)",
                    angle,
                    exposure_scale,
                )
            else:
                logger.info("SNAP: Applied per-angle white balance for %.2f deg", angle)
        else:
            logger.warning("SNAP: Failed to apply white balance for %.2f deg", angle)
        return wb_applied

    except ImportError as e:
        logger.warning("SNAP: White balance modules not available: %s", e)
        return False
    except Exception as e:
        logger.warning("SNAP: Error loading white balance calibration: %s", e)
        return False
