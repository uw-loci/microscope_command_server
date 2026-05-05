"""Calibration command handlers.

Handles white balance, polarizer, PPM, sunburst calibration, and noise
measurement commands:
WBCALIBR, WBSIMPLE, WBPPM, POLCAL, PPMSENS, PPMBIREF, SBCALIB,
GETNOISE, NOISCHAR

NOTE: Calibration handlers (WBCALIBR, WBSIMPLE, WBPPM, NOISCHAR) access
JAI camera properties via hardware.camera.properties (the lazily-initialized
JAICameraProperties instance on the JAICamera). All direct camera property
manipulation (finally-block cleanup) uses the Camera ABC methods instead.
"""

import socket
import struct
import time
import logging

import numpy as np

from microscope_control.hardware import Position
from microscope_command_server.server.protocol import END_MARKER, TCP_PORT
from microscope_command_server.server.handlers.utils import read_message_string

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WBCALIBR - Legacy white balance calibration
# ---------------------------------------------------------------------------

def handle_wbcalibr(conn, client, hardware, settings, **kwargs):
    """White balance calibration (legacy JAI per-channel).

    Reads --yaml, --output, --modality, --objective, --target,
    --tolerance, --defocus flags.  Runs JAIWhiteBalanceCalibrator.calibrate().

    Post-calibration cleanup resets per-channel mode so subsequent
    operations (autofocus, SNAP, acquisition) can use unified
    set_exposure() correctly.

    Response: SUCCESS:<status>|<path>|<exposures>|<gains>
              or FAILED:<reason>
    """
    config_manager = kwargs.get("config_manager")
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested white balance calibration", addr)

    # Read the message
    message_parts = []
    total_bytes = 0
    start_time = time.time()

    conn.settimeout(5.0)

    try:
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                logger.error(
                    "Connection closed while reading white balance message"
                )
                conn.sendall(b"FAILED:Connection closed")
                break

            message_parts.append(chunk.decode("utf-8"))
            total_bytes += len(chunk)
            logger.debug("WBCALIBR: received %d bytes so far", total_bytes)

            full_message = "".join(message_parts)

            if END_MARKER in full_message:
                message = full_message.replace(END_MARKER, "").strip()
                logger.info("WBCALIBR message: %s", message)

                # Parse the message
                params = {}

                flags = [
                    "--yaml",
                    "--output",
                    "--modality",
                    "--objective",
                    "--target",
                    "--tolerance",
                    "--defocus",
                ]

                for i, flag in enumerate(flags):
                    if flag in message:
                        start_idx = message.index(flag) + len(flag)
                        end_idx = len(message)
                        # Find the CLOSEST next flag (check all flags, not just remaining)
                        for next_flag in flags:
                            if next_flag != flag and next_flag in message[start_idx:]:
                                next_pos = message.index(next_flag, start_idx)
                                if next_pos < end_idx:
                                    end_idx = next_pos

                        value = message[start_idx:end_idx].strip()

                        if flag == "--yaml":
                            params["yaml_file_path"] = value
                        elif flag == "--output":
                            params["output_folder_path"] = value
                        elif flag == "--modality":
                            params["modality"] = value
                        elif flag == "--objective":
                            params["objective"] = value
                        elif flag == "--target":
                            params["target_intensity"] = float(value)
                        elif flag == "--tolerance":
                            params["tolerance"] = float(value)
                        elif flag == "--defocus":
                            params["defocus_um"] = float(value)

                # Validate required parameters
                required = ["yaml_file_path", "output_folder_path", "modality"]
                missing = [key for key in required if key not in params]
                if missing:
                    error_msg = f"Missing required parameters: {missing}"
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                    break

                # Send immediate acknowledgment to prevent client timeout
                try:
                    ack_response = f"STARTED:{params['output_folder_path']}".encode()
                    conn.sendall(ack_response)
                    logger.info(
                        "Sent STARTED acknowledgment for white balance calibration"
                    )

                    # Import the calibration module
                    # TODO: JAI-specific -- migrate to camera-agnostic interface
                    from microscope_control.jai import (
                        JAIWhiteBalanceCalibrator,
                        CalibrationConfig,
                    )
                    from pathlib import Path
                    from microscope_command_server.modality import (
                        get_config as get_modality_config,
                    )

                    # Build calibration config
                    wb_config = CalibrationConfig(
                        target_value=params.get("target_intensity", 180.0),
                        tolerance=params.get("tolerance", 5.0),
                        defocus_offset_um=params.get("defocus_um"),
                    )

                    # Create calibrator with hardware
                    jai_props = hardware.camera.properties
                    calibrator = JAIWhiteBalanceCalibrator(hardware, jai_props)

                    # Set up rotation callback if modality has rotation
                    mod_config = get_modality_config(params["modality"])
                    rotation_callback = None
                    if mod_config.has_rotation and hasattr(hardware, "set_psg_ticks"):
                        rotation_callback = hardware.set_psg_ticks

                    # Set up defocus callback if configured
                    defocus_callback = None
                    if wb_config.defocus_offset_um is not None:
                        def create_defocus_callback():
                            def defocus_fn(offset_um):
                                current_pos = hardware.get_current_position()
                                original_z = current_pos.z
                                new_z = original_z + offset_um
                                hardware.move_to_position(
                                    Position(hardware.get_current_position().x, hardware.get_current_position().y, new_z)
                                )
                                def restore():
                                    hardware.move_to_position(
                                        Position(hardware.get_current_position().x, hardware.get_current_position().y, original_z)
                                    )
                                return original_z, restore
                            return defocus_fn
                        defocus_callback = create_defocus_callback()

                    # Run calibration
                    output_path = Path(params["output_folder_path"])
                    result = calibrator.calibrate(
                        config=wb_config,
                        output_path=output_path,
                        rotation_callback=rotation_callback,
                        defocus_callback=defocus_callback,
                    )

                    # Format response
                    exp_str = (
                        f"exp_r:{result.exposures_ms['red']:.2f},"
                        f"exp_g:{result.exposures_ms['green']:.2f},"
                        f"exp_b:{result.exposures_ms['blue']:.2f}"
                    )
                    gain_str = (
                        f"gain_r:{result.gains['red']:.2f},"
                        f"gain_g:{result.gains['green']:.2f},"
                        f"gain_b:{result.gains['blue']:.2f}"
                    )
                    status = "CONVERGED" if result.converged else "NOT_CONVERGED"

                    response = f"SUCCESS:{status}|{output_path}|{exp_str}|{gain_str}"
                    conn.sendall(response.encode())
                    logger.info("White balance calibration completed: %s", status)

                except ImportError as e:
                    error_msg = f"JAI calibration module not available: {e}"
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                except Exception as e:
                    error_msg = f"White balance calibration failed: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                break

            if total_bytes > 100000:
                logger.error(
                    "White balance message exceeds maximum size"
                )
                conn.sendall(b"FAILED:Message too large")
                break

            if time.time() - start_time > 10:
                logger.error("Timeout reading white balance message")
                conn.sendall(b"FAILED:Timeout waiting for complete message")
                break

    except socket.timeout:
        logger.error("Timeout reading white balance message from %s", addr)
        conn.sendall(b"FAILED:Timeout reading message")
    except Exception as e:
        logger.error("Error in white balance calibration: %s", str(e), exc_info=True)
        conn.sendall(f"FAILED:{str(e)}".encode())
    finally:
        conn.settimeout(None)  # Reset to blocking mode
        # Reset per-channel mode so subsequent operations (autofocus,
        # SNAP, acquisition) can use unified set_exposure() correctly
        try:
            cam = hardware.camera
            cam.disable_individual_exposure()
            cam.disable_individual_gain()
            cam.set_rb_analog_gains(analog_red=1.0, analog_blue=1.0)
            logger.debug("Reset per-channel mode after WBCALIBR")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WBSIMPLE - Simple white balance (unified exposure per angle)
# ---------------------------------------------------------------------------

def handle_wbsimple(conn, client, hardware, settings, **kwargs):
    """Simple white balance calibration (JAI per-channel).

    Calibrates uncrossed angle first with per-channel exposure/gain,
    then calibrates remaining PPM angles using unified exposure mode
    while keeping the uncrossed analog gains for color balance.

    Post-calibration: applies result to camera for live view, or
    resets on failure.

    Response: SUCCESS:<path>|<status>|<exposures>|<gains>|angles:<n>
              or FAILED:<reason>
    """
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested simple white balance calibration", addr)

    # Track calibration result so the finally block can apply it
    # to the camera for live view (instead of resetting to defaults).
    _wb_calibration_result = None

    # Read the message using the same pattern as WBCALIBR
    message_parts = []
    total_bytes = 0
    start_time = time.time()

    conn.settimeout(5.0)

    try:
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                logger.error(
                    "Connection closed while reading WBSIMPLE message"
                )
                conn.sendall(b"FAILED:Connection closed")
                break

            message_parts.append(chunk.decode("utf-8"))
            total_bytes += len(chunk)
            logger.debug("WBSIMPLE: received %d bytes so far", total_bytes)

            full_message = "".join(message_parts)

            if END_MARKER in full_message:
                message = full_message.replace(END_MARKER, "").strip()
                logger.info("WBSIMPLE message: %s", message)

                # Parse the message
                params = {}

                flags = [
                    "--yaml",
                    "--objective",
                    "--detector",
                    "--output",
                    "--modality",
                    "--camera",
                    "--exposure",
                    "--target",
                    "--tolerance",
                    "--max_gain_db",
                    "--gain_threshold",
                    "--max_iterations",
                    "--calibrate_black_level",
                    "--base_gain",
                    "--exposure_soft_cap_ms",
                    "--boosted_max_gain_db",
                    "--gain_analog_rb_max",
                    "--target_positive",
                    "--target_negative",
                    "--target_crossed",
                ]

                # Helper to find a flag as a complete word (followed by space)
                def find_flag_position(msg, flag):
                    """Find flag position ensuring it's followed by a space."""
                    search_pattern = flag + " "
                    if search_pattern in msg:
                        return msg.index(search_pattern)
                    return -1

                for i, flag in enumerate(flags):
                    flag_pos = find_flag_position(message, flag)
                    if flag_pos >= 0:
                        start_idx = flag_pos + len(flag)
                        end_idx = len(message)
                        # Find the CLOSEST next flag
                        for next_flag in flags:
                            if next_flag != flag:
                                next_pos = find_flag_position(message[start_idx:], next_flag)
                                if next_pos >= 0:
                                    actual_pos = start_idx + next_pos
                                    if actual_pos < end_idx:
                                        end_idx = actual_pos

                        value = message[start_idx:end_idx].strip()

                        if flag == "--yaml":
                            params["yaml_file_path"] = value
                        elif flag == "--objective":
                            params["objective"] = value
                        elif flag == "--detector":
                            params["detector"] = value
                        elif flag == "--output":
                            params["output_folder_path"] = value
                        elif flag == "--modality":
                            params["modality"] = value
                        elif flag == "--camera":
                            params["camera"] = value
                        elif flag == "--exposure":
                            params["initial_exposure_ms"] = float(value)
                        elif flag == "--target":
                            params["target_intensity"] = float(value)
                        elif flag == "--tolerance":
                            params["tolerance"] = float(value)
                        elif flag == "--max_gain_db":
                            params["max_gain_db"] = float(value)
                        elif flag == "--gain_threshold":
                            params["gain_threshold"] = float(value)
                        elif flag == "--max_iterations":
                            params["max_iterations"] = int(value)
                        elif flag == "--calibrate_black_level":
                            params["calibrate_black_level"] = value.lower() == "true"
                        elif flag == "--base_gain":
                            params["base_gain"] = float(value)
                        elif flag == "--exposure_soft_cap_ms":
                            params["exposure_soft_cap_ms"] = float(value)
                        elif flag == "--boosted_max_gain_db":
                            params["boosted_max_gain_db"] = float(value)
                        elif flag == "--gain_analog_rb_max":
                            params["gain_analog_rb_max"] = float(value)
                        elif flag == "--target_positive":
                            params["target_positive"] = float(value)
                        elif flag == "--target_negative":
                            params["target_negative"] = float(value)
                        elif flag == "--target_crossed":
                            params["target_crossed"] = float(value)

                # Validate required parameters
                required = ["output_folder_path", "initial_exposure_ms"]
                missing = [key for key in required if key not in params]
                if missing:
                    error_msg = f"Missing required parameters: {missing}"
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                    break

                # Send immediate acknowledgment
                try:
                    ack_response = f"STARTED:{params['output_folder_path']}".encode()
                    conn.sendall(ack_response)
                    logger.info(
                        "Sent STARTED acknowledgment for WBSIMPLE"
                    )

                    # Import the calibration module
                    # TODO: JAI-specific -- migrate to camera-agnostic interface
                    from microscope_control.jai import (
                        JAIWhiteBalanceCalibrator,
                    )
                    from pathlib import Path

                    # Create calibrator with hardware
                    jai_props = hardware.camera.properties
                    calibrator = JAIWhiteBalanceCalibrator(hardware, jai_props)

                    # Run simple calibration at uncrossed (90 deg) first
                    output_path = Path(params["output_folder_path"])
                    calib_kwargs = dict(
                        target=params.get("target_intensity", 180.0),
                        tolerance=params.get("tolerance", 5.0),
                        output_path=output_path,
                        gain_threshold_ratio=params.get("gain_threshold"),
                        max_iterations=params.get("max_iterations"),
                        calibrate_black_level=params.get("calibrate_black_level"),
                        base_gain=params.get("base_gain"),
                        gain_analog_rb_max=params.get("gain_analog_rb_max"),
                    )

                    # Rotate to uncrossed (90 deg) for first calibration
                    if hasattr(hardware, "set_psg_ticks"):
                        hardware.set_psg_ticks(90)
                        logger.info("Simple WB: rotated to uncrossed (90 deg)")

                    uncrossed_result = calibrator.calibrate_simple(
                        initial_exposure_ms=params["initial_exposure_ms"],
                        **calib_kwargs,
                    )
                    logger.info(
                        "Simple WB uncrossed: R=%.2fms, G=%.2fms, B=%.2fms, converged=%s",
                        uncrossed_result.exposures_ms['red'],
                        uncrossed_result.exposures_ms['green'],
                        uncrossed_result.exposures_ms['blue'],
                        uncrossed_result.converged,
                    )

                    # Collect all angle results (uncrossed + remaining angles)
                    all_results = {"uncrossed": uncrossed_result}

                    # Load remaining PPM rotation angles from config
                    remaining_angles = []
                    try:
                        modality_name = params.get("modality", "ppm")
                        modalities = hardware.settings.get("modalities", {})
                        mod_config = modalities.get(modality_name, {})
                        rotation_angles = mod_config.get("rotation_angles", [])
                        for ra in rotation_angles:
                            name = ra.get("name")
                            tick = ra.get("tick")
                            if name and tick is not None and name != "uncrossed":
                                remaining_angles.append((name, float(tick)))
                        logger.info(
                            "Simple WB: will calibrate %d additional angles: %s",
                            len(remaining_angles),
                            [a[0] for a in remaining_angles],
                        )
                    except Exception as e:
                        logger.warning("Could not load rotation angles from config: %s", e)

                    # Calibrate remaining angles using UNIFIED exposure mode.
                    # Keep analog R/B gains from uncrossed calibration for color balance.
                    # Only adjust a single exposure time to reach target average intensity.
                    # This prevents the per-channel exposure * analog gain compounding
                    # that causes blue saturation at small angles.
                    from microscope_command_server.acquisition.workflow import (
                        get_target_intensity_for_angle,
                    )
                    from microscope_control.jai.calibration import WhiteBalanceResult

                    for angle_name, angle_deg in remaining_angles:
                        logger.info(
                            "Simple WB: calibrating %s (%.1f deg) with unified exposure...",
                            angle_name, angle_deg,
                        )
                        if hasattr(hardware, "set_psg_ticks"):
                            hardware.set_psg_ticks(angle_deg)

                        # Get per-angle target intensity
                        client_target_key = f"target_{angle_name}"
                        if client_target_key in params:
                            angle_target = params[client_target_key]
                            logger.info("  Target for %s: %.1f (from client)", angle_name, angle_target)
                        elif "yaml_file_path" in params:
                            angle_target = calib_kwargs["target"]
                            try:
                                val, src = get_target_intensity_for_angle(
                                    angle=angle_deg,
                                    modality=params.get("modality", "ppm"),
                                    config_path=Path(params["yaml_file_path"]),
                                )
                                angle_target = val
                                logger.info("  Target for %s: %.1f (from %s)", angle_name, val, src)
                            except Exception:
                                pass
                        else:
                            angle_target = calib_kwargs["target"]

                        try:
                            # Switch to unified exposure mode (single exposure for all channels)
                            # Keep analog gains from uncrossed calibration for color balance
                            jai_props.disable_individual_exposure()
                            jai_props.set_unified_gain(uncrossed_result.unified_gain)
                            jai_props.set_rb_analog_gains(
                                red=uncrossed_result.analog_red,
                                blue=uncrossed_result.analog_blue)
                            logger.info(
                                "  Unified mode: gain=%.2f, aR=%.3f, aB=%.3f",
                                uncrossed_result.unified_gain,
                                uncrossed_result.analog_red,
                                uncrossed_result.analog_blue,
                            )

                            # Start with uncrossed green exposure as initial guess
                            exposure_ms = uncrossed_result.exposures_ms["green"]
                            tolerance = calib_kwargs.get("tolerance", 5.0)
                            converged = False
                            max_iter = 15

                            for iteration in range(max_iter):
                                hardware.set_exposure(exposure_ms)
                                image, metadata = hardware.snap_image()
                                if image is None:
                                    raise RuntimeError("Failed to snap image")
                                measured = float(np.mean(image))
                                logger.info(
                                    "  Iter %d: exp=%.2fms, mean=%.1f (target=%.1f)",
                                    iteration, exposure_ms, measured, angle_target,
                                )
                                if abs(measured - angle_target) <= tolerance:
                                    converged = True
                                    logger.info("  Converged at iteration %d", iteration)
                                    break
                                if measured < 1.0:
                                    exposure_ms *= 5.0
                                else:
                                    exposure_ms *= angle_target / measured

                            # Build result with unified exposure for all channels
                            # (R/G/B all get the same exposure in unified mode)
                            angle_result = WhiteBalanceResult(
                                exposures_ms={"red": exposure_ms, "green": exposure_ms, "blue": exposure_ms},
                                black_levels={"red": 0, "green": 0, "blue": 0},
                                final_means={"red": measured, "green": measured, "blue": measured},
                                target_value=angle_target,
                                unified_gain=uncrossed_result.unified_gain,
                                analog_red=uncrossed_result.analog_red,
                                analog_blue=uncrossed_result.analog_blue,
                                wb_method="manual_simple",
                                converged=converged,
                                iterations=iteration + 1,
                            )
                            all_results[angle_name] = angle_result
                            logger.info(
                                "  %s: R=%.2fms, G=%.2fms, B=%.2fms, converged=%s",
                                angle_name,
                                angle_result.exposures_ms['red'],
                                angle_result.exposures_ms['green'],
                                angle_result.exposures_ms['blue'],
                                angle_result.converged,
                            )
                        except Exception as e:
                            logger.error("  Failed to calibrate %s: %s", angle_name, e)

                    # Save all angle results to imageprocessing config
                    if "yaml_file_path" in params:
                        wb_objective = params.get("objective")
                        wb_detector = params.get("detector")
                        wb_modality = params.get("modality", "ppm")
                        logger.info(
                            "Simple WB: saving %d angle(s) with objective=%s, detector=%s",
                            len(all_results), wb_objective, wb_detector,
                        )
                        for aname, aresult in all_results.items():
                            calibrator.update_imageprocessing_config(
                                config_path=Path(params["yaml_file_path"]),
                                result=aresult,
                                calibration_type="simple",
                                angle_name=aname,
                                modality=wb_modality,
                                objective=wb_objective,
                                detector=wb_detector,
                            )

                        # Also save simple_wb section with calibration values.
                        # This is the source of truth for Simple WB acquisition.
                        # Background collection does NOT overwrite these values,
                        # so they remain correct even if PPM WB later overwrites
                        # the shared exposures_ms section.
                        from microscope_command_server.acquisition.workflow import (
                            save_simple_wb_to_yaml,
                        )
                        uncrossed_for_base = all_results.get("uncrossed")
                        if uncrossed_for_base:
                            simple_wb_cal_results = {}
                            uncrossed_g = uncrossed_for_base.exposures_ms["green"]
                            for aname, aresult in all_results.items():
                                scale = (aresult.exposures_ms["green"] / uncrossed_g
                                         if uncrossed_g > 0 else 1.0)
                                simple_wb_cal_results[aname] = {
                                    "scale": round(scale, 3),
                                    "unified_gain": round(aresult.unified_gain, 3),
                                    "r": round(aresult.exposures_ms["red"], 2),
                                    "g": round(aresult.exposures_ms["green"], 2),
                                    "b": round(aresult.exposures_ms["blue"], 2),
                                }
                            base_exp = {
                                "r": round(uncrossed_for_base.exposures_ms["red"], 2),
                                "g": round(uncrossed_for_base.exposures_ms["green"], 2),
                                "b": round(uncrossed_for_base.exposures_ms["blue"], 2),
                                "unified_exposure_ms": round(
                                    uncrossed_for_base.exposures_ms["green"], 2),
                                "gains": {
                                    "unified_gain": round(uncrossed_for_base.unified_gain, 3),
                                    "analog_red": round(uncrossed_for_base.analog_red, 3),
                                    "analog_blue": round(uncrossed_for_base.analog_blue, 3),
                                    "wb_method": uncrossed_for_base.wb_method,
                                },
                            }
                            save_simple_wb_to_yaml(
                                config_path=Path(params["yaml_file_path"]),
                                simple_wb_results=simple_wb_cal_results,
                                base_exposures=base_exp,
                                modality=wb_modality,
                                objective=wb_objective,
                                detector=wb_detector,
                                logger=logger,
                            )
                            logger.info("Simple WB: saved calibration values to simple_wb section")

                    # Format response: uncrossed result for backward compatibility
                    result = uncrossed_result
                    exp_str = (
                        f"exp_r:{result.exposures_ms['red']:.2f},"
                        f"exp_g:{result.exposures_ms['green']:.2f},"
                        f"exp_b:{result.exposures_ms['blue']:.2f}"
                    )
                    gain_str = (
                        f"unified:{result.unified_gain:.3f},"
                        f"analog_r:{result.analog_red:.3f},"
                        f"analog_b:{result.analog_blue:.3f}"
                    )
                    all_converged = all(r.converged for r in all_results.values())
                    status = "CONVERGED" if all_converged else "NOT_CONVERGED"
                    n_angles = len(all_results)

                    response = f"SUCCESS:{output_path}|{status}|{exp_str}|{gain_str}|angles:{n_angles}"

                    # Append noise stats from uncrossed if available
                    if result.noise_stats is not None:
                        ns = result.noise_stats
                        response += (
                            f"|noise_r:{ns.channel_stddevs['red']:.2f},"
                            f"noise_g:{ns.channel_stddevs['green']:.2f},"
                            f"noise_b:{ns.channel_stddevs['blue']:.2f}"
                        )

                    conn.sendall(response.encode())
                    logger.info("WBSIMPLE completed: %d angles, all_converged=%s", n_angles, all_converged)
                    _wb_calibration_result = result

                except ImportError as e:
                    error_msg = f"JAI calibration module not available: {e}"
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                except Exception as e:
                    error_msg = f"WBSIMPLE failed: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                break

            if total_bytes > 100000:
                logger.error("WBSIMPLE message exceeds maximum size")
                conn.sendall(b"FAILED:Message too large")
                break

            if time.time() - start_time > 10:
                logger.error("Timeout reading WBSIMPLE message")
                conn.sendall(b"FAILED:Timeout waiting for complete message")
                break

    except socket.timeout:
        logger.error("Timeout reading WBSIMPLE message from %s", addr)
        conn.sendall(b"FAILED:Timeout reading message")
    except Exception as e:
        logger.error("Error in WBSIMPLE: %s", str(e), exc_info=True)
        conn.sendall(f"FAILED:{str(e)}".encode())
    finally:
        conn.settimeout(None)  # Reset to blocking mode
        # Apply calibration result to camera so live view shows
        # the white-balanced image, or reset on failure.
        try:
            cam = hardware.camera
            if (_wb_calibration_result is not None
                    and _wb_calibration_result.converged):
                cam.set_channel_exposures(
                    red=_wb_calibration_result.exposures_ms['red'],
                    green=_wb_calibration_result.exposures_ms['green'],
                    blue=_wb_calibration_result.exposures_ms['blue'],
                    auto_enable=True,
                )
                cam.set_unified_gain(
                    _wb_calibration_result.unified_gain)
                cam.set_rb_analog_gains(
                    analog_red=_wb_calibration_result.analog_red,
                    analog_blue=_wb_calibration_result.analog_blue)
                logger.info(
                    "Applied calibration to camera for live view: "
                    "R=%.2f G=%.2f B=%.2f, "
                    "unified=%.2f, aR=%.3f, aB=%.3f",
                    _wb_calibration_result.exposures_ms['red'],
                    _wb_calibration_result.exposures_ms['green'],
                    _wb_calibration_result.exposures_ms['blue'],
                    _wb_calibration_result.unified_gain,
                    _wb_calibration_result.analog_red,
                    _wb_calibration_result.analog_blue,
                )
            else:
                # Reset to clean state on failure
                cam.set_rb_analog_gains(analog_red=1.0, analog_blue=1.0)
                cam.set_unified_gain(1.0)
                cam.disable_individual_exposure()
                logger.debug("Reset camera state after WBSIMPLE "
                             "(calibration did not converge)")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WBPPM - PPM white balance (4 angles)
# ---------------------------------------------------------------------------

def handle_wbppm(conn, client, hardware, settings, **kwargs):
    """PPM white balance calibration at 4 polarizer angles.

    Per-channel exposure/gain calibration repeated at each of 4 PPM
    polarizer angles. Each angle may have a different target intensity.

    Post-calibration: applies uncrossed result to camera for live view,
    or resets on failure.

    Response: SUCCESS:<path>|<angle>:<exps>:<gains>:<Y/N>|...
              or FAILED:<reason>
    """
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested PPM white balance calibration (4 angles)", addr)

    # Track calibration results so the finally block can apply them
    _wb_rotation_results = None

    # Read the message
    message_parts = []
    total_bytes = 0
    start_time = time.time()

    conn.settimeout(5.0)

    try:
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                logger.error(
                    "Connection closed while reading WBPPM message"
                )
                conn.sendall(b"FAILED:Connection closed")
                break

            message_parts.append(chunk.decode("utf-8"))
            total_bytes += len(chunk)
            logger.debug("WBPPM: received %d bytes so far", total_bytes)

            full_message = "".join(message_parts)

            if END_MARKER in full_message:
                message = full_message.replace(END_MARKER, "").strip()
                logger.info("WBPPM message: %s", message)

                # Parse the message
                params = {}

                flags = [
                    "--yaml",
                    "--objective",
                    "--detector",
                    "--output",
                    "--camera",
                    "--positive_exp",
                    "--positive_angle",
                    "--target_positive",
                    "--negative_exp",
                    "--negative_angle",
                    "--target_negative",
                    "--crossed_exp",
                    "--crossed_angle",
                    "--target_crossed",
                    "--uncrossed_exp",
                    "--uncrossed_angle",
                    "--target_uncrossed",
                    "--target",
                    "--tolerance",
                    "--max_gain_db",
                    "--gain_threshold",
                    "--max_iterations",
                    "--calibrate_black_level",
                    "--base_gain",
                    "--exposure_soft_cap_ms",
                    "--boosted_max_gain_db",
                    "--gain_analog_rb_max",
                ]

                # Helper to find a flag as a complete word (followed by space)
                def find_flag_position(msg, flag):
                    """Find flag position ensuring it's followed by a space."""
                    search_pattern = flag + " "
                    if search_pattern in msg:
                        return msg.index(search_pattern)
                    return -1

                for i, flag in enumerate(flags):
                    flag_pos = find_flag_position(message, flag)
                    if flag_pos >= 0:
                        start_idx = flag_pos + len(flag)
                        end_idx = len(message)
                        # Find the CLOSEST next flag
                        for next_flag in flags:
                            if next_flag != flag:
                                next_pos = find_flag_position(message[start_idx:], next_flag)
                                if next_pos >= 0:
                                    actual_pos = start_idx + next_pos
                                    if actual_pos < end_idx:
                                        end_idx = actual_pos

                        value = message[start_idx:end_idx].strip()

                        if flag == "--yaml":
                            params["yaml_file_path"] = value
                        elif flag == "--objective":
                            params["objective"] = value
                        elif flag == "--detector":
                            params["detector"] = value
                        elif flag == "--output":
                            params["output_folder_path"] = value
                        elif flag == "--camera":
                            params["camera"] = value
                        elif flag == "--positive_exp":
                            params["positive_exp"] = float(value)
                        elif flag == "--positive_angle":
                            params["positive_angle"] = float(value)
                        elif flag == "--target_positive":
                            params["target_positive"] = float(value)
                        elif flag == "--negative_exp":
                            params["negative_exp"] = float(value)
                        elif flag == "--negative_angle":
                            params["negative_angle"] = float(value)
                        elif flag == "--target_negative":
                            params["target_negative"] = float(value)
                        elif flag == "--crossed_exp":
                            params["crossed_exp"] = float(value)
                        elif flag == "--crossed_angle":
                            params["crossed_angle"] = float(value)
                        elif flag == "--target_crossed":
                            params["target_crossed"] = float(value)
                        elif flag == "--uncrossed_exp":
                            params["uncrossed_exp"] = float(value)
                        elif flag == "--uncrossed_angle":
                            params["uncrossed_angle"] = float(value)
                        elif flag == "--target_uncrossed":
                            params["target_uncrossed"] = float(value)
                        elif flag == "--target":
                            params["target_intensity"] = float(value)
                        elif flag == "--tolerance":
                            params["tolerance"] = float(value)
                        elif flag == "--max_gain_db":
                            params["max_gain_db"] = float(value)
                        elif flag == "--gain_threshold":
                            params["gain_threshold"] = float(value)
                        elif flag == "--max_iterations":
                            params["max_iterations"] = int(value)
                        elif flag == "--calibrate_black_level":
                            params["calibrate_black_level"] = value.lower() == "true"
                        elif flag == "--base_gain":
                            params["base_gain"] = float(value)
                        elif flag == "--exposure_soft_cap_ms":
                            params["exposure_soft_cap_ms"] = float(value)
                        elif flag == "--boosted_max_gain_db":
                            params["boosted_max_gain_db"] = float(value)
                        elif flag == "--gain_analog_rb_max":
                            params["gain_analog_rb_max"] = float(value)

                # Validate required parameters
                required = [
                    "output_folder_path",
                    "positive_exp", "positive_angle",
                    "negative_exp", "negative_angle",
                    "crossed_exp", "crossed_angle",
                    "uncrossed_exp", "uncrossed_angle",
                ]
                missing = [key for key in required if key not in params]
                if missing:
                    error_msg = f"Missing required parameters: {missing}"
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                    break

                # Send immediate acknowledgment
                try:
                    ack_response = f"STARTED:{params['output_folder_path']}".encode()
                    conn.sendall(ack_response)
                    logger.info(
                        "Sent STARTED acknowledgment for WBPPM"
                    )

                    # Import the calibration module
                    # TODO: JAI-specific -- migrate to camera-agnostic interface
                    from microscope_control.jai import (
                        JAIWhiteBalanceCalibrator,
                    )
                    from pathlib import Path

                    # Build angle/exposure pairs
                    angle_exposures = {
                        "positive": (params["positive_angle"], params["positive_exp"]),
                        "negative": (params["negative_angle"], params["negative_exp"]),
                        "crossed": (params["crossed_angle"], params["crossed_exp"]),
                        "uncrossed": (params["uncrossed_angle"], params["uncrossed_exp"]),
                    }

                    # Build per-angle targets dictionary
                    # Priority: client-provided > YAML background_exposures > YAML target_intensities > default
                    per_angle_targets = {}

                    # Check if client provided per-angle targets
                    client_targets = {
                        "positive": params.get("target_positive"),
                        "negative": params.get("target_negative"),
                        "crossed": params.get("target_crossed"),
                        "uncrossed": params.get("target_uncrossed"),
                    }

                    # Load targets from YAML if not provided by client
                    yaml_targets_loaded = False
                    if "yaml_file_path" in params:
                        try:
                            from microscope_command_server.acquisition.workflow import (
                                get_target_intensity_for_angle,
                            )
                            for angle_name in ["positive", "negative", "crossed", "uncrossed"]:
                                if client_targets[angle_name] is not None:
                                    # Client provided explicit value
                                    per_angle_targets[angle_name] = client_targets[angle_name]
                                else:
                                    # Try YAML lookup
                                    angle_deg = params[f"{angle_name}_angle"]
                                    target_val, source = get_target_intensity_for_angle(
                                        angle=angle_deg,
                                        modality=params.get("modality", "ppm"),
                                        config_path=Path(params["yaml_file_path"]),
                                    )
                                    per_angle_targets[angle_name] = target_val
                                    logger.info(
                                        "WB target for %s: %s (from %s)",
                                        angle_name, target_val, source,
                                    )
                            yaml_targets_loaded = True
                        except Exception as e:
                            logger.warning("Failed to load targets from YAML: %s", e)

                    # If YAML loading failed, use client values or None
                    if not yaml_targets_loaded:
                        for angle_name in ["positive", "negative", "crossed", "uncrossed"]:
                            if client_targets[angle_name] is not None:
                                per_angle_targets[angle_name] = client_targets[angle_name]

                    # Create calibrator with hardware
                    jai_props = hardware.camera.properties
                    calibrator = JAIWhiteBalanceCalibrator(hardware, jai_props)

                    # Set up rotation callback
                    rotation_callback = None
                    if hasattr(hardware, "set_psg_ticks"):
                        rotation_callback = hardware.set_psg_ticks

                    # Run per-angle calibration
                    output_path = Path(params["output_folder_path"])
                    results = calibrator.calibrate_ppm(
                        angle_exposures=angle_exposures,
                        target=params.get("target_intensity", 180.0),
                        tolerance=params.get("tolerance", 5.0),
                        output_path=output_path,
                        rotation_callback=rotation_callback,
                        per_angle_targets=per_angle_targets if per_angle_targets else None,
                        gain_threshold_ratio=params.get("gain_threshold"),
                        max_iterations=params.get("max_iterations"),
                        calibrate_black_level=params.get("calibrate_black_level"),
                        base_gain=params.get("base_gain"),
                        gain_analog_rb_max=params.get("gain_analog_rb_max"),
                    )

                    # Update imageprocessing config for each angle
                    if "yaml_file_path" in params:
                        # Get objective/detector from command params (preferred) or hardware.settings (fallback)
                        wb_objective = params.get("objective")
                        wb_detector = params.get("detector")
                        if not wb_objective or not wb_detector:
                            if hasattr(hardware, 'settings') and hardware.settings:
                                wb_objective = wb_objective or hardware.settings.get("objective_in_use") or hardware.settings.get("objective")
                                wb_detector = wb_detector or hardware.settings.get("detector_in_use") or hardware.settings.get("detector")
                        logger.info(
                            "WB calibration: saving to imaging_profiles with objective=%s, detector=%s",
                            wb_objective, wb_detector,
                        )

                        wb_modality = params.get("modality", "ppm")
                        for angle_name, result in results.items():
                            calibrator.update_imageprocessing_config(
                                config_path=Path(params["yaml_file_path"]),
                                result=result,
                                calibration_type="per_angle",
                                angle_name=angle_name,
                                modality=wb_modality,
                                objective=wb_objective,
                                detector=wb_detector,
                            )

                    # Format response with results for all angles
                    # Format: SUCCESS:path|angle:exp_r,exp_g,exp_b:unified,aR,aB:Y/N|...
                    response_parts = [f"SUCCESS:{output_path}"]
                    all_converged = True
                    for name, result in results.items():
                        exp_str = (
                            f"{result.exposures_ms['red']:.2f},"
                            f"{result.exposures_ms['green']:.2f},"
                            f"{result.exposures_ms['blue']:.2f}"
                        )
                        gain_str = (
                            f"{result.unified_gain:.3f},"
                            f"{result.analog_red:.3f},"
                            f"{result.analog_blue:.3f}"
                        )
                        converged = "Y" if result.converged else "N"
                        response_parts.append(f"{name}:{exp_str}:{gain_str}:{converged}")
                        if not result.converged:
                            all_converged = False

                    response = "|".join(response_parts)
                    conn.sendall(response.encode())
                    logger.info("WBPPM completed: all_converged=%s", all_converged)
                    _wb_rotation_results = results

                except ImportError as e:
                    error_msg = f"JAI calibration module not available: {e}"
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                except Exception as e:
                    error_msg = f"WBPPM failed: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                break

            if total_bytes > 100000:
                logger.error("WBPPM message exceeds maximum size")
                conn.sendall(b"FAILED:Message too large")
                break

            if time.time() - start_time > 10:
                logger.error("Timeout reading WBPPM message")
                conn.sendall(b"FAILED:Timeout waiting for complete message")
                break

    except socket.timeout:
        logger.error("Timeout reading WBPPM message from %s", addr)
        conn.sendall(b"FAILED:Timeout reading message")
    except Exception as e:
        logger.error("Error in WBPPM: %s", str(e), exc_info=True)
        conn.sendall(f"FAILED:{str(e)}".encode())
    finally:
        conn.settimeout(None)  # Reset to blocking mode
        # Apply uncrossed calibration to camera so live view shows
        # the white-balanced image, or reset on failure.
        try:
            cam = hardware.camera
            # Use uncrossed (90 deg) result for live view -- it is
            # the brightest angle and most natural for visual QC.
            uncrossed = (
                _wb_rotation_results.get("uncrossed")
                if _wb_rotation_results else None
            )
            if uncrossed is not None and uncrossed.converged:
                cam.set_channel_exposures(
                    red=uncrossed.exposures_ms['red'],
                    green=uncrossed.exposures_ms['green'],
                    blue=uncrossed.exposures_ms['blue'],
                    auto_enable=True,
                )
                cam.set_unified_gain(uncrossed.unified_gain)
                cam.set_rb_analog_gains(
                    analog_red=uncrossed.analog_red,
                    analog_blue=uncrossed.analog_blue)
                logger.info(
                    "Applied uncrossed calibration to camera for "
                    "live view: R=%.2f G=%.2f B=%.2f, "
                    "unified=%.2f, aR=%.3f, aB=%.3f",
                    uncrossed.exposures_ms['red'],
                    uncrossed.exposures_ms['green'],
                    uncrossed.exposures_ms['blue'],
                    uncrossed.unified_gain,
                    uncrossed.analog_red,
                    uncrossed.analog_blue,
                )
            else:
                # Reset to clean state on failure
                cam.set_rb_analog_gains(analog_red=1.0, analog_blue=1.0)
                cam.set_unified_gain(1.0)
                cam.disable_individual_exposure()
                logger.debug("Reset camera state after WBPPM "
                             "(no uncrossed result to apply)")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# POLCAL - Polarizer calibration
# ---------------------------------------------------------------------------

def handle_polcal(conn, client, hardware, settings, **kwargs):
    """Polarizer calibration workflow.

    Reads --yaml, --output, --start, --end, --step, --exposure flags.
    Runs polarizer_calibration_workflow from acquisition.workflow.

    Response: SUCCESS:<report_path> or FAILED:<reason>
    """
    config_manager = kwargs.get("config_manager")
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested polarizer calibration", addr)

    try:
        message = read_message_string(conn)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read polarizer calibration message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    # Parse the message
    params = {}
    flags = [
        "--yaml",
        "--output",
        "--start",
        "--end",
        "--step",
        "--exposure",
    ]

    for i, flag in enumerate(flags):
        if flag in message:
            start_idx = message.index(flag) + len(flag)
            end_idx = len(message)
            for next_flag in flags[i + 1:]:
                if next_flag in message[start_idx:]:
                    next_pos = message.index(next_flag, start_idx)
                    if next_pos < end_idx:
                        end_idx = next_pos
                        break
            value = message[start_idx:end_idx].strip()
            if flag == "--yaml":
                params["yaml_file_path"] = value
            elif flag == "--output":
                params["output_folder_path"] = value
            elif flag == "--start":
                params["start_angle"] = float(value)
            elif flag == "--end":
                params["end_angle"] = float(value)
            elif flag == "--step":
                params["step_size"] = float(value)
            elif flag == "--exposure":
                params["exposure_ms"] = float(value)

    # Validate required parameters
    required = ["yaml_file_path", "output_folder_path"]
    missing = [key for key in required if key not in params]
    if missing:
        error_msg = f"Missing required parameters: {missing}"
        logger.error(error_msg)
        conn.sendall(f"FAILED:{error_msg}".encode())
        return

    # Set defaults for optional parameters
    params.setdefault("start_angle", 0.0)
    params.setdefault("end_angle", 360.0)
    params.setdefault("step_size", 5.0)
    params.setdefault("exposure_ms", 10.0)

    # Send immediate acknowledgment to prevent client timeout
    try:
        ack_response = f"STARTED:{params['output_folder_path']}".encode()
        conn.sendall(ack_response)
        logger.info("Sent STARTED acknowledgment for polarizer calibration")

        # Execute polarizer calibration workflow
        from microscope_command_server.acquisition.workflow import (
            polarizer_calibration_workflow,
        )

        report_path = polarizer_calibration_workflow(
            yaml_file_path=params["yaml_file_path"],
            output_folder_path=params["output_folder_path"],
            start_angle=params["start_angle"],
            end_angle=params["end_angle"],
            step_size=params["step_size"],
            exposure_ms=params["exposure_ms"],
            hardware=hardware,
            config_manager=config_manager,
            logger=logger,
        )

        # Send success response with report path
        response = f"SUCCESS:{report_path}".encode()
        conn.sendall(response)
        logger.info("Polarizer calibration completed. Report: %s", report_path)

    except Exception as e:
        logger.error("Polarizer calibration failed: %s", str(e), exc_info=True)
        response = f"FAILED:{str(e)}".encode()
        conn.sendall(response)


# ---------------------------------------------------------------------------
# PPMSENS - PPM rotation sensitivity test
# ---------------------------------------------------------------------------

def handle_ppmsens(conn, client, hardware, settings, **kwargs):
    """PPM rotation sensitivity test.

    Reads --yaml, --output, --test-type, --base-angle, --repeats flags.
    Delegates to microscope_command_server.modality.ppm.handle_sensitivity_test.

    Response: SUCCESS:<result_dir> or FAILED:<reason>
    """
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested PPM rotation sensitivity test", addr)

    try:
        message = read_message_string(conn)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read PPMSENS message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    # Parse parameters
    params = {}
    flags = ["--yaml", "--output", "--test-type", "--base-angle", "--repeats"]

    for i, flag in enumerate(flags):
        if flag in message:
            start_idx = message.index(flag) + len(flag)
            end_idx = len(message)
            for next_flag in flags[i + 1:]:
                if next_flag in message[start_idx:]:
                    next_pos = message.index(next_flag, start_idx)
                    if next_pos < end_idx:
                        end_idx = next_pos
                        break
            value = message[start_idx:end_idx].strip()

            if flag == "--yaml":
                params["yaml_file_path"] = value
            elif flag == "--output":
                params["output_folder_path"] = value
            elif flag == "--test-type":
                params["test_type"] = value
            elif flag == "--base-angle":
                params["base_angle"] = float(value)
            elif flag == "--repeats":
                params["n_repeats"] = int(value)

    # Set defaults
    params.setdefault("test_type", "repeatability")
    params.setdefault("base_angle", 7.0)
    params.setdefault("n_repeats", 10)

    # Validate required parameters
    required = ["yaml_file_path", "output_folder_path"]
    missing = [key for key in required if key not in params]
    if missing:
        error_msg = f"Missing required parameters: {missing}"
        logger.error(error_msg)
        conn.sendall(f"FAILED:{error_msg}".encode())
        return

    try:
        ack_response = f"STARTED:{params['output_folder_path']}".encode()
        conn.sendall(ack_response)
        logger.info("Sent STARTED acknowledgment for PPM sensitivity test")

        # Delegate to PPM modality handler
        from microscope_command_server.modality.ppm import handle_sensitivity_test

        result_dir = handle_sensitivity_test(
            params=params,
            port=TCP_PORT,
            _logger=logger,
        )

        if result_dir:
            response = f"SUCCESS:{result_dir}".encode()
            conn.sendall(response)
            logger.info("PPM sensitivity test completed: %s", result_dir)
        else:
            response = b"FAILED:Test did not complete successfully"
            conn.sendall(response)
            logger.error("PPM sensitivity test failed")

    except ImportError as e:
        logger.error("PPM sensitivity test module not available: %s", e)
        response = f"FAILED:Module not available - {e}".encode()
        conn.sendall(response)
    except Exception as e:
        logger.error("PPM sensitivity test failed: %s", str(e), exc_info=True)
        response = f"FAILED:{str(e)}".encode()
        conn.sendall(response)


# ---------------------------------------------------------------------------
# PPMBIREF - PPM birefringence maximization test
# ---------------------------------------------------------------------------

def handle_ppmbiref(conn, client, hardware, settings, **kwargs):
    """PPM birefringence maximization test.

    Reads --yaml, --output, --mode, --min-angle, --max-angle,
    --step, --exposure, --target-intensity flags. Creates progress
    and stage-move callbacks that communicate through the socket.

    Response: SUCCESS:<result_dir> or FAILED:<reason>
    """
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested PPM birefringence maximization test", addr)

    try:
        message = read_message_string(conn)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read PPMBIREF message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    # Parse parameters
    params = {}
    flags = ["--yaml", "--output", "--mode", "--min-angle", "--max-angle",
             "--step", "--exposure", "--target-intensity"]

    for i, flag in enumerate(flags):
        if flag in message:
            start_idx = message.index(flag) + len(flag)
            end_idx = len(message)
            for next_flag in flags[i + 1:]:
                if next_flag in message[start_idx:]:
                    next_pos = message.index(next_flag, start_idx)
                    if next_pos < end_idx:
                        end_idx = next_pos
                        break
            value = message[start_idx:end_idx].strip()

            if flag == "--yaml":
                params["yaml_file_path"] = value
            elif flag == "--output":
                params["output_folder_path"] = value
            elif flag == "--mode":
                params["exposure_mode"] = value
            elif flag == "--min-angle":
                params["min_angle"] = float(value)
            elif flag == "--max-angle":
                params["max_angle"] = float(value)
            elif flag == "--step":
                params["angle_step"] = float(value)
            elif flag == "--exposure":
                params["fixed_exposure_ms"] = float(value)
            elif flag == "--target-intensity":
                params["target_intensity"] = int(value)

    # Set defaults
    params.setdefault("exposure_mode", "interpolate")
    params.setdefault("min_angle", -10.0)
    params.setdefault("max_angle", 10.0)
    params.setdefault("angle_step", 0.5)  # Coarser default for server
    params.setdefault("target_intensity", 128)

    # Validate required parameters
    required = ["yaml_file_path", "output_folder_path"]
    missing = [key for key in required if key not in params]
    if missing:
        error_msg = f"Missing required parameters: {missing}"
        logger.error(error_msg)
        conn.sendall(f"FAILED:{error_msg}".encode())
        return

    # Validate fixed mode requires exposure
    if params["exposure_mode"] == "fixed" and "fixed_exposure_ms" not in params:
        error_msg = "fixed_exposure_ms required when mode=fixed"
        logger.error(error_msg)
        conn.sendall(f"FAILED:{error_msg}".encode())
        return

    try:
        ack_response = f"STARTED:{params['output_folder_path']}".encode()
        conn.sendall(ack_response)
        logger.info("Sent STARTED acknowledgment for PPM birefringence test")

        # Create progress callback to send updates through socket
        def send_progress(current, total):
            """Send progress update through socket."""
            try:
                progress_msg = f"PROGRESS:{current}:{total}".encode()
                conn.sendall(progress_msg)
                logger.debug("Sent progress: %d/%d", current, total)
            except Exception as e:
                logger.warning("Failed to send progress: %s", e)

        # Create stage move callback for calibrate mode
        def stage_move_callback():
            """
            Send STAGEMOVE message and wait for CONTINUE/ABORT response.
            Returns True if user confirmed, False if aborted.
            """
            try:
                # Send stage move request
                conn.sendall(b"STAGEMOVE:Background calibration complete. Move stage to tissue.")
                logger.info("Sent STAGEMOVE request, waiting for user confirmation...")

                # Wait indefinitely for user response (no timeout)
                # User may need significant time to find tissue and position stage
                conn.settimeout(None)  # No timeout - wait indefinitely
                response = conn.recv(1024).decode().strip()
                conn.settimeout(30.0)  # Restore normal timeout

                if response == "CONTINUE":
                    logger.info("User confirmed stage move, continuing...")
                    return True
                else:
                    logger.info("User response: %s, aborting...", response)
                    return False
            except Exception as e:
                logger.error("Stage move callback failed: %s", e)
                return False

        # Delegate to PPM modality handler
        from microscope_command_server.modality.ppm import handle_birefringence_test

        result_dir = handle_birefringence_test(
            params=params,
            port=TCP_PORT,
            progress_callback=send_progress,
            stage_move_callback=stage_move_callback,
            _logger=logger,
        )

        if result_dir:
            response = f"SUCCESS:{result_dir}".encode()
            conn.sendall(response)
            logger.info("PPM birefringence test completed: %s", result_dir)
        else:
            response = b"FAILED:Test did not complete successfully"
            conn.sendall(response)
            logger.error("PPM birefringence test failed")

    except ImportError as e:
        logger.error("PPM birefringence test module not available: %s", e)
        response = f"FAILED:Module not available - {e}".encode()
        conn.sendall(response)
    except Exception as e:
        logger.error("PPM birefringence test failed: %s", str(e), exc_info=True)
        response = f"FAILED:{str(e)}".encode()
        conn.sendall(response)


# ---------------------------------------------------------------------------
# SBCALIB - Sunburst calibration
# ---------------------------------------------------------------------------

def handle_sbcalib(conn, client, hardware, settings, **kwargs):
    """Sunburst calibration for hue-to-angle mapping.

    Reads --yaml, --output, --modality, --spokes, --saturation,
    --value, --name, --radius_inner, --radius_outer, --image_path,
    --center_y, --center_x flags.

    Response: SUCCESS:<json_result> or FAILED:<reason>
    """
    config_manager = kwargs.get("config_manager")
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested sunburst calibration", addr)

    try:
        message = read_message_string(conn)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read SBCALIB message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    # Parse parameters
    params = {}
    flags = ["--yaml", "--output", "--modality", "--spokes",
             "--saturation", "--value", "--name",
             "--radius_inner", "--radius_outer",
             "--image_path", "--center_y", "--center_x"]

    for i, flag in enumerate(flags):
        if flag in message:
            start_idx = message.index(flag) + len(flag)
            end_idx = len(message)
            for next_flag in flags[i + 1:]:
                if next_flag in message[start_idx:]:
                    next_pos = message.index(next_flag, start_idx)
                    if next_pos < end_idx:
                        end_idx = next_pos
                        break
            value = message[start_idx:end_idx].strip()

            if flag == "--yaml":
                params["yaml_file_path"] = value
            elif flag == "--output":
                params["output_folder_path"] = value
            elif flag == "--modality":
                params["modality"] = value
            elif flag == "--spokes":
                params["expected_spokes"] = int(value)
            elif flag == "--saturation":
                params["saturation_threshold"] = float(value)
            elif flag == "--value":
                params["value_threshold"] = float(value)
            elif flag == "--name":
                params["calibration_name"] = value
            elif flag == "--radius_inner":
                params["radius_inner"] = int(value)
            elif flag == "--radius_outer":
                params["radius_outer"] = int(value)
            elif flag == "--image_path":
                params["image_path"] = value
            elif flag == "--center_y":
                params["center_y"] = int(value)
            elif flag == "--center_x":
                params["center_x"] = int(value)

    # Set defaults
    params.setdefault("modality", "ppm_20x")
    params.setdefault("expected_spokes", 16)
    params.setdefault("saturation_threshold", 0.1)
    params.setdefault("value_threshold", 0.1)
    params.setdefault("calibration_name", None)
    params.setdefault("radius_inner", 30)
    params.setdefault("radius_outer", 150)
    params.setdefault("image_path", None)
    params.setdefault("center_y", None)
    params.setdefault("center_x", None)

    # Validate required parameters
    required = ["yaml_file_path", "output_folder_path"]
    missing = [key for key in required if key not in params]
    if missing:
        error_msg = f"Missing required parameters: {missing}"
        logger.error(error_msg)
        conn.sendall(f"FAILED:{error_msg}".encode())
        return

    try:
        ack_response = f"STARTED:{params['output_folder_path']}".encode()
        conn.sendall(ack_response)
        logger.info("Sent STARTED acknowledgment for sunburst calibration")

        # Run sunburst calibration workflow
        from microscope_command_server.calibration.sunburst_workflow import (
            run_sunburst_calibration,
        )

        # Build center tuple if both coordinates provided
        center = None
        if params["center_y"] is not None and params["center_x"] is not None:
            center = (params["center_y"], params["center_x"])

        result = run_sunburst_calibration(
            hardware=hardware,
            config_manager=config_manager,
            output_folder=params["output_folder_path"],
            modality=params["modality"],
            expected_spokes=params["expected_spokes"],
            saturation_threshold=params["saturation_threshold"],
            value_threshold=params["value_threshold"],
            calibration_name=params["calibration_name"],
            radius_inner=params["radius_inner"],
            radius_outer=params["radius_outer"],
            logger=logger,
            existing_image_path=params["image_path"],
            center=center,
        )

        # Send result as JSON (always SUCCESS: prefix with
        # full JSON so client gets image_path even on failure)
        import json
        result_json = json.dumps(result)
        response = f"SUCCESS:{result_json}".encode()
        conn.sendall(response)
        if result.get("success"):
            logger.info("Sunburst calibration successful. R^2=%.4f", result.get('r_squared', 0))
        else:
            logger.error("Sunburst calibration failed: %s", result.get('error', 'Unknown'))

    except ImportError as e:
        logger.error("Module not available: %s", e)
        response = f"FAILED:Module not available - {e}".encode()
        conn.sendall(response)
    except Exception as e:
        logger.error("Sunburst calibration failed: %s", str(e), exc_info=True)
        response = f"FAILED:{str(e)}".encode()
        conn.sendall(response)


# ---------------------------------------------------------------------------
# GETNOISE - Per-channel noise measurement
# ---------------------------------------------------------------------------

def handle_getnoise(conn, client, hardware, settings, **kwargs):
    """Get per-channel noise statistics via multi-frame analysis.

    Reads 1 byte for num_frames (0 = default 10).
    Response: 9 big-endian floats (R_mean, G_mean, B_mean, R_std,
    G_std, B_std, R_snr, G_snr, B_snr) or 9 zeros on error.
    """
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested noise measurement", addr)
    try:
        # Read 1 byte for num_frames (0 = default 10)
        nf_byte = conn.recv(1)
        num_frames = nf_byte[0] if nf_byte and nf_byte[0] > 0 else 10

        # TODO: JAI-specific -- migrate to camera-agnostic interface
        from microscope_control.jai import JAINoiseMeasurement
        noise_meter = JAINoiseMeasurement(hardware)
        stats = noise_meter.measure_noise(
            num_frames=num_frames, settle_frames=2
        )

        # Pack 9 floats: means (R,G,B), stddevs (R,G,B), SNRs (R,G,B)
        response = struct.pack(
            "!fffffffff",
            float(stats.channel_means["red"]),
            float(stats.channel_means["green"]),
            float(stats.channel_means["blue"]),
            float(stats.channel_stddevs["red"]),
            float(stats.channel_stddevs["green"]),
            float(stats.channel_stddevs["blue"]),
            float(stats.channel_snr["red"]),
            float(stats.channel_snr["green"]),
            float(stats.channel_snr["blue"]),
        )
        conn.sendall(response)
        logger.info(
            "Noise stats sent: R_snr=%.1f, G_snr=%.1f, B_snr=%.1f",
            stats.channel_snr['red'],
            stats.channel_snr['green'],
            stats.channel_snr['blue'],
        )
    except ImportError as e:
        logger.error("Noise measurement module not available: %s", e)
        # Send 9 zeros on error
        conn.sendall(struct.pack("!fffffffff", *([0.0] * 9)))
    except Exception as e:
        logger.error("Noise measurement failed: %s", e, exc_info=True)
        conn.sendall(struct.pack("!fffffffff", *([0.0] * 9)))


# ---------------------------------------------------------------------------
# NOISCHAR - JAI noise characterization
# ---------------------------------------------------------------------------

def handle_noischar(conn, client, hardware, settings, **kwargs):
    """JAI noise characterization across gain/exposure grid.

    Systematic noise characterization testing multiple gain/exposure
    combinations to find optimal SNR. Sends PROGRESS updates during
    the long-running characterization.

    Post-characterization: resets camera to unified gain 1.0, analog
    gains 1.0, disables individual exposure mode.

    Response: SUCCESS:<path>|<count>|<plots>|<bestGain>,<bestExp>
              or FAILED:<reason>
    """
    addr = client if isinstance(client, tuple) else getattr(client, "addr", client)
    logger.info("Client %s requested JAI noise characterization", addr)

    # Read the message using chunked pattern (same as WBSIMPLE)
    message_parts = []
    total_bytes = 0
    start_time = time.time()

    conn.settimeout(5.0)

    try:
        while True:
            chunk = conn.recv(1024)
            if not chunk:
                logger.error(
                    "Connection closed while reading NOISCHAR message"
                )
                conn.sendall(b"FAILED:Connection closed")
                break

            message_parts.append(chunk.decode("utf-8"))
            total_bytes += len(chunk)
            logger.debug("NOISCHAR: received %d bytes so far", total_bytes)

            full_message = "".join(message_parts)

            if END_MARKER in full_message:
                message = full_message.replace(END_MARKER, "").strip()
                logger.info("NOISCHAR message: %s", message)

                # Parse flags
                params = {}
                flags = [
                    "--output",
                    "--preset",
                    "--frames",
                    "--plots",
                    "--gains",
                    "--exposures",
                ]

                def find_flag_position(msg, flag):
                    """Find flag position ensuring it's followed by a space."""
                    search_pattern = flag + " "
                    if search_pattern in msg:
                        return msg.index(search_pattern)
                    return -1

                for i, flag in enumerate(flags):
                    flag_pos = find_flag_position(message, flag)
                    if flag_pos >= 0:
                        start_idx = flag_pos + len(flag)
                        end_idx = len(message)
                        for next_flag in flags:
                            if next_flag != flag:
                                next_pos = find_flag_position(
                                    message[start_idx:], next_flag
                                )
                                if next_pos >= 0:
                                    actual_pos = start_idx + next_pos
                                    if actual_pos < end_idx:
                                        end_idx = actual_pos

                        value = message[start_idx:end_idx].strip()

                        if flag == "--output":
                            params["output_path"] = value
                        elif flag == "--preset":
                            params["preset"] = value
                        elif flag == "--frames":
                            params["num_frames"] = int(value)
                        elif flag == "--plots":
                            params["generate_plots"] = (
                                value.lower() == "true"
                            )
                        elif flag == "--gains":
                            params["gains"] = [
                                float(v.strip())
                                for v in value.split(",")
                            ]
                        elif flag == "--exposures":
                            params["exposures"] = [
                                float(v.strip())
                                for v in value.split(",")
                            ]

                # Validate required parameters
                required = ["output_path"]
                missing = [
                    key for key in required if key not in params
                ]
                if missing:
                    error_msg = (
                        f"Missing required parameters: {missing}"
                    )
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                    break

                # Send immediate acknowledgment
                try:
                    from pathlib import Path

                    output_path = Path(params["output_path"])
                    output_path.mkdir(parents=True, exist_ok=True)

                    ack_response = (
                        f"STARTED:{params['output_path']}".encode()
                    )
                    conn.sendall(ack_response)
                    logger.info(
                        "Sent STARTED acknowledgment for NOISCHAR"
                    )

                    # Increase socket timeout for long-running
                    # characterization (up to 20 minutes)
                    conn.settimeout(1200.0)

                    # Import the characterization module
                    # TODO: JAI-specific -- migrate to camera-agnostic interface
                    from microscope_control.jai import (
                        JAINoiseCharacterization,
                    )

                    # Create characterization tool
                    jai_props = hardware.camera.properties
                    tool = JAINoiseCharacterization(
                        hardware,
                        jai_props,
                        num_frames=params.get("num_frames", 10),
                    )

                    # Build progress callback that sends PROGRESS
                    # messages back to the Java client
                    def progress_callback(current, total, msg=""):
                        try:
                            progress_msg = (
                                f"PROGRESS:{current}:{total}"
                            )
                            conn.sendall(progress_msg.encode())
                            logger.debug(
                                "NOISCHAR progress: %d/%d",
                                current, total,
                            )
                        except Exception as pe:
                            logger.warning(
                                "Failed to send progress: %s", pe
                            )

                    # Determine preset / custom gains+exposures
                    preset = params.get("preset", "full")
                    custom_gains = params.get("gains")
                    custom_exposures = params.get("exposures")

                    # Run characterization
                    is_quick = preset == "quick"
                    results = tool.run_characterization(
                        gains=custom_gains,
                        exposures=custom_exposures,
                        quick=is_quick,
                        progress_callback=progress_callback,
                    )

                    # Generate report/plots or just CSV
                    generate_plots = params.get(
                        "generate_plots", False
                    )
                    if generate_plots:
                        tool.generate_report(
                            results, output_path
                        )
                        logger.info(
                            "NOISCHAR: generated report with plots"
                        )
                    else:
                        # Just save CSV
                        results.to_csv(
                            output_path
                            / "noise_characterization.csv"
                        )
                        logger.info(
                            "NOISCHAR: saved CSV results only"
                        )

                    # Find best SNR from unsaturated results
                    best_gain = 0.0
                    best_exp = 0.0
                    best_snr = 0.0
                    total_configs = len(results.results)
                    for r in results.results:
                        if r.saturation_pct > 1.0:
                            continue
                        # Average SNR across channels
                        avg_snr = (
                            r.red_snr + r.green_snr + r.blue_snr
                        ) / 3.0
                        if avg_snr > best_snr:
                            best_snr = avg_snr
                            best_gain = r.unified_gain
                            best_exp = r.exposure_ms

                    # Format: SUCCESS:{path}|{count}|{plots}|
                    #         {bestGain},{bestExp}
                    plots_str = (
                        "true" if generate_plots else "false"
                    )
                    response = (
                        f"SUCCESS:{output_path}|"
                        f"{total_configs}|"
                        f"{plots_str}|"
                        f"{best_gain},{best_exp}"
                    )
                    conn.sendall(response.encode())
                    logger.info(
                        "NOISCHAR completed: %d configs, best SNR at "
                        "gain=%s, exp=%sms",
                        total_configs, best_gain, best_exp,
                    )

                except ImportError as e:
                    error_msg = (
                        f"JAI noise characterization module "
                        f"not available: {e}"
                    )
                    logger.error(error_msg)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                except Exception as e:
                    error_msg = f"NOISCHAR failed: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    conn.sendall(f"FAILED:{error_msg}".encode())
                break

            if total_bytes > 100000:
                logger.error(
                    "NOISCHAR message exceeds maximum size"
                )
                conn.sendall(b"FAILED:Message too large")
                break

            if time.time() - start_time > 10:
                logger.error("Timeout reading NOISCHAR message")
                conn.sendall(
                    b"FAILED:Timeout waiting for complete message"
                )
                break

    except socket.timeout:
        logger.error(
            "Timeout reading NOISCHAR message from %s", addr
        )
        conn.sendall(b"FAILED:Timeout reading message")
    except Exception as e:
        logger.error(
            "Error in NOISCHAR: %s", str(e), exc_info=True
        )
        conn.sendall(f"FAILED:{str(e)}".encode())
    finally:
        conn.settimeout(None)  # Reset to blocking mode
        # Reset camera to clean state after characterization
        try:
            cam = hardware.camera
            cam.set_rb_analog_gains(analog_red=1.0, analog_blue=1.0)
            cam.set_unified_gain(1.0)
            cam.disable_individual_exposure()
            logger.debug("Reset camera state after NOISCHAR")
        except Exception:
            pass
