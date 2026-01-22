"""Acquisition workflow and microscope-side operations for the command server.

This module contains the acquisition logic and helpers that interact with the
microscope hardware, separated from the socket server/transport logic.
"""

from __future__ import annotations

import time
from typing import Callable, List, Tuple, Optional, Dict, Any
from pathlib import Path
import shutil
import logging
import yaml

import numpy as np

from microscope_control.hardware import Position
from microscope_control.hardware.pycromanager import PycromanagerHardware
from microscope_control.autofocus.core import AutofocusUtils
from microscope_command_server.acquisition.tiles import TileConfigUtils
from ppm_library.imaging.writer import TifWriterUtils
from ppm_library.imaging.background import BackgroundCorrectionUtils
import shlex
import skimage.filters

logger = logging.getLogger(__name__)


def load_jai_calibration_from_imageprocessing(
    config_path: Path,
    per_angle: bool = False,
    modality: str = "ppm",
    objective: str = None,
    detector: str = None,
    logger=None,
) -> Optional[Dict[str, Any]]:
    """
    Load JAI white balance calibration from imageprocessing YAML.

    The calibration data is stored in the imaging_profiles section:
    imaging_profiles.{modality}.{objective}.{detector}.exposures_ms.{angle}.{r,g,b}

    Args:
        config_path: Path to the main config file (config_PPM.yml)
                    - imageprocessing file is derived from this
        per_angle: If True, load PPM per-angle calibration with R,G,B values
                  If False, load simple calibration (single exposure)
        modality: Modality name (e.g., "ppm", "brightfield")
        objective: Objective ID (e.g., "LOCI_OBJECTIVE_OLYMPUS_20X_POL_001")
        detector: Detector ID (e.g., "LOCI_DETECTOR_JAI_001")
        logger: Optional logger instance

    Returns:
        Dictionary with calibration data or None if not found.
        For PPM mode: {'angles': {'positive': {'exposures_ms': {'r': x, 'g': y, 'b': z}}, ...}}
    """
    config_path = Path(config_path)

    # Derive imageprocessing file path
    config_name = config_path.stem  # e.g., "config_PPM"
    if config_name.startswith("config_"):
        microscope_name = config_name[7:]  # e.g., "PPM"
        imageprocessing_name = f"imageprocessing_{microscope_name}.yml"
    else:
        imageprocessing_name = f"imageprocessing_{config_name}.yml"

    imageprocessing_path = config_path.parent / imageprocessing_name

    if not imageprocessing_path.exists():
        if logger:
            logger.info(f"No imageprocessing config found at {imageprocessing_path}")
        return None

    if not objective or not detector:
        if logger:
            logger.warning("Objective or detector not specified for calibration lookup")
        return None

    try:
        with open(imageprocessing_path, "r") as f:
            ip_data = yaml.safe_load(f) or {}

        # Navigate to imaging_profiles.{modality}.{objective}.{detector}
        imaging_profiles = ip_data.get("imaging_profiles", {})
        modality_profiles = imaging_profiles.get(modality, {})
        objective_profiles = modality_profiles.get(objective, {})
        detector_profile = objective_profiles.get(detector, {})

        if not detector_profile:
            if logger:
                logger.info(f"No profile found for {modality}/{objective}/{detector}")
            return None

        exposures_ms = detector_profile.get("exposures_ms", {})
        gains = detector_profile.get("gains", {})

        if not exposures_ms:
            if logger:
                logger.info(f"No exposures_ms found in profile for {modality}/{objective}/{detector}")
            return None

        if per_angle:
            # Build per-angle calibration structure from exposures_ms
            # Expected format in YAML:
            #   exposures_ms:
            #     positive: {all: 800, r: 750, g: 800, b: 850}
            #     negative: {all: 800, r: 750, g: 800, b: 850}
            #     ...
            angles_data = {}
            for angle_name, exp_data in exposures_ms.items():
                if isinstance(exp_data, dict) and 'r' in exp_data and 'g' in exp_data and 'b' in exp_data:
                    angles_data[angle_name] = {
                        'exposures_ms': {
                            'r': exp_data.get('r', 50.0),
                            'g': exp_data.get('g', 50.0),
                            'b': exp_data.get('b', 50.0),
                        }
                    }
                    # Add gains if available
                    if gains and angle_name in gains:
                        angles_data[angle_name]['gains'] = gains[angle_name]

            if angles_data:
                if logger:
                    logger.info(f"Loaded JAI PPM calibration for angles: {list(angles_data.keys())}")
                return {'angles': angles_data}
            else:
                if logger:
                    logger.info("No per-channel (r,g,b) exposure data found in exposures_ms")
                return None
        else:
            # Simple mode - return first available exposure settings
            if logger:
                logger.info(f"Loaded JAI simple calibration from {modality}/{objective}/{detector}")
            return {'exposures_ms': exposures_ms, 'gains': gains}

    except Exception as e:
        if logger:
            logger.warning(f"Failed to load JAI calibration from {imageprocessing_path}: {e}")
        return None


def apply_jai_calibration_for_angle(
    hardware: "PycromanagerHardware",
    jai_calibration: Dict[str, Any],
    angle: float,
    per_angle: bool = False,
    logger=None,
) -> bool:
    """
    Apply JAI white balance calibration settings before image capture.

    This enables individual exposure mode and sets per-channel exposures
    based on the calibration data.

    Args:
        hardware: PycromanagerHardware instance
        jai_calibration: Calibration data from load_jai_calibration_from_imageprocessing()
        angle: Current rotation angle (used for PPM mode to select angle-specific settings)
        per_angle: If True, use angle-specific settings from jai_ppm
                  If False, use single settings from jai_simple
        logger: Optional logger instance

    Returns:
        True if settings were applied, False otherwise
    """
    # Only applies to JAI camera
    try:
        camera_name = hardware.core.get_property("Core", "Camera")
        if "JAI" not in camera_name.upper():
            if logger:
                logger.debug(f"JAI calibration skipped - camera is {camera_name}")
            return False
    except Exception:
        return False

    try:
        from microscope_control.jai import JAICameraProperties

        # Get calibration settings for this angle
        if per_angle:
            # Map numeric angle to angle name
            angle_mapping = {90.0: "uncrossed", 0.0: "crossed", 7.0: "positive", -7.0: "negative"}
            angle_name = angle_mapping.get(angle)
            if not angle_name:
                # Try to find closest match
                for a, name in angle_mapping.items():
                    if abs(a - angle) < 1.0:
                        angle_name = name
                        break

            if not angle_name or "angles" not in jai_calibration:
                if logger:
                    logger.warning(f"No PPM calibration found for angle {angle}")
                return False

            angle_cal = jai_calibration["angles"].get(angle_name)
            if not angle_cal:
                if logger:
                    logger.warning(f"No PPM calibration found for angle {angle_name}")
                return False

            exposures = angle_cal.get("exposures_ms", {})
            gains = angle_cal.get("gains", {})
        else:
            # Simple mode - use same settings for all angles
            exposures = jai_calibration.get("exposures_ms", {})
            gains = jai_calibration.get("gains", {})

        if not exposures:
            if logger:
                logger.warning("No exposure data in JAI calibration")
            return False

        # Apply per-channel exposures
        jai_props = JAICameraProperties(hardware.core)
        jai_props.set_channel_exposures(
            red=exposures.get("r", 50.0),
            green=exposures.get("g", 50.0),
            blue=exposures.get("b", 50.0),
            auto_enable=True,  # Automatically enable individual exposure mode
        )

        exp_msg = (
            f"Applied JAI calibration for angle {angle}: "
            f"R={exposures.get('r'):.1f}ms, G={exposures.get('g'):.1f}ms, B={exposures.get('b'):.1f}ms"
        )

        # Apply per-channel gains if calibration required gain compensation
        gain_r = gains.get("r", 1.0)
        gain_g = gains.get("g", 1.0)
        gain_b = gains.get("b", 1.0)

        if gain_r != 1.0 or gain_g != 1.0 or gain_b != 1.0:
            jai_props.set_analog_gains(
                red=gain_r,
                green=gain_g,
                blue=gain_b,
                auto_enable=True,  # Automatically enable individual gain mode
            )
            exp_msg += f" | Gains: R={gain_r:.3f}, G={gain_g:.3f}, B={gain_b:.3f}"

        if logger:
            logger.info(exp_msg)

        return True

    except ImportError:
        if logger:
            logger.debug("JAI camera module not available - skipping calibration")
        return False
    except Exception as e:
        if logger:
            logger.warning(f"Failed to apply JAI calibration: {e}")
        return False


def load_and_apply_white_balance_settings(
    hardware: PycromanagerHardware,
    calibration_folder: str,
    detector: str,
    modality: str,
    objective: str,
    logger=None,
) -> bool:
    """
    Load and apply white balance calibration settings for JAI camera.

    Looks for white_balance_settings.yml in the calibration folder structure:
    {calibration_folder}/{detector}/{modality}/{objective}/white_balance_settings.yml

    Args:
        hardware: PycromanagerHardware instance
        calibration_folder: Base path for calibration data
        detector: Detector ID (e.g., "JAI")
        modality: Modality name (e.g., "ppm", "brightfield")
        objective: Objective ID (e.g., "20x")
        logger: Optional logger instance

    Returns:
        True if settings were loaded and applied, False otherwise
    """
    # Only applies to JAI camera
    camera_name = hardware.core.get_property("Core", "Camera")
    if camera_name != "JAICamera":
        if logger:
            logger.debug(f"White balance loading skipped - camera is {camera_name}, not JAI")
        return False

    try:
        from microscope_control.jai import JAICameraProperties

        # Build path to settings file
        wb_settings_path = (
            Path(calibration_folder)
            / detector
            / modality
            / objective
            / "white_balance_settings.yml"
        )

        if not wb_settings_path.exists():
            if logger:
                logger.info(f"No white balance settings found at {wb_settings_path}")
            return False

        # Load and apply settings
        jai_props = JAICameraProperties(hardware.core)
        success = jai_props.apply_white_balance_settings(str(wb_settings_path))

        if success and logger:
            logger.info(f"Applied white balance settings from {wb_settings_path}")

        return success

    except ImportError:
        if logger:
            logger.warning("JAI calibration module not available - skipping white balance loading")
        return False
    except Exception as e:
        if logger:
            logger.warning(f"Failed to load white balance settings: {e}")
        return False


def log_timing(logger, operation_name, start_time):
    """Log elapsed time for an operation in milliseconds.

    Args:
        logger: Logger instance
        operation_name: Description of the operation
        start_time: Start time from time.perf_counter()

    Returns:
        Current time for use as next start_time
    """
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f"  [TIMING] {operation_name}: {elapsed_ms:.1f}ms")
    return time.perf_counter()


def autofocus_with_manual_fallback(
    hardware: PycromanagerHardware,
    logger,
    request_manual_focus: Optional[Callable[[int], str]] = None,
    max_retries: int = 3,
    **autofocus_kwargs
):
    """
    Perform autofocus with manual focus fallback on failure.

    If autofocus fails (returns failure dict), prompts user for manual focus
    and retries. Shows dialog even on last attempt with retry button disabled.

    After manual focus completes, the stage XY position is restored to the
    original tile position (user may have moved XY to find tissue for focusing).

    Args:
        hardware: PycromanagerHardware instance
        logger: Logger instance
        request_manual_focus: Optional callback to request manual focus from user.
                             Callback receives retries_remaining (int) and returns
                             user choice: "retry", "skip", or "cancel".
                             If None, will raise exception on autofocus failure.
        max_retries: Maximum number of retry attempts after manual focus
        **autofocus_kwargs: Arguments to pass to hardware.autofocus()

    Returns:
        float: Best focus Z position on success

    Raises:
        RuntimeError: If user cancels acquisition or no callback provided
    """
    # Capture original XY position before any autofocus attempts
    # User may move XY during manual focus dialog to find tissue
    original_pos = hardware.get_current_position()
    original_x, original_y = original_pos.x, original_pos.y

    for attempt in range(max_retries):
        result = hardware.autofocus(**autofocus_kwargs)

        # Check if autofocus succeeded (returns float) or failed (returns dict)
        if isinstance(result, float):
            # Success!
            return result
        elif isinstance(result, dict) and result.get('success') == False:
            # Autofocus failed
            logger.warning(f"Autofocus failed (attempt {attempt + 1}/{max_retries}): {result['message']}")
            logger.warning(f"  Quality score: {result['quality_score']:.2f}, "
                          f"Prominence: {result['peak_prominence']:.2f}")

            if request_manual_focus is not None:
                # Always show dialog, even on last attempt
                retries_remaining = max_retries - attempt - 1
                logger.info(f"Requesting manual focus from user (retries remaining: {retries_remaining})...")
                user_choice = request_manual_focus(retries_remaining)  # Pass retries info

                if user_choice == "skip":
                    # User chose to use current focus - restore XY and return
                    current_pos = hardware.get_current_position()
                    if abs(current_pos.x - original_x) > 1.0 or abs(current_pos.y - original_y) > 1.0:
                        logger.info(f"Restoring XY position after manual focus: "
                                   f"({current_pos.x:.1f}, {current_pos.y:.1f}) -> ({original_x:.1f}, {original_y:.1f})")
                        # Position already imported at top of file (line 18)
                        restore_pos = Position(original_x, original_y, current_pos.z)
                        hardware.move_to_position(restore_pos)
                    logger.info("User chose to use current focus position")
                    return result['attempted_z']
                elif user_choice == "cancel":
                    # User chose to cancel acquisition
                    logger.warning("User cancelled acquisition during manual focus")
                    raise RuntimeError("Acquisition cancelled by user during manual focus")
                elif user_choice == "retry":
                    if retries_remaining > 0:
                        # IMPORTANT: Run autofocus at CURRENT position (where user found tissue)
                        # BEFORE restoring XY. This ensures autofocus runs where there's tissue.
                        logger.info(f"Running autofocus at current position (where user found tissue)...")
                        retry_result = hardware.autofocus(**autofocus_kwargs)

                        if isinstance(retry_result, float):
                            # Autofocus succeeded at current position - restore XY with new Z
                            logger.info(f"Autofocus succeeded at current position: Z={retry_result:.2f} um")
                            current_pos = hardware.get_current_position()
                            if abs(current_pos.x - original_x) > 1.0 or abs(current_pos.y - original_y) > 1.0:
                                logger.info(f"Restoring XY position: "
                                           f"({current_pos.x:.1f}, {current_pos.y:.1f}) -> ({original_x:.1f}, {original_y:.1f})")
                                # Position already imported at top of file (line 18)
                                restore_pos = Position(original_x, original_y, retry_result)
                                hardware.move_to_position(restore_pos)
                            return retry_result
                        else:
                            # Autofocus failed again - continue to next attempt
                            logger.warning(f"Autofocus retry failed: {retry_result.get('message', 'unknown error')}")
                            continue
                    else:
                        # No retries left - shouldn't happen since button should be disabled
                        logger.warning("User chose retry but no retries remaining - using current focus")
                        return result['attempted_z']
                else:
                    # Unknown choice - default to skip
                    logger.warning(f"Unknown user choice '{user_choice}' - using current focus")
                    return result['attempted_z']
            else:
                # No callback provided, raise exception
                raise RuntimeError(
                    f"Autofocus failed: {result['message']}. "
                    f"Quality score: {result['quality_score']:.2f}, "
                    f"Prominence: {result['peak_prominence']:.2f}"
                )
        else:
            # Unexpected return type
            raise RuntimeError(f"Unexpected autofocus return value: {result}")

    # Should never reach here
    raise RuntimeError("Autofocus retry loop exited unexpectedly")


def calculate_luminance_gain(r, g, b):
    """Calculate luminance-based gain from RGB values."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def parse_angles_exposures(angles_str, exposures_str=None) -> Tuple[List[float], List[int]]:
    """Parse angle and exposure strings from various formats."""
    angles: List[float] = []
    exposures: List[int] = []

    # Parse angles
    if isinstance(angles_str, list):
        angles = angles_str
    elif isinstance(angles_str, str):
        angles_str = angles_str.strip("()")
        if "," in angles_str:
            angles = [float(x.strip()) for x in angles_str.split(",")]
        elif angles_str:
            angles = [float(x) for x in angles_str.split()]

    # Parse exposures if provided
    if exposures_str:
        if isinstance(exposures_str, list):
            exposures = exposures_str
        elif isinstance(exposures_str, str):
            exposures_str = exposures_str.strip("()")
            if "," in exposures_str:
                exposures = [float(x.strip()) for x in exposures_str.split(",")]
            elif exposures_str:
                exposures = [float(x) for x in exposures_str.split()]

    # Default exposures if not provided
    if not exposures and angles:
        for angle in angles:
            if angle == 90.0:
                exposures.append(10.0)
            elif angle == 0.0:
                exposures.append(800.0)
            else:
                exposures.append(500.0)

    return angles, exposures


def parse_acquisition_message(message: str) -> dict:
    """Parse acquisition message in flag-based format."""
    # Remove END_MARKER if present
    message = message.replace(" END_MARKER", "").replace("END_MARKER", "").strip()

    # Parse flag-based format
    if "--" in message:
        # Parse flag-based format
        params = {}

        # Split by spaces but preserve quoted strings
        try:
            # For Windows compatibility, temporarily replace backslashes
            temp_message = message.replace("\\", "|||BACKSLASH|||")
            parts = shlex.split(temp_message)
            # Restore backslashes
            parts = [part.replace("|||BACKSLASH|||", "\\") for part in parts]
        except Exception:
            # Fallback to simple split if shlex fails
            parts = message.split()

        i = 0
        while i < len(parts):
            if parts[i] == "--yaml" and i + 1 < len(parts):
                params["yaml_file_path"] = parts[i + 1]
                i += 2
            elif parts[i] == "--projects" and i + 1 < len(parts):
                params["projects_folder_path"] = parts[i + 1]
                i += 2
            elif parts[i] == "--sample" and i + 1 < len(parts):
                params["sample_label"] = parts[i + 1]
                i += 2
            elif parts[i] == "--scan-type" and i + 1 < len(parts):
                params["scan_type"] = parts[i + 1]
                i += 2
            elif parts[i] == "--region" and i + 1 < len(parts):
                params["region_name"] = parts[i + 1]
                i += 2
            elif parts[i] == "--angles" and i + 1 < len(parts):
                params["angles_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--exposures" and i + 1 < len(parts):
                params["exposures_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--bg-correction" and i + 1 < len(parts):
                params["background_correction_enabled"] = parts[i + 1].lower() == "true"
                i += 2
            elif parts[i] == "--bg-method" and i + 1 < len(parts):
                params["background_correction_method"] = parts[i + 1]
                i += 2
            elif parts[i] == "--bg-folder" and i + 1 < len(parts):
                params["background_folder"] = parts[i + 1]
                i += 2
            elif parts[i] == "--bg-disabled-angles" and i + 1 < len(parts):
                params["background_disabled_angles_str"] = parts[i + 1]
                i += 2
            elif parts[i] == "--white-balance" and i + 1 < len(parts):
                params["white_balance_enabled"] = parts[i + 1].lower() == "true"
                i += 2
            elif parts[i] == "--wb-per-angle" and i + 1 < len(parts):
                params["white_balance_per_angle"] = parts[i + 1].lower() == "true"
                i += 2
            elif parts[i] == "--objective" and i + 1 < len(parts):
                params["objective"] = parts[i + 1]
                i += 2
            elif parts[i] == "--detector" and i + 1 < len(parts):
                params["detector"] = parts[i + 1]
                i += 2
            elif parts[i] == "--pixel-size" and i + 1 < len(parts):
                params["pixel_size"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--af-tiles" and i + 1 < len(parts):
                params["autofocus_tiles"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--af-steps" and i + 1 < len(parts):
                params["autofocus_steps"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "--af-range" and i + 1 < len(parts):
                params["autofocus_range"] = float(parts[i + 1])
                i += 2
            elif parts[i] == "--processing" and i + 1 < len(parts):
                params["processing_pipeline"] = parts[i + 1]
                i += 2
            elif parts[i] == "--hint-z" and i + 1 < len(parts):
                params["hint_z"] = float(parts[i + 1])
                i += 2
            else:
                i += 1

        # Parse angles and exposures if present
        angles, exposures = parse_angles_exposures(
            params.get("angles_str", "()"), params.get("exposures_str", None)
        )
        params["angles"] = angles
        params["exposures"] = exposures

        # Parse disabled angles for background correction
        disabled_angles = []
        disabled_angles_str = params.get("background_disabled_angles_str", "()")
        if disabled_angles_str and disabled_angles_str != "()":
            disabled_angles_str = disabled_angles_str.strip("()")
            if "," in disabled_angles_str:
                disabled_angles = [float(x.strip()) for x in disabled_angles_str.split(",")]
            elif disabled_angles_str:
                disabled_angles = [float(x) for x in disabled_angles_str.split()]
        params["background_disabled_angles"] = disabled_angles

        # Validate required parameters
        required = [
            "yaml_file_path",
            "projects_folder_path",
            "sample_label",
            "scan_type",
            "region_name",
        ]
        missing = [key for key in required if key not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        return params

    raise ValueError("Invalid acquisition message format - must use flag-based format with '--' parameters")


def get_angles_wb_from_settings(settings: Dict[str, Any]) -> Dict[float, List[float]]:
    """Extract white balance values for different angles from settings."""
    angles_wb = {}

    # Try to find white balance settings
    wb_settings = settings.get("white_balance", {})
    ppm_wb = wb_settings.get("ppm", {})

    # Map standard angle names to numeric values
    angle_mapping = {"crossed": 0.0, "uncrossed": 90.0, "positive": 7.0, "negative": -7.0}

    for angle_name, angle_value in angle_mapping.items():
        if angle_name in ppm_wb:
            wb_values = ppm_wb[angle_name]
            # Handle different formats
            if isinstance(wb_values, list):
                if len(wb_values) > 0 and isinstance(wb_values[0], str):
                    # Format: ["1.0 1.0 1.0"]
                    angles_wb[angle_value] = [float(x) for x in wb_values[0].split()]
                else:
                    # Format: [1.0, 1.0, 1.0]
                    angles_wb[angle_value] = wb_values
            elif isinstance(wb_values, str):
                # Format: "1.0 1.0 1.0"
                angles_wb[angle_value] = [float(x) for x in wb_values.split()]

    # Default fallback values if not found
    if not angles_wb:
        logger.warning("No white balance settings found, using defaults")
        angles_wb = {
            0.0: [1.0, 1.0, 1.0],  # crossed
            90.0: [1.2, 1.0, 1.1],  # uncrossed
            7.0: [1.0, 1.0, 1.0],  # positive
            -7.0: [1.0, 1.0, 1.0],  # negative
        }

    return angles_wb


def _acquisition_workflow(
    message: str,
    client_addr,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
    update_progress: Callable[[int, int], None],
    set_state: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    request_manual_focus: Optional[Callable[[], None]] = None,
    connection_config_path: Optional[str] = None,
):
    """Execute the main image acquisition workflow with progress and cancellation.

    Args:
        message: Acquisition command message
        client_addr: Client address for logging
        hardware: Hardware interface
        config_manager: Configuration manager
        logger: Logger instance
        update_progress: Callback to update progress (current, total)
        set_state: Callback to set acquisition state
        is_cancelled: Callback to check if cancelled
        request_manual_focus: Optional callback to request manual focus from user
                             when autofocus fails. If None, autofocus failures will
                             raise exceptions as before.
        connection_config_path: Optional path to config from initial CONFIG command,
                               used to warn if ACQUIRE uses different config.
    """

    logger.info(f"=== ACQUISITION WORKFLOW STARTED for client {client_addr} ===")

    try:
        # Parse the acquisition parameters
        params = parse_acquisition_message(message)

        modality = "_".join(params["scan_type"].split("_")[:2])

        logger.info("Acquisition parameters:")
        logger.info(f"  Client: {client_addr}")
        logger.info(f"  Modality: {modality}")
        logger.info(f"  Sample label: {params['sample_label']}")
        logger.info(f"  Scan type: {params['scan_type']}")
        logger.info(f"  Region: {params['region_name']}")
        logger.info(f"  Angles: {params['angles']} degrees")
        logger.info(f"  Exposures: {params['exposures']} ms")

        # Load the yaml file
        if not params["yaml_file_path"]:
            raise ValueError("YAML file path is required")
        if not Path(params["yaml_file_path"]).exists():
            raise FileNotFoundError(f"YAML file {params['yaml_file_path']} does not exist")

        # Load configuration using the config manager
        ppm_settings = config_manager.load_config_file(params["yaml_file_path"])
        loci_rsc_file = str(
            Path(params["yaml_file_path"]).parent / "resources" / "resources_LOCI.yml"
        )
        loci_resources = config_manager.load_config_file(loci_rsc_file)
        ppm_settings.update(loci_resources)
        hardware.settings = ppm_settings

        # SAFETY WARNING: Check if ACQUIRE yaml differs from CONFIG
        if connection_config_path:
            acquire_yaml = Path(params["yaml_file_path"]).resolve()
            connection_yaml = Path(connection_config_path).resolve()
            if acquire_yaml != connection_yaml:
                logger.warning("=" * 80)
                logger.warning("CONFIG MISMATCH WARNING")
                logger.warning(f"Connection CONFIG:  {connection_yaml}")
                logger.warning(f"ACQUIRE --yaml:     {acquire_yaml}")
                logger.warning("ACQUIRE yaml has overridden connection config for this acquisition")
                logger.warning("This may cause unexpected behavior or hardware misconfiguration!")
                logger.warning("=" * 80)

        # Re-initialize microscope-specific methods with updated settings
        # This is critical for PPM rotation to work correctly and to ensure
        # correct focus device configuration for autofocus
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Try to load and apply JAI white balance settings if available
        # Settings are stored in calibration folder after running WBCALIBRATE command
        wb_calibration_folder = params.get("white_balance_calibration_folder")
        if wb_calibration_folder:
            # Extract modality and objective from params for path construction
            wb_modality = BackgroundCorrectionUtils.get_modality_from_scan_type(
                params["scan_type"]
            )
            # Try to get objective from scan type (e.g., "ppm_20x" -> "20x")
            scan_parts = params["scan_type"].split("_")
            wb_objective = scan_parts[-1] if len(scan_parts) > 1 else "default"

            load_and_apply_white_balance_settings(
                hardware=hardware,
                calibration_folder=wb_calibration_folder,
                detector="JAI",
                modality=wb_modality,
                objective=wb_objective,
                logger=logger,
            )

        # Home rot-stage
        # hardware.home_psg()

        # Extract modality from scan type
        modality = BackgroundCorrectionUtils.get_modality_from_scan_type(params["scan_type"])
        logger.info(f"Using modality: {modality}")

        # Get processing settings from parameters
        background_correction_enabled = params.get("background_correction_enabled", False)
        background_correction_method = params.get("background_correction_method", "divide")
        background_disabled_angles = params.get("background_disabled_angles", [])
        white_balance_enabled = params.get("white_balance_enabled", True)

        # Log background correction configuration
        if background_correction_enabled:
            logger.info(
                f"Background correction enabled with method: {background_correction_method}"
            )
            if background_disabled_angles:
                logger.info(
                    f"Background correction will be disabled for angles: {background_disabled_angles}"
                )
        else:
            logger.info("Background correction disabled")

        # ======= WARNING FOR BOTH CORRECTIONS ENABLED =======
        if background_correction_enabled and white_balance_enabled:
            logger.warning("=" * 70)
            logger.warning("WARNING: Both background correction and white balance are enabled!")
            logger.warning("This may lead to over-correction of the images.")
            logger.warning("Consider using only one correction method.")
            logger.warning("=" * 70)

        # ======= BACKGROUND CORRECTION SETUP =======
        background_images = {}
        background_scaling_factors = {}

        if background_correction_enabled:
            background_dir = None

            # Priority 1: Message parameter
            if "background_folder" in params:
                background_dir = Path(params["background_folder"])
                logger.info(f"Using background folder from message: {background_dir}")
            else:
                # Priority 2: YAML configuration from imageprocessing config file
                # Try to load imageprocessing config (e.g., imageprocessing_PPM.yml)
                config_path = Path(params["yaml_file_path"])
                imageprocessing_path = config_path.parent / f"imageprocessing_{config_path.stem.replace('config_', '')}.yml"

                bc_settings = None
                if imageprocessing_path.exists():
                    try:
                        imageprocessing_config = config_manager.load_config_file(str(imageprocessing_path))
                        bc_config = imageprocessing_config.get("background_correction", {})
                        bc_settings = bc_config.get(modality, {})
                        logger.info(f"Loaded background correction settings from: {imageprocessing_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load imageprocessing config: {e}")
                else:
                    logger.warning(f"Imageprocessing config not found at: {imageprocessing_path}")

                if bc_settings and bc_settings.get("enabled") and bc_settings.get("base_folder"):
                    # For YAML config, construct path with modality subdirectory
                    background_dir = Path(bc_settings["base_folder"]) / modality
                    logger.info(f"Using background folder from YAML config: {background_dir}")

            # Load background images if directory is valid
            if background_dir and background_dir.exists():
                logger.info(f"Loading background images from: {background_dir}")
                background_images, background_scaling_factors, _ = (
                    BackgroundCorrectionUtils.load_background_images(
                        background_dir, params["angles"], logger
                    )
                )

                if background_images:
                    logger.info(f"Loaded {len(background_images)} background images")
                else:
                    logger.warning("No background images found - disabling background correction")
                    background_correction_enabled = False
            else:
                logger.warning(f"Background directory not found: {background_dir}")
                logger.warning("Disabling background correction")
                background_correction_enabled = False

        # ======= WHITE BALANCE SETUP =======
        angles_wb = {}
        white_balance_per_angle = params.get("white_balance_per_angle", False)

        # Load JAI hardware white balance calibration (per-channel exposures)
        # This is separate from software white balance (RGB multipliers applied post-capture)
        jai_calibration = None
        if white_balance_enabled:
            jai_calibration = load_jai_calibration_from_imageprocessing(
                config_path=Path(params["yaml_file_path"]),
                per_angle=white_balance_per_angle,
                logger=logger,
            )
            if jai_calibration:
                logger.info(
                    f"JAI hardware white balance enabled "
                    f"({'per-angle' if white_balance_per_angle else 'simple'} mode)"
                )
            else:
                logger.info("No JAI calibration found - using software white balance")

        if white_balance_enabled:
            # Load software white balance settings from configuration (RGB multipliers)
            angles_wb = get_angles_wb_from_settings(ppm_settings)

            if white_balance_per_angle:
                # Use per-angle white balance profiles (PPM mode)
                logger.info(f"Using per-angle white balance for {len(angles_wb)} angles")
            else:
                # Use single white balance profile (uncrossed/90deg) for all angles
                # This is the default for non-PPM or when per-angle is disabled
                uncrossed_profile = angles_wb.get(90.0, [1.0, 1.0, 1.0])
                logger.info(f"Using single white balance profile for all angles: {uncrossed_profile}")
                # Apply uncrossed profile to all angles that will be acquired
                for angle in params.get("angles", []):
                    angles_wb[angle] = uncrossed_profile

        # Set up output paths
        project_path = Path(params["projects_folder_path"]) / params["sample_label"]
        output_path = project_path / params["scan_type"] / params["region_name"]
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_path}")

        # Read tile positions
        tile_config_path = output_path / "TileConfiguration.txt"
        positions = TileConfigUtils.read_tile_config(tile_config_path, hardware.core)

        if not positions:
            logger.error(f"No positions found in {tile_config_path}")
            set_state("FAILED")
            return

        xy_positions = [(pos.x, pos.y) for pos, filename in positions]
        # except Exception as e:
        #   logger.warning("Falling back to older tileconfig reader: %s", e)
        #   xy_positions = TileConfigUtils.read_TileConfiguration_coordinates(tile_config_path)

        # Create angle subdirectories
        if params["angles"]:
            for angle in params["angles"]:
                angle_dir = output_path / str(angle)
                angle_dir.mkdir(exist_ok=True)
                shutil.copy2(tile_config_path, angle_dir / "TileConfiguration.txt")

        # Calculate total images and update progress
        total_images = (
            len(positions) * len(params["angles"]) if params["angles"] else len(positions)
        )

        # check if total image is 720_psg_degs (should be 360, MSN tested 720) x number_of_tiles < MM2 limit
        # for DDR25 limit is 536870.9 (thorlabs)degs or 268573 (ticks)
        # that is 372 tiles
        #total_rotation = 720 * len(positions)  #
        #if total_rotation > 2**18:  # 262144
        #    logger.error(
        #        f"Total rotation steps {total_rotation} exceed Micro-Manager limit of 536870. Acquisition aborted."
        #    )

        starting_position = hardware.get_current_position()

        update_progress(0, total_images)
        logger.info(
            f"Starting acquisition of {total_images} total images "
            f"({len(positions)} positions x {len(params['angles'])} angles)"
        )

        image_count = 0

        # Collect stage positions for TileConfiguration_Stage.txt
        # Format: list of (filename, x, y, z) tuples
        stage_positions_collected = []

        # Find autofocus positions
        fov = hardware.get_fov()

        # Load autofocus settings from separate autofocus_{microscope}.yml file
        af_n_tiles = 5  # default
        af_search_range = 50  # default
        af_n_steps = 11  # default
        af_interp_strength = 100  # default
        af_interp_kind = "quadratic"  # default
        af_score_metric_name = "laplacian_variance"  # default
        af_texture_threshold = 0.005  # default - tissue detection sensitivity
        af_tissue_area_threshold = 0.2  # default - minimum tissue coverage
        af_rgb_brightness_threshold = 225.0  # default - maximum RGB brightness for tissue (blank rejection)
        # Adaptive autofocus parameters
        af_adaptive_initial_step = 10.0  # default
        af_adaptive_min_step = 2.0  # default
        af_adaptive_max_steps = 25  # default
        af_adaptive_focus_threshold = 0.95  # default
        af_large_drift_threshold = 4.0  # default - um drift that triggers STANDARD autofocus fallback

        # Get objective from acquisition parameters (passed via command line)
        current_objective = params.get("objective", "")

        # Track whether autofocus settings were found for the objective
        af_settings_found = False

        try:
            # Derive autofocus config path from main config path
            # e.g., "config_PPM.yml" -> "autofocus_PPM.yml"
            config_path = Path(params["yaml_file_path"])
            config_name = config_path.stem  # "config_PPM"
            microscope_name = config_name.replace("config_", "")  # "PPM"
            autofocus_file = config_path.parent / f"autofocus_{microscope_name}.yml"

            if not autofocus_file.exists():
                error_msg = (
                    f"Autofocus configuration file not found: {autofocus_file}\n"
                    f"Cannot proceed with acquisition - autofocus settings are required for objective '{current_objective}'.\n"
                    f"Please create the autofocus configuration file with settings for your objectives."
                )
                logger.error(error_msg)
                set_state("FAILED", error_msg)
                return

            with open(autofocus_file, "r") as f:
                autofocus_config = yaml.safe_load(f)

            # Find settings for current objective
            af_settings_list = autofocus_config.get("autofocus_settings", [])
            for af_setting in af_settings_list:
                if af_setting.get("objective") == current_objective:
                    af_n_tiles = af_setting.get("n_tiles", af_n_tiles)
                    af_search_range = af_setting.get("search_range_um", af_search_range)
                    af_n_steps = af_setting.get("n_steps", af_n_steps)
                    af_interp_strength = af_setting.get("interp_strength", af_interp_strength)
                    af_interp_kind = af_setting.get("interp_kind", af_interp_kind)
                    af_score_metric_name = af_setting.get("score_metric", af_score_metric_name)
                    af_texture_threshold = af_setting.get("texture_threshold", af_texture_threshold)
                    af_tissue_area_threshold = af_setting.get("tissue_area_threshold", af_tissue_area_threshold)
                    af_rgb_brightness_threshold = af_setting.get("rgb_brightness_threshold", af_rgb_brightness_threshold)
                    af_adaptive_initial_step = af_setting.get("adaptive_initial_step_um", af_adaptive_initial_step)
                    af_adaptive_min_step = af_setting.get("adaptive_min_step_um", af_adaptive_min_step)
                    af_adaptive_max_steps = af_setting.get("adaptive_max_steps", af_adaptive_max_steps)
                    af_adaptive_focus_threshold = af_setting.get("adaptive_focus_threshold", af_adaptive_focus_threshold)
                    af_large_drift_threshold = af_setting.get("large_drift_threshold_um", af_large_drift_threshold)
                    logger.info(
                        f"Loaded autofocus settings for {current_objective}: "
                        f"n_steps={af_n_steps}, search_range={af_search_range}um, n_tiles={af_n_tiles}, "
                        f"interp_strength={af_interp_strength}, interp_kind={af_interp_kind}, "
                        f"score_metric={af_score_metric_name}, "
                        f"texture_threshold={af_texture_threshold}, tissue_area_threshold={af_tissue_area_threshold}, "
                        f"rgb_brightness_threshold={af_rgb_brightness_threshold}, "
                        f"adaptive: initial_step={af_adaptive_initial_step}um, min_step={af_adaptive_min_step}um, "
                        f"max_steps={af_adaptive_max_steps}, focus_threshold={af_adaptive_focus_threshold}"
                    )
                    af_settings_found = True
                    break

            # Validate that settings were found for the objective
            if not af_settings_found:
                available_objectives = [s.get("objective", "unknown") for s in af_settings_list]
                error_msg = (
                    f"No autofocus settings found for objective '{current_objective}' in {autofocus_file}\n"
                    f"Available objectives in config: {available_objectives}\n"
                    f"Cannot proceed with acquisition - please add autofocus settings for '{current_objective}' "
                    f"or verify the objective name matches the configuration."
                )
                logger.error(error_msg)
                set_state("FAILED", error_msg)
                return

        except Exception as e:
            error_msg = f"Error loading autofocus settings: {e}"
            logger.error(error_msg, exc_info=True)
            set_state("FAILED", error_msg)
            return

        # Map score metric name to function
        score_metric_map = {
            "laplacian_variance": AutofocusUtils.autofocus_profile_laplacian_variance,
            "sobel": AutofocusUtils.autofocus_profile_sobel,
            "brenner_gradient": AutofocusUtils.autofocus_profile_brenner_gradient,
            "robust_sharpness": AutofocusUtils.autofocus_profile_robust_sharpness_metric,
            "hybrid_sharpness": AutofocusUtils.autofocus_profile_hybrid_sharpness_metric,
        }
        af_score_metric = score_metric_map.get(
            af_score_metric_name, AutofocusUtils.autofocus_profile_laplacian_variance
        )

        # Calculate timing window size for progress estimation (3x autofocus n_tiles)
        # Use n_tiles (number of AF positions) not n_steps (Z-positions per AF)
        # This ensures we collect enough tiles to see the AF timing pattern
        # TODO: Move timing window multiplier (3) to settings/preferences instead of hardcoding
        timing_window_size = max(10, 3 * af_n_tiles)  # Minimum 10 tiles
        logger.info(f"Timing window size for progress estimation: {timing_window_size} tiles (3 x {af_n_tiles} AF positions, min 10)")

        # Write timing window to file for Java progress dialog
        # Include all information needed for accurate time estimation:
        # - timing_window_size: how many tiles to use for rolling average
        # - af_n_tiles: number of adaptive autofocus positions per annotation
        # - total_tiles: total number of tile positions in this annotation
        # - af_n_steps: number of Z-steps per autofocus operation (for reference)
        timing_metadata_path = output_path / "acquisition_metadata.txt"
        with open(timing_metadata_path, "w") as f:
            f.write(f"timing_window_size={timing_window_size}\n")
            f.write(f"af_n_tiles={af_n_tiles}\n")
            f.write(f"total_tiles={len(positions)}\n")
            f.write(f"af_n_steps={af_n_steps}\n")
            f.write(f"objective={current_objective}\n")
        logger.info(f"Wrote timing metadata to {timing_metadata_path}: "
                    f"window={timing_window_size}, af_positions={af_n_tiles}, tiles={len(positions)}")

        af_positions, af_min_distance = AutofocusUtils.get_autofocus_positions(
            fov, xy_positions, n_tiles=af_n_tiles
        )

        logger.info(f"Autofocus positions: {af_positions}")

        # Create dynamic autofocus positions set (can be modified during acquisition)
        dynamic_af_positions = set(af_positions)
        deferred_af_positions = set()  # Track positions where AF was deferred

        # Track whether we've performed the first successful autofocus with tissue
        # Use standard autofocus on first tissue, then adaptive for speed on subsequent
        first_tissue_autofocus_done = False

        metadata_txt_for_positions = output_path / "image_positions_metadata.txt"

        # Apply Z-focus hint if provided (predicted from tilt correction model)
        hint_z = params.get("hint_z")
        if hint_z is not None:
            current_z = hardware.get_current_position().z
            logger.info(f"Z-focus hint received: {hint_z:.2f} um (current Z: {current_z:.2f} um)")
            logger.info(f"Moving to predicted Z position before acquisition...")
            hardware.move_to_position(Position(z=hint_z))
            logger.info(f"Moved to predicted Z: {hint_z:.2f} um")

        # CRITICAL: Run autofocus BEFORE acquiring any tiles
        # Use the diagonal position (af_positions[0]) which is 1 FOV inward from the
        # corner to avoid focusing on buffer regions outside tissue.
        # The hint_z serves as a starting point for the autofocus search.
        if len(positions) > 0 and len(af_positions) > 0:
            # Get the first autofocus position (diagonal offset for large grids)
            first_af_idx = af_positions[0]
            first_af_pos, first_af_filename = positions[first_af_idx]
            logger.info(f"=== PRE-ACQUISITION AUTOFOCUS at position {first_af_idx} ===")
            logger.info(f"Using diagonal autofocus position: X={first_af_pos.x}, Y={first_af_pos.y}")

            # For PPM, set rotation to 90deg for autofocus and tissue detection
            exposure_90 = 2.0  # Default
            if "ppm" in modality.lower():
                hardware.set_psg_ticks(90.0)
                logger.info("Set rotation to 90 deg (uncrossed) for initial autofocus")
                if 90.0 in params["angles"]:
                    angle_idx = params["angles"].index(90.0)
                    if angle_idx < len(params["exposures"]):
                        exposure_90 = params["exposures"][angle_idx]
                hardware.set_exposure(exposure_90)
                logger.info(f"Set exposure to {exposure_90}ms for initial autofocus")

            # Calculate direction toward center for tissue search loop
            start_pos = np.array([first_af_pos.x, first_af_pos.y])
            center_pos = np.mean(xy_positions, axis=0)
            direction = center_pos - start_pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)

            # Tissue detection loop: try current position, then move 1 FOV toward center
            # After 3 attempts with no tissue, show manual dialog
            max_tissue_search_attempts = 3
            tissue_found = False
            search_pos = Position(first_af_pos.x, first_af_pos.y, hardware.get_current_position().z)
            fov_diagonal = np.sqrt(fov[0]**2 + fov[1]**2)

            for attempt in range(max_tissue_search_attempts):
                # Move to search position
                hardware.move_to_position(search_pos)
                logger.info(f"Tissue search attempt {attempt + 1}/{max_tissue_search_attempts}: "
                            f"X={search_pos.x:.1f}, Y={search_pos.y:.1f}")

                # Take test image for tissue detection
                test_img, _ = hardware.snap_image()

                # Ensure consistent format for tissue detection
                if test_img.dtype in [np.float32, np.float64]:
                    if test_img.max() <= 1.0 and test_img.min() >= 0.0:
                        test_img = (test_img * 255).astype(np.uint8)
                    else:
                        test_img = np.clip(test_img, 0, 255).astype(np.uint8)

                # Check for tissue
                has_tissue, tissue_stats = AutofocusUtils.has_sufficient_tissue(
                    test_img,
                    texture_threshold=af_texture_threshold,
                    tissue_area_threshold=af_tissue_area_threshold,
                    modality=modality,
                    logger=logger,
                    return_stats=True,
                    rgb_brightness_threshold=af_rgb_brightness_threshold,
                )

                if has_tissue:
                    logger.info(f"Tissue found at attempt {attempt + 1}")
                    tissue_found = True
                    break
                else:
                    reason = "blank tile (RGB)" if tissue_stats.get('brightness_rejected') else "insufficient texture/area"
                    logger.warning(f"No tissue at attempt {attempt + 1} ({reason}) - "
                                   f"texture={tissue_stats['texture']:.4f}, area={tissue_stats['area']:.3f}")

                    if attempt < max_tissue_search_attempts - 1:
                        # Move one FOV diagonal toward center for next attempt
                        new_xy = np.array([search_pos.x, search_pos.y]) + direction * fov_diagonal
                        search_pos = Position(new_xy[0], new_xy[1], search_pos.z)
                        logger.info(f"Moving 1 FOV diagonal toward center for next attempt")

            # Run autofocus (with manual fallback if no tissue found)
            try:
                if tissue_found:
                    # Standard autofocus with manual fallback - tissue was found
                    logger.info("Tissue found - running autofocus with manual fallback")
                    initial_z = autofocus_with_manual_fallback(
                        hardware=hardware,
                        request_manual_focus=request_manual_focus,
                        max_retries=3,
                        n_steps=af_n_steps,
                        search_range=af_search_range,
                        score_metric=af_score_metric,
                        diagnostic_output_path=str(output_path),
                        logger=logger,
                    )
                else:
                    # No tissue found after all search attempts - try autofocus anyway
                    # but with max_retries=0 so it goes immediately to manual dialog
                    # if autofocus fails (which it likely will on blank area)
                    logger.warning(f"No tissue found after {max_tissue_search_attempts} search attempts")
                    logger.warning("Attempting autofocus anyway - will go to manual dialog if it fails")
                    initial_z = autofocus_with_manual_fallback(
                        hardware=hardware,
                        request_manual_focus=request_manual_focus,
                        max_retries=0,  # No retries - go straight to manual if AF fails
                        n_steps=af_n_steps,
                        search_range=af_search_range,
                        score_metric=af_score_metric,
                        diagnostic_output_path=str(output_path),
                        logger=logger,
                    )

                logger.info(f"Initial autofocus completed: Z={initial_z:.2f} um")
                first_tissue_autofocus_done = True

                # Remove this position from dynamic_af_positions since we already did it
                dynamic_af_positions.discard(first_af_idx)

            except RuntimeError as e:
                logger.error(f"Initial autofocus failed: {e}")
                # Continue anyway - user may have chosen to skip or acquisition was cancelled
                if "cancelled" in str(e).lower():
                    set_state("CANCELLED")
                    return

            logger.info(f"=== Starting main acquisition loop ===")

        # Main acquisition loop
        for pos_idx, (pos, filename) in enumerate(positions):
            # Check for cancellation
            if is_cancelled():
                logger.warning(f"Acquisition cancelled by client {client_addr}")
                set_state("CANCELLED")
                return

            logger.info(f"Position {pos_idx + 1}/{len(positions)}: {filename}")

            # Start timing for this tile
            tile_start = time.perf_counter()

            # Ensure Z is current autofocus value
            pos.z = hardware.get_current_position().z

            # Move to position
            logger.info(f"Moving to position: X={pos.x}, Y={pos.y}, Z={pos.z}")
            t0 = time.perf_counter()
            hardware.move_to_position(pos)
            t0 = log_timing(logger, "Stage XY movement command", t0)

            # Perform autofocus if needed (with tissue detection)
            if pos_idx in dynamic_af_positions:
                logger.info(
                    f"Checking for autofocus at position {pos_idx}: X={pos.x}, Y={pos.y}, Z={pos.z}"
                )

                # For PPM, always autofocus at 90 deg (uncrossed polarizers - brightest, fastest)
                # This ensures consistent, fast autofocus regardless of angle sequence
                if "ppm" in modality.lower():
                    t_rot = time.perf_counter()
                    hardware.set_psg_ticks(90.0)
                    t_rot = log_timing(logger, "Rotation to 90deg for autofocus", t_rot)
                    logger.info("Set rotation to 90 deg (uncrossed) for PPM autofocus")
                    # CRITICAL: Set appropriate exposure for 90 deg before tissue detection
                    # Find the 90 deg exposure from acquisition parameters
                    exposure_90 = 2.0  # Default fallback

                    if 90.0 in params["angles"]:
                        angle_idx = params["angles"].index(90.0)
                        if angle_idx < len(params["exposures"]):
                            exposure_90 = params["exposures"][angle_idx]

                    t_exp = time.perf_counter()
                    hardware.set_exposure(exposure_90)
                    t_exp = log_timing(logger, "Set exposure for tissue detection", t_exp)
                    logger.info(f"Set exposure to {exposure_90}ms for 90 deg tissue detection")
                # Take a quick image to assess tissue content
                t_snap = time.perf_counter()
                test_img, _ = hardware.snap_image()
                t_snap = log_timing(logger, "Snap test image for tissue detection", t_snap)

                # Ensure consistent format for tissue detection
                if test_img.dtype in [np.float32, np.float64]:
                    # Check if already normalized (0-1 range)
                    if test_img.max() <= 1.0 and test_img.min() >= 0.0:
                        # Convert to uint8 to match expected format
                        test_img = (test_img * 255).astype(np.uint8)
                        logger.info(
                            "Converted normalized float image to uint8 for tissue detection"
                        )
                    else:
                        # Float but not normalized - clip and convert
                        test_img = np.clip(test_img, 0, 255).astype(np.uint8)
                        logger.info("Converted float image to uint8 for tissue detection")

                # Check if there's sufficient tissue for reliable autofocus
                # Use thresholds from autofocus config (per-objective settings)
                has_tissue, tissue_stats = AutofocusUtils.has_sufficient_tissue(
                    test_img,
                    texture_threshold=af_texture_threshold,
                    tissue_area_threshold=af_tissue_area_threshold,
                    modality=modality,
                    logger=logger,
                    return_stats=True,
                    rgb_brightness_threshold=af_rgb_brightness_threshold,
                )

                if has_tissue:
                    logger.info(f"Sufficient tissue detected - performing autofocus")
                    rgb_info = ""
                    if tissue_stats.get('rgb_mean') is not None:
                        rgb_info = f", RGB brightness={tissue_stats['avg_brightness']:.1f} (threshold<{tissue_stats['brightness_threshold']:.1f})"
                    logger.info(
                        f"  Tissue stats: texture={tissue_stats['texture']:.4f} (threshold={tissue_stats['texture_threshold']:.4f}), "
                        f"area={tissue_stats['area']:.3f} (threshold={tissue_stats['area_threshold']:.3f}){rgb_info}"
                    )

                    # Use STANDARD autofocus on first tissue position for accuracy
                    # Then use ADAPTIVE autofocus on subsequent positions for speed
                    if not first_tissue_autofocus_done:
                        logger.info(f"  First tissue position - using STANDARD autofocus for accuracy")
                        t_af = time.perf_counter()
                        new_z = autofocus_with_manual_fallback(
                            hardware=hardware,
                            logger=logger,
                            request_manual_focus=request_manual_focus,
                            max_retries=3,
                            move_stage_to_estimate=True,
                            n_steps=af_n_steps,
                            search_range=af_search_range,
                            interp_strength=af_interp_strength,
                            interp_kind=af_interp_kind,
                            score_metric=af_score_metric,
                            diagnostic_output_path=output_path,
                            position_index=pos_idx,
                        )
                        t_af = log_timing(logger, "STANDARD autofocus", t_af)
                        first_tissue_autofocus_done = True
                        logger.info(f"  Standard autofocus :: New Z {new_z}")
                    else:
                        # Get Z position before adaptive autofocus for drift detection
                        z_before_adaptive = hardware.get_current_position().z

                        logger.info(f"  Subsequent tissue position - using ADAPTIVE autofocus for speed")
                        t_af = time.perf_counter()
                        new_z = hardware.autofocus_adaptive_search(
                            initial_step_size=af_adaptive_initial_step,
                            min_step_size=af_adaptive_min_step,
                            focus_threshold=af_adaptive_focus_threshold,
                            max_total_steps=af_adaptive_max_steps,
                            score_metric=af_score_metric,
                            pop_a_plot=False,
                            move_stage_to_estimate=True,
                        )
                        t_af = log_timing(logger, "ADAPTIVE autofocus", t_af)

                        # Check for large drift and fall back to STANDARD autofocus if needed
                        drift = abs(new_z - z_before_adaptive)
                        logger.info(f"  Adaptive autofocus :: New Z {new_z} (drift: {new_z - z_before_adaptive:+.2f} um)")

                        if drift > af_large_drift_threshold:
                            logger.warning(
                                f"  Large drift detected ({drift:.2f} um > {af_large_drift_threshold:.2f} um threshold)!"
                            )
                            logger.warning(f"  Falling back to STANDARD autofocus to re-establish baseline...")

                            t_af_recovery = time.perf_counter()
                            new_z = autofocus_with_manual_fallback(
                                hardware=hardware,
                                logger=logger,
                                request_manual_focus=request_manual_focus,
                                max_retries=3,
                                move_stage_to_estimate=True,
                                n_steps=af_n_steps,
                                search_range=af_search_range,
                                interp_strength=af_interp_strength,
                                interp_kind=af_interp_kind,
                                score_metric=af_score_metric,
                            )
                            t_af_recovery = log_timing(logger, "STANDARD autofocus (drift recovery)", t_af_recovery)
                            logger.info(f"  STANDARD autofocus (drift recovery) :: New Z {new_z}")
                else:
                    reason = "blank tile (RGB)" if tissue_stats.get('brightness_rejected') else "insufficient texture/area"
                    logger.warning(
                        f"Insufficient tissue at position {pos_idx} ({reason}) - deferring autofocus"
                    )
                    rgb_info = ""
                    if tissue_stats.get('rgb_mean') is not None:
                        rgb_info = f", RGB brightness={tissue_stats['avg_brightness']:.1f} (threshold<{tissue_stats['brightness_threshold']:.1f})"
                    logger.warning(
                        f"  Tissue stats: texture={tissue_stats['texture']:.4f} (threshold={tissue_stats['texture_threshold']:.4f}), "
                        f"area={tissue_stats['area']:.3f} (threshold={tissue_stats['area_threshold']:.3f}){rgb_info}"
                    )

                    # Remove this position from autofocus list
                    dynamic_af_positions.discard(pos_idx)
                    deferred_af_positions.add(pos_idx)

                    # Try to find next suitable position for autofocus
                    next_af_pos = AutofocusUtils.defer_autofocus_to_next_tile(
                        current_pos_idx=pos_idx,
                        original_af_positions=af_positions,
                        total_positions=len(positions),
                        af_min_distance=af_min_distance,
                        positions=xy_positions,
                        logger=logger,
                    )

                    if next_af_pos is not None and next_af_pos < len(positions):
                        dynamic_af_positions.add(next_af_pos)
                        logger.info(f"Added position {next_af_pos} to autofocus queue")
                    else:
                        logger.warning(f"Could not find suitable position to defer autofocus to")

            # Collect stage position for this tile (after autofocus, before acquiring angles)
            # This captures the actual XYZ used for acquisition
            current_stage_pos = hardware.get_current_position()
            stage_positions_collected.append((
                filename,
                current_stage_pos.x,
                current_stage_pos.y,
                current_stage_pos.z
            ))

            if params["angles"]:
                # Storage for birefringence image calculation
                angle_images = {}

                # Multi-angle acquisition
                for angle_idx, angle in enumerate(params["angles"]):
                    # Check for cancellation
                    if is_cancelled():
                        logger.warning(f"Acquisition cancelled by client {client_addr}")
                        set_state("CANCELLED")
                        return

                    # Start timing for this angle
                    angle_start = time.perf_counter()

                    # Set rotation angle
                    # First angle of each position should reset to "a" polarization state
                    # is_sequence_start = angle_idx == 0
                    t_rot = time.perf_counter()
                    hardware.set_psg_ticks(angle)  # , is_sequence_start=is_sequence_start)
                    t_rot = log_timing(logger, f"Rotation to {angle}deg", t_rot)

                    # Backup check of angle - seem to be having hardware issues sometimes
                    # actual_angle = hardware.get_psg_ticks()
                    # angle_diff = min(abs(actual_angle - angle), 360 - abs(actual_angle - angle))
                    # if angle_diff > 5.0:
                    #     logger.warning(f"  Angle mismatch: requested {angle:.1f} deg, got {actual_angle:.1f} deg, retrying...")
                    #     hardware.set_psg_ticks(angle, is_sequence_start=False)
                    #     time.sleep(0.15)
                    #     actual_angle = hardware.get_psg_ticks()
                    # logger.info(f"  Angle set to {hardware.get_psg_ticks():.1f}")

                    # Set exposure time
                    # Priority 1: JAI hardware calibration (per-channel exposures for white balance)
                    # Priority 2: Single exposure from params
                    t_exp = time.perf_counter()
                    if jai_calibration is not None:
                        # Apply JAI per-channel exposures from calibration
                        applied = apply_jai_calibration_for_angle(
                            hardware=hardware,
                            jai_calibration=jai_calibration,
                            angle=angle,
                            per_angle=white_balance_per_angle,
                            logger=logger,
                        )
                        if not applied and angle_idx < len(params["exposures"]):
                            # Fall back to single exposure if JAI calibration failed
                            exposure_ms = params["exposures"][angle_idx]
                            hardware.set_exposure(exposure_ms)
                            logger.info(f"  JAI calibration failed, using single exposure: {exposure_ms}ms")
                    elif angle_idx < len(params["exposures"]):
                        # No JAI calibration - use single exposure from params
                        exposure_ms = params["exposures"][angle_idx]
                        hardware.set_exposure(exposure_ms)
                    t_exp = log_timing(logger, f"Set exposure for angle {angle}deg", t_exp)
                    logger.info(f"  Exposure set to {hardware.core.get_exposure()}")

                    # Acquire image
                    t_snap = time.perf_counter()
                    image, metadata = hardware.snap_image(debayering=False)
                    t_snap = log_timing(logger, f"Snap image at {angle}deg (includes camera+USB+internal processing)", t_snap)

                    if image is None:
                        logger.error(f"Failed to acquire image at angle {angle}")
                        continue

                    # Calculate image stats (numpy operation)
                    t_stats = time.perf_counter()
                    img_mean = image.mean((0,1))
                    t_stats = log_timing(logger, f"Calculate image stats at {angle}deg", t_stats)
                    logger.info(f"  Image shape: {image.shape}, mean: {img_mean}")

                    # Save raw (unprocessed) image for comparison
                    raw_output_path = output_path.parent / "Raw" / output_path.name
                    raw_image_path = raw_output_path / str(angle) / filename

                    t_mkdir = time.perf_counter()
                    if not raw_image_path.parent.exists():
                        raw_image_path.parent.mkdir(parents=True, exist_ok=True)
                    t_mkdir = log_timing(logger, f"Create directories at {angle}deg", t_mkdir)

                    try:
                        t_save_raw = time.perf_counter()
                        TifWriterUtils.ome_writer(  # raw
                            filename=str(raw_image_path),
                            pixel_size_um=hardware.core.get_pixel_size_um(),
                            data=image,
                        )
                        t_save_raw = log_timing(logger, f"Save raw image at {angle}deg (OME-TIFF write)", t_save_raw)
                        logger.info(f"  Saved raw image: {raw_image_path}")
                        write_position_metadata(
                            metadata_txt_for_positions, raw_image_path, hardware, modality
                        )
                    except Exception as e:
                        logger.warning(f"  Failed to save raw image: {e}")

                    # ======= APPLY BACKGROUND CORRECTION (STEP 1) =======
                    # Check if background correction is enabled, background exists, and angle is not disabled
                    if (
                        background_correction_enabled
                        and angle in background_images
                        and angle not in background_disabled_angles
                    ):
                        bg_img = background_images[angle]
                        logger.info(f"  Applying background correction for {angle} degrees")
                        logger.info(
                            f"    Background stats: mean={bg_img.mean():.1f}, std={bg_img.std():.1f}"
                        )

                        t_bg = time.perf_counter()
                        image = BackgroundCorrectionUtils.apply_flat_field_correction(
                            image,
                            background_images[angle],
                            background_scaling_factors[angle],
                            method=background_correction_method,
                        )
                        t_bg = log_timing(logger, f"Background correction at {angle}deg", t_bg)
                        logger.info(
                            f"    Correction applied with method: {background_correction_method}"
                        )
                        logger.info(f"    Post-correction RGB means: {image.mean(axis=(0,1))}")
                    elif background_correction_enabled and angle in background_disabled_angles:
                        logger.info(
                            f"  Background correction SKIPPED for {angle} deg (validation failed - exposure mismatch or missing background)"
                        )
                    elif background_correction_enabled and angle not in background_images:
                        logger.info(
                            f"  Background correction SKIPPED for {angle} deg (no background image available)"
                        )

                    # ======= APPLY WHITE BALANCE (STEP 2) =======
                    if white_balance_enabled:
                        # Use pre-configured white balance values
                        if angle in angles_wb:
                            wb_profile = angles_wb[angle]
                        else:
                            # Default neutral if angle not found
                            wb_profile = [1.0, 1.0, 1.0]
                            logger.warning(
                                f"    No white balance profile for {angle} deg, using neutral"
                            )

                        t_wb = time.perf_counter()
                        gain = calculate_luminance_gain(*wb_profile)
                        image = hardware.white_balance(
                            image, white_balance_profile=wb_profile, gain=gain
                        )
                        t_wb = log_timing(logger, f"White balance at {angle}deg", t_wb)
                        logger.info(
                            f"  Applied white balance: R={wb_profile[0]:.2f}, G={wb_profile[1]:.2f}, B={wb_profile[2]:.2f}"
                        )

                    # Save processed image
                    image_path = output_path / str(angle) / filename

                    if image_path.parent.exists():
                        t_save_proc = time.perf_counter()
                        TifWriterUtils.ome_writer(  # processed
                            filename=str(image_path),
                            pixel_size_um=hardware.core.get_pixel_size_um(),
                            data=image,
                        )
                        t_save_proc = log_timing(logger, f"Save processed image at {angle}deg", t_save_proc)
                        image_count += 1
                        update_progress(image_count, total_images)

                        # Store image for birefringence calculation
                        angle_images[angle] = image

                        # Log total time for this angle
                        angle_elapsed_ms = (time.perf_counter() - angle_start) * 1000
                        logger.info(f"  [TIMING] Total for angle {angle}deg: {angle_elapsed_ms:.1f}ms")
                    else:
                        logger.error(f"Failed to save {image_path} - parent directory missing")

                # Create birefringence image for this tile after all angles acquired
                positive_angles = [a for a in angle_images.keys() if a > 0 and a != 90]
                negative_angles = [a for a in angle_images.keys() if a < 0]

                if positive_angles and negative_angles:
                    pos_angle = min(positive_angles)
                    neg_angle = max(negative_angles)

                    # Set up birefringence directory and tile config source
                    biref_dir = output_path / f"{pos_angle}.biref"
                    tile_config_source = output_path / str(pos_angle) / "TileConfiguration.txt"

                    # Create normalized birefringence image
                    # Uses [I(+) - I(-)]/[I(+) + I(-)] to suppress H&E staining variations
                    t_biref = time.perf_counter()
                    TifWriterUtils.create_normalized_birefringence_tile(
                        pos_image=angle_images[pos_angle],
                        neg_image=angle_images[neg_angle],
                        output_dir=biref_dir,
                        filename=filename,
                        pixel_size_um=hardware.core.get_pixel_size_um(),
                        tile_config_source=tile_config_source,
                        logger=logger,
                    )
                    t_biref = log_timing(logger, f"Normalized birefringence calculation and save", t_biref)

                    # # Create sum image alongside birefringence image
                    # sum_dir = output_path / f"{pos_angle}.sum"
                    # TifWriterUtils.create_sum_tile(
                    #     pos_image=angle_images[pos_angle],
                    #     neg_image=angle_images[neg_angle],
                    #     output_dir=sum_dir,
                    #     filename=filename,
                    #     pixel_size_um=hardware.core.get_pixel_size_um(),
                    #     tile_config_source=tile_config_source,
                    #     logger=logger,
                    # )

            else:
                # Single image acquisition: no angles specified
                image, metadata = hardware.snap_image()
                image_path = output_path / filename

                if image_path.parent.exists():
                    TifWriterUtils.ome_writer(  # brightfield
                        filename=str(image_path),
                        pixel_size_um=hardware.core.get_pixel_size_um(),
                        data=image,
                    )
                    image_count += 1
                    update_progress(image_count, total_images)

                try:
                    write_position_metadata(
                        metadata_txt_for_positions, image_path, hardware, modality
                    )
                except Exception as e:
                    logger.warning(
                        f"  Failed to write position text {metadata_txt_for_positions}: {e}"
                    )

            # Log total time for this tile/position
            tile_elapsed_ms = (time.perf_counter() - tile_start) * 1000
            logger.info(f"[TIMING] === TOTAL TILE TIME: {tile_elapsed_ms:.1f}ms ({tile_elapsed_ms/1000:.2f}s) ===")

        # Save device properties
        current_props = hardware.get_device_properties()
        props_path = output_path / "MMproperties.txt"
        with open(props_path, "w") as fid:
            from pprint import pprint as dict_printer

            dict_printer(current_props, stream=fid)

        # Write TileConfiguration with stage coordinates including Z
        if stage_positions_collected:
            TileConfigUtils.write_tileconfig_stage(output_path, stage_positions_collected)

        # Get final Z position for tilt correction model
        final_z = hardware.get_current_position().z
        set_state("COMPLETED", final_z=final_z)
        logger.info("=== ACQUISITION COMPLETED SUCCESSFULLY ===")
        logger.info(f"Final Z position: {final_z:.2f} um")
        logger.info(f"Total images saved: {image_count}/{total_images}")
        logger.info(f"Output directory: {output_path}")

        # Report autofocus activity
        if deferred_af_positions:
            logger.info(
                f"Autofocus deferred at {len(deferred_af_positions)} positions due to insufficient tissue: {sorted(deferred_af_positions)}"
            )
        # else:
        #    logger.info(
        #    f"Autofocus completed at {len([p for p in af_positions if p not in deferred_af_positions])} positions")

    except Exception as e:
        logger.error("=== ACQUISITION FAILED ===")
        logger.error(f"Error: {str(e)}", exc_info=True)
        set_state("FAILED", str(e))
    finally:
        # Return to starting position
        logger.info("Returning to starting position")
        hardware.move_to_position(starting_position)


def write_position_metadata(metadata_txt_for_positions, raw_image_path, hardware, modality):
    pos_read = hardware.get_current_position()
    line = (
        f"filename = {raw_image_path} ; "
        f"(x,y,z) = ({pos_read.x},{pos_read.y},{round(pos_read.z, 3)}); "
    )

    # TODO: modality is chosen on the config , but then overwrittebn by the message parameter
    # here we should use the modality used for acquisition?
    # if modality.lower.count("ppm") > 0:  # user-set value passed from acquisition parameters
    # if hardware.settings.get("modality", "ppm") == "ppm":  # config set value from yaml

    if "ppm" in modality.lower():
        angle = (
            hardware.get_psg_ticks()
            if hardware.settings.get("ppm_optics", "ZCutQuartz") != "NA"
            else "NA"
        )
        line += f"r = {angle} ; "

    line += f"exposure (ms) = {hardware.core.get_exposure()}\n"

    with open(metadata_txt_for_positions, "a") as f:
        f.write(line)


def angle_to_name(angle: float) -> str:
    """
    Convert numeric angle to canonical name.

    Args:
        angle: Rotation angle in degrees

    Returns:
        Angle name (e.g., 'uncrossed', 'crossed', 'positive', 'negative')
    """
    abs_angle = abs(angle)

    if 88 <= abs_angle <= 92:
        return "uncrossed"
    elif abs_angle <= 3:
        return "crossed"
    elif 4 <= abs_angle <= 10:
        return "positive" if angle > 0 else "negative"
    else:
        return f"angle_{angle}"


def get_default_target_intensity(modality: str, angle: float) -> float:
    """
    Get hardcoded default target intensity for background acquisition.

    These defaults are used when no YAML configuration is available.
    The values are based on the optical properties of polarized light:
    - Crossed polarizers (0 deg): Very dim -> 125
    - Birefringence angles (5-7 deg): Moderate -> 160
    - Uncrossed (90 deg): Very bright -> 245
    - Intermediate: Standard -> 180

    Args:
        modality: Modality identifier (e.g., "ppm", "brightfield")
        angle: Rotation angle in degrees (for PPM)

    Returns:
        Target grayscale intensity (0-255)
    """
    modality_lower = modality.lower()

    # Brightfield modality
    if "brightfield" in modality_lower or "bf" in modality_lower:
        return 250.0

    # PPM modality - angle-specific targets
    if "ppm" in modality_lower:
        abs_angle = abs(angle)

        if 88 <= abs_angle <= 92:
            # Near-uncrossed region (around 90 deg) - brightest
            return 245.0
        elif 4 <= abs_angle <= 10:
            # Birefringence angles (5-7 deg and their neighbors)
            return 160.0
        elif abs_angle <= 3:
            # Near-crossed region (around 0 deg) - dimmest
            return 125.0
        else:
            # Intermediate angles (10-88 deg)
            return 180.0

    # Default fallback
    return 180.0


def load_calibration_targets_from_yaml(config_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load calibration targets from imageprocessing YAML file.

    Looks for the calibration_targets section which contains:
    - target_intensities: Per-angle default targets
    - background_exposures: Achieved intensities from prior background collection

    Args:
        config_path: Path to the main config file (config_PPM.yml)

    Returns:
        Dictionary with calibration_targets data or None if not found
    """
    config_path = Path(config_path)

    # Derive imageprocessing file path
    config_name = config_path.stem
    if config_name.startswith("config_"):
        microscope_name = config_name[7:]
        imageprocessing_name = f"imageprocessing_{microscope_name}.yml"
    else:
        imageprocessing_name = f"imageprocessing_{config_name}.yml"

    imageprocessing_path = config_path.parent / imageprocessing_name

    if not imageprocessing_path.exists():
        return None

    try:
        with open(imageprocessing_path, "r") as f:
            ip_data = yaml.safe_load(f) or {}
        return ip_data.get("calibration_targets")
    except Exception as e:
        logger.warning(f"Failed to load calibration targets from {imageprocessing_path}: {e}")
        return None


def get_target_intensity_for_angle(
    angle: float,
    modality: str = "ppm",
    config_path: Optional[Path] = None,
) -> Tuple[float, str]:
    """
    Get target intensity for a specific angle with YAML priority logic.

    Priority order:
    1. background_exposures.angles.{name}.achieved_intensity (from prior BG collection)
    2. calibration_targets.target_intensities.{name} (YAML configured)
    3. Hardcoded defaults (based on optical properties)

    This ensures white balance calibration uses the same target intensity
    as background collection, so white-balanced images match backgrounds.

    Args:
        angle: Rotation angle in degrees
        modality: Modality identifier (default: "ppm")
        config_path: Path to config file (optional, enables YAML lookup)

    Returns:
        Tuple of (target_intensity, source) where source describes where
        the value came from (e.g., "background_exposures", "yaml_config", "default")
    """
    angle_name = angle_to_name(angle)

    # Try YAML lookup if config_path provided
    if config_path is not None:
        cal_targets = load_calibration_targets_from_yaml(config_path)
        if cal_targets is not None:
            # Priority 1: Check background_exposures (achieved intensity from BG collection)
            bg_exposures = cal_targets.get("background_exposures", {})
            if bg_exposures and "angles" in bg_exposures:
                angle_data = bg_exposures["angles"].get(angle_name)
                if angle_data and "achieved_intensity" in angle_data:
                    return float(angle_data["achieved_intensity"]), "background_exposures"

            # Priority 2: Check configured target_intensities
            target_intensities = cal_targets.get("target_intensities", {})
            if angle_name in target_intensities:
                return float(target_intensities[angle_name]), "yaml_config"
            # Also check for 'default' key
            if "default" in target_intensities:
                return float(target_intensities["default"]), "yaml_config_default"

    # Priority 3: Hardcoded defaults
    return get_default_target_intensity(modality, angle), "default"


def get_target_intensity_for_background(modality: str, angle: float) -> float:
    """
    Get target intensity for background acquisition based on modality and angle.

    This is a convenience wrapper around get_target_intensity_for_angle() that
    only returns the intensity value (not the source). Use this for backward
    compatibility with existing code.

    For new code that needs to know where the value came from (e.g., to log
    whether YAML or defaults are being used), use get_target_intensity_for_angle().

    Args:
        modality: Modality identifier (e.g., "ppm", "brightfield")
        angle: Rotation angle in degrees (for PPM)

    Returns:
        Target grayscale intensity (0-255)

    Examples:
        >>> get_target_intensity_for_background("brightfield", 0)
        250.0
        >>> get_target_intensity_for_background("ppm", 90)
        245.0
        >>> get_target_intensity_for_background("ppm", 5)
        160.0
        >>> get_target_intensity_for_background("ppm", -5)
        160.0
        >>> get_target_intensity_for_background("ppm", 0)
        125.0
    """
    # For backward compatibility, use defaults only (no YAML lookup)
    # Callers that want YAML lookup should use get_target_intensity_for_angle()
    return get_default_target_intensity(modality, angle)


def save_background_exposures_to_yaml(
    config_path: Path,
    final_exposures: Dict[float, float],
    achieved_intensities: Dict[float, float],
    modality: str = "ppm",
    objective: Optional[str] = None,
    detector: Optional[str] = None,
) -> bool:
    """
    Save background collection exposures and achieved intensities to YAML.

    Updates the calibration_targets.background_exposures section in the
    imageprocessing YAML file. This data becomes the source of truth for
    target intensities in white balance calibration.

    Args:
        config_path: Path to the main config file (config_PPM.yml)
        final_exposures: Dictionary mapping angles to final exposure times (ms)
        achieved_intensities: Dictionary mapping angles to achieved median intensity
        modality: Modality name (e.g., "ppm")
        objective: Objective LOCI ID (optional)
        detector: Detector LOCI ID (optional)

    Returns:
        True if successfully saved, False otherwise
    """
    from datetime import datetime

    config_path = Path(config_path)

    # Derive imageprocessing file path
    config_name = config_path.stem
    if config_name.startswith("config_"):
        microscope_name = config_name[7:]
        imageprocessing_name = f"imageprocessing_{microscope_name}.yml"
    else:
        imageprocessing_name = f"imageprocessing_{config_name}.yml"

    imageprocessing_path = config_path.parent / imageprocessing_name

    try:
        # Load existing file or create empty dict
        if imageprocessing_path.exists():
            with open(imageprocessing_path, "r") as f:
                ip_data = yaml.safe_load(f) or {}
        else:
            ip_data = {}

        # Ensure calibration_targets section exists
        if "calibration_targets" not in ip_data:
            ip_data["calibration_targets"] = {}

        # Build background_exposures data
        angles_data = {}
        for angle, exposure_ms in final_exposures.items():
            angle_name = angle_to_name(angle)
            angles_data[angle_name] = {
                "angle_degrees": angle,
                "exposure_ms": round(exposure_ms, 2),
                "achieved_intensity": round(achieved_intensities.get(angle, 0.0), 1),
            }

        ip_data["calibration_targets"]["background_exposures"] = {
            "last_calibrated": datetime.now().isoformat(),
            "modality": modality,
            "objective": objective,
            "detector": detector,
            "angles": angles_data,
        }

        # Also ensure target_intensities has defaults if not present
        if "target_intensities" not in ip_data["calibration_targets"]:
            ip_data["calibration_targets"]["target_intensities"] = {
                "uncrossed": 245.0,
                "positive": 160.0,
                "negative": 160.0,
                "crossed": 125.0,
                "default": 180.0,
            }

        # Save updated file
        with open(imageprocessing_path, "w") as f:
            yaml.dump(ip_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved background exposures to {imageprocessing_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save background exposures to YAML: {e}")
        return False


def acquire_background_with_target_intensity(
    hardware: PycromanagerHardware,
    target_intensity: float,
    tolerance: float = 2.5,
    initial_exposure_ms: float = 100.0,
    max_iterations: int = 10,
    logger=None,
) -> Tuple[np.ndarray, float]:
    """
    Acquire background image with adaptive exposure to reach target intensity.

    Uses proportional control to iteratively adjust exposure time until the
    median image intensity is within tolerance of the target value. Median is
    used instead of mean as it is more robust to outliers and hot pixels.

    Args:
        hardware: Microscope hardware interface
        target_intensity: Target median grayscale value (0-255)
        tolerance: Acceptable deviation from target (default +/-2.5)
        initial_exposure_ms: Starting exposure time in milliseconds
        max_iterations: Maximum adjustment iterations
        logger: Logger instance for tracking convergence

    Returns:
        Tuple of (image, final_exposure_ms)
            image: Acquired image at target intensity
            final_exposure_ms: Final exposure time used

    Raises:
        RuntimeError: If image acquisition fails
    """
    # Exposure bounds to prevent extreme values
    MIN_EXPOSURE_MS = 0.0001
    MAX_EXPOSURE_MS = 5000.0

    # Set initial exposure
    current_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, initial_exposure_ms))
    hardware.set_exposure(current_exposure)

    if logger:
        logger.info(
            f"Starting adaptive exposure: target={target_intensity:.1f}, "
            f"tolerance={tolerance:.1f}, initial_exposure={current_exposure:.1f}ms"
        )

    last_image = None
    last_exposure = current_exposure

    for iteration in range(max_iterations):
        # Snap image (debayering auto-detected based on camera type)
        image, metadata = hardware.snap_image()

        if image is None:
            raise RuntimeError(f"Failed to acquire image at iteration {iteration}")

        # Calculate median intensity across all channels (more robust than mean)
        mean_intensity = float(np.median(image))

        # Store for potential use if we don't converge
        last_image = image
        last_exposure = current_exposure

        if logger:
            logger.info(
                f"  Iteration {iteration + 1}/{max_iterations}: "
                f"median={mean_intensity:.1f}, exposure={current_exposure:.1f}ms"
            )

        # Check convergence
        intensity_error = abs(mean_intensity - target_intensity)
        if intensity_error <= tolerance:
            if logger:
                logger.info(
                    f"Converged! Final: median={mean_intensity:.1f}, "
                    f"exposure={current_exposure:.1f}ms, iterations={iteration + 1}"
                )
            return image, current_exposure

        # Calculate proportional adjustment
        # If image is too dark, increase exposure; if too bright, decrease
        if mean_intensity >= 254.0:
            # Image is saturated - decrease exposure aggressively
            # Proportional control alone is too slow when saturated
            new_exposure = max(current_exposure * 0.5, MIN_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image saturated (median={mean_intensity:.1f}), halving exposure to {new_exposure:.1f}ms"
                )
            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)
        elif mean_intensity > 0:
            adjustment_ratio = target_intensity / mean_intensity
            new_exposure = current_exposure * adjustment_ratio

            # Clamp to bounds
            new_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, new_exposure))

            if logger:
                logger.info(
                    f"    Adjusting exposure: {current_exposure:.1f}ms -> {new_exposure:.1f}ms "
                    f"(ratio={adjustment_ratio:.2f})"
                )

            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)
        else:
            # Image is completely black, increase exposure significantly
            new_exposure = min(current_exposure * 2.0, MAX_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image completely black, doubling exposure to {new_exposure:.1f}ms"
                )
            current_exposure = new_exposure
            hardware.set_exposure(current_exposure)

    # Max iterations reached without convergence
    if logger:
        logger.warning(
            f"Did not converge after {max_iterations} iterations. "
            f"Using last image: median={float(np.median(last_image)):.1f}, exposure={last_exposure:.1f}ms"
        )

    return last_image, last_exposure


def acquire_background_with_biref_matching(
    hardware: PycromanagerHardware,
    reference_image: np.ndarray,
    tolerance: float = 5.0,
    initial_exposure_ms: float = 100.0,
    max_iterations: int = 10,
    logger=None,
) -> Tuple[np.ndarray, float, float]:
    """
    Acquire background image optimized to minimize birefringence against reference.

    Instead of matching overall intensity, this directly minimizes the
    birefringence metric (sum of absolute channel differences) against
    a reference image (typically the positive angle background).

    This ensures that when birefringence is calculated as:
        |R_pos - R_neg| + |G_pos - G_neg| + |B_pos - B_neg|
    the result is minimized for background regions.

    Args:
        hardware: Microscope hardware interface
        reference_image: Reference image to match against (e.g., +7 deg background)
        tolerance: Target mean birefringence value (default 5.0, ideal is 0)
        initial_exposure_ms: Starting exposure time in milliseconds
        max_iterations: Maximum adjustment iterations
        logger: Logger instance for tracking convergence

    Returns:
        Tuple of (image, final_exposure_ms, mean_biref)
            image: Acquired image that minimizes birefringence
            final_exposure_ms: Final exposure time used
            mean_biref: Achieved mean birefringence value

    Raises:
        RuntimeError: If image acquisition fails
    """
    MIN_EXPOSURE_MS = 0.0001
    MAX_EXPOSURE_MS = 5000.0

    current_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, initial_exposure_ms))
    hardware.set_exposure(current_exposure)

    # Convert reference to int16 for signed arithmetic
    ref_i16 = reference_image.astype(np.int16)
    ref_mean = float(np.mean(reference_image))

    if logger:
        logger.info(
            f"Starting biref-matched exposure: target_biref<={tolerance:.1f}, "
            f"ref_mean={ref_mean:.1f}, initial_exposure={current_exposure:.1f}ms"
        )

    best_biref = float('inf')
    best_image = None
    best_exposure = current_exposure

    for iteration in range(max_iterations):
        image, metadata = hardware.snap_image()

        if image is None:
            raise RuntimeError(f"Failed to acquire image at iteration {iteration}")

        img_i16 = image.astype(np.int16)

        # Calculate birefringence metric (same as ppm_angle_difference)
        # This is: |R_ref - R_img| + |G_ref - G_img| + |B_ref - B_img| per pixel
        abs_diff = np.abs(ref_i16 - img_i16)
        biref_per_pixel = np.sum(abs_diff, axis=2)
        mean_biref = float(np.mean(biref_per_pixel))

        # Calculate signed mean difference to determine adjustment direction
        img_mean = float(np.mean(image))
        signed_diff = ref_mean - img_mean

        # Track best result
        if mean_biref < best_biref:
            best_biref = mean_biref
            best_image = image.copy()
            best_exposure = current_exposure

        if logger:
            logger.info(
                f"  Iteration {iteration + 1}/{max_iterations}: "
                f"biref={mean_biref:.1f}, img_mean={img_mean:.1f}, "
                f"signed_diff={signed_diff:+.1f}, exposure={current_exposure:.1f}ms"
            )

        # Check convergence
        if mean_biref <= tolerance:
            if logger:
                logger.info(
                    f"Converged! Final biref={mean_biref:.1f}, "
                    f"exposure={current_exposure:.1f}ms, iterations={iteration + 1}"
                )
            return image, current_exposure, mean_biref

        # Check if we can improve further with exposure adjustment
        # If images have similar overall intensity but high biref, it means
        # per-channel ratios differ - exposure alone cannot fix this
        if abs(signed_diff) < 2.0 and iteration > 0:
            if logger:
                logger.warning(
                    f"    Images have similar intensity (diff={signed_diff:+.1f}) "
                    f"but biref={mean_biref:.1f}. Per-channel differences may not be "
                    f"correctable by exposure adjustment alone."
                )
            # Continue trying a few more iterations in case we can improve
            if iteration >= 3:
                break

        # Proportional adjustment based on mean intensity difference
        if img_mean >= 254.0:
            # Image saturated - decrease aggressively
            new_exposure = max(current_exposure * 0.5, MIN_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image saturated, halving exposure to {new_exposure:.1f}ms"
                )
        elif img_mean > 0:
            # Adjust to match reference intensity
            adjustment_ratio = ref_mean / img_mean
            new_exposure = current_exposure * adjustment_ratio
            new_exposure = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, new_exposure))

            if logger:
                logger.info(
                    f"    Adjusting: {current_exposure:.1f}ms -> {new_exposure:.1f}ms "
                    f"(ratio={adjustment_ratio:.3f})"
                )
        else:
            # Image completely black
            new_exposure = min(current_exposure * 2.0, MAX_EXPOSURE_MS)
            if logger:
                logger.warning(
                    f"    Image black, doubling exposure to {new_exposure:.1f}ms"
                )

        current_exposure = new_exposure
        hardware.set_exposure(current_exposure)

    # Return best result found
    if logger:
        logger.info(
            f"Max iterations reached. Using best result: "
            f"biref={best_biref:.1f}, exposure={best_exposure:.1f}ms"
        )

    return best_image, best_exposure, best_biref


def acquire_background_with_per_channel_adaptive(
    hardware: PycromanagerHardware,
    initial_exposures: Dict[str, float],
    target_intensity: float = 200.0,
    tolerance: float = 2.5,
    max_iterations: int = 10,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Acquire background image using per-channel exposure mode with adaptive scaling.

    Unlike acquire_background_with_target_intensity which uses a single exposure,
    this function maintains per-channel exposure ratios (for white balance) while
    scaling all channels proportionally to reach the target intensity.

    Args:
        hardware: Microscope hardware interface
        initial_exposures: Dict with 'r', 'g', 'b' exposure values in ms
                          e.g., {'r': 45.0, 'g': 50.0, 'b': 55.0}
        target_intensity: Target median grayscale value (0-255)
        tolerance: Acceptable deviation from target (default +/-2.5)
        max_iterations: Maximum adjustment iterations
        logger: Logger instance for tracking convergence

    Returns:
        Tuple of (image, final_exposures)
            image: Acquired image at target intensity
            final_exposures: Dict with final per-channel exposures {'r': x, 'g': y, 'b': z}

    Raises:
        RuntimeError: If image acquisition fails
    """
    try:
        from microscope_control.jai import JAICameraProperties
    except ImportError:
        if logger:
            logger.warning("JAI camera module not available - falling back to single exposure")
        # Fall back to regular adaptive exposure
        image, final_exp = acquire_background_with_target_intensity(
            hardware=hardware,
            target_intensity=target_intensity,
            tolerance=tolerance,
            initial_exposure_ms=initial_exposures.get('g', 100.0),
            max_iterations=max_iterations,
            logger=logger,
        )
        return image, {'r': final_exp, 'g': final_exp, 'b': final_exp}

    # Exposure bounds
    MIN_EXPOSURE_MS = 0.01
    MAX_EXPOSURE_MS = 5000.0

    # Get initial per-channel exposures
    exp_r = max(MIN_EXPOSURE_MS, initial_exposures.get('r', 50.0))
    exp_g = max(MIN_EXPOSURE_MS, initial_exposures.get('g', 50.0))
    exp_b = max(MIN_EXPOSURE_MS, initial_exposures.get('b', 50.0))

    # Calculate ratios relative to green (reference channel)
    ratio_r = exp_r / exp_g
    ratio_b = exp_b / exp_g

    jai_props = JAICameraProperties(hardware.core)

    if logger:
        logger.info(
            f"Starting per-channel adaptive exposure: target={target_intensity:.1f}, "
            f"initial R={exp_r:.1f}ms, G={exp_g:.1f}ms, B={exp_b:.1f}ms"
        )
        logger.info(f"  Channel ratios (R:G:B) = {ratio_r:.3f}:1.000:{ratio_b:.3f}")

    # Apply initial per-channel exposures
    jai_props.set_channel_exposures(red=exp_r, green=exp_g, blue=exp_b, auto_enable=True)

    last_image = None
    last_exposures = {'r': exp_r, 'g': exp_g, 'b': exp_b}

    for iteration in range(max_iterations):
        # Snap image
        image, metadata = hardware.snap_image()

        if image is None:
            raise RuntimeError(f"Failed to acquire image at iteration {iteration}")

        # Calculate median intensity
        median_intensity = float(np.median(image))

        last_image = image
        last_exposures = {'r': exp_r, 'g': exp_g, 'b': exp_b}

        if logger:
            logger.info(
                f"  Iteration {iteration + 1}/{max_iterations}: "
                f"median={median_intensity:.1f}, G_exp={exp_g:.1f}ms"
            )

        # Check convergence
        if abs(median_intensity - target_intensity) <= tolerance:
            if logger:
                logger.info(
                    f"Converged! Final: median={median_intensity:.1f}, "
                    f"R={exp_r:.1f}ms, G={exp_g:.1f}ms, B={exp_b:.1f}ms"
                )
            return image, {'r': exp_r, 'g': exp_g, 'b': exp_b}

        # Calculate scale factor
        if median_intensity >= 254.0:
            # Saturated - reduce aggressively
            scale = 0.5
            if logger:
                logger.warning(f"    Image saturated, halving exposures")
        elif median_intensity > 0:
            scale = target_intensity / median_intensity
        else:
            # Black image - increase
            scale = 2.0
            if logger:
                logger.warning(f"    Image black, doubling exposures")

        # Scale all channel exposures proportionally (maintaining ratios)
        exp_g = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, exp_g * scale))
        exp_r = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, exp_g * ratio_r))
        exp_b = max(MIN_EXPOSURE_MS, min(MAX_EXPOSURE_MS, exp_g * ratio_b))

        if logger:
            logger.info(
                f"    Scaled exposures: R={exp_r:.1f}ms, G={exp_g:.1f}ms, B={exp_b:.1f}ms"
            )

        # Apply scaled per-channel exposures
        jai_props.set_channel_exposures(red=exp_r, green=exp_g, blue=exp_b, auto_enable=True)

    # Max iterations reached
    if logger:
        logger.warning(
            f"Did not converge after {max_iterations} iterations. "
            f"Using last image: median={float(np.median(last_image)):.1f}"
        )

    return last_image, last_exposures


def simple_background_collection(
    yaml_file_path: str,
    output_folder_path: str,
    modality: str,
    angles_str: str,
    exposures_str: str,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
    update_progress: Callable[[int, int], None],
    use_per_angle_wb: bool = False,
):
    """
    Simplified background collection for BackgroundCollectionWorkflow.

    Acquires background images at current position using adaptive exposure
    to reach target intensities. Saves directly to correct folder structure
    for flat field correction.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds
        modality: Modality identifier (e.g., "ppm")
        angles_str: String of angles like "(0,90,5,-5)"
        exposures_str: String of initial exposure times like "(1.5,100,50,50)".
                      These are used as starting points for adaptive exposure.
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance
        update_progress: Progress callback function
        use_per_angle_wb: Whether to apply per-angle white balance calibration
                         before acquiring each background image

    Returns:
        Dict[float, float]: Dictionary mapping angles to final exposure times (ms)
                           e.g., {90.0: 1.2, 5.0: 45.8, ...}
    """
    logger.info("=== SIMPLE BACKGROUND COLLECTION STARTED ===")

    try:
        # Parse angles and exposures from client
        # Use client's exposures as initial values for adaptive exposure
        angles, exposures = parse_angles_exposures(angles_str, exposures_str)
        logger.info(f"Collecting backgrounds for angles: {angles} using adaptive exposure")
        logger.info(f"Initial exposures from client: {exposures}")

        # Load microscope configuration
        if not Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        # Load main configuration file
        settings = config_manager.load_config_file(yaml_file_path)

        # Load and merge LOCI resources (same pattern as regular acquisition workflow)
        loci_rsc_file = str(
            Path(__file__).parent / "configurations" / "resources" / "resources_LOCI.yml"
        )
        try:
            loci_resources = config_manager.load_config_file(loci_rsc_file)
            settings.update(loci_resources)
            logger.info("Loaded and merged LOCI resources for background collection")
        except FileNotFoundError:
            logger.warning(
                f"LOCI resources file not found at {loci_rsc_file}, continuing without device mappings"
            )

        hardware.settings = settings

        # Re-initialize microscope-specific methods with updated settings
        # This is critical for PPM rotation to work correctly
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Load JAI white balance calibration if per-angle mode requested
        jai_calibration = None
        if use_per_angle_wb:
            # Get objective and detector from settings for calibration lookup
            objective = settings.get("objective")
            detector = settings.get("detector")
            logger.info(f"Looking up calibration for modality={modality}, objective={objective}, detector={detector}")

            jai_calibration = load_jai_calibration_from_imageprocessing(
                config_path=Path(yaml_file_path),
                per_angle=True,
                modality=modality,
                objective=objective,
                detector=detector,
                logger=logger,
            )
            if jai_calibration:
                logger.info("Per-angle white balance calibration loaded for background collection")
            else:
                logger.warning("Per-angle white balance requested but no calibration found")

        # Get current position for reference
        current_pos = hardware.get_current_position()
        logger.info(
            f"Acquiring backgrounds at position: X={current_pos.x:.1f}, Y={current_pos.y:.1f}, Z={current_pos.z:.1f}"
        )

        # Create output directory structure
        output_path = Path(output_folder_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving backgrounds to: {output_path}")

        # Initialize progress
        total_images = len(angles)
        update_progress(0, total_images)

        # Track final exposures and achieved intensities for each angle
        final_exposures = {}
        achieved_intensities = {}

        # Track reference images for birefringence pair matching
        # When acquiring paired polarization angles (+7/-7 or +5/-5), the negative
        # angle should minimize birefringence against the positive angle's IMAGE,
        # not just match intensity. This uses the same metric as biref calculation:
        # sum(|R_pos - R_neg| + |G_pos - G_neg| + |B_pos - B_neg|)
        biref_pair_references = {}  # Maps positive angle -> reference image

        # Acquire background for each angle
        for angle_idx, angle in enumerate(angles):
            logger.info(f"Acquiring background {angle_idx + 1}/{total_images} for angle {angle}")

            # Set rotation angle if supported
            if hasattr(hardware, "set_psg_ticks"):
                hardware.set_psg_ticks(
                    angle  # , is_sequence_start=True
                )  # Each background is independent
                logger.info(f"Set angle to {angle}")

            # Get initial exposure from client
            initial_exposure_ms = exposures[angle_idx] if angle_idx < len(exposures) else 100.0
            logger.info(f"Initial exposure from client: {initial_exposure_ms:.2f}ms")

            # Choose acquisition method based on white balance mode
            if jai_calibration and use_per_angle_wb:
                # Per-angle white balance mode: use per-channel adaptive exposure
                # This maintains the R:G:B ratio while scaling to target intensity
                angle_mapping = {90.0: "uncrossed", 0.0: "crossed", 7.0: "positive", -7.0: "negative"}
                angle_name = angle_mapping.get(angle)
                if not angle_name:
                    for a, name in angle_mapping.items():
                        if abs(a - angle) < 1.0:
                            angle_name = name
                            break

                if angle_name and "angles" in jai_calibration:
                    angle_cal = jai_calibration["angles"].get(angle_name)
                    if angle_cal and "exposures_ms" in angle_cal:
                        per_channel_exp = angle_cal["exposures_ms"]
                        target_intensity = get_target_intensity_for_background(modality, angle)
                        logger.info(f"Using per-channel adaptive exposure for angle {angle}")
                        logger.info(f"  Initial R={per_channel_exp.get('r', 50):.1f}ms, "
                                   f"G={per_channel_exp.get('g', 50):.1f}ms, B={per_channel_exp.get('b', 50):.1f}ms")
                        try:
                            image, final_per_channel = acquire_background_with_per_channel_adaptive(
                                hardware=hardware,
                                initial_exposures=per_channel_exp,
                                target_intensity=target_intensity,
                                tolerance=2.5,
                                max_iterations=10,
                                logger=logger,
                            )
                            actual_intensity = float(np.median(image))
                            # Store green channel as reference exposure for compatibility
                            final_exposures[angle] = final_per_channel.get('g', 100.0)
                            achieved_intensities[angle] = actual_intensity
                            logger.info(
                                f"Acquired background: shape={image.shape}, median={actual_intensity:.1f}"
                            )

                            # Store reference for biref pair matching
                            if angle > 0 and angle != 90:
                                biref_pair_references[angle] = image.copy()
                                logger.info(f"Stored +{angle} image as reference for birefringence pair matching")
                        except RuntimeError as e:
                            logger.error(f"Failed to acquire background at angle {angle}: {e}")
                            continue
                    else:
                        logger.warning(f"No calibration for angle {angle_name}, falling back to standard mode")
                        # Fall through to standard acquisition below
                        jai_calibration = None  # Disable for this angle
                else:
                    logger.warning(f"Unknown angle {angle}, falling back to standard mode")
                    jai_calibration = None  # Disable for this angle

            # Standard acquisition mode (no per-angle white balance or fallback)
            if not (jai_calibration and use_per_angle_wb):
                # For negative polarization angles, use biref-matching against positive angle
                paired_positive = abs(angle)  # e.g., -7 pairs with 7
                if angle < 0 and angle != -90:  # Negative polarization angle (not -90 brightfield)
                    if paired_positive in biref_pair_references:
                        # Use biref-matching: minimize sum of abs channel differences
                        reference_image = biref_pair_references[paired_positive]
                        logger.info(
                            f"Biref pair matching: minimizing biref metric against +{paired_positive} reference"
                        )
                        try:
                            image, final_exposure, achieved_biref = acquire_background_with_biref_matching(
                                hardware=hardware,
                                reference_image=reference_image,
                                tolerance=5.0,  # Target mean biref <= 5
                                initial_exposure_ms=initial_exposure_ms,
                                max_iterations=10,
                                logger=logger,
                            )
                            actual_intensity = float(np.median(image))
                            logger.info(
                                f"Acquired background: shape={image.shape}, "
                                f"achieved_biref={achieved_biref:.1f}, median={actual_intensity:.1f}, "
                                f"final_exposure={final_exposure:.1f}ms"
                            )
                            final_exposures[angle] = final_exposure
                            achieved_intensities[angle] = actual_intensity
                        except RuntimeError as e:
                            logger.error(f"Failed to acquire background at angle {angle}: {e}")
                            continue
                    else:
                        # Positive angle hasn't been acquired yet - fall back to intensity matching
                        logger.warning(
                            f"Biref pair matching: +{paired_positive} not yet acquired. "
                            f"For best results, acquire positive angles before negative. "
                            f"Falling back to intensity-based matching."
                        )
                        target_intensity = get_target_intensity_for_background(modality, angle)
                        try:
                            image, final_exposure = acquire_background_with_target_intensity(
                                hardware=hardware,
                                target_intensity=target_intensity,
                                tolerance=2.5,
                                initial_exposure_ms=initial_exposure_ms,
                                max_iterations=10,
                                logger=logger,
                            )
                            actual_intensity = float(np.median(image))
                            logger.info(
                                f"Acquired background: shape={image.shape}, "
                                f"median={actual_intensity:.1f}, "
                                f"final_exposure={final_exposure:.1f}ms"
                            )
                            final_exposures[angle] = final_exposure
                            achieved_intensities[angle] = actual_intensity
                        except RuntimeError as e:
                            logger.error(f"Failed to acquire background at angle {angle}: {e}")
                            continue
                else:
                    # Non-biref angles (0, 90, positive angles): use standard intensity matching
                    target_intensity = get_target_intensity_for_background(modality, angle)
                    logger.info(f"Target intensity: {target_intensity:.1f}")

                    try:
                        image, final_exposure = acquire_background_with_target_intensity(
                            hardware=hardware,
                            target_intensity=target_intensity,
                            tolerance=2.5,
                            initial_exposure_ms=initial_exposure_ms,
                            max_iterations=10,
                            logger=logger,
                        )
                        actual_intensity = float(np.median(image))
                        logger.info(
                            f"Acquired background: shape={image.shape}, median={actual_intensity:.1f}, "
                            f"final_exposure={final_exposure:.1f}ms"
                        )
                        final_exposures[angle] = final_exposure
                        achieved_intensities[angle] = actual_intensity

                        # Store reference image for positive polarization angles
                        # This will be used by paired negative angles for biref matching
                        if angle > 0 and angle != 90:  # Positive polarization angles (not brightfield)
                            biref_pair_references[angle] = image.copy()
                            logger.info(
                                f"Stored +{angle} image as reference for birefringence pair matching"
                            )
                    except RuntimeError as e:
                        logger.error(f"Failed to acquire background at angle {angle}: {e}")

            # Save background image using new format: angle.tif (not in subdirectory)
            background_path = output_path / f"{angle}.tif"
            TifWriterUtils.ome_writer(  # background -single
                filename=str(background_path),
                pixel_size_um=hardware.core.get_pixel_size_um(),
                data=image,
            )

            logger.info(f"Saved background for {angle} deg to {background_path}")

            # Update progress
            update_progress(angle_idx + 1, total_images)

        logger.info("=== SIMPLE BACKGROUND COLLECTION COMPLETE ===")
        logger.info(f"Successfully collected {len(angles)} background images")

        # Save background exposures and achieved intensities to imageprocessing YAML
        # This data becomes the source of truth for white balance target intensities
        try:
            save_background_exposures_to_yaml(
                config_path=Path(yaml_file_path),
                final_exposures=final_exposures,
                achieved_intensities=achieved_intensities,
                modality=modality,
                objective=settings.get("objective"),
                detector=settings.get("detector"),
            )
            logger.info("Background exposures saved to imageprocessing YAML")
        except Exception as e:
            logger.warning(f"Failed to save background exposures to YAML: {e}")
            # Non-fatal - continue returning the exposures

        # Return final exposures for metadata writing
        return final_exposures

    except Exception as e:
        logger.error(f"Simple background collection failed: {str(e)}", exc_info=True)
        raise


def background_acquisition_workflow(
    yaml_file_path: str,
    output_folder_path: str,
    modality: str,
    angles_str: str,
    exposures_str: Optional[str],
    hardware: PycromanagerHardware,
    config_manager,
    logger,
):
    """
    Acquire background images for flat-field correction.

    IMPORTANT: Position the microscope at a blank area before calling this function.
    The system will acquire images at the current position using adaptive exposure
    to reach target intensities.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds (will create modality subfolder)
        modality: Modality identifier (e.g., "PPM_20x")
        angles_str: String of angles like "(0,90,5,-5)"
        exposures_str: String of initial exposure times like "(1.5,100,50,50)".
                      These are used as starting points for adaptive exposure.
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance

    Returns:
        Tuple[str, Dict[float, float]]: (output_path, final_exposures)
            output_path: Path where backgrounds were saved
            final_exposures: Dictionary mapping angles to final exposure times (ms)
    """
    logger.info("=== BACKGROUND ACQUISITION WORKFLOW STARTED ===")
    logger.warning("Ensure microscope is positioned at a clean, blank area!")

    # Get and log current position for reference
    current_pos = hardware.get_current_position()
    logger.info(
        f"Acquiring backgrounds at position: X={current_pos.x:.1f}, "
        f"Y={current_pos.y:.1f}, Z={current_pos.z:.1f}"
    )

    try:
        # Parse angles and exposures from client
        # Use client's exposures as initial values for adaptive exposure
        angles, exposures = parse_angles_exposures(angles_str, exposures_str)
        logger.info(f"Initial exposures from client: {exposures}")

        # Load the microscope configuration
        if not Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        settings = config_manager.load_config_file(yaml_file_path)
        hardware.settings = settings

        # Re-initialize microscope-specific methods with updated settings
        # This is critical for PPM rotation to work correctly
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Create output directory structure with modality
        output_path = Path(output_folder_path) / "backgrounds" / modality
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving backgrounds to: {output_path}")

        # Track final exposures and achieved intensities for each angle
        final_exposures = {}
        achieved_intensities = {}

        # Track reference images for birefringence pair matching
        # Uses the same metric as biref calculation: sum(|R_pos - R_neg| + ...)
        biref_pair_references = {}  # Maps positive angle -> reference image

        # Acquire background for each angle
        for angle_idx, angle in enumerate(angles):
            # Create angle subdirectory
            angle_dir = output_path / str(angle)
            angle_dir.mkdir(exist_ok=True)

            # Set rotation angle if PPM
            if hasattr(hardware, "set_psg_ticks"):
                hardware.set_psg_ticks(
                    angle  # , is_sequence_start=True
                )  # Each background is independent
                logger.info(f"Set angle to {angle}")

            # Use exposure from client as initial value for adaptive exposure
            initial_exposure_ms = exposures[angle_idx] if angle_idx < len(exposures) else 100.0
            logger.info(f"Initial exposure from client: {initial_exposure_ms:.2f}ms")

            # For negative polarization angles, use biref-matching against positive angle
            paired_positive = abs(angle)  # e.g., -7 pairs with 7
            if angle < 0 and angle != -90:  # Negative polarization angle
                if paired_positive in biref_pair_references:
                    # Use biref-matching: minimize sum of abs channel differences
                    reference_image = biref_pair_references[paired_positive]
                    logger.info(
                        f"Biref pair matching: minimizing biref metric against +{paired_positive} reference"
                    )
                    try:
                        image, final_exposure, achieved_biref = acquire_background_with_biref_matching(
                            hardware=hardware,
                            reference_image=reference_image,
                            tolerance=5.0,
                            initial_exposure_ms=initial_exposure_ms,
                            max_iterations=10,
                            logger=logger,
                        )
                        actual_intensity = float(np.median(image))
                        logger.info(
                            f"Acquired background: achieved_biref={achieved_biref:.1f}, "
                            f"median={actual_intensity:.1f}, final_exposure={final_exposure:.1f}ms"
                        )
                        final_exposures[angle] = final_exposure
                        achieved_intensities[angle] = actual_intensity
                    except RuntimeError as e:
                        logger.error(f"Failed to acquire background at angle {angle}: {e}")
                        continue
                else:
                    # Positive angle not yet acquired - fall back to intensity matching
                    logger.warning(
                        f"Biref pair matching: +{paired_positive} not yet acquired. "
                        f"Falling back to intensity-based matching."
                    )
                    target_intensity = get_target_intensity_for_background(modality, angle)
                    try:
                        image, final_exposure = acquire_background_with_target_intensity(
                            hardware=hardware,
                            target_intensity=target_intensity,
                            tolerance=2.5,
                            initial_exposure_ms=initial_exposure_ms,
                            max_iterations=10,
                            logger=logger,
                        )
                        actual_intensity = float(np.median(image))
                        logger.info(
                            f"Acquired background: median={actual_intensity:.1f}, "
                            f"final_exposure={final_exposure:.1f}ms"
                        )
                        final_exposures[angle] = final_exposure
                        achieved_intensities[angle] = actual_intensity
                    except RuntimeError as e:
                        logger.error(f"Failed to acquire background at angle {angle}: {e}")
                        continue
            else:
                # Non-biref angles: use standard intensity matching
                target_intensity = get_target_intensity_for_background(modality, angle)
                logger.info(f"Target intensity: {target_intensity:.1f}")

                try:
                    image, final_exposure = acquire_background_with_target_intensity(
                        hardware=hardware,
                        target_intensity=target_intensity,
                        tolerance=2.5,
                        initial_exposure_ms=initial_exposure_ms,
                        max_iterations=10,
                        logger=logger,
                    )
                    actual_intensity = float(np.median(image))
                    logger.info(
                        f"Acquired background: median={actual_intensity:.1f}, "
                        f"final_exposure={final_exposure:.1f}ms"
                    )
                    final_exposures[angle] = final_exposure
                    achieved_intensities[angle] = actual_intensity

                    # Store reference for positive polarization angles
                    if angle > 0 and angle != 90:
                        biref_pair_references[angle] = image.copy()
                        logger.info(
                            f"Stored +{angle} image as reference for birefringence pair matching"
                        )
                except RuntimeError as e:
                    logger.error(f"Failed to acquire background at angle {angle}: {e}")
                    continue

            # Save background image
            background_path = angle_dir / "background.tif"
            TifWriterUtils.ome_writer(  # background 2 with bkg-workflow
                filename=str(background_path),
                pixel_size_um=hardware.core.get_pixel_size_um(),
                data=image,
            )

            logger.info(f"Saved background for {angle} deg to {background_path}")

        logger.info("=== BACKGROUND ACQUISITION COMPLETE ===")

        # Save background exposures and achieved intensities to imageprocessing YAML
        try:
            save_background_exposures_to_yaml(
                config_path=Path(yaml_file_path),
                final_exposures=final_exposures,
                achieved_intensities=achieved_intensities,
                modality=modality,
                objective=settings.get("objective"),
                detector=settings.get("detector"),
            )
            logger.info("Background exposures saved to imageprocessing YAML")
        except Exception as e:
            logger.warning(f"Failed to save background exposures to YAML: {e}")

        return str(output_path), final_exposures

    except Exception as e:
        logger.error(f"Background acquisition failed: {str(e)}", exc_info=True)
        raise


def polarizer_calibration_workflow(
    yaml_file_path: str,
    output_folder_path: str,
    start_angle: float,
    end_angle: float,
    step_size: float,
    exposure_ms: float,
    hardware: PycromanagerHardware,
    config_manager,
    logger,
) -> str:
    """
    Calibrate PPM polarizer rotation stage to find crossed polarizer positions.

    IMPORTANT: Position microscope at uniform, bright background before calling.
    This workflow sweeps the rotation stage through angles, measures intensity,
    and determines optimal crossed polarizer positions for config_PPM.yml.

    Args:
        yaml_file_path: Path to microscope configuration YAML
        output_folder_path: Base folder for backgrounds (will write report at top level)
        start_angle: Starting angle for sweep (degrees)
        end_angle: Ending angle for sweep (degrees)
        step_size: Step size for sweep (degrees)
        exposure_ms: Exposure time (milliseconds)
        hardware: Microscope hardware interface
        config_manager: Configuration manager
        logger: Logger instance

    Returns:
        str: Path to the calibration report text file
    """
    logger.info("=== POLARIZER CALIBRATION WORKFLOW STARTED ===")
    logger.warning("Ensure microscope is positioned at a uniform, bright background!")

    # Get and log current position for reference
    current_pos = hardware.get_current_position()
    logger.info(
        f"Running calibration at position: X={current_pos.x:.1f}, "
        f"Y={current_pos.y:.1f}, Z={current_pos.z:.1f}"
    )

    try:
        # Load the microscope configuration
        if not Path(yaml_file_path).exists():
            raise FileNotFoundError(f"YAML file {yaml_file_path} does not exist")

        settings = config_manager.load_config_file(yaml_file_path)

        # Load and merge LOCI resources (required for rotation stage device lookup)
        loci_rsc_file = str(
            Path(yaml_file_path).parent / "resources" / "resources_LOCI.yml"
        )
        if Path(loci_rsc_file).exists():
            loci_resources = config_manager.load_config_file(loci_rsc_file)
            settings.update(loci_resources)
            logger.info("Loaded and merged LOCI resources")
        else:
            logger.warning(f"LOCI resources file not found: {loci_rsc_file}")

        hardware.settings = settings

        # Re-initialize microscope-specific methods
        if hasattr(hardware, "_initialize_microscope_methods"):
            hardware._initialize_microscope_methods()
            logger.info("Re-initialized hardware methods with updated settings")

        # Verify PPM is available
        if not hasattr(hardware, "set_psg_ticks"):
            raise RuntimeError(
                "PPM rotation stage methods not available. "
                "Check ppm_optics setting in configuration."
            )

        # Import the calibration utility
        from ppm_library.ppm.calibration import PolarizerCalibrationUtils

        # Run two-stage calibration to determine exact hardware offset
        logger.info(
            f"Starting two-stage hardware calibration: "
            f"Coarse: 0-360 deg in {step_size} deg steps, "
            f"Fine: +/-{step_size} deg in 0.1 deg steps"
        )
        logger.info(f"Exposure: {exposure_ms} ms")

        result = PolarizerCalibrationUtils.calibrate_hardware_offset_with_stability_check(
            hardware=hardware,
            num_runs=3,  # Run 3 times to validate stability
            stability_threshold_counts=50.0,  # Warn if variation > 0.05 deg
            coarse_range_deg=360.0,  # Full rotation
            coarse_step_deg=step_size,  # Use user-specified step size for coarse
            fine_range_deg=10.0,  # Increased from 5.0 for better safety margin
            fine_step_deg=0.1,  # Fine step for precise positioning
            exposure_ms=exposure_ms,
            channel=1,  # Green channel
            logger_instance=logger,
        )

        # Write calibration report
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"polarizer_calibration_{timestamp}.txt"
        report_path = Path(output_folder_path) / report_filename

        # Ensure output directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            # Get the first run's results for displaying exact positions
            primary_result = result.get('all_runs', [result])[0] if 'all_runs' in result else result
            hw_per_deg = result.get('hw_per_deg', primary_result.get('hw_per_deg', 1000.0))

            f.write("=" * 80 + "\n")
            f.write("PPM POLARIZER CALIBRATION RESULTS\n")
            f.write("=" * 80 + "\n\n")

            # ===== RESULTS FIRST - THE KEY VALUES =====
            f.write("CROSSED POLARIZER POSITIONS (use these values in config_PPM.yml):\n\n")

            f.write(f"  >>> ppm_pizstage_offset: {result['recommended_offset']:.1f} <<<\n\n")

            f.write(f"  Found {len(primary_result['exact_minima'])} crossed polarizer positions:\n\n")

            for i, (hw_pos, opt_angle) in enumerate(
                zip(primary_result["exact_minima"], primary_result["optical_angles"])
            ):
                # Find corresponding intensity
                intensity_str = ""
                for fine_result in primary_result["fine_results"]:
                    if abs(fine_result["exact_position"] - hw_pos) < 0.1:
                        intensity_str = f", intensity={fine_result['exact_intensity']:.1f}"
                        break
                f.write(f"    Position {i+1}: {hw_pos:.1f} counts ({opt_angle:.1f} deg optical){intensity_str}\n")

            f.write("\n")

            # Separation check
            if len(primary_result["exact_minima"]) >= 2:
                separation = abs(primary_result["exact_minima"][1] - primary_result["exact_minima"][0])
                separation_deg = separation / hw_per_deg
                f.write(f"  Separation: {separation_deg:.1f} deg (expected: 180.0 deg)\n")

            # Stability summary
            if 'offset_std' in result:
                stability_deg = result['offset_range'] / hw_per_deg
                f.write(f"  Stability: {'PASS' if result['is_stable'] else 'FAIL'} ")
                f.write(f"(variation: {stability_deg:.4f} deg across {len(result.get('all_runs', [result]))} runs)\n")

            f.write("\n")

            # ===== CONFIG RECOMMENDATIONS =====
            f.write("=" * 80 + "\n")
            f.write("CONFIG_PPM.YML UPDATE\n")
            f.write("=" * 80 + "\n\n")

            f.write("Update your config_PPM.yml with:\n\n")
            f.write(f"ppm_pizstage_offset: {result['recommended_offset']:.1f}\n\n")

            f.write("rotation_angles:\n")
            f.write("  - name: 'crossed'\n")
            f.write(f"    tick: 0   # Reference position (hardware: {result['recommended_offset']:.1f})\n")

            if len(primary_result["exact_minima"]) >= 2:
                other_angle = primary_result["optical_angles"][1]
                other_hw = primary_result["exact_minima"][1]
                f.write(f"    # OR tick: {other_angle:.0f}   # Alternate crossed (hardware: {other_hw:.1f})\n")

            f.write("  - name: 'uncrossed'\n")
            f.write("    tick: 90  # 90 deg from crossed (perpendicular)\n\n")

            # ===== CALIBRATION DETAILS =====
            f.write("=" * 80 + "\n")
            f.write("CALIBRATION DETAILS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {yaml_file_path}\n")
            f.write(f"Position: X={current_pos.x:.1f}, Y={current_pos.y:.1f}, Z={current_pos.z:.1f}\n")
            f.write(f"Rotation Device: {result['rotation_device']}\n\n")

            f.write("Parameters:\n")
            f.write(f"  Coarse: 0-360 deg in {step_size} deg steps\n")
            f.write(f"  Fine: +/-10 deg in 0.1 deg steps around each minimum\n")
            f.write(f"  Exposure: {exposure_ms} ms, Channel: Green\n")
            f.write(f"  Stability runs: {len(result.get('all_runs', [result]))}\n\n")

            coarse_intensities = primary_result["coarse_intensities"]
            f.write("Intensity Statistics:\n")
            f.write(f"  Range: {coarse_intensities.min():.1f} to {coarse_intensities.max():.1f}\n")
            f.write(f"  Dynamic Range: {coarse_intensities.max() / coarse_intensities.min():.1f}x\n\n")

            # ===== STABILITY CHECK DETAILS =====
            if 'offset_std' in result:
                f.write("=" * 80 + "\n")
                f.write("STABILITY CHECK\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Runs: {len(result.get('all_runs', [result]))}\n")
                f.write(f"Raw offsets: {result['individual_offsets']}\n")
                f.write(f"Normalized (mod 180 deg): {result['normalized_offsets']}\n")
                f.write(f"  Note: Crossed polarizers repeat every 180 deg, so positions\n")
                f.write(f"        differing by 180 deg are equivalent.\n\n")
                f.write(f"Std deviation: {result['offset_std']:.2f} counts ({result['offset_std']/hw_per_deg:.4f} deg)\n")
                f.write(f"Range: {result['offset_range']:.1f} counts ({result['offset_range']/hw_per_deg:.4f} deg)\n")
                f.write(f"Threshold: 50.0 counts (0.05 deg)\n")
                f.write(f"Result: {'PASS - Stable' if result['is_stable'] else 'FAIL - Unstable'}\n")
                if not result['is_stable']:
                    f.write(f"\nWARNING: Check polarizer/analyzer mounts for mechanical issues.\n")
                f.write("\n")

            # ===== RAW DATA =====
            f.write("=" * 80 + "\n")
            f.write("RAW DATA - COARSE SWEEP (RUN 1)\n")
            f.write("=" * 80 + "\n\n")

            f.write("Hardware Position (counts), Intensity\n")
            for hw_pos, intensity in zip(
                primary_result["coarse_hardware_positions"], primary_result["coarse_intensities"]
            ):
                f.write(f"{hw_pos:.1f}, {intensity:.2f}\n")

            # Write raw data for all runs if stability check was performed
            all_runs = result.get('all_runs', [primary_result])
            for run_idx, run_result in enumerate(all_runs, 1):
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"RAW DATA - FINE SWEEPS (RUN {run_idx})\n")
                f.write("=" * 80 + "\n\n")

                for i, fine_result in enumerate(run_result["fine_results"]):
                    f.write(
                        f"\nFine Sweep {i+1} (centered on {fine_result['approximate_position']:.1f}):\n"
                    )
                    f.write("Hardware Position (counts), Intensity\n")
                    for hw_pos, intensity in zip(
                        fine_result["fine_hw_positions"], fine_result["fine_intensities"]
                    ):
                        f.write(f"{hw_pos:.1f}, {intensity:.2f}\n")

        logger.info(f"Calibration report saved to: {report_path}")
        logger.info("=== POLARIZER CALIBRATION WORKFLOW COMPLETE ===")

        return str(report_path)

    except Exception as e:
        logger.error(f"Polarizer calibration failed: {str(e)}", exc_info=True)
        raise
