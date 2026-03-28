"""
Z-stack and time-lapse acquisition workflows.

Basic single-tile implementations designed for future multi-tile expansion.
Both workflows save individual frames as TIFF files with metadata.

Z-stack: Acquires images at multiple Z positions around the current focus.
Time-lapse: Acquires images at the current position at regular intervals.

Future expansion notes:
- Multi-tile Z-stack: iterate XY grid, at each position run Z-stack
- Multi-tile time-lapse: iterate XY grid at each time point
- Combined: XY grid x Z-stack x time points (requires careful ordering)
- Stitching: each Z plane or time point would need separate stitching
"""

import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


def acquire_z_stack(
    hardware,
    output_folder: str,
    z_start: float,
    z_end: float,
    z_step: float,
    modality: str = "brightfield",
    angles_str: str = "(0)",
    config_manager=None,
    wb_mode: str = "off",
    objective: str = None,
    detector: str = None,
    yaml_file_path: str = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    """
    Acquire a Z-stack at the current XY position.

    Moves Z through the specified range, snapping an image at each plane.
    If PPM modality, acquires all angles at each Z position.

    Args:
        hardware: PycromanagerHardware instance
        output_folder: Directory to save images
        z_start: Starting Z position in um (absolute stage coordinate)
        z_end: Ending Z position in um (absolute stage coordinate)
        z_step: Step size in um between Z planes
        modality: Imaging modality ("brightfield", "ppm", etc.)
        angles_str: Rotation angles for PPM (ignored for brightfield)
        config_manager: Config manager for WB calibration lookup
        wb_mode: White balance mode
        objective: Objective ID for calibration
        detector: Detector ID for calibration
        yaml_file_path: Path to config YAML
        progress_callback: Called with (current_plane, total_planes, message)

    Returns:
        Dict with results: n_planes, z_positions, output_folder, files
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate Z positions
    if z_step <= 0:
        raise ValueError(f"z_step must be positive, got {z_step}")

    z_positions = []
    z = z_start
    while z <= z_end + z_step * 0.01:  # small epsilon for float comparison
        z_positions.append(round(z, 3))
        z += z_step

    n_planes = len(z_positions)
    if n_planes == 0:
        raise ValueError(f"No Z positions in range [{z_start}, {z_end}] with step {z_step}")

    logger.info(f"=== Z-STACK ACQUISITION ===")
    logger.info(f"Z range: {z_start} to {z_end} um, step={z_step} um, {n_planes} planes")
    logger.info(f"Output: {output_path}")

    # Parse angles for PPM
    from microscope_command_server.acquisition.workflow import parse_angles_exposures
    angles = [0.0]
    if modality.lower().startswith("ppm"):
        angles, _ = parse_angles_exposures(angles_str, None)
        logger.info(f"PPM mode: {len(angles)} angles per Z plane")

    # Load WB calibration if needed
    jai_calibration = None
    if wb_mode not in ("off", None) and yaml_file_path:
        try:
            from microscope_command_server.acquisition.workflow import (
                load_jai_calibration_from_imageprocessing,
            )
            jai_calibration = load_jai_calibration_from_imageprocessing(
                config_path=Path(yaml_file_path),
                per_angle=True,
                modality=modality,
                objective=objective,
                detector=detector,
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Could not load WB calibration: {e}")

    saved_files = []
    start_time = time.time()

    for plane_idx, z_pos in enumerate(z_positions):
        if progress_callback:
            progress_callback(plane_idx, n_planes, f"Z={z_pos:.1f} um")

        # Move Z
        hardware.core.set_position(z_pos)
        hardware.core.wait_for_device(hardware.core.get_focus_device())
        actual_z = hardware.core.get_position()
        logger.info(f"Plane {plane_idx + 1}/{n_planes}: Z={actual_z:.2f} um (target={z_pos:.2f})")

        # Acquire at each angle
        for angle in angles:
            # Rotate if PPM
            if len(angles) > 1 and hasattr(hardware, "set_psg_ticks"):
                hardware.set_psg_ticks(angle)

            # Apply WB if calibration available
            if jai_calibration:
                _apply_wb_for_snap(hardware, jai_calibration, angle, modality, logger)

            # Snap
            image, metadata = hardware.snap_image()
            if image is None:
                logger.error(f"Failed to snap at Z={z_pos}, angle={angle}")
                continue

            # Save with descriptive filename
            angle_suffix = f"_angle{angle:.0f}" if len(angles) > 1 else ""
            filename = f"z{plane_idx:04d}_Z{z_pos:.1f}{angle_suffix}.tif"
            filepath = output_path / filename

            _save_image(image, filepath, {
                "z_position_um": actual_z,
                "z_index": plane_idx,
                "angle": angle,
                "modality": modality,
            })
            saved_files.append(str(filepath))

    elapsed = time.time() - start_time
    logger.info(f"=== Z-STACK COMPLETE: {n_planes} planes, {len(saved_files)} images, "
                f"{elapsed:.1f}s ===")

    if progress_callback:
        progress_callback(n_planes, n_planes, "Complete")

    return {
        "n_planes": n_planes,
        "z_positions": z_positions,
        "output_folder": str(output_path),
        "files": saved_files,
        "elapsed_seconds": elapsed,
    }


def acquire_time_lapse(
    hardware,
    output_folder: str,
    n_timepoints: int,
    interval_seconds: float,
    modality: str = "brightfield",
    angles_str: str = "(0)",
    config_manager=None,
    wb_mode: str = "off",
    objective: str = None,
    detector: str = None,
    yaml_file_path: str = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Acquire a time-lapse at the current position.

    Snaps images at regular intervals. If PPM modality, acquires all
    angles at each time point.

    Args:
        hardware: PycromanagerHardware instance
        output_folder: Directory to save images
        n_timepoints: Number of time points to acquire
        interval_seconds: Seconds between time points (0 = as fast as possible)
        modality: Imaging modality
        angles_str: Rotation angles for PPM
        config_manager: Config manager
        wb_mode: White balance mode
        objective: Objective ID
        detector: Detector ID
        yaml_file_path: Path to config YAML
        progress_callback: Called with (current_tp, total_tp, message)
        cancel_check: Returns True to abort acquisition

    Returns:
        Dict with results: n_timepoints, output_folder, files, elapsed
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if n_timepoints <= 0:
        raise ValueError(f"n_timepoints must be positive, got {n_timepoints}")

    logger.info(f"=== TIME-LAPSE ACQUISITION ===")
    logger.info(f"Timepoints: {n_timepoints}, interval: {interval_seconds}s")
    logger.info(f"Output: {output_path}")

    # Parse angles for PPM
    from microscope_command_server.acquisition.workflow import parse_angles_exposures
    angles = [0.0]
    if modality.lower().startswith("ppm"):
        angles, _ = parse_angles_exposures(angles_str, None)
        logger.info(f"PPM mode: {len(angles)} angles per timepoint")

    # Load WB calibration if needed
    jai_calibration = None
    if wb_mode not in ("off", None) and yaml_file_path:
        try:
            from microscope_command_server.acquisition.workflow import (
                load_jai_calibration_from_imageprocessing,
            )
            jai_calibration = load_jai_calibration_from_imageprocessing(
                config_path=Path(yaml_file_path),
                per_angle=True,
                modality=modality,
                objective=objective,
                detector=detector,
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Could not load WB calibration: {e}")

    saved_files = []
    start_time = time.time()
    t0 = time.time()

    for tp_idx in range(n_timepoints):
        # Check for cancellation
        if cancel_check and cancel_check():
            logger.info(f"Time-lapse cancelled at timepoint {tp_idx + 1}/{n_timepoints}")
            break

        tp_start = time.time()
        elapsed_since_start = tp_start - t0

        if progress_callback:
            progress_callback(tp_idx, n_timepoints, f"T={elapsed_since_start:.1f}s")

        logger.info(f"Timepoint {tp_idx + 1}/{n_timepoints} at T={elapsed_since_start:.1f}s")

        # Acquire at each angle
        for angle in angles:
            if len(angles) > 1 and hasattr(hardware, "set_psg_ticks"):
                hardware.set_psg_ticks(angle)

            if jai_calibration:
                _apply_wb_for_snap(hardware, jai_calibration, angle, modality, logger)

            image, metadata = hardware.snap_image()
            if image is None:
                logger.error(f"Failed to snap at T={tp_idx}, angle={angle}")
                continue

            angle_suffix = f"_angle{angle:.0f}" if len(angles) > 1 else ""
            filename = f"t{tp_idx:05d}_T{elapsed_since_start:.1f}s{angle_suffix}.tif"
            filepath = output_path / filename

            _save_image(image, filepath, {
                "timepoint": tp_idx,
                "elapsed_seconds": elapsed_since_start,
                "angle": angle,
                "modality": modality,
            })
            saved_files.append(str(filepath))

        # Wait for next timepoint (accounting for acquisition time)
        if tp_idx < n_timepoints - 1 and interval_seconds > 0:
            acquisition_time = time.time() - tp_start
            wait_time = max(0, interval_seconds - acquisition_time)
            if wait_time > 0:
                # Sleep in small increments to allow cancellation
                sleep_end = time.time() + wait_time
                while time.time() < sleep_end:
                    if cancel_check and cancel_check():
                        break
                    time.sleep(min(0.5, sleep_end - time.time()))

    elapsed = time.time() - start_time
    actual_tp = len(set(f.split("_T")[0] for f in [Path(f).stem for f in saved_files]))
    logger.info(f"=== TIME-LAPSE COMPLETE: {actual_tp} timepoints, "
                f"{len(saved_files)} images, {elapsed:.1f}s ===")

    if progress_callback:
        progress_callback(n_timepoints, n_timepoints, "Complete")

    return {
        "n_timepoints": actual_tp,
        "output_folder": str(output_path),
        "files": saved_files,
        "elapsed_seconds": elapsed,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _apply_wb_for_snap(hardware, jai_calibration, angle, modality, logger):
    """Apply white balance settings for a given angle before snapping."""
    try:
        from microscope_command_server.acquisition.workflow import (
            apply_jai_calibration_for_angle,
            angle_to_name,
        )
        angle_name = angle_to_name(angle, modality=modality)
        apply_jai_calibration_for_angle(
            hardware=hardware,
            jai_calibration=jai_calibration,
            angle=angle,
            modality=modality,
            logger=logger,
        )
    except Exception as e:
        logger.debug(f"Could not apply WB for angle {angle}: {e}")


def _save_image(image: np.ndarray, filepath: Path, metadata: dict):
    """Save an image as TIFF with basic metadata."""
    try:
        import tifffile
        tifffile.imwrite(str(filepath), image, metadata=metadata)
    except ImportError:
        # Fallback: save as raw numpy
        from PIL import Image as PILImage
        if image.ndim == 3 and image.shape[2] == 3:
            PILImage.fromarray(image).save(str(filepath))
        elif image.ndim == 2:
            PILImage.fromarray(image).save(str(filepath))
        else:
            np.save(str(filepath).replace(".tif", ".npy"), image)
    logger.debug(f"Saved: {filepath}")
