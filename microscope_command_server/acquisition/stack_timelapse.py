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
    projection: str = "none",
    background_correction_enabled: bool = False,
    background_folder: Optional[str] = None,
    background_correction_method: str = "divide",
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
        projection: Z-projection operator ("none","max","min","sum","mean","std")
        background_correction_enabled: When true, apply per-angle flat-field
            correction to every snap using images from background_folder.
        background_folder: Path to a directory containing per-angle background
            images. Layout matches the bounded-acquisition workflow's loader
            (BackgroundCorrectionUtils.load_background_images) -- typically
            {folder}/{angle}.tif or {folder}/{angle}/background.tif.
        background_correction_method: "divide" (flat-field) or "subtract".

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

    # Load per-angle background images (parity with main workflow).
    background_images, background_scaling_factors = _load_background_images_for_angles(
        enabled=background_correction_enabled,
        background_folder=background_folder,
        angles=angles,
    )
    bg_active = bool(background_images)

    saved_files = []
    # Buffer frames per angle so we can emit a single multi-plane OME-TIFF per
    # angle once the Z loop completes. Keyed by angle -> list of (z_index,
    # actual_z, image) in acquisition order.
    angle_frames: Dict[float, list] = {angle: [] for angle in angles}
    start_time = time.time()

    # Capture starting XYZ so we can restore the stage after the acquisition.
    # Users expect the stage to return to the original position (not sit at
    # the last Z plane of the stack).
    from microscope_control.hardware import Position
    try:
        start_position = hardware.get_current_position()
        logger.info(
            f"Z-stack start position: X={start_position.x:.2f}, "
            f"Y={start_position.y:.2f}, Z={start_position.z:.2f} um"
        )
    except Exception as e:
        logger.warning(f"Could not capture start position for restore: {e}")
        start_position = None

    try:
        for plane_idx, z_pos in enumerate(z_positions):
            if progress_callback:
                progress_callback(plane_idx, n_planes, f"Z={z_pos:.1f} um")

            # Move Z
            hardware.move_to_position(Position(z=z_pos))
            actual_z = hardware.get_z_position()
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

                # Apply per-angle flat-field correction when enabled. Mirrors
                # the main workflow's _acquire_tile_angles BG path so PPM
                # output from the single-point dialog matches what BoundedAcq
                # would have produced for the same modality/folder.
                if bg_active and angle in background_images:
                    image = _apply_bg_correction(
                        image=image,
                        bg_image=background_images[angle],
                        scaling=background_scaling_factors.get(angle),
                        method=background_correction_method,
                        angle=angle,
                    )

                angle_frames[angle].append((plane_idx, actual_z, image))
    finally:
        # Always restore the stage to the starting position, even if the
        # acquisition raised. Users expect XYZ to return where it started.
        if start_position is not None:
            try:
                hardware.move_to_position(start_position)
                logger.info(
                    f"Restored stage to start: X={start_position.x:.2f}, "
                    f"Y={start_position.y:.2f}, Z={start_position.z:.2f} um"
                )
            except Exception as e:
                logger.error(f"Failed to restore stage position: {e}")

    # Emit one OME-TIFF per angle using StackWriter. Non-PPM runs with a
    # single angle=0 and produces one zstack.ome.tiff with a real OME Z
    # dimension. PPM angles are written as independent sibling files
    # (zstack_angleNN.ome.tiff), per project decision that PPM angles stay
    # as separate files rather than combined channels.
    from microscope_imageprocessing.io.ome_writer import StackWriter

    pixel_size_um = _resolve_pixel_size_um(config_manager, modality, objective, detector)

    for angle, frames in angle_frames.items():
        if not frames:
            logger.warning(f"No frames captured for angle {angle}; skipping file write")
            continue
        # All frames share shape + dtype (same camera, same acquisition)
        first_img = frames[0][2]
        size_y, size_x = first_img.shape[:2]
        size_c = 3 if (first_img.ndim == 3 and first_img.shape[2] == 3) else 1
        is_rgb = size_c == 3

        angle_suffix = f"_angle{angle:.0f}" if len(angles) > 1 else ""
        out_file = output_path / f"zstack{angle_suffix}.ome.tiff"

        channel_names = ["RGB"] if is_rgb else [f"{modality}{angle_suffix}"]

        with StackWriter(
            output_path=out_file,
            size_t=1,
            size_z=len(frames),
            size_c=1,
            size_y=size_y,
            size_x=size_x,
            dtype=first_img.dtype,
            pixel_size_um=pixel_size_um,
            z_step_um=z_step,
            channel_names=channel_names,
            granularity="single",
            photometric="rgb" if is_rgb else "minisblack",
        ) as writer:
            for plane_idx, actual_z, image in frames:
                writer.write_frame(
                    image,
                    t=0,
                    z=plane_idx,
                    c=0,
                    plane_metadata={
                        "PositionZ": float(actual_z),
                        "angle": float(angle),
                    },
                )
        saved_files.append(str(out_file))
        logger.info(f"Wrote Z-stack OME-TIFF: {out_file} (SizeZ={len(frames)})")

    # Apply projection if requested -- one extra 2D file per angle alongside
    # the full Z-stack OME-TIFF.
    projected_file = None
    projection_name = projection
    if projection_name and projection_name != "none" and n_planes > 1:
        try:
            from microscope_command_server.acquisition.projections import get_projection
            projection_fn = get_projection(projection_name)

            for angle, frames in angle_frames.items():
                if not frames:
                    continue
                plane_images = [f[2] for f in frames]
                projected = projection_fn(plane_images)
                first_img = plane_images[0]
                size_y, size_x = first_img.shape[:2]
                is_rgb = first_img.ndim == 3 and first_img.shape[2] == 3

                angle_suffix = f"_angle{angle:.0f}" if len(angles) > 1 else ""
                proj_file = output_path / f"zstack_{projection_name}{angle_suffix}.ome.tiff"

                with StackWriter(
                    output_path=proj_file,
                    size_t=1,
                    size_z=1,
                    size_c=1,
                    size_y=size_y,
                    size_x=size_x,
                    dtype=projected.dtype,
                    pixel_size_um=pixel_size_um,
                    channel_names=["RGB" if is_rgb else f"{projection_name}{angle_suffix}"],
                    granularity="single",
                    photometric="rgb" if is_rgb else "minisblack",
                ) as writer:
                    writer.write_frame(projected, t=0, z=0, c=0)

                projected_file = str(proj_file)
                logger.info(f"Z-stack projection ({projection_name}): {proj_file}")
        except Exception as e:
            logger.warning(f"Z-stack projection failed: {e}")

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
        "projected_file": projected_file,
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
    background_correction_enabled: bool = False,
    background_folder: Optional[str] = None,
    background_correction_method: str = "divide",
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

    # Load per-angle background images (parity with main workflow).
    background_images, background_scaling_factors = _load_background_images_for_angles(
        enabled=background_correction_enabled,
        background_folder=background_folder,
        angles=angles,
    )
    bg_active = bool(background_images)

    saved_files = []
    # Buffer per-angle frames so we can emit a single multi-page OME-TIFF
    # per angle once the timepoint loop completes. Keyed by angle -> list of
    # (tp_idx, elapsed_since_start, image) in acquisition order.
    angle_frames: Dict[float, list] = {angle: [] for angle in angles}
    start_time = time.time()
    t0 = time.time()
    actual_tp = 0  # Number of timepoints actually acquired (for cancel case)

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

            # Per-angle flat-field correction (parity with main workflow).
            if bg_active and angle in background_images:
                image = _apply_bg_correction(
                    image=image,
                    bg_image=background_images[angle],
                    scaling=background_scaling_factors.get(angle),
                    method=background_correction_method,
                    angle=angle,
                )

            angle_frames[angle].append((tp_idx, elapsed_since_start, image))

        actual_tp = tp_idx + 1

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

    # Emit one multi-page OME-TIFF per angle using StackWriter. Non-PPM runs
    # with a single angle=0 and produces one timelapse.ome.tiff with a real
    # OME T dimension, TimeIncrement metadata, per-frame DeltaT. PPM angles
    # are written as independent sibling files (timelapse_angleNN.ome.tiff)
    # to mirror the Z-stack project decision (PPM angles stay separate
    # rather than combining into OME channels).
    from microscope_imageprocessing.io.ome_writer import StackWriter

    pixel_size_um = _resolve_pixel_size_um(config_manager, modality, objective, detector)

    for angle, frames in angle_frames.items():
        if not frames:
            logger.warning(f"No frames captured for angle {angle}; skipping file write")
            continue
        first_img = frames[0][2]
        size_y, size_x = first_img.shape[:2]
        is_rgb = first_img.ndim == 3 and first_img.shape[2] == 3

        angle_suffix = f"_angle{angle:.0f}" if len(angles) > 1 else ""
        out_file = output_path / f"timelapse{angle_suffix}.ome.tiff"

        channel_names = ["RGB"] if is_rgb else [f"{modality}{angle_suffix}"]

        with StackWriter(
            output_path=out_file,
            size_t=len(frames),
            size_z=1,
            size_c=1,
            size_y=size_y,
            size_x=size_x,
            dtype=first_img.dtype,
            pixel_size_um=pixel_size_um,
            time_increment_s=interval_seconds if interval_seconds > 0 else None,
            channel_names=channel_names,
            granularity="single",
            photometric="rgb" if is_rgb else "minisblack",
        ) as writer:
            for t_local, (tp_idx, elapsed_s, image) in enumerate(frames):
                writer.write_frame(
                    image,
                    t=t_local,
                    z=0,
                    c=0,
                    plane_metadata={
                        "delta_t_s": float(elapsed_s),
                        "angle": float(angle),
                    },
                )
        saved_files.append(str(out_file))
        logger.info(f"Wrote time-lapse OME-TIFF: {out_file} (SizeT={len(frames)})")

    elapsed = time.time() - start_time
    logger.info(f"=== TIME-LAPSE COMPLETE: {actual_tp} timepoints, "
                f"{len(saved_files)} files, {elapsed:.1f}s ===")

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

def _load_background_images_for_angles(
    enabled: bool,
    background_folder: Optional[str],
    angles: List[float],
):
    """Load per-angle background images for flat-field correction.

    Mirrors the main workflow's loader path (BackgroundCorrectionUtils.
    load_background_images) so the single-point dialog produces output
    consistent with BoundedAcq for the same modality + folder.

    Returns a (background_images, background_scaling_factors) tuple. Both
    are dicts keyed by angle (float). Returns ({}, {}) when correction is
    disabled, the folder is missing, or no images load successfully.
    """
    if not enabled or not background_folder:
        return {}, {}
    bg_dir = Path(background_folder)
    if not bg_dir.exists():
        logger.warning(
            "Background correction enabled but folder does not exist: %s",
            bg_dir,
        )
        return {}, {}
    try:
        from microscope_imageprocessing.correction.background import (
            BackgroundCorrectionUtils,
        )
        bg_images, scaling, _wb = BackgroundCorrectionUtils.load_background_images(
            bg_dir, angles, logger
        )
        if bg_images:
            logger.info(
                "Loaded %d background image(s) from %s for angles %s",
                len(bg_images),
                bg_dir,
                list(bg_images.keys()),
            )
        else:
            logger.warning(
                "No background images found in %s for angles %s -- "
                "correction will be skipped",
                bg_dir,
                angles,
            )
        return bg_images or {}, scaling or {}
    except Exception as e:
        logger.warning("Failed to load background images from %s: %s", bg_dir, e)
        return {}, {}


def _apply_bg_correction(image, bg_image, scaling, method: str, angle: float):
    """Apply flat-field (or subtractive) correction for a single snap.

    Returns the corrected image. Falls back to the original image on any
    exception so a per-angle failure doesn't lose the rest of the
    acquisition.
    """
    try:
        from microscope_imageprocessing.correction.background import (
            BackgroundCorrectionUtils,
        )
        return BackgroundCorrectionUtils.apply_flat_field_correction(
            image, bg_image, scaling, method=method,
        )
    except Exception as e:
        logger.warning(
            "Background correction failed at angle %s: %s -- using raw image",
            angle, e,
        )
        return image


def _resolve_pixel_size_um(config_manager, modality: str, objective, detector) -> float:
    """Best-effort pixel size lookup for OME-TIFF metadata.

    Tries the config manager's pixel-size resolver when available. Falls back
    to 1.0 um with a warning so the writer always has a numeric value (the
    OME-XML requires one). Downstream tools read PhysicalSizeX/Y from the
    emitted file and can correct scale if the true value is known elsewhere.
    """
    if config_manager is None:
        logger.warning("No config_manager available; using placeholder pixel_size_um=1.0")
        return 1.0

    for attr in ("get_pixel_size_um", "get_pixel_size", "pixel_size_um"):
        candidate = getattr(config_manager, attr, None)
        if candidate is None:
            continue
        try:
            value = candidate(modality, objective, detector) if callable(candidate) else candidate
            if value is not None:
                return float(value)
        except TypeError:
            try:
                value = candidate()
                if value is not None:
                    return float(value)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"pixel size resolver {attr} failed: {e}")

    logger.warning(
        "Could not resolve pixel_size_um from config_manager; "
        "using placeholder 1.0. Set modalities.%s.pixel_size_um in config or "
        "extend ConfigManager with get_pixel_size_um().",
        modality,
    )
    return 1.0


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
