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
    n_timepoints: int = 1,
    interval_seconds: float = 0.0,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict:
    """
    Acquire a Z-stack (optionally combined with time-lapse) at the current XY.

    Default behaviour (n_timepoints=1) is a single Z-stack: move through the
    Z range, snap at each plane, restore stage. With n_timepoints>1, the
    whole Z stack repeats at every timepoint with TimepointScheduler-driven
    pacing -- each timepoint starts at t0 + N*interval_seconds, drift bounded
    to a single interval. PPM angles iterate inside each (T, Z) point; output
    files keep the per-angle layout (timelapse acquisition naming conventions
    are reused so combined Z+T produces SizeT > 1 AND SizeZ > 1 in one
    OME-TIFF per angle).

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
        progress_callback: Called with (current_step, total_steps, message)
            where total_steps = n_timepoints * n_planes.
        projection: Z-projection operator ("none","max","min","sum","mean","std").
            Computed per timepoint when n_timepoints > 1.
        background_correction_enabled: When true, apply per-angle flat-field
            correction to every snap using images from background_folder.
        background_folder: Path to a directory containing per-angle background
            images. Layout matches the bounded-acquisition workflow's loader
            (BackgroundCorrectionUtils.load_background_images) -- typically
            {folder}/{angle}.tif or {folder}/{angle}/background.tif.
        background_correction_method: "divide" (flat-field) or "subtract".
        n_timepoints: Number of timepoints (default 1 = pure Z-stack).
            Combined with z_step, supports Z+T output (SizeT, SizeZ both > 1).
        interval_seconds: Seconds between timepoint START times when
            n_timepoints > 1. 0 means "as fast as possible". Pacing is
            anchored to t0 (NOT to the previous tp's completion) so slow
            iterations don't accumulate drift.
        cancel_check: Optional callable returning True to abort. Polled
            between timepoints during the interval wait.

    Returns:
        Dict with results: n_planes, n_timepoints, z_positions,
        output_folder, files, projected_file, elapsed_seconds.
    """
    output_path = Path(output_folder).resolve()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Hard-fail on a bad output path BEFORE acquiring 30+ images we
        # would then have nowhere to write. The handler-level try/except
        # surfaces this as FAILED:<reason> back to the client.
        raise OSError(
            f"Cannot create output folder {output_path!s}: {e}. Check that "
            f"the parent path exists and the drive is mounted."
        ) from e
    if not output_path.is_dir():
        raise OSError(f"Output folder did not materialize after mkdir: {output_path!s}")
    logger.info(f"Output folder ready: {output_path}")

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

    logger.info("=== Z-STACK ACQUISITION ===")
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
    # Validate T params
    if n_timepoints < 1:
        raise ValueError(f"n_timepoints must be >= 1, got {n_timepoints}")
    if interval_seconds < 0:
        raise ValueError(f"interval_seconds must be >= 0, got {interval_seconds}")
    has_t = n_timepoints > 1

    # Buffer frames per angle. For combined Z+T, the buffer holds one entry
    # per (t_idx, z_idx) within each angle so the writer can emit a single
    # OME-TIFF per angle with both SizeT and SizeZ. Keyed by
    # angle -> dict[(t_idx, z_idx)] = (actual_z, image, delta_t_s).
    angle_frames: Dict[float, Dict[Tuple[int, int], Tuple[float, Any, float]]] = {
        angle: {} for angle in angles
    }
    start_time = time.time()
    actual_n_timepoints = 0

    # Capture starting XYZ so we can restore the stage after the acquisition.
    # Users expect the stage to return to the original position (not sit at
    # the last Z plane of the stack).
    from microscope_control.hardware import Position

    try:
        start_position = hardware.get_current_position()
        logger.info(
            f"Acquisition start position: X={start_position.x:.2f}, "
            f"Y={start_position.y:.2f}, Z={start_position.z:.2f} um"
        )
    except Exception as e:
        logger.warning(f"Could not capture start position for restore: {e}")
        start_position = None

    # Optional T-outer scheduler. Only instantiated when n_timepoints > 1
    # so pure Z-stack callers don't pull in the scheduler module.
    scheduler = None
    if has_t:
        from microscope_command_server.acquisition.timepoint_scheduler import (
            TimepointScheduler,
        )

        t0 = time.monotonic()
        scheduler = TimepointScheduler(
            t0_monotonic=t0,
            interval_seconds=interval_seconds,
            logger=logger,
        )
        logger.info(
            f"=== Z+T ACQUISITION: {n_timepoints} timepoints x {n_planes} Z planes "
            f"x {len(angles)} angle(s), interval={interval_seconds}s ==="
        )

    total_steps = n_timepoints * n_planes
    try:
        for t_idx in range(n_timepoints):
            # Cancellation + scheduled wait before timepoints after the first.
            if cancel_check and cancel_check():
                logger.info(f"Acquisition cancelled at timepoint {t_idx + 1}/{n_timepoints}")
                break
            if has_t and t_idx > 0:
                scheduler.wait_until(t_idx, cancel_event=cancel_check)
                if cancel_check and cancel_check():
                    logger.info(
                        f"Acquisition cancelled during wait before timepoint "
                        f"{t_idx + 1}/{n_timepoints}"
                    )
                    break
            tp_start_wall = time.time()
            tp_elapsed = tp_start_wall - start_time
            if has_t:
                logger.info(f"=== Timepoint {t_idx + 1}/{n_timepoints} at T={tp_elapsed:.1f}s ===")

            for plane_idx, z_pos in enumerate(z_positions):
                step_idx = t_idx * n_planes + plane_idx
                if progress_callback:
                    msg = f"Z={z_pos:.1f} um"
                    if has_t:
                        msg = f"T{t_idx + 1}/{n_timepoints}, " + msg
                    progress_callback(step_idx, total_steps, msg)

                # Move Z
                hardware.move_to_position(Position(z=z_pos))
                actual_z = hardware.get_z_position()
                logger.info(
                    f"Plane {plane_idx + 1}/{n_planes}: Z={actual_z:.2f} um "
                    f"(target={z_pos:.2f})"
                )

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
                        logger.error(f"Failed to snap at T={t_idx} Z={z_pos}, angle={angle}")
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

                    angle_frames[angle][(t_idx, plane_idx)] = (
                        actual_z,
                        image,
                        tp_elapsed,
                    )
            actual_n_timepoints = t_idx + 1
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
    if not has_t:
        actual_n_timepoints = 1

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
        first_key = next(iter(frames))
        first_img = frames[first_key][1]
        size_y, size_x = first_img.shape[:2]
        is_rgb = first_img.ndim == 3 and first_img.shape[2] == 3

        angle_suffix = f"_angle{angle:.0f}" if len(angles) > 1 else ""
        # File stem: keep "zstack" for pure Z, "zstack_t" prefix when both
        # T and Z are >1 so downstream tools can tell from the filename
        # without having to read OME-XML.
        if has_t:
            out_file = output_path / f"zstack_t{angle_suffix}.ome.tiff"
        else:
            out_file = output_path / f"zstack{angle_suffix}.ome.tiff"

        channel_names = ["RGB"] if is_rgb else [f"{modality}{angle_suffix}"]

        with StackWriter(
            output_path=out_file,
            size_t=actual_n_timepoints,
            size_z=n_planes,
            size_c=1,
            size_y=size_y,
            size_x=size_x,
            dtype=first_img.dtype,
            pixel_size_um=pixel_size_um,
            z_step_um=z_step,
            time_increment_s=interval_seconds if has_t and interval_seconds > 0 else None,
            channel_names=channel_names,
            granularity="single",
            photometric="rgb" if is_rgb else "minisblack",
        ) as writer:
            for (t_idx, z_idx), (actual_z, image, delta_t_s) in frames.items():
                writer.write_frame(
                    image,
                    t=t_idx,
                    z=z_idx,
                    c=0,
                    plane_metadata={
                        "position_z_um": float(actual_z),
                        "delta_t_s": float(delta_t_s),
                        "angle": float(angle),
                    },
                )
        # Defensive check: the writer's `with` block exited cleanly above,
        # but verify the file actually landed on disk. A silent
        # not-on-disk would previously be reported as success in the log
        # while the user found nothing in the output folder.
        if not out_file.is_file():
            raise IOError(
                f"OME-TIFF writer reported success but file not on disk: "
                f"{out_file!s}. Check filesystem permissions, antivirus "
                f"quarantine, or remote-mount sync."
            )
        file_bytes = out_file.stat().st_size
        if file_bytes <= 0:
            raise IOError(f"OME-TIFF written but zero-bytes: {out_file!s}")
        saved_files.append(str(out_file))
        logger.info(
            f"Wrote OME-TIFF: {out_file} "
            f"(SizeT={actual_n_timepoints}, SizeZ={n_planes}, "
            f"{file_bytes / 1024 / 1024:.1f} MB)"
        )

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
                # Project per timepoint (so combined Z+T produces a SizeT
                # projection file alongside the full Z+T cube). For pure
                # Z-stack, this collapses to a single 2D plane as before.
                # Group buffer entries by t_idx -> list of (z_idx, image)
                # then run the projection function on the ordered images.
                from collections import defaultdict

                by_t: Dict[int, list] = defaultdict(list)
                for (t_idx, z_idx), (_actual_z, image, _delta_t) in frames.items():
                    by_t[t_idx].append((z_idx, image))
                if not by_t:
                    continue

                # Reference frame for shape / dtype.
                ref_t = next(iter(by_t))
                ref_z, ref_img = sorted(by_t[ref_t])[0]
                size_y, size_x = ref_img.shape[:2]
                is_rgb = ref_img.ndim == 3 and ref_img.shape[2] == 3

                angle_suffix = f"_angle{angle:.0f}" if len(angles) > 1 else ""
                proj_file = output_path / (f"zstack_{projection_name}{angle_suffix}.ome.tiff")

                with StackWriter(
                    output_path=proj_file,
                    size_t=actual_n_timepoints,
                    size_z=1,
                    size_c=1,
                    size_y=size_y,
                    size_x=size_x,
                    dtype=ref_img.dtype,
                    pixel_size_um=pixel_size_um,
                    time_increment_s=(interval_seconds if has_t and interval_seconds > 0 else None),
                    channel_names=["RGB" if is_rgb else f"{projection_name}{angle_suffix}"],
                    granularity="single",
                    photometric="rgb" if is_rgb else "minisblack",
                ) as writer:
                    for t_idx in range(actual_n_timepoints):
                        if t_idx not in by_t:
                            continue
                        ordered = [img for _z, img in sorted(by_t[t_idx])]
                        projected = projection_fn(ordered)
                        writer.write_frame(projected, t=t_idx, z=0, c=0)

                projected_file = str(proj_file)
                logger.info(
                    f"Projection ({projection_name}): {proj_file} " f"(SizeT={actual_n_timepoints})"
                )
        except Exception as e:
            logger.warning(f"Projection failed: {e}")

    elapsed = time.time() - start_time
    label = "Z+T" if has_t else "Z-STACK"
    logger.info(
        f"=== {label} COMPLETE: T={actual_n_timepoints}, Z={n_planes}, "
        f"{len(saved_files)} files, {elapsed:.1f}s ==="
    )

    if progress_callback:
        progress_callback(total_steps, total_steps, "Complete")

    return {
        "n_planes": n_planes,
        "n_timepoints": actual_n_timepoints,
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
    output_path = Path(output_folder).resolve()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Cannot create output folder {output_path!s}: {e}. Check that "
            f"the parent path exists and the drive is mounted."
        ) from e
    if not output_path.is_dir():
        raise OSError(f"Output folder did not materialize after mkdir: {output_path!s}")

    if n_timepoints <= 0:
        raise ValueError(f"n_timepoints must be positive, got {n_timepoints}")

    logger.info("=== TIME-LAPSE ACQUISITION ===")
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
        if not out_file.is_file():
            raise IOError(
                f"OME-TIFF writer reported success but file not on disk: "
                f"{out_file!s}. Check filesystem permissions, antivirus "
                f"quarantine, or remote-mount sync."
            )
        file_bytes = out_file.stat().st_size
        if file_bytes <= 0:
            raise IOError(f"OME-TIFF written but zero-bytes: {out_file!s}")
        saved_files.append(str(out_file))
        logger.info(
            f"Wrote time-lapse OME-TIFF: {out_file} (SizeT={len(frames)}, "
            f"{file_bytes / 1024 / 1024:.1f} MB)"
        )

    elapsed = time.time() - start_time
    logger.info(
        f"=== TIME-LAPSE COMPLETE: {actual_tp} timepoints, "
        f"{len(saved_files)} files, {elapsed:.1f}s ==="
    )

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
                "No background images found in %s for angles %s -- " "correction will be skipped",
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
            image,
            bg_image,
            scaling,
            method=method,
        )
    except Exception as e:
        logger.warning(
            "Background correction failed at angle %s: %s -- using raw image",
            angle,
            e,
        )
        return image


def _resolve_pixel_size_um(config_manager, modality: str, objective, detector) -> float:
    """Resolve pixel size for OME-TIFF metadata via ConfigManager.get_pixel_size.

    Reads from `acq_profiles.defaults[*].settings.pixel_size_xy_um[detector]`.
    Falls back to 1.0 um with a warning so the writer always has a numeric
    value (OME-XML requires one).
    """
    if config_manager is None:
        logger.warning("No config_manager available; using placeholder pixel_size_um=1.0")
        return 1.0

    if not objective or not detector:
        logger.warning(
            "Missing objective (%r) or detector (%r); using placeholder pixel_size_um=1.0",
            objective,
            detector,
        )
        return 1.0

    fn = getattr(config_manager, "get_pixel_size", None)
    if callable(fn):
        try:
            value = fn(objective, detector)
            if value is not None:
                return float(value)
        except Exception as e:
            logger.debug(f"get_pixel_size({objective!r}, {detector!r}) failed: {e}")

    logger.warning(
        "Could not resolve pixel_size_um for objective=%s detector=%s; "
        "using placeholder 1.0. Add the pair to "
        "acq_profiles.defaults[*].settings.pixel_size_xy_um in the active config.",
        objective,
        detector,
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

        if image.ndim == 3 and image.shape[2] == 3 or image.ndim == 2:
            PILImage.fromarray(image).save(str(filepath))
        else:
            np.save(str(filepath).replace(".tif", ".npy"), image)
    logger.debug(f"Saved: {filepath}")
