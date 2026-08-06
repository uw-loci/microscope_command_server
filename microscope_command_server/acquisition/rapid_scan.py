"""Rapid scan acquisition -- streaming XY tiled brightfield.

Adapts the streaming autofocus pattern (continuous camera + non-blocking
stage motion + circular-buffer frame grabbing) for XY tile acquisition.
The stage is slowed to match the effective frame capture rate, ensuring
adequate overlap between consecutive frames. Frame positions are
interpolated from wall-clock timestamps and known start/end positions.

No autofocus, no Z movement, no background correction, no debayering.
"""

import logging
import math
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Polling interval for frame grabs during streaming (seconds).
STREAM_POLL_SLEEP_S = 0.002

# After the stage reports idle, keep polling this long to catch
# the last in-motion frame (ms).
STREAM_TAIL_MS = 50.0

# Effective frame rate for full-frame ZMQ transfer (~12.7MB per JAI frame).
# Measured at ~2.5 fps on PPM (2026-04-22). The Pycromanager ZMQ bridge
# bottleneck is the image data transfer, not the camera frame rate.
ESTIMATED_EFFECTIVE_FPS = 2.5

# Stage speed property candidates (same as streaming AF).
SPEED_PROPERTY_CANDIDATES = ("MaxSpeed", "Velocity", "Speed", "MaxVelocity")

# The Prior ProScan MaxSpeed is a 1-100 percentage scale.
NORMAL_SPEED_VALUE = "100"


def _find_speed_property(core, device):
    """Return the first writable speed property on the device."""
    try:
        props = list(core.get_device_property_names(device))
    except Exception:
        return None
    for name in props:
        if name in SPEED_PROPERTY_CANDIDATES:
            try:
                if core.is_property_read_only(device, name):
                    continue
            except Exception:
                pass
            return name
    return None


def _try_set(core, device, prop, value):
    """Set a device property, returning True on success."""
    try:
        core.set_property(device, prop, str(value))
        return True
    except Exception as e:
        logger.debug("set_property(%s.%s=%s) failed: %s", device, prop, value, e)
        return False


def _process_raw_frame(arr, img_w, img_h, img_nch):
    """Reshape raw pixel buffer and convert BGRA -> RGB if needed."""
    if img_w <= 0 or img_h <= 0:
        return arr
    try:
        if img_nch > 1:
            arr = arr.reshape(img_h, img_w, img_nch)
            arr = arr[:, :, ::-1]  # BGRA -> ARGB
            if arr.shape[2] == 4:
                arr = arr[:, :, 1:]  # ARGB -> RGB
        else:
            arr = arr.reshape(img_h, img_w)
    except Exception as e:
        logger.debug("Frame reshape/reorder failed: %s", e)
    return arr


def acquire_rapid_scan(
    hardware,
    output_folder,
    center_x,
    center_y,
    width,
    height,
    overlap_percent,
    exposure_ms,
    fov_width,
    fov_height,
    binning=2,
    progress_dict=None,
):
    """Streaming XY tiled acquisition over a rectangular region.

    The stage is slowed so that at the effective ZMQ frame capture rate,
    consecutive frames have the desired overlap. Camera binning reduces
    the frame size transferred over ZMQ, increasing effective fps.

    Args:
        hardware: Hardware abstraction with .stage, .set_exposure, .core
        output_folder: Directory to save tiles and TileConfiguration.txt
        center_x, center_y: Center of scan region (stage um)
        width, height: Scan region size (um)
        overlap_percent: Tile overlap (0-50%)
        exposure_ms: Exposure time (max 0.5ms)
        fov_width, fov_height: Camera FOV (um)
        binning: Camera binning factor (1=full res, 2=2x2 binning).
                 Higher binning = faster ZMQ transfer = more frames/row.
        progress_dict: Optional dict for progress tracking

    Returns:
        dict with n_tiles, saved, elapsed_seconds, binning, etc.
    """
    import tifffile

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if exposure_ms > 0.5:
        raise ValueError(f"Exposure {exposure_ms}ms exceeds 0.5ms limit for rapid scan")

    # Set camera exposure
    logger.info("Setting exposure to %.3f ms", exposure_ms)
    hardware.set_exposure(exposure_ms)

    # Set camera binning for faster ZMQ transfer.
    # Binning=2 -> 2x2 -> 4x fewer pixels -> ~4x faster frame transfer.
    # FOV stays the same, pixel size doubles.
    # Routes through the Camera abstraction so the same set/restore path
    # works for any camera type and the new GETBIN/SETBIN socket commands
    # see consistent state.
    core = hardware.core
    camera_device = core.get_camera_device()
    camera = hardware.camera
    original_binning = None
    if binning > 1:
        try:
            original_binning = camera.get_binning()
            if original_binning is None:
                # We cannot read the current binning, so we could not put it
                # back afterwards -- the restore below is guarded on None and
                # would silently skip, leaving the camera binned for whatever
                # the user does next. Do not change hardware state you cannot
                # undo. Take the same path as a failed set_binning: run
                # unbinned.
                logger.warning(
                    "Camera did not report its current binning, so binning cannot be "
                    "restored after the scan; running unbinned instead of setting %d",
                    binning,
                )
                binning = 1
            else:
                camera.set_binning(binning)
                logger.info("Set camera binning=%d (was %s)", binning, original_binning)
        except Exception as e:
            logger.warning("Could not set binning=%d: %s", binning, e)
            binning = 1  # fall back to no binning

    # Compute row positions
    step_y = fov_height * (1.0 - overlap_percent / 100.0)
    n_rows = max(1, math.ceil(height / step_y))

    # X boundaries -- extend by half FOV for full edge coverage
    x_left = center_x - width / 2.0 - fov_width / 2.0
    x_right = center_x + width / 2.0 + fov_width / 2.0
    row_distance = abs(x_right - x_left)

    # Y positions
    start_y = center_y - (n_rows - 1) * step_y / 2.0
    row_y_positions = [start_y + row * step_y for row in range(n_rows)]

    # Calculate target stage velocity to achieve desired overlap.
    # At ESTIMATED_EFFECTIVE_FPS, the frame-to-frame step in X is:
    #   step_x = velocity / fps
    # For desired overlap:
    #   step_x = fov_width * (1 - overlap/100)
    # Therefore:
    #   velocity = fov_width * (1 - overlap/100) * fps
    target_step_x = fov_width * (1.0 - overlap_percent / 100.0)
    # Binning reduces frame size -> faster ZMQ transfer -> higher effective fps.
    # Binning=2 -> 4x fewer pixels -> ~4x faster transfer.
    effective_fps = ESTIMATED_EFFECTIVE_FPS * (binning * binning)
    target_velocity = target_step_x * effective_fps  # um/s

    logger.info(
        "Rapid scan (streaming): %d rows, Y step=%.1f um, "
        "X sweep=%.1f um, binning=%d, effective fps=%.0f, "
        "target velocity=%.0f um/s (%.1f mm/s)",
        n_rows,
        step_y,
        row_distance,
        binning,
        effective_fps,
        target_velocity,
        target_velocity / 1000.0,
    )
    logger.info(
        "  Target tile step: %.1f um at ~%.0f fps -> %.1f%% overlap",
        target_step_x,
        effective_fps,
        overlap_percent,
    )

    # ---- Hardware setup ----
    all_tiles = []
    start_time = time.time()

    if progress_dict is not None:
        progress_dict["total_rows"] = n_rows
        progress_dict["completed_rows"] = 0
        progress_dict["status"] = "scanning"

    # Cache image geometry
    try:
        img_w = int(core.get_image_width())
        img_h = int(core.get_image_height())
        img_nch = int(core.get_number_of_components())
    except Exception:
        img_w = img_h = 0
        img_nch = 1
    logger.info("Camera frame: %dx%d, %d channels", img_w, img_h, img_nch)

    # Save starting position to restore after scan
    start_stage_x = core.get_x_position()
    start_stage_y = core.get_y_position()
    logger.info("Saved starting position: (%.1f, %.1f)", start_stage_x, start_stage_y)

    # Pause stage position cache
    stage_cache = getattr(hardware, "_stage_cache", None)
    if stage_cache is None:
        stage_cache = getattr(hardware, "stage_cache", None)
    cache_was_running = False
    if stage_cache is not None:
        try:
            cache_was_running = stage_cache.is_running()
            if cache_was_running:
                stage_cache.pause()
                logger.info("Paused stage position cache")
        except Exception:
            pass

    # Stop any existing sequence (e.g., Live Viewer)
    try:
        if core.is_sequence_running():
            core.stop_sequence_acquisition()
            time.sleep(0.05)
            logger.info("Stopped pre-existing sequence acquisition")
    except Exception:
        pass

    # Set XY stage speed to match frame capture rate.
    # Try MaxSpeed directly (Prior ProScan 1-100% scale) -- the property
    # search can fail due to StrVector iteration issues over Pycromanager.
    xy_device = core.get_xy_stage_device()
    original_speed = None
    speed_prop = "MaxSpeed"  # Prior ProScan XY stage property

    try:
        original_speed = core.get_property(xy_device, speed_prop)
        logger.info("XY stage current %s=%s", speed_prop, original_speed)
    except Exception:
        # Try fallback property names
        for candidate in ("Velocity", "Speed", "MaxVelocity"):
            try:
                original_speed = core.get_property(xy_device, candidate)
                speed_prop = candidate
                logger.info("XY stage current %s=%s", speed_prop, original_speed)
                break
            except Exception:
                continue
        else:
            speed_prop = None
            logger.warning("No speed property found on XY device '%s'", xy_device)

    if speed_prop:
        # Prior ProScan MaxSpeed is 1-100. Observed calibration (PPM 2026-04-22):
        #   MaxSpeed=100 -> ~6800 um/s avg (short moves)
        #   MaxSpeed=8   -> ~2600 um/s avg
        # The mapping is NOT linear. Use 1 (minimum) for slow streaming.
        # At MaxSpeed=1, the stage should be ~800-1000 um/s, which matches
        # our target for ~2.5 fps with 320um tile steps.
        speed_pct = 1  # slowest available
        if target_velocity > 3000:
            speed_pct = max(1, min(100, int(target_velocity / 6800.0 * 100)))

        if _try_set(core, xy_device, speed_prop, str(speed_pct)):
            logger.info(
                "Set XY stage %s=%d (target %.0f um/s)",
                speed_prop,
                speed_pct,
                target_velocity,
            )
        else:
            logger.warning("Could not set XY stage speed")

    # Start continuous acquisition
    core.clear_circular_buffer()
    core.start_continuous_sequence_acquisition(0)
    logger.info("Started continuous acquisition for streaming scan")
    time.sleep(0.1)

    tile_idx = 0
    try:
        for row_idx in range(n_rows):
            row_y = row_y_positions[row_idx]

            # Serpentine
            if row_idx % 2 == 0:
                row_x_start, row_x_end = x_left, x_right
            else:
                row_x_start, row_x_end = x_right, x_left

            # Move to row start (blocking, at the reduced speed)
            hardware.stage.move_xy(row_x_start, row_y)

            # Flush stale frames
            try:
                core.clear_circular_buffer()
            except Exception:
                pass
            time.sleep(0.02)

            # Fire non-blocking move across the row
            hardware.stage.move_xy_no_wait(row_x_end, row_y)
            t_move_fired = time.perf_counter()

            logger.info(
                "Row %d/%d: (%.1f, %.1f) -> (%.1f, %.1f), %.0f um",
                row_idx + 1,
                n_rows,
                row_x_start,
                row_y,
                row_x_end,
                row_y,
                row_distance,
            )

            # ---- Frame capture loop ----
            # get_last_image() (fast peek) + remaining count for new-frame detection
            # + wall_ms timestamps for position interpolation.
            # NO position reads in the hot loop.
            hard_deadline_s = max(row_distance / 500.0, 5.0)  # 0.5mm/s absolute min

            last_remaining = -1
            raw_captures: List[Tuple[float, np.ndarray]] = []
            stage_idle_since = None

            while (time.perf_counter() - t_move_fired) < hard_deadline_s:
                t_now_ms = (time.perf_counter() - t_move_fired) * 1000.0

                # Detect new frame via remaining count delta
                try:
                    remaining = core.get_remaining_image_count()
                except Exception:
                    remaining = last_remaining

                if remaining > last_remaining:
                    try:
                        pixels = core.get_last_image()
                    except Exception:
                        pixels = None

                    if pixels is not None:
                        arr = np.asarray(pixels).copy()
                        raw_captures.append((t_now_ms, arr))

                    last_remaining = remaining

                # Check stage idle
                try:
                    stage_busy = core.device_busy(xy_device)
                except Exception:
                    stage_busy = True

                if not stage_busy:
                    if stage_idle_since is None:
                        stage_idle_since = time.perf_counter()
                    elif (time.perf_counter() - stage_idle_since) * 1000.0 > STREAM_TAIL_MS:
                        break
                else:
                    stage_idle_since = None

                time.sleep(STREAM_POLL_SLEEP_S)

            # Measure actual motion duration
            if stage_idle_since is not None:
                motion_duration_ms = (stage_idle_since - t_move_fired) * 1000.0
            else:
                motion_duration_ms = (time.perf_counter() - t_move_fired) * 1000.0

            row_elapsed_ms = (time.perf_counter() - t_move_fired) * 1000.0

            # ---- Filter: only keep frames captured DURING motion ----
            # Frames after stage_idle are stationary duplicates.
            motion_cutoff_ms = motion_duration_ms + 10.0  # small margin
            in_motion = [(t, arr) for t, arr in raw_captures if t <= motion_cutoff_ms]
            n_stationary = len(raw_captures) - len(in_motion)

            # ---- Interpolate XY from timestamps ----
            saved_this_row = 0
            for wall_ms, arr in in_motion:
                if motion_duration_ms > 0:
                    progress = min(wall_ms / motion_duration_ms, 1.0)
                else:
                    progress = 1.0

                interp_x = row_x_start + (row_x_end - row_x_start) * progress
                interp_y = row_y

                filename = f"{tile_idx}.tif"
                arr = _process_raw_frame(arr, img_w, img_h, img_nch)

                filepath = output_path / filename
                tifffile.imwrite(str(filepath), arr)
                all_tiles.append((filename, interp_x, interp_y))
                tile_idx += 1
                saved_this_row += 1

            actual_velocity = (
                row_distance / (motion_duration_ms / 1000.0) if motion_duration_ms > 0 else 0
            )
            logger.info(
                "  row %d: %d frames kept (%d stationary discarded), "
                "%.0fms motion, %.0f um/s actual, %.1f fps",
                row_idx + 1,
                saved_this_row,
                n_stationary,
                motion_duration_ms,
                actual_velocity,
                saved_this_row / (motion_duration_ms / 1000.0) if motion_duration_ms > 0 else 0,
            )

            if saved_this_row == 0:
                logger.warning("  row %d: NO in-motion frames captured!", row_idx + 1)

            if progress_dict is not None:
                progress_dict["completed_rows"] = row_idx + 1

    finally:
        # Stop continuous acquisition
        try:
            core.stop_sequence_acquisition()
        except Exception:
            pass
        try:
            core.clear_circular_buffer()
        except Exception:
            pass
        logger.info("Stopped continuous acquisition")

        # Restore camera binning
        if original_binning is not None:
            try:
                camera.set_binning(int(original_binning))
                logger.info("Restored camera binning=%s", original_binning)
            except Exception as e:
                logger.warning("Could not restore binning: %s", e)

        # Restore XY stage speed (BEFORE moving back, so return is at full speed)
        if speed_prop and original_speed is not None:
            if _try_set(core, xy_device, speed_prop, original_speed):
                logger.info("Restored XY stage %s=%s", speed_prop, original_speed)
            else:
                _try_set(core, xy_device, speed_prop, NORMAL_SPEED_VALUE)

        # Return stage to starting position
        try:
            hardware.stage.move_xy(start_stage_x, start_stage_y)
            logger.info(
                "Returned stage to starting position (%.1f, %.1f)", start_stage_x, start_stage_y
            )
        except Exception as e:
            logger.warning("Could not return stage to start: %s", e)

        # Restore stage cache
        if stage_cache is not None and cache_was_running:
            try:
                stage_cache.resume()
                logger.info("Resumed stage position cache")
            except Exception:
                pass

    # Write TileConfiguration.txt with interpolated positions
    config_path = output_path / "TileConfiguration.txt"
    with open(config_path, "w") as f:
        f.write("dim = 2\n")
        for filename, x, y in all_tiles:
            f.write(f"{filename}; ; ({x:.3f}, {y:.3f})\n")

    elapsed = time.time() - start_time
    n_tiles = len(all_tiles)

    if progress_dict is not None:
        progress_dict["status"] = "complete"
        progress_dict["total_tiles"] = n_tiles
        progress_dict["completed_tiles"] = n_tiles

    logger.info(
        "Rapid scan complete: %d tiles across %d rows, %.1fs (%.2fs/tile)",
        n_tiles,
        n_rows,
        elapsed,
        elapsed / n_tiles if n_tiles > 0 else 0,
    )

    return {
        "n_tiles": n_tiles,
        "saved": n_tiles,
        "output_folder": str(output_path),
        "elapsed_seconds": elapsed,
        "tile_config_path": str(config_path),
        "binning": binning,
    }
