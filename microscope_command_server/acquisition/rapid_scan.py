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
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Polling interval for frame grabs during streaming (seconds).
STREAM_POLL_SLEEP_S = 0.002

# After the stage reports idle, keep polling this long to catch
# the last in-motion frame (ms).
STREAM_TAIL_MS = 50.0

# Conservative effective frame rate for full-frame ZMQ transfer.
# Used to calculate the required stage velocity.
# The actual rate depends on frame size and system load.
ESTIMATED_EFFECTIVE_FPS = 5.0

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
    progress_dict=None,
):
    """Streaming XY tiled acquisition over a rectangular region.

    The stage is slowed so that at the effective ZMQ frame capture rate,
    consecutive frames have the desired overlap. This is the same principle
    as the streaming autofocus slowing the Z stage.

    Args:
        hardware: Hardware abstraction with .stage, .set_exposure, .core
        output_folder: Directory to save tiles and TileConfiguration.txt
        center_x, center_y: Center of scan region (stage um)
        width, height: Scan region size (um)
        overlap_percent: Tile overlap (0-50%)
        exposure_ms: Exposure time (max 0.5ms)
        fov_width, fov_height: Camera FOV (um)
        progress_dict: Optional dict for progress tracking

    Returns:
        dict with n_tiles, saved, elapsed_seconds, etc.
    """
    import tifffile

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if exposure_ms > 0.5:
        raise ValueError(
            f"Exposure {exposure_ms}ms exceeds 0.5ms limit for rapid scan"
        )

    # Set camera exposure
    logger.info("Setting exposure to %.3f ms", exposure_ms)
    hardware.set_exposure(exposure_ms)

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
    target_velocity = target_step_x * ESTIMATED_EFFECTIVE_FPS  # um/s

    logger.info(
        "Rapid scan (streaming): %d rows, Y step=%.1f um, "
        "X sweep=%.1f um, target velocity=%.0f um/s (%.1f mm/s)",
        n_rows, step_y, row_distance, target_velocity, target_velocity / 1000.0,
    )
    logger.info(
        "  Target tile step: %.1f um at ~%.0f fps -> %.1f%% overlap",
        target_step_x, ESTIMATED_EFFECTIVE_FPS, overlap_percent,
    )

    # ---- Hardware setup ----
    core = hardware.core
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

    # Find and set XY stage speed property
    xy_device = core.get_xy_stage_device()
    speed_prop = _find_speed_property(core, xy_device)
    original_speed = None

    if speed_prop:
        try:
            original_speed = core.get_property(xy_device, speed_prop)
        except Exception:
            pass

        # Calculate speed setting.
        # Prior ProScan MaxSpeed is 1-100 (percentage of max).
        # Max XY velocity is ~20 mm/s = 20000 um/s.
        # speed_pct = (target_velocity / max_velocity) * 100
        max_xy_velocity = 20000.0  # um/s, typical Prior ProScan
        speed_pct = max(1, min(100, int(target_velocity / max_xy_velocity * 100)))

        if _try_set(core, xy_device, speed_prop, str(speed_pct)):
            logger.info(
                "Set XY stage %s=%d%% (target %.0f um/s, was %s)",
                speed_prop, speed_pct, target_velocity, original_speed,
            )
        else:
            logger.warning("Could not set XY stage speed -- will use current speed")
    else:
        logger.warning("No speed property found on XY stage device '%s'", xy_device)

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
                row_idx + 1, n_rows,
                row_x_start, row_y, row_x_end, row_y, row_distance,
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

            actual_velocity = row_distance / (motion_duration_ms / 1000.0) if motion_duration_ms > 0 else 0
            logger.info(
                "  row %d: %d frames kept (%d stationary discarded), "
                "%.0fms motion, %.0f um/s actual, %.1f fps",
                row_idx + 1, saved_this_row, n_stationary,
                motion_duration_ms, actual_velocity,
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

        # Restore XY stage speed
        if speed_prop and original_speed is not None:
            if _try_set(core, xy_device, speed_prop, original_speed):
                logger.info("Restored XY stage %s=%s", speed_prop, original_speed)
            else:
                # Always try to restore to full speed
                _try_set(core, xy_device, speed_prop, NORMAL_SPEED_VALUE)

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
        n_tiles, n_rows, elapsed,
        elapsed / n_tiles if n_tiles > 0 else 0,
    )

    return {
        "n_tiles": n_tiles,
        "saved": n_tiles,
        "output_folder": str(output_path),
        "elapsed_seconds": elapsed,
        "tile_config_path": str(config_path),
    }
