"""Rapid scan acquisition -- streaming XY tiled brightfield.

Adapts the streaming autofocus pattern (continuous camera + non-blocking
stage motion + circular-buffer frame grabbing) for XY tile acquisition.
The stage moves continuously through each row while the camera streams
frames into the circular buffer. Frame positions are interpolated from
wall-clock timestamps and known start/end positions (same approach as
streaming AF's Z interpolation).

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
# 2ms matches the streaming AF cadence.
STREAM_POLL_SLEEP_S = 0.002

# After the stage reports idle, keep polling this long to catch
# trailing frames in the buffer (ms).
STREAM_TAIL_MS = 100.0


def _process_raw_frame(arr, img_w, img_h, img_nch):
    """Reshape raw pixel buffer and convert BGRA -> RGB if needed.

    Raw frames from core.get_last_image() are flat pixel buffers in the
    camera's native byte order. For 3-CCD cameras like JAI, this is
    BGRA (4 components). We need standard RGB TIFFs for the stitcher.
    """
    if img_w <= 0 or img_h <= 0:
        return arr

    try:
        if img_nch > 1:
            arr = arr.reshape(img_h, img_w, img_nch)
            # BGRA -> RGB: reverse channel order, drop alpha if 4 channels
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

    Adapts the streaming autofocus pattern: continuous camera acquisition
    into the circular buffer while the stage moves continuously through
    each row. Frame positions are interpolated from wall-clock timestamps
    and known row start/end positions -- NO position reads in the capture
    loop (same approach as streaming AF's Z interpolation).

    Args:
        hardware: Hardware abstraction with .stage, .set_exposure, .core
        output_folder: Directory to save tiles and TileConfiguration.txt
        center_x, center_y: Center of scan region (stage um)
        width, height: Scan region size (um)
        overlap_percent: Tile overlap -- controls Y row spacing (0-50%)
        exposure_ms: Exposure time (max 0.5ms)
        fov_width, fov_height: Camera FOV (um)
        progress_dict: Optional dict for progress tracking

    Returns:
        dict with n_tiles, saved, elapsed_seconds, output_folder,
              tile_config_path
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

    # Compute row positions.
    step_y = fov_height * (1.0 - overlap_percent / 100.0)
    n_rows = max(1, math.ceil(height / step_y))

    # X scan boundaries -- extend by half FOV so edges are fully covered
    x_left = center_x - width / 2.0 - fov_width / 2.0
    x_right = center_x + width / 2.0 + fov_width / 2.0

    # Y positions for each row
    start_y = center_y - (n_rows - 1) * step_y / 2.0
    row_y_positions = [start_y + row * step_y for row in range(n_rows)]

    logger.info(
        "Rapid scan (streaming): %d rows, Y step=%.1f um, "
        "X sweep=%.1f -> %.1f um (%.1f um), exposure=%.3f ms",
        n_rows, step_y, x_left, x_right, x_right - x_left, exposure_ms,
    )

    # ---- Streaming acquisition ----
    core = hardware.core
    all_tiles = []  # (filename, interp_x, interp_y)
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

    # Pause stage position cache to eliminate serial bus contention
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
        except Exception as e:
            logger.debug("Could not pause stage cache: %s", e)

    # Stop any existing sequence (e.g., Live Viewer)
    sequence_was_running = False
    try:
        if core.is_sequence_running():
            sequence_was_running = True
            core.stop_sequence_acquisition()
            time.sleep(0.05)
            logger.info("Stopped pre-existing sequence acquisition")
    except Exception as e:
        logger.debug("Could not stop existing sequence: %s", e)

    # Start continuous acquisition
    core.clear_circular_buffer()
    core.start_continuous_sequence_acquisition(0)
    logger.info("Started continuous acquisition for streaming scan")
    time.sleep(0.1)

    tile_idx = 0
    try:
        for row_idx in range(n_rows):
            row_y = row_y_positions[row_idx]

            # Serpentine: even rows L->R, odd rows R->L
            if row_idx % 2 == 0:
                row_x_start, row_x_end = x_left, x_right
            else:
                row_x_start, row_x_end = x_right, x_left

            row_distance = abs(row_x_end - row_x_start)

            # Move to row start (blocking)
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
                "Row %d/%d: streaming (%.1f, %.1f) -> (%.1f, %.1f), %.0f um",
                row_idx + 1, n_rows,
                row_x_start, row_y, row_x_end, row_y, row_distance,
            )

            # ---- Frame capture loop ----
            # Match the streaming AF pattern exactly:
            # - get_last_image() (fast peek, no ZMQ pop overhead)
            # - Track new frames via remaining_image_count delta
            # - Record wall_ms timestamps (NO position reads in loop)
            # - Interpolate XY from timing after row completes
            hard_deadline_s = max(row_distance / 5000.0, 3.0)  # 5mm/s min assumed
            xy_device = core.get_xy_stage_device()

            last_remaining = -1
            raw_captures: List[Tuple[float, np.ndarray]] = []  # (wall_ms, pixels)
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

                # Check stage idle for tail exit
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

            # Measure actual move duration for interpolation
            if stage_idle_since is not None:
                motion_duration_ms = (stage_idle_since - t_move_fired) * 1000.0
            else:
                # Stage didn't report idle before deadline -- use elapsed time
                motion_duration_ms = (time.perf_counter() - t_move_fired) * 1000.0

            row_elapsed_ms = (time.perf_counter() - t_move_fired) * 1000.0

            # ---- Interpolate XY positions from timestamps ----
            # Linear model: same as streaming AF's Z interpolation.
            # x(t) = x_start + (x_end - x_start) * (t / motion_duration)
            # Clamped to [x_start, x_end] for frames during accel/decel.
            direction = 1.0 if row_x_end >= row_x_start else -1.0
            saved_this_row = 0

            for wall_ms, arr in raw_captures:
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

            logger.info(
                "  row %d: %d frames in %.0fms (%.1f fps), "
                "motion=%.0fms, velocity=%.0f um/s",
                row_idx + 1, saved_this_row, row_elapsed_ms,
                saved_this_row / (row_elapsed_ms / 1000.0) if row_elapsed_ms > 0 else 0,
                motion_duration_ms,
                row_distance / (motion_duration_ms / 1000.0) if motion_duration_ms > 0 else 0,
            )

            if saved_this_row == 0:
                logger.warning("  row %d: NO FRAMES captured!", row_idx + 1)

            if progress_dict is not None:
                progress_dict["completed_rows"] = row_idx + 1

    finally:
        try:
            core.stop_sequence_acquisition()
        except Exception:
            pass
        try:
            core.clear_circular_buffer()
        except Exception:
            pass
        logger.info("Stopped continuous acquisition")

        # Restore stage cache
        if stage_cache is not None and cache_was_running:
            try:
                stage_cache.resume()
                logger.info("Resumed stage position cache")
            except Exception as e:
                logger.debug("Could not resume stage cache: %s", e)

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
