"""Rapid scan acquisition -- streaming XY tiled brightfield.

Adapts the streaming autofocus pattern (continuous camera + non-blocking
stage motion + circular-buffer frame grabbing) for XY tile acquisition.
The stage moves continuously through each row while the camera streams
frames into the circular buffer. Every captured frame is saved with its
actual XY position -- the stitcher handles placement from real coordinates.

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

    Raw frames from core.get_last_image() / pop_next_image() are flat
    pixel buffers in the camera's native byte order. For 3-CCD cameras
    like JAI, this is BGRA (4 components). We need standard RGB TIFFs
    for the stitcher.
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
    each row. Every frame is saved with its actual XY position read at
    capture time -- no snapping to a pre-planned grid.

    The overlap_percent and FOV determine row spacing (Y step between rows).
    Within each row, the stage moves continuously and frames are captured
    at whatever cadence the camera delivers.

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
    # Y step from overlap, X is continuous (no discrete columns).
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
    logger.info(
        "Rapid scan region: center=(%.1f, %.1f), size=%.1fx%.1f um, "
        "FOV=%.1fx%.1f um",
        center_x, center_y, width, height, fov_width, fov_height,
    )

    # ---- Streaming acquisition ----
    core = hardware.core
    all_tiles = []  # List of (filename, actual_x, actual_y)
    start_time = time.time()

    if progress_dict is not None:
        progress_dict["total_rows"] = n_rows
        progress_dict["completed_rows"] = 0
        progress_dict["status"] = "scanning"

    # Cache image geometry for reshape
    try:
        img_w = int(core.get_image_width())
        img_h = int(core.get_image_height())
        img_nch = int(core.get_number_of_components())
    except Exception:
        img_w = img_h = 0
        img_nch = 1
    logger.info("Camera frame: %dx%d, %d channels", img_w, img_h, img_nch)

    # Pause stage position cache to avoid serial bus contention
    stage_cache = getattr(hardware, "_stage_cache", None)
    if stage_cache is None:
        stage_cache = getattr(hardware, "stage_cache", None)
    cache_was_running = False
    if stage_cache is not None:
        try:
            cache_was_running = stage_cache.is_running()
            if cache_was_running:
                stage_cache.pause()
                logger.info("Paused stage position cache for streaming scan")
        except Exception as e:
            logger.debug("Could not pause stage cache: %s", e)

    # Stop any existing sequence (e.g., Live Viewer) before starting ours
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
    time.sleep(0.1)  # let first frames arrive

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

            # Flush stale frames from buffer
            try:
                core.clear_circular_buffer()
            except Exception:
                pass
            time.sleep(0.02)

            # Fire non-blocking move across the row
            hardware.stage.move_xy_no_wait(row_x_end, row_y)

            logger.info(
                "Row %d/%d: streaming (%.1f, %.1f) -> (%.1f, %.1f), "
                "%.0f um, %d tiles so far",
                row_idx + 1, n_rows,
                row_x_start, row_y, row_x_end, row_y, row_distance,
                tile_idx,
            )

            # Grab frames while stage moves.
            # Use stage busy state as primary exit condition, with a hard
            # deadline as safety net.
            est_velocity = 10000.0  # um/s, conservative
            est_duration_s = row_distance / max(est_velocity, 1.0)
            hard_deadline_s = max(est_duration_s * 5.0, 3.0)

            xy_device = core.get_xy_stage_device()
            t0 = time.perf_counter()
            deadline = t0 + hard_deadline_s
            row_frames: List[Tuple[np.ndarray, float, float]] = []
            last_remaining = -1
            stage_idle_since = None

            while time.perf_counter() < deadline:
                # Read position BEFORE image to minimize temporal offset
                try:
                    actual_x = core.get_x_position()
                    actual_y = core.get_y_position()
                except Exception:
                    actual_x = actual_y = float("nan")

                # Drain all available frames from the buffer
                try:
                    remaining = core.get_remaining_image_count()
                except Exception:
                    remaining = 0

                if remaining > 0:
                    try:
                        pixels = core.pop_next_image()
                    except Exception:
                        pixels = None

                    if pixels is not None:
                        arr = np.asarray(pixels).copy()
                        row_frames.append((arr, actual_x, actual_y))

                # Check if stage has finished moving
                try:
                    stage_busy = core.device_busy(xy_device)
                except Exception:
                    stage_busy = True

                if not stage_busy:
                    if stage_idle_since is None:
                        stage_idle_since = time.perf_counter()
                    # Tail: keep capturing for STREAM_TAIL_MS after stage stops
                    elif (time.perf_counter() - stage_idle_since) * 1000.0 > STREAM_TAIL_MS:
                        break
                else:
                    stage_idle_since = None

                time.sleep(STREAM_POLL_SLEEP_S)

            row_elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Warn on buffer overflow
            try:
                if core.is_buffer_overflowed():
                    logger.warning("  row %d: circular buffer overflowed!", row_idx + 1)
            except Exception:
                pass

            # Process and save all captured frames for this row
            n_saved_this_row = 0
            for arr, actual_x, actual_y in row_frames:
                filename = f"{tile_idx}.tif"
                arr = _process_raw_frame(arr, img_w, img_h, img_nch)

                filepath = output_path / filename
                tifffile.imwrite(str(filepath), arr)
                all_tiles.append((filename, actual_x, actual_y))
                tile_idx += 1
                n_saved_this_row += 1

            logger.info(
                "  row %d: %d frames in %.0fms (%.1f fps)",
                row_idx + 1, n_saved_this_row, row_elapsed_ms,
                n_saved_this_row / (row_elapsed_ms / 1000.0) if row_elapsed_ms > 0 else 0,
            )

            if n_saved_this_row == 0:
                logger.warning("  row %d: NO FRAMES captured!", row_idx + 1)

            if progress_dict is not None:
                progress_dict["completed_rows"] = row_idx + 1

    finally:
        # Always stop our continuous acquisition
        try:
            core.stop_sequence_acquisition()
        except Exception:
            pass
        try:
            core.clear_circular_buffer()
        except Exception:
            pass
        logger.info("Stopped continuous acquisition")

        # Restore stage position cache
        if stage_cache is not None and cache_was_running:
            try:
                stage_cache.resume()
                logger.info("Resumed stage position cache")
            except Exception as e:
                logger.debug("Could not resume stage cache: %s", e)

    # Write TileConfiguration.txt with ACTUAL captured positions
    config_path = output_path / "TileConfiguration.txt"
    with open(config_path, "w") as f:
        f.write("dim = 2\n")
        for filename, actual_x, actual_y in all_tiles:
            f.write(f"{filename}; ; ({actual_x:.3f}, {actual_y:.3f})\n")

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
