"""Rapid scan acquisition -- streaming XY tiled brightfield.

Adapts the streaming autofocus pattern (continuous camera + non-blocking
stage motion + circular-buffer frame grabbing) for XY tile acquisition.
The stage moves continuously through each row while the camera streams
frames into the circular buffer. Frames are grabbed at positions closest
to each planned tile center.

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
# 2ms matches the streaming AF cadence.
STREAM_POLL_SLEEP_S = 0.002

# After the stage should have arrived, keep polling this long
# to catch trailing frames (ms).
STREAM_TAIL_MS = 50.0


def _build_serpentine_grid(
    center_x, center_y, width, height,
    fov_width, fov_height, overlap_percent,
):
    """Compute tile positions in serpentine order, grouped by row.

    Returns:
        rows: list of lists, each inner list is [(x, y, filename), ...]
              Even rows go L->R, odd rows go R->L.
        n_cols, n_rows: grid dimensions
        step_x, step_y: tile step sizes in um
    """
    step_x = fov_width * (1.0 - overlap_percent / 100.0)
    step_y = fov_height * (1.0 - overlap_percent / 100.0)
    n_cols = max(1, math.ceil(width / step_x))
    n_rows = max(1, math.ceil(height / step_y))

    start_x = center_x - (n_cols - 1) * step_x / 2.0
    start_y = center_y - (n_rows - 1) * step_y / 2.0

    rows = []
    tile_idx = 0
    for row in range(n_rows):
        row_positions = []
        if row % 2 == 0:
            col_range = range(n_cols)
        else:
            col_range = range(n_cols - 1, -1, -1)
        for col in col_range:
            x = start_x + col * step_x
            y = start_y + row * step_y
            row_positions.append((x, y, f"{tile_idx}.tif"))
            tile_idx += 1
        rows.append(row_positions)

    return rows, n_cols, n_rows, step_x, step_y


def _grab_frames_during_move(
    core, row_positions, step_x, hard_deadline_s,
):
    """Pop frames from the circular buffer while the stage moves across a row.

    For each planned tile position, keeps the frame whose actual XY is
    closest to the tile center. Returns a list of (image_array, actual_x, actual_y)
    aligned 1:1 with row_positions.

    The stage move must already be issued (non-blocking) before calling this.
    """
    n_tiles = len(row_positions)
    # Best frame per tile: (image_array, distance_to_center)
    best_frames: List[Tuple[Optional[np.ndarray], float]] = [
        (None, float("inf")) for _ in range(n_tiles)
    ]

    # Cache image geometry for reshape
    try:
        img_w = int(core.get_image_width())
        img_h = int(core.get_image_height())
    except Exception:
        img_w = img_h = 0

    last_remaining = -1
    t0 = time.perf_counter()
    deadline = t0 + hard_deadline_s
    n_captured = 0

    while time.perf_counter() < deadline:
        # Check for new frames in the circular buffer
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
                # Read actual stage position at capture time
                try:
                    actual_x = core.get_x_position()
                    actual_y = core.get_y_position()
                except Exception:
                    actual_x = actual_y = float("nan")

                arr = np.asarray(pixels).copy()
                n_captured += 1

                # Assign to the closest tile position
                for i, (tx, ty, _) in enumerate(row_positions):
                    dist = math.sqrt((actual_x - tx) ** 2 + (actual_y - ty) ** 2)
                    if dist < best_frames[i][1]:
                        best_frames[i] = (arr, dist)

            last_remaining = remaining

        time.sleep(STREAM_POLL_SLEEP_S)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Reshape frames
    results = []
    for frame_data, dist in best_frames:
        if frame_data is not None and img_w > 0 and img_h > 0:
            try:
                nch = frame_data.size // (img_w * img_h)
                if nch > 1:
                    frame_data = frame_data.reshape(img_h, img_w, nch)
                else:
                    frame_data = frame_data.reshape(img_h, img_w)
            except Exception:
                pass
        results.append(frame_data)

    assigned = sum(1 for r in results if r is not None)
    logger.info(
        "  row grab: %d frames captured, %d/%d tiles assigned, %.0fms",
        n_captured, assigned, n_tiles, elapsed_ms,
    )
    return results


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
    into the circular buffer while the stage moves through each row.
    Frames are grabbed at positions closest to planned tile centers.

    No autofocus, no Z movement, brightfield only.

    Args:
        hardware: Hardware abstraction with .stage, .snap_image, .set_exposure,
                  .camera (for continuous acquisition), .core (MMCore)
        output_folder: Directory to save tiles and TileConfiguration.txt
        center_x, center_y: Center of scan region (stage um)
        width, height: Scan region size (um)
        overlap_percent: Tile overlap (0-50%)
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

    # Build serpentine grid grouped by row
    rows, n_cols, n_rows, step_x, step_y = _build_serpentine_grid(
        center_x, center_y, width, height,
        fov_width, fov_height, overlap_percent,
    )

    # Flatten for TileConfiguration and counting
    all_positions = [pos for row in rows for pos in row]
    n_tiles = len(all_positions)

    logger.info(
        "Rapid scan (streaming): %dx%d grid = %d tiles, "
        "step=(%.1f, %.1f) um, exposure=%.3f ms, overlap=%.1f%%",
        n_cols, n_rows, n_tiles, step_x, step_y, exposure_ms, overlap_percent,
    )
    logger.info(
        "Rapid scan region: center=(%.1f, %.1f), size=%.1fx%.1f um, "
        "FOV=%.1fx%.1f um",
        center_x, center_y, width, height, fov_width, fov_height,
    )
    for i, (px, py, fn) in enumerate(all_positions[:min(6, n_tiles)]):
        logger.info("  tile %d: (%.1f, %.1f) -> %s", i, px, py, fn)

    # Write TileConfiguration.txt
    config_path = output_path / "TileConfiguration.txt"
    with open(config_path, "w") as f:
        f.write("dim = 2\n")
        for x, y, filename in all_positions:
            f.write(f"{filename}; ; ({x:.3f}, {y:.3f})\n")
    logger.info("Wrote TileConfiguration.txt with %d entries", n_tiles)

    # ---- Streaming acquisition ----
    core = hardware.core
    saved_files = []
    start_time = time.time()

    if progress_dict is not None:
        progress_dict["total_tiles"] = n_tiles
        progress_dict["completed_tiles"] = 0
        progress_dict["status"] = "scanning"

    # Start continuous acquisition (circular buffer)
    core.clear_circular_buffer()
    core.start_continuous_sequence_acquisition(0)
    logger.info("Started continuous acquisition for streaming scan")
    time.sleep(0.1)  # let first frames arrive

    tiles_saved = 0
    try:
        for row_idx, row_positions in enumerate(rows):
            if not row_positions:
                continue

            # Row endpoints
            first_x, first_y = row_positions[0][0], row_positions[0][1]
            last_x, last_y = row_positions[-1][0], row_positions[-1][1]
            row_distance = math.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)

            # Move to the row start (blocking) -- need to be positioned
            # before streaming across the row
            hardware.stage.move_xy(first_x, first_y)

            # Flush stale frames from the buffer before this row
            try:
                core.clear_circular_buffer()
            except Exception:
                pass
            time.sleep(0.02)

            # Fire non-blocking move to the row end
            hardware.stage.move_xy_no_wait(last_x, last_y)

            # Estimate how long the move will take.
            # Prior stage at full speed: ~15-20 mm/s = 15000-20000 um/s
            # Conservative: 10000 um/s
            est_velocity = 10000.0  # um/s, conservative
            est_duration_s = row_distance / max(est_velocity, 1.0)
            hard_deadline_s = max(est_duration_s * 3.0, 2.0)

            logger.info(
                "Row %d/%d: streaming from (%.1f, %.1f) to (%.1f, %.1f), "
                "%.0f um, est %.1fs, deadline %.1fs, %d tiles",
                row_idx + 1, n_rows, first_x, first_y, last_x, last_y,
                row_distance, est_duration_s, hard_deadline_s, len(row_positions),
            )

            # Grab frames while stage moves
            frames = _grab_frames_during_move(
                core, row_positions, step_x, hard_deadline_s,
            )

            # Wait for stage to finish (should already be done)
            try:
                hardware.stage.wait_xy()
            except Exception:
                pass

            # Save tiles for this row
            for i, (x, y, filename) in enumerate(row_positions):
                frame = frames[i] if i < len(frames) else None
                if frame is not None:
                    filepath = output_path / filename
                    tifffile.imwrite(str(filepath), frame)
                    saved_files.append(str(filepath))
                    tiles_saved += 1
                else:
                    logger.warning("No frame captured for tile %s at (%.1f, %.1f)", filename, x, y)

            if progress_dict is not None:
                progress_dict["completed_tiles"] = tiles_saved

    finally:
        # Always stop continuous acquisition
        try:
            core.stop_sequence_acquisition()
        except Exception:
            pass
        try:
            core.clear_circular_buffer()
        except Exception:
            pass
        logger.info("Stopped continuous acquisition")

    elapsed = time.time() - start_time
    if progress_dict is not None:
        progress_dict["status"] = "complete"

    logger.info(
        "Rapid scan complete: %d/%d tiles saved, %.1fs (%.2fs/tile)",
        len(saved_files), n_tiles, elapsed,
        elapsed / n_tiles if n_tiles > 0 else 0,
    )

    return {
        "n_tiles": n_tiles,
        "saved": len(saved_files),
        "output_folder": str(output_path),
        "elapsed_seconds": elapsed,
        "tile_config_path": str(config_path),
    }
