"""Rapid scan acquisition -- fast tiled brightfield, no AF, no Z.

Traces a serpentine (boustrophedon) path through a rectangular region,
snapping one image per tile position with minimal overhead. No autofocus,
no Z movement, no background correction, no debayering.

This is a demonstration/prototype for streaming XY acquisition. Future
versions may use true continuous motion with circular-buffer frame grabbing.
"""

import logging
import math
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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
    """Fast tiled acquisition over a rectangular region.

    No autofocus, no Z movement, brightfield only.
    Serpentine path for minimal stage travel.

    Args:
        hardware: Hardware abstraction (must have move_xy, snap_image, set_exposure)
        output_folder: Directory to save tiles and TileConfiguration.txt
        center_x: Center X of scan region (stage um)
        center_y: Center Y of scan region (stage um)
        width: Width of scan region (um)
        height: Height of scan region (um)
        overlap_percent: Overlap between tiles (0-50%)
        exposure_ms: Exposure time (max 0.5ms)
        fov_width: Camera FOV width (um)
        fov_height: Camera FOV height (um)
        progress_dict: Optional dict to update with progress info

    Returns:
        dict with keys: n_tiles, saved, elapsed_seconds, output_folder,
                        tile_config_path
    """
    import tifffile

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate exposure cap
    if exposure_ms > 0.5:
        raise ValueError(
            f"Exposure {exposure_ms}ms exceeds 0.5ms limit for rapid scan"
        )

    # Set camera exposure
    logger.info("Setting exposure to %.3f ms", exposure_ms)
    hardware.set_exposure(exposure_ms)

    # Compute tile grid
    step_x = fov_width * (1.0 - overlap_percent / 100.0)
    step_y = fov_height * (1.0 - overlap_percent / 100.0)
    n_cols = max(1, math.ceil(width / step_x))
    n_rows = max(1, math.ceil(height / step_y))

    # Center the grid on the specified center point
    start_x = center_x - (n_cols - 1) * step_x / 2.0
    start_y = center_y - (n_rows - 1) * step_y / 2.0

    # Build serpentine position list
    positions = []  # List of (x, y, filename)
    tile_idx = 0
    for row in range(n_rows):
        if row % 2 == 0:
            col_range = range(n_cols)
        else:
            col_range = range(n_cols - 1, -1, -1)
        for col in col_range:
            x = start_x + col * step_x
            y = start_y + row * step_y
            positions.append((x, y, f"{tile_idx}.tif"))
            tile_idx += 1

    n_tiles = len(positions)
    logger.info(
        "Rapid scan: %dx%d = %d tiles, step=(%.1f, %.1f) um, "
        "exposure=%.3f ms, overlap=%.1f%%",
        n_cols, n_rows, n_tiles, step_x, step_y, exposure_ms, overlap_percent,
    )
    logger.info(
        "Rapid scan region: center=(%.1f, %.1f), size=%.1fx%.1f um",
        center_x, center_y, width, height,
    )

    # Write TileConfiguration.txt (stage coordinates in microns)
    config_path = output_path / "TileConfiguration.txt"
    with open(config_path, "w") as f:
        f.write("dim = 2\n")
        for x, y, filename in positions:
            f.write(f"{filename}; ; ({x:.3f}, {y:.3f})\n")
    logger.info("Wrote TileConfiguration.txt with %d entries", n_tiles)

    # Acquisition loop
    saved_files = []
    start_time = time.time()

    if progress_dict is not None:
        progress_dict["total_tiles"] = n_tiles
        progress_dict["completed_tiles"] = 0
        progress_dict["status"] = "scanning"

    for idx, (x, y, filename) in enumerate(positions):
        # Move stage (blocking)
        hardware.stage.move_xy(x, y)

        # Snap image
        image, _ = hardware.snap_image()

        if image is not None:
            filepath = output_path / filename
            tifffile.imwrite(str(filepath), image)
            saved_files.append(str(filepath))
        else:
            logger.warning(
                "Snap failed at tile %d (%.1f, %.1f)", idx, x, y
            )

        if progress_dict is not None:
            progress_dict["completed_tiles"] = idx + 1

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
