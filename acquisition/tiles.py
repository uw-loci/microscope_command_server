"""
Tile configuration utilities for QuPath integration.

This module contains utilities for reading and writing tile configuration
files used in QuPath for image stitching.
"""

import pathlib
import re
from typing import List, Tuple, Optional
import numpy as np
import logging

from microscope_control.hardware import Position

logger = logging.getLogger(__name__)


class TileConfigUtils:
    """Utilities for reading and writing tile configuration files."""

    def __init__(self):
        pass

    @staticmethod
    def read_tile_config(tile_config_path: pathlib.Path, core) -> List[Tuple[Position, str]]:
        """
        Read tile positions + filename from a QuPath-generated TileConfiguration.txt file.

        Args:
            tile_config_path: Path to TileConfiguration.txt file
            core: Pycromanager Core object for getting Z position

        Returns:
            List of tuples containing (Position, filename)
        """
        positions: List[Tuple[Position, str]] = []
        if tile_config_path.exists():
            # Get Z position once (same for all tiles in initial configuration)
            z = core.get_position()

            with open(tile_config_path, "r") as f:
                for line in f:
                    pattern = r"^([\w\-\.]+); ; \(\s*([\-\d.]+),\s*([\-\d.]+)"
                    m = re.match(pattern, line)
                    if m:
                        filename = m.group(1)
                        x = float(m.group(2))
                        y = float(m.group(3))
                        # Use same Z for all tiles (avoids 100+ hardware calls)
                        positions.append((Position(x, y, z), filename))
        else:
            logger.warning(f"Tile config file not found: {tile_config_path}")
        return positions

    @staticmethod
    def read_TileConfiguration_coordinates(tile_config_path):
        """
        Read tile XY coordinates from a TileConfiguration.txt file.

        Args:
            tile_config_path: Path to TileConfiguration.txt file

        Returns:
            dict: Dictionary mapping tile names to (x, y) coordinate tuples
        """
        coordinates = {}
        with open(tile_config_path, "r") as file:
            for line in file:
                # Extract tile name and coordinates using regular expression
                # Format: "tile_name.tif; ; (x, y)"
                match = re.search(r"([\w\-\.]+\.tif)[^(]*\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", line)
                if match:
                    tile_name = match.group(1)
                    x = float(match.group(2))
                    y = float(match.group(3))
                    coordinates[tile_name] = (x, y)
        return coordinates

    @staticmethod
    def write_tileconfig(
        target_foldername: Optional[str] = None,
        positions: Optional[list] = None,
        filename: str = "TileConfiguration.txt",
        pixel_size_um: float = 1.0,
        id1: int = 0,
        suffix_length: int = 3,
        tileconfig_path: Optional[str] = None,
    ):
        """
        Write a TileConfiguration.txt file.

        Args:
            target_foldername: Folder to create file in
            positions: List of (x, y) positions in micrometers
            filename: Name of output file (default: TileConfiguration.txt)
            pixel_size_um: Pixel size in micrometers for scaling coordinates
            id1: Starting index for tile numbering
            suffix_length: Number of digits for tile index
            tileconfig_path: Direct path to output file (overrides target_foldername/filename)
        """
        # If direct path provided, use it; otherwise construct from folder + filename
        if tileconfig_path is None and target_foldername is not None:
            target_folder_path = pathlib.Path(target_foldername)
            tileconfig_path = str(target_folder_path / filename)

        if tileconfig_path is not None and positions is not None:
            # Auto-calculate suffix length if not enough for all tiles
            num_tiles = len(positions)
            last_tile_index = id1 + num_tiles - 1
            min_suffix_length = len(str(last_tile_index))
            actual_suffix_length = max(suffix_length, min_suffix_length)

            with open(tileconfig_path, "w") as text_file:
                print("dim = {}".format(2), file=text_file)
                for ix, pos in enumerate(positions):
                    tile_index = id1 + ix
                    file_id = f"tile_{tile_index:0{actual_suffix_length}d}"
                    x, y = pos
                    print(
                        f"{file_id}.tif; ; ({x / pixel_size_um:.1f}, {y / pixel_size_um:.1f})",
                        file=text_file,
                    )

    @staticmethod
    def write_tileconfig_stage(
        output_path,
        tile_positions,
        filename: str = "TileConfiguration_Stage.txt",
    ):
        """
        Write a TileConfiguration file with actual stage coordinates including Z.

        This preserves the stage XYZ positions used during acquisition for
        later reference and analysis. The Z position is particularly useful
        for understanding focus variation across the sample.

        Args:
            output_path: Directory to write the file (Path or str)
            tile_positions: List of (x, y, z) or (filename, x, y, z) tuples with stage positions
            filename: Name of output file (default: TileConfiguration_Stage.txt)
        """
        # Convert to Path if string
        output_path = pathlib.Path(output_path) if isinstance(output_path, str) else output_path
        tileconfig_path = output_path / filename

        with open(tileconfig_path, "w") as f:
            # Use dim=3 to indicate 3D coordinates
            f.write("dim = 3\n")
            f.write("# Stage coordinates in micrometers (X, Y, Z)\n")

            for idx, pos in enumerate(tile_positions):
                # Check if position includes filename or just coordinates
                if len(pos) == 4:
                    # Format: (filename, x, y, z)
                    tile_filename, x, y, z = pos
                elif len(pos) == 3:
                    # Format: (x, y, z) - auto-generate filename
                    x, y, z = pos
                    tile_filename = f"tile_{idx:03d}.tif"
                else:
                    raise ValueError(f"Invalid position format: {pos}. Expected (x,y,z) or (filename,x,y,z)")

                # Format: filename; ; (x, y, z)
                f.write(f"{tile_filename}; ; ({x:.3f}, {y:.3f}, {z:.3f})\n")

        logger.info(f"Wrote stage TileConfiguration with {len(tile_positions)} positions to {tileconfig_path}")
