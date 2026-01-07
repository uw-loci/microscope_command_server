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
    def read_TileConfiguration_coordinates(tile_config_path) -> np.ndarray:
        """
        Read tile XY coordinates from a TileConfiguration.txt file.

        Args:
            tile_config_path: Path to TileConfiguration.txt file

        Returns:
            numpy array of XY coordinates
        """
        coordinates = []
        with open(tile_config_path, "r") as file:
            for line in file:
                # Extract coordinates using regular expression
                match = re.search(r"\((-?\d+\.\d+), (-?\d+\.\d+)\)", line)
                if match:
                    x, y = map(float, match.groups())
                    coordinates.append([x, y])
        return np.array(coordinates)

    @staticmethod
    def write_tileconfig(
        tileconfig_path: Optional[str] = None,
        target_foldername: Optional[str] = None,
        positions: Optional[list] = None,
        id1: str = "Tile",
        suffix_length: str = "06",
        pixel_size: float = 1.0,
    ):
        """
        Write a TileConfiguration.txt file.

        Args:
            tileconfig_path: Direct path to output file
            target_foldername: Folder to create file in
            positions: List of (x, y) positions
            id1: Prefix for tile names
            suffix_length: Number of digits for tile index
            pixel_size: Pixel size for scaling coordinates
        """
        if not tileconfig_path and target_foldername is not None:
            target_folder_path = pathlib.Path(target_foldername)
            tileconfig_path = str(target_folder_path / "TileConfiguration.txt")

        if tileconfig_path is not None and positions is not None:
            with open(tileconfig_path, "w") as text_file:
                print("dim = {}".format(2), file=text_file)
                for ix, pos in enumerate(positions):
                    file_id = f"{id1}_{ix:{suffix_length}}"
                    x, y = pos
                    print(
                        f"{file_id}.tif; ; ({x / pixel_size:.3f}, {y / pixel_size:.3f})",
                        file=text_file,
                    )

    @staticmethod
    def write_tileconfig_stage(
        output_path: pathlib.Path,
        tile_positions: List[Tuple[str, float, float, float]],
        filename: str = "TileConfiguration_Stage.txt",
    ):
        """
        Write a TileConfiguration file with actual stage coordinates including Z.

        This preserves the stage XYZ positions used during acquisition for
        later reference and analysis. The Z position is particularly useful
        for understanding focus variation across the sample.

        Args:
            output_path: Directory to write the file
            tile_positions: List of (filename, x, y, z) tuples with stage positions in micrometers
            filename: Name of output file (default: TileConfiguration_Stage.txt)
        """
        tileconfig_path = output_path / filename

        with open(tileconfig_path, "w") as f:
            # Use dim=3 to indicate 3D coordinates
            f.write("dim = 3\n")
            f.write("# Stage coordinates in micrometers (X, Y, Z)\n")

            for tile_filename, x, y, z in tile_positions:
                # Format: filename; ; (x, y, z)
                f.write(f"{tile_filename}; ; ({x:.3f}, {y:.3f}, {z:.3f})\n")

        logger.info(f"Wrote stage TileConfiguration with {len(tile_positions)} positions to {tileconfig_path}")
