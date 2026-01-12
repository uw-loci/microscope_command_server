"""
QuPath project path management utilities.

This module contains the QuPathProject class for managing
QuPath project structure and file paths.
"""

import pathlib
import uuid
import logging

logger = logging.getLogger(__name__)


class QuPathProject:
    """Class for managing QuPath project structure and file paths."""

    def __init__(
        self,
        projectsFolderPath: str = r"C:\Users\lociuser\Codes\MikeN\data\slides",
        sampleLabel: str = "2024_04_09_4",
        scan_type: str = "20x_bf_2",
        region: str = "1479_4696",
        tile_config: str = "TileConfiguration.txt",
    ):
        """
        Initialize QuPath project paths.

        Args:
            projectsFolderPath: Base path for all projects
            sampleLabel: Sample identifier
            scan_type: Type of scan performed
            region: Region identifier
            tile_config: Name of tile configuration file
        """
        self.path_tile_configuration = pathlib.Path(
            projectsFolderPath, sampleLabel, scan_type, region, tile_config
        )
        if self.path_tile_configuration.exists():
            self.path_qp_project = pathlib.Path(projectsFolderPath, sampleLabel)
            self.path_output = pathlib.Path(projectsFolderPath, sampleLabel, scan_type, region)
            self.acq_id = sampleLabel + "_ST_" + scan_type
        else:
            self.path_qp_project = pathlib.Path("undefined")
            self.path_output = pathlib.Path("undefined")
            self.acq_id = "undefined" + "_ScanType_" + "undefined"
            logger.warning(f"Tile configuration not found: {self.path_tile_configuration}")

    @staticmethod
    def uid() -> str:
        """Generate a unique identifier."""
        return uuid.uuid1().urn[9:]

    def __repr__(self):
        return (
            f"QuPath project: {self.path_qp_project}\n"
            f"TIF files: {self.path_output}\n"
            f"Acquisition ID: {self.acq_id}"
        )
