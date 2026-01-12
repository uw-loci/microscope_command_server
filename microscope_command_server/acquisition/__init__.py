"""
Acquisition package - Workflows for microscope image acquisition.

This package contains the core acquisition workflow logic,
tile configuration utilities, and QuPath project management.

Modules:
    workflow: Main acquisition workflow orchestration
    tiles: Tile configuration utilities (TileConfigUtils)
    project: QuPath project management (QuPathProject)
    pipeline: Text-based image processing pipeline
"""

from microscope_command_server.acquisition.tiles import TileConfigUtils
from microscope_command_server.acquisition.project import QuPathProject

__all__ = ["TileConfigUtils", "QuPathProject"]
