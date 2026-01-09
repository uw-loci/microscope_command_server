"""
Shared pytest fixtures for microscope_command_server tests.

Provides sample data for protocol, tile configuration, and server testing.
"""

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def sample_tile_configuration_txt():
    """
    Sample TileConfiguration.txt content for parsing tests.

    Returns:
        str: Sample TileConfiguration.txt format
    """
    return """# Define the number of dimensions we are working on
dim = 2

# Define the image coordinates
tile_0.tif; ; (0.0, 0.0)
tile_1.tif; ; (512.5, 0.0)
tile_2.tif; ; (1025.0, 0.0)
tile_3.tif; ; (0.0, 512.5)
tile_4.tif; ; (512.5, 512.5)
tile_5.tif; ; (1025.0, 512.5)
"""


@pytest.fixture
def sample_tile_positions():
    """
    Sample tile positions for TileConfiguration generation testing.

    Returns:
        list: List of (x, y) tuples representing tile positions
    """
    return [
        (0.0, 0.0),
        (512.5, 0.0),
        (1025.0, 0.0),
        (0.0, 512.5),
        (512.5, 512.5),
        (1025.0, 512.5)
    ]


@pytest.fixture
def sample_tile_positions_3d():
    """
    Sample 3D tile positions for stage TileConfiguration testing.

    Returns:
        list: List of (x, y, z) tuples representing stage positions
    """
    return [
        (1000.0, 2000.0, 5000.0),
        (1512.5, 2000.0, 5000.0),
        (2025.0, 2000.0, 5000.0),
        (1000.0, 2512.5, 5000.0),
        (1512.5, 2512.5, 5000.0),
        (2025.0, 2512.5, 5000.0)
    ]


@pytest.fixture
def temp_output_directory():
    """
    Create a temporary directory for test output files.

    Yields:
        Path: Temporary directory path
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_protocol_command_bytes():
    """
    Sample protocol command bytes for testing.

    Returns:
        dict: Dictionary mapping command names to byte values
    """
    return {
        'GETXY': b'GETXY___',
        'GETZ': b'GETZ____',
        'GETR': b'GETR____',
        'MOVEXY': b'MOVEXY__',
        'MOVEZ': b'MOVEZ___',
        'MOVER': b'MOVER___',
        'TESTAF': b'TESTAF__',
        'PPMSENS': b'PPMSENS_',
        'PPMBIREF': b'PPMBIREF'
    }


@pytest.fixture
def sample_microscope_config():
    """
    Sample microscope configuration for testing server functionality.

    Returns:
        dict: Microscope configuration dictionary
    """
    return {
        'microscope': {
            'name': 'TestMicroscope',
            'objectives': {
                '10x': {
                    'magnification': 10,
                    'pixel_size_um': 0.65
                },
                '20x': {
                    'magnification': 20,
                    'pixel_size_um': 0.325
                }
            },
            'stage': {
                'limits': {
                    'x': {'min': 0.0, 'max': 100000.0},
                    'y': {'min': 0.0, 'max': 75000.0},
                    'z': {'min': 0.0, 'max': 10000.0}
                }
            }
        }
    }


@pytest.fixture
def sample_acquisition_params():
    """
    Sample acquisition parameters for workflow testing.

    Returns:
        dict: Acquisition parameter dictionary
    """
    return {
        'objective': '20x',
        'modality': 'ppm_20x',
        'exposure_ms': 100,
        'overlap_percent': 10,
        'z_stack': False,
        'tile_width_um': 664.0,
        'tile_height_um': 558.0
    }
