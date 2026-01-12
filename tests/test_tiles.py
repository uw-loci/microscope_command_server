"""
Unit tests for TileConfiguration utilities.

Tests parsing and generation of TileConfiguration.txt files used by
Fiji/ImageJ for image stitching.
"""

import numpy as np
import pytest
from pathlib import Path
from microscope_command_server.acquisition.tiles import TileConfigUtils


class TestReadTileConfigurationCoordinates:
    """Test parsing of TileConfiguration.txt files."""

    def test_parse_standard_tileconfiguration(
        self, sample_tile_configuration_txt, temp_output_directory
    ):
        """Test parsing a standard TileConfiguration.txt file."""
        # Write sample content to temp file
        tilecfg_path = temp_output_directory / "TileConfiguration.txt"
        tilecfg_path.write_text(sample_tile_configuration_txt)

        # Parse the file
        coordinates = TileConfigUtils.read_TileConfiguration_coordinates(str(tilecfg_path))

        # Should return dictionary mapping tile names to coordinates
        assert isinstance(coordinates, dict)
        assert len(coordinates) == 6  # 6 tiles in sample

        # Check specific tile coordinates
        assert "tile_0.tif" in coordinates
        assert coordinates["tile_0.tif"] == (0.0, 0.0)

        assert "tile_1.tif" in coordinates
        assert coordinates["tile_1.tif"] == (512.5, 0.0)

        assert "tile_5.tif" in coordinates
        assert coordinates["tile_5.tif"] == (1025.0, 512.5)

    def test_parse_tileconfiguration_with_negative_coordinates(self, temp_output_directory):
        """Test parsing TileConfiguration with negative coordinates."""
        content = """# Define the number of dimensions we are working on
dim = 2

# Define the image coordinates
tile_0.tif; ; (-100.5, -200.0)
tile_1.tif; ; (100.0, -50.5)
"""
        tilecfg_path = temp_output_directory / "TileConfiguration.txt"
        tilecfg_path.write_text(content)

        coordinates = TileConfigUtils.read_TileConfiguration_coordinates(str(tilecfg_path))

        assert coordinates["tile_0.tif"] == (-100.5, -200.0)
        assert coordinates["tile_1.tif"] == (100.0, -50.5)

    def test_parse_tileconfiguration_empty_file(self, temp_output_directory):
        """Test parsing empty TileConfiguration file."""
        tilecfg_path = temp_output_directory / "TileConfiguration_empty.txt"
        tilecfg_path.write_text("# Only comments\ndim = 2\n")

        coordinates = TileConfigUtils.read_TileConfiguration_coordinates(str(tilecfg_path))

        # Should return empty dict
        assert isinstance(coordinates, dict)
        assert len(coordinates) == 0

    def test_parse_tileconfiguration_malformed_line(self, temp_output_directory):
        """Test handling of malformed coordinate lines."""
        content = """dim = 2
tile_0.tif; ; (0.0, 0.0)
tile_1_bad_line
tile_2.tif; ; (100.0, 100.0)
"""
        tilecfg_path = temp_output_directory / "TileConfiguration.txt"
        tilecfg_path.write_text(content)

        # Should skip malformed line and parse valid ones
        coordinates = TileConfigUtils.read_TileConfiguration_coordinates(str(tilecfg_path))

        assert "tile_0.tif" in coordinates
        assert "tile_2.tif" in coordinates
        assert "tile_1_bad_line" not in coordinates


class TestWriteTileconfig:
    """Test generation of TileConfiguration.txt files."""

    def test_write_basic_tileconfiguration(
        self, sample_tile_positions, temp_output_directory
    ):
        """Test writing basic 2D TileConfiguration file."""
        output_path = temp_output_directory / "output_tiles.txt"

        TileConfigUtils.write_tileconfig(
            str(temp_output_directory),
            sample_tile_positions,
            filename="output_tiles.txt",
            pixel_size_um=1.0
        )

        # Verify file was created
        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        assert "dim = 2" in content
        assert "tile_000.tif" in content
        assert "(0.0, 0.0)" in content

    def test_write_tileconfiguration_with_scaling(
        self, sample_tile_positions, temp_output_directory
    ):
        """Test that pixel_size_um scales coordinates correctly."""
        output_path = temp_output_directory / "scaled_tiles.txt"

        pixel_size = 0.5  # um/pixel

        TileConfigUtils.write_tileconfig(
            str(temp_output_directory),
            sample_tile_positions,
            filename="scaled_tiles.txt",
            pixel_size_um=pixel_size
        )

        # Read back and check coordinates are scaled
        content = output_path.read_text()

        # First position (0, 0) -> (0, 0) pixels
        assert "(0.0, 0.0)" in content

        # Second position (512.5, 0) um -> (1025.0, 0) pixels at 0.5 um/pixel
        assert "(1025.0, 0.0)" in content

    def test_write_tileconfiguration_custom_id(
        self, sample_tile_positions, temp_output_directory
    ):
        """Test writing TileConfiguration with custom id1 parameter."""
        output_path = temp_output_directory / "custom_id_tiles.txt"

        TileConfigUtils.write_tileconfig(
            str(temp_output_directory),
            sample_tile_positions,
            filename="custom_id_tiles.txt",
            pixel_size_um=1.0,
            id1=5,  # Start numbering from 5
            suffix_length=4
        )

        content = output_path.read_text()

        # Should have tile_0005, tile_0006, etc.
        assert "tile_0005.tif" in content
        assert "tile_0006.tif" in content

    def test_write_tileconfiguration_suffix_length(
        self, sample_tile_positions, temp_output_directory
    ):
        """Test writing TileConfiguration with custom suffix length."""
        output_path = temp_output_directory / "suffix_tiles.txt"

        TileConfigUtils.write_tileconfig(
            str(temp_output_directory),
            sample_tile_positions,
            filename="suffix_tiles.txt",
            pixel_size_um=1.0,
            id1=0,
            suffix_length=5  # 5-digit numbering
        )

        content = output_path.read_text()

        # Should have 5-digit tile IDs
        assert "tile_00000.tif" in content
        assert "tile_00001.tif" in content

    def test_write_tileconfiguration_empty_positions(self, temp_output_directory):
        """Test writing TileConfiguration with empty position list."""
        output_path = temp_output_directory / "empty_tiles.txt"

        TileConfigUtils.write_tileconfig(
            str(temp_output_directory),
            [],  # Empty list
            filename="empty_tiles.txt",
            pixel_size_um=1.0
        )

        content = output_path.read_text()

        # Should have dim = 2 but no tile entries
        assert "dim = 2" in content
        assert "tile_000.tif" not in content


class TestWriteTileconfigStage:
    """Test generation of stage coordinate TileConfiguration files (3D)."""

    def test_write_stage_tileconfiguration(
        self, sample_tile_positions_3d, temp_output_directory
    ):
        """Test writing 3D TileConfiguration with stage coordinates."""
        output_path = temp_output_directory / "stage_tiles.txt"

        TileConfigUtils.write_tileconfig_stage(
            str(temp_output_directory),
            sample_tile_positions_3d,
            filename="stage_tiles.txt"
        )

        # Verify file was created
        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        assert "dim = 3" in content  # 3D for stage coordinates
        assert "tile_000.tif" in content

        # Check that Z coordinates are present
        assert "5000.000" in content

    def test_write_stage_coordinate_precision(
        self, sample_tile_positions_3d, temp_output_directory
    ):
        """Test that stage coordinates are formatted with correct precision."""
        output_path = temp_output_directory / "precision_tiles.txt"

        TileConfigUtils.write_tileconfig_stage(
            str(temp_output_directory),
            sample_tile_positions_3d,
            filename="precision_tiles.txt"
        )

        content = output_path.read_text()

        # Should have 3 decimal places
        # First position: (1000.0, 2000.0, 5000.0)
        assert "(1000.000, 2000.000, 5000.000)" in content

    def test_write_stage_tileconfiguration_negative_coords(self, temp_output_directory):
        """Test stage TileConfiguration with negative coordinates."""
        positions_3d = [
            (-500.0, -1000.0, 5000.0),
            (500.0, -1000.0, 5000.0),
        ]

        output_path = temp_output_directory / "negative_stage_tiles.txt"

        TileConfigUtils.write_tileconfig_stage(
            str(temp_output_directory),
            positions_3d,
            filename="negative_stage_tiles.txt"
        )

        content = output_path.read_text()

        # Should handle negative coordinates
        assert "(-500.000, -1000.000, 5000.000)" in content
        assert "(500.000, -1000.000, 5000.000)" in content


class TestTileConfigurationRoundTrip:
    """Test round-trip conversion (write then read)."""

    def test_roundtrip_2d_tileconfiguration(
        self, sample_tile_positions, temp_output_directory
    ):
        """Test that writing and reading back preserves coordinates."""
        output_path = temp_output_directory / "roundtrip_tiles.txt"

        # Write
        TileConfigUtils.write_tileconfig(
            str(temp_output_directory),
            sample_tile_positions,
            filename="roundtrip_tiles.txt",
            pixel_size_um=1.0  # No scaling
        )

        # Read back
        coordinates = TileConfigUtils.read_TileConfiguration_coordinates(str(output_path))

        # Check coordinates match
        for i, (x, y) in enumerate(sample_tile_positions):
            tile_name = f"tile_{i:03d}.tif"
            assert tile_name in coordinates
            read_x, read_y = coordinates[tile_name]
            assert abs(read_x - x) < 0.01  # Allow tiny floating-point error
            assert abs(read_y - y) < 0.01


class TestTileConfigurationEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file(self):
        """Test reading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            TileConfigUtils.read_TileConfiguration_coordinates("/nonexistent/path.txt")

    def test_write_to_readonly_directory(self):
        """Test writing to read-only directory (if applicable)."""
        # Skip test if can't create read-only directory
        pytest.skip("Read-only directory test not implemented")

    def test_single_tile_configuration(self, temp_output_directory):
        """Test TileConfiguration with single tile."""
        positions = [(100.0, 200.0)]

        output_path = temp_output_directory / "single_tile.txt"

        TileConfigUtils.write_tileconfig(
            str(temp_output_directory),
            positions,
            filename="single_tile.txt",
            pixel_size_um=1.0
        )

        content = output_path.read_text()

        assert "tile_000.tif" in content
        assert "(100.0, 200.0)" in content

    def test_large_grid_configuration(self, temp_output_directory):
        """Test TileConfiguration with large grid (performance test)."""
        # Create 100x100 grid
        positions = [
            (float(x * 500), float(y * 500))
            for y in range(100)
            for x in range(100)
        ]

        output_path = temp_output_directory / "large_grid.txt"

        TileConfigUtils.write_tileconfig(
            str(temp_output_directory),
            positions,
            filename="large_grid.txt",
            pixel_size_um=1.0
        )

        # Should handle 10,000 tiles (indices 0-9999 need 4 digits)
        content = output_path.read_text()
        assert "tile_9999.tif" in content

    def test_very_large_coordinates(self, temp_output_directory):
        """Test TileConfiguration with very large coordinate values."""
        positions = [
            (1000000.0, 2000000.0),
            (1000500.0, 2000000.0),
        ]

        output_path = temp_output_directory / "large_coords.txt"

        TileConfigUtils.write_tileconfig(
            str(temp_output_directory),
            positions,
            filename="large_coords.txt",
            pixel_size_um=1.0
        )

        content = output_path.read_text()

        # Should handle large numbers
        assert "1000000.0" in content or "1e+06" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
