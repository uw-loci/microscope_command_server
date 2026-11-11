"""A PPM Z-stack produces birefringence.

Tiled acquisition with z_projection none keeps every plane for a 5D mosaic; no
per-angle image exists for the tile, so the projected-image birefringence never ran.
The single-point stack command wrote the per-angle files only. Neither produced any
birefringence for a PPM Z-stack (reported 2026-11-11).
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ppm_library")
tifffile = pytest.importorskip("tifffile")

from microscope_command_server.acquisition import stack_timelapse, workflow  # noqa: E402


def _plane(value):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[...] = value
    return img


class _InlinePool:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


def _ctx(tmp_path):
    return SimpleNamespace(
        output_path=tmp_path,
        preserve_z_planes=True,
        n_timepoints=1,
        current_timepoint=0,
        write_pool=_InlinePool(),
        hardware=SimpleNamespace(get_pixel_size_um=lambda: 0.5),
        logger=None,
        ppm_high_bit_input_max=None,
    )


def test_every_kept_plane_gets_a_biref_plane(tmp_path):
    (tmp_path / "7.0").mkdir()
    (tmp_path / "7.0" / "TileConfiguration.txt").write_text("tile_0.tif; ; (0, 0)\n")
    z_stack_images = {
        7.0: [_plane(200), _plane(150), _plane(100)],
        -7.0: [_plane(100), _plane(150), _plane(200)],
        0.0: [_plane(20)] * 3,
        90.0: [_plane(250)] * 3,
    }

    made = workflow._submit_biref_planes(_ctx(tmp_path), z_stack_images, "tile_0.tif", {})

    assert made
    root = tmp_path / "7.0.biref"
    assert (root / "TileConfiguration.txt").exists()
    planes = [tifffile.imread(root / f"z{z:03d}" / "tile_0.tif") for z in range(3)]
    assert planes[0].max() > 0 and planes[2].max() > 0
    assert planes[1].max() == 0, "equal intensities at +7 and -7: no birefringence"


def test_without_a_negative_angle_there_is_no_biref(tmp_path):
    z_stack_images = {7.0: [_plane(200)], 90.0: [_plane(250)]}
    assert not workflow._submit_biref_planes(_ctx(tmp_path), z_stack_images, "t.tif", {})
    assert not (tmp_path / "7.0.biref").exists()


def test_the_stack_command_writes_a_biref_stack(tmp_path):
    angle_frames = {
        7.0: {(0, z): (float(z), _plane(200 - 50 * z), 0.0) for z in range(3)},
        -7.0: {(0, z): (float(z), _plane(100 + 50 * z), 0.0) for z in range(3)},
        90.0: {(0, z): (float(z), _plane(250), 0.0) for z in range(3)},
    }

    out = stack_timelapse._write_biref_stack(
        angle_frames,
        tmp_path,
        has_t=False,
        size_t=1,
        size_z=3,
        pixel_size_um=0.5,
        z_step_um=1.0,
        time_increment_s=None,
        logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    assert out == tmp_path / "zstack_biref.ome.tiff"
    stack = tifffile.imread(out)
    assert stack.shape[-2:] == (16, 16)
    assert stack.reshape(3, 16, 16)[1].max() == 0


def test_the_stack_command_skips_biref_without_a_pair(tmp_path):
    angle_frames = {0.0: {(0, 0): (0.0, _plane(100), 0.0)}}
    out = stack_timelapse._write_biref_stack(
        angle_frames, tmp_path, False, 1, 1, 0.5, 1.0, None, logger=None
    )
    assert out is None
