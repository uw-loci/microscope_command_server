"""A Mean (or Sum) time projection of PPM writes two birefringence images to compare.

``7.0.biref/`` is the average of the per-timepoint birefringence, and
``7.0.biref_from_angles/`` is the birefringence of the time-averaged +7/-7 tiles. The
birefringence is a magnitude, so per-frame noise survives the first as a positive floor
and largely cancels in the second (requested 2026-11-11, to compare both).
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ppm_library")
tifffile = pytest.importorskip("tifffile")

from ppm_library.imaging.writer import TifWriterUtils  # noqa: E402

from microscope_command_server.acquisition import workflow  # noqa: E402


def _ctx(tmp_path, projection):
    out = tmp_path / "s" / "ppm_20x_1" / "bounds"
    out.mkdir(parents=True)
    return SimpleNamespace(
        output_path=out,
        n_timepoints=8,
        current_timepoint=0,
        preserve_z_planes=False,
        t_projection=projection,
        save_raw_tiles=False,
        params={"angles": [7.0, -7.0, 0.0, 90.0], "biref_min_intensity": 0},
        ppm_high_bit_input_max=None,
        hardware=SimpleNamespace(get_pixel_size_um=lambda: 0.5),
        logger=logging.getLogger("test"),
    )


def _write(ctx, rng):
    for angle in ("7.0", "-7.0"):
        (ctx.output_path / angle).mkdir()
        (ctx.output_path / angle / "TileConfiguration.txt").write_text("0.tif; ; (0, 0)\n")
    for t in range(ctx.n_timepoints):
        ctx.current_timepoint = t
        # No birefringence: both angles are 120 plus independent noise.
        pos = np.clip(120 + rng.normal(0, 6, (32, 32, 3)), 0, 255).astype(np.uint8)
        neg = np.clip(120 + rng.normal(0, 6, (32, 32, 3)), 0, 255).astype(np.uint8)
        for angle, img in (("7.0", pos), ("-7.0", neg)):
            workflow.ome_tiff_writer(
                filename=str(workflow._tile_dir(ctx, angle) / "0.tif"), pixel_size_um=0.5, data=img
            )
        TifWriterUtils.create_normalized_birefringence_tile(
            pos_image=pos,
            neg_image=neg,
            output_dir=workflow._tile_dir(ctx, "7.0.biref"),
            filename="0.tif",
            pixel_size_um=0.5,
        )


@pytest.mark.parametrize("projection", ["mean", "sum"])
def test_both_birefringence_images_are_written(tmp_path, projection):
    ctx = _ctx(tmp_path, projection)
    _write(ctx, np.random.default_rng(7))
    workflow._project_timepoints(ctx)

    averaged_biref = tifffile.imread(ctx.output_path / "7.0.biref" / "0.tif").astype(float)
    from_angles = tifffile.imread(ctx.output_path / "7.0.biref_from_angles" / "0.tif").astype(float)
    assert (ctx.output_path / "7.0.biref_from_angles" / "TileConfiguration.txt").exists()
    if projection == "mean":
        # The noise floor: averaging the angles first leaves well under half of it.
        assert from_angles.mean() < 0.5 * averaged_biref.mean()
    # The timepoints are reduced as before.
    assert not any(p.name.startswith("t0") for p in (ctx.output_path / "7.0").iterdir())


def test_max_keeps_only_the_per_timepoint_biref(tmp_path):
    ctx = _ctx(tmp_path, "max")
    _write(ctx, np.random.default_rng(7))
    workflow._project_timepoints(ctx)
    assert (ctx.output_path / "7.0.biref" / "0.tif").exists()
    assert not (ctx.output_path / "7.0.biref_from_angles").exists()
