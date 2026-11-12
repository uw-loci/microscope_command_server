"""A Mean (or Sum) time projection of PPM takes birefringence from the averaged angles.

Birefringence is a magnitude, so averaging each timepoint's keeps its noise as a positive
floor; averaging the +7/-7 tiles first lets the noise cancel. Both ways were compared on
four PPM 40x runs (same structure, about 18% lower background floor angle-first), and
angle-first was kept. Max, Min and Std still reduce each timepoint's birefringence.
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
def test_mean_and_sum_biref_come_from_the_averaged_angles(tmp_path, projection):
    ctx = _ctx(tmp_path, projection)
    _write(ctx, np.random.default_rng(7))
    # What averaging each timepoint's biref would give, for comparison.
    per_timepoint = [
        tifffile.imread(t / "0.tif").astype(float)
        for t in sorted((ctx.output_path / "7.0.biref").glob("t0*"))
    ]
    averaged_biref = np.mean(per_timepoint, axis=0)

    workflow._project_timepoints(ctx)

    biref = tifffile.imread(ctx.output_path / "7.0.biref" / "0.tif").astype(float)
    # The noise floor: averaging the angles first leaves well under half of it.
    assert biref.mean() < 0.5 * averaged_biref.mean()
    assert (ctx.output_path / "7.0.biref" / "TileConfiguration.txt").exists()
    # The timepoints are reduced as before, the biref's included, and nothing else is written.
    for group in ("7.0", "-7.0", "7.0.biref"):
        assert not any(p.name.startswith("t0") for p in (ctx.output_path / group).iterdir())
    assert sorted(p.name for p in ctx.output_path.iterdir()) == ["-7.0", "7.0", "7.0.biref"]


def test_max_reduces_the_per_timepoint_biref(tmp_path):
    ctx = _ctx(tmp_path, "max")
    _write(ctx, np.random.default_rng(7))
    expected = np.max(
        [tifffile.imread(t / "0.tif") for t in sorted((ctx.output_path / "7.0.biref").glob("t0*"))],
        axis=0,
    )
    workflow._project_timepoints(ctx)
    assert np.array_equal(tifffile.imread(ctx.output_path / "7.0.biref" / "0.tif"), expected)
