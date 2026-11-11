"""A time-lapse keeps every timepoint, and can reduce them like a Z projection.

2026-11-11: a 30-timepoint run of one position wrote 0.tif to the same place every
timepoint and left one tile. Each timepoint now goes to ``{group}/t{tt}/``; a time
projection (--t-projection mean|max|min|sum|std) reduces them to ``{group}/{tile}``.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from microscope_command_server.acquisition import workflow  # noqa: E402


def _ctx(tmp_path, n_timepoints=3, t_projection=None, save_raw=False):
    out = tmp_path / "sample" / "ppm_20x_1" / "bounds"
    out.mkdir(parents=True)
    return SimpleNamespace(
        output_path=out,
        n_timepoints=n_timepoints,
        current_timepoint=0,
        preserve_z_planes=False,
        t_projection=t_projection,
        save_raw_tiles=save_raw,
        hardware=SimpleNamespace(get_pixel_size_um=lambda: 0.5),
        logger=logging.getLogger("test"),
    )


def _write_timepoints(ctx, group, values):
    for t, value in enumerate(values):
        ctx.current_timepoint = t
        path = workflow._tile_dir(ctx, group) / "0.tif"
        workflow.ome_tiff_writer(
            filename=str(path), pixel_size_um=0.5, data=np.full((8, 8), value, dtype=np.uint8)
        )


def test_each_timepoint_gets_its_own_folder(tmp_path):
    ctx = _ctx(tmp_path)
    _write_timepoints(ctx, "90.0", [10, 20, 30])
    files = sorted(
        p.relative_to(ctx.output_path).as_posix() for p in ctx.output_path.rglob("*.tif")
    )
    assert files == ["90.0/t000/0.tif", "90.0/t001/0.tif", "90.0/t002/0.tif"]


def test_a_single_timepoint_keeps_the_old_layout(tmp_path):
    ctx = _ctx(tmp_path, n_timepoints=1)
    assert workflow._tile_dir(ctx, "90.0") == ctx.output_path / "90.0"
    assert workflow._timepoint_filename(ctx, "0.tif") == "0.tif"


def test_a_time_mean_replaces_the_timepoints(tmp_path):
    ctx = _ctx(tmp_path, t_projection="mean")
    _write_timepoints(ctx, "90.0", [10, 20, 30])
    _write_timepoints(ctx, None, [100, 110, 120])  # single-image modality, tiles at the top

    workflow._project_timepoints(ctx)

    assert tifffile.imread(ctx.output_path / "90.0" / "0.tif").mean() == pytest.approx(20)
    assert tifffile.imread(ctx.output_path / "0.tif").mean() == pytest.approx(110)
    assert not list(ctx.output_path.rglob("t00*"))


def test_kept_raw_timepoints_move_to_the_raw_folder(tmp_path):
    ctx = _ctx(tmp_path, t_projection="max", save_raw=True)
    _write_timepoints(ctx, "90.0", [10, 50, 30])

    workflow._project_timepoints(ctx)

    assert tifffile.imread(ctx.output_path / "90.0" / "0.tif").max() == 50
    raw = ctx.output_path.parent / "Raw" / "bounds" / "90.0"
    assert sorted(p.name for p in raw.iterdir()) == ["t000", "t001", "t002"]


def test_orientation_is_not_averaged(tmp_path):
    ctx = _ctx(tmp_path, t_projection="mean")
    _write_timepoints(ctx, "orientation", [10, 170, 10])

    workflow._project_timepoints(ctx)

    assert not (ctx.output_path / "orientation" / "0.tif").exists()
    assert (ctx.output_path / "orientation" / "t001" / "0.tif").exists()


def test_without_a_projection_every_timepoint_stays(tmp_path):
    ctx = _ctx(tmp_path, t_projection=None)
    _write_timepoints(ctx, "90.0", [10, 20, 30])
    workflow._project_timepoints(ctx)
    assert len(list((ctx.output_path / "90.0").glob("t*/0.tif"))) == 3


def test_the_projection_name_is_checked():
    log = logging.getLogger("test")
    assert workflow._resolve_time_projection("Mean", 5, log) == "mean"
    assert workflow._resolve_time_projection("none", 5, log) is None
    assert workflow._resolve_time_projection("mean", 1, log) is None
    assert workflow._resolve_time_projection("edf", 5, log) is None


def test_the_flag_is_parsed():
    params = workflow.parse_acquisition_message(
        "--yaml c.yml --projects p --sample s --scan-type ppm_20x_1 --region bounds "
        "--timepoints 30 --interval 2 --t-projection mean END_MARKER"
    )
    assert params["t_projection"] == "mean"
