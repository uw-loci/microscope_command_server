"""Where a time-lapse's timepoint loop sits: --t-order region | position | plane.

region   for t: for position: tile (every plane)     -- e.g. a well plate's rounds
position for position: for t: tile (every plane)
plane    for position: for plane: for t: tile (that plane)
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from microscope_command_server.acquisition import workflow

LOG = logging.getLogger("test")


def test_resolve_time_order():
    resolve = workflow._resolve_time_order
    assert resolve(None, 5, True, True, LOG) == "region"
    assert resolve("Position", 5, False, False, LOG) == "position"
    assert resolve("plane", 5, True, True, LOG) == "plane"
    # One timepoint: the order means nothing.
    assert resolve("plane", 1, True, False, LOG) == "region"
    # No Z-stack: each position's series.
    assert resolve("plane", 5, False, False, LOG) == "position"
    with pytest.raises(ValueError, match="z-projection none"):
        resolve("plane", 5, True, False, LOG)
    with pytest.raises(ValueError, match="Unknown --t-order"):
        resolve("sideways", 5, True, True, LOG)


def test_the_flag_is_parsed():
    params = workflow.parse_acquisition_message(
        "--yaml c.yml --projects p --sample s --scan-type ppm_20x_1 --region bounds"
        " --timepoints 3 --interval 0 --t-order Plane"
    )
    assert params["t_order"] == "plane"
    params = workflow.parse_acquisition_message(
        "--yaml c.yml --projects p --sample s --scan-type ppm_20x_1 --region bounds"
    )
    assert params["t_order"] == "region"


def _visits(monkeypatch, order, n_t=2, n_pos=2, planes=(-1.0, 0.0, 1.0)):
    seen = []

    def fake_position(ctx, pos_idx, pos, filename, revisit):
        z = [i for i, _ in workflow._z_planes(ctx)]
        seen.append((ctx.current_timepoint, pos_idx, tuple(z), revisit))
        return True

    monkeypatch.setattr(workflow, "_acquire_position", fake_position)
    ctx = SimpleNamespace(
        n_timepoints=n_t,
        t_order=order,
        interval_seconds=0.0,
        positions=[(SimpleNamespace(x=0, y=0), f"{i}.tif") for i in range(n_pos)],
        z_offsets=list(planes),
        z_plane_only=None,
        current_timepoint=0,
        is_cancelled=lambda: False,
        logger=LOG,
        t0=0.0,
        time_lapse_warning_fired=False,
    )
    workflow._run_acquisition_loops(ctx, None)
    assert ctx.z_plane_only is None
    return seen


def test_region_order_visits_every_position_each_timepoint(monkeypatch):
    all_planes = (0, 1, 2)
    assert _visits(monkeypatch, "region") == [
        (0, 0, all_planes, False),
        (0, 1, all_planes, False),
        (1, 0, all_planes, False),
        (1, 1, all_planes, False),
    ]


def test_position_order_runs_each_positions_timepoints(monkeypatch):
    all_planes = (0, 1, 2)
    assert _visits(monkeypatch, "position") == [
        (0, 0, all_planes, False),
        (1, 0, all_planes, True),
        (0, 1, all_planes, False),
        (1, 1, all_planes, True),
    ]


def test_plane_order_runs_each_planes_timepoints(monkeypatch):
    seen = _visits(monkeypatch, "plane", n_pos=1)
    assert seen == [
        (0, 0, (0,), False),
        (1, 0, (0,), True),
        (0, 0, (1,), True),
        (1, 0, (1,), True),
        (0, 0, (2,), True),
        (1, 0, (2,), True),
    ]


def test_a_skipped_tile_ends_that_positions_series(monkeypatch):
    seen = []

    def skip_first(ctx, pos_idx, pos, filename, revisit):
        seen.append((ctx.current_timepoint, pos_idx))
        return pos_idx != 0

    monkeypatch.setattr(workflow, "_acquire_position", skip_first)
    ctx = SimpleNamespace(
        n_timepoints=3,
        t_order="position",
        interval_seconds=0.0,
        positions=[(SimpleNamespace(x=0, y=0), "0.tif"), (SimpleNamespace(x=1, y=0), "1.tif")],
        z_offsets=[0.0],
        z_plane_only=None,
        current_timepoint=0,
        is_cancelled=lambda: False,
        logger=LOG,
        t0=0.0,
        time_lapse_warning_fired=False,
    )
    workflow._run_acquisition_loops(ctx, None)
    assert seen == [(0, 0), (0, 1), (1, 1), (2, 1)]


def test_plane_order_names_the_plane_it_is_on():
    ctx = SimpleNamespace(z_plane_only=(2, 1.5), z_offsets=[-1.5, 0.0, 1.5])
    assert workflow._z_planes(ctx) == [(2, 1.5)]
    assert workflow._z_plane_count(ctx) == 3
    ctx.z_plane_only = None
    assert workflow._z_planes(ctx) == [(0, -1.5), (1, 0.0), (2, 1.5)]


class _Hardware:
    def __init__(self):
        self.z = 0.0
        self.moves = []

    def get_current_position(self):
        return SimpleNamespace(x=0.0, y=0.0, z=self.z)

    def move_to_position(self, pos):
        if getattr(pos, "z", None) is not None:
            self.z = pos.z
            self.moves.append(round(pos.z, 3))

    def get_z_position(self):
        return self.z

    def snap_image(self):
        return np.full((4, 4), int(self.z) + 10, dtype=np.uint16), {}

    def get_pixel_size_um(self):
        return 0.5


@pytest.mark.parametrize("order,expected", [("region", [0, 1, 0, 1]), ("plane", [0, 0, 1, 1])])
def test_the_stack_command_orders_planes_and_timepoints(tmp_path, order, expected):
    pytest.importorskip("tifffile")
    from microscope_command_server.acquisition.stack_timelapse import acquire_z_stack

    hw = _Hardware()
    result = acquire_z_stack(
        hardware=hw,
        output_folder=str(tmp_path / "out"),
        z_start=0.0,
        z_end=1.0,
        z_step=1.0,
        n_timepoints=2,
        interval_seconds=0.0,
        t_order=order,
    )
    # Z moves in visit order (the final move restores the start position).
    assert hw.moves[:4] == expected
    assert result["n_timepoints"] == 2
    assert any(f.endswith("zstack_t.ome.tiff") for f in result["files"])
