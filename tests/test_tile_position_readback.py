"""The recorded per-tile stage position is the settled one.

2026-10-19, LC-PolScope 2x2 (channel modality, 5 channels per tile): tile 3 was
recorded in tile_manifest.csv and TileConfiguration_Stage.txt at X=9973.4 for a
commanded 9697.0, while image registration showed it was taken at 9697. A
non-AF tile's XY move is issued no-wait, and the acquisition loop read the
stage before ``wait_for_xy()`` -- partway through the 378 um move from tile 2.

The fake stage below reports a point along the move until it is waited on,
which is what a Prior stage behind Micro-Manager does.

ASCII-only per project policy.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from microscope_command_server.acquisition import workflow
from microscope_control.hardware import Position

WORKFLOW_SRC = Path(workflow.__file__)

# Tile 2 -> tile 3 of the failing run.
TILE2_XY = (10074.9, 1817.0)
TILE3_COMMANDED = Position(9697.0, 1817.0, -45.4)
TILE3_MID_MOVE_X = 9973.4


class MovingStage:
    """XY stage whose readback is mid-move until ``wait_for_xy()`` is called."""

    def __init__(self, start_xy, z=-45.4, mid_fraction=None, stuck_at=None, lag_reads=0):
        self.x, self.y = start_xy
        self.z = z
        self.target = None
        self.moving = False
        # Where along the move a read during motion lands (0 = start, 1 = target).
        self.mid_fraction = mid_fraction
        # A stage that never gets there: settles at this XY instead of the target.
        self.stuck_at = stuck_at
        # Reads after settling that still return the pre-wait value (a report that
        # lags the Busy flag), consumed one per read.
        self.lag_reads = lag_reads
        self.wait_calls = 0
        self.reads = []

    def move_xy_no_wait(self, x, y):
        self.start = (self.x, self.y)
        self.target = (x, y)
        self.moving = True

    def _mid(self):
        f = self.mid_fraction if self.mid_fraction is not None else 0.5
        sx, sy = self.start
        tx, ty = self.target
        return sx + (tx - sx) * f, sy + (ty - sy) * f

    def wait_for_xy(self):
        self.wait_calls += 1
        if self.moving:
            self._stale = self._mid()
            self.x, self.y = self.stuck_at if self.stuck_at is not None else self.target
            self.moving = False

    def get_current_position(self):
        if self.moving:
            xy = self._mid()
        elif self.lag_reads > 0:
            self.lag_reads -= 1
            xy = self._stale
        else:
            xy = (self.x, self.y)
        pos = Position(xy[0], xy[1], self.z)
        self.reads.append(pos)
        return pos


def _ctx(stage, name):
    return SimpleNamespace(
        hardware=stage,
        logger=logging.getLogger(name),
        stage_positions_collected=[],
    )


def _mid_fraction_for_tile3():
    sx, tx = TILE2_XY[0], TILE3_COMMANDED.x
    return (TILE3_MID_MOVE_X - sx) / (tx - sx)


@pytest.fixture(autouse=True)
def _no_reread_sleep(monkeypatch):
    monkeypatch.setattr(workflow, "_TILE_POSITION_REREAD_DELAY_S", 0.0)


def test_the_fake_reproduces_the_mid_move_readback():
    # Guard on the fake itself: read the way the old loop did and get 9973.4.
    stage = MovingStage(TILE2_XY, mid_fraction=_mid_fraction_for_tile3())
    stage.move_xy_no_wait(TILE3_COMMANDED.x, TILE3_COMMANDED.y)
    assert stage.get_current_position().x == pytest.approx(TILE3_MID_MOVE_X)


def test_a_no_wait_move_records_the_settled_position(caplog):
    stage = MovingStage(TILE2_XY, mid_fraction=_mid_fraction_for_tile3())
    ctx = _ctx(stage, "tile_pos_settled")
    stage.move_xy_no_wait(TILE3_COMMANDED.x, TILE3_COMMANDED.y)

    with caplog.at_level(logging.WARNING, logger="tile_pos_settled"):
        recorded = workflow._record_settled_tile_position(ctx, TILE3_COMMANDED, "3.tif", True)

    assert stage.wait_calls >= 1
    assert recorded.x == pytest.approx(9697.0)
    assert recorded.y == pytest.approx(1817.0)
    assert ctx.stage_positions_collected == [("3.tif", 9697.0, 1817.0, -45.4)]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_readback_that_lags_settling_is_read_again(caplog):
    # Waited on, but the first report still carries the mid-move value.
    stage = MovingStage(TILE2_XY, mid_fraction=_mid_fraction_for_tile3(), lag_reads=1)
    ctx = _ctx(stage, "tile_pos_lag")
    stage.move_xy_no_wait(TILE3_COMMANDED.x, TILE3_COMMANDED.y)

    with caplog.at_level(logging.WARNING, logger="tile_pos_lag"):
        recorded = workflow._record_settled_tile_position(ctx, TILE3_COMMANDED, "3.tif", True)

    assert stage.reads[0].x == pytest.approx(TILE3_MID_MOVE_X)
    assert recorded.x == pytest.approx(9697.0)
    assert ctx.stage_positions_collected[0][1] == pytest.approx(9697.0)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_real_positioning_failure_is_warned_and_recorded_as_reported(caplog):
    stage = MovingStage(TILE2_XY, stuck_at=(9973.4, 1816.8))
    ctx = _ctx(stage, "tile_pos_stuck")
    stage.move_xy_no_wait(TILE3_COMMANDED.x, TILE3_COMMANDED.y)

    with caplog.at_level(logging.WARNING, logger="tile_pos_stuck"):
        recorded = workflow._record_settled_tile_position(ctx, TILE3_COMMANDED, "3.tif", True)

    # The record is what the stage reports -- never the commanded value dressed up.
    assert recorded.x == pytest.approx(9973.4)
    assert ctx.stage_positions_collected == [("3.tif", 9973.4, 1816.8, -45.4)]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0]
    assert "3.tif" in msg
    assert "9973.40" in msg and "9697.00" in msg
    assert msg.isascii()


def test_a_blocking_move_is_not_waited_on_again():
    # AF tiles move with move_to_position (blocking); the read costs no extra wait.
    stage = MovingStage((9697.0, 1817.0))
    ctx = _ctx(stage, "tile_pos_blocking")
    recorded = workflow._record_settled_tile_position(ctx, TILE3_COMMANDED, "3.tif", False)
    assert stage.wait_calls == 0
    assert recorded.x == pytest.approx(9697.0)


def test_readback_noise_within_tolerance_is_silent(caplog):
    stage = MovingStage((9697.05, 1816.9))
    ctx = _ctx(stage, "tile_pos_noise")
    with caplog.at_level(logging.DEBUG, logger="tile_pos_noise"):
        workflow._record_settled_tile_position(ctx, TILE3_COMMANDED, "3.tif", False)
    assert stage.wait_calls == 0
    assert not caplog.records


def test_the_loop_records_positions_only_through_the_settled_read():
    """Both dispatch branches record after XY settles; nothing reads before.

    The angle branch keeps its rotation/XY overlap, so it records after its
    snaps; the channel/single branch records before its snaps, in place of
    the bare wait_for_xy() it used to make.
    """
    source = WORKFLOW_SRC.read_text()
    loop = "for pos_idx, (pos, filename) in enumerate(ctx.positions):"
    body = source.split(loop, 1)[1].split("_record_tile_measurement(", 1)[0]

    assert "stage_positions_collected.append" not in body
    assert body.count("_record_settled_tile_position(") == 2

    angle_branch, other_branch = body.split("\n                else:\n", 1)
    assert angle_branch.index("_acquire_tile_angles(") < angle_branch.index(
        "_record_settled_tile_position("
    )
    assert other_branch.index("_record_settled_tile_position(") < other_branch.index(
        "_acquire_tile_channels("
    )
