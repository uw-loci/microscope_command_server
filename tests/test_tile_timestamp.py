"""The per-tile acquisition timestamp is wall-clock time.

It read 1970-01-09 on every tile when the perf_counter() start time was passed to localtime().
"""

import logging
import time
from types import SimpleNamespace

from microscope_command_server.acquisition.workflow import _record_tile_measurement


def test_tile_timestamp_is_wall_clock_at_the_tile_start():
    ctx = SimpleNamespace(
        logger=logging.getLogger("t"),
        positions=[object()],
        tile_measurements=[],
        tile_measurements_stream=None,
        af_skip_count=0,
        af_strategy_name=None,
        af_tissue_tiles=None,
        modality="lcpolscope",
        progress_warning_fired=True,
    )
    before = time.time()
    _record_tile_measurement(
        ctx,
        0,
        "0.tif",
        time.perf_counter() - 30.0,
        False,
        "none",
        0.0,
        False,
        {},
        SimpleNamespace(x=0.0, y=0.0, z=1.0),
    )
    stamp = time.mktime(
        time.strptime(ctx.tile_measurements[0]["acq_timestamp_iso"], "%Y-%m-%dT%H:%M:%S")
    )
    assert abs(stamp - (before - 30.0)) < 3.0
