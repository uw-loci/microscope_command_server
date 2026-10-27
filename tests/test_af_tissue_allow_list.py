"""The autofocus allow-list (``--af-tissue-tiles``): no focus search starts on a tile the
client's macro image shows as glass.

On 2026-10-27 the server ran 28 focus searches in a sparse LC-PolScope region, most on bare
glass, because the plan, the two safety nets and the deferral all chose tiles by distance
alone and the camera-side ``dense_texture`` gate read glass and tissue alike on the Flash4
(0.014 either way). Each search on glass found a noise peak 44-63 um away and the
nearest-seed logic spread those Zs across the mosaic.

These tests pin the parser and the four pure decisions the allow-list adds. The per-tile
handler itself needs a live stage; its allow-list rule is the pure function tested here.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_workflow():
    path = ROOT / "microscope_command_server" / "acquisition" / "workflow.py"
    spec = importlib.util.spec_from_file_location("mcs_workflow_af_tissue", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcs_workflow_af_tissue"] = module
    spec.loader.exec_module(module)
    return module


wf = _load_workflow()
log = logging.getLogger("test_af_tissue_allow_list")

REQUIRED = "--yaml c.yml --projects /p --sample s --scan-type lcpolscope_20x_1 --region r"


# ---------------------------------------------------------------- parser


def test_the_list_is_parsed_as_ints_and_marks_the_region_scored():
    params = wf.parse_acquisition_message(f"{REQUIRED} --af-tissue-tiles 0,2,5,9 ENDOFSTR")
    assert params["af_tissue_tiles"] == [0, 2, 5, 9]
    assert params["af_tissue_scored"] is True


def test_none_means_scored_and_empty():
    params = wf.parse_acquisition_message(f"{REQUIRED} --af-tissue-tiles none ENDOFSTR")
    assert params["af_tissue_tiles"] == []
    assert params["af_tissue_scored"] is True


def test_absent_means_unscored():
    params = wf.parse_acquisition_message(f"{REQUIRED} ENDOFSTR")
    assert "af_tissue_tiles" not in params
    assert "af_tissue_scored" not in params


# ---------------------------------------------------------------- plan


def _grid(cols, rows, pitch=300.0):
    return [(c * pitch, r * pitch) for r in range(rows) for c in range(cols)]


def test_planned_points_off_the_list_move_to_the_nearest_tissue_tile():
    xy = _grid(6, 1)
    # Lattice planned 0 and 3; only 1 and 4 hold tissue.
    planned = wf._restrict_af_plan_to_tissue([0, 3], {1, 4}, xy, 600.0, None, log)
    assert planned == [1, 4]


def test_a_point_with_no_tissue_within_reach_is_dropped():
    xy = _grid(10, 1)
    # Planned 0 and 9; tissue only at 1 -- tile 9 is 2400 um from it, beyond 600.
    planned = wf._restrict_af_plan_to_tissue([0, 9], {1}, xy, 600.0, None, log)
    assert planned == [1]


def test_the_preferred_tile_stays_first():
    xy = _grid(6, 1)
    planned = wf._restrict_af_plan_to_tissue([0, 3, 5], {0, 3, 5}, xy, 600.0, 3, log)
    assert planned[0] == 3
    assert sorted(planned) == [0, 3, 5]


def test_none_reduces_the_plan_to_the_preferred_tile_alone():
    xy = _grid(6, 1)
    assert wf._restrict_af_plan_to_tissue([0, 3, 5], set(), xy, 600.0, 2, log) == [2]


def test_every_point_dropped_still_leaves_one_search_on_tissue():
    xy = _grid(10, 1)
    planned = wf._restrict_af_plan_to_tissue([9], {0}, xy, 600.0, None, log)
    assert planned == [0]


# ---------------------------------------------------------------- per tile


def test_a_glass_tile_never_searches_even_when_the_plan_or_nets_ask():
    needs, af_type, wanted, note = wf._apply_tissue_allow_list(7, True, {1, 9}, False)
    assert needs is False
    assert af_type == "skipped-glass"
    assert wanted is True, "the search due here is owed to the next tissue tile"
    assert "No tissue in tile 7" in note


def test_the_owed_search_runs_on_the_next_tissue_tile():
    needs, af_type, wanted, note = wf._apply_tissue_allow_list(9, False, {1, 9}, True)
    assert needs is True
    assert af_type is None
    assert wanted is False
    assert "deferred from glass" in note


def test_a_tissue_tile_with_nothing_due_is_left_alone():
    assert wf._apply_tissue_allow_list(9, False, {1, 9}, False) == (False, None, False, None)
    assert wf._apply_tissue_allow_list(9, True, {1, 9}, False) == (True, None, False, None)


def test_no_list_and_the_benchmark_change_nothing():
    assert wf._apply_tissue_allow_list(7, True, None, False) == (True, None, False, None)
    assert wf._apply_tissue_allow_list(7, True, {1}, True, af_benchmark=True) == (
        True,
        None,
        True,
        None,
    )


# ---------------------------------------------------------------- deferral


def test_a_deferral_target_off_the_list_is_replaced_by_the_next_tissue_tile_far_enough():
    xy = _grid(8, 1)
    # Proposed 3 is glass; 4 is tissue but only 600 um from 2 (< 0.7 * 1000); 6 qualifies.
    assert wf._next_tissue_tile_for_deferral(2, 3, {4, 6}, xy, 1000.0, log) == 6


def test_a_deferral_with_no_tissue_ahead_is_not_queued():
    xy = _grid(8, 1)
    assert wf._next_tissue_tile_for_deferral(6, 7, {0, 1}, xy, 300.0, log) is None
    assert wf._next_tissue_tile_for_deferral(2, 3, set(), xy, 300.0, log) is None


def test_deferral_is_untouched_without_a_list_or_when_already_on_it():
    xy = _grid(8, 1)
    assert wf._next_tissue_tile_for_deferral(2, 3, None, xy, 300.0, log) == 3
    assert wf._next_tissue_tile_for_deferral(2, 3, {3}, xy, 300.0, log) == 3
    assert wf._next_tissue_tile_for_deferral(2, None, {3}, xy, 300.0, log) is None


# ---------------------------------------------------------------- pre-acquisition search


def test_the_tissue_search_walks_tissue_tiles_nearest_the_centre_first():
    xy = _grid(5, 1)  # centre at x = 600 (tile 2)
    assert wf._nearest_tissue_tile_to_centre({0, 1, 4}, {0}, xy) == 1
    assert wf._nearest_tissue_tile_to_centre({0, 1, 4}, {0, 1}, xy) == 4
    assert wf._nearest_tissue_tile_to_centre({0, 1, 4}, {0, 1, 4}, xy) is None


def test_none_and_no_list_end_the_walk():
    xy = _grid(5, 1)
    assert wf._nearest_tissue_tile_to_centre(set(), {0}, xy) is None
    assert wf._nearest_tissue_tile_to_centre(None, {0}, xy) is None


@pytest.mark.parametrize("bad", [-1, 99])
def test_out_of_range_indices_on_the_list_are_ignored(bad):
    xy = _grid(4, 1)
    assert wf._restrict_af_plan_to_tissue([0], {bad, 2}, xy, 600.0, None, log) == [2]
    assert wf._nearest_tissue_tile_to_centre({bad}, set(), xy) is None
    assert isinstance(np.mean(np.asarray(xy, dtype=float), axis=0), np.ndarray)
