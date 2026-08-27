"""Search pattern for FINDTISS: where to look when a predicted position lands on glass.

Pure geometry -- no hardware, no MMCore. The handler that drives the stage over these
offsets cannot be tested here (it needs ``microscope_control`` -> ``pycromanager``),
which is exactly why the pattern lives in its own module.
"""

import math

import pytest

from microscope_command_server.server.tissue_search import (
    DEFAULT_MAX_ATTEMPTS,
    FAN_DEGREES,
    parse_direction,
    search_offsets,
)


def _radius(offset):
    return math.hypot(offset[0], offset[1])


def _bearing(offset):
    return math.degrees(math.atan2(offset[1], offset[0])) % 360.0


class TestSearchOffsets:
    def test_first_offset_is_always_the_starting_point(self):
        # Check where the transform actually put us before moving anywhere: at the
        # median 613 um error the camera is often still on tissue.
        assert search_offsets((1.0, 0.0), 446.0, 6)[0] == (0.0, 0.0)
        assert search_offsets(None, 446.0, 6)[0] == (0.0, 0.0)

    def test_returns_exactly_the_requested_number_of_positions(self):
        for n in range(1, 20):
            assert len(search_offsets((1.0, 0.0), 446.0, n)) == n

    def test_no_attempts_means_no_positions(self):
        assert search_offsets((1.0, 0.0), 446.0, 0) == []
        assert search_offsets((1.0, 0.0), 446.0, -3) == []

    def test_single_attempt_never_moves(self):
        assert search_offsets((1.0, 0.0), 446.0, 1) == [(0.0, 0.0)]

    def test_non_positive_step_never_moves(self):
        # Better to check one position than to march the stage in zero-length steps.
        assert search_offsets((1.0, 0.0), 0.0, 5) == [(0.0, 0.0)]
        assert search_offsets((1.0, 0.0), -10.0, 5) == [(0.0, 0.0)]

    def test_hint_is_tried_first_and_at_one_step(self):
        offsets = search_offsets((0.0, 1.0), 400.0, 4)
        assert _radius(offsets[1]) == pytest.approx(400.0)
        assert _bearing(offsets[1]) == pytest.approx(90.0)

    def test_hint_length_is_ignored_only_its_bearing_matters(self):
        short = search_offsets((0.001, 0.0), 400.0, 4)
        long = search_offsets((9999.0, 0.0), 400.0, 4)
        assert short == long

    def test_search_fans_either_side_of_the_hint(self):
        offsets = search_offsets((1.0, 0.0), 400.0, 4)
        bearings = sorted(_bearing(o) for o in offsets[1:])
        assert bearings == pytest.approx(sorted([0.0, FAN_DEGREES, 360.0 - FAN_DEGREES]))

    def test_radius_grows_one_step_per_ring(self):
        # An attempt budget must translate directly into a reach, so a caller can size it
        # against a measured landing error.
        offsets = search_offsets((1.0, 0.0), 400.0, 7)
        assert [_radius(o) for o in offsets] == pytest.approx(
            [0.0, 400.0, 400.0, 400.0, 800.0, 800.0, 800.0]
        )

    def test_reach_is_the_sizing_number_against_the_measured_landing_error(self):
        # The budget has to be chosen against a measurement, so pin down what each one
        # buys. At a 446 um FOV diagonal and 3 bearings per ring, reach is
        # 446 * ((n - 1) // 3): the default 7 covers the measured MEDIAN landing error
        # of 613 um outright, and the 1507 um worst case needs 13.
        def reach(n):
            return max(_radius(o) for o in search_offsets((1.0, 0.0), 446.0, n))

        assert reach(4) == pytest.approx(446.0)
        assert reach(4) < 613.0, "one ring does not reach the median error on its own"
        assert reach(DEFAULT_MAX_ATTEMPTS) >= 613.0
        assert reach(DEFAULT_MAX_ATTEMPTS) < 1507.0
        assert reach(13) >= 1507.0

    def test_default_budget_is_two_full_rings(self):
        # Sized so no bearing in the fan is left half-swept: a budget that stops mid-ring
        # would bias the search toward whichever side happens to be enumerated first.
        offsets = search_offsets((1.0, 0.0), 446.0, DEFAULT_MAX_ATTEMPTS)
        assert len(offsets) == DEFAULT_MAX_ATTEMPTS
        assert sum(1 for o in offsets if _radius(o) == pytest.approx(446.0)) == 3
        assert sum(1 for o in offsets if _radius(o) == pytest.approx(892.0)) == 3

    def test_no_hint_sweeps_the_compass(self):
        offsets = search_offsets(None, 400.0, 5)
        bearings = sorted(_bearing(o) for o in offsets[1:])
        assert bearings == pytest.approx([0.0, 90.0, 180.0, 270.0])

    @pytest.mark.parametrize("bad", [None, (), (1.0,), ("a", "b"), (float("nan"), 0.0), (0.0, 0.0)])
    def test_unusable_hint_degrades_to_the_compass_not_to_failure(self, bad):
        # A hint that does not define a direction -- including the degenerate case where
        # the predicted position already sits on the tissue centroid -- must not stop the
        # search; it just removes the bias.
        offsets = search_offsets(bad, 400.0, 5)
        assert sorted(_bearing(o) for o in offsets[1:]) == pytest.approx([0.0, 90.0, 180.0, 270.0])


class TestParseDirection:
    def test_parses_a_pair(self):
        assert parse_direction("1.5,-2.5") == (1.5, -2.5)

    def test_tolerates_surrounding_whitespace(self):
        assert parse_direction(" 1.0 , 2.0 ") == (1.0, 2.0)

    @pytest.mark.parametrize("bad", [None, "", "1.0", "1,2,3", "a,b", "nan,0", "1,inf"])
    def test_unusable_input_is_none_not_an_error(self, bad):
        # The search still works without a hint. Refusing to run because a hint was
        # malformed would turn a degraded search into a failed slide.
        assert parse_direction(bad) is None
