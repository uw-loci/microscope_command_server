"""Search pattern for FINDTISS: where to look when a predicted position lands on glass.

Pure geometry -- no hardware, no MMCore. The handler that drives the stage over these
offsets cannot be tested here (it needs ``microscope_control`` -> ``pycromanager``),
which is exactly why the pattern lives in its own module.
"""

import math

import pytest

from microscope_command_server.server.tissue_search import (
    DEFAULT_RINGS,
    FAN_DEGREES,
    MAX_ATTEMPTS_CEILING,
    default_max_attempts,
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

    def test_search_tries_the_hint_then_either_side_of_it(self):
        offsets = search_offsets((1.0, 0.0), 400.0, 4)
        assert [_bearing(o) for o in offsets[1:]] == pytest.approx(
            [0.0, FAN_DEGREES, 360.0 - FAN_DEGREES]
        )

    def test_a_hinted_ring_is_still_a_complete_ring(self):
        # The regression this guards is the one that made a wrong hint fatal. The hint is
        # measured in the transform's own frame, and the transform is off by the very offset
        # the search exists to defeat -- so its angular error is unbounded when the predicted
        # point sits near the tile-grid centre, which is exactly where the reference-tile
        # picker tends to aim. A pattern that only swept +/-45 deg of the hint would then
        # march two rings the wrong way and report NOTFOUND with tissue behind it.
        hinted = search_offsets((1.0, 0.0), 400.0, 9)
        unhinted = search_offsets(None, 400.0, 9)
        assert sorted(_bearing(o) for o in hinted[1:]) == pytest.approx(
            sorted(_bearing(o) for o in unhinted[1:])
        )
        # ... and the hint still decides the ORDER, which is the whole value of having one.
        assert _bearing(hinted[1]) == pytest.approx(0.0)

    def test_a_reversed_hint_still_reaches_the_tissue_behind_it(self):
        # A hint pointing 180 deg away from the truth must cost attempts, not the slide.
        offsets = search_offsets((1.0, 0.0), 400.0, default_max_attempts((1.0, 0.0)))
        opposite = [o for o in offsets[1:] if _bearing(o) == pytest.approx(180.0)]
        assert opposite, "the bearing opposite the hint is never visited"
        assert _radius(opposite[0]) == pytest.approx(400.0), "and it is reached on ring 1"

    def test_radius_grows_one_step_per_ring(self):
        # An attempt budget must translate directly into a reach, so a caller can size it
        # against a measured landing error.
        offsets = search_offsets((1.0, 0.0), 400.0, 11)
        assert [_radius(o) for o in offsets] == pytest.approx([0.0] + [400.0] * 8 + [800.0] * 2)

    def test_reach_is_the_sizing_number_against_the_measured_landing_error(self):
        # The budget has to be chosen against a measurement, so pin down what each one
        # buys. At a 446 um FOV diagonal and 8 bearings per ring, the radius swept in EVERY
        # direction is 446 * ((n - 1) // 8): the default covers the measured MEDIAN landing
        # error of 613 um outright, and the 1507 um worst case needs four rings.
        def reach(n):
            return max(_radius(o) for o in search_offsets((1.0, 0.0), 446.0, n))

        assert reach(9) == pytest.approx(446.0)
        assert reach(9) < 613.0, "one ring does not reach the median error on its own"
        assert reach(default_max_attempts((1.0, 0.0))) >= 613.0
        assert reach(default_max_attempts((1.0, 0.0))) < 1507.0
        assert reach(MAX_ATTEMPTS_CEILING) >= 1507.0, "the ceiling can still cover the worst case"

    @pytest.mark.parametrize("direction", [(1.0, 0.0), None])
    def test_default_budget_is_always_a_whole_number_of_rings(self, direction):
        # A budget that stops mid-ring biases the search toward whichever bearings are
        # enumerated first -- the exact thing the ring structure exists to prevent. With a
        # hint, "enumerated first" means "wherever the hint pointed", so a partial ring puts
        # the bias precisely where the hint is least trustworthy. Deriving the budget from
        # the bearing count is what keeps that from happening.
        n = default_max_attempts(direction)
        offsets = search_offsets(direction, 446.0, n)
        assert len(offsets) == n

        per_ring = (n - 1) // DEFAULT_RINGS
        for ring in range(1, DEFAULT_RINGS + 1):
            at_this_radius = sum(1 for o in offsets if _radius(o) == pytest.approx(ring * 446.0))
            assert at_this_radius == per_ring, f"ring {ring} is not fully swept"

    def test_hinting_does_not_change_the_budget_only_the_order(self):
        # Both sweep complete rings, so both cost the same in the worst case. What a hint
        # buys is an EARLY exit, which does not show up in the offset list -- it shows up in
        # the search returning at attempt 2 instead of attempt 7.
        assert default_max_attempts(None) == default_max_attempts((1.0, 0.0))

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
