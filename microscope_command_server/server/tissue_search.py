"""Where to look for tissue when a predicted stage position lands on blank glass.

Pure geometry, no hardware -- lives outside ``server.handlers`` for the same reason
``server.focus_geometry`` and ``server.probe_parsers`` do: importing the handlers
package pulls in ``microscope_control`` -> ``pycromanager``, which a unit test has no
business needing in order to check a search pattern.

Why this exists
---------------
A multi-slide batch predicts each slide's first alignment landmark from the base
transform. Measured on 8 slides across two 4-slide runs (2026-08-24), that prediction
lands a median of **613 um** from the tile it aimed at, worst case **1507 um**. The
second landmark, corrected by the first's translation, lands within **26 um** -- so the
error is very nearly a CONSTANT PER-SLIDE OFFSET, and only the FIRST landmark of each
slide needs help.

What that help has to achieve is narrower than it first appears. SIFT already matched
at 1507 um (796 inliers, confidence 0.999), so alignment reach is not the problem. What
breaks is AUTOFOCUS: at 613 um from target the camera is often over blank glass, where a
focus scan has no tissue to find. **So the search only has to land on tissue -- any
tissue -- not on the intended tile.** Once focus is real, SIFT does the rest.

That is why the pattern below is a coarse fan rather than a fine raster: it is looking
for the tissue mass, not for a specific field of view.
"""

import math
from typing import List, Optional, Sequence, Tuple

#: Rings to sweep when the caller does not specify a budget. Two, because at a 446 um FOV
#: diagonal that reaches 892 um, which covers the measured MEDIAN landing error of 613 um
#: outright. The 1507 um worst case needs four; the caller sizes that trade-off, since it
#: knows whether anyone is waiting.
DEFAULT_RINGS = 2

#: Hard cap on the attempt budget. Each attempt is a stage move plus a snap, so a
#: mistyped budget should degrade the search, not park the socket for ten minutes.
MAX_ATTEMPTS_CEILING = 25

# Lateral spread either side of the direction hint, in degrees. The hint says which way
# the tissue lies, but it is derived from a transform that has just been shown to be off
# by more than a field of view -- so the pattern hedges around it instead of marching
# down a single ray that a wrong hint would waste every attempt on.
FAN_DEGREES = 45.0

# Bearings tried at each radius when there is no usable direction hint: the four
# compass points, then the diagonals. Even coverage is the best available strategy when
# nothing says which way to go.
_COMPASS_DEGREES = (0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0)


def default_max_attempts(direction: Optional[Sequence[float]]) -> int:
    """Attempt budget for {DEFAULT_RINGS} complete rings, given whether there is a hint.

    Deliberately NOT a single constant. The two patterns have different bearing counts (3
    around a hint, 8 around the compass), so one number cannot mean "whole rings" for both:
    a fixed 7 sweeps two full rings when hinted, but leaves the unhinted ring 6/8 done --
    biasing the search toward whichever bearings happen to be enumerated first, which is the
    one property the ring structure exists to avoid. Searching unhinted costs more because
    knowing nothing about where the tissue lies genuinely is more work.
    """
    return 1 + DEFAULT_RINGS * len(_bearings_for(direction))


def search_offsets(
    direction: Optional[Sequence[float]],
    step_um: float,
    max_attempts: int,
) -> List[Tuple[float, float]]:
    """Stage-space (dx, dy) offsets to try, in order, starting from the predicted point.

    The first offset is always ``(0.0, 0.0)``: check where the transform actually put us
    before moving anywhere. Every later offset lies on a ring at a whole multiple of
    ``step_um``, so an attempt budget translates directly into a reach --
    ``step_um * ceil((max_attempts - 1) / bearings)``.

    :param direction: stage-space vector pointing at where tissue is believed to be. Its
        length is ignored (only the bearing is used). ``None``, too short to have a
        direction, or non-finite falls back to the compass pattern.
    :param step_um: radius increment, normally one camera FOV diagonal. Non-positive
        yields the no-move-at-all list.
    :param max_attempts: total positions to visit, including the starting one.
    :return: offsets in visit order; length is ``max(0, max_attempts)``.
    """
    if max_attempts <= 0:
        return []
    offsets: List[Tuple[float, float]] = [(0.0, 0.0)]
    if max_attempts == 1 or step_um <= 0:
        return offsets[:max_attempts]

    bearings = _bearings_for(direction)
    ring = 1
    while len(offsets) < max_attempts:
        radius = ring * step_um
        for bearing in bearings:
            if len(offsets) >= max_attempts:
                break
            radians = math.radians(bearing)
            offsets.append((radius * math.cos(radians), radius * math.sin(radians)))
        ring += 1
    return offsets


def _bearings_for(direction: Optional[Sequence[float]]) -> Tuple[float, ...]:
    """Bearings to sweep at each radius: hint-first fan, or the compass when unhinted."""
    heading = _heading_degrees(direction)
    if heading is None:
        return _COMPASS_DEGREES
    # Straight down the hint first, then either side of it. A hint that is right costs
    # one attempt; a hint that is 45 deg out costs two.
    return (heading, heading + FAN_DEGREES, heading - FAN_DEGREES)


def _heading_degrees(direction: Optional[Sequence[float]]) -> Optional[float]:
    """Bearing of ``direction`` in degrees, or None when it does not define one."""
    if direction is None or len(direction) < 2:
        return None
    try:
        dx = float(direction[0])
        dy = float(direction[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None
    # A hint shorter than this is numerically a point, not a direction. It happens when
    # the predicted position already sits on the tissue centroid the hint was measured
    # to -- in which case there is no preferred way to go and the compass is honest.
    if math.hypot(dx, dy) < 1e-9:
        return None
    return math.degrees(math.atan2(dy, dx))


def parse_direction(text: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse a ``"dx,dy"`` direction argument, returning None for anything unusable.

    Unusable input is deliberately not an error: the search still works without a hint,
    and refusing to run because a hint was malformed would turn a degraded search into a
    failed slide.
    """
    if not text:
        return None
    parts = str(text).split(",")
    if len(parts) != 2:
        return None
    try:
        dx = float(parts[0])
        dy = float(parts[1])
    except ValueError:
        return None
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None
    return (dx, dy)
