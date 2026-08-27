"""FINDTISS -- move the stage until the camera is looking at tissue.

Why this is a separate command from autofocus
---------------------------------------------
A multi-slide batch predicts each slide's first alignment landmark from the base
transform, and that prediction is measured to land a median 613 um (worst 1507 um) from
the tile it aimed at. At that distance the camera is frequently over blank glass, and a
focus scan there has nothing to find -- it reports a peak from coverslip contrast, or
walks its whole attempt budget and gives up. The alignment that follows is then done
against an out-of-focus view.

SIFT is NOT the problem: it matched at 1507 um with 796 inliers. So the fix is purely
"put the camera on tissue before focusing", which is an XY operation with no opinion
about Z. Keeping it as its own verb means the (already subtle, already tuned) streaming
autofocus path is untouched, and the caller can order the two itself:

    MOVE -> FINDTISS -> STRMAFZ -> SIFT

Tissue is decided by the SAME strategy validity check the acquisition path uses for its
tissue/background decision (``texture_and_area`` and friends, thresholds from
``autofocus_<scope>.yml``), so there is no new metric to calibrate and no second
definition of "has content" to drift.

Exposure is deliberately NOT adjusted. The caller has just put the modality into its
alignment reference state (for PPM, the calibrated uncrossed angle and exposure), and
SIFT is about to match against that state -- so a brightness-chasing loop here would
silently change the thing the next step depends on. Frames that are unusable for the
validity check are reported as such rather than corrected.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from microscope_command_server.server.handlers.utils import parse_flags, read_message_string
from microscope_command_server.server.tissue_search import (
    DEFAULT_MAX_ATTEMPTS,
    MAX_ATTEMPTS_CEILING,
    parse_direction,
    search_offsets,
)

logger = logging.getLogger(__name__)


def _reply(conn, text: str) -> None:
    """Send one response line, tolerating a socket the client has already dropped.

    A failed send is logged and swallowed: by the time a reply goes out the stage has
    already been moved, and raising here would replace a useful log line about where the
    search ended with a traceback about a closed socket.
    """
    try:
        conn.sendall(text.encode())
    except Exception as e:
        logger.error("FINDTISS:reply send failed (%s): %s", text.split(":", 1)[0], e)


def _as_uint8(img):
    """Normalise a snapped frame to uint8, which is what the validity checks expect."""
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.dtype in (np.float32, np.float64):
        if arr.size and arr.max() <= 1.0 and arr.min() >= 0.0:
            return (arr * 255).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)
    if arr.dtype == np.uint16:
        # 16-bit frames scale by the actual range rather than a blind >>8: a 12-bit
        # sensor's frames would otherwise land in the bottom 1/16 of the uint8 range and
        # read as uniformly dark, i.e. "no tissue" everywhere.
        peak = float(arr.max()) if arr.size else 0.0
        if peak <= 0:
            return np.zeros(arr.shape, dtype=np.uint8)
        return np.clip(arr.astype(np.float32) * (255.0 / peak), 0, 255).astype(np.uint8)
    return arr.astype(np.uint8, copy=False)


def _resolve_validity(
    yaml_path: Optional[str], objective: Optional[str]
) -> Tuple[str, Dict[str, Any]]:
    """The strategy validity check name + thresholds for this scope and objective.

    Falls back to the shipped ``texture_and_area`` defaults when the autofocus YAML has
    nothing to say, which is the same thing the acquisition path does.
    """
    af_entry: Dict[str, Any] = {}
    if yaml_path:
        try:
            from microscope_command_server.server.handlers.streaming_focus import (
                _load_autofocus_yaml_for_objective,
            )

            af_entry = _load_autofocus_yaml_for_objective(yaml_path, objective) or {}
        except Exception as e:
            logger.warning("FINDTISS:could not read autofocus yaml (%s); using defaults", e)
    return (
        "texture_and_area",
        {
            "texture_threshold": float(af_entry.get("texture_threshold", 0.010)),
            "tissue_area_threshold": float(af_entry.get("tissue_area_threshold", 0.200)),
            "rgb_brightness_threshold": float(af_entry.get("rgb_brightness_threshold", 240.0)),
        },
    )


def _fov_diagonal_um(hardware) -> Optional[float]:
    """One camera FOV diagonal, the natural step for a search that must not skip ground."""
    try:
        fov = hardware.get_fov()
    except Exception as e:
        logger.warning("FINDTISS:could not read FOV: %s", e)
        return None
    try:
        w = float(fov[0])
        h = float(fov[1])
    except (TypeError, ValueError, IndexError):
        return None
    if w <= 0 or h <= 0:
        return None
    return float(np.hypot(w, h))


def handle_findtissue(conn, client, hardware, settings, **kwargs):
    """Entry point for FINDTISS.

    Payload flags (all optional except that a usable step must be derivable):
      ``--yaml``          main config path, used to find ``autofocus_<scope>.yml``
      ``--objective``     objective id, for the per-objective validity thresholds
      ``--dir``           ``"dx,dy"`` stage-space hint toward believed tissue
      ``--step``          radius increment in um (default: one FOV diagonal)
      ``--max-attempts``  positions to visit including the start

    Response, one line:
      ``FOUND:<x>:<y>:<attempt>:<of>``   stage left at the position that has tissue
      ``NOTFOUND:<x>:<y>:<of>``          every position checked was background; the stage
                                        is back where it started
      ``FAILED:<reason>``                the search could not be performed at all

    NOTFOUND deliberately returns the stage to the starting point. A search that found
    nothing has no more reason to prefer its last guess than its first, and leaving the
    stage somewhere the caller did not ask for would silently corrupt the alignment
    prediction it is about to use.
    """
    addr = getattr(client, "addr", client)
    try:
        message = read_message_string(conn)
    except Exception as e:
        logger.error("FINDTISS:failed to read payload from %s: %s", addr, e)
        _reply(conn, f"FAILED:payload-read-error: {e}")
        return

    params = parse_flags(message, ["--yaml", "--objective", "--dir", "--step", "--max-attempts"])
    direction = parse_direction(params.get("dir"))
    if params.get("dir") and direction is None:
        logger.warning(
            "FINDTISS:ignoring unusable --dir %r; searching the compass instead",
            params.get("dir"),
        )

    max_attempts = DEFAULT_MAX_ATTEMPTS
    if params.get("max_attempts"):
        try:
            requested = int(params["max_attempts"])
            if requested >= 1:
                max_attempts = min(requested, MAX_ATTEMPTS_CEILING)
            else:
                logger.warning("FINDTISS:ignoring --max-attempts < 1: %r", params["max_attempts"])
        except ValueError:
            logger.warning(
                "FINDTISS:ignoring non-integer --max-attempts: %r", params["max_attempts"]
            )

    step_um: Optional[float] = None
    if params.get("step"):
        try:
            candidate = float(params["step"])
            if candidate > 0:
                step_um = candidate
            else:
                logger.warning("FINDTISS:ignoring non-positive --step: %r", params["step"])
        except ValueError:
            logger.warning("FINDTISS:ignoring non-numeric --step: %r", params["step"])
    if step_um is None:
        step_um = _fov_diagonal_um(hardware)
    if step_um is None:
        _reply(conn, "FAILED:no --step and the camera FOV is unknown")
        return

    try:
        start = hardware.get_current_position()
        start_x = float(start.x)
        start_y = float(start.y)
        start_z = float(start.z)
    except Exception as e:
        logger.error("FINDTISS:could not read the stage position: %s", e)
        _reply(conn, f"FAILED:get-position: {e}")
        return

    validity_name, validity_kwargs = _resolve_validity(params.get("yaml"), params.get("objective"))
    try:
        from microscope_imageprocessing.focus import resolve_validity_check

        check = resolve_validity_check(validity_name)
    except Exception as e:
        logger.error("FINDTISS:validity check %s unavailable: %s", validity_name, e)
        _reply(conn, f"FAILED:validity-check-unavailable: {e}")
        return

    from microscope_control.hardware import Position

    offsets = search_offsets(direction, step_um, max_attempts)
    logger.info(
        "FINDTISS:searching %d position(s), step %.1f um, hint %s (validity=%s %s)",
        len(offsets),
        step_um,
        "none" if direction is None else f"({direction[0]:.3f}, {direction[1]:.3f})",
        validity_name,
        validity_kwargs,
    )

    for index, (dx, dy) in enumerate(offsets):
        attempt = index + 1
        x = start_x + dx
        y = start_y + dy
        try:
            hardware.move_to_position(Position(x, y, start_z))
        except Exception as e:
            # A move that fails is usually a stage limit -- keep looking at the offsets
            # that are still reachable rather than abandoning the search.
            logger.warning(
                "FINDTISS:attempt %d/%d could not move to (%.1f, %.1f): %s",
                attempt,
                len(offsets),
                x,
                y,
                e,
            )
            continue

        img = None
        try:
            snapped = hardware.snap_image()
            img = snapped[0] if isinstance(snapped, tuple) else snapped
        except Exception as e:
            logger.warning("FINDTISS:attempt %d/%d snap failed: %s", attempt, len(offsets), e)
        img = _as_uint8(img)
        if img is None:
            continue

        try:
            ok, stats = check(img, **validity_kwargs)
        except Exception as e:
            logger.warning(
                "FINDTISS:attempt %d/%d validity check failed: %s", attempt, len(offsets), e
            )
            continue

        if ok:
            logger.info(
                "FINDTISS:tissue at attempt %d/%d, (%.1f, %.1f), offset (%+.1f, %+.1f) um: %s",
                attempt,
                len(offsets),
                x,
                y,
                dx,
                dy,
                stats,
            )
            _reply(conn, f"FOUND:{x:.3f}:{y:.3f}:{attempt}:{len(offsets)}")
            return
        logger.info(
            "FINDTISS:attempt %d/%d at (%.1f, %.1f) is background: %s",
            attempt,
            len(offsets),
            x,
            y,
            stats,
        )

    logger.warning(
        "FINDTISS:no tissue at any of %d position(s); returning to (%.1f, %.1f)",
        len(offsets),
        start_x,
        start_y,
    )
    try:
        hardware.move_to_position(Position(start_x, start_y, start_z))
    except Exception as e:
        logger.error("FINDTISS:could not return to the starting position: %s", e)
    _reply(conn, f"NOTFOUND:{start_x:.3f}:{start_y:.3f}:{len(offsets)}")
