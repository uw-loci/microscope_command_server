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
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from microscope_command_server.server.focus_validity import load_autofocus_doc, resolve_validity
from microscope_command_server.server.handlers.utils import parse_flags, read_message_string
from microscope_command_server.server.tissue_search import (
    MAX_ATTEMPTS_CEILING,
    default_max_attempts,
    parse_direction,
    search_offsets,
)

logger = logging.getLogger(__name__)

#: How long to wait for a frame from a running stream before falling back to a blocking
#: snap. Generous at any realistic frame rate (30 fps is 33 ms), and the fallback is
#: correct anyway -- this only decides how long we hope to save 400 ms.
FRESH_FRAME_WAIT_S = 0.25


def _return_to_start(hardware, x: float, y: float, z: float) -> None:
    """Put the stage back where the search began.

    Both the cancelled and the found-nothing paths owe the caller this: neither has any
    reason to prefer the last position tried, and leaving the stage somewhere the caller did
    not ask for would silently invalidate the prediction it is about to align against.
    """
    try:
        from microscope_control.hardware import Position

        hardware.move_to_position(Position(x, y, z))
    except Exception as e:
        logger.error("FINDTISS:could not return to the starting position (%.1f, %.1f): %s", x, y, e)


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


def _fresh_frame(hardware):
    """One frame that is definitely from AFTER the stage move, as cheaply as possible.

    Two reasons this is not just ``hardware.snap_image()``.

    STALENESS, which would be a wrong answer rather than a slow one. FINDTISS runs with the
    Live Viewer streaming -- ``AlignmentLivePrep`` puts the rig there so SIFT can work -- so
    the camera has a circular buffer of frames captured before and during the move we just
    made. Reading one of those evaluates the PREVIOUS position, which is how a search
    reports tissue at a place that does not have any. Draining the buffer first and taking
    the next frame to arrive is what makes the frame belong to where the stage now is.

    COST. A blocking snap on the JAI is ~400 ms -- the streaming-AF handler calls it "the
    single biggest fixed cost" and pops a stream frame instead for exactly this reason.
    Over a 17-position sweep that is seven seconds of pure snap overhead per slide.

    Falls back to a plain snap whenever no sequence is running or the buffer path does not
    produce a frame in time; a snap is always correct, just slower.

    A side benefit worth keeping: a stream frame carries the LIVE exposure, which is the
    state SIFT is about to match against. Snap and live exposures are separately held on the
    JAI and have drifted apart before, so judging tissue from the same frames the alignment
    will use is the more honest test as well as the cheaper one.
    """
    core = getattr(hardware, "core", None)
    sequence_running = False
    if core is not None:
        try:
            sequence_running = bool(core.is_sequence_running())
        except Exception as e:
            logger.debug("FINDTISS:could not query the sequence state (%s); snapping", e)

    if sequence_running:
        try:
            from microscope_command_server.server.handlers.streaming_focus import (
                _pop_image_as_numpy,
            )

            # Everything already buffered predates this position. Drop it.
            core.clear_circular_buffer()
            deadline = time.perf_counter() + FRESH_FRAME_WAIT_S
            while time.perf_counter() < deadline:
                if int(core.get_remaining_image_count()) > 0:
                    img = _pop_image_as_numpy(core)
                    if img is not None:
                        return img
                    break
                time.sleep(0.003)
            logger.info(
                "FINDTISS:no stream frame within %.2fs; snapping instead", FRESH_FRAME_WAIT_S
            )
        except Exception as e:
            logger.info("FINDTISS:stream frame unavailable (%s); snapping instead", e)

    snapped = hardware.snap_image()
    return snapped[0] if isinstance(snapped, tuple) else snapped


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
    yaml_path: Optional[str], modality: Optional[str], objective: Optional[str]
) -> Tuple[str, Dict[str, Any]]:
    """The validity check name + parameters this modality and objective actually use.

    Delegates to the shared resolver so this uses the SAME chain the acquisition path
    does -- modality binding, strategy, then the binding's ``validity_params`` overrides.
    Reading only the flat per-objective keys would discard exactly the tuning PPM and
    LC-PolScope depend on (a widened ``tissue_mask_range``, and on LC-PolScope a
    ``tissue_area_threshold`` its config says the default value rejects valid fields at),
    so the search would report "no tissue" while looking straight at some.
    """
    return resolve_validity(load_autofocus_doc(yaml_path), modality, objective)


def _fov_diagonal_um(hardware) -> Optional[float]:
    """One camera FOV diagonal -- the default search step.

    Deliberately coarse, and it does leave gaps: stepping 446 um along X with a 357 x 267 um
    field skips 89 um. The largest step that could not skip anything on an arbitrary bearing
    is the field's SHORT side, which would need far more positions for the same reach. Gaps
    are the right trade here because the target is a tissue mass many fields across, not a
    particular field -- the same choice the acquisition path's own first-tile search makes.
    """
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
      ``--modality``      active modality, which selects the strategy binding whose
                          ``validity_params`` define tissue for this light path
      ``--objective``     objective id, for the per-objective fallback thresholds
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
    # Reuse the streaming-AF abort signal rather than inventing a second one. From the
    # operator's side this search and the focus scan that follows it are ONE action -- the
    # Live Viewer has already turned its Autofocus button into a Cancel toggle before the
    # search starts -- so the Cancel they press must stop whichever half is running. Without
    # this the button is live but inert for the whole search, and the stage keeps stepping.
    abort_event = None
    try:
        from microscope_command_server.server.handlers.streaming_focus import (
            _client_ip,
            _get_af_abort_event,
        )

        abort_ip = _client_ip(addr)
        if abort_ip is not None:
            abort_event = _get_af_abort_event(abort_ip)
            # Clear anything left set by an earlier scan, exactly as handle_streaming_focus
            # does on entry -- a stale signal would abort this search before it began.
            abort_event.clear()
    except Exception as e:
        logger.warning(
            "FINDTISS:could not resolve the abort signal (%s); the search will not be cancellable",
            e,
        )

    try:
        message = read_message_string(conn)
    except Exception as e:
        logger.error("FINDTISS:failed to read payload from %s: %s", addr, e)
        _reply(conn, f"FAILED:payload-read-error: {e}")
        return

    params = parse_flags(
        message, ["--yaml", "--modality", "--objective", "--dir", "--step", "--max-attempts"]
    )
    direction = parse_direction(params.get("dir"))
    if params.get("dir") and direction is None:
        logger.warning(
            "FINDTISS:ignoring unusable --dir %r; searching the compass instead",
            params.get("dir"),
        )

    # Depends on the hint: an unhinted sweep has more bearings per ring, so "two rings"
    # is a different number of positions. See default_max_attempts.
    max_attempts = default_max_attempts(direction)
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

    validity_name, validity_kwargs = _resolve_validity(
        params.get("yaml"), params.get("modality"), params.get("objective")
    )
    if validity_name == "always_false":
        # A modality bound to a manual-only strategy has no automatic tissue test at all.
        # Walking the search pattern would move the stage several times and then report
        # NOTFOUND by construction, so say why instead and leave the stage alone.
        logger.info(
            "FINDTISS:modality %r uses the always_false validity check; no automatic "
            "tissue test exists for it",
            params.get("modality"),
        )
        _reply(conn, "FAILED:no-automatic-tissue-check-for-this-modality")
        return
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
        # Between positions, not mid-move: a stage move is a blocking hardware call, and
        # tearing one down part-way is how you lose track of where the stage is.
        if abort_event is not None and abort_event.is_set():
            logger.warning(
                "FINDTISS:cancelled by the operator at attempt %d/%d; returning to (%.1f, %.1f)",
                attempt,
                len(offsets),
                start_x,
                start_y,
            )
            _return_to_start(hardware, start_x, start_y, start_z)
            _reply(conn, f"ABORTED:{start_x:.3f}:{start_y:.3f}:{attempt - 1}")
            return
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
            img = _fresh_frame(hardware)
        except Exception as e:
            logger.warning(
                "FINDTISS:attempt %d/%d frame capture failed: %s", attempt, len(offsets), e
            )
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
    _return_to_start(hardware, start_x, start_y, start_z)
    _reply(conn, f"NOTFOUND:{start_x:.3f}:{start_y:.3f}:{len(offsets)}")
