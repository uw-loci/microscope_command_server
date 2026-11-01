"""Flat-field profile command handlers.

FLATFIT starts a BaSiC fit from a folder of tiles and FLATPOLL asks how it went. The fit can take
minutes, and a client must not hold its command socket that long: cancel and status polls share it.
It reads files only, so it needs no hardware, but the tiles must be visible from this machine.
"""

import json
import logging
import socket
import threading
from pathlib import Path

from microscope_command_server.server.handlers.utils import read_message_string

logger = logging.getLogger(__name__)

TILE_CONFIG = "TileConfiguration.txt"

_job_lock = threading.Lock()
_job_thread = None
_job_result = None


def _fit(request):
    """Run the fit; returns the FLATPOLL reply."""
    try:
        # Lazy so a server on an older microscope_imageprocessing still starts
        from microscope_imageprocessing.correction import fit_profile, read_tile_planes
    except ImportError as e:
        return f"FAILED:microscope_imageprocessing has no flat-field fit; update it ({e})"

    channel = request.get("channel")
    tiles_path = Path(request["tiles"])
    profile_folder = Path(request["profile"])
    channel_folders = []
    if channel is None and tiles_path.is_dir():
        channel_folders = sorted(
            d for d in tiles_path.iterdir() if d.is_dir() and (d / TILE_CONFIG).is_file()
        )
    targets = [(d.name, d, profile_folder / d.name) for d in channel_folders]
    targets = targets or [(None, tiles_path, profile_folder)]

    summaries = []
    try:
        for name, source, destination in targets:
            planes = read_tile_planes(source, channel=channel)
            logger.info("FLATFIT: read %d planes from %s", len(planes), source)
            profile = fit_profile(
                planes,
                offset=request.get("offset"),
                metadata={
                    "source": str(source),
                    "channel": channel,  # a MicroManager index; Tiles to Pyramid reads it
                    "channel_id": name,
                    "label": request.get("label", ""),
                },
            )
            profile.save(destination)
            summaries.append(
                {
                    "channel": name,
                    "offset": profile.offset,
                    "flatfield_min": profile.metadata["flatfield_min"],
                    "flatfield_max": profile.metadata["flatfield_max"],
                    "images_used": profile.metadata["images_used"],
                    "images_read": profile.metadata["images_read"],
                }
            )
    except Exception as e:
        logger.error("FLATFIT failed", exc_info=True)
        return f"FAILED:{e}"

    logger.info("FLATFIT complete: %s, saved to %s", summaries, profile_folder)
    return f"SUCCESS:{json.dumps({'profiles': summaries})}"


def _run_job(request):
    global _job_result
    result = _fit(request)
    with _job_lock:
        _job_result = result


def handle_flatfit(conn, client, hardware, settings, **kwargs):
    """Start fitting a flat-field profile from tiles.

    The payload is one JSON object, since folder paths can hold spaces:
    ``{"tiles": <folder>, "profile": <folder>, "channel": <int|null>, "offset": <float|null>,
    "label": <text>}`` then END_MARKER.

    With no channel given and channel folders under ``tiles`` (each holding a
    TileConfiguration.txt, as a QPSC region does), every channel is fitted on its own and saved to
    ``<profile>/<channel id>/``, which is where an acquisition looks for it.

    Response, newline-terminated: ``STARTED``, or ``FAILED:<reason>``.
    """
    global _job_thread, _job_result
    addr = kwargs.get(
        "addr", client if isinstance(client, tuple) else getattr(client, "addr", client)
    )
    logger.info("Client %s requested a flat-field fit", addr)

    def reply(text):
        conn.sendall(f"{text}\n".encode("ascii", errors="replace"))

    try:
        request = json.loads(read_message_string(conn))
        request["tiles"], request["profile"]
    except (socket.timeout, ConnectionError, ValueError, KeyError, TypeError) as e:
        logger.error("Bad FLATFIT message: %s", e)
        reply(f"FAILED:Bad FLATFIT message: {e}")
        return

    with _job_lock:
        if _job_thread is not None and _job_thread.is_alive():
            reply("FAILED:A flat-field fit is already running")
            return
        _job_result = None
        _job_thread = threading.Thread(
            target=_run_job, args=(request,), name="flatfield-fit", daemon=True
        )
        _job_thread.start()
    reply("STARTED")


def handle_flatpoll(conn, client, hardware, settings, **kwargs):
    """Report the fit: ``RUNNING``, ``SUCCESS:<json>`` or ``FAILED:<reason>``, newline-terminated.

    ``SUCCESS:{"profiles": [...]}`` has channel, offset, flatfield_min, flatfield_max, images_used
    and images_read for each profile. The result stays until the next FLATFIT.
    """
    with _job_lock:
        if _job_thread is None:
            result = "FAILED:No flat-field fit has been started"
        else:
            result = _job_result or "RUNNING"
    conn.sendall(f"{result}\n".encode("ascii", errors="replace"))
