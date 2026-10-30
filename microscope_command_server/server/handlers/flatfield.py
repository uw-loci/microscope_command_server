"""Flat-field profile command handler.

Handles FLATFIT: fit a BaSiC flat-field profile from a folder of tiles. Reads files only, so it needs
no hardware, but the tiles must be visible from this machine.
"""

import json
import logging
import socket

from microscope_command_server.server.handlers.utils import read_message_string

logger = logging.getLogger(__name__)


def handle_flatfit(conn, client, hardware, settings, **kwargs):
    """Fit a flat-field profile from tiles and save it.

    The payload is one JSON object, since folder paths can hold spaces:
    ``{"tiles": <folder>, "profile": <folder>, "channel": <int|null>, "offset": <float|null>,
    "label": <text>}`` then END_MARKER.

    Response: ``SUCCESS:<json>`` with offset, flatfield_min, flatfield_max, images_used and
    images_read, or ``FAILED:<reason>``.
    """
    addr = kwargs.get(
        "addr", client if isinstance(client, tuple) else getattr(client, "addr", client)
    )
    logger.info("Client %s requested a flat-field fit", addr)

    def fail(reason):
        logger.error("FLATFIT failed: %s", reason)
        conn.sendall(f"FAILED:{reason}".encode("ascii", errors="replace"))

    try:
        request = json.loads(read_message_string(conn))
        tiles = request["tiles"]
        profile_folder = request["profile"]
    except (socket.timeout, ConnectionError, ValueError, KeyError, TypeError) as e:
        fail(f"Bad FLATFIT message: {e}")
        return

    try:
        # Lazy so a server on an older microscope_imageprocessing still starts
        from microscope_imageprocessing.correction import fit_profile, read_tile_planes
    except ImportError as e:
        fail(f"microscope_imageprocessing has no flat-field fit; update it ({e})")
        return

    channel = request.get("channel")
    try:
        planes = read_tile_planes(tiles, channel=channel)
        logger.info("FLATFIT: read %d planes from %s", len(planes), tiles)
        profile = fit_profile(
            planes,
            offset=request.get("offset"),
            metadata={"source": tiles, "channel": channel, "label": request.get("label", "")},
        )
        profile.save(profile_folder)
    except Exception as e:
        logger.error("FLATFIT failed", exc_info=True)
        fail(str(e))
        return

    summary = {key: profile.metadata[key] for key in ("flatfield_min", "flatfield_max")}
    summary.update(
        offset=profile.offset,
        images_used=profile.metadata["images_used"],
        images_read=profile.metadata["images_read"],
    )
    logger.info("FLATFIT complete: %s, saved to %s", summary, profile_folder)
    conn.sendall(f"SUCCESS:{json.dumps(summary)}".encode("ascii"))
