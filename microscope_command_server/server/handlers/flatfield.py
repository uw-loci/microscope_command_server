"""Flat-field profile command handler.

Handles FLATFIT: fit a BaSiC flat-field profile from a folder of tiles. Reads files only, so it needs
no hardware, but the tiles must be visible from this machine.
"""

import json
import logging
import socket
from pathlib import Path

from microscope_command_server.server.handlers.utils import read_message_string

logger = logging.getLogger(__name__)

TILE_CONFIG = "TileConfiguration.txt"


def handle_flatfit(conn, client, hardware, settings, **kwargs):
    """Fit a flat-field profile from tiles and save it.

    The payload is one JSON object, since folder paths can hold spaces:
    ``{"tiles": <folder>, "profile": <folder>, "channel": <int|null>, "offset": <float|null>,
    "label": <text>}`` then END_MARKER.

    With no channel given and channel folders under ``tiles`` (each holding a
    TileConfiguration.txt, as a QPSC region does), every channel is fitted on its own and saved to
    ``<profile>/<channel id>/``, which is where an acquisition looks for it.

    Response: ``SUCCESS:<json>``, ``{"profiles": [...]}`` with channel, offset, flatfield_min,
    flatfield_max, images_used and images_read for each, or ``FAILED:<reason>``.
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
    tiles_path = Path(tiles)
    channel_folders = []
    if channel is None and tiles_path.is_dir():
        channel_folders = sorted(
            d for d in tiles_path.iterdir() if d.is_dir() and (d / TILE_CONFIG).is_file()
        )
    targets = [(d.name, d, Path(profile_folder) / d.name) for d in channel_folders]
    targets = targets or [(None, tiles_path, Path(profile_folder))]

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
                    "channel": channel if name is None else name,
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
        fail(str(e))
        return

    logger.info("FLATFIT complete: %s, saved to %s", summaries, profile_folder)
    conn.sendall(f"SUCCESS:{json.dumps({'profiles': summaries})}".encode("ascii"))
