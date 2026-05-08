"""Handler for the RPDSCAN (rapid scan) command.

Fast tiled brightfield acquisition with no autofocus, no Z movement,
and exposure capped at 0.5ms. Serpentine path through a rectangular region.
"""

import logging
import socket

from microscope_command_server.server.handlers.utils import read_message_string

logger = logging.getLogger(__name__)


def handle_rapid_scan(conn, client, hardware, settings, **kwargs):
    """Rapid scan handler -- fast tiled acquisition over a rectangle.

    Command: RPDSCAN (8 bytes: rpdscan_)

    Reads flags:
        --output <path>     Output folder for tiles + TileConfiguration.txt
        --center-x <float>  Center X of region (stage um)
        --center-y <float>  Center Y of region (stage um)
        --width <float>     Region width (um)
        --height <float>    Region height (um)
        --overlap <float>   Tile overlap percentage (0-50)
        --exposure <float>  Exposure time in ms (max 0.5)
        --fov-w <float>     Camera FOV width (um)
        --fov-h <float>     Camera FOV height (um)

    Response:
        SUCCESS:<n_tiles>:<elapsed_seconds>
        FAILED:<reason>
    """
    addr = kwargs.get(
        "addr",
        client if isinstance(client, tuple) else getattr(client, "addr", client),
    )
    logger.info("Client %s requested rapid scan", addr)

    try:
        message = read_message_string(conn, chunk_size=4096)
    except (socket.timeout, ConnectionError, ValueError) as e:
        logger.error("Failed to read RPDSCAN message from %s: %s", addr, e)
        conn.sendall(f"FAILED:{str(e)}".encode())
        return

    logger.info("RPDSCAN message: %s", message)

    # Parse parameters
    params = {}
    parts = message.split()
    i = 0
    while i < len(parts):
        if parts[i] == "--output" and i + 1 < len(parts):
            params["output"] = parts[i + 1]
            i += 2
        elif parts[i] == "--center-x" and i + 1 < len(parts):
            params["center_x"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--center-y" and i + 1 < len(parts):
            params["center_y"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--width" and i + 1 < len(parts):
            params["width"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--height" and i + 1 < len(parts):
            params["height"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--overlap" and i + 1 < len(parts):
            params["overlap"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--exposure" and i + 1 < len(parts):
            params["exposure"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--fov-w" and i + 1 < len(parts):
            params["fov_w"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--fov-h" and i + 1 < len(parts):
            params["fov_h"] = float(parts[i + 1])
            i += 2
        elif parts[i] == "--binning" and i + 1 < len(parts):
            params["binning"] = int(parts[i + 1])
            i += 2
        else:
            i += 1

    # Validate required params
    required = [
        "output",
        "center_x",
        "center_y",
        "width",
        "height",
        "overlap",
        "exposure",
        "fov_w",
        "fov_h",
    ]
    for req in required:
        if req not in params:
            flag_name = req.replace("_", "-")
            conn.sendall(f"FAILED:Missing --{flag_name}".encode())
            return

    # Validate exposure cap
    if params["exposure"] > 0.5:
        conn.sendall(b"FAILED:Exposure exceeds 0.5ms limit")
        return

    # Run acquisition
    try:
        from microscope_command_server.acquisition.rapid_scan import acquire_rapid_scan

        # Use server-level progress dict if available
        progress_dict = settings.get("acquisition_progress") if settings else None

        binning = int(params.get("binning", 2))

        result = acquire_rapid_scan(
            hardware=hardware,
            output_folder=params["output"],
            center_x=params["center_x"],
            center_y=params["center_y"],
            width=params["width"],
            height=params["height"],
            overlap_percent=params["overlap"],
            exposure_ms=params["exposure"],
            fov_width=params["fov_w"],
            fov_height=params["fov_h"],
            binning=binning,
            progress_dict=progress_dict,
        )

        response = (
            f"SUCCESS:{result['n_tiles']}:{result['elapsed_seconds']:.1f}" f":{result['binning']}"
        )
        conn.sendall(response.encode())
        logger.info(
            "RPDSCAN complete: %d tiles in %.1fs (binning=%d)",
            result["n_tiles"],
            result["elapsed_seconds"],
            result["binning"],
        )
    except Exception as e:
        logger.error("RPDSCAN failed: %s", e, exc_info=True)
        conn.sendall(f"FAILED:{str(e)}".encode())
