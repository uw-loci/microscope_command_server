"""Position and movement command handlers.

Handles stage position queries and movement commands:
GETXY, GETZ, GETXYZ, GETFOV, GETPXSZ, GETR, GETZF,
MOVE, MOVEZ, MOVZNW, MOVEXYZ, MOVER
"""

import struct
import time
import logging

from microscope_control.hardware import Position

logger = logging.getLogger(__name__)


def handle_getxy(conn, client, hardware, settings, **kwargs):
    """Return current XY stage position as two big-endian floats."""
    logger.debug("Client %s requested XY position", client.addr)
    try:
        pos = hardware.get_current_position()
        conn.sendall(struct.pack("!ff", pos.x, pos.y))
        logger.debug("Sent XY position: (%.1f, %.1f)", pos.x, pos.y)
    except Exception as e:
        logger.error("Failed to get XY position: %s", e, exc_info=True)
        conn.sendall(b"HW_ERROR")


def handle_getz(conn, client, hardware, settings, **kwargs):
    """Return current Z position as one big-endian float."""
    logger.debug("Client %s requested Z position", client.addr)
    try:
        pos = hardware.get_current_position()
        conn.sendall(struct.pack("!f", pos.z))
        logger.debug("Sent Z position: %.2f", pos.z)
    except Exception as e:
        logger.error("Failed to get Z position: %s", e, exc_info=True)
        conn.sendall(b"HWERR")


def handle_getxyz(conn, client, hardware, settings, **kwargs):
    """Return current XYZ position as three big-endian floats."""
    logger.debug("Client %s requested XYZ position", client.addr)
    try:
        pos = hardware.get_current_position()
        conn.sendall(struct.pack("!fff", pos.x, pos.y, pos.z))
        logger.debug("Sent XYZ: (%.1f, %.1f, %.2f)", pos.x, pos.y, pos.z)
    except Exception as e:
        logger.error("Failed to get XYZ position: %s", e, exc_info=True)
        conn.sendall(struct.pack("!fff", 0.0, 0.0, 0.0))


def handle_getfov(conn, client, hardware, settings, **kwargs):
    """Return camera field of view as two big-endian floats (um)."""
    server_configured = kwargs.get("server_configured", False)
    if not server_configured:
        logger.warning("GETFOV: Blocked - server not configured")
        conn.sendall(struct.pack("!ff", -1.0, -1.0))
        return
    try:
        fov_x, fov_y = hardware.get_fov()
        conn.sendall(struct.pack("!ff", fov_x, fov_y))
        logger.debug("Sent FOV: (%.1f, %.1f)", fov_x, fov_y)
    except Exception as e:
        logger.error("Failed to get FOV: %s", e)
        conn.sendall(struct.pack("!ff", 0.0, 0.0))


def handle_getpxsz(conn, client, hardware, settings, **kwargs):
    """Return pixel size in micrometers as one big-endian float."""
    server_configured = kwargs.get("server_configured", False)
    if not server_configured:
        logger.warning("GETPXSZ: Blocked - server not configured")
        conn.sendall(struct.pack("!f", 0.0))
        return
    try:
        pixel_size = hardware.get_pixel_size_um()
        conn.sendall(struct.pack("!f", float(pixel_size)))
        logger.debug("Sent pixel size: %.4f um/px", pixel_size)
    except Exception as e:
        logger.error("Failed to get pixel size: %s", e)
        conn.sendall(struct.pack("!f", 0.0))


def handle_getr(conn, client, hardware, settings, **kwargs):
    """Return current rotation angle as one big-endian float (degrees).

    Returns NaN if no rotation stage is configured (not an error condition).
    """
    logger.debug("Client %s requested rotation angle", client.addr)
    try:
        angle = hardware.get_psg_ticks()
        conn.sendall(struct.pack("!f", angle))
        logger.debug("Sent rotation angle: %.1f deg", angle)
    except RuntimeError as e:
        if "No rotation stage" in str(e):
            # Not an error -- microscope simply has no rotation hardware
            conn.sendall(struct.pack("!f", float("nan")))
            logger.debug("No rotation stage configured, returned NaN")
        else:
            logger.error("Failed to get rotation angle: %s", e, exc_info=True)
            conn.sendall(b"HWERR")
    except Exception as e:
        logger.error("Failed to get rotation angle: %s", e, exc_info=True)
        conn.sendall(b"HWERR")


def handle_getzf(conn, client, hardware, settings, **kwargs):
    """Return fast Z position (no XY read) as one big-endian float."""
    try:
        z = hardware.get_z_position()
        conn.sendall(struct.pack("!f", z))
    except Exception as e:
        logger.error("Failed fast Z read: %s", e, exc_info=True)
        conn.sendall(struct.pack("!f", 0.0))


def handle_move(conn, client, hardware, settings, **kwargs):
    """Move XY stage to position (read 8 bytes: two floats)."""
    coords = conn.recv(8)
    if len(coords) == 8:
        x, y = struct.unpack("!ff", coords)
        logger.info("Client %s requested move to: X=%.1f, Y=%.1f", client.addr, x, y)
        try:
            t0 = time.perf_counter()
            hardware.move_to_position(Position(x, y))
            t_ms = (time.perf_counter() - t0) * 1000
            logger.info("MOVE completed to X=%.1f, Y=%.1f in %.0fms", x, y, t_ms)
        except Exception as e:
            logger.error("Failed to move to XY position: %s", e, exc_info=True)
    else:
        logger.error("Client %s sent incomplete move coordinates", client.addr)


def handle_movez(conn, client, hardware, settings, **kwargs):
    """Move Z stage to position (read 4 bytes: one float)."""
    z = conn.recv(4)
    z_position = struct.unpack("!f", z)[0]
    logger.info("Client %s requested move to Z=%.2f", client.addr, z_position)
    try:
        hardware.move_to_position(Position(z=z_position))
        logger.info("Move completed to Z=%.2f", z_position)
    except Exception as e:
        logger.error("Failed to move to Z position: %s", e, exc_info=True)


def handle_movznw(conn, client, hardware, settings, **kwargs):
    """Non-blocking Z move (for sweep focus). Read 4 bytes: one float."""
    z = conn.recv(4)
    z_position = struct.unpack("!f", z)[0]
    logger.debug("Client %s non-blocking Z move to %.2f", client.addr, z_position)
    try:
        hardware.set_z_no_wait(z_position)
    except Exception as e:
        logger.error("Failed non-blocking Z move: %s", e, exc_info=True)


def handle_movexyz(conn, client, hardware, settings, **kwargs):
    """Move to XYZ position (read 12 bytes: three floats)."""
    xyz_data = conn.recv(12)
    x, y, z = struct.unpack("!fff", xyz_data)
    logger.info("Client %s requested move to XYZ=(%.1f, %.1f, %.2f)", client.addr, x, y, z)
    try:
        hardware.move_to_position(Position(x, y, z))
        logger.info("Successfully moved to XYZ: (%.1f, %.1f, %.2f)", x, y, z)
    except Exception as e:
        logger.error("Failed to move to XYZ (%.1f, %.1f, %.2f): %s", x, y, z, e, exc_info=True)


def handle_mover(conn, client, hardware, settings, **kwargs):
    """Move rotation stage (read 4 bytes: one float, angle in degrees)."""
    coords = conn.recv(4)
    angle = struct.unpack("!f", coords)[0]
    logger.info("Client %s requested rotation to %.1f deg", client.addr, angle)
    try:
        hardware.set_psg_ticks(angle)
        hardware.wait_for_rotation()
        logger.info("Rotation completed to %.1f deg", angle)
    except Exception as e:
        logger.error("Failed to rotate to %.1f deg: %s", angle, e, exc_info=True)
