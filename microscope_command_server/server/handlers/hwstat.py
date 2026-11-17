"""HWSTAT: which part of the microscope is not answering.

Reply: a 4-byte big-endian length, then UTF-8 JSON
``{"stage": {...}, "camera": {...}, "micromanager": {...}}``, each part
``{"ok": bool, "since": epoch seconds or null, "message": str, "failures": int}``.
The states come from failures the server sees anyway (see microscope_control's
hardware/health.py); nothing is probed, so this answers at once even while a part hangs.
"""

import json
import logging
import struct

from microscope_control.hardware.health import health

logger = logging.getLogger(__name__)


def hardware_status(hardware) -> dict:
    """The per-part status for ``hardware`` (its stage cache, when it has one)."""
    return health.snapshot(getattr(hardware, "stage_cache", None))


def handle_hwstat(conn, client, hardware, settings, **kwargs):
    """Send the per-part hardware status."""
    try:
        report = hardware_status(hardware)
    except Exception as exc:  # report, never take the socket down
        logger.error("HWSTAT failed: %s", exc)
        report = {"error": str(exc)}
    payload = json.dumps(report).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)) + payload)
    return None
