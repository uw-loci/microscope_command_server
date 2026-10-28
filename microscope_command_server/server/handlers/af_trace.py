"""AFTRACE: the autofocus events after a sequence number, for the operator's display.

Payload: 8 bytes, the last sequence number the client has seen (unsigned
big-endian; 0 for none). Reply: 4-byte big-endian length, then that many bytes
of UTF-8 JSON, the dict from :meth:`AutofocusEventLog.since`. Reads only the
in-memory log, never the hardware, so it answers on either socket while a
search or an acquisition runs.
"""

import json
import logging
import struct

from microscope_command_server.server.af_events import AF_EVENTS

logger = logging.getLogger(__name__)


def _recv_exact(conn, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("client closed during AFTRACE payload")
        data += chunk
    return data


def handle_aftrace(conn, client, hardware, settings, **kwargs):
    """Send the autofocus events newer than the client's last sequence number."""
    (since,) = struct.unpack(">Q", _recv_exact(conn, 8))
    payload = json.dumps(AF_EVENTS.since(since)).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)
