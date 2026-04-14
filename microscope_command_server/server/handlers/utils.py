"""Shared utilities for command handlers.

Provides common patterns used across multiple handler groups,
such as reading variable-length message strings from the socket.
"""

import socket
import time
import logging

from microscope_command_server.server.protocol import END_MARKER

logger = logging.getLogger(__name__)


def read_message_string(conn, timeout=5.0, max_bytes=10000, chunk_size=1024):
    """Read a variable-length message string terminated by END_MARKER.

    Many commands send a variable-length text payload after the 8-byte
    command header. This function reads chunks until the END_MARKER
    sentinel is found, then strips it and returns the clean message.

    Args:
        conn: Socket connection to read from.
        timeout: Socket timeout in seconds for each recv() call.
        max_bytes: Maximum total bytes to read before aborting.
        chunk_size: Size of each recv() call.

    Returns:
        The message string with END_MARKER removed and whitespace stripped,
        or None if reading failed (timeout, connection closed, too large).

    Raises:
        socket.timeout: If the socket times out waiting for data.
        ConnectionError: If the connection is closed by the remote end.
        ValueError: If the message exceeds max_bytes.
    """
    message_parts = []
    total_bytes = 0
    start_time = time.time()

    conn.settimeout(timeout)

    try:
        while True:
            chunk = conn.recv(chunk_size)
            if not chunk:
                raise ConnectionError("Connection closed while reading message")

            message_parts.append(chunk.decode("utf-8"))
            total_bytes += len(chunk)

            full_message = "".join(message_parts)

            if END_MARKER in full_message:
                # Remove the end marker (handle both ",ENDOFSTR" and "ENDOFSTR")
                message = full_message.replace("," + END_MARKER, "").replace(
                    END_MARKER, ""
                ).strip()
                logger.debug(
                    "Read complete message (%d bytes) in %.2fs",
                    total_bytes,
                    time.time() - start_time,
                )
                return message

            # Safety check for message size
            if total_bytes > max_bytes:
                raise ValueError(
                    f"Message too large: {total_bytes} bytes (max {max_bytes})"
                )

            # Timeout check (wall clock, separate from socket timeout)
            if time.time() - start_time > timeout * 2:
                raise socket.timeout(
                    f"Timeout reading message after {time.time() - start_time:.1f}s"
                )
    finally:
        conn.settimeout(None)


def parse_flags(message, flags):
    """Parse flag-based parameters from a message string.

    Extracts values for flags like --yaml, --output, etc. from a
    space-separated message string. Handles flags whose values may
    contain spaces (e.g. file paths) by finding the next flag --
    from ANY position in the flags list -- as the delimiter.

    Order-independence is important: clients may serialize flags in
    any order, and the flags list the handler declares does not
    necessarily match the on-wire order. The previous implementation
    only scanned flags[i+1:] and break'd at the first hit, which
    meant that if the client sent '--yaml X --modality Y --range Z'
    but the handler declared flags=['--yaml', '--range', '--modality'],
    the '--yaml' value would be terminated by the NEXT declared flag
    that happened to be in the message ('--range') rather than the
    actually-closest flag ('--modality'). That caused '--modality Y'
    to be swallowed into the '--yaml' value and '--modality' itself
    to be parsed out of the leftover text, corrupting both values.

    Args:
        message: The message string to parse.
        flags: List of flag strings (e.g. ['--yaml', '--output']).

    Returns:
        Dict mapping flag names (without leading dashes, with internal
        dashes replaced by underscores) to their string values. Only
        includes flags that were found in the message.
    """
    result = {}
    for flag in flags:
        if flag not in message:
            continue
        start_idx = message.index(flag) + len(flag)
        end_idx = len(message)
        # Walk ALL declared flags (not just flags later in the list)
        # and take the nearest one after start_idx as the terminator.
        # This is the minimum-next-flag-position search, not the
        # first-match search the old implementation did.
        for other_flag in flags:
            if other_flag == flag:
                continue
            if other_flag not in message[start_idx:]:
                continue
            pos = message.index(other_flag, start_idx)
            if pos < end_idx:
                end_idx = pos
        value = message[start_idx:end_idx].strip()
        # Store with flag name minus leading dashes.
        key = flag.lstrip("-").replace("-", "_")
        result[key] = value
    return result
