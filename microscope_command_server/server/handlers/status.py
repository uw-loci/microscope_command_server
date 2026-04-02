"""Acquisition status and coordination command handlers.

Handles acquisition lifecycle queries and user-interaction coordination:
STATUS, PROGRESS, CANCEL, REQMANF, ACKMF, SKIPAF, REQHWER, ACKHWER

These commands access per-client state (acquisition status, progress,
manual focus events, hardware error events) via global dictionaries
keyed by client address.
"""

import struct
import logging

logger = logging.getLogger(__name__)


def handle_status(conn, client, hardware, settings, **kwargs):
    """Return current acquisition status.

    Uses global dicts: acquisition_states, acquisition_locks,
    acquisition_failure_messages, acquisition_final_z, acquisition_saturation_summary.

    Response formats:
    - FAILED: 'FAILED: <message>' (up to 250 bytes)
    - COMPLETED: 'COMPLETED|final_z:<z>|sat:<summary>' (variable length)
    - Other: state name padded to 16 bytes
    """
    # TODO: use client.state instead of global dict
    acquisition_states = kwargs["acquisition_states"]
    acquisition_locks = kwargs["acquisition_locks"]
    acquisition_failure_messages = kwargs["acquisition_failure_messages"]
    acquisition_final_z = kwargs["acquisition_final_z"]
    acquisition_saturation_summary = kwargs["acquisition_saturation_summary"]
    addr = client.addr

    with acquisition_locks[addr]:
        state = acquisition_states[addr]
        # If state is FAILED and we have an error message, send it
        if state.value == "FAILED" and addr in acquisition_failure_messages:
            # Send "FAILED: <message>" format (truncated to fit in response)
            error_msg = acquisition_failure_messages.get(addr) or "Unknown error"
            # Java client expects to parse this format
            state_str = f"FAILED: {error_msg}"[:250]  # Reasonable limit for error message
            # Pad to 16 bytes minimum for compatibility, but can be longer
            response = state_str.encode("utf-8")
            conn.sendall(response)
            logger.debug(
                "Sent FAILED status with message to %s: %s...",
                addr, error_msg[:50],
            )
        # If state is COMPLETED, include final_z and saturation summary
        elif state.value == "COMPLETED" and addr in acquisition_final_z:
            final_z = acquisition_final_z[addr]
            state_str = f"COMPLETED|final_z:{final_z:.2f}"
            # Append saturation summary if available
            sat_summary = acquisition_saturation_summary.get(addr)
            if sat_summary:
                state_str += f"|sat:{sat_summary}"
            response = state_str.encode("utf-8")
            conn.sendall(response)
            logger.debug("Sent COMPLETED status to %s: %s", addr, state_str)
        else:
            # Send state as 16-byte string (padded)
            state_str = state.value.ljust(16)[:16]
            conn.sendall(state_str.encode())
            logger.debug("Sent acquisition status to %s: %s", addr, state.value)


def handle_progress(conn, client, hardware, settings, **kwargs):
    """Return acquisition progress as two unsigned ints (current, total).

    Response: 8 bytes (two big-endian unsigned 32-bit ints).
    """
    # TODO: use client.state instead of global dict
    acquisition_locks = kwargs["acquisition_locks"]
    acquisition_progress = kwargs["acquisition_progress"]
    addr = client.addr

    with acquisition_locks[addr]:
        current, total = acquisition_progress[addr]
    # Send as two integers
    response = struct.pack("!II", current, total)
    conn.sendall(response)
    logger.debug("Sent progress to %s: %d/%d", addr, current, total)


def handle_cancel(conn, client, hardware, settings, **kwargs):
    """Cancel a running acquisition.

    Sets acquisition state to CANCELLING and signals the cancel event.
    Response: 'ACK' (3 bytes).
    """
    # TODO: use client.state instead of global dict
    acquisition_states = kwargs["acquisition_states"]
    acquisition_locks = kwargs["acquisition_locks"]
    acquisition_cancel_events = kwargs["acquisition_cancel_events"]
    addr = client.addr

    logger.warning("Client %s requested acquisition cancellation", addr)
    with acquisition_locks[addr]:
        if acquisition_states[addr].value == "RUNNING":
            acquisition_states[addr] = _get_state_enum("CANCELLING", kwargs)
            acquisition_cancel_events[addr].set()
            logger.info("Cancellation initiated for %s", addr)
    # Send acknowledgment
    conn.sendall(b"ACK")


def handle_reqmanf(conn, client, hardware, settings, **kwargs):
    """Check if manual focus is requested by the acquisition thread.

    Response: 'NEEDEDnn' (8 bytes, nn = retries remaining 00-99)
    or 'IDLE____' (8 bytes).
    """
    # TODO: use client.state instead of global dict
    manual_focus_request_events = kwargs["manual_focus_request_events"]
    manual_focus_retries_remaining = kwargs["manual_focus_retries_remaining"]
    addr = client.addr

    if manual_focus_request_events[addr].is_set():
        # Manual focus needed - send request status with retries remaining (8 bytes exactly)
        retries = manual_focus_retries_remaining.get(addr, 0)
        # Format: "NEEDEDnn" where nn is 00-99
        response = f"NEEDED{retries:02d}".encode("utf-8")
        conn.sendall(response)
        logger.debug(
            "Sent manual focus request to %s (retries remaining: %d)",
            addr, retries,
        )
    else:
        # No manual focus needed (8 bytes exactly)
        conn.sendall(b"IDLE____")
        logger.debug("Manual focus not needed for %s", addr)


def handle_ackmf(conn, client, hardware, settings, **kwargs):
    """Acknowledge manual focus - client chose to retry autofocus.

    Response: 'ACK' (3 bytes).
    """
    # TODO: use client.state instead of global dict
    manual_focus_user_choice = kwargs["manual_focus_user_choice"]
    manual_focus_complete_events = kwargs["manual_focus_complete_events"]
    addr = client.addr

    manual_focus_user_choice[addr] = "retry"
    manual_focus_complete_events[addr].set()
    conn.sendall(b"ACK")
    logger.info("Manual focus acknowledged by client %s - will retry autofocus", addr)


def handle_skipaf(conn, client, hardware, settings, **kwargs):
    """Skip autofocus retry - client chose to use current focus position.

    Response: 'ACK' (3 bytes).
    """
    # TODO: use client.state instead of global dict
    manual_focus_user_choice = kwargs["manual_focus_user_choice"]
    manual_focus_complete_events = kwargs["manual_focus_complete_events"]
    addr = client.addr

    manual_focus_user_choice[addr] = "skip"
    manual_focus_complete_events[addr].set()
    conn.sendall(b"ACK")
    logger.info("Manual focus acknowledged by client %s - using current focus", addr)


def handle_reqhwer(conn, client, hardware, settings, **kwargs):
    """Check if hardware error recovery is requested.

    Response: 'HWERR___' (8 bytes) + 4-byte length + message bytes if error present,
    or 'IDLE____' (8 bytes) if no error.
    """
    # TODO: use client.state instead of global dict
    hardware_error_request_events = kwargs["hardware_error_request_events"]
    hardware_error_message = kwargs["hardware_error_message"]
    addr = client.addr

    if hardware_error_request_events[addr].is_set():
        # Hardware error - send error message
        err_msg = hardware_error_message.get(addr, "Unknown hardware error")
        # Encode as: 8-byte status + 4-byte length (big-endian) + message bytes
        msg_bytes = err_msg.encode("utf-8")
        length = len(msg_bytes)
        conn.sendall(b"HWERR___")  # 8-byte status: error present
        conn.sendall(length.to_bytes(4, "big"))
        conn.sendall(msg_bytes)
        logger.debug("Sent hardware error to %s: %s", addr, err_msg[:100])
    else:
        conn.sendall(b"IDLE____")  # 8 bytes: no error


def handle_ackhwer(conn, client, hardware, settings, **kwargs):
    """Acknowledge hardware error - user chose retry/skip/cancel.

    Protocol: 8 bytes (user choice padded with underscores).
    Response: 'ACK' (3 bytes).
    """
    # TODO: use client.state instead of global dict
    hardware_error_user_choice = kwargs["hardware_error_user_choice"]
    hardware_error_complete_events = kwargs["hardware_error_complete_events"]
    addr = client.addr

    # Read 8 bytes: user choice (padded to 8 with underscores)
    choice_data = conn.recv(8)
    choice = choice_data.decode("utf-8").strip().rstrip("_")
    hardware_error_user_choice[addr] = choice
    hardware_error_complete_events[addr].set()
    conn.sendall(b"ACK")
    logger.info("Hardware error acknowledged by %s: %s", addr, choice)


def _get_state_enum(state_name, kwargs):
    """Get AcquisitionState enum value by name.

    Uses the AcquisitionState class from the server module.
    """
    # TODO: use client.state instead of global dict -- this helper goes away
    # Import the enum from the same module that defines the global dicts
    from microscope_command_server.server.qp_server import AcquisitionState
    return AcquisitionState(state_name)
