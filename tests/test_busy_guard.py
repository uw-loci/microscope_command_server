"""Stage-moving commands refuse to run under someone else's acquisition.

The 2026-10-12 case: the client timed out of an acquisition without cancelling it, the
server thread sat at a manual-focus prompt in state RUNNING, and the alignment gate's
STRMAFZ then moved Z by 30 um under that thread before the next ACQUIRE was refused.
"""

from enum import Enum
from pathlib import Path

from microscope_command_server.server.handlers.busy import acquisition_in_progress, busy_reply

HANDLER_SRC = (
    Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "server"
    / "handlers"
    / "streaming_focus.py"
)


class State(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


MAIN = ("127.0.0.1", 25460)
AUX = ("127.0.0.1", 25461)


def test_nothing_running_is_not_busy():
    assert acquisition_in_progress({MAIN: State.IDLE, AUX: State.COMPLETED}) is None
    assert acquisition_in_progress({}) is None
    assert acquisition_in_progress(None) is None


def test_a_running_acquisition_on_any_socket_is_busy():
    # The acquisition may be on the other of QuPath's two sockets.
    assert acquisition_in_progress({AUX: State.IDLE, MAIN: State.RUNNING}) == MAIN


def test_cancelling_still_owns_the_stage():
    # A cancel is acknowledged before the thread has stopped moving.
    assert acquisition_in_progress({MAIN: State.CANCELLING}) == MAIN


def test_finished_states_release_it():
    for done in (State.COMPLETED, State.FAILED):
        assert acquisition_in_progress({MAIN: done}) is None


def test_string_states_work_too():
    # The admission tests mirror the enum as plain strings; so may older callers.
    assert acquisition_in_progress({MAIN: "RUNNING"}) == MAIN
    assert acquisition_in_progress({MAIN: "IDLE"}) is None


def test_the_reply_is_a_failed_the_client_already_understands():
    reply = busy_reply(MAIN, "move the focus stage")
    assert reply.startswith("FAILED:")
    assert "25460" in reply
    assert reply.isascii()


def test_streaming_focus_asks_before_it_touches_the_stage():
    """The guard has to precede the first hardware access, or it is decoration."""
    src = HANDLER_SRC.read_text()
    handler_at = src.index("def handle_streaming_focus(")
    body = src[handler_at:]
    guard_at = body.index('acquisition_in_progress(kwargs.get("acquisition_states"))')
    first_hardware = body.index("STREAM_AF:focus device =")
    assert guard_at < first_hardware
    # And a refusal returns rather than falling through into the scan.
    refusal = body[guard_at : guard_at + 500]
    assert "return" in refusal
