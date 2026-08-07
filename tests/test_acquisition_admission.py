"""Only one acquisition may run at a time, across every connected client.

There is one stage. Two acquisitions driving it interleave their moves, so each
loop's snap lands wherever the other loop just went. Nothing errors: the
acquisition reports success, the stitch reports success, and only the pixels are
wrong -- which is how it went unnoticed through a 42-minute run on OWS3 on
2026-08-06 and surfaced as a scrambled mosaic.

The guard used to be keyed by client address, which made it per-CONNECTION. That
is not the same as per-microscope: one QuPath opens a main and an auxiliary
socket, each with its own source port and therefore its own addr, so each could
admit an acquisition while the other was already running.
"""

import sys
import threading
import types

import pytest

# The handlers package imports microscope_control, which only exists on a machine
# with a microscope attached. Admission control is pure logic, so it is stubbed
# here rather than skipped -- a guard against silently corrupting a 42-minute
# acquisition should be verifiable on any machine, not only at the scope.
if "microscope_control" not in sys.modules:
    _hardware = types.ModuleType("microscope_control.hardware")
    _hardware.Position = type("Position", (), {})
    _root = types.ModuleType("microscope_control")
    _root.hardware = _hardware
    sys.modules.setdefault("microscope_control", _root)
    sys.modules.setdefault("microscope_control.hardware", _hardware)

from microscope_command_server.server.handlers.acquisition import handle_acquire  # noqa: E402


class AcquisitionState:
    """Mirror of the server enum, as strings, which is all the handler compares."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FakeConn:
    """Socket stand-in that records what the server answered."""

    def __init__(self, message=b""):
        self._message = message
        self.sent = b""
        self.timeout = None

    def settimeout(self, value):
        self.timeout = value

    def recv(self, _size):
        chunk, self._message = self._message, b""
        return chunk

    def sendall(self, data):
        self.sent += data


ADDR_MAIN = ("127.0.0.1", 51000)
ADDR_AUX = ("127.0.0.1", 51001)


def make_kwargs(states):
    """Handler kwargs sharing one state dict, as the real server does."""
    addrs = list(states)
    return {
        "acquisition_admission_lock": threading.Lock(),
        "acquisition_locks": {a: threading.Lock() for a in addrs},
        "acquisition_states": states,
        "acquisition_progress": dict.fromkeys(addrs, (0, 0)),
        "acquisition_cancel_events": {a: threading.Event() for a in addrs},
        "AcquisitionState": AcquisitionState,
        "acquisitionWorkflow": lambda message, addr: None,
    }


@pytest.mark.parametrize("blocking_state", [AcquisitionState.RUNNING, AcquisitionState.CANCELLING])
def test_second_connection_is_refused_while_another_client_acquires(blocking_state):
    # The regression. Two addrs means two connections from the same QuPath; the
    # old per-addr guard saw an IDLE state for this one and let it through.
    states = {ADDR_MAIN: blocking_state, ADDR_AUX: AcquisitionState.IDLE}
    kwargs = make_kwargs(states)
    conn = FakeConn(b"--sample Test ENDOFSTR")

    thread = handle_acquire(conn, None, None, None, addr=ADDR_AUX, **kwargs)

    assert thread is None, "a second acquisition must not start while one is in flight"
    assert states[ADDR_AUX] == AcquisitionState.IDLE, "a refused client must not be marked RUNNING"
    assert states[ADDR_MAIN] == blocking_state, "the running acquisition must be left alone"


def test_refusal_is_answered_rather_than_left_hanging():
    # The client blocks on a 16-byte acknowledgment and requires it to begin with
    # "STARTED". Returning silently would leave it waiting on an acquisition that
    # will never run, so the refusal has to be spoken.
    states = {ADDR_MAIN: AcquisitionState.RUNNING, ADDR_AUX: AcquisitionState.IDLE}
    conn = FakeConn(b"--sample Test ENDOFSTR")

    handle_acquire(conn, None, None, None, addr=ADDR_AUX, **make_kwargs(states))

    assert len(conn.sent) == 16, "the client reads exactly 16 bytes"
    reply = conn.sent.decode().strip()
    assert reply.startswith("BUSY"), reply
    assert not reply.startswith("STARTED"), "a refusal must not read as an acceptance"


def test_an_idle_server_still_admits_an_acquisition():
    # The guard must not be so eager that it blocks the normal case; a terminal
    # state from a previous run is not an acquisition in flight.
    states = {ADDR_MAIN: AcquisitionState.COMPLETED, ADDR_AUX: AcquisitionState.IDLE}
    kwargs = make_kwargs(states)
    conn = FakeConn(b"--sample Test ENDOFSTR")

    thread = handle_acquire(conn, None, None, None, addr=ADDR_AUX, **kwargs)

    assert states[ADDR_AUX] == AcquisitionState.RUNNING
    assert conn.sent.decode().strip().startswith("STARTED")
    if thread is not None:
        thread.join(timeout=5)


def test_concurrent_requests_admit_exactly_one():
    # Both connections ask at once, which is the shape of the real failure -- the
    # decision has to be atomic, not merely checked.
    addrs = [("127.0.0.1", 52000 + i) for i in range(8)]
    states = dict.fromkeys(addrs, AcquisitionState.IDLE)
    kwargs = make_kwargs(states)
    conns = {a: FakeConn(b"--sample Test ENDOFSTR") for a in addrs}
    threads = []
    barrier = threading.Barrier(len(addrs))

    def request(addr):
        barrier.wait()
        result = handle_acquire(conns[addr], None, None, None, addr=addr, **kwargs)
        if result is not None:
            threads.append(result)

    workers = [threading.Thread(target=request, args=(a,)) for a in addrs]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=10)

    admitted = [a for a in addrs if states[a] == AcquisitionState.RUNNING]
    assert (
        len(admitted) == 1
    ), "exactly one of {} concurrent requests may be admitted, got {}".format(
        len(addrs), len(admitted)
    )
    for a in addrs:
        if a not in admitted:
            assert conns[a].sent.decode().strip().startswith("BUSY")
    for t in threads:
        t.join(timeout=5)
