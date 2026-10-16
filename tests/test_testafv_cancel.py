"""TESTAFV hands the client's Cancel (ABORTAF) to the validation and reports CANCELLED.

The validation itself is replaced here: what is under test is the handler's wiring, that
the per-IP abort event reaches should_abort and that a cancelled result is sent as
CANCELLED rather than FAILED. The validation's own cancel is tested on the demo hardware
in test_demo_microscope.py.
"""

import socket
import sys
import types

import pytest

from microscope_command_server.server.protocol import END_MARKER


def _read_all(sock):
    sock.settimeout(5)
    chunks = []
    while True:
        try:
            data = sock.recv(4096)
        except socket.timeout:
            break
        if not data:
            break
        chunks.append(data)
    return b"".join(chunks).decode()


def test_a_cancel_reaches_the_validation_and_is_reported_as_cancelled(monkeypatch):
    from microscope_command_server.server.handlers import autofocus as af_handlers
    from microscope_command_server.server.handlers.streaming_focus import _get_af_abort_event

    seen = {}

    def fake_validation(**kwargs):
        should_abort = kwargs["should_abort"]
        seen["before"] = should_abort()
        _get_af_abort_event("127.0.0.1").set()  # what ABORTAF on the aux socket does
        seen["after"] = should_abort()
        return {"success": False, "cancelled": True, "message": "Autofocus validation cancelled"}

    fake_module = types.ModuleType("microscope_control.autofocus.test")
    fake_module.test_autofocus_validation = fake_validation
    monkeypatch.setitem(sys.modules, "microscope_control.autofocus.test", fake_module)

    server_side, client_side = socket.socketpair()
    try:
        client_side.sendall(
            ("--yaml /x/config_Demo.yml --output /x/out --objective OBJ " + END_MARKER).encode()
        )
        af_handlers.handle_testafv(server_side, ("127.0.0.1", 50000), hardware=None, settings={})
        server_side.close()
        reply = _read_all(client_side)
    finally:
        client_side.close()

    assert seen == {
        "before": False,
        "after": True,
    }, "the handler clears, then sees, the abort event"
    assert reply.startswith("STARTED:")
    assert "CANCELLED:Autofocus validation cancelled" in reply
    assert "FAILED" not in reply


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
