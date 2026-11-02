"""TESTADAF: which search runs, and what the client is told."""

import socket

from microscope_command_server.server.handlers.autofocus import handle_testadaf
from microscope_command_server.server.protocol import END_MARKER


def _run(monkeypatch, message, sweep=None, search=None):
    import microscope_control.autofocus.test as af

    calls = {}

    def record(name, reply):
        def fake(**kwargs):
            calls[name] = kwargs
            return reply

        return fake

    monkeypatch.setattr(af, "test_adaptive_autofocus_at_current_position", record("sweep", sweep))
    monkeypatch.setattr(af, "focus_search_at_current_position", record("search", search))
    server_side, client_side = socket.socketpair()
    try:
        client_side.sendall((message + " " + END_MARKER).encode())
        handle_testadaf(server_side, ("127.0.0.1", 0), None, None)
        client_side.settimeout(5)
        return calls, client_side.recv(65536).decode()
    finally:
        server_side.close()
        client_side.close()


FOUND = {
    "success": True,
    "initial_z": 53.4,
    "final_z": 33.1,
    "z_shift": -20.3,
    "message": "Focus search completed. Z shift: -20.30 um.",
}
PATH = "--yaml C:\\a b\\c.yml --output C:\\out --objective OBJ_20X"


def test_default_is_still_a_drift_check(monkeypatch):
    calls, _ = _run(monkeypatch, PATH, sweep=FOUND)
    assert list(calls) == ["sweep"]
    assert calls["sweep"]["objective"] == "OBJ_20X"


def test_wide_search_flag_runs_the_search(monkeypatch):
    calls, reply = _run(monkeypatch, PATH + " --wide-search", search=FOUND)
    assert list(calls) == ["search"]
    assert calls["search"]["objective"] == "OBJ_20X"
    assert calls["search"]["yaml_file_path"] == "C:\\a b\\c.yml"
    assert calls["search"]["should_abort"] is not None
    assert reply.endswith("SUCCESS:Focus search completed. Z shift: -20.30 um.|53.40:33.10:-20.30")


def test_no_focus_found_reaches_the_client_as_failed(monkeypatch):
    held = {"success": False, "message": "No focus found, Z left at 53.40 um: no peak"}
    _, reply = _run(monkeypatch, PATH + " --wide-search", search=held)
    assert reply.endswith("FAILED:No focus found, Z left at 53.40 um: no peak")


def test_cancelled_search_reports_cancelled(monkeypatch):
    done = {"cancelled": True, "success": False, "message": "Focus search cancelled"}
    _, reply = _run(monkeypatch, PATH + " --wide-search", search=done)
    assert reply.endswith("CANCELLED:Focus search cancelled")
