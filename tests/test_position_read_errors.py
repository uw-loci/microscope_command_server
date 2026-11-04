"""GETXY / GETZ / GETXYZ / GETZF when the stage does not answer."""

import logging
import struct

import pytest

from microscope_command_server.server.handlers import position
from microscope_control.hardware import Position
from microscope_control.hardware.stage_cache import StageUnavailableError

JAVA_ERROR = (
    'java.lang.Exception: Error in device "XYStage": Serial command failed.  '
    "Is the device connected to the serial port? (14)\n"
    "mmcorej.MMCoreJJNI.CMMCore_getXPosition__SWIG_1(Native Method)"
)


class FakeConn:
    def __init__(self):
        self.sent = b""

    def sendall(self, data):
        self.sent += data


class FakeClient:
    addr = ("127.0.0.1", 5000)


class FakeCache:
    def __init__(self, result):
        self.result = result

    def get_cached_position(self, max_age_ms=None):
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeHardware:
    def __init__(self, result):
        self.stage_cache = FakeCache(result)


@pytest.fixture(autouse=True)
def _fresh_trace_memory(monkeypatch):
    monkeypatch.setattr(position, "_traced_read_errors", set())


HANDLERS = [
    (position.handle_getxy, b"HW_ERROR"),
    (position.handle_getz, b"HWERR"),
    (position.handle_getxyz, b"HW_ERROR____"),
    (position.handle_getzf, b"HWERR"),
]


def _call(handler, result):
    conn = FakeConn()
    handler(conn, FakeClient(), FakeHardware(result), {})
    return conn.sent


@pytest.mark.parametrize("handler,marker", HANDLERS)
def test_stage_down_replies_the_error_marker_and_logs_only_at_debug(handler, marker, caplog):
    caplog.set_level(logging.DEBUG, logger=position.__name__)
    sent = _call(handler, StageUnavailableError("Stage not responding (3 failed reads over 2 s)"))
    assert sent == marker
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize("handler,marker", HANDLERS)
def test_a_repeated_error_logs_one_line_and_one_traceback(handler, marker, caplog):
    caplog.set_level(logging.DEBUG, logger=position.__name__)
    for _ in range(3):
        assert _call(handler, Exception(JAVA_ERROR)) == marker

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 3
    assert all("mmcorej" not in r.getMessage() for r in errors)
    assert [bool(r.exc_info) for r in errors] == [True, False, False]


def test_a_readable_stage_still_answers_with_floats():
    sent = _call(position.handle_getxyz, Position(1.5, -2.5, 30.0))
    assert struct.unpack("!fff", sent) == (1.5, -2.5, 30.0)
