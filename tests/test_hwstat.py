"""HWSTAT: the per-part hardware status, answered at once from failures already seen."""

import json
import struct

from microscope_command_server.server.handlers.hwstat import handle_hwstat
from microscope_control.hardware.health import CAMERA, health


class _Conn:
    def __init__(self):
        self.sent = b""

    def sendall(self, data):
        self.sent += data

    def reply(self):
        (n,) = struct.unpack("!I", self.sent[:4])
        return json.loads(self.sent[4 : 4 + n])


class _Cache:
    def down_status(self):
        return {"ok": False, "since": 1.0, "message": "Stage not responding", "failures": 4}


class _Hardware:
    stage_cache = _Cache()


def test_the_reply_names_each_part_and_the_failing_one():
    health.succeeded(CAMERA)
    health.failed(CAMERA, RuntimeError("snap failed"))
    try:
        conn = _Conn()
        handle_hwstat(conn, None, _Hardware(), {})
        report = conn.reply()
    finally:
        health.succeeded(CAMERA)
    assert report["camera"]["ok"] is False
    assert report["camera"]["message"] == "snap failed"
    assert report["stage"]["ok"] is False and report["stage"]["known"] is True
    assert report["micromanager"]["ok"] is True


def test_hardware_without_a_stage_cache_reports_the_stage_as_unknown():
    conn = _Conn()
    handle_hwstat(conn, None, object(), {})
    assert conn.reply()["stage"]["known"] is False
