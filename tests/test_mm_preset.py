"""MMPRESET: read or apply a Micro-Manager preset, refusing what could move a turret mid-run."""

import json
import struct
import threading

import pytest

pycromanager_hw = pytest.importorskip("microscope_control.hardware.pycromanager")

from microscope_command_server.demo.virtual_microscope import DemoCore, DemoStudio  # noqa: E402
from microscope_command_server.server.handlers.presets import (  # noqa: E402
    handle_mmpreset,
    preset_report,
)


@pytest.fixture
def hardware():
    core = DemoCore(xy_speed_um_per_s=1e6, z_speed_um_per_s=1e6)
    # An "Objective Lens" group keyed on a labelling property, as on the LC-PolScope.
    core._devices[core.LAMP].properties["State"].allowed = ["0", "1"]
    core._config_groups["Lamp"] = {
        "Off": [(core.LAMP, "State", "0")],
        "On": [(core.LAMP, "State", "1")],
    }
    return pycromanager_hw.PycromanagerHardware(core, DemoStudio(core), {"modalities": {}})


class _Conn:
    def __init__(self, request):
        body = json.dumps(request).encode()
        self.inbox = struct.pack("!I", len(body)) + body
        self.sent = b""

    def recv(self, n):
        out, self.inbox = self.inbox[:n], self.inbox[n:]
        return out

    def sendall(self, data):
        self.sent += data

    def reply(self):
        (n,) = struct.unpack("!I", self.sent[:4])
        return json.loads(self.sent[4 : 4 + n])


def _call(hardware, request, **kwargs):
    conn = _Conn(request)
    kwargs.setdefault("server_configured", True)
    handle_mmpreset(conn, None, hardware, {}, **kwargs)
    return conn.reply()


def test_get_reports_the_current_preset_and_changes_nothing(hardware):
    report = preset_report(hardware.core, "Lamp", "Off")
    assert report["group_exists"] and report["preset_exists"]
    assert report["current"] == "On"
    assert hardware.core.get_property("DLamp", "State") == "1"


def test_set_applies_the_preset(hardware):
    reply = _call(hardware, {"action": "set", "group": "Lamp", "preset": "Off"})
    assert reply["applied"] and reply["previous"] == "On" and reply["current"] == "Off"
    assert hardware.core.get_property("DLamp", "State") == "0"
    assert reply["pixel_size_um"] == pytest.approx(0.65)


def test_set_to_the_current_preset_does_nothing(hardware):
    reply = _call(hardware, {"action": "set", "group": "Lamp", "preset": "On"})
    assert reply["applied"] is False and "error" not in reply


def test_names_that_do_not_exist_are_reported_with_what_does(hardware):
    reply = _call(hardware, {"action": "set", "group": "Objective", "preset": "20x"})
    assert reply["applied"] is False and "no group 'Objective'" in reply["error"]
    assert "Lamp" in reply["groups"]
    reply = _call(hardware, {"action": "set", "group": "Lamp", "preset": "Dim"})
    assert "no preset 'Dim'" in reply["error"] and reply["presets"] == ["Off", "On"]


def test_no_set_during_an_acquisition_or_before_configuration(hardware):
    running = threading.Thread(target=threading.Event().wait, args=(5,), daemon=True)
    running.start()
    reply = _call(
        hardware,
        {"action": "set", "group": "Lamp", "preset": "Off"},
        acquisition_thread=running,
    )
    assert reply["applied"] is False and "acquisition is running" in reply["error"]
    reply = _call(
        hardware, {"action": "set", "group": "Lamp", "preset": "Off"}, server_configured=False
    )
    assert reply["applied"] is False and "not configured" in reply["error"]
    assert hardware.core.get_property("DLamp", "State") == "1"
