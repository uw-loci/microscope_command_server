"""GETPROPL: report the range Micro-Manager accepts for a numeric property.

The UI bounds its intensity spinner with this, so the failure that matters is
a fabricated range: any answer other than real limits must come back as
"unavailable" (0x00) so the caller keeps its own default instead.
"""

import importlib.util
import pathlib
import struct

# Import the handler module by path: microscope_command_server.server.handlers.__init__
# pulls in microscope_control (pycromanager), which is not installed off-rig.
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "server"
    / "handlers"
    / "illumination.py"
)
_spec = importlib.util.spec_from_file_location("_illumination_handlers", _MODULE_PATH)
_illumination = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_illumination)
handle_getpropl = _illumination.handle_getpropl

UNAVAILABLE = struct.pack(">Bff", 0x00, 0.0, 0.0)


class FakeConn:
    def __init__(self, payload=b""):
        self._payload = payload
        self.sent = b""

    def recv(self, n):
        chunk, self._payload = self._payload[:n], self._payload[n:]
        return chunk

    def sendall(self, data):
        self.sent += data


class FakeClient:
    addr = ("127.0.0.1", 0)


class FakeCore:
    def __init__(self, limits=None, raises=False):
        self._limits = limits or {}
        self._raises = raises

    def has_property_limits(self, device, prop):
        if self._raises:
            raise RuntimeError("no such device")
        return (device, prop) in self._limits

    def get_property_lower_limit(self, device, prop):
        return self._limits[(device, prop)][0]

    def get_property_upper_limit(self, device, prop):
        return self._limits[(device, prop)][1]


class FakeHardware:
    def __init__(self, core):
        self.core = core


def payload(device, prop):
    return device.encode().ljust(32, b"\x00") + prop.encode().ljust(32, b"\x00")


def run(payload_bytes, core):
    conn = FakeConn(payload_bytes)
    handle_getpropl(conn, FakeClient(), FakeHardware(core), None)
    return conn.sent


def test_real_limits_are_reported():
    core = FakeCore({("DLED", "Intensity-385nm"): (0.0, 100.0)})
    sent = run(payload("DLED", "Intensity-385nm"), core)
    assert len(sent) == 9
    available, low, high = struct.unpack(">Bff", sent)
    assert available == 0x01
    assert low == 0.0
    assert high == 100.0


def test_a_property_without_limits_is_unavailable_not_zero_to_zero():
    core = FakeCore({("DLED", "Intensity-385nm"): (0.0, 100.0)})
    assert run(payload("DLED", "Enable-385nm"), core) == UNAVAILABLE


def test_an_unknown_device_is_unavailable_rather_than_an_exception():
    assert run(payload("NoSuchDevice", "Intensity"), FakeCore(raises=True)) == UNAVAILABLE


def test_degenerate_limits_are_refused():
    # A device that answers 0..0 would otherwise pin the spinner shut.
    core = FakeCore({("DiaLamp", "Intensity"): (0.0, 0.0)})
    assert run(payload("DiaLamp", "Intensity"), core) == UNAVAILABLE


def test_short_payload_is_unavailable():
    core = FakeCore({("DLED", "Intensity-385nm"): (0.0, 100.0)})
    assert run(b"DLED", core) == UNAVAILABLE


def test_empty_device_or_property_is_unavailable():
    core = FakeCore({("DLED", "Intensity-385nm"): (0.0, 100.0)})
    assert run(payload("", "Intensity-385nm"), core) == UNAVAILABLE
    assert run(payload("DLED", ""), core) == UNAVAILABLE


def test_hardware_without_a_core_is_unavailable():
    conn = FakeConn(payload("DLED", "Intensity-385nm"))

    class NoCore:
        core = None

    handle_getpropl(conn, FakeClient(), NoCore(), None)
    assert conn.sent == UNAVAILABLE


def test_the_reply_is_always_nine_bytes():
    # The Java client reads a fixed-size reply; a short one would desync the socket.
    core = FakeCore({("DLED", "Intensity-385nm"): (0.0, 100.0)})
    for p in [
        payload("DLED", "Intensity-385nm"),
        payload("DLED", "Enable-385nm"),
        b"",
        payload("", ""),
    ]:
        assert len(run(p, core)) == 9
