"""Regression tests for five defects found by reading the two ends of the socket together.

1. ``--af-disabled`` matched its parser branch and never advanced, so the
   acquisition thread spun in the parser and the client saw RUNNING at 0 of 0.
2. A short ``FAILED`` status reply left one of the client's two reads waiting
   for the 30 s socket timeout.
3. Stage moves were accepted before CONFIG, against the server's own rule that
   nothing moves on the generic startup configuration.
4. Move payloads were unpacked from a single ``recv()``, which may be short.
5. The modality registry returned the first registered prefix, so a ``bf_if``
   name resolved through ``bf``, and ``lsm``, ``2p`` and ``confocal`` had no
   entry at all.

microscope_control is not installed on the dev machine, so it is stubbed before
the modules under test are loaded. microscope_imageprocessing is stubbed only
where it is missing too: the modality package needs the real one.

ASCII-only per project policy.
"""

import importlib.util
import struct
import sys
import threading
import types
from pathlib import Path

import pytest


def _install_stub(name, attrs=None):
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for attr, value in (attrs or {}).items():
        if not hasattr(mod, attr):
            setattr(mod, attr, value)
    return mod


_install_stub("microscope_control")
_install_stub(
    "microscope_control.hardware",
    {
        "Position": type("Position", (), {}),
        "PycromanagerHardware": type("PycromanagerHardware", (), {}),
    },
)
_install_stub(
    "microscope_control.hardware.pycromanager",
    {"PycromanagerHardware": type("PycromanagerHardware", (), {})},
)
_install_stub("microscope_control.autofocus")
_install_stub(
    "microscope_control.autofocus.core", {"AutofocusUtils": type("AutofocusUtils", (), {})}
)
try:
    import microscope_imageprocessing.correction.background  # noqa: F401
    import microscope_imageprocessing.io.writer  # noqa: F401
except ImportError:
    _install_stub("microscope_imageprocessing")
    _install_stub("microscope_imageprocessing.io")
    _install_stub("microscope_imageprocessing.io.writer", {"ome_tiff_writer": lambda *a, **k: None})
    _install_stub("microscope_imageprocessing.correction")
    _install_stub(
        "microscope_imageprocessing.correction.background",
        {"BackgroundCorrectionUtils": type("BackgroundCorrectionUtils", (), {})},
    )

from microscope_command_server.modality import get_config, registry  # noqa: E402
from microscope_command_server.modality.brightfield import BRIGHTFIELD_CONFIG  # noqa: E402
from microscope_command_server.modality.shg import SHG_CONFIG  # noqa: E402
from microscope_command_server.modality.widefield import WIDEFIELD_CONFIG  # noqa: E402
from microscope_command_server.server.handlers import position, status  # noqa: E402


def _load_workflow():
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "microscope_command_server" / "acquisition" / "workflow.py"
    spec = importlib.util.spec_from_file_location("mcs_workflow_parser_fixes", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcs_workflow_parser_fixes"] = module
    spec.loader.exec_module(module)
    return module


workflow_mod = _load_workflow()

REQUIRED = "--yaml c.yml --projects /p --sample s --scan-type bf_10x_1 --region r"


def _parse_with_deadline(message, seconds=5.0):
    """Parse on a daemon thread, so a parser that never returns fails the test
    and does not hang the whole run."""
    result = {}

    def run():
        result["params"] = workflow_mod.parse_acquisition_message(message)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "parse_acquisition_message did not return"
    return result["params"]


# ---------------------------------------------------------------- 1. parser


@pytest.mark.parametrize(
    "tail",
    ["--af-disabled", "--af-disabled --z-stack --z-start -5 --z-end 5 --z-step 1"],
    ids=["last-flag", "followed-by-z-stack"],
)
def test_af_disabled_does_not_hang_the_parser(tail):
    params = _parse_with_deadline(f"{REQUIRED} {tail} ENDOFSTR")
    assert params["af_disabled"] is True
    assert params["region_name"] == "r"


def test_flags_after_af_disabled_are_still_read():
    params = _parse_with_deadline(f"{REQUIRED} --af-disabled --z-stack --z-step 1.5")
    assert params.get("z_stack") is True
    assert params.get("z_step") == 1.5


def test_end_marker_is_removed_by_its_value():
    params = _parse_with_deadline(f"{REQUIRED} ENDOFSTR")
    assert params["region_name"] == "r"
    assert "ENDOFSTR" not in " ".join(str(v) for v in params.values())


# ---------------------------------------------------------- 2. FAILED reply


class _State:
    def __init__(self, value):
        self.value = value


class _Client:
    addr = ("127.0.0.1", 51000)


class _Conn:
    """Socket stand-in. ``chunks`` are returned one per recv(), cut to size."""

    def __init__(self, *chunks):
        self._chunks = list(chunks)
        self.sent = b""

    def recv(self, size):
        if not self._chunks:
            return b""
        head = self._chunks[0]
        out, rest = head[:size], head[size:]
        if rest:
            self._chunks[0] = rest
        else:
            self._chunks.pop(0)
        return out

    def sendall(self, data):
        self.sent += data


def _status_reply(message):
    addr = _Client.addr
    conn = _Conn()
    status.handle_status(
        conn,
        _Client(),
        None,
        {},
        acquisition_states={addr: _State("FAILED")},
        acquisition_locks={addr: threading.Lock()},
        acquisition_failure_messages={addr: message},
        acquisition_final_z={},
        acquisition_saturation_summary={},
    )
    return conn.sent


@pytest.mark.parametrize("message", ["'z'", "x", "12345678"])
def test_short_failed_reply_is_longer_than_the_first_read(message):
    reply = _status_reply(message)
    # The client reads 16 bytes and then reads again; both must find data.
    assert len(reply) > 16
    assert reply.decode().strip() == f"FAILED: {message}"


def test_long_failed_reply_is_not_padded_or_cut_short():
    message = "Saturation abort: " + "x" * 100
    assert _status_reply(message).decode() == f"FAILED: {message}"


# ------------------------------------------------------- 3 and 4. the moves


class _Position:
    def __init__(self, x=None, y=None, z=None):
        self.x, self.y, self.z = x, y, z


class _Hardware:
    def __init__(self):
        self.calls = []
        self.core = types.SimpleNamespace(is_sequence_running=lambda: False)

    def move_to_position(self, pos):
        self.calls.append(("move", pos.x, pos.y, pos.z))

    def set_z_no_wait(self, z):
        self.calls.append(("z_no_wait", z))

    def set_psg_ticks(self, angle):
        self.calls.append(("rotate", angle))

    def wait_for_rotation(self):
        pass


@pytest.fixture(autouse=True)
def _real_position(monkeypatch):
    monkeypatch.setattr(position, "Position", _Position)


MOVES = [
    (position.handle_move, struct.pack("!ff", 100.0, 200.0), ("move", 100.0, 200.0, None)),
    (position.handle_movez, struct.pack("!f", 50.0), ("move", None, None, 50.0)),
    (position.handle_movznw, struct.pack("!f", 50.0), ("z_no_wait", 50.0)),
    (position.handle_movexyz, struct.pack("!fff", 1.0, 2.0, 3.0), ("move", 1.0, 2.0, 3.0)),
    (position.handle_mover, struct.pack("!f", 7.0), ("rotate", 7.0)),
]
MOVE_IDS = ["MOVE", "MOVEZ", "MOVZNW", "MOVEXYZ", "MOVER"]


@pytest.mark.parametrize("handler,payload,expected", MOVES, ids=MOVE_IDS)
def test_move_runs_once_the_server_is_configured(handler, payload, expected):
    hardware = _Hardware()
    handler(_Conn(payload), _Client(), hardware, {}, server_configured=True)
    assert hardware.calls == [expected]


@pytest.mark.parametrize("handler,payload,expected", MOVES, ids=MOVE_IDS)
def test_move_is_refused_before_config_and_its_payload_is_consumed(handler, payload, expected):
    hardware = _Hardware()
    conn = _Conn(payload + b"getxy___")
    handler(conn, _Client(), hardware, {}, server_configured=False)
    assert hardware.calls == []
    # What is left on the socket is the next command, not the tail of this one.
    assert conn.recv(8) == b"getxy___"


@pytest.mark.parametrize("handler,payload,expected", MOVES, ids=MOVE_IDS)
def test_move_payload_split_across_two_reads_is_reassembled(handler, payload, expected):
    hardware = _Hardware()
    handler(_Conn(payload[:3], payload[3:]), _Client(), hardware, {}, server_configured=True)
    assert hardware.calls == [expected]


@pytest.mark.parametrize("handler,payload,expected", MOVES, ids=MOVE_IDS)
def test_truncated_move_payload_is_dropped_without_raising(handler, payload, expected):
    hardware = _Hardware()
    handler(_Conn(payload[:-1]), _Client(), hardware, {}, server_configured=True)
    assert hardware.calls == []


# ------------------------------------------------------------- 5. registry


@pytest.mark.parametrize("name", ["bf_if", "BF_IF_20x", "bf_if_10x_1"])
def test_bf_if_names_do_not_resolve_through_bf(name):
    assert get_config(name) is WIDEFIELD_CONFIG


@pytest.mark.parametrize("name", ["bf", "bf_10x", "Brightfield_40x"])
def test_plain_brightfield_names_still_resolve_to_brightfield(name):
    assert get_config(name) is BRIGHTFIELD_CONFIG


@pytest.mark.parametrize("name", ["shg_20x", "lsm_20x", "2p_25x", "confocal_40x"])
def test_laser_scanning_prefixes_share_the_shg_config(name):
    assert get_config(name) is SHG_CONFIG


def test_longest_prefix_wins_whatever_the_registration_order(monkeypatch):
    short, long_ = object(), object()
    for order in (("ab", "ab_cd"), ("ab_cd", "ab")):
        monkeypatch.setattr(registry, "_registry", {})
        for prefix in order:
            registry.register(prefix, long_ if prefix == "ab_cd" else short)
        assert registry.get_config("ab_cd_20x") is long_
        assert registry.get_config("ab_20x") is short


def test_unknown_and_missing_names_get_the_default():
    assert get_config("nothing_registered") is registry._default
    assert get_config(None) is registry._default
