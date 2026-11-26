"""Record a demo session, replay it, and catch deliberate changes to the code under test.

The checked-in trace under ``tests/data/trace_demo_session`` was recorded by
``python -m tests.trace_support DIR --no-autofocus``: CONFIG, position reads,
moves, SETEXP, three SNAPs, a MOVEZ and a four-tile Brightfield acquisition
with autofocus off, over the demo microscope with frames stored as digests.
Replaying it against the unchanged code must give zero differences; a flipped
stage sign or a shifted tile order must show up as grouped differences.
"""

import hashlib
import json
import socket
import struct
from pathlib import Path

import pytest

pycromanager_hw = pytest.importorskip("microscope_control.hardware.pycromanager")

from microscope_control.trace import Trace  # noqa: E402
from microscope_control.trace.diff import KIND_ARGS, KIND_REPLY_VALUE  # noqa: E402

from microscope_command_server.server.protocol import END_MARKER, ExtendedCommand  # noqa: E402
from microscope_command_server.trace import (  # noqa: E402
    RecordingConnection,
    SessionRecorder,
    command_class,
    command_name,
    command_scope,
)
from microscope_command_server.trace.recorder import LARGE_REPLY_BYTES  # noqa: E402
from microscope_command_server.trace.replayer import (  # noqa: E402
    ReplayConnection,
    rewrite_payload,
    replay_trace,
)
from tests.trace_support import CHECKED_IN_TRACE, DEMO_BUNDLE, run_session  # noqa: E402

EXPECTED_COMMANDS = (
    ["CONFIG", "GETXYZ", "MOVE", "GETXY", "SETEXP"]
    + ["MOVE", "SNAP"] * 3
    + [
        "MOVEZ",
        "ACQUIRE",
        "STATUS",
        "PROGRESS",
        "STATUS",
        "PROGRESS",
        "STATUS",
        "PROGRESS",
        "STATUS",
    ]
    + ["DISCONNECT"]
)


def _ascii_only(path: Path) -> bool:
    return all(b < 128 for b in path.read_bytes())


# -- the trace format ------------------------------------------------------------


def test_checked_in_trace_has_the_documented_shape():
    trace = Trace(CHECKED_IN_TRACE)
    assert trace.manifest["format_version"] == 1
    assert trace.manifest["demo"] is True and trace.manifest["frame_mode"] == "hash"
    for key in ("config_files", "inputs", "paths", "connections", "versions", "git", "start_time"):
        assert key in trace.manifest, key
    assert [c["command"] for c in trace.commands()] == EXPECTED_COMMANDS
    assert {c["class"] for c in trace.commands()} == {"action", "read", "poll"}
    assert (CHECKED_IN_TRACE / "config" / "config_Demo.yml").exists()
    assert (CHECKED_IN_TRACE / "config" / "resources" / "resources_LOCI.yml").exists()
    assert (CHECKED_IN_TRACE / "inputs" / "stage_13" / "TileConfiguration.txt").exists()
    for name in ("manifest.json", "hardware.jsonl", "socket.jsonl"):
        assert _ascii_only(CHECKED_IN_TRACE / name), name
    # Frames are digests, never pixels.
    for rec in trace.hardware_calls():
        if rec["method"] == "get_tagged_image":
            pix = rec["ret"]["__tagged_image__"]["pix"]["__ndarray__"]
            assert set(pix) == {"shape", "dtype", "digest", "min", "max", "mean"}
    assert not (CHECKED_IN_TRACE / "frames").exists()


def test_command_names_and_classes():
    assert command_name(ExtendedCommand.ACQUIRE) == "ACQUIRE"
    assert command_name(b"zzzzzzzz") == "UNKNOWN:7a7a7a7a7a7a7a7a"
    assert command_class("ACQUIRE") == "action"
    assert command_class("STATUS") == "poll" and command_class("GETXY") == "read"


# -- the recording connection -----------------------------------------------------


class _FakeSocket:
    def __init__(self, inbound: bytes):
        self.inbound = inbound
        self.sent = b""
        self.closed = False

    def recv(self, n):
        out, self.inbound = self.inbound[:n], self.inbound[n:]
        return out

    def sendall(self, data):
        self.sent += data

    def settimeout(self, t):
        self.timeout = t

    def close(self):
        self.closed = True


def test_recording_connection_records_both_directions_and_copies_the_config(tmp_path):
    recorder = SessionRecorder(tmp_path / "traces", frame_mode="hash", demo=True, label="t")
    yaml_path = DEMO_BUNDLE / "config_Demo.yml"
    path = str(yaml_path).encode()
    sock = _FakeSocket(struct.pack("!I", len(path)) + path)
    conn = recorder.wrap_connection(sock, ("127.0.0.1", 50001))
    assert isinstance(conn, RecordingConnection)
    conn.settimeout(5.0)  # passes through
    assert sock.timeout == 5.0
    with command_scope(conn, ExtendedCommand.CONFIG):
        (n,) = struct.unpack("!I", conn.recv(4))
        assert conn.recv(n) == path
        conn.sendall(b"CFG___OK")
        conn.sendall(b"x" * (LARGE_REPLY_BYTES + 1))
    conn.close()
    recorder.close()

    trace = Trace(recorder.directory)
    kinds = [(r["kind"], r.get("command")) for r in trace.socket]
    assert kinds[:2] == [("connect", ""), ("command", "CONFIG")]
    c2s = [r for r in trace.socket if r["kind"] == "c2s"]
    assert [r["n"] for r in c2s] == [4, len(path)]
    s2c = [r for r in trace.socket if r["kind"] == "s2c"]
    assert bytes.fromhex(s2c[0]["bytes"]) == b"CFG___OK"
    assert (
        "bytes" not in s2c[1]
        and s2c[1]["digest"]
        == hashlib.blake2b(b"x" * (LARGE_REPLY_BYTES + 1), digest_size=16).hexdigest()
    )
    assert trace.socket[-1]["kind"] == "close" and sock.closed
    entry = trace.manifest["config_files"][0]
    assert entry["dir"] == str(DEMO_BUNDLE) and "config_Demo.yml" in entry["copied"]
    assert (recorder.directory / "config" / "config_Demo.yml").exists()
    assert trace.manifest["connections"] == [{"conn": 1, "addr": ["127.0.0.1", 50001]}]


def test_polls_do_not_take_over_the_stage_label(tmp_path):
    recorder = SessionRecorder(tmp_path / "traces", frame_mode="hash", demo=True, label="t")
    conn = recorder.wrap_connection(_FakeSocket(b""), ("127.0.0.1", 1))
    with command_scope(conn, ExtendedCommand.ACQUIRE):
        pass
    assert recorder.writer.stage == 1 and recorder.writer.stage_name == "ACQUIRE"
    with command_scope(conn, ExtendedCommand.STATUS):
        pass
    with command_scope(conn, ExtendedCommand.GETXYZ):
        pass
    assert recorder.writer.stage == 1  # hardware calls during the polls still belong to ACQUIRE
    with command_scope(conn, ExtendedCommand.MOVE):
        pass
    assert recorder.writer.stage == 4
    recorder.close()


def test_command_scope_is_a_no_op_for_a_plain_socket():
    a, b = socket.socketpair()
    try:
        with command_scope(a, ExtendedCommand.STATUS):
            a.sendall(b"hi")
        assert b.recv(2) == b"hi"
    finally:
        a.close()
        b.close()


def test_acquire_messages_copy_the_tile_configuration(tmp_path):
    region = tmp_path / "proj" / "S" / "scan" / "r1"
    region.mkdir(parents=True)
    (region / "TileConfiguration.txt").write_text("dim = 2\n")
    recorder = SessionRecorder(tmp_path / "traces", frame_mode="hash", demo=True, label="t")
    message = f"--yaml {tmp_path}/c.yml --projects {tmp_path}/proj --sample S --scan-type scan --region r1 {END_MARKER}"
    conn = recorder.wrap_connection(_FakeSocket(message.encode()), ("127.0.0.1", 1))
    with command_scope(conn, ExtendedCommand.ACQUIRE):
        conn.recv(65536)
    recorder.close()
    manifest = Trace(recorder.directory).manifest
    assert (recorder.directory / "inputs" / "stage_1" / "TileConfiguration.txt").exists()
    assert manifest["inputs"][0]["region_dir"] == str(region)
    assert {p["flag"] for p in manifest["paths"]} == {"yaml", "projects"}


# -- payload rewriting ---------------------------------------------------------------


def test_rewrite_payload_relocates_paths_and_keeps_binary_payloads():
    rewrites = [("C:\\Users\\scope\\configs", "/out/config")]
    text = f"--yaml C:\\Users\\scope\\configs\\config_PPM.yml --exposure 10 {END_MARKER}".encode()
    out = rewrite_payload("SNAP", text, rewrites).decode()
    assert out.startswith("--yaml /out/config/config_PPM.yml --exposure 10")
    cfg = b"C:\\Users\\scope\\configs\\config_PPM.yml"
    out = rewrite_payload("CONFIG", struct.pack("!I", len(cfg)) + cfg, rewrites)
    (n,) = struct.unpack("!I", out[:4])
    assert out[4 : 4 + n] == b"/out/config/config_PPM.yml"
    floats = struct.pack("!ff", 1.0, 2.0)
    assert rewrite_payload("MOVE", floats, rewrites) == floats


def test_config_reply_versions_are_environment_not_behavior():
    from microscope_command_server.trace.replayer import _compare_reply

    def reply(versions):
        body = json.dumps(versions).encode()
        return b"CFG___OK" + struct.pack("!I", len(body)) + body

    recorded = reply({"Microscope Command Server": "1.6.0", "ppm-library": "1.2.0"})
    records = [{"n": len(recorded), "bytes": recorded.hex()}]
    other_versions = reply({"Microscope Command Server": "1.7.0", "ppm-library": "not installed"})
    assert _compare_reply(1, "CONFIG", records, other_versions, 1e-6) == []
    fewer_keys = reply({"Microscope Command Server": "1.6.0"})
    diffs = _compare_reply(1, "CONFIG", records, fewer_keys, 1e-6)
    assert [d.field for d in diffs] == ["keys"] and diffs[0].expected == ["ppm-library"]
    failed = b"CFG_FAIL" + struct.pack("!I", 2) + b"{}"
    assert [d.field for d in _compare_reply(1, "CONFIG", records, failed, 1e-6)] == ["status"]


# -- the replay connection -------------------------------------------------------------


def test_replay_connection_blocks_times_out_and_closes():
    from microscope_command_server.trace.replayer import ReplayReport

    conn = ReplayConnection(1, ("127.0.0.1", 1), ReplayReport(Path("."), Path(".")))
    conn.settimeout(0.05)
    with pytest.raises(socket.timeout):
        conn.recv(8)
    conn.push(1, "MOVE", b"abc")
    assert conn.recv(2) == b"ab" and conn.recv(8) == b"c"
    conn.sendall(b"reply")
    assert conn.take(5, 1.0) == b"reply"
    assert conn.take(3, 0.05) == b""
    conn.close_inbound()
    assert conn.recv(8) == b""


# -- end to end: record, replay, alter -------------------------------------------------


def test_the_checked_in_trace_replays_against_the_current_code_with_no_differences(tmp_path):
    report = replay_trace(CHECKED_IN_TRACE, output_dir=tmp_path / "replay", stage_timeout_s=60.0)
    assert report.stages == len(EXPECTED_COMMANDS)
    assert report.hardware_summary.startswith("48/48 recorded hardware commands consumed")
    assert report.differences == [], report.text()
    assert report.differences_path.exists()
    summary = json.loads((tmp_path / "replay" / "replay_summary.json").read_text())
    assert summary["differences"] == 0
    # The acquisition wrote its tiles where the rewritten --projects pointed.
    tiles = list((tmp_path / "replay" / "paths").rglob("tile_*.tif"))
    assert len(tiles) >= 4, tiles


def test_a_flipped_stage_sign_is_caught_and_grouped(tmp_path, monkeypatch):
    stage_mod = pytest.importorskip("microscope_control.hardware.stage")
    for name in ("move_xy", "move_xy_no_wait"):
        original = getattr(stage_mod.PycromanagerStage, name)

        def flipped(self, x, y, _original=original):
            _original(self, -x, y)

        monkeypatch.setattr(stage_mod.PycromanagerStage, name, flipped)
    report = replay_trace(CHECKED_IN_TRACE, output_dir=tmp_path / "replay", stage_timeout_s=60.0)
    assert report.differences, "the sign flip went unnoticed"
    assert {d.kind for d in report.differences} == {KIND_ARGS}
    groups = report.grouped()
    assert set(groups) == {
        "MOVE|hardware_call_args|core.set_xy_position|args[0]",
        "ACQUIRE|hardware_call_args|core.set_xy_position|args[0]",
    }
    # Moves to x=0 flip to -0.0, which is within tolerance: only the non-zero X moves differ.
    assert len(groups["MOVE|hardware_call_args|core.set_xy_position|args[0]"]) == 2
    assert len(groups["ACQUIRE|hardware_call_args|core.set_xy_position|args[0]"]) == 2
    first = groups["MOVE|hardware_call_args|core.set_xy_position|args[0]"][0]
    assert first.expected == [1000.0, 2000.0] and first.actual == [-1000.0, 2000.0]


def test_a_shifted_tile_order_is_caught_under_the_acquire_stage(tmp_path, monkeypatch):
    from microscope_command_server.acquisition import workflow

    original = workflow.TileConfigUtils.read_tile_config

    def shifted(tile_config_path, core):
        positions = original(tile_config_path, core)
        return positions[1:] + positions[:1]

    monkeypatch.setattr(workflow.TileConfigUtils, "read_tile_config", staticmethod(shifted))
    report = replay_trace(CHECKED_IN_TRACE, output_dir=tmp_path / "replay", stage_timeout_s=60.0)
    assert report.differences, "the shifted tile order went unnoticed"
    # The first thing the replay sees is the first tile's move going to the second tile's place.
    first = report.differences[0]
    assert first.fingerprint == "ACQUIRE|hardware_call_args|core.set_xy_position|args[0]"
    assert first.expected == [0.0, 0.0] and first.actual == [300.0, 0.0]
    # Everything after it is the same run going wrong, and it all belongs to the acquisition.
    assert {d.command for d in report.differences} <= {"ACQUIRE", "STATUS"}
    assert "first difference: [stage 13 ACQUIRE] hardware_call_args" in report.text()


def test_a_fresh_recording_of_the_same_session_matches_the_checked_in_trace(tmp_path):
    """The recorder is deterministic for this session and the checked-in trace is current."""
    trace_dir = run_session(
        tmp_path / "traces", tmp_path / "work", with_autofocus=False, label="fresh"
    )
    fresh, pinned = Trace(trace_dir), Trace(CHECKED_IN_TRACE)

    def actions(trace):  # how often the client polled depends on timing
        return [c["command"] for c in trace.commands() if c["class"] != "poll"]

    assert actions(fresh) == actions(pinned)

    def commands(trace):
        from microscope_control.trace.replay import is_query

        return [
            (r["method"], r["args"]) for r in trace.hardware_calls() if not is_query(r["method"])
        ]

    assert commands(fresh) == commands(pinned)
    assert _ascii_only(trace_dir / "hardware.jsonl") and _ascii_only(trace_dir / "socket.jsonl")
    report = replay_trace(trace_dir, output_dir=tmp_path / "replay", stage_timeout_s=60.0)
    assert report.differences == [], report.text()


def test_replay_reports_a_run_that_ends_in_another_state(tmp_path, monkeypatch):
    from microscope_command_server.acquisition import workflow

    def no_tiles(tile_config_path, core):
        return []

    monkeypatch.setattr(workflow.TileConfigUtils, "read_tile_config", staticmethod(no_tiles))
    report = replay_trace(CHECKED_IN_TRACE, output_dir=tmp_path / "replay", stage_timeout_s=20.0)
    states = [d for d in report.differences if d.kind == KIND_REPLY_VALUE and d.field == "state"]
    assert states and states[0].command == "STATUS" and "FAILED" in str(states[0].actual)
