"""The autofocus event log the operator's display reads (server/af_events.py, AFTRACE).

Covers the log itself, the AFTRACE reply, streaming autofocus outcomes, and the
stop-and-snap searches feeding it through microscope_control's autofocus listener.
"""

import json
import math
import struct
from types import SimpleNamespace

import numpy as np

from microscope_command_server.server.af_events import AutofocusEventLog
from microscope_command_server.server.handlers import COMMAND_HANDLERS
from microscope_command_server.server.handlers import streaming_focus
from microscope_command_server.server.handlers.af_trace import handle_aftrace


def _types(events):
    return [e["type"] for e in events]


def test_a_search_is_start_samples_end_with_one_run_id():
    log = AutofocusEventLog()
    with log.context("Tile 3 of 10", tile=2, x=100.0, y=-50.0):
        log.af_started("sweep", {"z0": -300.0, "n_steps": 3, "search_range": 20.0})
        for z, score in [(-310.0, 1.0), (-300.0, 3.0), (-290.0, 2.0)]:
            log.af_sample(z, score, 40.0)
        log.af_finished("ok", -301.2, {})
    events = log.since(0)["events"]
    assert _types(events) == ["start", "sample", "sample", "sample", "end"]
    assert len({e["run"] for e in events}) == 1
    start, end = events[0], events[-1]
    assert start["context"] == "Tile 3 of 10" and start["tile"] == 2 and start["x"] == 100.0
    assert start["method"] == "sweep" and start["n_steps"] == 3
    assert end["status"] == "ok" and end["z"] == -301.2 and end["method"] == "sweep"
    assert end["context"] == "Tile 3 of 10"  # a reader of the record need not find the start
    assert [e["seq"] for e in events] == [1, 2, 3, 4, 5]
    # The label is the caller's: it is gone once the block ends.
    log.af_started("sweep", {})
    assert "context" not in log.since(5)["events"][0]


def test_since_returns_only_newer_events_and_names_the_running_search():
    log = AutofocusEventLog()
    log.af_started("drift_check", {})
    log.af_sample(1.0, 2.0)
    reply = log.since(0)
    assert reply["seq"] == 2 and reply["active"]["method"] == "drift_check"
    log.af_sample(2.0, 3.0)
    newer = log.since(2)
    assert _types(newer["events"]) == ["sample"] and newer["events"][0]["z"] == 2.0
    log.af_finished("ok", 2.0, {})
    assert log.since(4)["active"] is None
    assert log.since(4)["events"] == []


def test_a_reader_that_falls_behind_is_told_and_pages_through():
    log = AutofocusEventLog(capacity=10)
    log.af_started("sweep", {})
    for i in range(20):
        log.af_sample(float(i), float(i))
    reply = log.since(0, limit=4)
    assert reply["first"] == 12  # events 1..11 were dropped
    assert reply["more"] is True and len(reply["events"]) == 4
    assert reply["events"][0]["seq"] == 12


def test_a_nested_search_keeps_its_own_samples():
    log = AutofocusEventLog()
    log.af_started("streaming", {})
    log.af_started("sweep", {})  # a fallback started inside the first
    log.af_sample(5.0, 1.0)
    log.af_finished("ok", 5.0, {})
    log.af_sample(6.0, 1.0)
    log.af_finished("failed", None, {"reason": "no peak"})
    events = log.since(0)["events"]
    outer, inner = events[0]["run"], events[1]["run"]
    assert events[2]["run"] == inner and events[3]["run"] == inner
    assert events[4]["run"] == outer and events[5]["status"] == "failed"


def test_non_finite_values_never_reach_the_json():
    log = AutofocusEventLog()
    log.af_started("sweep", {"score_metric": "brenner_gradient"})
    log.af_sample(1.0, math.nan, math.inf)
    sample = log.since(0)["events"][1]
    assert sample["score"] is None and sample["spread"] is None
    json.dumps(log.since(0), allow_nan=False)


def test_a_samples_without_a_search_is_ignored():
    log = AutofocusEventLog()
    log.af_sample(1.0, 2.0)
    log.af_finished("ok", 1.0, {})
    assert log.since(0)["events"] == []


def test_the_acquisition_record_gets_every_event(tmp_path):
    log = AutofocusEventLog()
    path = tmp_path / "af_trace.ndjson"
    log.attach_file(path)
    log.af_started("sweep", {})
    log.af_sample(1.0, 2.0)
    log.af_finished("ok", 1.0, {})
    log.detach_file()
    log.af_started("sweep", {})  # after the run: not recorded
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert _types(lines) == ["start", "sample", "end"]


class _Conn:
    def __init__(self, payload: bytes):
        self._in = payload
        self.sent = b""

    def recv(self, n):
        chunk, self._in = self._in[:n], self._in[n:]
        return chunk

    def sendall(self, data):
        self.sent += data


def test_aftrace_replies_with_length_prefixed_json(monkeypatch):
    log = AutofocusEventLog()
    log.af_started("sweep", {})
    log.af_sample(1.0, 2.0)
    monkeypatch.setattr("microscope_command_server.server.handlers.af_trace.AF_EVENTS", log)
    conn = _Conn(struct.pack(">Q", 1))
    handle_aftrace(conn, None, None, None)
    (length,) = struct.unpack(">I", conn.sent[:4])
    reply = json.loads(conn.sent[4 : 4 + length].decode("utf-8"))
    assert _types(reply["events"]) == ["sample"] and reply["active"]["method"] == "sweep"
    assert COMMAND_HANDLERS[b"aftrace_"] is handle_aftrace


def _run_streaming(monkeypatch, reply: str, busy=False):
    log = AutofocusEventLog()
    monkeypatch.setattr(streaming_focus, "AF_EVENTS", log)

    def fake_impl(conn, client, hardware, settings, **kwargs):
        conn.sendall(reply.encode("utf-8"))

    monkeypatch.setattr(streaming_focus, "_handle_streaming_focus", fake_impl)
    monkeypatch.setattr(streaming_focus, "acquisition_in_progress", lambda states: busy)
    conn = _Conn(b"")
    streaming_focus.handle_streaming_focus(conn, None, None, None)
    assert conn.sent.decode("utf-8") == reply  # the client still gets the reply
    return log.since(0)["events"]


def test_streaming_outcomes_map_to_statuses(monkeypatch):
    ok = _run_streaming(monkeypatch, "SUCCESS:-300.0:-296.5:3.5:180:40.0")
    assert ok[0]["method"] == "streaming" and ok[-1]["status"] == "ok" and ok[-1]["z"] == -296.5
    failed = _run_streaming(monkeypatch, "FAILED:peak at the edge")
    assert failed[-1]["status"] == "failed" and failed[-1]["reason"] == "peak at the edge"
    assert _run_streaming(monkeypatch, "ABORTED:client")[-1]["status"] == "aborted"
    assert _run_streaming(monkeypatch, "UNAVAILABLE:no calibration")[-1]["status"] == "unavailable"


def test_a_streaming_request_refused_as_busy_publishes_nothing(monkeypatch):
    assert _run_streaming(monkeypatch, "BUSY____", busy=True) == []


def test_the_hardware_searches_feed_the_log():
    from microscope_control.hardware.base import Position
    from microscope_control.hardware.pycromanager import PycromanagerHardware

    hw = PycromanagerHardware.__new__(PycromanagerHardware)
    hw.settings = {"stage": {"limits": {"z_um": {"low": -1000.0, "high": 1000.0}}}}
    log = AutofocusEventLog()
    hw.autofocus_listener = log
    z = {"z": 0.0}
    hw._active_detector_id = "fake"
    hw._camera_registry = {
        "fake": SimpleNamespace(stop_if_streaming=lambda: None, extract_green_channel=lambda i: i)
    }
    hw.get_current_position = lambda: Position(0.0, 0.0, z["z"])
    hw.move_to_position = lambda pos: z.update(z=float(pos.z))
    hw.snap_image = lambda: (
        np.full((8, 8), 1000.0 * np.exp(-0.5 * (z["z"] / 4.0) ** 2) + 10.0),
        {},
    )
    with log.context("Initial focus"):
        hw.autofocus(
            n_steps=9,
            search_range=30.0,
            score_metric=lambda img: float(np.mean(img)),
            move_stage_to_estimate=False,
        )
    events = log.since(0)["events"]
    assert _types(events) == ["start"] + ["sample"] * 9 + ["end"]
    assert events[0]["context"] == "Initial focus" and events[0]["method"] == "sweep"
    assert events[-1]["status"] == "ok" and abs(events[-1]["z"]) <= 4.0
