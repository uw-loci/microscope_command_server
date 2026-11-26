"""Replay a recorded session against the server code now checked out.

The server's own ``handle_client`` runs in a thread per recorded connection,
on a :class:`PycromanagerHardware` built over a
:class:`microscope_control.trace.ReplayCore` that answers every Micro-Manager
call from the trace. The recorded client commands are fed in order through an
in-memory socket; each reply is compared with the recorded one and every
hardware call with the recorded sequence. Differences are written as JSONL
records for ``python -m microscope_control.trace.diff``.

Command line::

    python -m microscope_command_server.trace.replayer TRACE_DIR \
        [--out differences.jsonl] [--output-dir DIR] [--rewrite OLD=NEW ...] \
        [--abs-tol 1e-6] [--tolerance set_xy_position=0.5] [--stage-timeout 120]

Exit status 0 means no differences.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import struct
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from microscope_control.trace import (
    Difference,
    ReplaySession,
    Trace,
    group_by_fingerprint,
    write_differences,
)
from microscope_control.trace.diff import (
    KIND_REPLAY_ERROR,
    KIND_REPLY_BYTES,
    KIND_REPLY_LENGTH,
    KIND_REPLY_TIMEOUT,
    KIND_REPLY_VALUE,
    format_groups,
)

from microscope_command_server.server.protocol import END_MARKER
from microscope_command_server.trace.recorder import _text_preview, command_class

logger = logging.getLogger(__name__)

# Position reads answered from the server's background-polled stage cache. Which
# poll the cache last ran is a matter of timing the replay cannot reproduce, so
# these replies are waited for (to stay in step) but not compared; a stage bug
# still shows in the set_xy_position / set_position arguments.
CACHED_READ_COMMANDS = frozenset({"GETXY", "GETZ", "GETXYZ"})

# Replies that carry their own length: an optional fixed status word, then a
# big-endian unsigned length, then that many bytes. They are read by length
# rather than by the recorded size, since the body can legitimately differ in
# size (CONFIG answers with the installed package versions).
LENGTH_PREFIXED_REPLIES: Dict[str, Tuple[int, str]] = {
    "CONFIG": (8, "!I"),
    "MMPRESET": (0, "!I"),
    "HWSTAT": (0, "!I"),
    "AFTRACE": (0, "!I"),
}

# CONFIG's JSON body lists package versions and git refs: environment, not
# behavior. Its keys are compared, its values are not.
ENVIRONMENT_JSON_REPLIES = frozenset({"CONFIG"})

# Fixed-format replies that are compared field by field, within a tolerance.
REPLY_FORMATS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "GETXY": ("!ff", ("x", "y")),
    "GETXYZ": ("!fff", ("x", "y", "z")),
    "GETZ": ("!f", ("z",)),
    "GETZF": ("!f", ("z",)),
    "GETR": ("!f", ("angle",)),
    "GETFOV": ("!ff", ("fov_x", "fov_y")),
    "GETPXSZ": ("!f", ("pixel_size_um",)),
    "PROGRESS": ("!II", ("current", "total")),
}


class ReplayConnection:
    """An in-memory stand-in for the client socket ``handle_client`` reads and writes."""

    def __init__(self, conn_id: int, addr: Tuple[str, int], report: "ReplayReport"):
        self.conn_id = conn_id
        self.addr = addr
        self._report = report
        self._cond = threading.Condition()
        self._inbound = bytearray()
        self._outbound = bytearray()
        self._closed = False
        self._timeout: Optional[float] = None
        self._pending: Deque[Tuple[int, str]] = deque()
        self.stage = 0
        self.command = ""

    # -- the server's hook -------------------------------------------------------

    def begin_command(self, code: bytes) -> None:
        with self._cond:
            if self._pending:
                self.stage, self.command = self._pending.popleft()
        # Polls and reads do not take over the label, as when recording: the
        # calls an acquisition thread makes stay with ACQUIRE.
        if command_class(self.command) == "action":
            self._report.current = (self.stage, self.command)

    def end_command(self) -> None:
        pass

    # -- socket surface ------------------------------------------------------------

    def recv(self, n: int) -> bytes:
        with self._cond:
            deadline = None if self._timeout is None else time.monotonic() + self._timeout
            while not self._inbound and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise socket.timeout("timed out")
                self._cond.wait(remaining)
            data = bytes(self._inbound[:n])
            del self._inbound[:n]
            return data

    def sendall(self, data: bytes) -> None:
        with self._cond:
            self._outbound += bytes(data)
            self._cond.notify_all()

    def send(self, data: bytes) -> int:
        self.sendall(data)
        return len(data)

    def settimeout(self, timeout: Optional[float]) -> None:
        self._timeout = timeout

    def gettimeout(self) -> Optional[float]:
        return self._timeout

    def setblocking(self, flag: bool) -> None:
        self._timeout = None if flag else 0.0

    def getpeername(self) -> Tuple[str, int]:
        return self.addr

    def fileno(self) -> int:
        return -1

    def shutdown(self, how: int) -> None:
        pass

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # -- the driver's side ----------------------------------------------------------

    def push(self, stage: int, command: str, data: bytes) -> None:
        with self._cond:
            self._pending.append((stage, command))
            self._inbound += data
            self._cond.notify_all()

    def close_inbound(self) -> None:
        self.close()

    def take(self, n: int, timeout: float) -> bytes:
        """Up to ``n`` reply bytes, waiting at most ``timeout`` s for them to arrive."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while len(self._outbound) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            data = bytes(self._outbound[:n])
            del self._outbound[:n]
            return data

    def take_message(self, timeout: float, settle_s: float) -> bytes:
        """Everything the server sends within ``settle_s`` of its first reply byte."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._outbound:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return b""
                self._cond.wait(remaining)
        return self.drain(settle_s)

    def drain(self, settle_s: float) -> bytes:
        """Whatever else arrives within ``settle_s``; empty when the server stayed quiet."""
        time.sleep(settle_s)
        with self._cond:
            data = bytes(self._outbound)
            self._outbound.clear()
            return data


@dataclass
class ReplayReport:
    trace_dir: Path
    output_dir: Path
    differences: List[Difference] = field(default_factory=list)
    stages: int = 0
    hardware_summary: str = ""
    differences_path: Optional[Path] = None
    current: Tuple[int, str] = (0, "")

    @property
    def ok(self) -> bool:
        return not self.differences

    def grouped(self) -> Dict[str, List[Difference]]:
        return group_by_fingerprint(self.differences)

    def text(self) -> str:
        lines = [
            f"trace:   {self.trace_dir}",
            f"output:  {self.output_dir}",
            f"stages replayed: {self.stages}",
            f"hardware: {self.hardware_summary}",
        ]
        if self.differences_path is not None:
            lines.append(f"differences: {self.differences_path}")
        if self.differences:
            first = min(
                self.differences,
                key=lambda d: (d.stage, d.seq if d.seq is not None else 10**12),
            )
            lines.append("first difference: " + first.one_line())
        lines.append(format_groups(self.grouped()))
        return "\n".join(lines)


def _records_for_stage(trace: Trace, stage: int, kind: str, conn: int) -> List[Dict[str, Any]]:
    return [
        r
        for r in trace.socket
        if r.get("kind") == kind and r.get("stage") == stage and r.get("conn") == conn
    ]


def _rewrite_text(text: str, rewrites: Sequence[Tuple[str, str]]) -> str:
    """Replace recorded path prefixes; separators after a replaced prefix follow this host."""
    for old, new in rewrites:
        if not old:
            continue
        pattern = re.compile(re.escape(old) + r"([\\/][^\s\"']*)?")

        def repl(match: "re.Match[str]") -> str:
            tail = match.group(1) or ""
            return new + tail.replace("\\", "/").replace("/", os.sep)

        text = pattern.sub(repl, text)
    return text


def rewrite_payload(command: str, payload: bytes, rewrites: Sequence[Tuple[str, str]]) -> bytes:
    """Apply path rewrites to a text payload; binary payloads are returned unchanged."""
    if not rewrites or not payload:
        return payload
    if command == "CONFIG" and len(payload) >= 4:
        (n,) = struct.unpack("!I", payload[:4])
        path = payload[4 : 4 + n].decode("utf-8", "replace")
        new_path = _rewrite_text(path, rewrites).encode("utf-8")
        return struct.pack("!I", len(new_path)) + new_path + payload[4 + n :]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload
    if END_MARKER not in text and not text.lstrip().startswith("--"):
        return payload
    return _rewrite_text(text, rewrites).encode("utf-8")


# Time stamps the server puts in file names and messages; masked before text
# replies are compared, since a replay runs at another time by definition.
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T _]\d{2}:\d{2}:\d{2}(?:\.\d+)?|\d{8}[_T-]\d{6}(?:\.\d+)?"
)


def _text_or_none(data: bytes) -> Optional[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if text.isprintable() else None


def _masked(text: str) -> str:
    return _TIMESTAMP_RE.sub("<TIME>", text)


def _first_difference_window(expected: str, actual: str, width: int = 60) -> Tuple[str, str]:
    """The two texts around the first character that differs."""
    i = 0
    while i < min(len(expected), len(actual)) and expected[i] == actual[i]:
        i += 1
    start = max(0, i - width // 3)
    prefix = "..." if start else ""
    return prefix + expected[start : start + width], prefix + actual[start : start + width]


def _expected_bytes(records: List[Dict[str, Any]]) -> Tuple[int, bytes]:
    total = sum(int(r.get("n", 0)) for r in records)
    raw = b"".join(bytes.fromhex(r["bytes"]) for r in records if "bytes" in r)
    return total, raw


def _take_reply(conn: "ReplayConnection", command: str, total: int, timeout: float) -> bytes:
    """The reply bytes: by recorded size, or by the length a prefixed reply announces."""
    prefixed = LENGTH_PREFIXED_REPLIES.get(command)
    if prefixed is None:
        return conn.take(total, timeout)
    word, fmt = prefixed
    head = conn.take(word + struct.calcsize(fmt), timeout)
    if len(head) < word + struct.calcsize(fmt):
        return head
    (n,) = struct.unpack(fmt, head[word:])
    return head + conn.take(n, timeout)


def _compare_environment_json(
    stage: int, command: str, expected: bytes, got: bytes, word: int, fmt: str
) -> List[Difference]:
    """Status word exact; the JSON body's keys exact; its values are environment."""
    size = struct.calcsize(fmt)
    if expected[:word] != got[:word]:
        return [
            Difference(
                stage,
                command,
                KIND_REPLY_BYTES,
                "reply",
                "status",
                _text_preview(expected[:word]),
                _text_preview(got[:word]),
            )
        ]
    try:
        exp_json = json.loads(expected[word + size :].decode("utf-8"))
        got_json = json.loads(got[word + size :].decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        if expected == got:
            return []
        return [
            Difference(
                stage,
                command,
                KIND_REPLY_BYTES,
                "reply",
                "body",
                _text_preview(expected),
                _text_preview(got),
                "not JSON on both sides",
            )
        ]
    if isinstance(exp_json, dict) and isinstance(got_json, dict) and set(exp_json) != set(got_json):
        return [
            Difference(
                stage,
                command,
                KIND_REPLY_VALUE,
                "reply",
                "keys",
                sorted(set(exp_json) - set(got_json)),
                sorted(set(got_json) - set(exp_json)),
                "recorded-only keys vs replay-only keys; values are environment and not compared",
            )
        ]
    return []


def _compare_reply(
    stage: int,
    command: str,
    records: List[Dict[str, Any]],
    got: bytes,
    abs_tol: float,
) -> List[Difference]:
    """Differences between the recorded reply records and the bytes the replay produced."""
    total, expected_all = _expected_bytes(records)
    if command in ENVIRONMENT_JSON_REPLIES and command in LENGTH_PREFIXED_REPLIES:
        word, fmt = LENGTH_PREFIXED_REPLIES[command]
        return _compare_environment_json(stage, command, expected_all, got, word, fmt)
    if len(got) != total:
        return [
            Difference(
                stage,
                command,
                KIND_REPLY_LENGTH,
                "reply",
                "n",
                total,
                len(got),
                f"recorded {_text_preview(b''.join(bytes.fromhex(r['bytes']) for r in records if 'bytes' in r))!r} "
                f"got {_text_preview(got)!r}",
            )
        ]
    diffs: List[Difference] = []
    offset = 0
    for rec in records:
        n = int(rec.get("n", 0))
        chunk = got[offset : offset + n]
        offset += n
        if "bytes" in rec:
            expected = bytes.fromhex(rec["bytes"])
            if chunk == expected:
                continue
            exp_text, got_text = _text_or_none(expected), _text_or_none(chunk)
            if exp_text is not None and got_text is not None:
                if _masked(exp_text) == _masked(got_text):
                    continue
                exp_show, got_show = _first_difference_window(exp_text, got_text)
                diffs.append(
                    Difference(
                        stage,
                        command,
                        KIND_REPLY_BYTES,
                        "reply",
                        "text",
                        exp_show,
                        got_show,
                        "time stamps masked; shown around the first differing character",
                    )
                )
                continue
            fmt = REPLY_FORMATS.get(command)
            if fmt and struct.calcsize(fmt[0]) == n:
                exp_vals = struct.unpack(fmt[0], expected)
                got_vals = struct.unpack(fmt[0], chunk)
                for name, e, g in zip(fmt[1], exp_vals, got_vals):
                    if abs(float(e) - float(g)) > abs_tol:
                        diffs.append(
                            Difference(
                                stage,
                                command,
                                KIND_REPLY_VALUE,
                                "reply",
                                name,
                                e,
                                g,
                                f"tolerance {abs_tol}",
                            )
                        )
                continue
            diffs.append(
                Difference(
                    stage,
                    command,
                    KIND_REPLY_BYTES,
                    "reply",
                    "bytes",
                    _text_preview(expected),
                    _text_preview(chunk),
                )
            )
        elif "digest" in rec:
            digest = hashlib.blake2b(chunk, digest_size=16).hexdigest()
            if digest != rec["digest"]:
                diffs.append(
                    Difference(
                        stage,
                        command,
                        KIND_REPLY_BYTES,
                        "reply",
                        "digest",
                        rec["digest"],
                        digest,
                        f"{n} bytes; recorded as a digest only",
                    )
                )
    return diffs


def _relocated(output_dir: Path, recorded: str) -> Path:
    """A folder under ``output_dir`` for a recorded path: its last name plus a short hash."""
    name = re.split(r"[\\/]+", recorded.rstrip("\\/"))[-1] or "root"
    digest = hashlib.blake2b(recorded.encode("utf-8"), digest_size=3).hexdigest()
    return output_dir / "paths" / f"{name}_{digest}"


# STATUS replies that end an acquisition. A replay that reaches one of these
# while the recording shows another is not going to catch up; say so at once.
_TERMINAL_STATES = (b"COMPLETED", b"FAILED", b"CANCELLED")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _masked_numbers(data: bytes) -> Optional[str]:
    text = _text_or_none(data)
    return None if text is None else _NUMBER_RE.sub("<N>", _masked(text))


def _poll_reached(name: str, got: bytes, acceptable: "set[bytes]") -> bool:
    """A poll's reply counts as reached when it is a recorded answer, or has passed it.

    Text answers match with their numbers masked (``COMPLETED|final_z:4.97``
    is the same state at any Z). PROGRESS carries counters: a replay that
    finished faster than the recording reports a count the client never saw,
    so any count at or beyond the recorded one (same total) is the same point
    in the run or later.
    """
    if got in acceptable:
        return True
    got_text = _masked_numbers(got)
    if got_text is not None and any(_masked_numbers(a) == got_text for a in acceptable):
        return True
    fmt = REPLY_FORMATS.get(name)
    if fmt is None or len(got) != struct.calcsize(fmt[0]):
        return False
    got_vals = struct.unpack(fmt[0], got)
    for expected in acceptable:
        if len(expected) != len(got):
            continue
        exp_vals = struct.unpack(fmt[0], expected)
        if name == "PROGRESS":
            if got_vals[1] == exp_vals[1] and got_vals[0] >= exp_vals[0]:
                return True
        elif all(abs(float(g) - float(e)) <= 1e-6 for g, e in zip(got_vals, exp_vals)):
            return True
    return False


def _default_rewrites(trace: Trace, output_dir: Path) -> List[Tuple[str, str]]:
    """Where recorded paths go on this machine.

    The config folder CONFIG named becomes a copy of the trace's ``config/``
    under the output dir (so the session log the server writes beside the YAML
    lands there, not in the trace). Every other folder the client named
    (``--output``, ``--projects``, background and white-balance folders) gets
    a folder of its own under ``output/paths/``.
    """
    rewrites: List[Tuple[str, str]] = []
    config_dirs = set()
    config_copy = trace.directory / "config"
    if config_copy.is_dir():
        target = output_dir / "config"
        if not target.exists():
            shutil.copytree(config_copy, target)
        for entry in trace.manifest.get("config_files", []):
            if entry.get("dir"):
                config_dirs.add(entry["dir"])
                rewrites.append((entry["dir"], str(target)))
    seen = set(config_dirs)
    for entry in trace.manifest.get("paths", []):
        value = entry.get("value", "")
        if not value or value in seen:
            continue
        if entry.get("flag") == "yaml":
            parent = re.sub(r"[\\/][^\\/]*$", "", value)
            if parent in config_dirs or not parent:
                continue
            value = parent
            if value in seen:
                continue
        seen.add(value)
        rewrites.append((value, str(_relocated(output_dir, value))))
    for entry in trace.manifest.get("inputs", []):
        projects = entry.get("flags", {}).get("projects")
        if projects and projects not in seen:
            seen.add(projects)
            rewrites.append((projects, str(_relocated(output_dir, projects))))
    # Longest prefixes first so a parent folder does not shadow a child.
    rewrites.sort(key=lambda kv: -len(kv[0]))
    return rewrites


def _rewrite_replies(
    records: List[Dict[str, Any]], rewrites: Sequence[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """Recorded replies with the recorded paths relocated, so text replies compare fairly."""
    if not rewrites:
        return records
    out: List[Dict[str, Any]] = []
    for rec in records:
        if "bytes" not in rec:
            out.append(rec)
            continue
        raw = bytes.fromhex(rec["bytes"])
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            out.append(rec)
            continue
        if not text.isprintable():
            out.append(rec)
            continue
        new = _rewrite_text(text, rewrites).encode("utf-8")
        if new == raw:
            out.append(rec)
        else:
            out.append(dict(rec, bytes=new.hex(), n=len(new)))
    return out


def _stage_inputs(
    trace: Trace, stage: int, output_dir: Path, rewrites: Sequence[Tuple[str, str]]
) -> None:
    """Put the TileConfiguration.txt an ACQUIRE read where the rewritten message points."""
    for entry in trace.manifest.get("inputs", []):
        if entry.get("stage") != stage or not entry.get("copied"):
            continue
        region = _rewrite_text(entry.get("region_dir", ""), rewrites)
        if not region:
            continue
        Path(region).mkdir(parents=True, exist_ok=True)
        for rel in entry["copied"]:
            src = trace.directory / rel
            if src.is_file():
                shutil.copy2(src, Path(region) / src.name)


def replay_trace(
    trace_dir: Path,
    output_dir: Optional[Path] = None,
    rewrites: Optional[Sequence[Tuple[str, str]]] = None,
    abs_tol: float = 1e-6,
    tolerances: Optional[Dict[str, float]] = None,
    stage_timeout_s: float = 120.0,
    poll_interval_s: float = 0.05,
    settle_s: float = 0.1,
    mode: str = "record",
    differences_path: Optional[Path] = None,
) -> ReplayReport:
    """Run the recorded session against the current server code; see the module docstring."""
    from microscope_command_server.server import qp_server

    trace = Trace(Path(trace_dir))
    output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="trace_replay_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = ReplayReport(trace_dir=trace.directory, output_dir=output_dir)
    all_rewrites = _default_rewrites(trace, output_dir) + list(rewrites or [])

    session = ReplaySession(
        trace,
        abs_tol=abs_tol,
        tolerances=tolerances,
        mode=mode,
        stage_of=lambda: report.current,
    )
    qp_server.reset_connection_state()
    qp_server.load_startup_settings()
    qp_server.build_hardware(session.core(), session.studio())

    connections: Dict[int, ReplayConnection] = {}
    threads: Dict[int, threading.Thread] = {}
    for entry in trace.manifest.get("connections", []):
        conn_id = int(entry["conn"])
        addr = tuple(entry.get("addr", ["127.0.0.1", 50000 + conn_id]))
        addr = (str(addr[0]), int(addr[1]))
        conn = ReplayConnection(conn_id, addr, report)
        connections[conn_id] = conn
        thread = threading.Thread(
            target=qp_server.handle_client,
            args=(conn, addr),
            daemon=True,
            name=f"replay-conn-{conn_id}",
        )
        thread.start()
        threads[conn_id] = thread

    commands = sorted(trace.commands(), key=lambda r: int(r["stage"]))
    gave_up: "set[Tuple[int, str]]" = set()
    try:
        for rec in commands:
            stage = int(rec["stage"])
            name = str(rec["command"])
            conn = connections.get(int(rec.get("conn", 1)))
            if conn is None:
                report.differences.append(
                    Difference(
                        stage, name, KIND_REPLAY_ERROR, "connection", "conn", rec.get("conn"), None
                    )
                )
                continue
            klass = rec.get("class") or command_class(name)
            code = bytes.fromhex(rec["code"])
            _, payload = _expected_bytes(_records_for_stage(trace, stage, "c2s", conn.conn_id))
            payload = rewrite_payload(name, payload, all_rewrites)
            _stage_inputs(trace, stage, output_dir, all_rewrites)
            replies = _rewrite_replies(
                _records_for_stage(trace, stage, "s2c", conn.conn_id), all_rewrites
            )
            total, expected_raw = _expected_bytes(replies)
            report.stages += 1

            if klass == "poll":
                # A poll's recorded answer may be transient (RUNNING before
                # COMPLETED). The replay is ahead of the recording when the
                # answer is one a later poll of the same command recorded, so
                # those count as reached too.
                acceptable = {expected_raw}
                for later in commands:
                    if (
                        int(later["stage"]) > stage
                        and later["command"] == name
                        and later.get("conn") == rec.get("conn")
                    ):
                        later_replies = _rewrite_replies(
                            _records_for_stage(trace, int(later["stage"]), "s2c", conn.conn_id),
                            all_rewrites,
                        )
                        acceptable.add(_expected_bytes(later_replies)[1])
                if (conn.conn_id, name) in gave_up:
                    report.differences.append(
                        Difference(
                            stage,
                            name,
                            KIND_REPLY_TIMEOUT,
                            "reply",
                            "bytes",
                            _text_preview(expected_raw),
                            None,
                            "not sent: an earlier poll of this command never got its recorded answer",
                        )
                    )
                    continue
                deadline = time.monotonic() + stage_timeout_s
                got = b""
                while True:
                    conn.push(stage, name, code + payload)
                    got = conn.take_message(min(5.0, stage_timeout_s), settle_s)
                    if _poll_reached(name, got, acceptable):
                        break
                    if name == "STATUS" and got.startswith(_TERMINAL_STATES):
                        report.differences.append(
                            Difference(
                                stage,
                                name,
                                KIND_REPLY_VALUE,
                                "reply",
                                "state",
                                _text_preview(expected_raw),
                                _text_preview(got),
                                "the run ended in another state than recorded",
                            )
                        )
                        gave_up.add((conn.conn_id, name))
                        break
                    if time.monotonic() >= deadline:
                        gave_up.add((conn.conn_id, name))
                        report.differences.append(
                            Difference(
                                stage,
                                name,
                                KIND_REPLY_TIMEOUT,
                                "reply",
                                "bytes",
                                _text_preview(expected_raw),
                                _text_preview(got),
                                f"the recorded answer did not come within {stage_timeout_s} s",
                            )
                        )
                        break
                    time.sleep(poll_interval_s)
                continue

            if klass == "action":
                gave_up.difference_update({k for k in gave_up if k[0] == conn.conn_id})
            conn.push(stage, name, code + payload)
            got = _take_reply(conn, name, total, stage_timeout_s)
            if len(got) < total and name not in LENGTH_PREFIXED_REPLIES:
                report.differences.append(
                    Difference(
                        stage,
                        name,
                        KIND_REPLY_TIMEOUT,
                        "reply",
                        "n",
                        total,
                        len(got),
                        f"only {len(got)} of {total} reply bytes within {stage_timeout_s} s: {_text_preview(got)!r}",
                    )
                )
                continue
            if name not in CACHED_READ_COMMANDS:
                report.differences.extend(_compare_reply(stage, name, replies, got, abs_tol))
            extra = conn.drain(settle_s)
            if extra:
                report.differences.append(
                    Difference(
                        stage,
                        name,
                        KIND_REPLY_LENGTH,
                        "reply",
                        "extra",
                        0,
                        len(extra),
                        f"unexpected bytes after the reply: {_text_preview(extra)!r}",
                    )
                )
    finally:
        for conn in connections.values():
            conn.close_inbound()
        for thread in threads.values():
            thread.join(timeout=30.0)
        hardware = getattr(qp_server, "hardware", None)
        cache = getattr(hardware, "stage_cache", None) if hardware is not None else None
        if cache is not None:
            with contextlib.suppress(Exception):
                cache.stop()

    report.differences.extend(d for d in session.finish() if d not in report.differences)
    # Within a stage, hardware differences come in trace order and the reply's
    # differences (no seq) after them.
    report.differences.sort(key=lambda d: (d.stage, d.seq if d.seq is not None else 10**12))
    report.hardware_summary = session.summary()
    report.differences_path = (
        Path(differences_path) if differences_path else output_dir / "differences.jsonl"
    )
    if report.differences_path.exists():
        report.differences_path.unlink()
    write_differences(report.differences_path, report.differences)
    (output_dir / "replay_summary.json").write_text(
        json.dumps(
            {
                "trace": str(trace.directory),
                "stages": report.stages,
                "hardware": report.hardware_summary,
                "differences": len(report.differences),
                "groups": {fp: len(items) for fp, items in report.grouped().items()},
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n"
    )
    return report


def _parse_tolerances(items: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        method, _, value = item.partition("=")
        if not method or not value:
            raise argparse.ArgumentTypeError(f"--tolerance wants METHOD=VALUE, not {item!r}")
        out[method] = float(value)
    return out


def _parse_rewrites(items: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items:
        old, sep, new = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--rewrite wants OLD=NEW, not {item!r}")
        out.append((old, new))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m microscope_command_server.trace replay",
        description="Replay a recorded session against the current server code.",
    )
    parser.add_argument("trace_dir", help="a session directory written with --trace-dir")
    parser.add_argument(
        "--out", help="where to write differences.jsonl (default: inside --output-dir)"
    )
    parser.add_argument(
        "--output-dir", help="where tiles, logs and the config copy go (default: a temp dir)"
    )
    parser.add_argument(
        "--rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="replace a recorded path prefix in client messages (repeatable)",
    )
    parser.add_argument(
        "--abs-tol", type=float, default=1e-6, help="numeric tolerance for arguments and replies"
    )
    parser.add_argument(
        "--tolerance",
        action="append",
        default=[],
        metavar="METHOD=VALUE",
        help="per-Core-method tolerance, e.g. set_xy_position=0.5 (repeatable)",
    )
    parser.add_argument(
        "--stage-timeout", type=float, default=120.0, help="seconds to wait for one reply"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop the code under test at the first hardware difference instead of noting it and going on",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="only the summary, no server log"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    report = replay_trace(
        Path(args.trace_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        rewrites=_parse_rewrites(args.rewrite),
        abs_tol=args.abs_tol,
        tolerances=_parse_tolerances(args.tolerance),
        stage_timeout_s=args.stage_timeout,
        mode="strict" if args.strict else "record",
        differences_path=Path(args.out) if args.out else None,
    )
    print(report.text())
    return 0 if report.ok else 1
