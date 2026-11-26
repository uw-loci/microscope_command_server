"""Record a server session: client traffic and every Micro-Manager Core call.

Switched on by ``--trace-dir DIR`` or the ``MICROSCOPE_TRACE_DIR`` environment
variable (off by default). The recorder wraps the Core and Studio objects in
:class:`microscope_control.trace.RecordingProxy` and every accepted client
socket in :class:`RecordingConnection`, so the server's own code runs
unchanged. One session is one directory::

    <trace-dir>/session_<stamp>/
        manifest.json      server version, package git refs, config files, timing
        hardware.jsonl     every Core/Studio call: args, return value, duration
        socket.jsonl       every client command, its payload and the replies
        config/            a copy of the YAML bundle CONFIG pointed at
        inputs/stage_N/    the TileConfiguration.txt an ACQUIRE read

A *stage* is one client command; hardware calls are labeled with the stage in
progress so a replay can say which command a difference belongs to. Polling
commands (STATUS, PROGRESS, the REQ* prompts) and position reads do not take
over the label, so the calls an acquisition thread makes stay with ACQUIRE.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import platform
import shutil
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from microscope_control.trace import (
    TRACE_DIR_ENV,
    TraceWriter,
    frame_mode_from_env,
    new_trace_dir,
    record_core,
    record_studio,
)
from microscope_control.trace.writer import SOCKET_STREAM

from microscope_command_server.server.protocol import END_MARKER, ExtendedCommand

logger = logging.getLogger(__name__)

# Message flags whose values are folders or files on the recording machine. The
# replayer points them somewhere on its own machine.
PATH_FLAGS = ("yaml", "output", "projects", "bg-folder", "wb-folder", "flatfield-folder")

# Replies longer than this are stored as a digest and length, not raw bytes
# (live-view frames would otherwise dominate the file).
LARGE_REPLY_BYTES = 4096

# Commands the client repeats while waiting for something: their replies change
# with timing, so a replay re-asks until it sees the recorded answer instead of
# comparing the first one.
POLL_COMMANDS = frozenset(
    {
        "STATUS",
        "PROGRESS",
        "REQMANF",
        "REQHWER",
        "REQTWARN",
        "REQFINSH",
        "REQSAT",
        "FLATPOLL",
        "LCREADY",
        "HWSTAT",
        "AFTRACE",
        "GETLIVE",
    }
)

# Position and setting reads the client polls for its displays. Their replies
# are compared with a tolerance; they never take over the stage label.
READ_COMMANDS = frozenset(
    {
        "GETXY",
        "GETZ",
        "GETXYZ",
        "GETR",
        "GETZF",
        "GETFOV",
        "GETPXSZ",
        "GETEXP",
        "GETCAM",
        "GETMODE",
        "GETBIN",
        "GETGAIN",
        "GETCAP",
        "GETILLM",
        "GETPROPL",
        "GETFRAME",
    }
)

_CODE_TO_NAME: Dict[bytes, str] = {
    value: name
    for name, value in vars(ExtendedCommand).items()
    if isinstance(value, bytes) and not name.startswith("_")
}


def command_name(code: bytes) -> str:
    """The protocol name of an 8-byte command code, or ``UNKNOWN:<hex>``."""
    return _CODE_TO_NAME.get(bytes(code), "UNKNOWN:" + bytes(code).hex())


def command_class(name: str) -> str:
    """``poll``, ``read`` or ``action``; see :data:`POLL_COMMANDS` and :data:`READ_COMMANDS`."""
    if name in POLL_COMMANDS:
        return "poll"
    if name in READ_COMMANDS:
        return "read"
    return "action"


def _ascii(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _text_preview(data: bytes, limit: int = 120) -> str:
    """Printable ASCII view of a payload for reading the trace by eye."""
    shown = data[:limit].decode("ascii", "backslashreplace")
    shown = "".join(ch if 32 <= ord(ch) < 127 else "." for ch in shown)
    return shown + ("..." if len(data) > limit else "")


def _parse_message_flags(text: str) -> Dict[str, str]:
    """``--flag value`` pairs of an ACQUIRE-style message (values may be quoted)."""
    import shlex

    try:
        parts = shlex.split(text)
    except ValueError:
        parts = text.split()
    flags: Dict[str, str] = {}
    for i, part in enumerate(parts):
        if part.startswith("--") and i + 1 < len(parts) and not parts[i + 1].startswith("--"):
            flags[part[2:]] = parts[i + 1]
    return flags


@contextlib.contextmanager
def command_scope(conn: Any, code: bytes) -> Iterator[None]:
    """Marks the handling of one command on ``conn``.

    The dispatcher wraps each handler call in this. A recording or replay
    connection learns from it where one command ends and the next begins; a
    plain socket has no ``begin_command`` and nothing happens.
    """
    begin = getattr(conn, "begin_command", None)
    if begin is None:
        yield
        return
    begin(code)
    try:
        yield
    finally:
        conn.end_command()


class RecordingConnection:
    """A socket proxy that records what the client sent and what the server replied.

    Everything the handlers use (``recv``, ``sendall``, ``send``, ``settimeout``,
    ``close``, ...) passes straight through to the real socket.
    """

    def __init__(self, sock: Any, recorder: "SessionRecorder", conn_id: int, addr: Any):
        self._sock = sock
        self._recorder = recorder
        self.conn_id = conn_id
        self.addr = addr
        self.stage = 0
        self.command = ""
        self._in_command = False
        self._payload: List[bytes] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sock, name)

    # -- the server's hook -----------------------------------------------------

    def begin_command(self, code: bytes) -> None:
        """Called by the dispatcher once it has read a command code."""
        self.stage, self.command = self._recorder.begin_command(self, code)
        self._in_command = True
        self._payload = []

    def end_command(self) -> None:
        """Called when the handler returns; lets the recorder copy the inputs it named."""
        self._in_command = False
        payload = b"".join(self._payload)
        self._payload = []
        self._recorder.end_command(self, payload)

    # -- socket surface ----------------------------------------------------------

    def recv(self, n: int) -> bytes:
        data = self._sock.recv(n)
        if self._in_command and data:
            self._payload.append(data)
            self._recorder.record_bytes(self, "c2s", data)
        return data

    def sendall(self, data: bytes) -> None:
        self._sock.sendall(data)
        self._recorder.record_bytes(self, "s2c", bytes(data))

    def send(self, data: bytes) -> int:
        sent = self._sock.send(data)
        self._recorder.record_bytes(self, "s2c", bytes(data[:sent]))
        return sent

    def close(self) -> None:
        self._recorder.record_event(self, "close")
        self._sock.close()


class SessionRecorder:
    """Owns the trace directory for one server process."""

    def __init__(
        self,
        trace_dir: Path,
        frame_mode: Optional[str] = None,
        demo: bool = False,
        label: str = "session",
    ):
        self.frame_mode = frame_mode or frame_mode_from_env()
        self.directory = new_trace_dir(Path(trace_dir), label)
        self.writer = TraceWriter(self.directory, self._manifest(demo))
        self._lock = threading.Lock()
        self._stage = 0
        self._conn_count = 0
        self._copied_config_dirs: Dict[str, str] = {}
        logger.info("Trace recording to %s (frames: %s)", self.directory, self.frame_mode)

    # -- construction ------------------------------------------------------------

    @classmethod
    def from_flag_or_env(
        cls, trace_dir: Optional[str], demo: bool = False
    ) -> Optional["SessionRecorder"]:
        """A recorder when ``--trace-dir`` or MICROSCOPE_TRACE_DIR names a directory, else None."""
        chosen = trace_dir or os.environ.get(TRACE_DIR_ENV, "").strip()
        if not chosen:
            return None
        return cls(Path(chosen), demo=demo)

    def _manifest(self, demo: bool) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "recorder": "microscope_command_server.trace",
            "demo": bool(demo),
            "frame_mode": self.frame_mode,
            "python": platform.python_version(),
            "platform": platform.system(),
            "argv": [_ascii(a) for a in sys.argv],
            "config_files": [],
            "inputs": [],
            "paths": [],
            "connections": [],
        }
        try:
            from microscope_command_server.version_info import collect_git_info, collect_versions

            info["versions"] = {k: _ascii(str(v)) for k, v in collect_versions().items()}
            info["git"] = {k: _ascii(str(v)) for k, v in collect_git_info().items()}
        except Exception as exc:  # noqa: BLE001 - provenance is best effort
            info["versions_error"] = _ascii(str(exc))
        return info

    # -- wrapping ----------------------------------------------------------------

    def wrap_core(self, core: Any) -> Any:
        return record_core(core, self.writer, self.frame_mode)

    def wrap_studio(self, studio: Any) -> Any:
        return record_studio(studio, self.writer, self.frame_mode)

    def wrap_connection(self, sock: Any, addr: Any) -> RecordingConnection:
        with self._lock:
            self._conn_count += 1
            conn_id = self._conn_count
        conn = RecordingConnection(sock, self, conn_id, addr)
        self.writer.manifest_append("connections", {"conn": conn_id, "addr": list(addr)})
        self.writer.append(
            SOCKET_STREAM,
            {"kind": "connect", "conn": conn_id, "addr": list(addr), "command": ""},
        )
        return conn

    # -- what the connections report ----------------------------------------------

    def begin_command(self, conn: RecordingConnection, code: bytes) -> Tuple[int, str]:
        name = command_name(code)
        klass = command_class(name)
        with self._lock:
            self._stage += 1
            stage = self._stage
        if klass == "action":
            self.writer.set_stage(stage, name)
        self.writer.append(
            SOCKET_STREAM,
            {
                "kind": "command",
                "conn": conn.conn_id,
                "stage": stage,
                "command": name,
                "class": klass,
                "code": bytes(code).hex(),
                "thread": threading.current_thread().name,
            },
        )
        return stage, name

    def end_command(self, conn: RecordingConnection, payload: bytes) -> None:
        self.writer.append(
            SOCKET_STREAM,
            {"kind": "end", "conn": conn.conn_id, "stage": conn.stage, "command": conn.command},
        )
        try:
            if conn.command == "CONFIG" and len(payload) >= 4:
                (n,) = struct.unpack("!I", payload[:4])
                self.note_config_path(payload[4 : 4 + n].decode("utf-8", "replace"))
            elif END_MARKER.encode("ascii") in payload:
                text = payload.decode("utf-8", "replace").replace(END_MARKER, "")
                self.note_message_paths(conn.stage, conn.command, text)
                if conn.command == "ACQUIRE":
                    self.note_acquire_message(conn.stage, conn.command, text)
        except Exception as exc:  # noqa: BLE001 - never let bookkeeping break a session
            logger.warning("Trace recorder could not copy inputs for %s: %s", conn.command, exc)

    def record_bytes(self, conn: RecordingConnection, direction: str, data: bytes) -> None:
        record: Dict[str, Any] = {
            "kind": direction,
            "conn": conn.conn_id,
            "stage": conn.stage,
            "command": conn.command,
            "n": len(data),
            "thread": threading.current_thread().name,
        }
        if len(data) > LARGE_REPLY_BYTES:
            record["digest"] = hashlib.blake2b(data, digest_size=16).hexdigest()
            record["text"] = _text_preview(data, 40)
        else:
            record["bytes"] = data.hex()
            record["text"] = _text_preview(data)
        self.writer.append(SOCKET_STREAM, record)

    def record_event(self, conn: RecordingConnection, event: str) -> None:
        self.writer.append(
            SOCKET_STREAM,
            {"kind": event, "conn": conn.conn_id, "stage": conn.stage, "command": conn.command},
        )

    # -- inputs the replay will need ----------------------------------------------

    def note_config_path(self, config_path: str) -> None:
        """Copy the YAML bundle CONFIG pointed at into ``config/`` and note it."""
        source = Path(config_path)
        entry: Dict[str, Any] = {"path": _ascii(config_path), "stage": self.writer.stage}
        if source.is_file():
            target_dir = self.directory / "config"
            target_dir.mkdir(exist_ok=True)
            copied: List[str] = []
            for item in sorted(source.parent.iterdir()):
                if item.is_file() and item.suffix.lower() in (".yml", ".yaml"):
                    shutil.copy2(item, target_dir / item.name)
                    copied.append(item.name)
                elif item.is_dir() and item.name == "resources":
                    shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)
                    copied.append(item.name + "/")
            entry["copied"] = copied
            entry["dir"] = _ascii(str(source.parent))
            self._copied_config_dirs[str(source.parent)] = "config"
        else:
            entry["missing"] = True
        self.writer.manifest_append("config_files", entry)

    def note_message_paths(self, stage: int, command: str, text: str) -> None:
        """Note every path-valued flag of a text message so a replay can relocate it."""
        flags = _parse_message_flags(text)
        for key in PATH_FLAGS:
            if key in flags:
                self.writer.manifest_append(
                    "paths",
                    {"stage": stage, "command": command, "flag": key, "value": _ascii(flags[key])},
                )

    def note_acquire_message(self, stage: int, command: str, text: str) -> None:
        """Copy the TileConfiguration.txt an acquisition reads into ``inputs/stage_N/``."""
        flags = _parse_message_flags(text)
        entry: Dict[str, Any] = {"stage": stage, "command": command, "flags": {}}
        for key in ("yaml", "projects", "sample", "scan-type", "region", "output"):
            if key in flags:
                entry["flags"][key] = _ascii(flags[key])
        if all(k in flags for k in ("projects", "sample", "scan-type", "region")):
            region_dir = (
                Path(flags["projects"]) / flags["sample"] / flags["scan-type"] / flags["region"]
            )
            tile_config = region_dir / "TileConfiguration.txt"
            if tile_config.is_file():
                target = self.directory / "inputs" / f"stage_{stage}"
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tile_config, target / tile_config.name)
                entry["copied"] = [f"inputs/stage_{stage}/{tile_config.name}"]
                entry["region_dir"] = _ascii(str(region_dir))
            else:
                entry["missing"] = _ascii(str(tile_config))
        self.writer.manifest_append("inputs", entry)

    # -- the end -------------------------------------------------------------------

    def close(self) -> None:
        self.writer.close()
        logger.info("Trace closed: %s", self.directory)
