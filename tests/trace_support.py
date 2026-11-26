"""A scripted client session against the demo microscope, recorded as a trace.

Used by the trace tests and by ``python -m tests.trace_support DIR`` to build
the synthetic trace checked in under ``tests/data``. The server's own
``handle_client`` runs on one end of a socket pair over a ``DemoCore`` wrapped
by the session recorder, exactly as ``--demo --trace-dir`` would run it; the
other end plays QuPath: CONFIG, position reads, moves, exposure, snaps, an
autofocus test and a small tile acquisition with autofocus.
"""

from __future__ import annotations

import shutil
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from microscope_command_server.server.protocol import END_MARKER, ExtendedCommand

DATA_DIR = Path(__file__).parent / "data"
DEMO_BUNDLE = DATA_DIR / "demo_bundle"
CHECKED_IN_TRACE = DATA_DIR / "trace_demo_session"

# A 2 x 2 tile grid; the demo slide has tissue near the origin.
TILE_GRID: List[Tuple[str, float, float]] = [
    ("tile_0.tif", 0.0, 0.0),
    ("tile_1.tif", 300.0, 0.0),
    ("tile_2.tif", 0.0, 300.0),
    ("tile_3.tif", 300.0, 300.0),
]


def stage_demo_bundle(work_dir: Path) -> Path:
    """Copy the demo YAML bundle under ``work_dir/config`` and return config_Demo.yml."""
    target = work_dir / "config"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for item in DEMO_BUNDLE.iterdir():
        if item.name.startswith("resources_"):
            # The bundle QPSC extracts puts the resources file in resources/.
            (target / "resources").mkdir(exist_ok=True)
            shutil.copy2(item, target / "resources" / item.name)
        elif item.suffix == ".yml":
            shutil.copy2(item, target / item.name)
    return target / "config_Demo.yml"


def write_tile_configuration(region_dir: Path, tiles: List[Tuple[str, float, float]]) -> Path:
    region_dir.mkdir(parents=True, exist_ok=True)
    path = region_dir / "TileConfiguration.txt"
    lines = [
        "# Define the number of dimensions we are working on",
        "dim = 2",
        "",
        "# Define the image coordinates",
    ]
    lines += [f"{name}; ; ({x:.1f}, {y:.1f})" for name, x, y in tiles]
    path.write_text("\n".join(lines) + "\n")
    return path


class ScriptedClient:
    """The QuPath side of the protocol, enough for a short session."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.sock.settimeout(60.0)

    def _recv_exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("server closed the connection")
            data += chunk
        return data

    def config(self, yaml_path: Path) -> str:
        path = str(yaml_path).encode("utf-8")
        self.sock.sendall(ExtendedCommand.CONFIG + struct.pack("!I", len(path)) + path)
        head = self._recv_exact(8)
        (n,) = struct.unpack("!I", self._recv_exact(4))
        body = self._recv_exact(n)
        if head != b"CFG___OK":
            raise RuntimeError(f"CONFIG failed: {head!r} {body[:200]!r}")
        return body.decode("utf-8")

    def get_xyz(self) -> Tuple[float, float, float]:
        self.sock.sendall(ExtendedCommand.GETXYZ)
        return struct.unpack("!fff", self._recv_exact(12))

    def get_xy(self) -> Tuple[float, float]:
        self.sock.sendall(ExtendedCommand.GETXY)
        return struct.unpack("!ff", self._recv_exact(8))

    def move_xy(self, x: float, y: float) -> None:
        self.sock.sendall(ExtendedCommand.MOVE + struct.pack("!ff", x, y))

    def move_z(self, z: float) -> None:
        self.sock.sendall(ExtendedCommand.MOVEZ + struct.pack("!f", z))

    def set_exposure(self, ms: float) -> bytes:
        self.sock.sendall(ExtendedCommand.SETEXP + bytes([1]) + struct.pack("!f", ms))
        return self._recv_exact(8)

    def _text_reply(self) -> str:
        return self.sock.recv(65536).decode("utf-8", "replace")

    def snap(self, output_dir: Path, exposure_ms: float, angle: float = 0.0) -> str:
        message = f"--angle {angle} --exposure {exposure_ms} --output {output_dir} {END_MARKER}"
        self.sock.sendall(ExtendedCommand.SNAP + message.encode("utf-8"))
        reply = self._text_reply()
        if not reply.startswith("SUCCESS:"):
            raise RuntimeError(f"SNAP failed: {reply}")
        return reply[8:].strip()

    def test_autofocus(self, yaml_path: Path, output_dir: Path, objective: str) -> str:
        message = f"--yaml {yaml_path} --output {output_dir} --objective {objective} {END_MARKER}"
        self.sock.sendall(ExtendedCommand.TESTAF + message.encode("utf-8"))
        started = self._text_reply()
        if not started.startswith("STARTED:"):
            raise RuntimeError(f"TESTAF refused: {started}")
        return self._text_reply()

    def acquire(self, message: str) -> str:
        self.sock.sendall(ExtendedCommand.ACQUIRE + (message + " " + END_MARKER).encode("utf-8"))
        return self._recv_exact(16).decode("ascii").strip()

    def status(self) -> str:
        self.sock.sendall(ExtendedCommand.STATUS)
        return self._recv_exact(16).decode("ascii").strip()

    def progress(self) -> Tuple[int, int]:
        self.sock.sendall(ExtendedCommand.PROGRESS)
        return struct.unpack("!II", self._recv_exact(8))

    def wait_until_done(self, timeout_s: float = 300.0, poll_s: float = 0.5) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.status()
            if state.startswith(("COMPLETED", "FAILED", "CANCELLED")):
                return state
            self.progress()
            time.sleep(poll_s)
        raise TimeoutError("acquisition did not finish")

    def disconnect(self) -> None:
        self.sock.sendall(ExtendedCommand.DISCONNECT)


def acquire_message(
    yaml_path: Path, projects: Path, sample: str, scan_type: str, region: str, af: bool
) -> str:
    parts = [
        f"--yaml {yaml_path}",
        f"--projects {projects}",
        f"--sample {sample}",
        f"--scan-type {scan_type}",
        f"--region {region}",
        "--objective DEMO_OBJECTIVE_10X",
        "--detector DEMO_CAMERA",
        "--pixel-size 0.65",
        "--bg-correction false",
        "--wb-mode off",
    ]
    if af:
        parts += ["--af-tiles 1", "--af-steps 7", "--af-range 40.0"]
    else:
        parts += ["--af-disabled true"]
    return " ".join(parts)


def default_script(
    client: ScriptedClient,
    work: Dict[str, Path],
    with_acquire: bool = True,
    with_autofocus: bool = True,
) -> None:
    """The scripted QuPath session: connect, read, move, expose, snap, autofocus, acquire.

    ``with_autofocus=False`` leaves out TESTAF and runs the acquisition with
    autofocus disabled: the session then depends on no pixel values, so it
    replays exactly from a trace that holds frame digests only.
    """
    yaml_path = work["yaml"]
    client.config(yaml_path)
    client.get_xyz()
    client.move_xy(1000.0, 2000.0)
    client.get_xy()
    client.set_exposure(12.5)
    snaps = work["snaps"]
    snaps.mkdir(parents=True, exist_ok=True)
    for i, (x, y) in enumerate([(0.0, 0.0), (250.0, 0.0), (0.0, 250.0)]):
        client.move_xy(x, y)
        client.snap(snaps / f"snap_{i}", 12.5)
    client.move_z(0.0)
    if with_autofocus:
        client.test_autofocus(yaml_path, work["af"], "DEMO_OBJECTIVE_10X")
    if with_acquire:
        projects = work["projects"]
        region_dir = projects / "Sample1" / "Brightfield_10x_1" / "bounds"
        write_tile_configuration(region_dir, TILE_GRID)
        message = acquire_message(
            yaml_path, projects, "Sample1", "Brightfield_10x_1", "bounds", af=with_autofocus
        )
        ack = client.acquire(message)
        if not ack.startswith("STARTED"):
            raise RuntimeError(f"ACQUIRE refused: {ack}")
        client.wait_until_done()
    client.disconnect()


def run_session(
    trace_parent: Path,
    work_dir: Path,
    script: Callable[[ScriptedClient, Dict[str, Path]], None] = default_script,
    label: str = "demo_session",
    frame_mode: str = "hash",
    core_speed_um_per_s: float = 1e6,
    with_autofocus: bool = True,
) -> Path:
    """Record one scripted session over the demo core; returns the trace directory."""
    from microscope_command_server.demo.virtual_microscope import DemoCore, DemoStudio
    from microscope_command_server.server import qp_server
    from microscope_command_server.trace.recorder import SessionRecorder

    work_dir = Path(work_dir)
    work = {
        "yaml": stage_demo_bundle(work_dir),
        "snaps": work_dir / "snaps",
        "af": work_dir / "af",
        "projects": work_dir / "projects",
    }
    recorder = SessionRecorder(Path(trace_parent), frame_mode=frame_mode, demo=True, label=label)
    core = DemoCore(xy_speed_um_per_s=core_speed_um_per_s, z_speed_um_per_s=core_speed_um_per_s)
    qp_server.reset_connection_state()
    qp_server.load_startup_settings()
    qp_server.trace_recorder = recorder
    try:
        qp_server.build_hardware(recorder.wrap_core(core), recorder.wrap_studio(DemoStudio(core)))
        client_end, server_end = socket.socketpair()
        addr = ("127.0.0.1", 50001)
        thread = threading.Thread(
            target=qp_server.handle_client, args=(server_end, addr), daemon=True
        )
        thread.start()
        try:
            if script is default_script:
                default_script(ScriptedClient(client_end), work, with_autofocus=with_autofocus)
            else:
                script(ScriptedClient(client_end), work)
        finally:
            client_end.close()
            thread.join(timeout=60.0)
    finally:
        qp_server.trace_recorder = None
        cache = getattr(qp_server.hardware, "stage_cache", None)
        if cache is not None:
            cache.stop()
        recorder.close()
    return recorder.directory


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Record the scripted demo session as a trace.")
    parser.add_argument("trace_parent", help="directory to create the session folder in")
    parser.add_argument("--work-dir", help="where tiles and snaps go (default: a temp dir)")
    parser.add_argument("--label", default="demo_session")
    parser.add_argument("--frames", default="hash", choices=("none", "hash", "preview", "full"))
    parser.add_argument(
        "--no-autofocus",
        action="store_true",
        help="leave out TESTAF and acquire with autofocus disabled (replays from digests alone)",
    )
    args = parser.parse_args(argv)
    work_dir = (
        Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="trace_work_"))
    )
    path = run_session(
        Path(args.trace_parent),
        work_dir,
        label=args.label,
        frame_mode=args.frames,
        with_autofocus=not args.no_autofocus,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
