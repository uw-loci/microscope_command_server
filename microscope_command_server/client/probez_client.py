"""Standalone client for the PROBEZ diagnostic command.

Sends a CONFIG followed by a PROBEZ to a running microscope command
server. The interesting output lives in the server session log -- this
script only kicks the probe off and waits for the OK/FAIL response.

Usage:

    python -m microscope_command_server.client.probez_client \
        --config C:/path/to/config_PPM.yml

    python -m microscope_command_server.client.probez_client \
        --config C:/path/to/config_PPM.yml --host 127.0.0.1 --port 5000

After the probe finishes, find the newest server_session_*.log in the
same directory as the config file (under a 'logs' subfolder) and send
it back for analysis. Filter with:

    parse_server_log.py <log> --grep "PROBEZ" --short-time --no-level

The probe is read-mostly -- it snapshots every writable property on
the focus device, runs its tests, and restores all modified state
(including Z position) in a finally block on the server side.
"""

import argparse
import socket
import struct
import sys
import time

from microscope_command_server.server.protocol import ExtendedCommand, TCP_PORT


def _send_config(sock: socket.socket, config_path: str) -> bool:
    """Send CONFIG + path bytes and wait for CFG___OK."""
    path_bytes = config_path.encode("utf-8")
    sock.sendall(ExtendedCommand.CONFIG)
    sock.sendall(struct.pack("!I", len(path_bytes)))
    sock.sendall(path_bytes)

    response = sock.recv(8)
    if response == b"CFG___OK":
        # Drain the version-info payload (length-prefixed) so the
        # next command doesn't read stale bytes.
        length_bytes = sock.recv(4)
        if len(length_bytes) == 4:
            length = struct.unpack("!I", length_bytes)[0]
            if length > 0:
                remaining = length
                while remaining > 0:
                    chunk = sock.recv(min(remaining, 4096))
                    if not chunk:
                        break
                    remaining -= len(chunk)
        print("CONFIG OK")
        return True

    if response == b"CFG_BLCK":
        # Length-prefixed error message
        length_bytes = sock.recv(4)
        length = struct.unpack("!I", length_bytes)[0] if len(length_bytes) == 4 else 0
        err = sock.recv(length).decode("utf-8", errors="replace") if length else ""
        print(f"CONFIG BLOCKED: {err}", file=sys.stderr)
        return False

    if response == b"CFG_FAIL":
        length_bytes = sock.recv(4)
        length = struct.unpack("!I", length_bytes)[0] if len(length_bytes) == 4 else 0
        err = sock.recv(length).decode("utf-8", errors="replace") if length else ""
        print(f"CONFIG FAIL: {err}", file=sys.stderr)
        return False

    print(f"CONFIG unexpected response: {response!r}", file=sys.stderr)
    return False


def _send_probez(sock: socket.socket) -> bool:
    """Send PROBEZ and wait for PROBEZOK / PROBEZFL."""
    print("Sending PROBEZ...")
    print("The server will spend ~15-40 seconds running the probe battery.")
    print("All diagnostic output goes to the server session log.")
    sock.sendall(ExtendedCommand.PROBEZ)

    # The probe can take a while; give the server plenty of breathing room.
    sock.settimeout(120.0)
    response = sock.recv(8)
    if response == b"PROBEZOK":
        print("PROBEZ completed successfully")
        print("Check the server session log for 'PROBEZ [step-N]' entries.")
        return True
    if response == b"PROBEZFL":
        print("PROBEZ failed -- check the server session log for details.", file=sys.stderr)
        return False
    print(f"PROBEZ unexpected response: {response!r}", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config",
        required=True,
        help="Absolute path to the microscope config YAML (e.g. config_PPM.yml)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=TCP_PORT, help=f"Server port (default {TCP_PORT})"
    )
    args = parser.parse_args()

    print(f"Connecting to {args.host}:{args.port}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(10.0)
        sock.connect((args.host, args.port))
        sock.settimeout(30.0)

        if not _send_config(sock, args.config):
            return 2

        # Give the server a beat to finish any post-CONFIG initialization
        # logging before we hit it with the probe command.
        time.sleep(0.1)

        ok = _send_probez(sock)

        try:
            sock.sendall(ExtendedCommand.DISCONNECT)
        except Exception:
            pass

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
