"""Background profiles: the server records the profile QPSC resolved the folder under, never
derives a folder itself, and says in acquisition_metadata.json what was actually applied."""

import json
import struct

import numpy as np
import tifffile

from microscope_command_server.acquisition import workflow as workflow_mod
from microscope_command_server.server.handlers import live
from microscope_command_server.server.protocol import END_MARKER


class FakeConn:
    def __init__(self, payload=b""):
        self._payload = payload
        self.sent = b""
        self.timeout = None

    def recv(self, n):
        chunk, self._payload = self._payload[:n], self._payload[n:]
        return chunk

    def sendall(self, data):
        self.sent += data

    def settimeout(self, t):
        self.timeout = t

    def gettimeout(self):
        return self.timeout


class FakeClient:
    addr = ("127.0.0.1", 0)


class FakeHardware:
    def __init__(self, frame):
        self._frame = frame

    def get_live_frame(self):
        return self._frame, None

    def get_rotation_position(self):
        raise RuntimeError("no rotation stage")


def test_the_parser_keeps_the_background_profile():
    params = workflow_mod.parse_acquisition_message(
        "--yaml c.yml --projects p --sample s --scan-type bf_10x --region r --angles () "
        "--exposures (10.0) --bg-correction true --bg-method divide "
        "--bg-folder D:/bg/profiles/carrier4/DET/bf/10x --bg-profile carrier4 --wb-mode off"
    )
    assert params["background_profile"] == "carrier4"
    assert params["background_folder"] == "D:/bg/profiles/carrier4/DET/bf/10x"


def test_metadata_says_which_background_and_whether_it_was_applied(tmp_path):
    params = {
        "sample_label": "s",
        "scan_type": "bf_10x",
        "angles": [],
        "exposures": [10.0],
        "background_correction_enabled": True,
        "background_correction_method": "divide",
        "background_profile": "carrier4",
        "background_folder": "D:/bg/profiles/carrier4/DET/bf/10x",
        "background_disabled_angles": [7.0],
        "background_correction_applied": False,
        "background_angles_loaded": [],
        "background_channels_loaded": [],
    }
    workflow_mod.write_acquisition_metadata(
        output_path=tmp_path, params=params, tile_measurements=[], sat_monitor=None
    )
    meta = json.loads((tmp_path / "acquisition_metadata.json").read_text())
    acq = meta["acquisition_parameters"] if "acquisition_parameters" in meta else meta
    found = _find_key(meta, "bg_profile")
    assert found == "carrier4"
    assert _find_key(meta, "bg_folder") == "D:/bg/profiles/carrier4/DET/bf/10x"
    assert _find_key(meta, "bg_correction_applied") is False
    assert _find_key(meta, "bg_disabled_angles") == [7.0]
    assert _find_key(meta, "bg_correction_enabled") is True
    del acq


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def test_correctframe_uses_the_folder_it_is_sent(tmp_path):
    bg = np.full((4, 6), 2000, dtype=np.uint16)
    bg[:, :3] = 1000  # left half darker: the flat-field doubles the left half
    tifffile.imwrite(tmp_path / "background.tif", bg)
    frame = np.full((4, 6), 1000, dtype=np.uint16)
    conn = FakeConn(
        (f"--bg-folder {tmp_path} --bg-method divide --modality bf" + END_MARKER).encode()
    )
    live.handle_correctframe(conn, FakeClient(), FakeHardware(frame), {})
    assert not conn.sent.startswith(b"FAILED"), conn.sent[:80]
    w, h, ch, bpp, n = struct.unpack(">5i", conn.sent[:20])
    assert (w, h, ch, bpp) == (6, 4, 1, 2)
    corrected = np.frombuffer(conn.sent[20:], dtype=">u2").reshape(h, w)
    assert corrected[0, 0] > corrected[0, 5], "the darker background side is lifted"
    assert conn.timeout is None, "the socket timeout is restored"


def test_correctframe_refuses_without_a_folder_and_with_a_missing_one(tmp_path):
    conn = FakeConn(("--bg-method divide" + END_MARKER).encode())
    live.handle_correctframe(conn, FakeClient(), FakeHardware(np.zeros((2, 2), np.uint16)), {})
    assert conn.sent.startswith(b"FAILED:No --bg-folder")

    conn = FakeConn((f"--bg-folder {tmp_path / 'nope'}" + END_MARKER).encode())
    live.handle_correctframe(conn, FakeClient(), FakeHardware(np.zeros((2, 2), np.uint16)), {})
    assert conn.sent.startswith(b"FAILED:Background folder not found")


def test_a_path_with_spaces_survives_the_flag_reader():
    conn = FakeConn(("--bg-folder C:/My Data/bg --bg-method divide" + END_MARKER).encode())
    params = live._read_flag_message(conn, ("--bg-folder", "--bg-method"))
    assert params == {"--bg-folder": "C:/My Data/bg", "--bg-method": "divide"}
