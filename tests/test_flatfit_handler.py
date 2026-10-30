"""FLATFIT over a real socket pair."""

import json
import socket

import numpy as np
import tifffile as tf

from microscope_command_server.server.handlers.flatfield import handle_flatfit
from microscope_command_server.server.protocol import END_MARKER


def _call(payload):
    server_side, client_side = socket.socketpair()
    try:
        client_side.sendall(payload + END_MARKER.encode())
        handle_flatfit(server_side, ("test", 0), None, None)
        client_side.settimeout(5)
        return client_side.recv(4096).decode("ascii")
    finally:
        server_side.close()
        client_side.close()


def test_fits_a_folder_with_spaces_in_its_name(tmp_path):
    rng = np.random.default_rng(0)
    tiles = tmp_path / "tile folder"
    tiles.mkdir()
    y, x = np.mgrid[0:64, 0:64] / 63.0
    for index in range(10):
        signal = rng.uniform(500, 900) * (1 + 0.2 * (x - 0.5))
        tile = 6500 + signal + rng.normal(0, 20, (64, 64))
        tf.imwrite(tiles / f"{index}.tif", tile.astype(np.uint16))
    profile = tmp_path / "my profile"

    reply = _call(
        json.dumps(
            {"tiles": str(tiles), "profile": str(profile), "offset": 6500.0, "label": "top PMT"}
        ).encode()
    )

    assert reply.startswith("SUCCESS:")
    summary = json.loads(reply[len("SUCCESS:") :])
    assert summary["offset"] == 6500.0
    assert summary["images_used"] == 10
    assert summary["flatfield_max"] > summary["flatfield_min"]
    assert (profile / "flatfield.tif").exists()
    assert "top PMT" in (profile / "profile.yml").read_text()


def test_a_missing_folder_is_reported_not_raised(tmp_path):
    reply = _call(json.dumps({"tiles": str(tmp_path), "profile": str(tmp_path / "p")}).encode())
    assert reply.startswith("FAILED:No TIFF files")


def test_a_bad_message_is_reported():
    assert _call(b"--tiles nope").startswith("FAILED:Bad FLATFIT message")
