"""FLATFIT over a real socket pair."""

import json
import socket
import threading
import time

import numpy as np
import tifffile as tf

from microscope_command_server.server.handlers import flatfield as flatfield_handler
from microscope_command_server.server.handlers.flatfield import handle_flatfit, handle_flatpoll
from microscope_command_server.server.protocol import END_MARKER


def _exchange(handler, payload=b""):
    server_side, client_side = socket.socketpair()
    try:
        client_side.sendall(payload)
        handler(server_side, ("test", 0), None, None)
        client_side.settimeout(5)
        return client_side.recv(65536).decode("ascii").rstrip("\n")
    finally:
        server_side.close()
        client_side.close()


def _call(payload):
    """Start a fit and poll until it ends, as the client does."""
    reply = _exchange(handle_flatfit, payload + END_MARKER.encode())
    if reply != "STARTED":
        return reply
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        reply = _exchange(handle_flatpoll)
        if reply != "RUNNING":
            return reply
        time.sleep(0.02)
    raise AssertionError("the fit never finished")


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
    summaries = json.loads(reply[len("SUCCESS:") :])["profiles"]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["channel"] is None
    assert summary["offset"] == 6500.0
    assert summary["images_used"] == 10
    assert summary["flatfield_max"] > summary["flatfield_min"]
    assert (profile / "flatfield.tif").exists()
    assert "top PMT" in (profile / "profile.yml").read_text()


def test_a_region_of_channel_folders_fits_each_channel(tmp_path):
    rng = np.random.default_rng(1)
    y, x = np.mgrid[0:64, 0:64] / 63.0
    region = tmp_path / "region"
    for name, slope in (("DAPI", 0.3), ("FITC", -0.3)):
        (region / name).mkdir(parents=True)
        (region / name / "TileConfiguration.txt").write_text("dim = 2\n")
        for index in range(10):
            signal = rng.uniform(500, 900) * (1 + slope * (x - 0.5))
            tile = 6500 + signal + rng.normal(0, 20, (64, 64))
            tf.imwrite(region / name / f"{index}.tif", tile.astype(np.uint16))
    profile = tmp_path / "set"

    reply = _call(
        json.dumps({"tiles": str(region), "profile": str(profile), "offset": 6500.0}).encode()
    )

    summaries = json.loads(reply[len("SUCCESS:") :])["profiles"]
    assert [s["channel"] for s in summaries] == ["DAPI", "FITC"]
    assert (profile / "DAPI" / "flatfield.tif").exists()
    assert (profile / "FITC" / "flatfield.tif").exists()
    dapi = tf.imread(profile / "DAPI" / "flatfield.tif")
    fitc = tf.imread(profile / "FITC" / "flatfield.tif")
    assert dapi[:, -1].mean() > dapi[:, 0].mean()
    assert fitc[:, -1].mean() < fitc[:, 0].mean()


def test_a_missing_folder_is_reported_not_raised(tmp_path):
    reply = _call(json.dumps({"tiles": str(tmp_path), "profile": str(tmp_path / "p")}).encode())
    assert reply.startswith("FAILED:No TIFF files")


def test_a_second_fit_is_refused_while_one_runs(tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(flatfield_handler, "_fit", lambda request: release.wait(5) and "SUCCESS:{}")
    payload = json.dumps({"tiles": str(tmp_path), "profile": str(tmp_path)}).encode()
    assert _exchange(handle_flatfit, payload + END_MARKER.encode()) == "STARTED"
    assert _exchange(handle_flatpoll) == "RUNNING"
    second = _exchange(handle_flatfit, payload + END_MARKER.encode())
    assert second == "FAILED:A flat-field fit is already running"
    release.set()
    flatfield_handler._job_thread.join(5)
    assert _exchange(handle_flatpoll) == "SUCCESS:{}"


def test_a_bad_message_is_reported():
    assert _call(b"--tiles nope").startswith("FAILED:Bad FLATFIT message")
