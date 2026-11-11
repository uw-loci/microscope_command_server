"""PPM white balance tells the client which angle it is calibrating.

The calibrator calls the rotation callback once per angle, just before calibrating it, so
the handler reports PROGRESS:<n>:<of>:<angle deg> there; QPSC highlights that angle.
"""

import pathlib

HANDLER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "server"
    / "handlers"
    / "calibration.py"
)


def test_each_angle_is_reported_before_it_is_rotated_to():
    source = HANDLER.read_text()
    body = source[source.index("def handle_wbppm(") :]
    body = body[: body.index("\ndef ")]
    callback = body[body.index("def rotation_callback(angle_deg):") :]
    callback = callback[: callback.index("results = calibrator.calibrate_ppm(")]
    assert "PROGRESS:{progress['n']}:{n_angles}:{float(angle_deg)}" in callback
    assert callback.index("conn.sendall(") < callback.index("rotate(angle_deg)")
    assert "rotation_callback=rotation_callback" in body
