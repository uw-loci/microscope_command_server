"""LCSWEEP: measuring each crystal's own voltage curve, against a rig whose adapter table is wrong.

The fake crystals follow a smooth true curve. The fake adapter converts waves to volts through a
table that matches it except for a kink near LC-B's working point, as the LC-PolScope's
mmgr_dal_MeadowlarkLC.csv does across 4.1-4.6 V (2026-10-28). The sweep must recover the true
curve, see the table's error, and report the palette's real offsets.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("polscope_library")
calibration = pytest.importorskip("polscope_library.calibration")
if not hasattr(calibration, "SWEEP_GEOMETRY"):
    pytest.skip("polscope_library predates SWEEP_GEOMETRY", allow_module_level=True)

from microscope_command_server.calibration.lc_voltage_sweep import (  # noqa: E402
    run_lc_voltage_sweep,
)

LC = "MeadowlarkLC"


def true_waves(volts):
    """What the crystal really does: smooth, 1.6 waves at 0 V down to 0.07 at 20 V."""
    return 0.07 + 1.53 / (1.0 + (np.asarray(volts, dtype=float) / 2.2) ** 1.6)


def table_waves(volts):
    """The adapter's belief: the true curve plus a bump near 0.45 waves (about 4.4 V)."""
    v = np.asarray(volts, dtype=float)
    return true_waves(v) + 0.02 * np.exp(-(((v - 4.4) / 0.25) ** 2))


_GRID = np.linspace(0.0, 20.0, 20001)


def table_volts(waves):
    w = table_waves(_GRID)
    return float(np.interp(waves, w[::-1], _GRID[::-1]))


class SweepCore:
    def __init__(self):
        self.volts = {"LCA": table_volts(0.25), "LCB": table_volts(0.5)}
        self.props = {("FakeCam", "PixelType"): "12bit", ("FakeCam", "GainAuto"): "Off"}

    # adapter
    def set_property(self, device, prop, value):
        if device == LC and prop.startswith("Voltage (V) LC-"):
            self.volts["LC" + prop[-1]] = float(value)
        elif device == LC and prop.startswith("Retardance LC-") and prop.endswith("[in waves]"):
            self.volts["LC" + prop.split("LC-")[1][0]] = table_volts(float(value))
        else:
            self.props[(device, prop)] = value

    def get_property(self, device, prop):
        if device == LC and prop.startswith("Voltage (V) LC-"):
            return f"{self.volts['LC' + prop[-1]]:.4f}"
        if device == LC and prop.endswith("[in waves]"):
            return f"{float(table_waves(self.volts['LC' + prop.split('LC-')[1][0]])):.4f}"
        return self.props[(device, prop)]

    def has_property_limits(self, device, prop):
        return device == LC and prop.startswith("Voltage")

    def get_property_lower_limit(self, device, prop):
        return 0.0

    def get_property_upper_limit(self, device, prop):
        return 20.0

    def get_device_property_names(self, device):
        return [p for (d, p) in self.props if d == device]

    def wait_for_device(self, device):
        pass

    # camera
    def get_camera_device(self):
        return "FakeCam"

    def get_image_bit_depth(self):
        return 12

    def get_bytes_per_pixel(self):
        return 2

    def get_image_width(self):
        return 4

    def get_image_height(self):
        return 4


class SweepHardware:
    """Counts at full transmission are 6x the 12-bit range at 50 ms, as a real sweep sees."""

    def __init__(self):
        self.core = SweepCore()
        self.exposure = 50.0

    def snap_image(self):
        from polscope_library.calibration.simulator import SimulatedPolScope

        scope = SimulatedPolScope(read_noise=0.0, transmittance=24000.0, black_level=100.0)
        a = float(true_waves(self.core.volts["LCA"]))
        b = float(true_waves(self.core.volts["LCB"]))
        value = 100.0 + (scope.intensity_at(a, b) - 100.0) * self.exposure / 50.0
        return np.full((4, 4), min(value, 4095.0), dtype=np.uint16), {}

    def get_camera_name(self):
        return "FakeCam"

    def get_exposure(self):
        return self.exposure

    def set_exposure(self, ms):
        self.exposure = float(ms)


def _yaml():
    def state(sid, a, b):
        return {
            "id": sid,
            "device_properties": [
                {"device": LC, "property": "Retardance LC-A [in waves]", "value": a},
                {"device": LC, "property": "Retardance LC-B [in waves]", "value": b},
            ],
        }

    # The rig's 10-26 shape: LC-A symmetric, LC-B lopsided in commanded waves.
    return {
        "modalities": {
            "lcpolscope": {
                "reconstruction": {"wavelength_nm": 549, "black_level": 100, "lc_settle_ms": 0},
                "channels": [
                    state("State0", 0.25, 0.44),
                    state("State1", 0.22, 0.44),
                    state("State2", 0.25, 0.454),
                    state("State3", 0.28, 0.44),
                    state("State4", 0.25, 0.415),
                ],
            }
        }
    }


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    out = tmp_path_factory.mktemp("sweep")
    hw = SweepHardware()
    result = run_lc_voltage_sweep(hw, _yaml(), output_folder=str(out), step_v=0.02)
    return hw, result, json.loads(Path(result["metadata_path"]).read_text())


def test_both_crystals_are_measured_without_clipping(swept):
    _, result, record = swept
    assert result["success"] is True, result
    for axis in ("LCA", "LCB"):
        entry = record["axes"][axis]
        assert "error" not in entry, entry.get("error")
        assert entry["clipped_points"] == 0
        assert entry["exposure_ms"] < 50.0, "the exposure is chosen for the sweep, not reused"


@pytest.mark.parametrize("axis", ["LCA", "LCB"])
def test_measured_curve_is_the_true_one(swept, axis):
    _, _, record = swept
    c = record["axes"][axis]["curve"]
    v = np.asarray(c["volts"])
    err = np.abs(np.asarray(c["waves"]) - true_waves(v))
    assert err.max() < 0.01, f"{axis}: worst {err.max():.4f} waves at {v[np.argmax(err)]:.2f} V"


def test_the_table_error_is_found_where_it_is(swept):
    _, result, _ = swept
    worst = result["axes"]["LCB"]["table_minus_measured_waves"]
    assert worst["max_abs"] == pytest.approx(0.02, abs=0.006)
    assert worst["at_volts"] == pytest.approx(4.4, abs=0.3)


def test_palette_rows_report_commanded_and_measured_offsets(swept):
    _, result, _ = swept
    rows = result["palette"]
    for state in ("State2", "State4"):
        row = rows[state]
        assert row["lcb_waves_measured"] == pytest.approx(
            float(true_waves(row["lcb_volts"])), abs=0.01
        )
        assert "lcb_offset_measured" in row and "lcb_offset_volts" in row
    assert any(line.startswith("State2: LC-B") for line in result["palette_lines"])


def test_crystals_exposure_and_files_are_left_tidy(swept):
    hw, result, _ = swept
    assert hw.exposure == 50.0
    # Back to the extinction the sweep found the adapter at (its waves, through its table).
    assert hw.core.volts["LCA"] == pytest.approx(table_volts(0.25), abs=0.01)
    text = Path(result["curve_csv_path"]).read_text()
    assert "Not the Meadowlark adapter's file format" in text
    assert text.splitlines()[2].startswith("Voltage(mv),549-A,549-B")


def test_the_socket_handler_runs_a_sweep(tmp_path):
    from types import SimpleNamespace

    from microscope_command_server.server.handlers.calibration import handle_lcsweep

    message = f"--output {tmp_path} --axes LCB --step-v 0.05 ENDOFSTR".encode()
    sent = []
    conn = SimpleNamespace(
        recv=lambda n, buf=[message]: buf.pop() if buf else b"",
        sendall=sent.append,
        settimeout=lambda t: None,
    )
    handle_lcsweep(conn, SimpleNamespace(addr="t"), SweepHardware(), _yaml())
    assert sent[0] == f"STARTED:{tmp_path}".encode()
    final = sent[-1].decode()
    assert final.startswith("SUCCESS:"), final[:300]
    summary = json.loads(final[len("SUCCESS:") :])
    assert set(summary["axes"]) == {"LCB"}
    assert summary["success"] is True


def test_the_socket_handler_needs_an_output_folder():
    from types import SimpleNamespace

    from microscope_command_server.server.handlers.calibration import handle_lcsweep

    sent = []
    conn = SimpleNamespace(
        recv=lambda n, buf=[b"--axes LCA ENDOFSTR"]: buf.pop() if buf else b"",
        sendall=sent.append,
        settimeout=lambda t: None,
    )
    handle_lcsweep(conn, SimpleNamespace(addr="t"), SweepHardware(), _yaml())
    assert sent == [b"FAILED:--output is required"]
