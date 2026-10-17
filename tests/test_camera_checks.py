"""Camera settings that make polarimetric intensities meaningless.

The 2026-10-16 rig run had Gain Auto: Continuous and Gamma Enabled: 1 on an
8-bit pixel format. Every state the search settled on read 173 counts, the
camera's auto-gain target; the calibration reported "good".
"""

from microscope_command_server.calibration.camera_checks import camera_setting_problems

RIG_2026_10_16 = {
    "ADC Bit Depth": "Bit10",
    "Binning": "1",
    "Black Level": "0.0000",
    "Exposure Auto": "Off",
    "Gain": "0.0000",
    "Gain Auto": "Continuous",
    "Gamma": "0.8000",
    "Gamma Enabled": "1",
    "Pixel Format": "Mono8",
}


def test_the_rig_camera_is_refused_for_auto_gain_and_warned_for_gamma_and_8_bit():
    fatal, advisory = camera_setting_problems(RIG_2026_10_16, 8)
    assert len(fatal) == 1 and "Gain Auto" in fatal[0] and "Off" in fatal[0]
    assert any("Gamma Enabled" in w for w in advisory)
    assert any("8-bit" in w and "12- or 16-bit" in w for w in advisory)
    assert len(advisory) == 2


def test_a_linear_12_bit_camera_with_everything_off_is_clean():
    props = dict(
        RIG_2026_10_16, **{"Gain Auto": "Off", "Gamma Enabled": "0", "Pixel Format": "Mono16"}
    )
    assert camera_setting_problems(props, 12) == ([], [])


def test_auto_exposure_is_fatal_too():
    fatal, _ = camera_setting_problems({"ExposureAuto": "Once"}, 12)
    assert len(fatal) == 1 and "exposure" in fatal[0]


def test_gamma_without_an_enable_flag_is_judged_by_its_value():
    _, advisory = camera_setting_problems({"Gamma": "0.8"}, 12)
    assert len(advisory) == 1 and "1.0" in advisory[0]
    assert camera_setting_problems({"Gamma": "1.0"}, 12) == ([], [])


def test_a_camera_without_these_properties_reports_nothing():
    assert camera_setting_problems({"Binning": "1"}, None) == ([], [])


def test_significant_bits_takes_the_smaller_of_the_adapter_and_the_pixel_format():
    from microscope_command_server.calibration.camera_checks import significant_bits

    # Mono12p unpacked into a 16-bit buffer: the values still run 0 to 4095.
    assert significant_bits(16, "Mono12p") == 12
    assert significant_bits(16, "Mono12Packed") == 12
    assert significant_bits(16, "Mono16") == 16
    assert significant_bits(8, "Mono8") == 8
    assert significant_bits(12, None) == 12
    assert significant_bits(None, "Mono12p") == 12
    assert significant_bits(16, "Raw") == 16
    assert significant_bits(None, None) is None


def test_lcready_reports_camera_problems_and_the_adapter_wavelength():
    import json
    import struct
    from types import SimpleNamespace

    from microscope_command_server.server.handlers.calibration import handle_lcready

    class Core:
        props = {
            ("Cam", "Gain Auto"): "Continuous",
            ("Cam", "Pixel Format"): "Mono12p",
            ("MeadowlarkLC", "Wavelength"): "546.0000",
        }

        def get_camera_device(self):
            return "Cam"

        def get_image_bit_depth(self):
            return 16

        def get_device_property_names(self, device):
            return [p for (d, p) in self.props if d == device]

        def get_property(self, device, prop):
            return self.props[(device, prop)]

    sent = []
    conn = SimpleNamespace(sendall=sent.append)
    hardware = SimpleNamespace(
        core=Core(),
        get_exposure=lambda: 33.0,
        settings={"modalities": {"lcpolscope": {"reconstruction": {"wavelength_nm": 549}}}},
    )
    handle_lcready(conn, SimpleNamespace(addr="t"), hardware, None)
    (length,) = struct.unpack("!I", sent[0])
    report = json.loads(sent[1].decode("utf-8"))
    assert length == len(sent[1])
    assert report["camera"]["significant_bits"] == 12, "Mono12p in a 16-bit buffer"
    assert any("Gain Auto" in p for p in report["camera_fatal"])
    assert report["adapter_wavelength_nm"] == 546.0
    assert report["reconstruction_wavelength_nm"] == 549


def test_the_intercept_of_counts_against_exposure_is_the_offset():
    from microscope_command_server.calibration.camera_checks import (
        black_level_from_exposure_series,
    )

    est = black_level_from_exposure_series([10, 20, 30, 40], [150, 200, 250, 300])
    assert est["black_level"] == 100.0 and est["slope_per_ms"] == 5.0 and est["r_squared"] == 1.0
    assert black_level_from_exposure_series([10, 20], [1, 2]) is None, "two points is not a fit"
    assert (
        black_level_from_exposure_series([10, 10, 10], [1, 2, 3]) is None
    ), "no spread in exposure"
