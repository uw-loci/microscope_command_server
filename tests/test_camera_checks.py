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
