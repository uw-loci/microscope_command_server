"""FlatfieldCorrector: one profile, per-channel profiles, and the ways a run stays uncorrected."""

import numpy as np

from microscope_command_server.acquisition.flatfield import FlatfieldCorrector
from microscope_command_server.acquisition.workflow import parse_acquisition_message
from microscope_command_server.server.handlers.utils import split_flag_message
from microscope_imageprocessing.correction import FlatfieldProfile


def _save(folder, value, offset=100.0):
    flatfield = np.full((4, 4), value, dtype=np.float32)
    FlatfieldProfile(flatfield=flatfield, offset=offset).save(folder)


def _plane(value=300):
    return np.full((4, 4), value, dtype=np.uint16)


def test_no_folder_changes_nothing():
    corrector = FlatfieldCorrector.load(None)
    image = _plane()
    assert corrector.apply(image) is image
    assert corrector.loaded == []


def test_a_single_profile_covers_every_channel(tmp_path):
    _save(tmp_path, 2.0)
    corrector = FlatfieldCorrector.load(str(tmp_path), ["DAPI", "FITC"])
    assert corrector.loaded == ["all"]
    assert corrector.apply(_plane(), "FITC")[0, 0] == 200
    assert corrector.apply(_plane())[0, 0] == 200


def test_each_channel_uses_its_own_profile(tmp_path):
    _save(tmp_path / "DAPI", 2.0)
    _save(tmp_path / "FITC", 4.0)
    corrector = FlatfieldCorrector.load(str(tmp_path), ["DAPI", "FITC", "Cy5"])
    assert corrector.loaded == ["DAPI", "FITC"]
    assert corrector.apply(_plane(), "DAPI")[0, 0] == 200
    assert corrector.apply(_plane(), "FITC")[0, 0] == 150
    assert corrector.apply(_plane(), "Cy5")[0, 0] == 300


def test_a_folder_with_no_profile_warns_and_stays_uncorrected(tmp_path, caplog):
    corrector = FlatfieldCorrector.load(str(tmp_path))
    assert corrector.apply(_plane())[0, 0] == 300
    assert "No flat-field profile" in caplog.text


def test_a_wrong_size_warns_once_and_the_run_goes_on(tmp_path, caplog):
    _save(tmp_path, 2.0)
    corrector = FlatfieldCorrector.load(str(tmp_path))
    big = np.full((8, 8), 300, dtype=np.uint16)
    assert corrector.apply(big)[0, 0] == 300
    assert corrector.apply(big)[0, 0] == 300
    assert caplog.text.count("another scan size") == 1


def test_clipped_pixels_are_counted(tmp_path):
    _save(tmp_path, 0.5, offset=0.0)
    corrector = FlatfieldCorrector.load(str(tmp_path))
    corrector.apply(np.full((4, 4), 200, dtype=np.uint8))
    assert corrector.clipped_pixels == 16


def test_zstack_flags_keep_a_quoted_folder_and_every_apostrophe():
    assert split_flag_message('--flatfield-folder "F:\\flat field\\a" --x 1') == [
        "--flatfield-folder",
        "F:\\flat field\\a",
        "--x",
        "1",
    ]
    assert split_flag_message("--output C:\\Bob's\\Mike's\\run --x 1") == [
        "--output",
        "C:\\Bob's\\Mike's\\run",
        "--x",
        "1",
    ]


def test_a_wrong_sized_channel_does_not_fall_through_to_the_common_profile(tmp_path):
    _save(tmp_path, 2.0)
    profile = FlatfieldProfile(flatfield=np.full((8, 8), 2.0, dtype=np.float32), offset=100.0)
    profile.save(tmp_path / "DAPI")
    corrector = FlatfieldCorrector.load(str(tmp_path), ["DAPI"])
    image = _plane()
    assert corrector.apply(image, "DAPI") is image
    assert corrector.apply(image, "DAPI") is image
    assert corrector.loaded == ["all"]


def test_the_acquire_message_carries_a_folder_with_spaces():
    params = parse_acquisition_message(
        '--yaml y --projects p --sample s --scan-type t --region r --flatfield-folder "F:\\flat field\\top PMT" ENDOFSTR'
    )
    assert params["flatfield_folder"] == "F:\\flat field\\top PMT"
