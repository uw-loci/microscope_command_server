"""LC-PolScope drift correction (reconstruction.drift_correction).

A uniform leftover that grows after the background is subtracted from copies of the tiles. These
tests build five-state images of glass carrying a known leftover and of tissue carrying a known
specimen retardance plus the same leftover, then check that:
  - the YAML block is read strictly (a typo is refused, not ignored);
  - glass is found by brightness against the background, and tissue is not;
  - the leftover measured on glass is the one put in;
  - the model follows the leftover in time (tile glass, reference visits, both);
  - the corrected copies carry the specimen alone, beside untouched originals, in folders QPSC
    stitches;
  - a background-spot visit returns the stage exactly where it was.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from microscope_command_server.modality.lcpolscope_drift import (
    DRIFT_FOLDER,
    DRIFT_PREFIX,
    DriftLog,
    DEFAULT_REFERENCE_INTERVAL_MIN,
    DriftSettings,
    drift_settings,
    drift_settings_for_run,
    fit_drift_model,
    glass_mask,
    remove_vector,
    vector_nm_axis,
)

SWING, WAVELENGTH = 0.03, 549.0
SHAPE = (96, 128)


def _vec(nm, axis_deg):
    a = np.radians(2 * axis_deg)
    return nm * np.cos(a), nm * np.sin(a)


def _states(ret_nm, axis_rad, intensity):
    """Five state images of a field with the given retardance (nm), slow axis and brightness."""
    from polscope_library import calculate_stokes_to_intensity_matrix, mmul, stokes_after_adr

    ret = np.asarray(ret_nm, float) * 2 * np.pi / WAVELENGTH
    s = stokes_after_adr(
        ret, np.asarray(axis_rad, float), np.asarray(intensity, float), np.ones(SHAPE)
    )
    states = mmul(calculate_stokes_to_intensity_matrix(SWING), np.stack(s))
    return {f"State{i}": states[i] for i in range(5)}


def _tile(leftover, specimen=(10.0, 30.0)):
    """Left half glass (leftover only), right half tissue (half the transmittance, as stained
    tissue absorbs; specimen + leftover added as vectors, the first-order model the correction
    inverts)."""
    lx, ly = _vec(*leftover)
    sx, sy = _vec(*specimen)
    vx = np.full(SHAPE, lx)
    vy = np.full(SHAPE, ly)
    vx[:, SHAPE[1] // 2 :] += sx
    vy[:, SHAPE[1] // 2 :] += sy
    ret = np.hypot(vx, vy)
    axis = (np.arctan2(vy, vx) / 2) % np.pi
    inten = np.full(SHAPE, 2000.0)
    inten[:, SHAPE[1] // 2 :] = 1000.0
    return _states(ret, axis, inten)


def _background():
    return _states(np.full(SHAPE, 1e-6), np.zeros(SHAPE), np.full(SHAPE, 2000.0))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", [None, {}, {"tile_glass": False}, {"reference_interval_min": 0}])
def test_off_unless_a_measurement_is_asked_for(block):
    cfg = {} if block is None else {"drift_correction": block}
    assert drift_settings(cfg) is None


def test_settings_are_read():
    s = drift_settings(
        {
            "drift_correction": {
                "tile_glass": True,
                "reference_interval_min": 10,
                "reference_frames": 3,
            }
        }
    )
    assert s.tile_glass and s.reference_interval_min == 10 and s.reference_frames == 3
    assert "every 10 min" in s.describe() and "tile glass" in s.describe()


@pytest.mark.parametrize(
    "block, match",
    [
        ({"tile_glas": True}, "unknown key"),
        ({"tile_glass": "yes"}, "true or false"),
        ({"reference_interval_min": -5}, "reference_interval_min"),
        ({"tile_glass": True, "glass_min_ratio": 1.2}, "glass_min_ratio"),
        ({"tile_glass": True, "time_bandwidth_min": 0}, "time_bandwidth_min"),
        ("on", "mapping"),
    ],
)
def test_a_bad_block_is_refused(block, match):
    with pytest.raises(ValueError, match=match):
        drift_settings({"drift_correction": block})


RECON_WITH_BLOCK = {
    "drift_correction": {"tile_glass": False, "reference_interval_min": 3, "glass_max_ratio": 1.1}
}


def test_without_a_dialog_choice_the_yaml_decides():
    assert drift_settings_for_run({}, None) is None
    assert drift_settings_for_run(RECON_WITH_BLOCK, None).reference_interval_min == 3


def test_the_dialog_can_switch_the_yaml_block_off():
    assert drift_settings_for_run(RECON_WITH_BLOCK, False) is None


def test_the_dialog_switches_it_on_without_a_yaml_block():
    s = drift_settings_for_run({}, True)
    assert s.tile_glass and s.reference_interval_min == DEFAULT_REFERENCE_INTERVAL_MIN


def test_the_dialog_interval_wins_and_the_yaml_keeps_its_thresholds():
    s = drift_settings_for_run(RECON_WITH_BLOCK, True, 7)
    assert s.tile_glass and s.reference_interval_min == 7 and s.glass_max_ratio == 1.1
    # 0 = tile glass only, no visits to the background spot.
    assert drift_settings_for_run(RECON_WITH_BLOCK, True, 0).reference_interval_min == 0
    # No interval from the dialog: the YAML's.
    assert drift_settings_for_run(RECON_WITH_BLOCK, True).reference_interval_min == 3


def test_the_run_flags_are_parsed():
    from microscope_command_server.acquisition.workflow import parse_acquisition_message

    base = "--yaml c.yml --projects p --sample s --scan-type t --region r"
    on = parse_acquisition_message(base + " --bg-drift on --bg-drift-interval-min 5 ENDOFSTR")
    assert on["bg_drift"] is True and on["bg_drift_interval_min"] == 5.0
    assert parse_acquisition_message(base + " --bg-drift OFF ENDOFSTR")["bg_drift"] is False
    assert "bg_drift" not in parse_acquisition_message(base + " --bg-drift maybe ENDOFSTR")
    assert "bg_drift" not in parse_acquisition_message(base + " ENDOFSTR")


# ---------------------------------------------------------------------------
# Measurement and removal
# ---------------------------------------------------------------------------


def test_glass_is_found_by_brightness_against_the_background():
    pytest.importorskip("polscope_library")
    tile = _tile((0.8, 15.0))
    bg = _background()
    mask = glass_mask(list(tile.values()), list(bg.values()), 0.0, DriftSettings(tile_glass=True))
    assert mask[:, 5 : SHAPE[1] // 2 - 5].all()
    assert not mask[:, SHAPE[1] // 2 :].any()


def test_glass_still_reads_as_glass_at_another_exposure():
    pytest.importorskip("polscope_library")
    tile = {k: 2 * v for k, v in _tile((0.8, 15.0)).items()}  # twice the background's exposure
    bg = _background()
    s = DriftSettings(tile_glass=True)
    assert not glass_mask(list(tile.values()), list(bg.values()), 0.0, s).any()
    assert glass_mask(list(tile.values()), list(bg.values()), 0.0, s, exposure_ratio=2.0)[
        :, 10:50
    ].all()


def test_remove_vector_recovers_the_specimen():
    sx, sy = _vec(10.0, 30.0)
    lx, ly = _vec(0.8, 15.0)
    r = np.full((4, 4), np.hypot(sx + lx, sy + ly))
    o = np.full((4, 4), (np.arctan2(sy + ly, sx + lx) / 2) % np.pi)
    r2, o2 = remove_vector(r, o, lx, ly)
    np.testing.assert_allclose(r2, 10.0, atol=1e-9)
    np.testing.assert_allclose(np.degrees(o2), 30.0, atol=1e-7)


def test_vector_nm_axis():
    nm, axis = vector_nm_axis(*_vec(0.5, 170.0))
    assert nm == pytest.approx(0.5)
    assert axis == pytest.approx(170.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _log_with_glass(times_min, nm_per_min, axis=15.0, **settings):
    log = DriftLog(settings=DriftSettings(tile_glass=True, **settings), background_time=0.0)
    for i, t in enumerate(times_min):
        vx, vy = _vec(0.3 + nm_per_min * t, axis)
        log.add_tile(f"{i}.tif", t * 60.0, {}, (vx, vy, 1000, 0.5))
    return log


def test_tile_glass_model_follows_a_steady_rise():
    log = _log_with_glass(np.arange(5, 65, 0.5), 0.0125)  # 0.36 -> 1.1 nm, like run 018
    model = fit_drift_model(log)
    assert model.kind.startswith("tile glass")
    px, py = model.predict(np.array([10.0, 40.0]) * 60)
    np.testing.assert_allclose(np.hypot(px, py), [0.425, 0.8], atol=0.005)


def test_tile_glass_model_ignores_one_contaminated_tile():
    log = _log_with_glass(np.arange(5, 35, 0.5), 0.01)
    log.add_tile("bad.tif", 20 * 60.0, {}, (*_vec(3.0, 80.0), 1000, 0.5))
    px, py = fit_drift_model(log).predict(np.array([20.0 * 60]))
    assert np.hypot(px, py)[0] == pytest.approx(0.5, abs=0.01)


def test_tile_glass_model_is_held_constant_outside_the_measured_times():
    log = _log_with_glass(np.arange(20, 30, 0.5), 0.01)
    px, py = fit_drift_model(log).predict(np.array([0.0, 100.0]) * 60)
    np.testing.assert_allclose(np.hypot(px, py), [0.5, 0.595], atol=0.01)


def test_reference_visits_are_interpolated_from_zero_at_the_background():
    log = DriftLog(settings=DriftSettings(reference_interval_min=10), background_time=0.0)
    for t, nm in ((5, 0.3), (15, 0.5), (25, 0.6)):
        log.add_reference(t * 60.0, *_vec(nm, 15.0), 4096)
    model = fit_drift_model(log)
    assert model.detail["anchored_at_background"]
    px, py = model.predict(np.array([2.5, 10.0, 40.0]) * 60)
    np.testing.assert_allclose(np.hypot(px, py), [0.15, 0.4, 0.6], atol=1e-9)


def test_tile_glass_shifts_the_reference_curve_to_the_scanned_area():
    log = DriftLog(
        settings=DriftSettings(tile_glass=True, reference_interval_min=10), background_time=0.0
    )
    for t, nm in ((5, 0.3), (15, 0.5), (25, 0.7)):
        log.add_reference(t * 60.0, *_vec(nm, 15.0), 4096)
    for i, t in enumerate(np.arange(6, 24, 1.0)):
        log.add_tile(f"{i}.tif", t * 60.0, {}, (*_vec(0.25 + 0.02 * t + 0.2, 15.0), 1000, 0.5))
    model = fit_drift_model(log)
    assert "offset" in model.kind
    # curve 0.4 nm at 10 min; the scanned glass reads 0.25 nm above the spot throughout
    assert model.detail["offset_from_tile_glass_nm"][0] == pytest.approx(
        _vec(0.25, 15.0)[0], abs=0.005
    )
    px, py = model.predict(np.array([10.0 * 60]))
    assert np.hypot(px, py)[0] == pytest.approx(0.65, abs=0.01)


def test_no_model_from_too_little():
    log = _log_with_glass([5, 6], 0.01)
    assert fit_drift_model(log) is None


# ---------------------------------------------------------------------------
# End to end through the tile reconstruction (A) and the corrected copies (C)
# ---------------------------------------------------------------------------


def _writer(record):
    import tifffile

    def write(*, filename, pixel_size_um, data, channel_name, map_annotations):
        tifffile.imwrite(filename, np.asarray(data))
        record[filename] = dict(map_annotations)

    return write


def _reconstruct_tile(tmp_path, log, name, leftover, t, writer):
    from microscope_command_server.modality.lcpolscope import submit_tile_reconstruction

    class Pool:
        def submit(self, fn, **kw):
            kw["tile_time"] = t
            fn(**kw)

    (tmp_path / "State0").mkdir(exist_ok=True)
    (tmp_path / "State0" / "TileConfiguration.txt").write_text("dim = 2\n")
    bg = _background()
    submit_tile_reconstruction(
        channel_images=_tile(leftover),
        channel_order=[f"State{i}" for i in range(5)],
        reconstruction_cfg={"swing_waves": SWING, "wavelength_nm": WAVELENGTH, "scheme": "5-State"},
        output_path=tmp_path,
        filename=name,
        pixel_size_um=0.27,
        write_pool=Pool(),
        ome_writer=writer,
        background_images=bg,
        drift_log=log,
    )


def test_corrected_copies_carry_the_specimen_alone(tmp_path):
    pytest.importorskip("polscope_library")
    import tifffile

    from microscope_command_server.modality.lcpolscope import apply_drift_correction

    written = {}
    writer = _writer(written)
    log = DriftLog(
        settings=DriftSettings(tile_glass=True), background_time=0.0, folder=tmp_path / DRIFT_FOLDER
    )
    for i, minutes in enumerate(range(5, 65, 5)):
        _reconstruct_tile(
            tmp_path, log, f"{i}.tif", (0.3 + 0.0125 * minutes, 15.0), minutes * 60.0, writer
        )
    assert all("vx" in r for r in log.tiles.values())

    summary = apply_drift_correction(tmp_path, log, writer)
    assert summary["applied"] and summary["corrected_tiles"] == 12
    assert set(summary["folders"]) == {
        DRIFT_PREFIX + "retardance",
        DRIFT_PREFIX + "orientation",
        DRIFT_PREFIX + "orientation_color",
    }
    for name in ("0.tif", "11.tif"):
        orig = tifffile.imread(tmp_path / "retardance" / name) / 100.0
        corr = tifffile.imread(tmp_path / (DRIFT_PREFIX + "retardance") / name) / 100.0
        ori = tifffile.imread(tmp_path / (DRIFT_PREFIX + "orientation") / name) / 100.0
        glass, tissue = (slice(None), slice(10, 50)), (slice(None), slice(74, 118))
        assert np.median(orig[glass]) > 0.3  # the original keeps its leftover
        assert np.median(corr[glass]) < 0.03
        assert np.median(corr[tissue]) == pytest.approx(10.0, abs=0.05)
        assert np.median(ori[tissue]) == pytest.approx(30.0, abs=0.2)
        assert (tmp_path / (DRIFT_PREFIX + "retardance") / "TileConfiguration.txt").is_file()
    meta = written[str(tmp_path / (DRIFT_PREFIX + "retardance") / "11.tif")]
    assert meta["polscope.quantity"] == "retardance"
    assert meta["polscope.drift_removed_nm"] == pytest.approx(0.3 + 0.0125 * 60, abs=0.02)
    assert meta["polscope.drift_minutes_after_background"] == pytest.approx(60.0)
    saved = json.loads((tmp_path / DRIFT_FOLDER / "drift_model.json").read_text())
    assert saved["corrected_tiles"] == 12
    lines = (tmp_path / DRIFT_FOLDER / "drift_measurements.ndjson").read_text().splitlines()
    assert len(lines) == 12


def test_nothing_is_written_without_enough_measurements(tmp_path):
    pytest.importorskip("polscope_library")
    from microscope_command_server.modality.lcpolscope import apply_drift_correction

    writer = _writer({})
    log = DriftLog(settings=DriftSettings(tile_glass=True), folder=tmp_path / DRIFT_FOLDER)
    _reconstruct_tile(tmp_path, log, "0.tif", (0.5, 15.0), 60.0, writer)
    summary = apply_drift_correction(tmp_path, log, writer)
    assert not summary["applied"]
    assert not (tmp_path / (DRIFT_PREFIX + "retardance")).exists()


# ---------------------------------------------------------------------------
# Reference visit (B)
# ---------------------------------------------------------------------------


class _Hardware:
    def __init__(self, fields):
        self.fields = fields
        self.pos = SimpleNamespace(x=100.0, y=200.0, z=30.0)
        self.moves = []
        self.channel = None

    def get_current_position(self):
        return SimpleNamespace(**vars(self.pos))

    def move_to_position(self, p):
        self.moves.append((p.x, p.y, p.z))
        for axis in ("x", "y", "z"):
            if getattr(p, axis) is not None:
                setattr(self.pos, axis, getattr(p, axis))

    def set_exposure(self, ms):
        pass

    def snap_image(self):
        return self.fields[self.channel], {}


def test_reference_visit_records_the_spot_and_returns_the_stage(monkeypatch, tmp_path):
    pytest.importorskip("polscope_library")
    from microscope_command_server.acquisition import workflow

    leftover = (0.4, 15.0)
    spot = _states(
        np.full(SHAPE, leftover[0]), np.full(SHAPE, np.radians(leftover[1])), np.full(SHAPE, 2000.0)
    )
    hw = _Hardware(spot)
    plan = [{"id": f"State{i}", "exposure_ms": 100} for i in range(5)]
    monkeypatch.setattr(workflow, "resolve_channel_plan", lambda *a, **k: plan)

    def apply(hardware, entry, logger, preset_cache=None):
        hardware.channel = entry["id"]

    monkeypatch.setattr(workflow, "apply_channel_hardware_state", apply)
    log = DriftLog(
        settings=DriftSettings(reference_interval_min=10),
        reference_xyz=(5000.0, 600.0, 50.0),
        folder=tmp_path / DRIFT_FOLDER,
    )
    ctx = SimpleNamespace(
        lcps_drift=log,
        hardware=hw,
        logger=workflow.logging.getLogger("test"),
        is_cancelled=lambda: False,
        ppm_settings={
            "modalities": {
                "lcpolscope": {
                    "reconstruction": {
                        "swing_waves": SWING,
                        "wavelength_nm": WAVELENGTH,
                        "scheme": "5-State",
                    }
                }
            }
        },
        params={},
        channel_preset_cache={},
        channel_background_images=_background(),
    )
    workflow._maybe_visit_drift_reference(ctx)
    assert len(log.references) == 1
    nm, axis = vector_nm_axis(log.references[0]["vx"], log.references[0]["vy"])
    assert nm == pytest.approx(0.4, abs=0.01) and axis == pytest.approx(15.0, abs=0.5)
    # XY at the tile's Z, then Z; back in the reverse order, to exactly where it was.
    assert hw.moves == [
        (5000.0, 600.0, None),
        (None, None, 50.0),
        (None, None, 30.0),
        (100.0, 200.0, None),
    ]
    # Not due again until the interval has passed; the last visit always runs.
    workflow._maybe_visit_drift_reference(ctx)
    assert len(log.references) == 1
    workflow._maybe_visit_drift_reference(ctx, final=True)
    assert len(log.references) == 2
    # Only the first and the last visit keep their five state images.
    import tifffile

    for sub in ("reference_first", "reference_last"):
        files = sorted(p.name for p in (tmp_path / DRIFT_FOLDER / sub).iterdir())
        assert files == [f"State{i}.tif" for i in range(5)]
    img = tifffile.imread(tmp_path / DRIFT_FOLDER / "reference_first" / "State1.tif")
    assert img.dtype == np.uint16
    np.testing.assert_allclose(img, np.rint(spot["State1"]), atol=0)
    assert [r.get("images") for r in log.references] == ["reference_first", "reference_last"]


def _setup_ctx(recon, provenance=None, backgrounds=True):
    from microscope_command_server.acquisition import workflow

    return SimpleNamespace(
        modality="lcpolscope_20x",
        ppm_settings={"modalities": {"lcpolscope": {"reconstruction": recon}}},
        lcps_background_provenance=provenance or {},
        channel_background_images={"State0": np.zeros((2, 2))} if backgrounds else {},
        params={"channels": ["State0"], "channel_exposures": [100.0]},
        output_path=None,
        logger=workflow.logging.getLogger("test"),
    )


def test_setup_needs_a_background_and_refuses_the_estimated_background_too(tmp_path):
    from microscope_command_server.acquisition import workflow

    on = {"drift_correction": {"tile_glass": True}}
    prov = {
        "collected_at": "2026-11-17T16:58:08",
        "exposure_ms": {"State0": 50.0},
        "folder": str(tmp_path),
    }
    assert workflow._setup_lcpolscope_drift(_setup_ctx(on, prov, backgrounds=False)) is None
    both = {**on, "estimated_background": {"enabled": True}}
    assert workflow._setup_lcpolscope_drift(_setup_ctx(both, prov)) is None
    ctx = _setup_ctx(on, prov)
    ctx.output_path = tmp_path
    log = workflow._setup_lcpolscope_drift(ctx)
    assert log is not None and log.exposure_ratio == pytest.approx(2.0)
    # Kept with the background and calibration, where tile cleanup leaves files alone.
    assert log.folder == tmp_path / "_protected" / DRIFT_FOLDER
    assert log.reference_xyz is None  # no background_settings.yml in the folder
    (tmp_path / "background_settings.yml").write_text(
        "context:\n  stage_xy_um: [30294.0, 772.3]\n  stage_z_um: 50.9\n"
    )
    assert workflow._setup_lcpolscope_drift(ctx).reference_xyz == (30294.0, 772.3, 50.9)
