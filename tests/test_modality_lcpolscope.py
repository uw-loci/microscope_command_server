"""LC-PolScope modality registration and its capability invariants.

These are cheap guards on decisions that fail *silently* if reversed. The
Stokes inversion has no self-check: if the five states stop sharing one
radiometric scale, or the modality starts being treated as an angle modality,
the reconstruction still produces a plausible-looking retardance and
orientation map that is simply wrong. Nothing downstream will notice.
"""

import pytest

from microscope_command_server.modality import get_config, registered_prefixes
from microscope_command_server.modality.lcpolscope import (
    EXTINCTION_CHANNEL_ID,
    LCPOLSCOPE_CONFIG,
)


@pytest.mark.parametrize("modality", ["lcpolscope", "lcpolscope_20x", "lcps", "lcps_20x"])
def test_prefixes_resolve_to_the_lcpolscope_config(modality):
    """Both spellings the Java ModalityRegistry uses must resolve here too."""
    assert get_config(modality) is LCPOLSCOPE_CONFIG


def test_prefixes_are_registered():
    prefixes = registered_prefixes()
    assert "lcpolscope" in prefixes
    assert "lcps" in prefixes


def test_lcpolscope_does_not_collide_with_other_modalities():
    """Prefix matching is first-match-wins, so a collision would silently
    hand LC-PolScope another modality's rotation and white-balance settings."""
    for other in ["ppm_20x", "brightfield", "fl_20x", "widefield", "shg"]:
        assert get_config(other) is not LCPOLSCOPE_CONFIG


def test_is_a_channel_modality_not_an_angle_modality():
    """States are set electrically by the liquid crystal -- there is no stage
    to rotate, and no angle to move to before autofocus."""
    assert LCPOLSCOPE_CONFIG.has_rotation is False
    assert LCPOLSCOPE_CONFIG.autofocus_angle is None
    assert LCPOLSCOPE_CONFIG.default_angle_count == 1


def test_no_white_balance():
    """Monochrome camera at a single wavelength. A white balance would also
    rescale the five states unequally, biasing the inversion."""
    assert LCPOLSCOPE_CONFIG.wb_settings_key is None


def test_no_per_state_intensity_targets():
    """The per-state brightness spread is signal, not something to normalize.

    State0 is dark *by design* -- it is the extinction state. Per-state
    intensity targets would drive the states to a common brightness and
    destroy exactly the modulation the reconstruction measures.
    """
    assert LCPOLSCOPE_CONFIG.angle_intensity_targets == {}
    for angle in [0.0, 7.0, 45.0, 90.0]:
        assert LCPOLSCOPE_CONFIG.get_target_intensity(angle) == 200.0


def test_derived_outputs_are_separate_directories():
    """Retardance and orientation stitch independently, so they need their own
    tile directories -- and orientation must stay separable because it is
    axial data that cannot be resampled like an ordinary scalar."""
    assert LCPOLSCOPE_CONFIG.post_processing_suffixes == ["retardance", "orientation"]


def test_extinction_channel_id_matches_the_java_handler():
    """LCPolScopeModalityHandler.EXTINCTION_CHANNEL_ID must agree; autofocus
    skips this state on both sides."""
    assert EXTINCTION_CHANNEL_ID == "State0"


# ---------------------------------------------------------------------------
# Per-tile reconstruction guards
#
# These run without polscope_library: check_reconstruction_inputs deliberately
# validates before anything is imported or computed, so a misconfigured run is
# refused on the first tile rather than after an hour of imaging.
# ---------------------------------------------------------------------------


class _FakePool:
    """Stand-in for ctx.write_pool that records submissions instead of running them."""

    def __init__(self):
        self.submissions = []

    def submit(self, fn, **kwargs):
        self.submissions.append((fn, kwargs))


def _submit(**overrides):
    from microscope_command_server.modality import lcpolscope

    pool = _FakePool()
    kwargs = {
        "channel_images": {f"State{i}": object() for i in range(5)},
        "channel_order": [f"State{i}" for i in range(5)],
        "reconstruction_cfg": {
            "swing_waves": 0.03,
            "wavelength_nm": 549,
            "scheme": "5-State",
        },
        "output_path": None,
        "filename": "tile_0_0.tif",
        "pixel_size_um": 0.1715,
        "write_pool": pool,
        "ome_writer": None,
    }
    kwargs.update(overrides)
    queued = lcpolscope.submit_tile_reconstruction(**kwargs)
    return queued, pool


def test_raw_tile_background_correction_is_refused():
    """QLIPP corrects in Stokes space, not by dividing each state tile.

    background_correction_enabled arrives as a socket parameter, so the
    'enabled: false' in config_LCPolScope.yml does not gate it -- this check
    is the only thing standing between a flat-field divide and a silently
    biased orientation map.
    """
    from microscope_command_server.modality.lcpolscope import ReconstructionRefused

    with pytest.raises(ReconstructionRefused, match="flat-field corrected"):
        _submit(raw_tiles_flat_fielded=True)


def test_partial_state_set_is_refused():
    from microscope_command_server.modality.lcpolscope import ReconstructionRefused

    with pytest.raises(ReconstructionRefused, match="4 or 5 polarization states"):
        _submit(
            channel_images={f"State{i}": object() for i in range(3)},
            channel_order=[f"State{i}" for i in range(3)],
        )


def test_missing_state_image_is_refused_not_silently_dropped():
    """A dropped state must not reconstruct from whatever else is present."""
    from microscope_command_server.modality.lcpolscope import ReconstructionRefused

    with pytest.raises(ReconstructionRefused, match="missing state image"):
        _submit(channel_images={f"State{i}": object() for i in range(4)})


def test_states_are_passed_in_calibration_order_not_dict_order():
    """Order is positional; a permutation rotates or mirrors orientation."""
    order = ["State0", "State3", "State1", "State4", "State2"]
    images = {cid: f"img-{cid}" for cid in order}
    # Build the dict in a different order than the plan.
    shuffled = {cid: images[cid] for cid in sorted(order)}

    queued, pool = _submit(channel_images=shuffled, channel_order=order)

    assert queued is True
    assert len(pool.submissions) == 1
    _, kwargs = pool.submissions[0]
    assert kwargs["state_images"] == [images[cid] for cid in order]


@pytest.mark.parametrize("missing_key", ["swing_waves", "wavelength_nm"])
def test_incomplete_reconstruction_config_skips_rather_than_guesses(missing_key):
    """No default swing or wavelength: both are calibration facts, and a wrong
    value produces a wrong retardance scale with no visible symptom. Raw state
    images are still saved, so the run stays reconstructable offline."""
    cfg = {"swing_waves": 0.03, "wavelength_nm": 549}
    cfg.pop(missing_key)

    queued, pool = _submit(reconstruction_cfg=cfg)

    assert queued is False
    assert pool.submissions == []


def test_reconstruction_runs_end_to_end(tmp_path):
    """The queued callable actually reconstructs and writes both outputs."""
    pytest.importorskip("polscope_library")
    import numpy as np

    from microscope_command_server.modality.lcpolscope import (
        ORIENTATION_DIR,
        RETARDANCE_DIR,
    )

    written = {}

    def fake_writer(filename, pixel_size_um, data, **kw):
        written[filename] = (data, kw)

    states = [np.full((8, 8), v, dtype=float) for v in (10.0, 60.0, 55.0, 50.0, 45.0)]
    queued, pool = _submit(
        channel_images=dict(zip([f"State{i}" for i in range(5)], states)),
        output_path=tmp_path,
        ome_writer=fake_writer,
    )
    assert queued is True

    fn, kwargs = pool.submissions[0]
    fn(**kwargs)

    assert (tmp_path / RETARDANCE_DIR).is_dir()
    assert (tmp_path / ORIENTATION_DIR).is_dir()
    assert len(written) == 2
    for data, _ in written.values():
        assert data.shape == (8, 8)


def test_derived_dirs_get_a_tile_configuration(tmp_path):
    """Without TileConfiguration.txt the stitcher cannot place derived tiles,
    and the mosaic comes out empty while the per-tile files look fine."""
    pytest.importorskip("polscope_library")
    import numpy as np

    from microscope_command_server.modality.lcpolscope import (
        ORIENTATION_DIR,
        RETARDANCE_DIR,
    )

    # The acquisition writes the layout into the first state's directory.
    state0 = tmp_path / "State0"
    state0.mkdir()
    (state0 / "TileConfiguration.txt").write_text("dim = 2\ntile_0_0.tif; ; (0.0, 0.0)\n")

    states = [np.full((4, 4), v, dtype=float) for v in (10.0, 60.0, 55.0, 50.0, 45.0)]
    _, pool = _submit(
        channel_images=dict(zip([f"State{i}" for i in range(5)], states)),
        output_path=tmp_path,
        ome_writer=lambda **kw: None,
    )
    fn, kwargs = pool.submissions[0]
    fn(**kwargs)

    for subdir in (RETARDANCE_DIR, ORIENTATION_DIR):
        copied = tmp_path / subdir / "TileConfiguration.txt"
        assert copied.exists(), f"{subdir} has no tile layout"
        assert copied.read_text() == (state0 / "TileConfiguration.txt").read_text()


def test_missing_tile_configuration_does_not_break_reconstruction(tmp_path):
    """It is written during acquisition, so early tiles may reconstruct before
    it exists. That must not fail the tile."""
    pytest.importorskip("polscope_library")
    import numpy as np

    states = [np.full((4, 4), v, dtype=float) for v in (10.0, 60.0, 55.0, 50.0, 45.0)]
    _, pool = _submit(
        channel_images=dict(zip([f"State{i}" for i in range(5)], states)),
        output_path=tmp_path,
        ome_writer=lambda **kw: None,
    )
    fn, kwargs = pool.submissions[0]
    fn(**kwargs)  # must not raise


def test_partial_background_set_is_refused():
    """The correction is per-state in Stokes space, so filling the gaps with
    uncorrected states biases the result rather than merely weakening it."""
    from microscope_command_server.modality.lcpolscope import ReconstructionRefused

    with pytest.raises(ReconstructionRefused, match="partial background set"):
        _submit(background_images={f"State{i}": object() for i in range(3)})


def test_no_backgrounds_at_all_is_allowed():
    """Background-free reconstruction is valid, just lower quality."""
    queued, pool = _submit(background_images=None)
    assert queued is True
    assert pool.submissions[0][1]["background_images"] is None


def test_backgrounds_are_ordered_like_the_states():
    """reconstruct() pairs states and backgrounds positionally. A mismatched
    order corrects each state with another state's background, which neither
    raises nor looks wrong."""
    order = ["State0", "State3", "State1", "State4", "State2"]
    bgs = {cid: f"bg-{cid}" for cid in order}
    shuffled = {cid: bgs[cid] for cid in sorted(order)}

    _, pool = _submit(
        channel_images={cid: f"img-{cid}" for cid in order},
        channel_order=order,
        background_images=shuffled,
    )

    assert pool.submissions[0][1]["background_images"] == [bgs[cid] for cid in order]


def test_background_reaches_the_reconstruction(tmp_path):
    """A background must actually change the result, or it is not being used."""
    pytest.importorskip("polscope_library")
    import numpy as np

    def run(background_images):
        captured = {}
        _, pool = _submit(
            channel_images={
                f"State{i}": np.full((4, 4), v, dtype=float)
                for i, v in enumerate((10.0, 60.0, 55.0, 50.0, 45.0))
            },
            background_images=background_images,
            output_path=tmp_path,
            ome_writer=lambda filename, data, **kw: captured.setdefault(
                filename.rsplit("/", 2)[-2], data
            ),
        )
        fn, kwargs = pool.submissions[0]
        fn(**kwargs)
        return captured

    plain = run(None)
    corrected = run(
        {
            f"State{i}": np.full((4, 4), v, dtype=float)
            for i, v in enumerate((9.0, 58.0, 54.0, 49.0, 44.0))
        }
    )

    assert set(plain) == set(corrected)
    assert not np.allclose(plain["retardance"], corrected["retardance"])


# ---------------------------------------------------------------------------
# State ordering
# ---------------------------------------------------------------------------


def test_state_order_reorders_before_inversion():
    """OpenPolScope acquires in Oldenbourg order -- pairs (1,4) and (2,3) --
    so State3 and State4 swap before reconstruct() sees them. Verified against
    real reference-slide data: this permutation matches OpenPolScope's own
    orientation channel to 0.07 deg, every other permutation to 17-90 deg."""
    ids = [f"State{i}" for i in range(5)]
    queued, pool = _submit(
        channel_images={cid: f"img-{cid}" for cid in ids},
        channel_order=ids,
        reconstruction_cfg={
            "swing_waves": 0.03,
            "wavelength_nm": 549,
            "scheme": "5-State",
            "state_order": ["State0", "State1", "State2", "State4", "State3"],
        },
    )
    assert queued is True
    assert pool.submissions[0][1]["state_images"] == [
        "img-State0",
        "img-State1",
        "img-State2",
        "img-State4",
        "img-State3",
    ]


def test_backgrounds_follow_the_same_reordering():
    """Background states are paired positionally with sample states, so a
    reorder that moves one must move the other."""
    ids = [f"State{i}" for i in range(5)]
    _, pool = _submit(
        channel_images={cid: f"img-{cid}" for cid in ids},
        channel_order=ids,
        background_images={cid: f"bg-{cid}" for cid in ids},
        reconstruction_cfg={
            "swing_waves": 0.03,
            "wavelength_nm": 549,
            "state_order": ["State0", "State1", "State2", "State4", "State3"],
        },
    )
    assert pool.submissions[0][1]["background_images"] == [
        "bg-State0",
        "bg-State1",
        "bg-State2",
        "bg-State4",
        "bg-State3",
    ]


def test_omitted_state_order_keeps_acquisition_order():
    """recOrder's own naming is already (ext, +S1, +S2, -S1, -S2)."""
    ids = [f"State{i}" for i in range(5)]
    _, pool = _submit(channel_images={cid: f"img-{cid}" for cid in ids}, channel_order=ids)
    assert pool.submissions[0][1]["state_images"] == [f"img-{c}" for c in ids]


@pytest.mark.parametrize(
    "bad",
    [
        ["State0", "State1", "State2", "State3"],  # too few
        ["State0", "State1", "State2", "State3", "State3"],  # duplicate
        ["State0", "State1", "State2", "State3", "State9"],  # unknown id
    ],
)
def test_state_order_must_be_a_permutation(bad):
    """It reorders states; it does not select or invent them."""
    from microscope_command_server.modality.lcpolscope import ReconstructionRefused

    with pytest.raises(ReconstructionRefused, match="permutation"):
        _submit(
            reconstruction_cfg={
                "swing_waves": 0.03,
                "wavelength_nm": 549,
                "state_order": bad,
            }
        )


def test_derived_tiles_carry_their_provenance_and_handling_rules(tmp_path):
    """A reader cannot infer from the pixels that orientation is axial, nor
    which frame it is measured in. Both must travel with the file."""
    pytest.importorskip("polscope_library")
    import numpy as np

    from microscope_command_server.modality.lcpolscope import (
        ORIENTATION_DIR,
        RETARDANCE_DIR,
    )

    seen = {}

    def writer(*, filename, pixel_size_um, data, channel_name, map_annotations):
        seen[filename.rsplit("/", 2)[-2]] = (channel_name, map_annotations)

    _, pool = _submit(
        channel_images={
            f"State{i}": np.full((4, 4), v, dtype=float)
            for i, v in enumerate((10.0, 60.0, 55.0, 50.0, 45.0))
        },
        reconstruction_cfg={
            "swing_waves": 0.03,
            "wavelength_nm": 549,
            "scheme": "5-State",
            "state_order": ["State0", "State1", "State2", "State4", "State3"],
        },
        output_path=tmp_path,
        ome_writer=writer,
    )
    fn, kwargs = pool.submissions[0]
    fn(**kwargs)

    ret_name, ret_meta = seen[RETARDANCE_DIR]
    ori_name, ori_meta = seen[ORIENTATION_DIR]

    assert "nm" in ret_name and "Orientation" in ori_name
    assert ret_meta["polscope.units"] == "nanometres"
    assert ret_meta["polscope.axial"] == "false"
    assert ori_meta["polscope.units"] == "radians"
    assert ori_meta["polscope.axial"] == "true"
    assert "sin(2t)" in ori_meta["polscope.resample"]
    assert "mirror" in ori_meta["polscope.frame"]

    # The reconstruction parameters travel with BOTH channels, so a file found
    # on its own can still be traced back to the calibration that made it.
    for meta in (ret_meta, ori_meta):
        assert meta["polscope.wavelength_nm"] == 549
        assert meta["polscope.swing_waves"] == 0.03
        assert meta["polscope.scheme"] == "5-State"
        assert meta["polscope.state_order"] == "State0,State1,State2,State4,State3"
        assert meta["polscope.background_corrected"] == "false"
