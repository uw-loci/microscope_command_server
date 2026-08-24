"""LC-PolScope modality configuration.

Quantitative polarized light using a liquid-crystal universal compensator
(Meadowlark D5020). Five polarization states are set *electrically* per tile
via MicroManager ConfigGroup presets -- there is no rotation stage, so this is
a channel modality (State0..State4) and not an angle modality like PPM.

The reconstruction is QLIPP-style Stokes inversion (Guo et al., eLife 2020;9:
e55502), performed by ``polscope_library``. Nothing in this module does math;
it declares capabilities so core acquisition code can dispatch without
hardcoding the modality name.

Two invariants this modality does NOT enforce alone
---------------------------------------------------
1. **All five states share one exposure and gain.** The Stokes inversion treats
   the five intensities as samples of one radiometric scale, so a per-state
   difference biases retardance and orientation with *no visible symptom* --
   the images still look fine. This is held by three cooperating layers: the
   Java ``LCPolScopeModalityHandler.enforceEqualExposure``, equal ``exposure_ms``
   on every channel in ``config_LCPolScope.yml``, and the deliberate absence of
   ``channel_overrides`` on the LC-PolScope acquisition profiles. Do not add
   per-channel exposure tuning here or in the profile.

2. **State order is positional.** The states are consumed in calibration order;
   a permutation silently rotates or mirrors the orientation map rather than
   raising. If a calibration's provenance is unknown, identify it from data
   with ``polscope-scheme-check`` before trusting any orientation output.
"""

import logging

from .config import ModalityConfig
from .registry import register

logger = logging.getLogger(__name__)


# The extinction state. Excluded from autofocus: it is dark by design, so a
# focus metric computed on it is dominated by noise.
EXTINCTION_CHANNEL_ID = "State0"


LCPOLSCOPE_CONFIG = ModalityConfig(
    # No rotation stage -- polarization states are set electrically, so there
    # is no angle to move to before autofocus.
    autofocus_angle=None,
    has_rotation=False,
    # Channel modality: one "angle" per tile, five channels within it.
    default_angle_count=1,
    # Polarized transmitted light through a monochrome camera at a single
    # wavelength (549 nm). White balance is meaningless here, and applying one
    # would rescale the states unequally -- see invariant 1 above.
    wb_settings_key=None,
    # Deliberately no angle_intensity_targets: the per-state brightness spread
    # (extinction is dark by design) is signal, not something to normalize away.
    default_target_intensity=200.0,
    # Derived outputs written per tile by the reconstruction hook. Plain
    # directory names (not ".suffix" like PPM) because they sit alongside
    # State0..State4 rather than qualifying one of them, and the stitcher
    # treats every tile directory the same way.
    post_processing_suffixes=["retardance", "orientation"],
)


def register_lcpolscope():
    """Register LC-PolScope modality config with the registry."""
    register("lcpolscope", LCPOLSCOPE_CONFIG)
    register("lcps", LCPOLSCOPE_CONFIG)
    logger.debug("Registered LC-PolScope modality config")


# ---------------------------------------------------------------------------
# Per-tile reconstruction
# ---------------------------------------------------------------------------

RETARDANCE_DIR = "retardance"
ORIENTATION_DIR = "orientation"


class ReconstructionRefused(Exception):
    """Raised when the acquisition state makes a valid inversion impossible.

    Deliberately an error rather than a warning. Every failure mode this
    guards against produces a plausible-looking retardance and orientation
    map that is simply wrong, and nothing downstream can detect it.
    """


def check_reconstruction_inputs(channel_ids, raw_tiles_flat_fielded, background_images=None):
    """Validate acquisition state before any state images are reconstructed.

    Args:
        channel_ids: Channel ids acquired for this tile, in acquisition order.
        raw_tiles_flat_fielded: Whether a flat-field correction was already
            applied to the state images being handed in.
        background_images: Background state images keyed by channel id, or
            None. Must cover every state or none of them.

    Raises:
        ReconstructionRefused: if the inversion could not be trusted.
    """
    if raw_tiles_flat_fielded:
        raise ReconstructionRefused(
            "The state tiles were flat-field corrected before reconstruction, "
            "which is incompatible with QLIPP. QLIPP corrects in Stokes space "
            "using background *intensities* passed to reconstruct(); dividing "
            "each state tile by a flat field first is a different operation "
            "and biases retardance and orientation with no visible symptom. "
            "The acquisition path is supposed to skip the divide for this "
            "modality and route the backgrounds here instead."
        )

    n = len(channel_ids)
    if n not in (4, 5):
        raise ReconstructionRefused(
            f"Expected 4 or 5 polarization states, got {n} ({list(channel_ids)}). "
            "The scheme is fixed by the calibration; a partial state set cannot "
            "be inverted."
        )

    if background_images:
        missing = [cid for cid in channel_ids if cid not in background_images]
        if missing:
            raise ReconstructionRefused(
                f"Background images cover only {sorted(background_images)} but the "
                f"acquisition uses {list(channel_ids)}; missing {missing}. A partial "
                "background set cannot be used: the correction is a per-state "
                "subtraction in Stokes space, so filling the gaps with uncorrected "
                "states would bias the result rather than merely weaken it. Supply a "
                "background for every state, or none."
            )


def _copy_tile_configuration(source_dir, out_dir):
    """Give a derived tile directory the tile layout the stitcher needs.

    Without a TileConfiguration.txt the stitcher cannot place these tiles, so
    the retardance and orientation mosaics silently come out empty or
    scrambled while the per-tile files look perfectly fine on disk.

    Copied per tile rather than once, because the source file is written
    during acquisition and does not necessarily exist yet when the first tiles
    are reconstructed. Skipped once the destination has it.
    """
    import shutil

    dest = out_dir / "TileConfiguration.txt"
    if dest.exists() or source_dir is None:
        return
    source = source_dir / "TileConfiguration.txt"
    if source.exists():
        shutil.copy2(source, dest)


def _reconstruct_and_write(
    state_images,
    reconstruction_cfg,
    background_images,
    output_path,
    filename,
    pixel_size_um,
    ome_writer,
    log,
    tile_config_source=None,
):
    """Reconstruct one tile and write retardance + orientation. Runs in the write pool."""
    from polscope_library import reconstruct

    result = reconstruct(
        intensities=state_images,
        swing=float(reconstruction_cfg["swing_waves"]),
        wavelength_nm=float(reconstruction_cfg["wavelength_nm"]),
        scheme=str(reconstruction_cfg.get("scheme", "5-State")),
        background_intensities=background_images,
    )

    for subdir, data in (
        (RETARDANCE_DIR, result.retardance_nm),
        # Orientation is axial data in [0, pi). Written raw here on purpose --
        # any downsampling or blending of these pixels must go through
        # sin(2*theta)/cos(2*theta), never an arithmetic mean, or 179 deg and
        # 1 deg average to 90 deg: perpendicular to the truth and entirely
        # plausible-looking.
        (ORIENTATION_DIR, result.orientation_rad),
    ):
        out_dir = output_path / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        _copy_tile_configuration(tile_config_source, out_dir)
        ome_writer(
            filename=str(out_dir / filename),
            pixel_size_um=pixel_size_um,
            data=data,
        )
    if log is not None:
        log.debug("Reconstructed LC-PolScope tile %s", filename)


def submit_tile_reconstruction(
    *,
    channel_images,
    channel_order,
    reconstruction_cfg,
    output_path,
    filename,
    pixel_size_um,
    write_pool,
    ome_writer,
    raw_tiles_flat_fielded=False,
    background_images=None,
    logger_=None,
):
    """Queue birefringence reconstruction for one acquired tile.

    ``channel_order`` is authoritative: the states are consumed positionally
    in calibration order, and a permutation silently rotates or mirrors the
    orientation map rather than raising. It must come from the acquisition
    profile, never from dict iteration order.

    ``background_images`` is a per-state mapping (channel id -> array) of a
    specimen-free, slightly defocused field. It is applied by the inversion in
    Stokes space, NOT by dividing the raw tiles -- see check_reconstruction_inputs.

    Returns True if work was queued, False if it was skipped.

    Raises:
        ReconstructionRefused: if the inputs cannot yield a trustworthy result.
    """
    log = logger_ or logger

    missing = [cid for cid in channel_order if cid not in channel_images]
    if missing:
        raise ReconstructionRefused(
            f"Tile {filename} is missing state image(s) {missing}; "
            f"have {sorted(channel_images)}."
        )

    check_reconstruction_inputs(
        channel_order, raw_tiles_flat_fielded, background_images=background_images
    )

    if not reconstruction_cfg:
        log.warning(
            "No modalities.lcpolscope.reconstruction block; skipping "
            "reconstruction for %s. Raw state images are still saved.",
            filename,
        )
        return False
    for required in ("swing_waves", "wavelength_nm"):
        if reconstruction_cfg.get(required) is None:
            log.warning(
                "reconstruction.%s is not set; skipping reconstruction for %s. "
                "Raw state images are still saved.",
                required,
                filename,
            )
            return False

    ordered = [channel_images[cid] for cid in channel_order]
    # Backgrounds must be ordered exactly like the states: reconstruct() pairs
    # them positionally, so a mismatched order corrects each state with another
    # state's background -- which does not raise and does not look wrong.
    ordered_background = (
        [background_images[cid] for cid in channel_order] if background_images else None
    )
    # The first state's directory is the tile-layout reference; every channel
    # is imaged at the same positions, so any of them would do.
    tile_config_source = output_path / str(channel_order[0]) if output_path is not None else None
    write_pool.submit(
        _reconstruct_and_write,
        tile_config_source=tile_config_source,
        state_images=ordered,
        reconstruction_cfg=reconstruction_cfg,
        background_images=ordered_background,
        output_path=output_path,
        filename=filename,
        pixel_size_um=pixel_size_um,
        ome_writer=ome_writer,
        log=log,
    )
    return True
