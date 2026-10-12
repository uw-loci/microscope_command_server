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

import numpy as np

from microscope_imageprocessing.io import (
    RESAMPLE_ANGULAR_180,
    RESAMPLE_LINEAR,
    channel_handling,
)

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
    # wavelength (546 nm). White balance is meaningless here, and applying one
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

# Derived tiles are written as uint16, not float, because the stitcher cannot
# consume anything else: ChunkCompositor builds TYPE_USHORT_GRAY / TYPE_BYTE_GRAY
# and PyramidLevelGenerator branches only on is16Bit. Float tiles would
# reconstruct correctly and then be unstitchable.
#
# ORIENTATION uses OpenPolScope's own convention -- 0..18000 spanning 0..180
# degrees, i.e. hundredths of a degree -- so our files are directly comparable
# with theirs. Confirmed from their 2026-08-25 output, whose orientation channel
# maxes at exactly 18000.
ORIENTATION_COUNTS_PER_DEGREE = 100.0
ORIENTATION_MAX_COUNTS = 18000

# RETARDANCE uses an explicit scale of our own rather than OpenPolScope's.
# Theirs is a single linear factor measured at ~1097.5 counts/nm. That is close
# to 2*wavelength, but which wavelength is unclear and the agreement is not
# clean enough to settle it: 2*549 = 1098 (0.05% away) fits better than
# 2*546 = 1092 (0.5% away), yet the formula is undocumented, so the whole
# resemblance may be a coincidence. Treat it as a loose end, not as evidence
# for a wavelength. Either way, reproducing their scale would mean guessing. Hundredths of a nanometre
# is far finer than the noise and spans 0..655 nm, while the algorithm caps
# retardance at a quarter wave (137 nm at 546 nm), so there is ample headroom.
RETARDANCE_COUNTS_PER_NM = 100.0
_UINT16_MAX = 65535

# modalities.lcpolscope.reconstruction.method. Validated up front so that an
# unknown method is refused once at acquisition start, not raised per tile
# inside the write pool. The labels are what the OME metadata carries.
DEFAULT_RECONSTRUCTION_METHOD = "qlipp"
RECONSTRUCTION_LABELS = {
    "qlipp": "QLIPP Stokes inversion (polscope_library)",
    "openpolscope": "OpenPolScope 3.20b 5-frame, bit-exact (polscope_library)",
}


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
                "background set cannot be used: the background states are inverted to "
                "Stokes parameters and collapsed into ONE Mueller matrix, whose "
                "inverse is applied to the sample. A missing state corrupts that "
                "whole matrix rather than degrading one channel. Supply a "
                "background for every state, or none."
            )

    # Checked here rather than left to the import inside the write pool.
    # polscope_library is an optional extra, so a rig set up without it is a
    # realistic configuration -- and if that surfaces as an ImportError raised
    # on a worker thread at the first tile write, the operator learns about it
    # somewhere in a log, per tile, partway into a slide. As a refusal it is
    # reported once, up front, in the same place as every other reason a
    # reconstruction cannot be trusted, and the run still saves its raw states.
    try:
        import polscope_library  # noqa: F401
    except ImportError as exc:
        raise ReconstructionRefused(
            f"polscope_library is not installed in the server environment ({exc}). "
            "The raw state images are still saved, so this run can be "
            "reconstructed offline once it is. Install it from the checkout "
            "next to this repo -- 'pip install -e ../polscope_library', or run "
            "update_env.bat with the environment active. It is not on PyPI, so "
            "the [polscope] extra cannot fetch it on its own."
        ) from exc


def _to_uint16(values, counts_per_unit, max_counts=_UINT16_MAX):
    """Scale a physical map to uint16 counts, reporting whether it clipped.

    Returns (array, clipped). Clipping is reported rather than silently
    absorbed: a saturated retardance map still looks like data.
    """
    scaled = np.rint(np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0) * counts_per_unit)
    clipped = bool(np.any(scaled > max_counts) or np.any(scaled < 0))
    return np.clip(scaled, 0, max_counts).astype(np.uint16), clipped


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
    state_order=None,
    lc_state_readback=None,
):
    """Reconstruct one tile and write retardance + orientation. Runs in the write pool."""
    from polscope_library import reconstruct

    method = str(reconstruction_cfg.get("method", DEFAULT_RECONSTRUCTION_METHOD))
    black_level = float(reconstruction_cfg.get("black_level", 0.0))
    result = reconstruct(
        intensities=state_images,
        swing_waves=float(reconstruction_cfg["swing_waves"]),
        wavelength_nm=float(reconstruction_cfg["wavelength_nm"]),
        scheme=str(reconstruction_cfg.get("scheme", "5-State")),
        background_intensities=background_images,
        method=method,
        black_level=black_level,
    )

    shared = {
        "polscope.wavelength_nm": reconstruction_cfg.get("wavelength_nm"),
        "polscope.swing_waves": reconstruction_cfg.get("swing_waves"),
        "polscope.scheme": reconstruction_cfg.get("scheme", "5-State"),
        "polscope.state_order": ",".join(str(c) for c in (state_order or [])),
        "polscope.background_corrected": "true" if background_images else "false",
        # Which algorithm made this file. The two agree without a background
        # and differ by ~0.5% with one, so a file found on its own has to say.
        "polscope.reconstruction": RECONSTRUCTION_LABELS[method],
        "polscope.reconstruction_method": method,
        "polscope.black_level": black_level,
    }
    # Which LC calibration set the states. Written into the YAML by QPSC's
    # "Apply to acquisition"; absent on a rig still running OpenPolScope's
    # presets, and then deliberately absent here too.
    for key in ("lc_calibration_file", "lc_calibrated_at", "lc_extinction_ratio"):
        if reconstruction_cfg.get(key) is not None:
            shared["polscope." + key] = str(reconstruction_cfg[key])
    # What the crystals READ when each state was applied, against what the
    # YAML commanded -- the acquisition-side half of the trace the calibration
    # file starts. "agrees" is the one word to look at.
    if lc_state_readback:
        shared["polscope.lc_readback"] = lc_readback_summary(lc_state_readback)
        shared["polscope.lc_readback_agrees"] = (
            "true"
            if all(v["agrees"] for e in lc_state_readback.values() for v in e.values())
            else "false"
        )
    # Resampling policy uses the shared vocabulary rather than a polscope-specific
    # key: a mask, a label map or an object-id channel needs the same protection,
    # and a reader should only have to understand one convention.
    per_channel = {
        RETARDANCE_DIR: {
            "polscope.quantity": "retardance",
            "polscope.units": "nanometres",
            "polscope.counts_per_unit": RETARDANCE_COUNTS_PER_NM,
            "polscope.to_physical": "nanometres = counts / 100",
            # Ordinary non-negative scalar. Nothing special.
            **channel_handling(RESAMPLE_LINEAR),
        },
        ORIENTATION_DIR: {
            "polscope.quantity": "slow_axis_orientation",
            "polscope.units": "degrees",
            "polscope.range": "[0,180)",
            "polscope.counts_per_unit": ORIENTATION_COUNTS_PER_DEGREE,
            "polscope.to_physical": "degrees = counts / 100 (OpenPolScope convention)",
            # The frame note stays polscope-local: it is about how the values
            # relate to image geometry, not about how they may be combined.
            "polscope.frame": "image (y-down); a single mirror negates the angle",
            **channel_handling(
                RESAMPLE_ANGULAR_180,
                # 0..18000 counts span the full 0..180 degree cycle. A reader
                # holding counts cannot convert them to angles without this, so
                # microscope_imageprocessing requires it for angular policies --
                # without it the declared circular averaging would silently
                # degrade to nearest-neighbour, which is the exact failure the
                # policy exists to prevent.
                period=ORIENTATION_MAX_COUNTS,
                reason=(
                    "axial slow-axis angle: 0 and 180 degrees are the same physical "
                    "axis, so the mean of 179 and 1 is 90 -- perpendicular to the truth"
                ),
            ),
        },
    }
    channel_label = {
        RETARDANCE_DIR: "Retardance (nm)",
        ORIENTATION_DIR: "Slow Axis Orientation (deg x100, axial)",
    }

    retardance_counts, ret_clipped = _to_uint16(result.retardance_nm, RETARDANCE_COUNTS_PER_NM)
    orientation_counts, _ = _to_uint16(
        np.rad2deg(result.orientation_rad), ORIENTATION_COUNTS_PER_DEGREE, ORIENTATION_MAX_COUNTS
    )
    if ret_clipped and log is not None:
        log.warning(
            "Retardance clipped at %.1f nm while writing %s; the stored map is "
            "saturated there and must not be read quantitatively.",
            _UINT16_MAX / RETARDANCE_COUNTS_PER_NM,
            filename,
        )

    for subdir, data in (
        (RETARDANCE_DIR, retardance_counts),
        # Orientation is axial data in [0, pi). Written raw here on purpose --
        # any downsampling or blending of these pixels must go through
        # sin(2*theta)/cos(2*theta), never an arithmetic mean, or 179 deg and
        # 1 deg average to 90 deg: perpendicular to the truth and entirely
        # plausible-looking.
        (ORIENTATION_DIR, orientation_counts),
    ):
        out_dir = output_path / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        _copy_tile_configuration(tile_config_source, out_dir)
        ome_writer(
            filename=str(out_dir / filename),
            pixel_size_um=pixel_size_um,
            data=data,
            channel_name=channel_label[subdir],
            map_annotations={**shared, **per_channel[subdir]},
        )
    if log is not None:
        log.debug("Reconstructed LC-PolScope tile %s", filename)


#: Largest commanded-vs-read-back difference, in waves, that counts as the same
#: setting. The calibration records what the crystals READ after each move, so
#: a reproduced state should agree to the adapter's own rounding.
LC_READBACK_TOLERANCE_WAVES = 0.001


def record_state_readback(ctx, hardware, ch_entry, logger) -> None:
    """On an LC-PolScope run, read the crystals back the first time each state is applied.

    No-op for every other modality, and for repeat visits to a state. Closes the
    loop the calibration opened: the YAML says what each state SHOULD be, this
    records what the device reports it IS, and the comparison goes to the log
    and (via the reconstruction) into every derived tile's metadata. The first
    call also logs where the LC settings came from -- QPSC's own calibration,
    or the presets the rig shipped with -- so an acquisition can never again be
    silently running a calibration nobody applied.

    Stored on the context as ``_lc_state_readback``:
    ``{channel_id: {property: {"commanded": str, "read": str, "agrees": bool}}}``.
    """
    from .registry import get_config

    if get_config(getattr(ctx, "modality", "") or "") is not LCPOLSCOPE_CONFIG:
        return
    readback = getattr(ctx, "_lc_state_readback", None)
    if readback is None:
        readback = {}
        ctx._lc_state_readback = readback
        modalities = (ctx.ppm_settings or {}).get("modalities", {}) or {}
        recon = (modalities.get("lcpolscope") or {}).get("reconstruction") or {}
        if recon.get("lc_calibrated_at") or recon.get("lc_calibration_file"):
            logger.info(
                "LC states come from QPSC's LC calibration of %s (extinction ratio %s, %s)",
                recon.get("lc_calibrated_at", "?"),
                recon.get("lc_extinction_ratio", "?"),
                recon.get("lc_calibration_file", "no file recorded"),
            )
        else:
            logger.warning(
                "LC states come from the YAML device_properties / Micro-Manager Channel "
                "presets; no QPSC LC calibration has been applied to this configuration "
                "(modalities.lcpolscope.reconstruction has no lc_calibrated_at)."
            )
    ch_id = str(ch_entry.get("id"))
    if ch_id in readback:
        return
    core = getattr(hardware, "core", None)
    entry: dict = {}
    for prop in ch_entry.get("device_properties", []) or []:
        if not isinstance(prop, dict):
            continue
        device, name, commanded = prop.get("device"), prop.get("property"), prop.get("value")
        if not (device and name) or core is None:
            continue
        try:
            read = str(core.get_property(str(device), str(name)))
        except Exception as exc:
            logger.warning("LC state %s: could not read back %s.%s: %s", ch_id, device, name, exc)
            continue
        try:
            agrees = abs(float(read) - float(commanded)) <= LC_READBACK_TOLERANCE_WAVES
        except (TypeError, ValueError):
            agrees = str(read).strip() == str(commanded).strip()
        entry[str(name)] = {"commanded": str(commanded), "read": read, "agrees": agrees}
        if agrees:
            logger.info("LC state %s: %s commanded %s, read back %s", ch_id, name, commanded, read)
        else:
            logger.warning(
                "LC state %s: %s commanded %s but the device reads %s -- the acquisition is "
                "NOT reproducing the configured state; reconstruction from it will be biased.",
                ch_id,
                name,
                commanded,
                read,
            )
    readback[ch_id] = entry


def lc_readback_summary(lc_state_readback) -> str:
    """One line per acquisition: ``State0 Retardance LC-A [in waves]=0.2520 ...``.

    Read values only; the commanded ones are in the YAML the metadata already
    points at, and a disagreement is flagged by ``polscope.lc_readback_agrees``.
    """
    parts = []
    for state, props in lc_state_readback.items():
        settings = " ".join(f"{name}={v['read']}" for name, v in props.items())
        parts.append(f"{state} {settings}".strip())
    return "; ".join(parts)


def resolve_state_order(channel_order, state_order):
    """Map acquisition order to the order ``reconstruct()`` expects.

    The five states are consumed positionally as (extinction, +S1, +S2, -S1,
    -S2). Which acquired state plays which role is a property of the software
    that ran the calibration, NOT of the microscope:

      * recOrder names them ext, I0, I45, I90, I135, which is already
        (ext, +S1, +S2, -S1, -S2) -- pairs (1,3) and (2,4).
      * OpenPolScope 3.20 acquires in the Oldenbourg order -- pairs (1,4) and
        (2,3) -- so State3 and State4 must be swapped before inversion.

    Getting this wrong rotates or mirrors the orientation map and raises
    nothing, so the order is configuration rather than an assumption baked
    into code. ``state_order`` is a list of channel ids in reconstruction
    order; omitting it means the acquisition order is already correct.

    Raises:
        ReconstructionRefused: if state_order is not a permutation of the
            acquired channels.
    """
    if not state_order:
        return list(channel_order)
    if sorted(state_order) != sorted(channel_order):
        raise ReconstructionRefused(
            f"reconstruction.state_order {list(state_order)} is not a permutation of "
            f"the acquired channels {list(channel_order)}. It must name every acquired "
            "state exactly once -- it reorders them, it does not select them."
        )
    return list(state_order)


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
    lc_state_readback=None,
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

    method = str(reconstruction_cfg.get("method", DEFAULT_RECONSTRUCTION_METHOD))
    if method not in RECONSTRUCTION_LABELS:
        raise ReconstructionRefused(
            f"reconstruction.method is {method!r}; expected one of {sorted(RECONSTRUCTION_LABELS)}. "
            "Refused up front rather than per tile: a typo here would otherwise fail inside the "
            "write pool on every tile of the slide."
        )
    if method == "openpolscope" and str(reconstruction_cfg.get("scheme", "5-State")) != "5-State":
        raise ReconstructionRefused(
            "reconstruction.method 'openpolscope' is 5-State only (OpenPolScope's 4-frame path has "
            "not been verified against real output); use method 'qlipp' for a 4-State scheme."
        )

    recon_order = resolve_state_order(channel_order, reconstruction_cfg.get("state_order"))
    ordered = [channel_images[cid] for cid in recon_order]
    # Backgrounds must be ordered exactly like the states: reconstruct() pairs
    # them positionally, so a mismatched order corrects each state with another
    # state's background -- which does not raise and does not look wrong.
    ordered_background = (
        [background_images[cid] for cid in recon_order] if background_images else None
    )
    # The first state's directory is the tile-layout reference; every channel
    # is imaged at the same positions, so any of them would do.
    tile_config_source = output_path / str(channel_order[0]) if output_path is not None else None
    write_pool.submit(
        _reconstruct_and_write,
        tile_config_source=tile_config_source,
        state_order=recon_order,
        state_images=ordered,
        reconstruction_cfg=reconstruction_cfg,
        background_images=ordered_background,
        lc_state_readback=lc_state_readback,
        output_path=output_path,
        filename=filename,
        pixel_size_um=pixel_size_um,
        ome_writer=ome_writer,
        log=log,
    )
    return True
