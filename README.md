# Microscope Command Server

Socket-based command server for remote microscope control and QuPath integration.

> **Part of the [QPSC (QuPath Scope Control)](https://github.com/uw-loci/qupath-extension-qpsc) system.**
> For complete installation and setup instructions, see the [QPSC Installation Guide](https://github.com/uw-loci/qupath-extension-qpsc/blob/main/documentation/INSTALLATION.md).

## Features

- **Socket Server**: TCP/IP server for remote microscope control
- **QuPath Integration**: Designed for QuPath annotation-driven acquisition
- **Client Library**: Python functions for stage control and acquisition
- **Acquisition Workflows**: Multi-tile, multi-modality acquisition orchestration
- **Real-time Monitoring**: Progress tracking and cancellation support
- **Multi-Channel Widefield IF / BF+IF**: Vendor-agnostic channel library driven
  by Micro-Manager ConfigGroup presets and device property writes

## Multi-Channel Acquisition (Widefield IF, BF+IF)

The BGACQUIRE command now supports a vendor-agnostic channel-based acquisition
branch used by widefield immunofluorescence (IF) and combined brightfield + IF
(BF+IF) workflows. When a command carries `--channels` and `--channel-exposures`
in place of `--angles` / `--exposures`, the single-image tile loop iterates the
resolved channel plan once per tile position, writing one TIFF per channel per
tile. The Python server does not know anything about specific illuminators; it
drives the hardware entirely through `core.setConfig(group, preset)` and
`core.setProperty(device, property, value)`, so the same code path serves
CoolLED, Lumencor, DLED, Colibri, and custom builds without modification.

See the cross-repo overview at
`QPSC/docs/multichannel-if-overview.md` for the full pipeline description,
YAML schema, and end-to-end BF+IF example. This section covers only the
Python server's slice of the pipeline.

### New BGACQUIRE flags

The BGACQUIRE (and ACQUIRE) acquisition message parser accepts optional flags
on top of the existing angle-based flags:

**For multi-channel acquisition (widefield IF, BF+IF):**
- `--channels "(id1,id2,...)"` -- ordered list of channel ids to acquire at
  every tile position. Ids must match entries in the modality's channel
  library declared in the microscope YAML.
- `--channel-exposures "(exp1,exp2,...)"` -- per-channel exposures in
  milliseconds. Must be the same length and order as `--channels`; missing
  or non-positive entries fall back to the channel library's default
  `exposure_ms`.

**For background collection:**
- `--profile <key>` -- acquisition profile key to apply for background
  collection (e.g., `profile=Brightfield_10x` or `profile=IF_40x`). If
  omitted, the profile is resolved automatically from modality and objective.
  The profile determines which illumination intensity and device properties are
  applied during background collection.
- `--channels "(id1,id2,...)"` -- when used with BGACQUIRE, enables
  per-channel background collection: the server collects one background image
  per channel instead of per-angle, saving as `{channel_id}.tif` under the
  output folder. Useful for widefield illumination systems where per-channel
  backgrounds (with channel-specific optics and illumination) better represent
  the correction needed during acquisition.

When `--channels` is present for acquisition, the server takes the channel
acquisition branch in `acquisition/workflow.py`. `--channels` is mutually
exclusive with `--angles`. If both are supplied (for example, a stale angle
field from an older client), the server logs a warning and clears the angles
so the channel path is the single source of truth.

### How it works

Three helpers in `microscope_command_server/acquisition/workflow.py` implement
the channel path:

- `resolve_channel_plan(ppm_settings, scan_type, channel_ids, channel_exposures)`
  -- resolves the profile (`acquisition_profiles.<scan_type>`), looks up its
  `modality`, then reads `modalities.<modality>.channels` from the YAML and
  filters / reorders to the requested ids. For each channel it merges in the
  profile's `channel_overrides.<id>.device_properties` and returns an ordered
  list of channel plan dicts containing `id`, `display_name`, `exposure_ms`,
  `mm_setup_presets`, `device_properties`, and optional `settle_ms`.
- `_merge_device_property_overrides(library_props, override_props)` -- private
  helper mirroring the Java-side merge rule
  (`MicroscopeConfigManager.mergeDevicePropertyOverrides`) exactly: match by
  `(device, property)` tuple, replace the value in place when matched, append
  to the end of the list otherwise. This lets a profile tune one property on
  one channel with a single YAML line without redeclaring the whole channel.
- `apply_channel_hardware_state(hardware, channel_plan_entry, logger_)` --
  applies `mm_setup_presets` via `core.setConfig(group, preset)` followed by
  `core.waitForConfig`, then applies `device_properties` via
  `core.setProperty(device, property, value)` and calls `core.waitForDevice`
  on every touched device. This is the critical settle pass that stops
  back-to-back channel transitions from racing the camera snap on serial LED
  controllers. An optional `settle_ms` field on the channel entry adds a
  dumb-sleep fallback for hardware whose `isBusy()` reports complete too
  early (some filter turrets, reflector wheels, serial LED controllers).

Inside the "Single image acquisition: no rotation angles" block of the tile
loop, the server checks for a non-empty channel plan. If present, it iterates
the plan for the current tile position: apply channel state, set the channel
exposure, snap, (optionally) flat-field correct, saturation-check, and write
the per-channel TIFF. The tile loop then `continue`s past the default
single-snap path. If no channel plan is resolved, the tile loop falls back to
the existing single-snap behavior -- see "Backward compatibility" below.

### File layout on disk

Per-tile the channel branch writes one TIFF per channel into a per-channel
subdirectory under the existing annotation output folder:

```
{projectsFolder}/{sample}/{scan_type}/{annotation}/
    {channel_id_1}/tile_0_0.tif
    {channel_id_1}/tile_0_1.tif
    {channel_id_2}/tile_0_0.tif
    {channel_id_2}/tile_0_1.tif
    ...
    TileConfiguration.txt
```

This mirrors the PPM per-angle layout exactly -- channel ids double as
subdirectory names. The stitcher
(`qupath-extension-tiles-to-pyramid`) can then isolate each channel at
stitch time by pointing its existing per-axis stitching helper at each
channel subdirectory, without any channel-aware logic in the stitcher
itself.

### Per-channel background correction (opt-in)

When the tile loop loads background images for an acquisition, the channel
branch additionally looks for per-channel flat-field images under the
background directory:

```
{background_dir}/{channel_id}/background.tif
```

Any channel whose file is present is flat-field corrected via
`BackgroundCorrectionUtils.apply_flat_field_correction` using the configured
method (`divide` by default). Channels whose file is missing are skipped
silently -- they are acquired without correction. The loader also accepts
the flat alternates `{background_dir}/{channel_id}.tif` and
`{channel_id}.tiff` for convenience.

This is the channel-axis analog of the PPM per-angle background path: the
key is the channel id rather than the rotation angle, but the correction
call and the missing-file behavior are the same.

### Backward compatibility

BGACQUIRE commands that do not pass `--channels` fall through unchanged:

- Commands with `--angles` take the multi-angle branch (PPM and similar).
- Commands with neither angles nor channels take the default single-snap
  branch (brightfield, single-snap fluorescence, laser scanning).

Modalities whose YAML has no `channels:` library never enter the channel
branch, so existing profiles keep working without any YAML edits.

## Acquisition Loop Ordering (--inner-axis)

The BGACQUIRE (and ACQUIRE) acquisition message parser accepts an optional
`--inner-axis` flag to control the nesting order of hardware sweeps. This
affects performance by changing the frequency of expensive hardware transitions
(rotation moves for PPM, filter-cube changes for widefield).

### Flag syntax

```
--inner-axis <value>
```

Allowed values:
- `z` -- Z-position is the innermost loop
- `channel` -- Channel is the innermost loop
- `angle` -- Rotation angle is the innermost loop

### PPM (Multi-Angle) Acquisition

Default behavior (`--inner-axis angle` or omitted): **z-outer / angle-inner**
- Outer loop: z-planes
- Inner loop: angles
- **Effect**: At each z-plane, every angle is re-acquired before z advances.
  Tight per-z registration across angles, but a 5-z x 4-angle field pays
  20 rotation-stage moves per tile. This is the historical PPM ordering.

Alternative mode (`--inner-axis z`): **angle-outer / z-inner**
- Outer loop: angles
- Inner loop: z-planes
- **Effect**: Each angle sweeps its full z-stack before rotating to the next
  angle. Fewer rotation moves (one per angle per tile instead of one per
  angle-z pair). The per-angle WB / JAI-calibration / exposure block also
  hoists outside the inner z loop. Faster for thicker tissue z-stacks
  (especially at 40x).

### Widefield (Multi-Channel) Acquisition

Default behavior (`--inner-axis z` or omitted): **channel-outer / z-inner**
- Outer loop: channels
- Inner loop: z-planes
- **Effect**: Each channel sweeps its full z-stack before switching to the
  next channel. Fewer filter-cube changes (one per channel per tile instead
  of one per channel-z pair). Optimized for fixed slides where focus drift
  is not a concern.

Alternative mode (`--inner-axis channel`): **z-outer / channel-inner**
- Outer loop: z-planes
- Inner loop: channels
- **Effect**: At each z-plane, every channel is re-acquired before z advances.
  Tighter per-channel z registration -- right for live-cell or drifting
  samples where minutes between channel acquisitions could cause axial
  decorrelation. Costs channels x z_planes filter switches per tile.

### Backward compatibility

Omitting `--inner-axis` preserves byte-identical behavior to pre-flag
acquisitions:
- PPM defaults to z-outer / angle-inner (the historical ordering)
- Widefield defaults to channel-outer / z-inner (fewer filter changes)

## Z-Stack Acquisition (--z-stack, --z-projection)

The BGACQUIRE (and ACQUIRE) acquisition message parser accepts optional flags
to control Z-stack (axial) image acquisition and output format.

### Flag syntax

**Z-stack enable and range:**
```
--z-stack true
--z-start <micrometers>
--z-end <micrometers>
--z-step <micrometers>
```

**Projection and output format:**
```
--z-projection <method>
```

### Z-stack parameters

- `--z-stack true|false` -- Enable Z-stack acquisition (acquire multiple planes
  at different focal depths). Default: `false`. When enabled, requires
  `--z-start`, `--z-end`, and `--z-step`.
- `--z-start <um>` -- Starting Z position (micrometers). Often a negative value
  (e.g., `-5.0`) to acquire above the focal plane.
- `--z-end <um>` -- Ending Z position (micrometers). Often a positive value
  (e.g., `+5.0`) to acquire below the focal plane.
- `--z-step <um>` -- Distance between successive planes (micrometers). Typical
  values: `0.5` to `2.0` depending on sample refractive index and objective.

The server computes the number of planes as `ceil(|z_end - z_start| / z_step)`.

### Z-projection methods

- `max` -- Maximum intensity projection (default). Each tile output is a single
  2D image (the brightest pixel at each xy position across the Z-stack).
- `mean` -- Mean intensity projection. Single 2D output per tile.
- `edf` -- Extended Depth of Field (EDF) projection. Computes a single 2D image
  by selecting pixels from the Z-stack according to a sharpness metric (see
  "EDF tuning parameters" below). Each tile output is a single 2D image.
- `none` -- Preserve individual Z-planes without projection. Output structure
  changes to a per-plane layout, enabling 5D stitching (x, y, z, channel/angle,
  time). See "File layout on disk" below.

### EDF tuning parameters

When using `--z-projection edf`, three optional parameters control the sharpness
metric and filtering behavior:

- `--edf-metric <name>` -- Per-pixel sharpness map (default: `tenengrad`).
  Valid values are exactly `tenengrad`, `modified_laplacian` and `variance` --
  the names in `microscope_imageprocessing.focus.sharpness_maps`. Anything else
  raises rather than silently substituting, so a typo fails before the
  acquisition starts. `tenengrad` matches the autofocus metric of the same
  name; `modified_laplacian` peaks more sharply in Z but is noisier;
  `variance` is the most forgiving of noise. Only meaningful with `edf`.
- `--edf-window <pixels>` -- Averaging window for the sharpness map, in pixels
  (integer, >= 1; default 9). Raw per-pixel sharpness is too noisy to choose a
  plane from, so it is averaged over this window first -- the setting that
  matters most. Raise it if the fused output looks blocky or speckled; lower it
  if boundaries between in-focus regions look smeared. Scales with pixel size.
  Only meaningful with `edf`.
- `--edf-index-smooth <pixels>` -- Median-filter size applied to the map of
  which plane each pixel selected, in pixels (integer, >= 0; default 5; 0
  disables). Real focal surfaces are smooth, so this removes pixels that chose
  an odd plane for no physical reason. Raise it for a tilted but flat sample;
  lower it where focus genuinely steps (a fold, a torn section), because a
  large median bridges the step and picks a plane sharp on neither side.
  Only meaningful with `edf`.

The defaults are reasoned starting points, not measured optima.

**Example:**
```
--z-projection edf --edf-metric tenengrad --edf-window 9 --edf-index-smooth 5
```

### File layout on disk

**When `--z-projection max|mean` (default projection behavior):**
Single 2D image per tile per channel/angle:
```
{projectsFolder}/{sample}/{scan_type}/{annotation}/
    {angle_or_channel}/tile_000.tif
    {angle_or_channel}/tile_001.tif
    ...
```

**When `--z-projection none` (preserve individual planes):**
Each Z-plane is written to a per-Z-plane subdirectory. For single-timepoint
acquisitions, the layout is `{group}/z{zz}/{filename}`:
```
{projectsFolder}/{sample}/{scan_type}/{annotation}/
    {angle_or_channel}/z000/tile_000.tif
    {angle_or_channel}/z000/tile_001.tif
    {angle_or_channel}/z001/tile_000.tif
    {angle_or_channel}/z001/tile_001.tif
    ...
```

For time-lapse acquisitions (with `--timepoints > 1`), a `t{tt}/` segment is
added to track each timepoint, enabling reassembly into a true 5D mosaic:
```
{projectsFolder}/{sample}/{scan_type}/{annotation}/
    {angle_or_channel}/t000/z000/tile_000.tif
    {angle_or_channel}/t000/z001/tile_000.tif
    ...
    {angle_or_channel}/t001/z000/tile_000.tif
    {angle_or_channel}/t001/z001/tile_000.tif
    ...
```

The stitcher (`qupath-extension-tiles-to-pyramid`) can then assemble all planes
(and timepoints) into a unified 5D (x, y, z, channel/angle, time) mosaic by
reading these directory structures.

### Use case: 5D confocal / light-sheet stacks

Set `--z-projection none` when acquiring thick (volumetric) samples that require
a full Z-stack for analysis downstream (e.g., volume rendering, 3D
reconstruction, confocal microscopy). Single-timepoint 3D stacks produce
`{group}/z{zz}/` layouts byte-identical to the existing `--save-raw`
behavior. Time-lapse 3D stacks add `t{tt}/` segments for temporal registration
and 5D reassembly.

### Backward compatibility

Omitting `--z-stack` or setting it to `false` disables Z-stack acquisition
entirely. Omitting `--z-projection` defaults to `max` (maximum intensity
projection), preserving the original single-image-per-tile output.

## LC-PolScope Acquisition (Automatic Birefringence Reconstruction)

LC-PolScope acquisitions automatically reconstruct birefringence (retardance and
orientation) maps from polarization states when properly configured. The server
queues reconstruction for each tile after all state images are acquired.

### Configuration Requirements

Reconstruction requires three settings in the microscope YAML under
`modalities.lcpolscope.reconstruction`:

```yaml
modalities:
  lcpolscope:
    reconstruction:
      swing_waves: 0.03          # Calibration swing amplitude (required)
      wavelength_nm: 546         # Light wavelength in nanometers (required)
      scheme: "5-State"          # Polarization scheme; default "5-State"
      state_order:               # Optional: state reordering before reconstruction
        - State0
        - State1
        - State2
        - State4
        - State3
```

- **`swing_waves`**: The calibration swing amplitude used during system
  calibration. A fixed value per microscope.
- **`wavelength_nm`**: Illumination wavelength in nanometers. Must match the
  wavelength used during acquisition and calibration.
- **`scheme`**: Reconstruction scheme (default `"5-State"`). The only accepted
  values are `"5-State"` and `"4-State"`; the scheme is fixed by how the system
  was calibrated and is not a free choice at acquisition time.
- **`state_order`** (optional): A permutation of the acquired state ids that
  reorders them before inversion. Used when the acquisition order differs from
  the calibration order (e.g., when using OpenPolScope, which acquires in the
  Oldenbourg order with pairs (1,4) and (2,3), requiring State3 and State4 to
  swap). Omitting this field means states are consumed in acquisition order,
  which is correct for recOrder and similar tools. Must list every acquired
  state exactly once; specifying a partial list or unknown state ids will
  raise an error and skip reconstruction.

**If this block is missing or incomplete**, the server logs a warning and skips
reconstruction; raw state images are still saved and can be reconstructed
offline.

### State Image Acquisition

Acquire 4 or 5 polarization states (depending on your scheme) as separate
channels using the standard multi-channel acquisition flags:

```
--channels "(State0,State1,State2,State3,State4)"
--channel-exposures "(100,100,100,100,100)"
```

The state ids must match exactly (case-sensitive); Micro-Manager presets should
map to these state names in your configuration.

**One critical invariant that fails silently:** The Stokes inversion treats
the state intensities as samples of a single radiometric scale, so a per-state
difference biases the result.

- **All states must share one exposure and gain.** Three layers cooperate to
  hold this: the QuPath extension equalises the exposures before sending them,
  every channel in `config_LCPolScope.yml` carries the same `exposure_ms`,
  and the LC-PolScope acquisition profiles deliberately carry no
  `channel_overrides`. Do not add per-channel exposure tuning to any of them.

**State order is configurable.** The `state_order` parameter in the YAML
reorders states before inversion to match the calibration scheme. When omitted,
states are consumed in acquisition order (correct for recOrder). A wrong
permutation rotates or mirrors the orientation map without raising an error.
If you did not run the calibration yourself, identify it from the data with
`polscope-scheme-check` (shipped with `polscope-library`) before trusting
any orientation output. The correct order is a property of the software that
ran the calibration -- not of the microscope -- so it can change without any
hardware changing, and must be re-checked after a software switch.

### Automatic Reconstruction Behavior

After all state images for a tile are acquired and saved:
1. The server validates the state set (4 or 5 images, in calibration order)
2. If valid, reconstruction is queued in the background write pool
3. Retardance and orientation maps are computed and written as OME-TIFF files
4. A refusal to reconstruct does not abort the acquisition — raw states remain
   saved and can be reconstructed offline

### Constraints and Validation

**Background Correction**: QLIPP applies the background in Stokes space, using
background *intensities* passed to the reconstruction library. Flat-field
division of the raw state tiles is a different operation and biases retardance
and orientation with no visible symptom -- the images still look right.

The server routes this correctly on its own: for LC-PolScope the acquisition
loop skips the flat-field divide, so raw states reach the reconstruction, which
applies the background itself. No special flag selects this behaviour; it
follows from the modality.

You do still have to ask for backgrounds. Per-channel background images are
only loaded when the acquisition is sent with `--bg-correction true` -- placing
files in the background folder without that flag has no effect.

**A partial background set is refused.** The correction is not a per-state
subtraction. The background states are inverted to Stokes parameters, an
attenuating-depolarizing-retarder Mueller matrix is estimated from them, and its
inverse is applied to the sample's Stokes vector. All states feed that one
matrix, so a missing state corrupts the entire correction rather than degrading
a single channel. Supply a background for every state, or none. Reconstructing with
no background at all is valid, just lower quality.

Refusals do not abort the acquisition. Raw state images are still saved, so the
run can be reconstructed offline afterwards, and a refusal is logged once per
acquisition rather than once per tile.

### Output File Layout

Derived retardance and orientation maps are written to sibling directories
alongside the state images:

```
{projectsFolder}/{sample}/{scan_type}/{annotation}/
    State0/tile_0_0.tif
    State1/tile_0_0.tif
    State2/tile_0_0.tif
    State3/tile_0_0.tif
    State4/tile_0_0.tif
    retardance/tile_0_0.tif          # Retardance (uint16, hundredths of nm)
    orientation/tile_0_0.tif         # Orientation (uint16, hundredths of degrees, [0, 18000) = [0, 180) degrees)
    TileConfiguration.txt
```

Each is a tile directory in its own right, with a `TileConfiguration.txt`
copied from the state directories, so the stitcher
(`qupath-extension-tiles-to-pyramid`) stitches each into its own mosaic.

Each derived tile file carries OME-XML metadata recording the reconstruction
parameters (wavelength, swing waves, scheme, state order), whether background
correction was applied, and critical handling rules for downstream processing.
This metadata travels with the file, so a reader can identify both the
calibration used to produce it and the processing constraints that must be
respected.

**Orientation is axial data and must not be resampled as an ordinary scalar.**
0 and pi are the same physical orientation, so the mean of 179 degrees and 1
degree is 90 degrees -- perpendicular to the truth, and entirely
plausible-looking. Anything that averages these pixels (blending in a stitch
seam, pyramid downsampling) has to go through sin(2*theta)/cos(2*theta), or
encode the angle as hue, rather than averaging the angle directly. Retardance
is an ordinary scalar and has no such constraint. The metadata in the OME-XML
flags which channel is axial and how it may be resampled.

### Backward Compatibility

Omitting the `modalities.lcpolscope.reconstruction` block disables
reconstruction. Raw state images are still acquired and saved normally, and
reconstruction can be performed offline using the `polscope-library` directly.

## LC-PolScope Calibration (LCCALIB)

The `LCCALIB` command calibrates the liquid crystals on an LC-PolScope system,
finding the extinction point and swing states. It produces a calibration palette
and metadata file that guides all subsequent LC-PolScope acquisitions.

### When to calibrate

- **First setup**: After installing or aligning new liquid crystals
- **Configuration change**: When wavelength or scheme changes
- **Whenever the extinction ratio drops**: it is the health metric, so a
  falling value is the signal to recalibrate. How often that happens on this
  rig is not yet known.

### Command syntax

```
LCCALIB --yaml config.yml --output /path/to/output [--modality lcpolscope] \
        [--swing 0.03] [--scheme "5-State"] [--wavelength 546.0] \
        [--black-level 100.0] [--strategy single_pass] ENDOFSTR
```

### Parameters

- **`--yaml config.yml`** (required) -- Microscope configuration file. The server
  reads `modalities.<modality>.reconstruction` for default swing, wavelength, scheme,
  and settle time if not overridden on the command line.
- **`--output /path/to/output`** (required) -- Folder where the calibration palette
  and metadata file are written. The server creates this folder if it doesn't exist.
- **`--modality lcpolscope`** (optional) -- Which modality to calibrate. Default:
  `lcpolscope`. Only relevant if the YAML defines multiple LC-PolScope configurations.
- **`--swing 0.03`** (optional) -- Calibration swing amplitude in waves. If omitted,
  taken from `modalities.<modality>.reconstruction.swing_waves` in the YAML.
- **`--scheme "5-State"`** (optional) -- Polarization scheme to calibrate. Accepted
  values: `"5-State"` or `"4-State"`. If omitted, taken from the YAML; defaults to
  `"5-State"`. **The scheme is fixed by your hardware and calibration procedure**;
  it is not a free choice at runtime.
- **`--wavelength 546.0`** (optional) -- Illumination wavelength in nanometers.
  If omitted, taken from the YAML; defaults to 546.0 nm.
- **`--black-level 100.0`** (optional) -- Dark-frame intensity used to correct the
  raw measurements. If omitted, the calibration falls through its black-level chain, in this order:
  (1) an explicit value here or in the YAML, (2) an averaged dark frame if lamp
  control is configured, (3) zero, with a warning. The value matters: a 50-count
  error moves the extinction ratio by roughly 10%.
- **`--strategy single_pass`** (optional) -- Search strategy. `single_pass`
  (default) makes one pass per state; `iterative` repeats, re-centring on the
  crystals, until the residual is small enough. The second costs roughly three
  times the exposures, so it is a trade rather than an upgrade.

### Response sequence

The command sends multiple responses as it progresses:

1. **`STARTED:<output_folder>`** -- Calibration has begun; acknowledges the output
   path the operator requested.

2. **`PROGRESS:<current>:<total>:<message>`** -- Updates during the search, in
   the same four-field shape PPMBIREF uses.

3. **`SUCCESS:<json>`** -- Calibration succeeded. The JSON payload always contains:
   ```json
   {
     "success": true,
     "scheme": "5-State",
     "swing_waves": 0.03,
     "wavelength_nm": 546.0,
     "lc_control_mode": "MM-Retardance",
     "strategy": "single_pass",
     "black_level": 100.0,
     "black_level_source": "configured|measured|assumed",
     "extinction_ratio": 150.5,
     "assessment": "good|acceptable|poor|unmeasurable",
     "palette": {
       "State0": [0.2500, 0.5000],
       "State1": [0.2200, 0.5000],
       "...": "[LC-A, LC-B] in waves, one entry per state"
     },
     "state_intensities": {...},
     "exposures": 42,
     "elapsed_s": 15.3,
     "warnings": [],
     "output_folder": "/path/to/output",
     "metadata_path": "/path/to/output/lc_calibration_YYYYMMDD_HHMMSS.json"
   }
   ```

   **Key fields:**
   - `success: true` always indicates a palette was produced
   - `extinction_ratio` is the quality metric; higher is better (>100 is good)
   - `assessment` is a human-readable quality summary
   - `palette` contains the retardance values (in waves) for each state and LC axis
   - `warnings` lists any issues that don't prevent use (e.g., marginal extinction)
   - `metadata_path` points to a JSON file with the full trace (per-exposure details)

4. **`FAILED:<reason>`** -- Calibration could not complete. The reason string
   describes the error (e.g., "could not reach the liquid crystals", "polscope_library
   is not installed").

### Extinction ratio interpretation

The extinction ratio is the primary quality metric, computed from the measured
intensities at the calibration swing. Higher is better.

- **100 or above**: Good. These are recOrder's bands; this rig reached 267.
- **80-100**: Acceptable. Usable, but check alignment and optical cleanliness.
- **Below 80**: Poor. Recorded with a warning rather than rejected.
- **Unmeasurable**: The search failed to find a valid extinction point. The hardware
  or configuration may need attention.

A poor result is still returned (not rejected), so the operator can inspect the data
and decide whether to retry or investigate the hardware.

### Using the calibration result

The generated palette should be stored and then referenced in your microscope YAML
under `modalities.lcpolscope.reconstruction`:

```yaml
modalities:
  lcpolscope:
    reconstruction:
      swing_waves: 0.03
      wavelength_nm: 546.0
      scheme: "5-State"
      palette:
        State0: [0.0, 0.0]
        State1: [0.25, 0.0]
        ...
```

Or the palette can be burned into the microscope's persistent configuration so
acquisitions always use the correct calibration.

### Black-level options

The `--black-level` parameter controls the dark-frame correction, which improves
extinction ratio by accounting for detector noise and dark current:

- **Omitted or 0**: Falls through the library's chain: measure if a lamp is
  configured, use the YAML value if present, otherwise assume zero.
- **Positive value**: Use the supplied dark level directly.

A measured dark frame (via a lamp) is preferred when available, as it captures
the actual hardware behavior on the day.

### LC control mode

The server currently supports **`MM-Retardance`** mode: commands the liquid
crystals in retardance (wave) units, and the Micro-Manager device adapter
converts to voltage. Voltage mode (`MM-Voltage`) is not yet wired up and is
refused with an error message directing the operator to use retardance mode.

## PPM Acquisition Options (--ppm-high-bit-depth)

The BGACQUIRE (and ACQUIRE) acquisition message parser accepts optional flags
specific to PPM (polarized-light microscopy) acquisitions.

### High-bit-depth angle capture

```
--ppm-high-bit-depth true|false
```

**Parameters:**

- `--ppm-high-bit-depth true|false` -- Enable high-bit-depth capture for PPM
  angle frames (opt-in). Default: `false`.
  - When `true`: PPM angle frames are captured at the camera's native
    high-bit PixelFormat (e.g., 12-bit on JAI cameras) instead of the standard
    8-bit. This provides higher precision inputs to the birefringence
    calculation, potentially improving measurement accuracy for low-intensity
    samples.
  - When `false` or omitted: Uses the standard 8-bit capture path; behavior is
    byte-identical to prior releases.

**Constraints:**

- This flag is only honored on cameras that implement
  `set_high_bit_mode()` (currently the JAI camera interface).
- If specified on unsupported camera hardware, the server logs a warning and
  silently falls back to 8-bit capture.
- Scoped to angle acquisition only: autofocus and non-PPM imaging always use
  standard 8-bit capture regardless of this flag.

**Use case:** When acquiring PPM images from weakly birefringent samples or at
high zoom factors, enabling high-bit-depth capture can improve the signal-to-noise
ratio of birefringence measurements by working with full camera precision
instead of quantized 8-bit data.

### Backward compatibility

Omitting `--ppm-high-bit-depth` or setting it to `false` preserves the standard
8-bit capture behavior, ensuring acquisitions are unchanged from prior releases.

## Finding tissue before autofocus (FINDTISS)

`FINDTISS` moves the stage in **XY** until the camera is looking at tissue. It never
changes Z, exposure, or any camera setting -- the caller orders the pair itself:

```
MOVE -> FINDTISS -> STRMAFZ
```

### Why it exists

A multi-slide batch predicts each slide's first alignment landmark from the base
transform. Measured over 8 slides (2026-08-24) that prediction lands a median **613 um**
from its target, worst case **1507 um**, which frequently puts the camera over blank
glass. A focus scan there commits to coverslip contrast or walks its whole attempt budget
and gives up.

The second landmark, corrected by the first's translation, lands within **26 um**. So the
base transform's error is very nearly a constant per-slide offset, and **only the first
landmark of a slide needs this** -- which is the only place QPSC sends it.

Alignment reach is *not* what is being fixed: SIFT matched at 1507 um with 796 inliers and
0.999 confidence. The search only has to put tissue -- any tissue -- in view so the focus
scan has something real to find.

### Parameters

```
FINDTISS --yaml config_PPM.yml [--objective <id>] [--dir <dx>,<dy>] \
         [--step <um>] [--max-attempts <n>] ENDOFSTR
```

- `--dir` is a stage-space hint toward where tissue is believed to be; only its bearing is
  used. QPSC computes it as the vector from the predicted position toward the centre of the
  tile grid. Malformed input is ignored with a warning rather than failing the command --
  the search still works unhinted.
- `--step` defaults to one camera FOV diagonal, the coarsest step that cannot skip ground.
- `--max-attempts` counts the starting position. Default 7, capped at 25.

### Search pattern

`server/tissue_search.py` is pure geometry and unit-tested (`tests/test_tissue_search.py`).
The first position is always where the caller already is. After that, positions lie on
rings at whole multiples of `--step`: with a hint, three bearings per ring (down the hint,
then +/-45 deg); without one, the four compass points then the diagonals. Reach is
therefore `step * ((max_attempts - 1) // bearings_per_ring)`, so an attempt budget converts
directly into a distance -- which is how the default was sized against the measurement
above.

Tissue is decided by the **same strategy validity check the acquisition path uses**
(`texture_and_area` and friends, thresholds from `autofocus_<scope>.yml`), so there is no
new metric to calibrate and no second definition of "has content" to drift.

### Exposure is deliberately not adjusted

The caller has just put the modality into its alignment reference state -- for PPM, the
calibrated uncrossed angle and exposure -- and SIFT is about to match against that state.
A brightness-chasing loop here would silently change what the next step depends on. This
differs from the acquisition path's first-tile tissue search, which *does* double exposure:
that one owns the camera state, this one borrows it.

### Responses

- `FOUND:<x>:<y>:<attempt>:<of>` -- the stage is standing at `(x, y)`.
- `NOTFOUND:<x>:<y>:<of>` -- everything searched was background, and **the stage has been
  put back where the search started**. A search that found nothing has no reason to prefer
  its last guess over its first, and leaving the stage elsewhere would silently invalidate
  the caller's own prediction.
- `FAILED:<reason>` -- could not run at all. Nothing moved.

## Autofocus / Streaming Focus (--safe-z, --approach-max, --tissue-gate)

The `STRMAFZ` command supports multiple autofocus strategies. The default
edge-retry walk is always available; an alternative approach-from-safe-Z strategy
can be enabled via parameters measured during a separate validation run.

### Two autofocus strategies

**Edge-Retry Walk (default):**
Starts from wherever the stage currently is and scans a window; if the peak
appears to lie past an edge of that window, it shifts the window in that
direction and scans again, up to `--max-attempts` times. Needs no operator
measurement and works from any starting position, but each continuation is an
inference that moving further -- possibly toward the sample -- is correct.

**Approach-from-Safe-Z (new):**
A bounded single-pass scan that retracts to a known-safe position first, then
approaches the sample once. The operator measures the safe Z (clear of sample)
and the maximum distance to travel during a separate validation run. This
strategy:
- Guarantees a clear retraction on failure
- Bounds travel distance (no open-ended walk)
- Optionally validates tissue before committing to a focus peak

### Flag syntax

```
--safe-z <micrometers>         # Retraction Z (where the sample is definitely not)
--approach-max <micrometers>   # Distance to scan from safe Z toward the sample
--tissue-gate 1                # Require tissue validation before committing (omit to disable)
```

### Approach-from-safe-Z parameters

- **`--safe-z <um>`** -- the RETRACTED Z, where the objective is clearly clear of
  the sample for the insert and objective in use. Not the coverslip and not the
  sample plane -- those are what it must stay away from. The stage retracts here
  before scanning, and returns here if the scan fails. **Required to activate
  approach mode; no safe default, because a guessed retraction could be on the
  wrong side of the sample.**
  
- **`--approach-max <um>`** -- Maximum distance to travel from safe Z toward
  the sample (positive or negative, depending on Z direction). The actual scan
  span is clamped by stage limits. **Required to activate approach mode; no
  safe default.** Typically measured from a validation run as the distance from
  safe Z to the auto-detected focus peak, with a safety margin.
  
- **`--tissue-gate 1`** -- If set, each focus peak candidate is
  tested for tissue presence before committing. Peaks that fail the tissue
  check (e.g., coverslip reflections producing metric peaks but no tissue
  texture) are rejected and the scan continues to the next peak. Omitted means
  no tissue check: commit to the first prominent peak. QPSC sets this when its
  validation run found surfaces BEFORE focus, which is exactly when committing
  to the first peak would land on glass.

**Activation rule:** Approach-from-safe-Z is enabled ONLY when BOTH
`--safe-z` AND `--approach-max` are provided. If only one is given, the server
logs a warning and falls back to the edge-retry walk.

### Tissue gate validation

When `--tissue-gate true`, each focus peak undergoes a texture-based tissue
presence check before commitment. The check uses the same validity criteria
applied during acquisition (via `resolve_validity_check("texture_and_area")`),
configured by the microscope YAML:

```yaml
autofocus:
  texture_threshold: 0.010        # Texture variance threshold (default 0.010)
  tissue_area_threshold: 0.200    # Tissue coverage % (default 20%)
  rgb_brightness_threshold: 240.0 # Max RGB value for coverslip rejection (default 240)
```

A peak passes the gate if the image at that Z shows sufficient texture (cell
structure, not uniform glass) and has tissue in the required fraction of pixels.
Coverslip reflections are typically rejected because they produce uniform,
bright pixels rather than textured features.

### Example: Validating and using approach-from-safe-Z

**Step 1: Validation run (operator measurement)**
```
STREAMING_FOCUS --yaml config.yml --metric tenengrad --range 100
# Server scans across the sample, prints the detected focus peak Z.
# Operator measures the safe Z (e.g., at the coverslip) = 2.00 mm.
# Difference: peak_Z - safe_Z = 0.042 mm -> measure approach_max as 0.050 mm (with margin).
```

**Step 2: Subsequent acquisitions**
```
STREAMING_FOCUS --yaml config.yml --metric tenengrad \
  --safe-z 2.00 --approach-max 0.050 --tissue-gate true
# Server retracts to 2.00 mm, scans 0.050 mm toward the sample,
# finds the first prominent peak, validates tissue, and commits.
# On failure, stage returns to 2.00 mm (safe).
```

### Backward compatibility

Omitting `--safe-z` and `--approach-max` (or providing only one) preserves the
default edge-retry walk, ensuring autofocus behavior is unchanged when these
parameters are not supplied. The `--tissue-gate` parameter has no effect when
approach mode is not active.

## Autofocus Profiling (--z-start, --z-end)

The `STRMAFZ` command supports an explicit profiling mode to acquire the focus
metric over a named interval without committing to any focus result. This is
used for validation workflows that need the raw metric profile over a specific
region to analyze focus behavior and measure safe-Z distances.

### Profiling mode parameters

```
--z-start <micrometers>   # Start Z position for the profiling scan
--z-end <micrometers>     # End Z position for the profiling scan
```

**Activation rule:** Profiling mode is enabled ONLY when BOTH `--z-start` AND
`--z-end` are provided. If only one is given, the server logs a warning and
ignores both. Profiling mode is **mutually exclusive** with approach-from-safe-Z
mode (`--safe-z` / `--approach-max`).

### Profiling behavior

- **Single pass:** The stage scans exactly once from `z_start` to `z_end`, in
  whichever Z direction that requires.
- **No focus decision:** The metric profile is recorded and returned, but no peak
  is committed. The focus is not updated.
- **Return to start:** After the scan, the stage returns to `z_start`, so the
  acquisition leaves the Z position unchanged from before the profiling run.
- **Direction-agnostic:** Unlike the edge-retry walk (which always increments Z),
  profiling supports scans in either direction: `z_end` can be greater than or
  less than `z_start`.

### Use case: Focus approach validation

Profiling is used during Focus Approach Validation to measure safe-Z distances
and validate that the sample is reachable:

1. **Step 1: Profiling run** - Scan across the expected sample region and capture
   the metric profile.
2. **Step 2: Analyze profile** - Identify the focus peak, the safe-Z (clear of
   sample), and the approach distance.
3. **Step 3: Subsequent acquisitions** - Use the measured values with
   `--safe-z` and `--approach-max` for bounded, repeatable focusing.

### Example: Profiling scan

```
STREAMING_FOCUS --yaml config.yml --metric tenengrad \
  --z-start -0.500 --z-end 0.500
# Server scans from -0.500 mm to +0.500 mm, records the metric profile,
# and returns the stage to -0.500 mm without committing to any focus.
```

### Backward compatibility

Omitting `--z-start` and `--z-end` (or providing only one) disables profiling
mode, ensuring autofocus behavior is unchanged when these parameters are not
supplied.

## Autofocus diagnostics (`--dump`, `--dump-label`, `--dump-frames`)

`STRMAFZ` can write the scan out to disk for offline analysis.

```
--dump 1              enable the dump (CSV traces + manifest)
--dump-label <text>   append a label to the folder name
--dump-frames 1       ALSO write one TIF per sample
```

`--dump 1` writes the traces and the manifest. Frames are **not** written unless
`--dump-frames` is also set: they are several hundred TIFs, often ~750 MB per scan,
whereas the traces are a few kB and are what the analysis actually reads.

`--dump-label` exists because a focus-approach validation performs two scans -- one
over tissue and one over blank slide -- and both would otherwise land as
`streaming_af_<timestamp>`, indistinguishable without opening them. Anything outside
`[A-Za-z0-9_-]` is replaced with `-`, and the result is truncated to 40 characters.

### Layout

```
<yaml_dir>/autofocus_tests/streaming_af_<TIMESTAMP>[_<LABEL>]/
    samples.csv       idx, wall_ms, z_assumed_um, z_actual_um, metric
    z_poll.csv        wall_ms, z_actual_um   -- the raw stage-position poll
    manifest.json     scan parameters and summary
    frames/           ONLY when --dump-frames is set
        frame_0000_t000093ms_zass-001.072.tif
        ...
```

`samples.csv` carries both Z columns on purpose. `z_actual_um` is interpolated from
the poll trace and is what the focus fit uses; `z_assumed_um` is the old
`wall_ms * velocity` model, kept so the two can be compared. They diverge whenever
`slow_speed_um_per_s` is mis-calibrated, and the divergence grows with distance
travelled -- see *Where a sample's Z comes from* in the QPSC autofocus documentation.

`manifest.json` records `z_start`, `z_end`, `velocity_um_s_configured`,
`motion_duration_ms`, `metric_name`, `n_kept_samples`, `n_z_poll_samples`,
`frames_written`, and the observed average velocity from the poll trace.

The retry walk gives each attempt its own `attempt_<N>/` subfolder; a single-pass
profiling or approach scan writes straight into the dump root. Readers should handle
both.

### Example

A profiling traverse from a retracted safe Z of 0 um in to -267 um, over tissue,
keeping only the traces:

```
STRMAFZ --yaml config_PPM.yml --modality ppm --range 267.0 \
        --z-start 0.0 --z-end -267.0 \
        --max-attempts 1 --dump 1 --dump-label tissue
```

Repeat it over a bare part of the same slide with `--dump-label blank`. Any peak
present in both is a surface rather than the sample.

## Installation

**Part of [QPSC (QuPath Scope Control)](https://github.com/uw-loci/QPSC)**

**Requirements:**
- Python 3.9 or later
- pip (Python package installer)
- Git (for `pip install git+https://...` commands)

**Important**: This package depends on `microscope-imageprocessing` (required) and `microscope-control` (required).
Optional dependencies:
- `ppm-library` -- only needed for PPM (polarized light) modality support
- `polscope-library` -- only needed for LC-PolScope birefringence reconstruction

See the [QPSC Installation Guide](https://github.com/uw-loci/QPSC#automated-installation-windows) for complete setup instructions.

### Quick Install (from GitHub)

**Install dependencies first:**
```bash
# 1. Install microscope-imageprocessing (required - background correction, OME-TIFF I/O)
pip install git+https://github.com/uw-loci/microscope_imageprocessing.git

# 2. Install microscope-control (required - hardware abstraction)
pip install git+https://github.com/uw-loci/microscope_control.git

# 3. (Optional) Install ppm-library for PPM modality support
pip install git+https://github.com/uw-loci/ppm_library.git

# 4. (Optional) Install polscope-library for LC-PolScope birefringence reconstruction
pip install git+https://github.com/uw-loci/polscope_library.git

# 5. Then install microscope_command_server
pip install git+https://github.com/uw-loci/microscope_command_server.git
```

### Development Install (editable mode)

```bash
git clone https://github.com/uw-loci/microscope_command_server.git
cd microscope_command_server
pip install -e .
```

**For automated setup**, use the [QPSC setup script](https://github.com/uw-loci/QPSC/blob/main/PPM-QuPath.ps1).

### Troubleshooting Installation

#### Problem: `ModuleNotFoundError: No module named 'microscope_command_server'`

**Cause:** Package not installed correctly or virtual environment not activated.

**Solution:**

1. **Ensure virtual environment is activated:**
   ```bash
   # Windows
   path\to\venv_qpsc\Scripts\Activate.ps1

   # Linux/macOS
   source path/to/venv_qpsc/bin/activate
   ```

2. **Reinstall the package:**
   ```bash
   pip install -e . --force-reinstall
   ```

3. **Verify installation:**
   ```bash
   pip show microscope-command-server
   ```

#### Problem: Entry point `microscope-command-server` command not found

**Cause:** Entry points not registered or PATH not updated.

**Solution:**

Try running the server directly:
```bash
# Using Python module
python -m microscope_command_server.server.qp_server

# Or with PYTHONPATH set (if needed)
export PYTHONPATH="/path/to/parent/directory:$PYTHONPATH"
microscope-command-server
```

#### Problem: Port 5000 already in use

**Symptom:** `OSError: [Errno 48] Address already in use`

**Cause:** Another server instance or application is using port 5000.

**Solution:**
```bash
# Find process using port 5000
# Windows:
netstat -ano | findstr :5000
# macOS/Linux:
lsof -i :5000

# Kill the process if safe
```

For more troubleshooting, see the [QPSC Installation Guide](https://github.com/uw-loci/QPSC#troubleshooting-python-package-installation).

## Quick Start

### Server Side

```python
from microscope_command_server.server.qp_server import run_server

# Start server
run_server(host='0.0.0.0', port=5000)
```

Or run from command line:
```bash
# Option 1: Entry point command (NOTE: uses hyphens, not underscores!)
microscope-command-server

# Option 2: Python module syntax
python -m microscope_command_server.server.qp_server
```

**Common mistake:** The command is `microscope-command-server` (with **hyphens**), not `microscope_command_server` (with underscores).

### Client Side

```python
from microscope_command_server.client import get_stageXY, move_stageXY

# Get current position
x, y = get_stageXY()

# Move stage
move_stageXY(x + 1000, y + 1000)
```

## Architecture

The server coordinates between QuPath (Java) and the microscope hardware (Python/Micro-Manager):

```
QuPath Extension -> Socket Client -> Microscope Server
                                          |
                  +-------+---------------+-------+-------+
                  |       |               |       |       |
          Microscope Microscope      PPM Lib  PolScope   ...
           Control ImageProcessing (opt)     Lib(opt)
              |        |                      |
              v        v                      v
        Micro-Manager Debayering,      Birefringence
          Hardware    Background,     reconstruction
                    OME-TIFF I/O,     per-tile
                    Z-stack projections
```

## Server Configuration

The microscope command server uses a **dynamic configuration** approach:

### Startup
- Server loads a minimal generic configuration (`config_generic.yml`)
- Connects to Micro-Manager (hardware must be available)
- Waits for client connections

### During Acquisition
- Client sends ACQUIRE command with `--yaml /path/to/config.yml` parameter
- Server loads microscope-specific config from the provided path
- Hardware settings are updated dynamically
- Microscope-specific methods (e.g., PPM rotation) are initialized

### Exploratory Commands
Commands like GETXY, MOVE, GETZ use the most recently loaded config:
- Before first ACQUIRE: Uses generic startup config with permissive stage limits
- After ACQUIRE: Uses the microscope-specific config from that acquisition

**Note**: Always provide the `--yaml` parameter in ACQUIRE commands to ensure correct microscope configuration.

## Testing

This package includes automated unit tests for components that can be tested without hardware.

### Automated Unit Tests

Pytest-compatible unit tests are located in the `tests/` directory:
- **`tests/test_tiles.py`** - Tests for TileConfiguration.txt parsing and generation

These tests:
- Run without hardware (use synthetic test data and temp files)
- Can be integrated into CI/CD pipelines
- Test protocol handling, tile configuration, and utility functions

**Running Unit Tests:**

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_tiles.py

# Run with coverage report
pytest --cov=microscope_command_server --cov-report=html

# View coverage report
open htmlcov/index.html  # or xdg-open on Linux
```

**Test Coverage:**

Current automated tests achieve ~60-70% coverage for testable components:
- ✅ TileConfiguration parsing (coordinates extraction)
- ✅ TileConfiguration generation (2D pixel coordinates and 3D stage coordinates)
- ⏸️ Socket protocol (future test expansion)
- ⏸️ Server communication (requires integration testing)

**Hardware Diagnostic Tools:**

This package does not include standalone diagnostic tools. Hardware testing is performed via:
- The `TESTAF` and `TESTADAF` server commands (call diagnostic functions from `microscope_control`)
- The `PPMSENS` and `PPMBIREF` server commands (call diagnostic functions from `ppm_library`)

See the `microscope_control` and `ppm_library` documentation for details on these diagnostic tools.

## License

MIT License - see [LICENSE](LICENSE) for details.

## AI-Assisted Development

This project was developed with assistance from [Claude](https://claude.ai) (Anthropic). Claude was used as a development tool for code generation, architecture design, debugging, and documentation throughout the project.
