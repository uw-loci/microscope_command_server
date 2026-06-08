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
- `none` -- Preserve individual Z-planes without projection. Output structure
  changes to a per-plane layout, enabling 5D stitching (x, y, z, channel/angle,
  time). See "File layout on disk" below.

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

## Installation

**Part of [QPSC (QuPath Scope Control)](https://github.com/uw-loci/QPSC)**

**Requirements:**
- Python 3.9 or later
- pip (Python package installer)
- Git (for `pip install git+https://...` commands)

**Important**: This package depends on `microscope-imageprocessing` (required) and `microscope-control` (required).
`ppm-library` is an **optional** dependency, only needed for PPM (polarized light) modality support.
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

# 4. Then install microscope_command_server
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
                          +---------------+---------------+
                          |               |               |
                  Microscope       Microscope        PPM Library
                   Control       ImageProcessing     (optional)
                      |               |                  |
                      v               v                  v
              Micro-Manager     Debayering,        PPM-specific
                Hardware        Background,        analysis and
                               OME-TIFF I/O,       calibration
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
