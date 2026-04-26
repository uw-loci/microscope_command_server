from enum import Enum


TCP_PORT = 5000  # Default port number for the server, can be changed as needed
END_MARKER = "ENDOFSTR"


class Command(Enum):
    GETXY = b"getxy___"
    GETZ = b"getz____"
    MOVEZ = b"move_z__"
    GETR = b"getr____"
    MOVER = b"move_r__"
    MOVE = b"move____"
    ACQUIRE = b"acquire_"
    SHUTDOWN = b"shutdown"
    DISCONNECT = b"quitclnt"
    FOV = b"getfov__"


## CMD echo | set /p="shutdown" | ncat.exe 127.0.0.1 5000

for command in Command:
    if len(command.value) != 8:
        raise ValueError(f"Command {command.name} must be exactly 8 bytes long.")
    if not isinstance(command.value, bytes):
        raise TypeError(f"Command {command.name} must be of type bytes.")


# Extend the Command enum with new commands
class ExtendedCommand:
    """Extended commands for enhanced acquisition control."""

    # Existing commands from Command enum
    GETXY = Command.GETXY.value
    GETZ = Command.GETZ.value
    MOVEZ = Command.MOVEZ.value
    MOVE = Command.MOVE.value
    GETR = Command.GETR.value
    MOVER = Command.MOVER.value
    SHUTDOWN = Command.SHUTDOWN.value
    DISCONNECT = Command.DISCONNECT.value
    GETFOV = Command.FOV.value
    ACQUIRE = Command.ACQUIRE.value

    # New commands (8 bytes each)
    CONFIG = b"config__"  # Set microscope configuration file (CRITICAL - must be first command)
    RECONFG = b"reconfg_"  # Re-read YAML configs from disk (after calibration writes)
    STATUS = b"status__"  # Get acquisition status
    PROGRESS = b"progress"  # Get acquisition progress
    CANCEL = b"cancel__"  # Cancel acquisition
    BGACQUIRE = b"bgacquir"  # Acquire background images
    POLCAL = b"polcal__"  # Calibrate polarizer rotation stage
    TESTAF = b"testaf__"  # Test standard autofocus at current position
    TESTADAF = b"testadaf"  # Test adaptive autofocus at current position
    TESTAFV = b"testafv_"  # Test autofocus validation (sweep + recovery from defocus)
    REQMANF = b"reqmanf_"  # Check if manual focus is requested
    ACKMF = b"ackmf___"  # Acknowledge manual focus - retry autofocus
    SKIPAF = b"skipaf__"  # Skip autofocus retry - use current focus
    REQHWER = b"reqhwer_"  # Check if hardware error recovery is requested
    ACKHWER = b"ackhwer_"  # Acknowledge hardware error - retry/skip/cancel
    AFBENCH = b"afbench_"  # Run autofocus parameter benchmark
    SNAP = b"snap____"  # Simple snap with fixed exposure (no adaptive)
    ZSTACK = b"zstack__"  # Z-stack acquisition at current XY (multi-Z, single tile)
    TLAPSE = b"tlapse__"  # Time-lapse acquisition at current position (repeat over time)
    SIFTAL = b"siftal__"  # SIFT auto-alignment: snap + match against WSI region file

    # PPM Testing Commands (for QPSC menu integration)
    PPMSENS = b"ppmsens_"  # PPM Rotation Sensitivity Test
    PPMBIREF = b"ppmbiref"  # PPM Birefringence Maximization Test
    SBCALIB = b"sbcalib_"  # Sunburst Calibration for hue-to-angle mapping

    # ---- JAI-Specific: White Balance Calibration Commands ----
    # These commands require a JAI trilinear color camera with per-channel
    # exposure and gain control. They use the JAIWhiteBalanceCalibrator from
    # microscope_control.jai to iteratively converge on target intensity.
    # Non-JAI cameras should use software white balance (see pipeline.py).
    WBCALIBR = b"wbcalibr"  # White Balance Calibration for JAI camera (legacy)
    WBSIMPLE = b"wbsimple"  # Simple WB at single exposure (JAI per-channel)
    WBPPM = b"wbppm___"  # PPM WB at 4 polarizer angles (JAI per-channel)

    # ---- JAI-Specific: Camera Control Commands (Camera Control dialog) ----
    # Per-channel (count>=3) paths require JAI. Unified (count=1) paths for
    # SETEXP use generic hardware.set_exposure(), but SETGAIN unified path
    # still uses JAI's set_unified_gain(). GETMODE/SETMODE are JAI-only.
    GETCAM = b"getcam__"  # Get camera name from Core (generic)
    GETMODE = b"getmode_"  # Get exposure/gain mode flags (JAI: individual vs unified)
    SETMODE = b"setmode_"  # Set exposure/gain mode flags (JAI-only, 2-byte payload)
    GETEXP = b"getexp__"  # Get exposure values (unified or JAI per-channel RGB)
    SETEXP = b"setexp__"  # Set exposure (count=1: generic, count>=3: JAI per-channel)
    GETGAIN = b"getgain_"  # Get gain values (unified or JAI per-channel RGB)
    SETGAIN = b"setgain_"  # Set gain (count=1: JAI unified, count>=3: JAI per-channel)
    SETCAM = b"setcam__"   # Compound: set mode + exposures + gains atomically (1 round-trip)

    # Binning (Camera Control v2 phase 1)
    GETBIN = b"getbin__"  # Get current + available binning factors (response: count + ints + current)
    SETBIN = b"setbin__"  # Set binning factor (1-byte unsigned payload)

    # Capabilities (Camera Control v2 phase 2)
    # Single round-trip query that returns everything Camera Control v2
    # needs to render. Optional 32-byte payload: profile name to scope the
    # answer (e.g. "if I were to apply Brightfield_10x, what controls?");
    # empty payload returns the current state.
    # Response: 4-byte big-endian length + UTF-8 JSON blob.
    GETCAP = b"getcap__"

    # NOTE: SETWBMD (camera WB mode control) was removed -- JAI hardware AWB
    # cannot be reliably controlled through Pycromanager. Set AWB manually in
    # MicroManager's Device Property Browser.

    # Live Mode Control Commands
    GETLIVE = b"getlive_"  # Check if live mode is currently running
    SETLIVE = b"setlive_"  # Set live mode on (1) or off (0)

    # Noise measurement
    GETNOISE = b"getnoise"  # Get per-channel noise stats (multi-frame temporal analysis)
    NOISCHAR = b"noischar"  # JAI noise characterization across gain/exposure grid

    # Pixel size query
    GETPXSZ = b"getpxsz_"  # Get MicroManager pixel size (um/pixel)

    # Live Viewer Commands (core-level, bypasses MM studio/live window)
    GETFRAME = b"getframe"  # Get latest frame from MM circular buffer
    STRTSEQ = b"strtseq_"  # Start continuous sequence acquisition (core-level)
    STOPSEQ = b"stopseq_"  # Stop continuous sequence acquisition (core-level)

    # 3D Position Commands (voxel support)
    GETXYZ = b"getxyz__"  # Get XYZ position as single command
    MOVEXYZ = b"movexyz_"  # Move to XYZ position as single command

    # Sweep Focus Commands
    MOVZNW = b"movznw__"  # Move Z non-blocking (no wait_for_device) - for sweep focus
    GETZF = b"getzf___"  # Get Z position only (fast, no X/Y read) - for sweep focus

    # Z-stage diagnostic probe (one-shot characterization run)
    PROBEZ = b"probez__"  # Run Z-stage timing/streaming probe, log results

    # Streaming autofocus -- continuous-Z autofocus via streamed frames
    # during non-blocking stage motion. Replaces the stepped sweep
    # drift check on calibrated hardware.
    STRMAFZ = b"strmafz_"  # Streaming autofocus scan
    RPDSCAN = b"rpdscan_"  # Rapid scan -- fast tiled brightfield, no AF, no Z

    # Illumination & Profile Commands
    GETILLM = b"getillm_"  # Get illumination state (power, range, on/off)
    SETILLM = b"setillm_"  # Set illumination power (4-byte float)
    APPLYPR = b"applypr_"  # Apply acquisition profile (calls apply_mode_setup)
    # Apply a single channel from a profile's channel library: cube + light
    # source switch + per-channel intensity property write + exposure. Used
    # by Live Viewer's per-channel preview radios. Empty channel id deactivates
    # all illumination for the profile's modality.
    # Payload: 32-byte profile name + 32-byte channel id (null-padded UTF-8).
    APPLYCH = b"applych_"
