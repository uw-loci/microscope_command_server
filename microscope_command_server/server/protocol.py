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
    STATUS = b"status__"  # Get acquisition status
    PROGRESS = b"progress"  # Get acquisition progress
    CANCEL = b"cancel__"  # Cancel acquisition
    BGACQUIRE = b"bgacquir"  # Acquire background images
    POLCAL = b"polcal__"  # Calibrate polarizer rotation stage
    TESTAF = b"testaf__"  # Test standard autofocus at current position
    TESTADAF = b"testadaf"  # Test adaptive autofocus at current position
    REQMANF = b"reqmanf_"  # Check if manual focus is requested
    ACKMF = b"ackmf___"  # Acknowledge manual focus - retry autofocus
    SKIPAF = b"skipaf__"  # Skip autofocus retry - use current focus
    REQHWER = b"reqhwer_"  # Check if hardware error recovery is requested
    ACKHWER = b"ackhwer_"  # Acknowledge hardware error - retry/skip/cancel
    AFBENCH = b"afbench_"  # Run autofocus parameter benchmark
    SNAP = b"snap____"  # Simple snap with fixed exposure (no adaptive)

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
