"""Command handlers for the microscope socket server.

The COMMAND_HANDLERS dict maps 8-byte command bytes to handler functions.
Each handler has the signature:

    def handle_xxx(conn, client, hardware, settings, **kwargs)

Special return values:
    'DISCONNECT' -- caller should break the client loop
    'SHUTDOWN'   -- caller should signal server shutdown
    dict         -- CONFIG handler returns updated server state
"""

from microscope_command_server.server.protocol import ExtendedCommand

from microscope_command_server.server.handlers.position import (
    handle_getxy,
    handle_getz,
    handle_getxyz,
    handle_getfov,
    handle_getpxsz,
    handle_getr,
    handle_getzf,
    handle_move,
    handle_movez,
    handle_movznw,
    handle_movexyz,
    handle_mover,
)
from microscope_command_server.server.handlers.camera import (
    handle_getcam,
    handle_getmode,
    handle_setmode,
    handle_getexp,
    handle_setexp,
    handle_getgain,
    handle_setgain,
    handle_setcam,
    handle_getbin,
    handle_setbin,
    handle_getcap,
)
from microscope_command_server.server.handlers.live import (
    handle_getlive,
    handle_setlive,
    handle_getframe,
    handle_correctframe,
    handle_strtseq,
    handle_stopseq,
    handle_snap,
)
from microscope_command_server.server.handlers.status import (
    handle_status,
    handle_progress,
    handle_cancel,
    handle_reqmanf,
    handle_ackmf,
    handle_skipaf,
    handle_reqhwer,
    handle_ackhwer,
    handle_reqtwarn,
    handle_reqsat,
    handle_acksat,
)
from microscope_command_server.server.handlers.autofocus import (
    handle_testaf,
    handle_testadaf,
    handle_testafv,
    handle_afbench,
)
from microscope_command_server.server.handlers.calibration import (
    handle_wbcalibr,
    handle_wbsimple,
    handle_wbppm,
    handle_polcal,
    handle_ppmsens,
    handle_ppmbiref,
    handle_sbcalib,
    handle_getnoise,
    handle_noischar,
)
from microscope_command_server.server.handlers.acquisition import (
    handle_acquire,
    handle_bgacquire,
    handle_zstack,
    handle_tlapse,
)
from microscope_command_server.server.handlers.illumination import (
    handle_getillm,
    handle_setillm,
    handle_setilmd,
    handle_setprop,
    handle_applypr,
    handle_applych,
)
from microscope_command_server.server.handlers.system import (
    handle_config,
    handle_reconfig,
    handle_disconnect,
    handle_shutdown,
    handle_siftal,
    handle_siftim,
    handle_get_log,
)
from microscope_command_server.server.handlers.probez import handle_probez
from microscope_command_server.server.handlers.probe_stage_af import handle_probe_stage_af
from microscope_command_server.server.handlers.streaming_focus import (
    handle_streaming_focus,
    handle_abort_af,
)
from microscope_command_server.server.handlers.rapid_scan import handle_rapid_scan

# Map 8-byte command bytes -> handler function
COMMAND_HANDLERS = {
    # Position & movement
    ExtendedCommand.GETXY: handle_getxy,
    ExtendedCommand.GETZ: handle_getz,
    ExtendedCommand.GETXYZ: handle_getxyz,
    ExtendedCommand.GETFOV: handle_getfov,
    ExtendedCommand.GETPXSZ: handle_getpxsz,
    ExtendedCommand.GETR: handle_getr,
    ExtendedCommand.GETZF: handle_getzf,
    ExtendedCommand.MOVE: handle_move,
    ExtendedCommand.MOVEZ: handle_movez,
    ExtendedCommand.MOVZNW: handle_movznw,
    ExtendedCommand.MOVEXYZ: handle_movexyz,
    ExtendedCommand.MOVER: handle_mover,
    # Camera control
    ExtendedCommand.GETCAM: handle_getcam,
    ExtendedCommand.GETMODE: handle_getmode,
    ExtendedCommand.SETMODE: handle_setmode,
    ExtendedCommand.GETEXP: handle_getexp,
    ExtendedCommand.SETEXP: handle_setexp,
    ExtendedCommand.GETGAIN: handle_getgain,
    ExtendedCommand.SETGAIN: handle_setgain,
    ExtendedCommand.SETCAM: handle_setcam,
    ExtendedCommand.GETBIN: handle_getbin,
    ExtendedCommand.SETBIN: handle_setbin,
    ExtendedCommand.GETCAP: handle_getcap,
    # Live mode & snapshot
    ExtendedCommand.GETLIVE: handle_getlive,
    ExtendedCommand.SETLIVE: handle_setlive,
    ExtendedCommand.GETFRAME: handle_getframe,
    ExtendedCommand.CORRECTFRAME: handle_correctframe,
    ExtendedCommand.STRTSEQ: handle_strtseq,
    ExtendedCommand.STOPSEQ: handle_stopseq,
    ExtendedCommand.SNAP: handle_snap,
    # Status & coordination
    ExtendedCommand.STATUS: handle_status,
    ExtendedCommand.PROGRESS: handle_progress,
    ExtendedCommand.CANCEL: handle_cancel,
    ExtendedCommand.REQMANF: handle_reqmanf,
    ExtendedCommand.ACKMF: handle_ackmf,
    ExtendedCommand.SKIPAF: handle_skipaf,
    ExtendedCommand.REQHWER: handle_reqhwer,
    ExtendedCommand.ACKHWER: handle_ackhwer,
    ExtendedCommand.REQTWARN: handle_reqtwarn,
    ExtendedCommand.REQSAT: handle_reqsat,
    ExtendedCommand.ACKSAT: handle_acksat,
    # Autofocus testing
    ExtendedCommand.TESTAF: handle_testaf,
    ExtendedCommand.TESTADAF: handle_testadaf,
    ExtendedCommand.TESTAFV: handle_testafv,
    ExtendedCommand.AFBENCH: handle_afbench,
    # Calibration
    ExtendedCommand.WBCALIBR: handle_wbcalibr,
    ExtendedCommand.WBSIMPLE: handle_wbsimple,
    ExtendedCommand.WBPPM: handle_wbppm,
    ExtendedCommand.POLCAL: handle_polcal,
    ExtendedCommand.PPMSENS: handle_ppmsens,
    ExtendedCommand.PPMBIREF: handle_ppmbiref,
    ExtendedCommand.SBCALIB: handle_sbcalib,
    ExtendedCommand.GETNOISE: handle_getnoise,
    ExtendedCommand.NOISCHAR: handle_noischar,
    # Acquisition
    ExtendedCommand.ACQUIRE: handle_acquire,
    ExtendedCommand.BGACQUIRE: handle_bgacquire,
    ExtendedCommand.ZSTACK: handle_zstack,
    ExtendedCommand.TLAPSE: handle_tlapse,
    # Illumination & profile
    ExtendedCommand.GETILLM: handle_getillm,
    ExtendedCommand.SETILLM: handle_setillm,
    ExtendedCommand.SETILLMD: handle_setilmd,
    ExtendedCommand.SETPROP: handle_setprop,
    ExtendedCommand.APPLYPR: handle_applypr,
    ExtendedCommand.APPLYCH: handle_applych,
    # System & alignment
    ExtendedCommand.CONFIG: handle_config,
    ExtendedCommand.RECONFG: handle_reconfig,
    ExtendedCommand.DISCONNECT: handle_disconnect,
    ExtendedCommand.SHUTDOWN: handle_shutdown,
    ExtendedCommand.SIFTAL: handle_siftal,
    ExtendedCommand.SIFTIM: handle_siftim,
    # Diagnostics: session-log tail for the in-app bug reporter
    ExtendedCommand.GETLOG: handle_get_log,
    # Z-stage diagnostic probe
    ExtendedCommand.PROBEZ: handle_probez,
    # Setup-wizard stage probe (recommends streaming_af YAML values)
    ExtendedCommand.PRBSAFZ: handle_probe_stage_af,
    # Streaming autofocus
    ExtendedCommand.STRMAFZ: handle_streaming_focus,
    ExtendedCommand.ABORTAF: handle_abort_af,
    # Rapid scan
    ExtendedCommand.RPDSCAN: handle_rapid_scan,
}
