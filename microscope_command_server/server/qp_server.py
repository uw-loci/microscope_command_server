"""
Microscope Command Server
=========================

A socket-based server that provides remote control of a microscope through Micro-Manager.
Can be used by any client software (QuPath, custom applications, scripts, etc.).
Handles stage movement, image acquisition, and multi-angle imaging workflows.

Enhanced Features:
- Acquisition status monitoring
- Real-time progress updates
- Acquisition cancellation support
- Non-blocking socket communication during acquisition
- Improved state management and logging
"""

import socket
import threading
import struct
import sys
import pathlib
import time
import enum
from threading import Lock
import logging
from datetime import datetime

import numpy as np

from microscope_control.config import ConfigManager
from microscope_command_server.modality import get_config as get_modality_config


def check_for_existing_server(host: str, port: int, timeout: float = 2.0) -> bool:
    """
    Check if a server is already running on the specified host and port.

    Attempts to connect to the port and send a simple query command.
    If successful, another server instance is already running.

    Args:
        host: The host to check (typically 127.0.0.1 for localhost)
        port: The port to check
        timeout: Connection timeout in seconds

    Returns:
        True if a server is already running, False otherwise
    """
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(timeout)
        test_socket.connect((host, port))

        # Try to send a simple command to verify it's our server
        # GETXY command is safe and quick
        test_socket.sendall(b"getxy___")

        # If we get here without exception, a server is responding
        test_socket.close()
        return True

    except (ConnectionRefusedError, socket.timeout, OSError):
        # Connection refused or timeout means no server is running
        return False
    except Exception:
        # Any other error - assume no server running
        return False
    finally:
        try:
            test_socket.close()
        except Exception:
            pass
from microscope_control.hardware import Position
from microscope_control.hardware.pycromanager import (
    PycromanagerHardware,
    init_pycromanager,
    MicroManagerConnectionError,
)
from microscope_command_server.server.protocol import ExtendedCommand, TCP_PORT, END_MARKER
from microscope_command_server.acquisition.workflow import _acquisition_workflow


# Configure logging - boot/pre-connection logging goes to console + fallback file
current_file_path = pathlib.Path(__file__).resolve()
base_dir = current_file_path.parent
log_dir = base_dir / "server_logfiles"
log_dir.mkdir(parents=True, exist_ok=True)
boot_log_filename = log_dir / f'qp_server_boot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(boot_log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Session log handler - added to root logger when CONFIG provides a config path,
# removed on client disconnect. Writes to <config_dir>/logs/server_session_*.log
_session_log_handler = None


def _start_session_logging(config_path: str) -> None:
    """
    Start session-based file logging in the config file's parent directory.

    Creates a log file at <config_parent>/logs/server_session_YYYYMMDD_HHMMSS.log.
    The handler is added to the root logger so all module loggers are captured.
    The handler flushes immediately on each log record (no buffered data lost on crash).

    Args:
        config_path: Path to the YAML config file sent by QuPath via CONFIG command
    """
    global _session_log_handler

    # Remove any existing session handler first
    _stop_session_logging()

    try:
        config_dir = pathlib.Path(config_path).resolve().parent
        session_log_dir = config_dir / "logs"
        session_log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_file = session_log_dir / f"server_session_{timestamp}.log"

        # Create a handler that flushes immediately via a custom emit override
        handler = logging.FileHandler(session_log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

        # Add to root logger so all child loggers are captured
        logging.getLogger().addHandler(handler)
        _session_log_handler = handler

        logger.info(f"Session logging started: {session_log_file}")
    except Exception as e:
        logger.error(f"Failed to start session logging: {e}", exc_info=True)


def _stop_session_logging() -> None:
    """
    Stop session-based file logging and clean up the handler.

    Flushes and closes the session log handler, then removes it from the root logger.
    """
    global _session_log_handler

    if _session_log_handler is not None:
        try:
            logger.info("Session logging stopped")
            _session_log_handler.flush()
            _session_log_handler.close()
            logging.getLogger().removeHandler(_session_log_handler)
        except Exception as e:
            logger.debug(f"Error closing session log handler: {e}")
        finally:
            _session_log_handler = None


# Server configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = TCP_PORT  # Default: 5000

# Threading events for coordination
shutdown_event = threading.Event()


# Global acquisition state management
class AcquisitionState(enum.Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Global acquisition tracking
acquisition_states = {}  # addr -> AcquisitionState
acquisition_progress = {}  # addr -> (current, total)
acquisition_locks = {}  # addr -> Lock
acquisition_cancel_events = {}  # addr -> Event
acquisition_failure_messages = {}  # addr -> str (error message when FAILED)
acquisition_final_z = {}  # addr -> float (final Z position when COMPLETED, for tilt model)
manual_focus_request_events = {}  # addr -> Event (set when manual focus needed)
manual_focus_complete_events = {}  # addr -> Event (set when user acknowledges)
manual_focus_user_choice = {}  # addr -> str ("retry", "skip", "cancel")
manual_focus_retries_remaining = {}  # addr -> int (number of retries remaining)

# Server configuration state - CRITICAL FOR SAFETY
# NEVER allow hardware operations with generic config - could damage microscope!
server_configured = False  # True only after CONFIG command received with valid microscope config
active_connection_addr = None  # Track single active client connection (blocks other connections)
active_connection_config_path = None  # Path to config file provided by active connection
connection_state_lock = Lock()  # Protect connection state from race conditions

# AWB state tracking -- True when Camera AWB corrections are present in analog gains.
# Set True when SETWBMD mode=1 (Continuous) or mode=2 (Once) completes.
# Remains True when mode=0 (Off) -- Off doesn't clear analog gain registers.
# Reset to False only when analog gains are explicitly cleared (e.g., by simple/per_angle mode setup).
awb_calibrated = False


def init_pycromanager_with_logger():
    """
    Initialize Pycro-Manager connection to Micro-Manager.

    Provides clean error messages if connection fails.
    """
    logger.info("Initializing Pycro-Manager connection...")
    try:
        core, studio = init_pycromanager()
        if not core:
            logger.error("Failed to initialize Micro-Manager connection")
            sys.exit(1)
        logger.info("Pycro-Manager initialized successfully")
        return core, studio

    except MicroManagerConnectionError as e:
        # Clean, informative error message
        logger.error("")
        logger.error("=" * 70)
        logger.error("MICRO-MANAGER CONNECTION FAILED")
        logger.error("=" * 70)
        logger.error("")
        for line in str(e).split('\n'):
            logger.error(line)
        logger.error("")
        logger.error("=" * 70)
        logger.error("Server cannot start without Micro-Manager.")
        logger.error("Please fix the issue above and restart the server.")
        logger.error("=" * 70)
        sys.exit(1)

    except Exception as e:
        # Unexpected error - still provide clean output
        logger.error("")
        logger.error("=" * 70)
        logger.error("UNEXPECTED ERROR CONNECTING TO MICRO-MANAGER")
        logger.error("=" * 70)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error: {e}")
        logger.error("")
        logger.error("Please ensure Micro-Manager is running and responsive.")
        logger.error("=" * 70)
        sys.exit(1)


# OPTION 3 HELPER (COMMENTED OUT): Auto-reconnect to MM on first command
# Uncomment this function and use it in handle_client() if using Option 3 lazy connection mode
# def ensure_micromanager_connected():
#     """
#     Attempt to connect to Micro-Manager if not already connected.
#     Used with Option 3 (lazy connection mode).
#
#     Returns:
#         bool: True if connected, False if connection failed
#     """
#     global core, studio, hardware
#
#     if core is not None and hardware is not None:
#         return True  # Already connected
#
#     logger.info("Attempting to connect to Micro-Manager...")
#     try:
#         core, studio = init_pycromanager()
#         if core:
#             hardware = PycromanagerHardware(core, studio, startup_settings)
#             logger.info("Successfully connected to Micro-Manager")
#             return True
#         else:
#             logger.error("Failed to connect to Micro-Manager - core is None")
#             return False
#     except Exception as e:
#         logger.error(f"Exception while connecting to Micro-Manager: {e}")
#         return False


# Initialize hardware connections
logger.info("Loading generic startup configuration...")
config_manager = ConfigManager()

## GENERIC CONFIG loaded for exploratory XYZ movements
# Actual microscope-specific config loaded during ACQUIRE command via --yaml parameter
package_dir = pathlib.Path(__file__).parent.parent

# Try to load generic config
generic_config_path = package_dir / "configurations" / "config_generic.yml"
if generic_config_path.exists():
    logger.info(f"Loading generic startup config from {generic_config_path}")
    startup_settings = config_manager.load_config_file(str(generic_config_path))
else:
    # Fallback to hardcoded minimal config if file doesn't exist
    logger.warning("Generic config file not found, using hardcoded minimal defaults")
    startup_settings = {
        "microscope": {"name": "Generic", "type": "Unconfigured"},
        "stage": {
            "stage_id": "GENERIC_STAGE",
            "limits": {
                "x_um": {"low": -100000, "high": 100000},
                "y_um": {"low": -100000, "high": 100000},
                "z_um": {"low": -20000, "high": 20000}
            }
        },
        "ppm_optics": "NA",
        "modalities": {},
        "hardware": {},
        "id_stage": {},
        "id_detector": {},
        "id_camera": {}
    }

# Load LOCI resources if available (for device lookup during ACQUIRE)
loci_rsc_file = package_dir / "configurations" / "resources" / "resources_LOCI.yml"
if loci_rsc_file.exists():
    loci_resources = config_manager.load_config_file(str(loci_rsc_file))
    startup_settings.update(loci_resources)
    logger.info("Loaded LOCI resources for hardware device lookup")
else:
    logger.warning("LOCI resources file not found - device lookups may fail during ACQUIRE")

# Initialize hardware with generic config (will be replaced during ACQUIRE)
# OPTION 1 (ACTIVE): Fail-fast - require Micro-Manager at startup
logger.info("Initializing Micro-Manager connection...")
core, studio = init_pycromanager_with_logger()
hardware = PycromanagerHardware(core, studio, startup_settings)
logger.info("Hardware initialized with generic config")
logger.info("Server ready - microscope-specific config will be loaded from ACQUIRE --yaml parameter")

# OPTION 3 (COMMENTED OUT): Allow server to start without MM, auto-reconnect on first command
# Uncomment this section and comment out Option 1 above to enable lazy connection mode
# logger.info("Micro-Manager connection will be attempted on first command")
# core = None
# studio = None
# hardware = None
# logger.warning("Server starting without Micro-Manager - commands will fail until MM connected")


def acquisitionWorkflow(message, client_addr):
    """Wrapper for acquisition workflow with state management."""

    def _update_progress(current: int, total: int):
        with acquisition_locks[client_addr]:
            acquisition_progress[client_addr] = (current, total)

    def _set_state(state_str: str, error_message: str = None, final_z: float = None):
        with acquisition_locks[client_addr]:
            try:
                new_state = AcquisitionState[state_str]
                acquisition_states[client_addr] = new_state
                # Store error message if state is FAILED
                if new_state == AcquisitionState.FAILED and error_message:
                    acquisition_failure_messages[client_addr] = error_message
                # Store final Z position if state is COMPLETED (for tilt correction model)
                if new_state == AcquisitionState.COMPLETED and final_z is not None:
                    acquisition_final_z[client_addr] = final_z
            except KeyError:
                acquisition_states[client_addr] = AcquisitionState.FAILED
                if error_message:
                    acquisition_failure_messages[client_addr] = error_message

    def _is_cancelled() -> bool:
        return acquisition_cancel_events[client_addr].is_set()

    def _request_manual_focus(retries_remaining: int):
        """Signal manual focus needed and wait for user acknowledgment.

        Args:
            retries_remaining: Number of autofocus retries remaining after this

        Returns:
            str: User's choice - "retry", "skip", or "cancel"
        """
        logger.info(f"Manual focus requested for client {client_addr} (retries remaining: {retries_remaining})")
        # Store retries remaining so REQMANF can return it
        manual_focus_retries_remaining[client_addr] = retries_remaining
        # Set request event to signal client
        manual_focus_request_events[client_addr].set()
        # Clear previous choice
        manual_focus_user_choice[client_addr] = None
        # Wait for user to acknowledge (blocks acquisition thread)
        logger.info("Waiting for manual focus acknowledgment from user...")
        manual_focus_complete_events[client_addr].wait()
        # Get user's choice
        user_choice = manual_focus_user_choice[client_addr] or "cancel"
        # Clear events for next potential use
        manual_focus_request_events[client_addr].clear()
        manual_focus_complete_events[client_addr].clear()
        manual_focus_user_choice[client_addr] = None
        manual_focus_retries_remaining[client_addr] = 0
        logger.info(f"Manual focus acknowledged, user chose: {user_choice}")
        return user_choice

    return _acquisition_workflow(
        message=message,
        client_addr=client_addr,
        hardware=hardware,
        config_manager=config_manager,
        logger=logger,
        update_progress=_update_progress,
        set_state=_set_state,
        is_cancelled=_is_cancelled,
        request_manual_focus=_request_manual_focus,
        connection_config_path=active_connection_config_path,
        awb_calibrated=awb_calibrated,
    )


def handle_client(conn, addr):
    """
    Handle commands from a connected client with enhanced acquisition control.
    """
    logger.info(f">>> New client connected from {addr}")

    # Initialize client state
    acquisition_locks[addr] = Lock()
    acquisition_states[addr] = AcquisitionState.IDLE
    acquisition_progress[addr] = (0, 0)
    acquisition_cancel_events[addr] = threading.Event()
    acquisition_failure_messages[addr] = None
    manual_focus_request_events[addr] = threading.Event()
    manual_focus_complete_events[addr] = threading.Event()
    manual_focus_user_choice[addr] = None
    manual_focus_retries_remaining[addr] = 0

    acquisition_thread = None

    try:
        while True:
            # All commands are 8 bytes
            data = conn.recv(8)
            if not data:
                logger.info(f"Client {addr} disconnected (no data)")
                break

            logger.debug(f"Received command from {addr}: {data}")

            # Connection management commands
            if data == ExtendedCommand.DISCONNECT:
                logger.info(f"Client {addr} requested to disconnect")
                break

            if data == ExtendedCommand.SHUTDOWN:
                logger.warning(f"Client {addr} requested server shutdown")
                shutdown_event.set()
                break

            # CONFIG command - MUST be first command sent by client (safety critical)
            if data == ExtendedCommand.CONFIG:
                global server_configured, active_connection_addr, active_connection_config_path

                logger.info(f"Client {addr} sent CONFIG command")

                try:
                    # Read config file path: 4 bytes length + path string
                    path_length_bytes = conn.recv(4)
                    if not path_length_bytes:
                        logger.error("CONFIG: No path length received")
                        conn.sendall(b"CFG_FAIL")
                        continue

                    path_length = struct.unpack("!I", path_length_bytes)[0]
                    logger.debug(f"CONFIG: Expecting config path of {path_length} bytes")

                    config_path_bytes = conn.recv(path_length)
                    config_path = config_path_bytes.decode("utf-8")
                    logger.info(f"CONFIG: Received config path: {config_path}")

                    # Check connection locking
                    with connection_state_lock:
                        if active_connection_addr is not None and active_connection_addr != addr:
                            # Another connection exists - check if same IP (likely reconnect)
                            # addr is (ip, port) tuple - compare IP only
                            active_ip = active_connection_addr[0]
                            new_ip = addr[0]

                            if active_ip == new_ip:
                                # Same IP reconnecting - allow takeover (previous connection likely crashed)
                                logger.warning(f"CONFIG: Same IP reconnecting - taking over from {active_connection_addr}")
                                logger.warning("CONFIG: Previous connection may have been improperly closed")
                                # Clear the old connection state (will be set to new addr below)
                                active_connection_addr = None
                                active_connection_config_path = None
                            else:
                                # Different IP - reject this CONFIG
                                logger.warning(f"CONFIG: Rejected - connection {active_connection_addr} already active")
                                error_msg = f"BLOCKED: Active connection from {active_connection_addr}".encode("utf-8")
                                error_length = struct.pack("!I", len(error_msg))
                                conn.sendall(b"CFG_BLCK" + error_length + error_msg)
                                continue

                    # Load the config file
                    new_settings = config_manager.load_config_file(config_path)

                    # Validate essential config sections exist
                    # Note: id_detector specs come from resources file, not main config
                    # Main config has hardware.detectors which lists detector IDs
                    required_sections = ["microscope", "stage"]
                    missing = [s for s in required_sections if s not in new_settings or not new_settings[s]]
                    if missing:
                        error_msg = f"Config missing required sections: {', '.join(missing)}"
                        logger.error(f"CONFIG: {error_msg}")
                        error_bytes = error_msg.encode("utf-8")
                        error_length = struct.pack("!I", len(error_bytes))
                        conn.sendall(b"CFG_FAIL" + error_length + error_bytes)
                        continue

                    # Update hardware with new configuration
                    hardware.settings = new_settings
                    hardware._initialize_microscope_methods()

                    # Mark server as configured and track active connection
                    with connection_state_lock:
                        server_configured = True
                        active_connection_addr = addr
                        active_connection_config_path = config_path

                    microscope_name = new_settings.get("microscope", {}).get("name", "Unknown")
                    logger.info(f"CONFIG: Successfully loaded config for microscope: {microscope_name}")
                    logger.info(f"CONFIG: Server now configured and ready for operations")

                    # Start session logging to <config_dir>/logs/
                    _start_session_logging(config_path)

                    # Send success response
                    conn.sendall(b"CFG___OK")

                except FileNotFoundError as e:
                    error_msg = f"Config file not found: {config_path}"
                    logger.error(f"CONFIG: {error_msg}")
                    error_bytes = error_msg.encode("utf-8")
                    error_length = struct.pack("!I", len(error_bytes))
                    conn.sendall(b"CFG_FAIL" + error_length + error_bytes)
                except Exception as e:
                    error_msg = f"Failed to load config: {str(e)}"
                    logger.error(f"CONFIG: {error_msg}", exc_info=True)
                    error_bytes = error_msg.encode("utf-8")
                    error_length = struct.pack("!I", len(error_bytes))
                    conn.sendall(b"CFG_FAIL" + error_length + error_bytes)

                continue

            # Position query commands
            if data == ExtendedCommand.GETXY:
                # OPTION 3 USAGE (COMMENTED OUT): Check MM connection before hardware access
                # if not ensure_micromanager_connected():
                #     logger.error("Micro-Manager not connected - cannot get XY position")
                #     error_msg = f"MM_NOCON".ljust(8)[:8]
                #     conn.sendall(error_msg.encode("utf-8"))
                #     continue

                logger.debug(f"Client {addr} requested XY position")
                try:
                    current_position_xyz = hardware.get_current_position()
                    response = struct.pack("!ff", current_position_xyz.x, current_position_xyz.y)
                    conn.sendall(response)
                    logger.debug(
                        f"Sent XY position to {addr}: ({current_position_xyz.x}, {current_position_xyz.y})"
                    )
                except Exception as e:
                    logger.error(f"Failed to get XY position: {e}", exc_info=True)
                    # Send error message (8 bytes to match expected response size)
                    error_msg = f"HW_ERROR".ljust(8)[:8]
                    conn.sendall(error_msg.encode("utf-8"))
                continue

            if data == ExtendedCommand.GETZ:
                logger.debug(f"Client {addr} requested Z position")
                try:
                    current_position_xyz = hardware.get_current_position()
                    response = struct.pack("!f", current_position_xyz.z)
                    conn.sendall(response)
                    logger.debug(f"Sent Z position to {addr}: {current_position_xyz.z}")
                except Exception as e:
                    logger.error(f"Failed to get Z position: {e}", exc_info=True)
                    # Send error message (4 bytes to match expected response size)
                    error_msg = "HWERR"[:4]
                    conn.sendall(error_msg.encode("utf-8"))
                continue

            if data == ExtendedCommand.GETFOV:
                logger.debug(f"Client {addr} requested Field of View")

                # SAFETY CHECK: Require CONFIG before GETFOV
                if not server_configured:
                    logger.warning(f"GETFOV: Blocked - server not configured (CONFIG command required first)")
                    error_response = struct.pack("!ff", -1.0, -1.0)  # Negative values indicate error
                    conn.sendall(error_response)
                    continue

                try:
                    current_fov_x, current_fov_y = hardware.get_fov()
                    response = struct.pack("!ff", current_fov_x, current_fov_y)
                    conn.sendall(response)
                    logger.debug(f"Sent FOV to {addr}: ({current_fov_x}, {current_fov_y})")
                except Exception as e:
                    logger.error(f"Failed to get FOV: {e}")
                    # Send error response
                    response = struct.pack("!ff", 0.0, 0.0)
                    conn.sendall(response)
                continue

            if data == ExtendedCommand.GETPXSZ:
                logger.debug(f"Client {addr} requested pixel size")

                # SAFETY CHECK: Require CONFIG before GETPXSZ
                if not server_configured:
                    logger.warning(f"GETPXSZ: Blocked - server not configured (CONFIG command required first)")
                    response = struct.pack("!f", 0.0)
                    conn.sendall(response)
                    continue

                try:
                    pixel_size = hardware.core.get_pixel_size_um()
                    response = struct.pack("!f", float(pixel_size))
                    conn.sendall(response)
                    logger.debug(f"Sent pixel size to {addr}: {pixel_size} um/pixel")
                except Exception as e:
                    logger.error(f"Failed to get pixel size: {e}")
                    response = struct.pack("!f", 0.0)
                    conn.sendall(response)
                continue

            if data == ExtendedCommand.GETR:
                logger.debug(f"Client {addr} requested rotation angle")
                try:
                    angle = hardware.get_psg_ticks()
                    response = struct.pack("!f", angle)
                    conn.sendall(response)
                    logger.debug(f"Sent rotation angle to {addr}: {angle} deg")
                except Exception as e:
                    logger.error(f"Failed to get rotation angle: {e}", exc_info=True)
                    # Send error message (4 bytes to match expected response size)
                    error_msg = "HWERR"[:4]
                    conn.sendall(error_msg.encode("utf-8"))
                continue

            # Movement commands
            if data == ExtendedCommand.MOVE:
                coords = conn.recv(8)
                if len(coords) == 8:
                    x, y = struct.unpack("!ff", coords)
                    logger.info(f"Client {addr} requested move to: X={x}, Y={y}")
                    try:
                        t0 = time.perf_counter()
                        hardware.move_to_position(Position(x, y))
                        t_ms = (time.perf_counter() - t0) * 1000
                        logger.info(f"MOVE completed to X={x}, Y={y} in {t_ms:.0f}ms")
                    except Exception as e:
                        logger.error(f"Failed to move to XY position: {e}", exc_info=True)
                        # No response expected for movement commands, but log the error
                else:
                    logger.error(f"Client {addr} sent incomplete move coordinates")
                continue

            if data == ExtendedCommand.MOVEZ:
                z = conn.recv(4)
                z_position = struct.unpack("!f", z)[0]
                logger.info(f"Client {addr} requested move to Z={z_position}")
                try:
                    hardware.move_to_position(Position(z=z_position))
                    logger.info(f"Move completed to Z={z_position}")
                except Exception as e:
                    logger.error(f"Failed to move to Z position: {e}", exc_info=True)
                continue

            if data == ExtendedCommand.MOVER:
                coords = conn.recv(4)
                angle = struct.unpack("!f", coords)[0]
                logger.info(f"Client {addr} requested rotation to {angle} deg")
                try:
                    hardware.set_psg_ticks(
                        angle
                    )  # , is_sequence_start=True)  # Single rotation command
                    logger.info(f"Rotation completed to {angle} deg")
                except Exception as e:
                    logger.error(f"Failed to rotate stage: {e}", exc_info=True)
                continue

            # ============ ACQUISITION STATUS COMMANDS ============

            # Status query command
            if data == ExtendedCommand.STATUS:
                with acquisition_locks[addr]:
                    state = acquisition_states[addr]
                    # If state is FAILED and we have an error message, send it
                    if state == AcquisitionState.FAILED and addr in acquisition_failure_messages:
                        # Send "FAILED: <message>" format (truncated to fit in response)
                        error_msg = acquisition_failure_messages[addr]
                        # Java client expects to parse this format
                        state_str = f"FAILED: {error_msg}"[:250]  # Reasonable limit for error message
                        # Pad to 16 bytes minimum for compatibility, but can be longer
                        response = state_str.encode('utf-8')
                        conn.sendall(response)
                        logger.debug(f"Sent FAILED status with message to {addr}: {error_msg[:50]}...")
                    # If state is COMPLETED and we have final_z, include it for tilt model
                    elif state == AcquisitionState.COMPLETED and addr in acquisition_final_z:
                        final_z = acquisition_final_z[addr]
                        # Send "COMPLETED|final_z:<value>" format
                        state_str = f"COMPLETED|final_z:{final_z:.2f}"
                        response = state_str.encode('utf-8')
                        conn.sendall(response)
                        logger.debug(f"Sent COMPLETED status with final_z to {addr}: {final_z:.2f}")
                    else:
                        # Send state as 16-byte string (padded)
                        state_str = state.value.ljust(16)[:16]
                        conn.sendall(state_str.encode())
                        logger.debug(f"Sent acquisition status to {addr}: {state.value}")
                continue

            # Progress query command
            if data == ExtendedCommand.PROGRESS:
                with acquisition_locks[addr]:
                    current, total = acquisition_progress[addr]
                # Send as two integers
                response = struct.pack("!II", current, total)
                conn.sendall(response)
                logger.debug(f"Sent progress to {addr}: {current}/{total}")
                continue

            # Cancel acquisition command
            if data == ExtendedCommand.CANCEL:
                logger.warning(f"Client {addr} requested acquisition cancellation")
                with acquisition_locks[addr]:
                    if acquisition_states[addr] == AcquisitionState.RUNNING:
                        acquisition_states[addr] = AcquisitionState.CANCELLING
                        acquisition_cancel_events[addr].set()
                        logger.info(f"Cancellation initiated for {addr}")
                # Send acknowledgment
                conn.sendall(b"ACK")
                continue

            # ============ MANUAL FOCUS REQUEST/ACKNOWLEDGMENT ============

            if data == ExtendedCommand.REQMANF:
                # Check if manual focus is requested
                if manual_focus_request_events[addr].is_set():
                    # Manual focus needed - send request status with retries remaining (8 bytes exactly)
                    retries = manual_focus_retries_remaining.get(addr, 0)
                    # Format: "NEEDEDnn" where nn is 00-99
                    response = f"NEEDED{retries:02d}".encode('utf-8')
                    conn.sendall(response)
                    logger.debug(f"Sent manual focus request to {addr} (retries remaining: {retries})")
                else:
                    # No manual focus needed (8 bytes exactly)
                    conn.sendall(b"IDLE____")
                    logger.debug(f"Manual focus not needed for {addr}")
                continue

            # Manual focus acknowledgment - retry autofocus
            if data == ExtendedCommand.ACKMF:
                # Client chose to retry autofocus after manual adjustment
                manual_focus_user_choice[addr] = "retry"
                manual_focus_complete_events[addr].set()
                conn.sendall(b"ACK")
                logger.info(f"Manual focus acknowledged by client {addr} - will retry autofocus")
                continue

            # Skip autofocus retry - use current focus
            if data == ExtendedCommand.SKIPAF:
                # Client chose to use current focus position
                manual_focus_user_choice[addr] = "skip"
                manual_focus_complete_events[addr].set()
                conn.sendall(b"ACK")
                logger.info(f"Manual focus acknowledged by client {addr} - using current focus")
                continue

            # ============ ACQUISITION COMMAND ============

            if data == ExtendedCommand.ACQUIRE:
                logger.info(f"Client {addr} requested acquisition workflow")

                # Check if already running
                with acquisition_locks[addr]:
                    if acquisition_states[addr] == AcquisitionState.RUNNING:
                        logger.warning(f"Acquisition already running for {addr}")
                        continue
                    # Set state to RUNNING immediately
                    acquisition_states[addr] = AcquisitionState.RUNNING
                    acquisition_progress[addr] = (0, 0)

                # Read the full message immediately
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                # Set a timeout for reading
                conn.settimeout(5.0)

                try:
                    while True:
                        # Read in chunks
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                f"Connection closed while reading acquisition message from {addr}"
                            )
                            with acquisition_locks[addr]:
                                acquisition_states[addr] = AcquisitionState.FAILED
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        # Check if we have the end marker
                        full_message = "".join(message_parts)
                        if END_MARKER in full_message:
                            # Remove the end marker
                            message = full_message.replace("," + END_MARKER, "").replace(
                                END_MARKER, ""
                            )
                            logger.debug(
                                f"Received complete acquisition message ({total_bytes} bytes) "
                                f"in {time.time() - start_time:.2f}s"
                            )

                            # Clear cancellation event
                            acquisition_cancel_events[addr].clear()

                            # Start acquisition in separate thread
                            acquisition_thread = threading.Thread(
                                target=acquisitionWorkflow,
                                args=(message, addr),
                                daemon=True,
                                name=f"Acquisition-{addr}",
                            )
                            acquisition_thread.start()

                            logger.info(f"Acquisition thread started for {addr}")

                            # Send acknowledgment to prevent client timeout
                            # Format matches BGACQUIRE pattern for consistency
                            ack_response = "STARTED:ACQUIRE".ljust(16)[:16].encode()
                            conn.sendall(ack_response)
                            logger.debug(f"Sent ACQUIRE acknowledgment to {addr}")
                            break

                        # Safety check for message size
                        if total_bytes > 10000:  # 10KB max
                            logger.error(
                                f"Acquisition message too large from {addr}: {total_bytes} bytes"
                            )
                            with acquisition_locks[addr]:
                                acquisition_states[addr] = AcquisitionState.FAILED
                            break

                        # Timeout check
                        if time.time() - start_time > 10:
                            logger.error(f"Timeout reading acquisition message from {addr}")
                            with acquisition_locks[addr]:
                                acquisition_states[addr] = AcquisitionState.FAILED
                            break

                except socket.timeout:
                    logger.error(f"Socket timeout reading acquisition message from {addr}")
                    with acquisition_locks[addr]:
                        acquisition_states[addr] = AcquisitionState.FAILED
                except Exception as e:
                    logger.error(f"Error reading acquisition message from {addr}: {e}")
                    with acquisition_locks[addr]:
                        acquisition_states[addr] = AcquisitionState.FAILED
                finally:
                    # Reset socket to blocking mode
                    conn.settimeout(None)

                continue

            if data == ExtendedCommand.BGACQUIRE:
                logger.info(f"Client {addr} requested background acquisition")

                # Read the message using the same pattern as ACQUIRE command
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading background acquisition message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Check for boolean flags first (no value)
                            use_per_angle_wb = "--use_per_angle_wb" in message

                            # Split by known flags to avoid issues with spaces in paths
                            # Include --wb-mode as a valued flag
                            flags = ["--yaml", "--output", "--modality", "--angles", "--exposures", "--wb-mode", "--objective", "--detector"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    # Find where this flag starts
                                    start_idx = message.index(flag) + len(flag)

                                    # Find where the next flag starts (or use end of string)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1 :]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break
                                    # Also check for boolean flag --use_per_angle_wb
                                    if "--use_per_angle_wb" in message[start_idx:]:
                                        wb_pos = message.index("--use_per_angle_wb", start_idx)
                                        if wb_pos < end_idx:
                                            end_idx = wb_pos

                                    # Extract the value and clean it up
                                    value = message[start_idx:end_idx].strip()

                                    # Map to the parameter name
                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--modality":
                                        params["modality"] = value
                                    elif flag == "--angles":
                                        params["angles_str"] = value
                                    elif flag == "--exposures":
                                        params["exposures_str"] = value
                                    elif flag == "--wb-mode":
                                        params["wb_mode"] = value.lower()
                                    elif flag == "--objective":
                                        params["objective"] = value
                                    elif flag == "--detector":
                                        params["detector"] = value

                            # Resolve wb_mode: prefer explicit --wb-mode, fall back to boolean flag
                            if "wb_mode" in params:
                                logger.info(f"WB mode for background acquisition: {params['wb_mode']}")
                            elif use_per_angle_wb:
                                params["wb_mode"] = "per_angle"
                                logger.info("Per-angle white balance enabled for background acquisition (legacy flag)")
                            # If neither --wb-mode nor --use_per_angle_wb, leave wb_mode unset
                            # and let simple_background_collection use its default

                            # Keep legacy flag for backward compat
                            params["use_per_angle_wb"] = use_per_angle_wb

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path", "modality"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # SAFETY WARNING: Check if ACQUIRE yaml differs from CONFIG
                            if active_connection_config_path:
                                acquire_yaml = pathlib.Path(params["yaml_file_path"]).resolve()
                                connection_yaml = pathlib.Path(active_connection_config_path).resolve()
                                if acquire_yaml != connection_yaml:
                                    logger.warning("=" * 80)
                                    logger.warning("CONFIG MISMATCH WARNING")
                                    logger.warning(f"Connection CONFIG:  {connection_yaml}")
                                    logger.warning(f"ACQUIRE --yaml:     {acquire_yaml}")
                                    logger.warning("ACQUIRE yaml will override connection config for this acquisition")
                                    logger.warning("This may cause unexpected behavior or hardware misconfiguration!")
                                    logger.warning("=" * 80)

                            # Send immediate acknowledgment to prevent client timeout
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for background acquisition"
                                )

                                # Execute background acquisition using simplified collection
                                from microscope_command_server.acquisition.workflow import (
                                    simple_background_collection,
                                )

                                # Create progress update function for this client
                                def update_progress(current, total):
                                    with acquisition_locks[addr]:
                                        acquisition_progress[addr] = (current, total)

                                final_exposures = simple_background_collection(
                                    yaml_file_path=params["yaml_file_path"],
                                    output_folder_path=params["output_folder_path"],
                                    modality=params["modality"],
                                    angles_str=params.get("angles_str", "()"),
                                    exposures_str=params.get("exposures_str", "()"),
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    logger=logger,
                                    update_progress=update_progress,
                                    use_per_angle_wb=params.get("use_per_angle_wb", False),
                                    wb_mode=params.get("wb_mode"),
                                    objective=params.get("objective"),
                                    detector=params.get("detector"),
                                )

                                # Format exposures as angle:exposure pairs
                                # e.g., "90:137.1,7:245.8,-7:155.2"
                                exposures_formatted = ",".join(
                                    f"{angle}:{exposure:.2f}"
                                    for angle, exposure in sorted(final_exposures.items())
                                )

                                # Send success response with output path and final exposures
                                response = f"SUCCESS:{params['output_folder_path']}|{exposures_formatted}".encode()
                                conn.sendall(response)
                                logger.info(
                                    f"Background acquisition completed successfully with exposures: {exposures_formatted}"
                                )

                            except Exception as e:
                                logger.error(
                                    f"Background acquisition failed: {str(e)}", exc_info=True
                                )
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            # We found and processed the END_MARKER, so break the while loop
                            break

                        # Safety checks for the while loop
                        if total_bytes > 10000:  # 10KB max
                            logger.error(
                                f"Background acquisition message too large: {total_bytes} bytes"
                            )
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading background acquisition message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading background acquisition message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in background acquisition: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode

                continue

            if data == ExtendedCommand.WBCALIBR:
                logger.info(f"Client {addr} requested white balance calibration")

                # Read the message using the same pattern as BGACQUIRE
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading white balance message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)
                        logger.debug(f"WBCALIBR: received {total_bytes} bytes so far")

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()
                            logger.info(f"WBCALIBR message: {message}")

                            # Parse the message
                            params = {}

                            # Parse flags: --yaml, --output, --modality, --objective,
                            #              --target, --tolerance, --defocus
                            flags = [
                                "--yaml",
                                "--output",
                                "--modality",
                                "--objective",
                                "--target",
                                "--tolerance",
                                "--defocus",
                            ]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    # Find the CLOSEST next flag (check all flags, not just remaining ones)
                                    for next_flag in flags:
                                        if next_flag != flag and next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos

                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--modality":
                                        params["modality"] = value
                                    elif flag == "--objective":
                                        params["objective"] = value
                                    elif flag == "--target":
                                        params["target_intensity"] = float(value)
                                    elif flag == "--tolerance":
                                        params["tolerance"] = float(value)
                                    elif flag == "--defocus":
                                        params["defocus_um"] = float(value)

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path", "modality"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment to prevent client timeout
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for white balance calibration"
                                )

                                # Import the calibration module
                                from microscope_control.jai import (
                                    JAIWhiteBalanceCalibrator,
                                    CalibrationConfig,
                                    JAICameraProperties,
                                )
                                from pathlib import Path

                                # Build calibration config
                                wb_config = CalibrationConfig(
                                    target_value=params.get("target_intensity", 180.0),
                                    tolerance=params.get("tolerance", 5.0),
                                    defocus_offset_um=params.get("defocus_um"),
                                )

                                # Create calibrator with hardware
                                jai_props = JAICameraProperties(hardware.core)
                                calibrator = JAIWhiteBalanceCalibrator(hardware, jai_props)

                                # Set up rotation callback if modality has rotation
                                mod_config = get_modality_config(params["modality"])
                                rotation_callback = None
                                if mod_config.has_rotation and hasattr(hardware, "set_psg_ticks"):
                                    rotation_callback = hardware.set_psg_ticks

                                # Set up defocus callback if configured
                                defocus_callback = None
                                if wb_config.defocus_offset_um is not None:
                                    def create_defocus_callback():
                                        def defocus_fn(offset_um):
                                            current_pos = hardware.get_current_position()
                                            original_z = current_pos.z
                                            new_z = original_z + offset_um
                                            hardware.move_to_position(
                                                hardware.get_current_position()._replace(z=new_z)
                                            )
                                            def restore():
                                                hardware.move_to_position(
                                                    hardware.get_current_position()._replace(z=original_z)
                                                )
                                            return original_z, restore
                                        return defocus_fn
                                    defocus_callback = create_defocus_callback()

                                # Run calibration
                                output_path = Path(params["output_folder_path"])
                                result = calibrator.calibrate(
                                    config=wb_config,
                                    output_path=output_path,
                                    rotation_callback=rotation_callback,
                                    defocus_callback=defocus_callback,
                                )

                                # Format response
                                exp_str = (
                                    f"exp_r:{result.exposures_ms['red']:.2f},"
                                    f"exp_g:{result.exposures_ms['green']:.2f},"
                                    f"exp_b:{result.exposures_ms['blue']:.2f}"
                                )
                                gain_str = (
                                    f"gain_r:{result.gains['red']:.2f},"
                                    f"gain_g:{result.gains['green']:.2f},"
                                    f"gain_b:{result.gains['blue']:.2f}"
                                )
                                status = "CONVERGED" if result.converged else "NOT_CONVERGED"

                                response = f"SUCCESS:{status}|{output_path}|{exp_str}|{gain_str}"
                                conn.sendall(response.encode())
                                logger.info(f"White balance calibration completed: {status}")

                            except ImportError as e:
                                error_msg = f"JAI calibration module not available: {e}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                            except Exception as e:
                                error_msg = f"White balance calibration failed: {str(e)}"
                                logger.error(error_msg, exc_info=True)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                            break

                        if total_bytes > 100000:
                            logger.error(
                                "White balance message exceeds maximum size"
                            )
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading white balance message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading white balance message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in white balance calibration: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode
                    # Reset per-channel mode so subsequent operations (autofocus,
                    # SNAP, acquisition) can use unified set_exposure() correctly
                    try:
                        from microscope_control.jai import JAICameraProperties
                        jai_props = JAICameraProperties(hardware.core)
                        jai_props.disable_individual_exposure()
                        jai_props.disable_individual_gain()
                        jai_props.set_rb_analog_gains(red=1.0, blue=1.0)
                        logger.debug("Reset per-channel mode after WBCALIBR")
                    except (ImportError, Exception):
                        pass

                continue

            # ==================== WBSIMPLE: Simple White Balance ====================
            # JAI-SPECIFIC: Per-channel exposure/gain calibration for a single
            # imaging condition (one exposure, one target intensity).
            #
            # Protocol: 8-byte command, then variable-length text message with
            #   flag-based parameters (--yaml, --output, --camera, --exposure,
            #   --target, --tolerance, --max_gain_db, etc.)
            #
            # Response sequence:
            #   1. Immediately sends "STARTED:{output_path}" acknowledgment
            #   2. Runs iterative calibration (may take seconds to minutes)
            #   3. Sends "SUCCESS:{path}|CONVERGED|exp_r:...|gain_r:..." or
            #      "FAILED:{reason}" on completion
            #
            # Post-calibration cleanup (in finally block):
            #   Resets camera to unified exposure/gain mode and sets all analog
            #   gains to 1.0. This prevents per-channel settings from leaking
            #   into subsequent non-calibrated acquisitions.
            if data == ExtendedCommand.WBSIMPLE:
                logger.info(f"Client {addr} requested simple white balance calibration")

                # Track calibration result so the finally block can apply it
                # to the camera for live view (instead of resetting to defaults).
                _wb_calibration_result = None

                # Read the message using the same pattern as WBCALIBR
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading WBSIMPLE message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)
                        logger.debug(f"WBSIMPLE: received {total_bytes} bytes so far")

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()
                            logger.info(f"WBSIMPLE message: {message}")

                            # Parse the message
                            params = {}

                            # Parse flags: --yaml, --objective, --detector, --output,
                            #              --camera, --exposure, --target, --tolerance,
                            #              --max_gain_db, --gain_threshold,
                            #              --max_iterations, --calibrate_black_level,
                            #              --base_gain, --exposure_soft_cap_ms, --boosted_max_gain_db
                            flags = [
                                "--yaml",
                                "--objective",
                                "--detector",
                                "--output",
                                "--camera",
                                "--exposure",
                                "--target",
                                "--tolerance",
                                "--max_gain_db",
                                "--gain_threshold",
                                "--max_iterations",
                                "--calibrate_black_level",
                                "--base_gain",
                                "--exposure_soft_cap_ms",
                                "--boosted_max_gain_db",
                            ]

                            # Helper to find a flag as a complete word (followed by space)
                            def find_flag_position(msg, flag):
                                """Find flag position ensuring it's followed by a space."""
                                search_pattern = flag + " "
                                if search_pattern in msg:
                                    return msg.index(search_pattern)
                                return -1

                            for i, flag in enumerate(flags):
                                flag_pos = find_flag_position(message, flag)
                                if flag_pos >= 0:
                                    start_idx = flag_pos + len(flag)
                                    end_idx = len(message)
                                    # Find the CLOSEST next flag
                                    for next_flag in flags:
                                        if next_flag != flag:
                                            next_pos = find_flag_position(message[start_idx:], next_flag)
                                            if next_pos >= 0:
                                                actual_pos = start_idx + next_pos
                                                if actual_pos < end_idx:
                                                    end_idx = actual_pos

                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--objective":
                                        params["objective"] = value
                                    elif flag == "--detector":
                                        params["detector"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--camera":
                                        params["camera"] = value
                                    elif flag == "--exposure":
                                        params["initial_exposure_ms"] = float(value)
                                    elif flag == "--target":
                                        params["target_intensity"] = float(value)
                                    elif flag == "--tolerance":
                                        params["tolerance"] = float(value)
                                    elif flag == "--max_gain_db":
                                        params["max_gain_db"] = float(value)
                                    elif flag == "--gain_threshold":
                                        params["gain_threshold"] = float(value)
                                    elif flag == "--max_iterations":
                                        params["max_iterations"] = int(value)
                                    elif flag == "--calibrate_black_level":
                                        params["calibrate_black_level"] = value.lower() == "true"
                                    elif flag == "--base_gain":
                                        params["base_gain"] = float(value)
                                    elif flag == "--exposure_soft_cap_ms":
                                        params["exposure_soft_cap_ms"] = float(value)
                                    elif flag == "--boosted_max_gain_db":
                                        params["boosted_max_gain_db"] = float(value)

                            # Validate required parameters
                            required = ["output_folder_path", "initial_exposure_ms"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for WBSIMPLE"
                                )

                                # Import the calibration module
                                from microscope_control.jai import (
                                    JAIWhiteBalanceCalibrator,
                                    JAICameraProperties,
                                )
                                from pathlib import Path

                                # Create calibrator with hardware
                                jai_props = JAICameraProperties(hardware.core)
                                calibrator = JAIWhiteBalanceCalibrator(hardware, jai_props)

                                # Run simple calibration using the new method
                                output_path = Path(params["output_folder_path"])
                                result = calibrator.calibrate_simple(
                                    initial_exposure_ms=params["initial_exposure_ms"],
                                    target=params.get("target_intensity", 180.0),
                                    tolerance=params.get("tolerance", 5.0),
                                    output_path=output_path,
                                    max_gain_db=params.get("max_gain_db"),
                                    gain_threshold_ratio=params.get("gain_threshold"),
                                    max_iterations=params.get("max_iterations"),
                                    calibrate_black_level=params.get("calibrate_black_level"),
                                    base_gain=params.get("base_gain"),
                                    exposure_soft_cap_ms=params.get("exposure_soft_cap_ms"),
                                    boosted_max_gain_db=params.get("boosted_max_gain_db"),
                                )

                                # Update imageprocessing config if yaml path provided
                                if "yaml_file_path" in params:
                                    wb_objective = params.get("objective")
                                    wb_detector = params.get("detector")
                                    logger.info(
                                        f"Simple WB: saving config with objective={wb_objective}, "
                                        f"detector={wb_detector}"
                                    )
                                    calibrator.update_imageprocessing_config(
                                        config_path=Path(params["yaml_file_path"]),
                                        result=result,
                                        calibration_type="simple",
                                        angle_name="uncrossed",  # Simple WB calibrates at 90 deg (uncrossed)
                                        modality=params.get("modality"),
                                        objective=wb_objective,
                                        detector=wb_detector,
                                    )

                                # Format response with new gain model
                                exp_str = (
                                    f"exp_r:{result.exposures_ms['red']:.2f},"
                                    f"exp_g:{result.exposures_ms['green']:.2f},"
                                    f"exp_b:{result.exposures_ms['blue']:.2f}"
                                )
                                gain_str = (
                                    f"unified:{result.unified_gain:.3f},"
                                    f"analog_r:{result.analog_red:.3f},"
                                    f"analog_b:{result.analog_blue:.3f}"
                                )
                                status = "CONVERGED" if result.converged else "NOT_CONVERGED"

                                response = f"SUCCESS:{output_path}|{status}|{exp_str}|{gain_str}"

                                # Append noise stats if available
                                if result.noise_stats is not None:
                                    ns = result.noise_stats
                                    response += (
                                        f"|noise_r:{ns.channel_stddevs['red']:.2f},"
                                        f"noise_g:{ns.channel_stddevs['green']:.2f},"
                                        f"noise_b:{ns.channel_stddevs['blue']:.2f}"
                                    )

                                conn.sendall(response.encode())
                                logger.info(f"WBSIMPLE completed: {status}")
                                _wb_calibration_result = result

                            except ImportError as e:
                                error_msg = f"JAI calibration module not available: {e}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                            except Exception as e:
                                error_msg = f"WBSIMPLE failed: {str(e)}"
                                logger.error(error_msg, exc_info=True)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                            break

                        if total_bytes > 100000:
                            logger.error("WBSIMPLE message exceeds maximum size")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading WBSIMPLE message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading WBSIMPLE message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in WBSIMPLE: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode
                    # Apply calibration result to camera so live view shows
                    # the white-balanced image, or reset on failure.
                    try:
                        from microscope_control.jai import JAICameraProperties
                        jai_props = JAICameraProperties(hardware.core)
                        if (_wb_calibration_result is not None
                                and _wb_calibration_result.converged):
                            jai_props.set_channel_exposures(
                                red=_wb_calibration_result.exposures_ms['red'],
                                green=_wb_calibration_result.exposures_ms['green'],
                                blue=_wb_calibration_result.exposures_ms['blue'],
                                auto_enable=True,
                            )
                            jai_props.set_unified_gain(
                                _wb_calibration_result.unified_gain)
                            jai_props.set_rb_analog_gains(
                                red=_wb_calibration_result.analog_red,
                                blue=_wb_calibration_result.analog_blue)
                            logger.info(
                                "Applied calibration to camera for live view: "
                                "R=%.2f G=%.2f B=%.2f, "
                                "unified=%.2f, aR=%.3f, aB=%.3f",
                                _wb_calibration_result.exposures_ms['red'],
                                _wb_calibration_result.exposures_ms['green'],
                                _wb_calibration_result.exposures_ms['blue'],
                                _wb_calibration_result.unified_gain,
                                _wb_calibration_result.analog_red,
                                _wb_calibration_result.analog_blue,
                            )
                        else:
                            # Reset to clean state on failure
                            jai_props.set_rb_analog_gains(red=1.0, blue=1.0)
                            jai_props.set_unified_gain(1.0)
                            jai_props.disable_individual_exposure()
                            logger.debug("Reset camera state after WBSIMPLE "
                                         "(calibration did not converge)")
                    except (ImportError, Exception):
                        pass

                continue

            # ==================== WBPPM: PPM White Balance (4 angles) ====================
            # JAI-SPECIFIC: Per-channel exposure/gain calibration repeated at
            # each of 4 PPM polarizer angles (positive, negative, crossed,
            # uncrossed). Each angle may have a different target intensity
            # because optical transmission varies dramatically with polarizer
            # orientation (e.g., crossed ~125 vs uncrossed ~245).
            #
            # Protocol: 8-byte command, then variable-length text with
            #   per-angle flags (--positive_exp, --positive_angle, --target_positive, ...)
            #   plus shared calibration tuning flags (same as WBSIMPLE).
            #
            # Per-angle target priority (highest to lowest):
            #   1. Client-provided --target_{angle} flags
            #   2. YAML background_exposures.angles.{angle}.achieved_intensity
            #   3. YAML calibration_targets.target_intensities.{angle}
            #   4. Default fallback: 180.0
            #
            # Gain reset between angles: The calibrator resets per-channel
            # gains to 1.0 at the start of each angle's calibration to ensure
            # clean convergence without carryover from the previous angle.
            #
            # Response: "SUCCESS:{path}|{angle}:{exp_r},{exp_g},{exp_b}:{gain_r},{gain_g},{gain_b}:{Y/N}|..."
            # Post-calibration: Same mode reset as WBSIMPLE (unified mode, gains to 1.0).
            if data == ExtendedCommand.WBPPM:
                logger.info(f"Client {addr} requested PPM white balance calibration (4 angles)")

                # Track calibration results so the finally block can apply them
                # to the camera for live view (using uncrossed angle settings).
                _wb_rotation_results = None

                # Read the message
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading WBPPM message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)
                        logger.debug(f"WBPPM: received {total_bytes} bytes so far")

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()
                            logger.info(f"WBPPM message: {message}")

                            # Parse the message
                            params = {}

                            # Parse flags for PPM white balance:
                            # --yaml, --output, --camera,
                            # --positive_exp, --positive_angle, --target_positive,
                            # --negative_exp, --negative_angle, --target_negative,
                            # --crossed_exp, --crossed_angle, --target_crossed,
                            # --uncrossed_exp, --uncrossed_angle, --target_uncrossed,
                            # --target, --tolerance,
                            # --max_gain_db, --gain_threshold, --max_iterations, --calibrate_black_level,
                            # --base_gain, --exposure_soft_cap_ms, --boosted_max_gain_db
                            flags = [
                                "--yaml",
                                "--objective",
                                "--detector",
                                "--output",
                                "--camera",
                                "--positive_exp",
                                "--positive_angle",
                                "--target_positive",
                                "--negative_exp",
                                "--negative_angle",
                                "--target_negative",
                                "--crossed_exp",
                                "--crossed_angle",
                                "--target_crossed",
                                "--uncrossed_exp",
                                "--uncrossed_angle",
                                "--target_uncrossed",
                                "--target",
                                "--tolerance",
                                "--max_gain_db",
                                "--gain_threshold",
                                "--max_iterations",
                                "--calibrate_black_level",
                                "--base_gain",
                                "--exposure_soft_cap_ms",
                                "--boosted_max_gain_db",
                            ]

                            # Helper to find a flag as a complete word (followed by space)
                            def find_flag_position(msg, flag):
                                """Find flag position ensuring it's followed by a space."""
                                search_pattern = flag + " "
                                if search_pattern in msg:
                                    return msg.index(search_pattern)
                                return -1

                            for i, flag in enumerate(flags):
                                flag_pos = find_flag_position(message, flag)
                                if flag_pos >= 0:
                                    start_idx = flag_pos + len(flag)
                                    end_idx = len(message)
                                    # Find the CLOSEST next flag
                                    for next_flag in flags:
                                        if next_flag != flag:
                                            next_pos = find_flag_position(message[start_idx:], next_flag)
                                            if next_pos >= 0:
                                                actual_pos = start_idx + next_pos
                                                if actual_pos < end_idx:
                                                    end_idx = actual_pos

                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--objective":
                                        params["objective"] = value
                                    elif flag == "--detector":
                                        params["detector"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--camera":
                                        params["camera"] = value
                                    elif flag == "--positive_exp":
                                        params["positive_exp"] = float(value)
                                    elif flag == "--positive_angle":
                                        params["positive_angle"] = float(value)
                                    elif flag == "--target_positive":
                                        params["target_positive"] = float(value)
                                    elif flag == "--negative_exp":
                                        params["negative_exp"] = float(value)
                                    elif flag == "--negative_angle":
                                        params["negative_angle"] = float(value)
                                    elif flag == "--target_negative":
                                        params["target_negative"] = float(value)
                                    elif flag == "--crossed_exp":
                                        params["crossed_exp"] = float(value)
                                    elif flag == "--crossed_angle":
                                        params["crossed_angle"] = float(value)
                                    elif flag == "--target_crossed":
                                        params["target_crossed"] = float(value)
                                    elif flag == "--uncrossed_exp":
                                        params["uncrossed_exp"] = float(value)
                                    elif flag == "--uncrossed_angle":
                                        params["uncrossed_angle"] = float(value)
                                    elif flag == "--target_uncrossed":
                                        params["target_uncrossed"] = float(value)
                                    elif flag == "--target":
                                        params["target_intensity"] = float(value)
                                    elif flag == "--tolerance":
                                        params["tolerance"] = float(value)
                                    elif flag == "--max_gain_db":
                                        params["max_gain_db"] = float(value)
                                    elif flag == "--gain_threshold":
                                        params["gain_threshold"] = float(value)
                                    elif flag == "--max_iterations":
                                        params["max_iterations"] = int(value)
                                    elif flag == "--calibrate_black_level":
                                        params["calibrate_black_level"] = value.lower() == "true"
                                    elif flag == "--base_gain":
                                        params["base_gain"] = float(value)
                                    elif flag == "--exposure_soft_cap_ms":
                                        params["exposure_soft_cap_ms"] = float(value)
                                    elif flag == "--boosted_max_gain_db":
                                        params["boosted_max_gain_db"] = float(value)

                            # Validate required parameters
                            required = [
                                "output_folder_path",
                                "positive_exp", "positive_angle",
                                "negative_exp", "negative_angle",
                                "crossed_exp", "crossed_angle",
                                "uncrossed_exp", "uncrossed_angle",
                            ]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for WBPPM"
                                )

                                # Import the calibration module
                                from microscope_control.jai import (
                                    JAIWhiteBalanceCalibrator,
                                    JAICameraProperties,
                                )
                                from pathlib import Path

                                # Build angle/exposure pairs
                                angle_exposures = {
                                    "positive": (params["positive_angle"], params["positive_exp"]),
                                    "negative": (params["negative_angle"], params["negative_exp"]),
                                    "crossed": (params["crossed_angle"], params["crossed_exp"]),
                                    "uncrossed": (params["uncrossed_angle"], params["uncrossed_exp"]),
                                }

                                # Build per-angle targets dictionary
                                # Priority: client-provided > YAML background_exposures > YAML target_intensities > default
                                per_angle_targets = {}

                                # Check if client provided per-angle targets
                                client_targets = {
                                    "positive": params.get("target_positive"),
                                    "negative": params.get("target_negative"),
                                    "crossed": params.get("target_crossed"),
                                    "uncrossed": params.get("target_uncrossed"),
                                }

                                # Load targets from YAML if not provided by client
                                yaml_targets_loaded = False
                                if "yaml_file_path" in params:
                                    try:
                                        from microscope_command_server.acquisition.workflow import (
                                            get_target_intensity_for_angle,
                                        )
                                        for angle_name in ["positive", "negative", "crossed", "uncrossed"]:
                                            if client_targets[angle_name] is not None:
                                                # Client provided explicit value
                                                per_angle_targets[angle_name] = client_targets[angle_name]
                                            else:
                                                # Try YAML lookup
                                                angle_deg = params[f"{angle_name}_angle"]
                                                target_val, source = get_target_intensity_for_angle(
                                                    angle=angle_deg,
                                                    modality=params.get("modality", "ppm"),
                                                    config_path=Path(params["yaml_file_path"]),
                                                )
                                                per_angle_targets[angle_name] = target_val
                                                logger.info(
                                                    f"WB target for {angle_name}: {target_val} (from {source})"
                                                )
                                        yaml_targets_loaded = True
                                    except Exception as e:
                                        logger.warning(f"Failed to load targets from YAML: {e}")

                                # If YAML loading failed, use client values or None
                                if not yaml_targets_loaded:
                                    for angle_name in ["positive", "negative", "crossed", "uncrossed"]:
                                        if client_targets[angle_name] is not None:
                                            per_angle_targets[angle_name] = client_targets[angle_name]

                                # Create calibrator with hardware
                                jai_props = JAICameraProperties(hardware.core)
                                calibrator = JAIWhiteBalanceCalibrator(hardware, jai_props)

                                # Set up rotation callback
                                rotation_callback = None
                                if hasattr(hardware, "set_psg_ticks"):
                                    rotation_callback = hardware.set_psg_ticks

                                # Run per-angle calibration
                                output_path = Path(params["output_folder_path"])
                                results = calibrator.calibrate_ppm(
                                    angle_exposures=angle_exposures,
                                    target=params.get("target_intensity", 180.0),
                                    tolerance=params.get("tolerance", 5.0),
                                    output_path=output_path,
                                    rotation_callback=rotation_callback,
                                    per_angle_targets=per_angle_targets if per_angle_targets else None,
                                    max_gain_db=params.get("max_gain_db"),
                                    gain_threshold_ratio=params.get("gain_threshold"),
                                    max_iterations=params.get("max_iterations"),
                                    calibrate_black_level=params.get("calibrate_black_level"),
                                    base_gain=params.get("base_gain"),
                                    exposure_soft_cap_ms=params.get("exposure_soft_cap_ms"),
                                    boosted_max_gain_db=params.get("boosted_max_gain_db"),
                                )

                                # Update imageprocessing config for each angle
                                if "yaml_file_path" in params:
                                    # Get objective/detector from command params (preferred) or hardware.settings (fallback)
                                    wb_objective = params.get("objective")
                                    wb_detector = params.get("detector")
                                    if not wb_objective or not wb_detector:
                                        if hasattr(hardware, 'settings') and hardware.settings:
                                            wb_objective = wb_objective or hardware.settings.get("objective_in_use") or hardware.settings.get("objective")
                                            wb_detector = wb_detector or hardware.settings.get("detector_in_use") or hardware.settings.get("detector")
                                    logger.info(f"WB calibration: saving to imaging_profiles with objective={wb_objective}, detector={wb_detector}")

                                    wb_modality = params.get("modality", "ppm")
                                    for angle_name, result in results.items():
                                        calibrator.update_imageprocessing_config(
                                            config_path=Path(params["yaml_file_path"]),
                                            result=result,
                                            calibration_type="per_angle",
                                            angle_name=angle_name,
                                            modality=wb_modality,
                                            objective=wb_objective,
                                            detector=wb_detector,
                                        )

                                # Format response with results for all angles
                                # Format: SUCCESS:path|angle:exp_r,exp_g,exp_b:unified,aR,aB:Y/N|...
                                response_parts = [f"SUCCESS:{output_path}"]
                                all_converged = True
                                for name, result in results.items():
                                    exp_str = (
                                        f"{result.exposures_ms['red']:.2f},"
                                        f"{result.exposures_ms['green']:.2f},"
                                        f"{result.exposures_ms['blue']:.2f}"
                                    )
                                    gain_str = (
                                        f"{result.unified_gain:.3f},"
                                        f"{result.analog_red:.3f},"
                                        f"{result.analog_blue:.3f}"
                                    )
                                    converged = "Y" if result.converged else "N"
                                    response_parts.append(f"{name}:{exp_str}:{gain_str}:{converged}")
                                    if not result.converged:
                                        all_converged = False

                                response = "|".join(response_parts)
                                conn.sendall(response.encode())
                                logger.info(f"WBPPM completed: all_converged={all_converged}")
                                _wb_rotation_results = results

                            except ImportError as e:
                                error_msg = f"JAI calibration module not available: {e}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                            except Exception as e:
                                error_msg = f"WBPPM failed: {str(e)}"
                                logger.error(error_msg, exc_info=True)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                            break

                        if total_bytes > 100000:
                            logger.error("WBPPM message exceeds maximum size")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading WBPPM message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading WBPPM message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in WBPPM: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode
                    # Apply uncrossed calibration to camera so live view shows
                    # the white-balanced image, or reset on failure.
                    try:
                        from microscope_control.jai import JAICameraProperties
                        jai_props = JAICameraProperties(hardware.core)
                        # Use uncrossed (90 deg) result for live view -- it is
                        # the brightest angle and most natural for visual QC.
                        uncrossed = (
                            _wb_rotation_results.get("uncrossed")
                            if _wb_rotation_results else None
                        )
                        if uncrossed is not None and uncrossed.converged:
                            jai_props.set_channel_exposures(
                                red=uncrossed.exposures_ms['red'],
                                green=uncrossed.exposures_ms['green'],
                                blue=uncrossed.exposures_ms['blue'],
                                auto_enable=True,
                            )
                            jai_props.set_unified_gain(uncrossed.unified_gain)
                            jai_props.set_rb_analog_gains(
                                red=uncrossed.analog_red,
                                blue=uncrossed.analog_blue)
                            logger.info(
                                "Applied uncrossed calibration to camera for "
                                "live view: R=%.2f G=%.2f B=%.2f, "
                                "unified=%.2f, aR=%.3f, aB=%.3f",
                                uncrossed.exposures_ms['red'],
                                uncrossed.exposures_ms['green'],
                                uncrossed.exposures_ms['blue'],
                                uncrossed.unified_gain,
                                uncrossed.analog_red,
                                uncrossed.analog_blue,
                            )
                        else:
                            # Reset to clean state on failure
                            jai_props.set_rb_analog_gains(red=1.0, blue=1.0)
                            jai_props.set_unified_gain(1.0)
                            jai_props.disable_individual_exposure()
                            logger.debug("Reset camera state after WBPPM "
                                         "(no uncrossed result to apply)")
                    except (ImportError, Exception):
                        pass

                continue

            if data == ExtendedCommand.SNAP:
                logger.info(f"Client {addr} requested simple snap (fixed exposure)")
                snap_start_time = time.time()

                # Read the message with parameters
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error("Connection closed while reading snap message")
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)
                        logger.debug(f"SNAP: received {total_bytes} bytes so far")

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Parse flags: --angle, --exposure, --output, --debayer, --white_balance, --yaml, --objective, --detector, --exp_r, --exp_g, --exp_b
                            flags = ["--angle", "--exposure", "--output", "--debayer", "--white_balance", "--yaml", "--objective", "--detector", "--exp_r", "--exp_g", "--exp_b"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1:]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break

                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--angle":
                                        params["angle"] = float(value)
                                    elif flag == "--exposure":
                                        params["exposure_ms"] = float(value)
                                    elif flag == "--output":
                                        params["output_path"] = value
                                    elif flag == "--debayer":
                                        # Support "auto", "true"/"1"/"yes", "false"/"0"/"no"
                                        val = value.lower()
                                        if val == "auto":
                                            params["debayer"] = "auto"
                                        else:
                                            params["debayer"] = val in ("true", "1", "yes")
                                    elif flag == "--white_balance":
                                        params["white_balance"] = value.lower() in ("true", "1", "yes")
                                    elif flag == "--yaml":
                                        params["yaml_path"] = value
                                    elif flag == "--objective":
                                        params["objective"] = value
                                    elif flag == "--detector":
                                        params["detector"] = value
                                    elif flag == "--exp_r":
                                        params["exp_r"] = float(value)
                                    elif flag == "--exp_g":
                                        params["exp_g"] = float(value)
                                    elif flag == "--exp_b":
                                        params["exp_b"] = float(value)

                            # Validate required parameters
                            required = ["angle", "exposure_ms", "output_path"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            try:
                                import tifffile
                                from pathlib import Path

                                angle = params["angle"]
                                exposure_ms = params["exposure_ms"]
                                output_path = Path(params["output_path"])
                                debayer = params.get("debayer", "auto")
                                use_white_balance = params.get("white_balance", False)
                                yaml_path = params.get("yaml_path")

                                # Per-channel exposures for direct control (e.g., WB calibration loops)
                                exp_r = params.get("exp_r")
                                exp_g = params.get("exp_g")
                                exp_b = params.get("exp_b")

                                # Create output directory if needed
                                output_path.parent.mkdir(parents=True, exist_ok=True)

                                # Priority 1: Direct per-channel exposures (for calibration loops)
                                # Priority 2: WB calibration lookup from YAML
                                # Priority 3: Unified exposure
                                wb_applied = False
                                if exp_r is not None and exp_g is not None and exp_b is not None:
                                    # Direct per-channel control - used for WB calibration loops
                                    try:
                                        from microscope_control.jai import JAICameraProperties
                                        jai_props = JAICameraProperties(hardware.core)
                                        jai_props.set_channel_exposures(
                                            red=exp_r,
                                            green=exp_g,
                                            blue=exp_b,
                                            auto_enable=True,
                                        )
                                        wb_applied = True
                                        logger.info(
                                            f"SNAP: Applied direct per-channel exposures: "
                                            f"R={exp_r:.2f}ms, G={exp_g:.2f}ms, B={exp_b:.2f}ms"
                                        )
                                    except (ImportError, Exception) as e:
                                        logger.warning(f"SNAP: Failed to set per-channel exposures: {e}")
                                        wb_applied = False

                                elif use_white_balance and yaml_path:
                                    try:
                                        from microscope_command_server.acquisition.workflow import (
                                            load_jai_calibration_from_imageprocessing,
                                            apply_jai_calibration_for_angle,
                                            get_interpolated_calibration_for_angle,
                                        )

                                        # Get objective/detector from params or hardware.settings
                                        wb_objective = params.get("objective")
                                        wb_detector = params.get("detector")
                                        if not wb_objective or not wb_detector:
                                            if hasattr(hardware, 'settings') and hardware.settings:
                                                wb_objective = wb_objective or hardware.settings.get("objective_in_use") or hardware.settings.get("objective")
                                                wb_detector = wb_detector or hardware.settings.get("detector_in_use") or hardware.settings.get("detector")

                                        if wb_objective and wb_detector:
                                            # Derive modality from YAML config path
                                            # (e.g. config_PPM.yml -> "PPM")
                                            snap_modality = params.get("modality", "ppm")
                                            jai_cal = load_jai_calibration_from_imageprocessing(
                                                config_path=Path(yaml_path),
                                                per_angle=True,
                                                modality=snap_modality,
                                                objective=wb_objective,
                                                detector=wb_detector,
                                                logger=logger,
                                            )
                                            if jai_cal:
                                                # Calculate exposure scale factor to allow adaptive
                                                # exposure control while preserving WB color ratios.
                                                # The calibration provides per-channel exposures for
                                                # color balance; we scale them by the ratio of the
                                                # adaptive exposure_ms to the calibration base exposure.
                                                exposure_scale = None
                                                if "angles" in jai_cal:
                                                    angle_cal = get_interpolated_calibration_for_angle(
                                                        angle=angle,
                                                        angles_cal=jai_cal["angles"],
                                                        logger=logger,
                                                    )
                                                    if angle_cal:
                                                        cal_exposures = angle_cal.get("exposures_ms", {})
                                                        base_exp = (
                                                            cal_exposures.get("r", 50.0) +
                                                            cal_exposures.get("g", 50.0) +
                                                            cal_exposures.get("b", 50.0)
                                                        ) / 3.0
                                                        if base_exp > 0:
                                                            exposure_scale = exposure_ms / base_exp
                                                            logger.debug(
                                                                f"SNAP: WB exposure scale={exposure_scale:.2f}x "
                                                                f"(adaptive={exposure_ms:.1f}ms / base={base_exp:.1f}ms)"
                                                            )

                                                wb_applied, exp_info = apply_jai_calibration_for_angle(
                                                    hardware=hardware,
                                                    jai_calibration=jai_cal,
                                                    angle=angle,
                                                    per_angle=True,
                                                    logger=logger,
                                                    exposure_scale=exposure_scale,
                                                )
                                                if wb_applied:
                                                    if exposure_scale is not None and exposure_scale != 1.0:
                                                        logger.info(
                                                            f"SNAP: Applied WB with intensity scaling for {angle:.2f} deg "
                                                            f"(scale={exposure_scale:.2f}x)"
                                                        )
                                                    else:
                                                        logger.info(f"SNAP: Applied per-angle white balance for {angle:.2f} deg")
                                                else:
                                                    logger.warning(f"SNAP: Failed to apply white balance for {angle:.2f} deg")
                                            else:
                                                logger.warning(f"SNAP: No JAI calibration found in {yaml_path}")
                                        else:
                                            logger.warning(f"SNAP: Cannot apply WB - missing objective ({wb_objective}) or detector ({wb_detector})")
                                    except ImportError as e:
                                        logger.warning(f"SNAP: White balance modules not available: {e}")
                                    except Exception as e:
                                        logger.warning(f"SNAP: Error loading white balance calibration: {e}")

                                # If white balance was not applied, use the default behavior:
                                # disable per-channel mode and use unified exposure
                                if not wb_applied:
                                    try:
                                        from microscope_control.jai import JAICameraProperties
                                        jai_props = JAICameraProperties(hardware.core)
                                        jai_props.disable_individual_exposure()
                                        jai_props.disable_individual_gain()
                                        # Don't reset analog gains - preserve WB color balance
                                    except (ImportError, Exception):
                                        pass  # Not a JAI camera or module not available

                                    # Set unified exposure (fixed - no adaptive adjustment!)
                                    hardware.set_exposure(exposure_ms)
                                    logger.info(f"Set exposure to {exposure_ms:.2f} ms (FIXED)")
                                else:
                                    # White balance was applied - per-channel exposures are set,
                                    # so we don't call hardware.set_exposure() which would be ignored
                                    # (or potentially interfere with per-channel mode)
                                    logger.debug(f"SNAP: Using per-channel exposures from WB calibration (exposure_ms={exposure_ms:.2f} ignored)")

                                # Set rotation angle
                                if hasattr(hardware, "set_psg_ticks"):
                                    hardware.set_psg_ticks(angle)
                                    logger.info(f"Set rotation angle to {angle:.2f} deg")

                                # Snap image with simple acquisition
                                image, metadata = hardware.snap_image(debayering=debayer)

                                if image is None:
                                    raise RuntimeError("snap_image returned None")

                                # Save the image
                                tifffile.imwrite(
                                    str(output_path),
                                    image,
                                    compression="zlib",
                                    compressionargs={"level": 6},
                                )

                                elapsed = time.time() - snap_start_time
                                logger.info(
                                    f"SNAP complete: {output_path.name}, "
                                    f"angle={angle:.2f}deg, exposure={exposure_ms:.2f}ms, "
                                    f"shape={image.shape}, median={float(image.mean()):.1f}, "
                                    f"total_time={elapsed:.2f}s"
                                )

                                # Send success response
                                response = f"SUCCESS:{output_path}".encode()
                                conn.sendall(response)
                                logger.debug(f"SNAP: sent SUCCESS response")

                            except Exception as e:
                                logger.error(f"SNAP failed: {str(e)}", exc_info=True)
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            break

                        # Safety checks
                        if total_bytes > 10000:
                            logger.error(f"SNAP message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading SNAP message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading SNAP message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in SNAP: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)

                continue

            if data == ExtendedCommand.TESTAF:
                logger.info(f"Client {addr} requested autofocus test")

                # Read the message using the same pattern as BGACQUIRE
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error("Connection closed while reading autofocus test message")
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Split by known flags to avoid issues with spaces in paths
                            flags = ["--yaml", "--output", "--objective"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    # Find where this flag starts
                                    start_idx = message.index(flag) + len(flag)

                                    # Find where the next flag starts (or use end of string)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1 :]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break

                                    # Extract the value and clean it up
                                    value = message[start_idx:end_idx].strip()

                                    # Map to the parameter name
                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--objective":
                                        params["objective"] = value

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path", "objective"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment to prevent client timeout
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for standard autofocus test"
                                )

                                # Execute STANDARD autofocus test
                                from microscope_control.autofocus.test import (
                                    test_standard_autofocus_at_current_position,
                                )

                                result = test_standard_autofocus_at_current_position(
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    yaml_file_path=params["yaml_file_path"],
                                    output_folder_path=params["output_folder_path"],
                                    objective=params["objective"],
                                    logger=logger,
                                )

                                if result["success"]:
                                    # Format result as: SUCCESS:plot_path|initial_z:final_z:z_shift
                                    result_data = f"{result['initial_z']:.2f}:{result['final_z']:.2f}:{result['z_shift']:.2f}"
                                    response = (
                                        f"SUCCESS:{result['plot_path']}|{result_data}".encode()
                                    )
                                    conn.sendall(response)
                                    logger.info(f"Autofocus test completed: {result['message']}")
                                else:
                                    response = f"FAILED:{result['message']}".encode()
                                    conn.sendall(response)
                                    logger.error(f"Autofocus test failed: {result['message']}")

                            except Exception as e:
                                logger.error(f"Autofocus test failed: {str(e)}", exc_info=True)
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            # We found and processed the END_MARKER, so break the while loop
                            break

                        # Safety checks for the while loop
                        if total_bytes > 10000:  # 10KB max
                            logger.error(f"Autofocus test message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading autofocus test message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading autofocus test message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in autofocus test: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode

                continue

            if data == ExtendedCommand.TESTADAF:
                logger.info(f"Client {addr} requested adaptive autofocus test")

                # Read the message using the same pattern as TESTAF
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading adaptive autofocus test message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Split by known flags to avoid issues with spaces in paths
                            flags = ["--yaml", "--output", "--objective"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    # Find where this flag starts
                                    start_idx = message.index(flag) + len(flag)

                                    # Find where the next flag starts (or use end of string)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1 :]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break

                                    # Extract the value and clean it up
                                    value = message[start_idx:end_idx].strip()

                                    # Map to the parameter name
                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--objective":
                                        params["objective"] = value

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path", "objective"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment to prevent client timeout
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for adaptive autofocus test"
                                )

                                # Execute ADAPTIVE autofocus test
                                from microscope_control.autofocus.test import (
                                    test_adaptive_autofocus_at_current_position,
                                )

                                result = test_adaptive_autofocus_at_current_position(
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    yaml_file_path=params["yaml_file_path"],
                                    output_folder_path=params["output_folder_path"],
                                    objective=params["objective"],
                                    logger=logger,
                                )

                                if result["success"]:
                                    # Format result as: SUCCESS:message|initial_z:final_z:z_shift
                                    result_data = f"{result['initial_z']:.2f}:{result['final_z']:.2f}:{result['z_shift']:.2f}"
                                    response = f"SUCCESS:{result['message']}|{result_data}".encode()
                                    conn.sendall(response)
                                    logger.info(
                                        f"Adaptive autofocus test completed: {result['message']}"
                                    )
                                else:
                                    response = f"FAILED:{result['message']}".encode()
                                    conn.sendall(response)
                                    logger.error(
                                        f"Adaptive autofocus test failed: {result['message']}"
                                    )

                            except Exception as e:
                                logger.error(
                                    f"Adaptive autofocus test failed: {str(e)}", exc_info=True
                                )
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            # We found and processed the END_MARKER, so break the while loop
                            break

                        # Safety checks for the while loop
                        if total_bytes > 10000:  # 10KB max
                            logger.error(
                                f"Adaptive autofocus test message too large: {total_bytes} bytes"
                            )
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading adaptive autofocus test message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading adaptive autofocus test message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in adaptive autofocus test: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode

                continue

            if data == ExtendedCommand.AFBENCH:
                logger.info(f"Client {addr} requested autofocus benchmark")

                # Read the message with parameters
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading benchmark message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Split by known flags
                            flags = ["--reference_z", "--output", "--distances", "--quick", "--objective"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1:]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break

                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--reference_z":
                                        params["reference_z"] = float(value)
                                    elif flag == "--output":
                                        params["output_folder"] = value
                                    elif flag == "--distances":
                                        # Parse comma-separated distances
                                        params["test_distances"] = [float(d.strip()) for d in value.split(",")]
                                    elif flag == "--quick":
                                        params["quick_mode"] = value.lower() in ("true", "1", "yes")
                                    elif flag == "--objective":
                                        params["objective"] = value

                            # Validate required parameters
                            required = ["reference_z", "output_folder"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment
                            try:
                                ack_response = f"STARTED:{params['output_folder']}".encode()
                                conn.sendall(ack_response)
                                logger.info("Sent STARTED acknowledgment for autofocus benchmark")

                                # Create progress callback that sends socket updates
                                # This keeps the connection alive during long benchmarks
                                # Format: PROGRESS:current:total:message (consistent with PPMBIREF)
                                def send_progress(current: int, total: int, status_msg: str):
                                    """Send progress update to keep connection alive."""
                                    try:
                                        progress_msg = f"PROGRESS:{current}:{total}:{status_msg}"
                                        conn.sendall(progress_msg.encode())
                                    except Exception as e:
                                        logger.warning(f"Failed to send progress update: {e}")

                                # Execute benchmark with progress callback
                                from microscope_control.autofocus.benchmark import (
                                    run_autofocus_benchmark_from_server,
                                )

                                result = run_autofocus_benchmark_from_server(
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    reference_z=params["reference_z"],
                                    output_folder=params["output_folder"],
                                    test_distances=params.get("test_distances"),
                                    quick_mode=params.get("quick_mode", False),
                                    objective=params.get("objective"),
                                    logger=logger,
                                    progress_callback=send_progress,
                                )

                                # Check for safety violation
                                if result.get("safety_violation"):
                                    error_msg = result.get("error", "Safety limit exceeded")
                                    response = f"FAILED:SAFETY:{error_msg}".encode()
                                    conn.sendall(response)
                                    logger.error(f"Autofocus benchmark SAFETY VIOLATION: {error_msg}")
                                else:
                                    # Format response
                                    success_rate = result.get("success_rate", 0)
                                    total_trials = result.get("total_trials", 0)
                                    results_dir = result.get("results_directory", "")

                                    response = f"SUCCESS:Benchmark complete. {total_trials} trials, {success_rate:.1%} success rate|{results_dir}".encode()
                                    conn.sendall(response)
                                    logger.info(f"Autofocus benchmark completed: {total_trials} trials")

                            except Exception as e:
                                logger.error(
                                    f"Autofocus benchmark failed: {str(e)}", exc_info=True
                                )
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            break

                        # Safety checks
                        if total_bytes > 10000:
                            logger.error(f"Benchmark message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading benchmark message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading benchmark message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in autofocus benchmark: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)

                continue

            if data == ExtendedCommand.POLCAL:
                logger.info(f"Client {addr} requested polarizer calibration")

                # Read the message using the same pattern as BGACQUIRE
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading polarizer calibration message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse the message
                            params = {}

                            # Split by known flags
                            flags = [
                                "--yaml",
                                "--output",
                                "--start",
                                "--end",
                                "--step",
                                "--exposure",
                            ]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    # Find where this flag starts
                                    start_idx = message.index(flag) + len(flag)

                                    # Find where the next flag starts (or use end of string)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1 :]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break

                                    # Extract the value and clean it up
                                    value = message[start_idx:end_idx].strip()

                                    # Map to the parameter name
                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--start":
                                        params["start_angle"] = float(value)
                                    elif flag == "--end":
                                        params["end_angle"] = float(value)
                                    elif flag == "--step":
                                        params["step_size"] = float(value)
                                    elif flag == "--exposure":
                                        params["exposure_ms"] = float(value)

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Set defaults for optional parameters
                            params.setdefault("start_angle", 0.0)
                            params.setdefault("end_angle", 360.0)
                            params.setdefault("step_size", 5.0)
                            params.setdefault("exposure_ms", 10.0)

                            # Send immediate acknowledgment to prevent client timeout
                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info("Sent STARTED acknowledgment for polarizer calibration")

                                # Execute polarizer calibration workflow
                                from microscope_command_server.acquisition.workflow import (
                                    polarizer_calibration_workflow,
                                )

                                report_path = polarizer_calibration_workflow(
                                    yaml_file_path=params["yaml_file_path"],
                                    output_folder_path=params["output_folder_path"],
                                    start_angle=params["start_angle"],
                                    end_angle=params["end_angle"],
                                    step_size=params["step_size"],
                                    exposure_ms=params["exposure_ms"],
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    logger=logger,
                                )

                                # Send success response with report path
                                response = f"SUCCESS:{report_path}".encode()
                                conn.sendall(response)

                                logger.info(
                                    f"Polarizer calibration completed. Report: {report_path}"
                                )

                            except Exception as e:
                                logger.error(
                                    f"Polarizer calibration failed: {str(e)}", exc_info=True
                                )
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            # We found and processed the END_MARKER, so break the while loop
                            break

                        # Safety checks for the while loop
                        if total_bytes > 10000:  # 10KB max
                            logger.error(
                                f"Polarizer calibration message too large: {total_bytes} bytes"
                            )
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading polarizer calibration message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading polarizer calibration message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in polarizer calibration: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode

                continue

            # ============ PPM TESTING COMMANDS ============

            if data == ExtendedCommand.PPMSENS:
                logger.info(f"Client {addr} requested PPM rotation sensitivity test")

                # Read the message with parameters
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error("Connection closed while reading PPMSENS message")
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse parameters
                            params = {}
                            flags = ["--yaml", "--output", "--test-type", "--base-angle", "--repeats"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1:]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break
                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--test-type":
                                        params["test_type"] = value
                                    elif flag == "--base-angle":
                                        params["base_angle"] = float(value)
                                    elif flag == "--repeats":
                                        params["n_repeats"] = int(value)

                            # Set defaults
                            params.setdefault("test_type", "repeatability")
                            params.setdefault("base_angle", 7.0)
                            params.setdefault("n_repeats", 10)

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info("Sent STARTED acknowledgment for PPM sensitivity test")

                                # Delegate to PPM modality handler
                                from microscope_command_server.modality.ppm import handle_sensitivity_test

                                result_dir = handle_sensitivity_test(
                                    params=params,
                                    port=PORT,
                                    _logger=logger,
                                )

                                if result_dir:
                                    response = f"SUCCESS:{result_dir}".encode()
                                    conn.sendall(response)
                                    logger.info(f"PPM sensitivity test completed: {result_dir}")
                                else:
                                    response = b"FAILED:Test did not complete successfully"
                                    conn.sendall(response)
                                    logger.error("PPM sensitivity test failed")

                            except ImportError as e:
                                logger.error(f"PPM sensitivity test module not available: {e}")
                                response = f"FAILED:Module not available - {e}".encode()
                                conn.sendall(response)
                            except Exception as e:
                                logger.error(f"PPM sensitivity test failed: {str(e)}", exc_info=True)
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            break

                        # Safety checks
                        if total_bytes > 10000:
                            logger.error(f"PPMSENS message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading PPMSENS message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading PPMSENS message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in PPMSENS: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)

                continue

            if data == ExtendedCommand.PPMBIREF:
                logger.info(f"Client {addr} requested PPM birefringence maximization test")

                # Read the message with parameters
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error("Connection closed while reading PPMBIREF message")
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse parameters
                            params = {}
                            flags = ["--yaml", "--output", "--mode", "--min-angle", "--max-angle",
                                    "--step", "--exposure", "--target-intensity"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1:]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break
                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--mode":
                                        params["exposure_mode"] = value
                                    elif flag == "--min-angle":
                                        params["min_angle"] = float(value)
                                    elif flag == "--max-angle":
                                        params["max_angle"] = float(value)
                                    elif flag == "--step":
                                        params["angle_step"] = float(value)
                                    elif flag == "--exposure":
                                        params["fixed_exposure_ms"] = float(value)
                                    elif flag == "--target-intensity":
                                        params["target_intensity"] = int(value)

                            # Set defaults
                            params.setdefault("exposure_mode", "interpolate")
                            params.setdefault("min_angle", -10.0)
                            params.setdefault("max_angle", 10.0)
                            params.setdefault("angle_step", 0.5)  # Coarser default for server
                            params.setdefault("target_intensity", 128)

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Validate fixed mode requires exposure
                            if params["exposure_mode"] == "fixed" and "fixed_exposure_ms" not in params:
                                error_msg = "fixed_exposure_ms required when mode=fixed"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info("Sent STARTED acknowledgment for PPM birefringence test")

                                # Create progress callback to send updates through socket
                                def send_progress(current: int, total: int):
                                    """Send progress update through socket."""
                                    try:
                                        progress_msg = f"PROGRESS:{current}:{total}".encode()
                                        conn.sendall(progress_msg)
                                        logger.debug(f"Sent progress: {current}/{total}")
                                    except Exception as e:
                                        logger.warning(f"Failed to send progress: {e}")

                                # Create stage move callback for calibrate mode
                                def stage_move_callback() -> bool:
                                    """
                                    Send STAGEMOVE message and wait for CONTINUE/ABORT response.
                                    Returns True if user confirmed, False if aborted.
                                    """
                                    try:
                                        # Send stage move request
                                        conn.sendall(b"STAGEMOVE:Background calibration complete. Move stage to tissue.")
                                        logger.info("Sent STAGEMOVE request, waiting for user confirmation...")

                                        # Wait indefinitely for user response (no timeout)
                                        # User may need significant time to find tissue and position stage
                                        conn.settimeout(None)  # No timeout - wait indefinitely
                                        response = conn.recv(1024).decode().strip()
                                        conn.settimeout(30.0)  # Restore normal timeout

                                        if response == "CONTINUE":
                                            logger.info("User confirmed stage move, continuing...")
                                            return True
                                        else:
                                            logger.info(f"User response: {response}, aborting...")
                                            return False
                                    except Exception as e:
                                        logger.error(f"Stage move callback failed: {e}")
                                        return False

                                # Delegate to PPM modality handler
                                from microscope_command_server.modality.ppm import handle_birefringence_test

                                result_dir = handle_birefringence_test(
                                    params=params,
                                    port=PORT,
                                    progress_callback=send_progress,
                                    stage_move_callback=stage_move_callback,
                                    _logger=logger,
                                )

                                if result_dir:
                                    response = f"SUCCESS:{result_dir}".encode()
                                    conn.sendall(response)
                                    logger.info(f"PPM birefringence test completed: {result_dir}")
                                else:
                                    response = b"FAILED:Test did not complete successfully"
                                    conn.sendall(response)
                                    logger.error("PPM birefringence test failed")

                            except ImportError as e:
                                logger.error(f"PPM birefringence test module not available: {e}")
                                response = f"FAILED:Module not available - {e}".encode()
                                conn.sendall(response)
                            except Exception as e:
                                logger.error(f"PPM birefringence test failed: {str(e)}", exc_info=True)
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            break

                        # Safety checks
                        if total_bytes > 10000:
                            logger.error(f"PPMBIREF message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading PPMBIREF message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading PPMBIREF message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in PPMBIREF: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)

                continue

            # ==================== SBCALIB - Sunburst Calibration ====================
            if data == ExtendedCommand.SBCALIB:
                logger.info(f"Client {addr} requested sunburst calibration")

                # Read the message with parameters
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error("Connection closed while reading SBCALIB message")
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()

                            # Parse parameters
                            params = {}
                            flags = ["--yaml", "--output", "--modality", "--spokes",
                                    "--saturation", "--value", "--name",
                                    "--radius_inner", "--radius_outer",
                                    "--image_path", "--center_y", "--center_x"]

                            for i, flag in enumerate(flags):
                                if flag in message:
                                    start_idx = message.index(flag) + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags[i + 1:]:
                                        if next_flag in message[start_idx:]:
                                            next_pos = message.index(next_flag, start_idx)
                                            if next_pos < end_idx:
                                                end_idx = next_pos
                                                break
                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--yaml":
                                        params["yaml_file_path"] = value
                                    elif flag == "--output":
                                        params["output_folder_path"] = value
                                    elif flag == "--modality":
                                        params["modality"] = value
                                    elif flag == "--spokes":
                                        params["expected_spokes"] = int(value)
                                    elif flag == "--saturation":
                                        params["saturation_threshold"] = float(value)
                                    elif flag == "--value":
                                        params["value_threshold"] = float(value)
                                    elif flag == "--name":
                                        params["calibration_name"] = value
                                    elif flag == "--radius_inner":
                                        params["radius_inner"] = int(value)
                                    elif flag == "--radius_outer":
                                        params["radius_outer"] = int(value)
                                    elif flag == "--image_path":
                                        params["image_path"] = value
                                    elif flag == "--center_y":
                                        params["center_y"] = int(value)
                                    elif flag == "--center_x":
                                        params["center_x"] = int(value)

                            # Set defaults
                            # Sunburst calibration is currently PPM-specific
                            params.setdefault("modality", "ppm_20x")
                            params.setdefault("expected_spokes", 16)
                            params.setdefault("saturation_threshold", 0.1)
                            params.setdefault("value_threshold", 0.1)
                            params.setdefault("calibration_name", None)
                            params.setdefault("radius_inner", 30)
                            params.setdefault("radius_outer", 150)
                            params.setdefault("image_path", None)
                            params.setdefault("center_y", None)
                            params.setdefault("center_x", None)

                            # Validate required parameters
                            required = ["yaml_file_path", "output_folder_path"]
                            missing = [key for key in required if key not in params]
                            if missing:
                                error_msg = f"Missing required parameters: {missing}"
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            try:
                                ack_response = f"STARTED:{params['output_folder_path']}".encode()
                                conn.sendall(ack_response)
                                logger.info("Sent STARTED acknowledgment for sunburst calibration")

                                # Run sunburst calibration workflow
                                from microscope_command_server.calibration.sunburst_workflow import (
                                    run_sunburst_calibration,
                                )

                                # Build center tuple if both coordinates provided
                                center = None
                                if params["center_y"] is not None and params["center_x"] is not None:
                                    center = (params["center_y"], params["center_x"])

                                result = run_sunburst_calibration(
                                    hardware=hardware,
                                    config_manager=config_manager,
                                    output_folder=params["output_folder_path"],
                                    modality=params["modality"],
                                    expected_spokes=params["expected_spokes"],
                                    saturation_threshold=params["saturation_threshold"],
                                    value_threshold=params["value_threshold"],
                                    calibration_name=params["calibration_name"],
                                    radius_inner=params["radius_inner"],
                                    radius_outer=params["radius_outer"],
                                    logger=logger,
                                    existing_image_path=params["image_path"],
                                    center=center,
                                )

                                # Send result as JSON (always SUCCESS: prefix with
                                # full JSON so client gets image_path even on failure)
                                import json
                                result_json = json.dumps(result)
                                response = f"SUCCESS:{result_json}".encode()
                                conn.sendall(response)
                                if result.get("success"):
                                    logger.info(f"Sunburst calibration successful. R^2={result.get('r_squared', 0):.4f}")
                                else:
                                    logger.error(f"Sunburst calibration failed: {result.get('error', 'Unknown')}")

                            except ImportError as e:
                                logger.error(f"Module not available: {e}")
                                response = f"FAILED:Module not available - {e}".encode()
                                conn.sendall(response)
                            except Exception as e:
                                logger.error(f"Sunburst calibration failed: {str(e)}", exc_info=True)
                                response = f"FAILED:{str(e)}".encode()
                                conn.sendall(response)

                            break

                        # Safety checks
                        if total_bytes > 10000:
                            logger.error(f"SBCALIB message too large: {total_bytes} bytes")
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading SBCALIB message")
                            conn.sendall(b"FAILED:Timeout waiting for complete message")
                            break

                except socket.timeout:
                    logger.error(f"Timeout reading SBCALIB message from {addr}")
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(f"Error in SBCALIB: {str(e)}", exc_info=True)
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)

                continue

            # ==================== Camera Control Commands ====================

            # GETCAM - Get camera name from Core
            if data == ExtendedCommand.GETCAM:
                logger.debug(f"Client {addr} requested camera name")
                try:
                    camera_name = hardware.core.get_property("Core", "Camera")
                    # Pad or truncate to 32 bytes
                    camera_name_bytes = camera_name.encode("utf-8")[:32].ljust(32, b"\x00")
                    conn.sendall(camera_name_bytes)
                    logger.info(f"Sent camera name to {addr}: {camera_name}")
                except Exception as e:
                    logger.error(f"Failed to get camera name: {e}")
                    # Send error response (32 bytes, starts with ERROR)
                    error_msg = f"ERROR:{str(e)[:23]}"
                    conn.sendall(error_msg.encode("utf-8").ljust(32, b"\x00"))
                continue

            # GETMODE - Get exposure/gain mode flags (individual vs unified)
            # Gain is always reported as unified (0) since individual gain mode
            # is no longer used. R/B analog gains work in unified mode.
            if data == ExtendedCommand.GETMODE:
                logger.debug(f"Client {addr} requested camera mode flags")
                try:
                    from microscope_control.jai import JAICameraProperties
                    jai_props = JAICameraProperties(hardware.core)

                    if jai_props.validate_camera():
                        exp_individual = jai_props.is_individual_exposure_enabled()
                        # Gain is always unified in new model
                        mode_str = f"JAI_EXP:{1 if exp_individual else 0}_GAIN:0"
                        conn.sendall(mode_str.encode("utf-8").ljust(16, b"\x00"))
                        logger.info(f"Sent JAI mode flags: exp_ind={exp_individual}, gain_ind=false")
                    else:
                        conn.sendall(b"UNIFIED_________")
                        logger.info("Non-JAI camera - sent UNIFIED mode")
                except ImportError:
                    conn.sendall(b"UNIFIED_________")
                    logger.info("JAI module not available - sent UNIFIED mode")
                except Exception as e:
                    logger.error(f"Failed to get camera mode: {e}")
                    error_msg = f"ERROR:{str(e)[:8]}"
                    conn.sendall(error_msg.encode("utf-8").ljust(16, b"\x00"))
                continue

            # SETMODE - Set exposure/gain mode flags
            # JAI-SPECIFIC: Sets exposure mode (individual or unified).
            # Gain mode byte is accepted but ignored - gain is always unified.
            # R/B analog gains work in unified mode via set_rb_analog_gains().
            #
            # Protocol: 8-byte command + 2 bytes [exp_mode, gain_mode]
            #   exp_mode:  1 = individual (R,G,B separate), 0 = unified
            #   gain_mode: ignored (always unified), logged if True requested
            #
            # Response: "ACK_____" on success, "ERR_NJAI" if not JAI, "ERR_MODE" on failure.
            if data == ExtendedCommand.SETMODE:
                logger.debug(f"Client {addr} requested to set camera mode")
                try:
                    # Read 2 bytes: [exp_mode, gain_mode]
                    mode_bytes = conn.recv(2)
                    if len(mode_bytes) != 2:
                        raise ValueError("Expected 2 bytes for mode flags")

                    exp_individual = mode_bytes[0] == 1
                    gain_individual = mode_bytes[1] == 1

                    if gain_individual:
                        logger.warning(
                            "Individual gain mode requested but ignored - "
                            "gain is always unified. Use R/B analog gains instead."
                        )

                    logger.info(f"Setting mode: exp_individual={exp_individual}, gain_individual=false(forced)")

                    # Safety net: stop any active streaming before changing camera properties.
                    # JAI cameras cannot change ExposureIsIndividual while hardware is busy.
                    stopped_sequence = False
                    stopped_studio_live = False
                    try:
                        if hardware.core.is_sequence_running():
                            logger.warning("Core sequence running during SETMODE - auto-stopping")
                            hardware.core.stop_sequence_acquisition()
                            stopped_sequence = True
                            time.sleep(0.2)
                    except Exception as seq_err:
                        logger.debug(f"Could not check/stop sequence: {seq_err}")
                    try:
                        if hardware.studio is not None and hardware.studio.live().is_live_mode_on():
                            logger.warning("MM Studio live mode on during SETMODE - auto-stopping")
                            hardware.studio.live().set_live_mode(False)
                            stopped_studio_live = True
                            time.sleep(0.2)
                    except Exception as live_err:
                        logger.debug(f"Could not check/stop studio live: {live_err}")

                    from microscope_control.jai import JAICameraProperties
                    jai_props = JAICameraProperties(hardware.core)

                    if not jai_props.validate_camera():
                        raise RuntimeError("JAI camera not active - cannot set individual mode")

                    if exp_individual:
                        jai_props.enable_individual_exposure()
                    else:
                        jai_props.disable_individual_exposure()

                    # Always ensure gain is unified
                    jai_props.disable_individual_gain()

                    conn.sendall(b"ACK_____")
                    if stopped_sequence or stopped_studio_live:
                        logger.info("Camera mode set successfully (auto-stopped streaming first)")
                    else:
                        logger.info("Camera mode set successfully")
                except ImportError:
                    conn.sendall(b"ERR_NJAI")
                    logger.error("JAI module not available")
                except Exception as e:
                    logger.error(f"Failed to set camera mode: {e}")
                    conn.sendall(b"ERR_MODE")
                continue

            # GETEXP - Get exposure values (unified or per-channel RGB)
            if data == ExtendedCommand.GETEXP:
                logger.debug(f"Client {addr} requested exposure values")
                try:
                    from microscope_control.jai import JAICameraProperties
                    jai_props = JAICameraProperties(hardware.core)

                    if jai_props.validate_camera() and jai_props.is_individual_exposure_enabled():
                        # JAI with individual exposures - return 4 floats (all, R, G, B)
                        exposures = jai_props.get_channel_exposures()
                        # Get unified exposure as well for "all" value
                        all_exp = hardware.core.get_exposure()
                        response = struct.pack("!ffff",
                            float(all_exp),
                            float(exposures["red"]),
                            float(exposures["green"]),
                            float(exposures["blue"]))
                        conn.sendall(response)
                        logger.info(f"Sent per-channel exposures: all={all_exp}, R={exposures['red']}, G={exposures['green']}, B={exposures['blue']}")
                    else:
                        # Unified exposure - return 1 float
                        exposure = hardware.core.get_exposure()
                        response = struct.pack("!f", float(exposure))
                        conn.sendall(response)
                        logger.info(f"Sent unified exposure: {exposure}")
                except ImportError:
                    # JAI module not available - get unified exposure
                    exposure = hardware.core.get_exposure()
                    response = struct.pack("!f", float(exposure))
                    conn.sendall(response)
                    logger.info(f"Sent unified exposure (no JAI): {exposure}")
                except Exception as e:
                    logger.error(f"Failed to get exposure: {e}")
                    # Send error as negative value
                    conn.sendall(struct.pack("!f", -1.0))
                continue

            # SETEXP - Set exposure values
            # MIXED: count=1 is GENERIC (calls hardware.set_exposure for any camera),
            # count>=3 is JAI-SPECIFIC (sets per-channel R,G,B exposures via
            # JAICameraProperties.set_channel_exposures with auto_enable=True,
            # which implicitly enables individual exposure mode).
            #
            # Protocol: 8-byte command + 1 count byte + (count * 4) bytes of
            #   big-endian floats (exposure values in ms).
            #
            # Response: "ACK_____" on success, "ERR_NJAI" if JAI module unavailable
            #   for per-channel, "ERR_EXPO" on other failure.
            if data == ExtendedCommand.SETEXP:
                logger.debug(f"Client {addr} requested to set exposure")
                try:
                    # Read count byte first
                    count_byte = conn.recv(1)
                    count = count_byte[0]
                    logger.debug(f"SETEXP: expecting {count} exposure values")

                    # Read float values
                    float_data = conn.recv(count * 4)
                    if len(float_data) != count * 4:
                        raise ValueError(f"Expected {count * 4} bytes, got {len(float_data)}")

                    exposures = struct.unpack(f"!{'f' * count}", float_data)
                    logger.info(f"Setting exposures: {exposures}")

                    if count == 1:
                        # Unified exposure
                        hardware.set_exposure(exposures[0])
                        logger.info(f"Set unified exposure to {exposures[0]} ms")
                    elif count >= 3:
                        # Per-channel exposures (R, G, B)
                        from microscope_control.jai import JAICameraProperties
                        jai_props = JAICameraProperties(hardware.core)
                        jai_props.set_channel_exposures(
                            red=exposures[0],
                            green=exposures[1],
                            blue=exposures[2],
                            auto_enable=True
                        )
                        logger.info(f"Set per-channel exposures: R={exposures[0]}, G={exposures[1]}, B={exposures[2]}")

                    conn.sendall(b"ACK_____")
                except ImportError:
                    conn.sendall(b"ERR_NJAI")
                    logger.error("JAI module not available for per-channel exposure")
                except Exception as e:
                    logger.error(f"Failed to set exposure: {e}")
                    conn.sendall(b"ERR_EXPO")
                continue

            # GETGAIN - Get gain values
            # Always returns 3 floats: [unified_gain, analog_red, analog_blue]
            if data == ExtendedCommand.GETGAIN:
                logger.debug(f"Client {addr} requested gain values")
                try:
                    from microscope_control.jai import JAICameraProperties
                    jai_props = JAICameraProperties(hardware.core)

                    if jai_props.validate_camera():
                        unified = jai_props.get_unified_gain()
                        rb_gains = jai_props.get_rb_analog_gains()
                        response = struct.pack("!fff",
                            float(unified),
                            float(rb_gains["red"]),
                            float(rb_gains["blue"]))
                        conn.sendall(response)
                        logger.info(
                            f"Sent gains: unified={unified}, "
                            f"analog_red={rb_gains['red']}, analog_blue={rb_gains['blue']}"
                        )
                    else:
                        # Not JAI - return defaults
                        response = struct.pack("!fff", 1.0, 1.0, 1.0)
                        conn.sendall(response)
                        logger.info("Non-JAI camera - sent default gains (1.0, 1.0, 1.0)")
                except ImportError:
                    response = struct.pack("!fff", 1.0, 1.0, 1.0)
                    conn.sendall(response)
                    logger.info("JAI module not available - sent default gains")
                except Exception as e:
                    logger.error(f"Failed to get gain: {e}")
                    conn.sendall(struct.pack("!fff", -1.0, -1.0, -1.0))
                continue

            # SETGAIN - Set gain values
            # JAI-SPECIFIC (both paths):
            #   count=1: Sets unified gain via set_unified_gain (range 1.0-8.0)
            #   count=3: Sets [unified_gain, analog_red, analog_blue]
            #            - unified gain applied to all channels
            #            - analog_red/blue applied via set_rb_analog_gains (0.47-4.0)
            #            - Does NOT enable individual gain mode
            #
            # Protocol: 8-byte command + 1 count byte + (count * 4) bytes floats.
            # Response: "ACK_____", "ERR_NJAI", or "ERR_GAIN".
            if data == ExtendedCommand.SETGAIN:
                logger.debug(f"Client {addr} requested to set gain")
                try:
                    # Read count byte first
                    count_byte = conn.recv(1)
                    count = count_byte[0]
                    logger.debug(f"SETGAIN: expecting {count} gain values")

                    # Read float values
                    float_data = conn.recv(count * 4)
                    if len(float_data) != count * 4:
                        raise ValueError(f"Expected {count * 4} bytes, got {len(float_data)}")

                    gains = struct.unpack(f"!{'f' * count}", float_data)
                    logger.info(f"Setting gains: {gains}")

                    from microscope_control.jai import JAICameraProperties
                    jai_props = JAICameraProperties(hardware.core)

                    # Stop any active streaming before changing gain properties
                    # (same pattern as SETMODE handler)
                    try:
                        if hardware.core.is_sequence_running():
                            hardware.core.stop_sequence_acquisition()
                            time.sleep(0.2)
                    except Exception:
                        pass

                    if count == 1:
                        # Unified gain only
                        jai_props.set_unified_gain(gains[0])
                        logger.info(f"Set unified gain: {gains[0]}")
                    elif count >= 3:
                        # New semantics: [unified_gain, analog_red, analog_blue]
                        jai_props.set_unified_gain(gains[0])
                        jai_props.set_rb_analog_gains(red=gains[1], blue=gains[2])
                        logger.info(
                            f"Set gains: unified={gains[0]}, "
                            f"analog_red={gains[1]}, analog_blue={gains[2]}"
                        )

                    conn.sendall(b"ACK_____")
                except ImportError:
                    conn.sendall(b"ERR_NJAI")
                    logger.error("JAI module not available for gain control")
                except Exception as e:
                    logger.error(f"Failed to set gain: {e}")
                    conn.sendall(b"ERR_GAIN")
                continue

            # ==================== White Balance Mode Control ====================

            # SETWBMD - Set camera white balance mode (0=Off, 1=Continuous, 2=Once)
            # JAI-SPECIFIC: Controls the camera's built-in hardware auto white
            # balance feature. This is SEPARATE from the calibration commands
            # (WBSIMPLE/WBPPM) which manually compute and apply per-channel
            # exposure/gain values.
            #
            # WARNING: Hardware auto-WB (Continuous/Once) adjusts internal
            # camera parameters that are NOT saved or reproducible. For
            # reproducible scientific imaging, use WBSIMPLE/WBPPM calibration
            # instead and keep hardware WB mode set to Off.
            #
            # Protocol: 8-byte command + 1 byte mode value.
            #   0 = Off (disable auto WB)
            #   1 = Continuous (camera auto-adjusts WB every frame)
            #   2 = Once (camera runs single auto-WB then stops)
            #
            # Response: "ACK_____", "ERR_NJAI", or "ERR_WBMD".
            if data == ExtendedCommand.SETWBMD:
                logger.debug(f"Client {addr} requested to set WB mode")
                try:
                    mode_byte = conn.recv(1)
                    mode = mode_byte[0]
                    logger.info(f"Setting WB mode: {mode}")

                    global awb_calibrated
                    from microscope_control.jai import JAICameraProperties
                    jai_props = JAICameraProperties(hardware.core)

                    if mode == 0:
                        # Set Off WITHOUT wait_for_device (wait=False).
                        # The JAI camera's wait_for_device clears internal AWB
                        # corrections accumulated during Continuous mode.
                        # MicroManager's GUI does not call wait_for_device,
                        # which is why AWB corrections persist via MM but not
                        # via our code. Using wait=False preserves corrections.
                        jai_props._set_property(
                            jai_props.WHITE_BALANCE, "Off", wait=False
                        )
                        # Note: does NOT clear analog gain corrections.
                        # awb_calibrated stays True if AWB was previously run.
                        logger.info("Set white balance mode to Off (AWB corrections preserved)")
                    elif mode == 1:
                        jai_props.set_white_balance_mode("Continuous")
                        awb_calibrated = True
                        logger.info("Set white balance mode to Continuous (AWB active)")
                    elif mode == 2:
                        # Set the camera's native "Once" mode directly.
                        # The camera runs a single AWB calibration and then
                        # auto-returns to Off. This is the simple property set
                        # used by the Camera Control UI.
                        # NOTE: run_auto_white_balance() (mode 3) is a separate
                        # full calibration routine used by automated workflows.
                        jai_props._set_property(
                            jai_props.WHITE_BALANCE, "Once", wait=False
                        )
                        awb_calibrated = True
                        logger.info("Set white balance mode to Once (native one-shot AWB)")
                    elif mode == 3:
                        # Full AWB calibration routine: starts streaming, sets
                        # Continuous, drains buffer for 3s, then sets Off.
                        # Used by automated workflows (WB Comparison Test).
                        jai_props.run_auto_white_balance()
                        awb_calibrated = True
                        logger.info("Ran AWB Continuous calibration (internal corrections active)")
                    else:
                        logger.warning(f"Unknown WB mode: {mode}")

                    conn.sendall(b"ACK_____")
                except ImportError:
                    conn.sendall(b"ERR_NJAI")
                    logger.error("JAI module not available for WB mode control")
                except Exception as e:
                    logger.error(f"Failed to set WB mode: {e}")
                    conn.sendall(b"ERR_WBMD")
                continue

            # ==================== Live Mode Control Commands ====================

            # GETLIVE - Check if live mode is currently running
            if data == ExtendedCommand.GETLIVE:
                logger.debug(f"Client {addr} requested live mode status")
                try:
                    is_live = False
                    # Check if sequence is running (indicates live mode)
                    if hardware.core.is_sequence_running():
                        is_live = True
                    # Also check via studio if available
                    elif hardware.studio is not None:
                        try:
                            is_live = hardware.studio.live().is_live_mode_on()
                        except Exception:
                            pass  # Fall back to is_sequence_running result

                    # Response: 1 byte (0 = not live, 1 = live)
                    conn.sendall(bytes([1 if is_live else 0]))
                    logger.info(f"Live mode status: {'ON' if is_live else 'OFF'}")
                except Exception as e:
                    logger.error(f"Failed to get live mode status: {e}")
                    conn.sendall(bytes([0]))  # Default to not live on error
                continue

            # SETLIVE - Set live mode on or off
            # When turning OFF, also stops core-level sequence acquisition so that
            # SETLIVE OFF is comprehensive (matches what GETLIVE reports).
            if data == ExtendedCommand.SETLIVE:
                logger.debug(f"Client {addr} requested to set live mode")
                try:
                    # Read 1 byte: 0 = off, 1 = on
                    enable_byte = conn.recv(1)
                    if len(enable_byte) != 1:
                        raise ValueError("Expected 1 byte for live mode flag")

                    enable_live = enable_byte[0] == 1
                    logger.info(f"Setting live mode: {'ON' if enable_live else 'OFF'}")

                    if not enable_live:
                        # Also stop core-level sequence acquisition (QPSC Live Viewer uses this)
                        try:
                            if hardware.core.is_sequence_running():
                                hardware.core.stop_sequence_acquisition()
                                logger.info("Stopped core sequence acquisition via SETLIVE OFF")
                        except Exception as seq_err:
                            logger.debug(f"Could not stop core sequence: {seq_err}")

                    if hardware.studio is not None:
                        hardware.studio.live().set_live_mode(enable_live)
                        conn.sendall(b"ACK_____")
                        logger.info(f"Live mode set to {'ON' if enable_live else 'OFF'}")
                    else:
                        # No studio available - cannot control live mode
                        conn.sendall(b"ERR_NSTD")
                        logger.warning("No studio available to control live mode")
                except Exception as e:
                    logger.error(f"Failed to set live mode: {e}")
                    conn.sendall(b"ERR_LIVE")
                continue

            # ==================== Live Viewer Commands ====================

            # GETFRAME - Get latest frame from MM circular buffer (for live viewer)
            if data == ExtendedCommand.GETFRAME:
                try:
                    image, meta = hardware.get_live_frame()
                    if image is None:
                        # No frame available - send zero header
                        conn.sendall(struct.pack(">5i", 0, 0, 0, 0, 0))
                        continue

                    h, w = image.shape[:2]
                    channels = 1 if image.ndim == 2 else image.shape[2]
                    bpp = image.dtype.itemsize

                    # Convert uint16 to big-endian for wire transfer
                    if image.dtype == np.uint16:
                        image = image.astype(">u2")

                    raw_bytes = np.ascontiguousarray(image).tobytes()
                    header = struct.pack(">5i", w, h, channels, bpp, len(raw_bytes))
                    conn.sendall(header + raw_bytes)
                except Exception as e:
                    logger.error(f"GETFRAME failed: {e}")
                    try:
                        conn.sendall(struct.pack(">5i", 0, 0, 0, 0, 0))
                    except Exception:
                        pass
                continue

            # STRTSEQ - Start continuous sequence acquisition (core-level, bypasses MM live window)
            if data == ExtendedCommand.STRTSEQ:
                logger.info(f"Client {addr} requested start continuous acquisition")
                try:
                    hardware.start_continuous_acquisition()
                    conn.sendall(b"ACK_____")
                    logger.info("Continuous sequence acquisition started")
                except Exception as e:
                    logger.error(f"Failed to start continuous acquisition: {e}")
                    conn.sendall(b"ERR_SEQ_")
                continue

            # STOPSEQ - Stop continuous sequence acquisition (core-level)
            if data == ExtendedCommand.STOPSEQ:
                logger.info(f"Client {addr} requested stop continuous acquisition")
                try:
                    hardware.stop_continuous_acquisition()
                    conn.sendall(b"ACK_____")
                    logger.info("Continuous sequence acquisition stopped")
                except Exception as e:
                    logger.error(f"Failed to stop continuous acquisition: {e}")
                    conn.sendall(b"ERR_SEQ_")
                continue

            # GETNOISE - Get per-channel noise statistics via multi-frame analysis
            # Protocol: 8-byte command + 1 byte (num_frames, 0 = default 10)
            # Response: 9 big-endian floats:
            #   R_mean, G_mean, B_mean, R_std, G_std, B_std, R_snr, G_snr, B_snr
            if data == ExtendedCommand.GETNOISE:
                logger.info(f"Client {addr} requested noise measurement")
                try:
                    # Read 1 byte for num_frames (0 = default 10)
                    nf_byte = conn.recv(1)
                    num_frames = nf_byte[0] if nf_byte and nf_byte[0] > 0 else 10

                    from microscope_control.jai import JAINoiseMeasurement
                    noise_meter = JAINoiseMeasurement(hardware)
                    stats = noise_meter.measure_noise(
                        num_frames=num_frames, settle_frames=2
                    )

                    # Pack 9 floats: means (R,G,B), stddevs (R,G,B), SNRs (R,G,B)
                    response = struct.pack(
                        "!fffffffff",
                        float(stats.channel_means["red"]),
                        float(stats.channel_means["green"]),
                        float(stats.channel_means["blue"]),
                        float(stats.channel_stddevs["red"]),
                        float(stats.channel_stddevs["green"]),
                        float(stats.channel_stddevs["blue"]),
                        float(stats.channel_snr["red"]),
                        float(stats.channel_snr["green"]),
                        float(stats.channel_snr["blue"]),
                    )
                    conn.sendall(response)
                    logger.info(
                        f"Noise stats sent: R_snr={stats.channel_snr['red']:.1f}, "
                        f"G_snr={stats.channel_snr['green']:.1f}, "
                        f"B_snr={stats.channel_snr['blue']:.1f}"
                    )
                except ImportError as e:
                    logger.error(f"Noise measurement module not available: {e}")
                    # Send 9 zeros on error
                    conn.sendall(struct.pack("!fffffffff", *([0.0] * 9)))
                except Exception as e:
                    logger.error(f"Noise measurement failed: {e}", exc_info=True)
                    conn.sendall(struct.pack("!fffffffff", *([0.0] * 9)))
                continue

            # ==================== NOISCHAR: JAI Noise Characterization ====================
            # JAI-SPECIFIC: Systematic noise characterization across a grid of gain and
            # exposure settings. Tests multiple combinations to find optimal SNR.
            #
            # Protocol: 8-byte command, then variable-length text message with
            #   flag-based parameters (--output, --preset, --frames, --plots,
            #   --gains, --exposures)
            #
            # Response sequence:
            #   1. Immediately sends "STARTED:{output_path}" acknowledgment
            #   2. Sends "PROGRESS:{current}:{total}" after each configuration
            #   3. Sends "SUCCESS:{path}|{totalConfigs}|{plots}|{bestGain},{bestExp}"
            #      or "FAILED:{reason}" on completion
            #
            # Post-characterization cleanup (in finally block):
            #   Resets camera to unified gain 1.0, analog gains 1.0, disables
            #   individual exposure mode.
            if data == ExtendedCommand.NOISCHAR:
                logger.info(f"Client {addr} requested JAI noise characterization")

                # Read the message using chunked pattern (same as WBSIMPLE)
                message_parts = []
                total_bytes = 0
                start_time = time.time()

                conn.settimeout(5.0)

                try:
                    while True:
                        chunk = conn.recv(1024)
                        if not chunk:
                            logger.error(
                                "Connection closed while reading NOISCHAR message"
                            )
                            conn.sendall(b"FAILED:Connection closed")
                            break

                        message_parts.append(chunk.decode("utf-8"))
                        total_bytes += len(chunk)
                        logger.debug(f"NOISCHAR: received {total_bytes} bytes so far")

                        full_message = "".join(message_parts)

                        if END_MARKER in full_message:
                            message = full_message.replace(END_MARKER, "").strip()
                            logger.info(f"NOISCHAR message: {message}")

                            # Parse flags
                            params = {}
                            flags = [
                                "--output",
                                "--preset",
                                "--frames",
                                "--plots",
                                "--gains",
                                "--exposures",
                            ]

                            def find_flag_position(msg, flag):
                                """Find flag position ensuring it's followed by a space."""
                                search_pattern = flag + " "
                                if search_pattern in msg:
                                    return msg.index(search_pattern)
                                return -1

                            for i, flag in enumerate(flags):
                                flag_pos = find_flag_position(message, flag)
                                if flag_pos >= 0:
                                    start_idx = flag_pos + len(flag)
                                    end_idx = len(message)
                                    for next_flag in flags:
                                        if next_flag != flag:
                                            next_pos = find_flag_position(
                                                message[start_idx:], next_flag
                                            )
                                            if next_pos >= 0:
                                                actual_pos = start_idx + next_pos
                                                if actual_pos < end_idx:
                                                    end_idx = actual_pos

                                    value = message[start_idx:end_idx].strip()

                                    if flag == "--output":
                                        params["output_path"] = value
                                    elif flag == "--preset":
                                        params["preset"] = value
                                    elif flag == "--frames":
                                        params["num_frames"] = int(value)
                                    elif flag == "--plots":
                                        params["generate_plots"] = (
                                            value.lower() == "true"
                                        )
                                    elif flag == "--gains":
                                        params["gains"] = [
                                            float(v.strip())
                                            for v in value.split(",")
                                        ]
                                    elif flag == "--exposures":
                                        params["exposures"] = [
                                            float(v.strip())
                                            for v in value.split(",")
                                        ]

                            # Validate required parameters
                            required = ["output_path"]
                            missing = [
                                key for key in required if key not in params
                            ]
                            if missing:
                                error_msg = (
                                    f"Missing required parameters: {missing}"
                                )
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                                break

                            # Send immediate acknowledgment
                            try:
                                from pathlib import Path

                                output_path = Path(params["output_path"])
                                output_path.mkdir(parents=True, exist_ok=True)

                                ack_response = (
                                    f"STARTED:{params['output_path']}".encode()
                                )
                                conn.sendall(ack_response)
                                logger.info(
                                    "Sent STARTED acknowledgment for NOISCHAR"
                                )

                                # Increase socket timeout for long-running
                                # characterization (up to 20 minutes)
                                conn.settimeout(1200.0)

                                # Import the characterization module
                                from microscope_control.jai import (
                                    JAINoiseCharacterization,
                                    JAICameraProperties,
                                )

                                # Create characterization tool
                                jai_props = JAICameraProperties(hardware.core)
                                tool = JAINoiseCharacterization(
                                    hardware,
                                    jai_props,
                                    num_frames=params.get("num_frames", 10),
                                )

                                # Build progress callback that sends PROGRESS
                                # messages back to the Java client
                                def progress_callback(current, total, msg=""):
                                    try:
                                        progress_msg = (
                                            f"PROGRESS:{current}:{total}"
                                        )
                                        conn.sendall(progress_msg.encode())
                                        logger.debug(
                                            f"NOISCHAR progress: "
                                            f"{current}/{total}"
                                        )
                                    except Exception as pe:
                                        logger.warning(
                                            f"Failed to send progress: {pe}"
                                        )

                                # Determine preset / custom gains+exposures
                                preset = params.get("preset", "full")
                                custom_gains = params.get("gains")
                                custom_exposures = params.get("exposures")

                                # Run characterization
                                is_quick = preset == "quick"
                                results = tool.run_characterization(
                                    gains=custom_gains,
                                    exposures=custom_exposures,
                                    quick=is_quick,
                                    progress_callback=progress_callback,
                                )

                                # Generate report/plots or just CSV
                                generate_plots = params.get(
                                    "generate_plots", False
                                )
                                if generate_plots:
                                    tool.generate_report(
                                        results, output_path
                                    )
                                    logger.info(
                                        "NOISCHAR: generated report with plots"
                                    )
                                else:
                                    # Just save CSV
                                    results.to_csv(
                                        output_path
                                        / "noise_characterization.csv"
                                    )
                                    logger.info(
                                        "NOISCHAR: saved CSV results only"
                                    )

                                # Find best SNR from unsaturated results
                                best_gain = 0.0
                                best_exp = 0.0
                                best_snr = 0.0
                                total_configs = len(results.results)
                                for r in results.results:
                                    if r.saturation_pct > 1.0:
                                        continue
                                    # Average SNR across channels
                                    avg_snr = (
                                        r.red_snr + r.green_snr + r.blue_snr
                                    ) / 3.0
                                    if avg_snr > best_snr:
                                        best_snr = avg_snr
                                        best_gain = r.unified_gain
                                        best_exp = r.exposure_ms

                                # Format: SUCCESS:{path}|{count}|{plots}|
                                #         {bestGain},{bestExp}
                                plots_str = (
                                    "true" if generate_plots else "false"
                                )
                                response = (
                                    f"SUCCESS:{output_path}|"
                                    f"{total_configs}|"
                                    f"{plots_str}|"
                                    f"{best_gain},{best_exp}"
                                )
                                conn.sendall(response.encode())
                                logger.info(
                                    f"NOISCHAR completed: {total_configs} "
                                    f"configs, best SNR at "
                                    f"gain={best_gain}, "
                                    f"exp={best_exp}ms"
                                )

                            except ImportError as e:
                                error_msg = (
                                    f"JAI noise characterization module "
                                    f"not available: {e}"
                                )
                                logger.error(error_msg)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                            except Exception as e:
                                error_msg = f"NOISCHAR failed: {str(e)}"
                                logger.error(error_msg, exc_info=True)
                                conn.sendall(f"FAILED:{error_msg}".encode())
                            break

                        if total_bytes > 100000:
                            logger.error(
                                "NOISCHAR message exceeds maximum size"
                            )
                            conn.sendall(b"FAILED:Message too large")
                            break

                        if time.time() - start_time > 10:
                            logger.error("Timeout reading NOISCHAR message")
                            conn.sendall(
                                b"FAILED:Timeout waiting for complete message"
                            )
                            break

                except socket.timeout:
                    logger.error(
                        f"Timeout reading NOISCHAR message from {addr}"
                    )
                    conn.sendall(b"FAILED:Timeout reading message")
                except Exception as e:
                    logger.error(
                        f"Error in NOISCHAR: {str(e)}", exc_info=True
                    )
                    conn.sendall(f"FAILED:{str(e)}".encode())
                finally:
                    conn.settimeout(None)  # Reset to blocking mode
                    # Reset camera to clean state after characterization
                    try:
                        from microscope_control.jai import JAICameraProperties

                        jai_props = JAICameraProperties(hardware.core)
                        jai_props.set_rb_analog_gains(red=1.0, blue=1.0)
                        jai_props.set_unified_gain(1.0)
                        jai_props.disable_individual_exposure()
                        logger.debug("Reset camera state after NOISCHAR")
                    except (ImportError, Exception):
                        pass

                continue

            # Unknown command
            logger.warning(f"Unknown command from {addr}: {data}")

    except Exception as e:
        logger.error(f"Error handling client {addr}: {str(e)}", exc_info=True)
    finally:
        # Cleanup
        if acquisition_thread and acquisition_thread.is_alive():
            logger.info(f"Cancelling acquisition for disconnected client {addr}")
            acquisition_cancel_events[addr].set()
            acquisition_thread.join(timeout=10)

        # Remove client state
        if addr in acquisition_locks:
            del acquisition_locks[addr]
        if addr in acquisition_states:
            del acquisition_states[addr]
        if addr in acquisition_progress:
            del acquisition_progress[addr]
        if addr in acquisition_cancel_events:
            del acquisition_cancel_events[addr]
        if addr in acquisition_failure_messages:
            del acquisition_failure_messages[addr]
        if addr in acquisition_final_z:
            del acquisition_final_z[addr]

        # Clear active connection if this was the active client
        # NOTE: global statement removed - these are module-level variables accessed via 'connection_state_lock'
        # global server_configured, active_connection_addr, active_connection_config_path
        with connection_state_lock:
            if active_connection_addr == addr:
                logger.info(f"Active connection {addr} disconnected - server now UNCONFIGURED")
                logger.info("Next connection will need to provide CONFIG command")
                # Stop session logging before clearing state
                _stop_session_logging()
                server_configured = False
                active_connection_addr = None
                active_connection_config_path = None

        conn.close()
        logger.info(f"<<< Client {addr} disconnected and cleaned up")


def main():
    """Main server loop that accepts client connections and spawns handler threads."""
    logger.info("=" * 60)
    logger.info("Microscope Command Server")
    logger.info("=" * 60)

    # Check for existing server instance BEFORE attempting to bind
    logger.info("Checking for existing server instance...")
    if check_for_existing_server("127.0.0.1", PORT):
        logger.error("=" * 60)
        logger.error("ANOTHER SERVER INSTANCE IS ALREADY RUNNING!")
        logger.error("=" * 60)
        logger.error(f"A server is already listening on port {PORT}.")
        logger.error("Please close the existing server before starting a new one.")
        logger.error("")
        logger.error("To find the existing server:")
        logger.error("  Windows: Use Task Manager to find python.exe processes")
        logger.error("  Linux: Run 'lsof -i :5000' or 'netstat -tlnp | grep 5000'")
        logger.error("=" * 60)
        print("\n" + "=" * 60)
        print("ERROR: Another server instance is already running on port {}!".format(PORT))
        print("Please close the existing server before starting a new one.")
        print("=" * 60 + "\n")
        sys.exit(1)

    logger.info("No existing server instance found. Proceeding with startup...")

    logger.info(f"Server configuration:")
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Port: {PORT}")
    logger.info(f"  Micro-Manager core: {'Connected' if core else 'Not connected'}")
    logger.info(f"  Hardware: {'Initialized' if hardware else 'Not initialized'}")

    # Log loaded configuration
    microscope_info = startup_settings.get("microscope", {})
    logger.info(f"  Microscope: {microscope_info.get('name', 'Unknown')}")
    logger.info(f"  Type: {microscope_info.get('type', 'Unknown')}")

    logger.info("Features:")
    logger.info("  - Status monitoring")
    logger.info("  - Progress tracking")
    logger.info("  - Cancellation support")
    logger.info("  - Enhanced logging")
    logger.info("  - Multi-instance detection")
    logger.info("=" * 60)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        logger.info(f"Server listening on {HOST}:{PORT}")
        logger.info("Ready for connections...")

        threads = []

        while not shutdown_event.is_set():
            try:
                s.settimeout(1.0)
                conn, addr = s.accept()
                thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                thread.start()
                threads.append(thread)
            except socket.timeout:
                continue
            except OSError:
                break

        logger.info("Server shutting down. Waiting for client threads to finish...")
        shutdown_event.set()

        for t in threads:
            t.join(timeout=5.0)

        logger.info("Server has shut down.")


if __name__ == "__main__":
    main()
