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
from microscope_command_server.version_info import collect_versions, format_log_header


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


_session_log_config_path = None  # Track which config started the current session


def _start_session_logging(config_path: str) -> None:
    """
    Start session-based file logging in the config file's parent directory.

    Creates a log file at <config_parent>/logs/server_session_YYYYMMDD_HHMMSS.log.
    The handler is added to the root logger so all module loggers are captured.
    The handler flushes immediately on each log record (no buffered data lost on crash).

    If a session log is already active for the same config path, keeps the
    existing log file rather than creating a new one (prevents log splitting
    when QuPath sends multiple CONFIG commands during a single session).

    Args:
        config_path: Path to the YAML config file sent by QuPath via CONFIG command
    """
    global _session_log_handler, _session_log_config_path

    # If already logging for the same config, keep existing log
    if _session_log_handler is not None and _session_log_config_path == config_path:
        logger.info("Session logging already active for this config, continuing in same log")
        return

    # Different config or no active handler -- start fresh
    _stop_session_logging()
    _session_log_config_path = config_path

    try:
        config_dir = pathlib.Path(config_path).resolve().parent
        session_log_dir = config_dir / "logs"
        session_log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_file = session_log_dir / f"server_session_{timestamp}.log"

        # Create a handler that flushes after every log record.
        # Without this, Python buffers log output and an 18-hour acquisition
        # can lose all diagnostic data if the server crashes.
        class FlushingFileHandler(logging.FileHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        handler = FlushingFileHandler(session_log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

        # Add to root logger so all child loggers are captured
        logging.getLogger().addHandler(handler)
        _session_log_handler = handler

        logger.info(f"Session logging started: {session_log_file}")
        logger.info(format_log_header())
    except Exception as e:
        logger.error(f"Failed to start session logging: {e}", exc_info=True)


def _stop_session_logging() -> None:
    """
    Stop session-based file logging and clean up the handler.

    Flushes and closes the session log handler, then removes it from the root logger.
    """
    global _session_log_handler, _session_log_config_path

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
            _session_log_config_path = None


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
acquisition_saturation_summary = {}  # addr -> str (saturation summary when COMPLETED)
manual_focus_request_events = {}  # addr -> Event (set when manual focus needed)
manual_focus_complete_events = {}  # addr -> Event (set when user acknowledges)
manual_focus_user_choice = {}  # addr -> str ("retry", "skip", "cancel")
manual_focus_retries_remaining = {}  # addr -> int (number of retries remaining)
hardware_error_request_events = {}  # addr -> Event (set when hardware error needs user decision)
hardware_error_complete_events = {}  # addr -> Event (set when user acknowledges hardware error)
hardware_error_user_choice = {}  # addr -> str ("retry", "skip", "cancel")
hardware_error_message = {}  # addr -> str (error message string per client)

# Server configuration state - CRITICAL FOR SAFETY
# NEVER allow hardware operations with generic config - could damage microscope!
server_configured = False  # True only after CONFIG command received with valid microscope config
active_connection_addr = None  # Track single active client connection (blocks other connections)
active_connection_config_path = None  # Path to config file provided by active connection
connection_state_lock = Lock()  # Protect connection state from race conditions
# Track all connected clients from the configured IP so we only unconfigure
# when ALL connections from that IP disconnect (Java uses main + aux sockets).
active_ip_connections = set()  # Set of (ip, port) tuples from the active IP



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


def acquisitionWorkflow(message, client_addr):
    """Wrapper for acquisition workflow with state management."""

    def _update_progress(current: int, total: int):
        with acquisition_locks[client_addr]:
            acquisition_progress[client_addr] = (current, total)

    def _set_state(state_str: str, error_message: str = None, final_z: float = None,
                   saturation_summary: str = None):
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
                # Store saturation summary if provided
                if new_state == AcquisitionState.COMPLETED and saturation_summary:
                    acquisition_saturation_summary[client_addr] = saturation_summary
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

    def _request_hardware_error_recovery(error_message: str) -> str:
        """Signal hardware error and wait for user to choose retry/skip/cancel.

        Args:
            error_message: Detailed error message to show to user

        Returns:
            str: User's choice - "retry", "skip", or "cancel"
        """
        logger.info(f"Hardware error recovery requested for {client_addr}")
        hardware_error_message[client_addr] = error_message
        hardware_error_request_events[client_addr].set()
        hardware_error_user_choice[client_addr] = None
        logger.info("Waiting for user to resolve hardware error...")
        hardware_error_complete_events[client_addr].wait()
        user_choice = hardware_error_user_choice[client_addr] or "cancel"
        # Clear events
        hardware_error_request_events[client_addr].clear()
        hardware_error_complete_events[client_addr].clear()
        hardware_error_user_choice[client_addr] = None
        hardware_error_message[client_addr] = ""
        logger.info(f"Hardware error resolved, user chose: {user_choice}")
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
        request_hardware_error_recovery=_request_hardware_error_recovery,
        connection_config_path=active_connection_config_path,
    )


def handle_client(conn, addr):
    """Handle commands from a connected client via dispatch table.

    Each 8-byte command is looked up in COMMAND_HANDLERS and dispatched
    to the appropriate handler function. Per-client state is tracked in
    the global dicts (will migrate to ClientState in a future pass).
    """
    global server_configured, active_connection_addr, active_connection_config_path, startup_settings
    from microscope_command_server.server.handlers import COMMAND_HANDLERS
    from microscope_command_server.server.client_state import ClientState

    logger.info(f">>> New client connected from {addr}")

    # Track this connection for disconnect cleanup
    with connection_state_lock:
        active_ip_connections.add(addr)

    # Initialize per-client state in global dicts
    # (legacy pattern -- will migrate to ClientState object)
    acquisition_locks[addr] = Lock()
    acquisition_states[addr] = AcquisitionState.IDLE
    acquisition_progress[addr] = (0, 0)
    acquisition_cancel_events[addr] = threading.Event()
    acquisition_failure_messages[addr] = None
    manual_focus_request_events[addr] = threading.Event()
    manual_focus_complete_events[addr] = threading.Event()
    manual_focus_user_choice[addr] = None
    manual_focus_retries_remaining[addr] = 0
    hardware_error_request_events[addr] = threading.Event()
    hardware_error_complete_events[addr] = threading.Event()
    hardware_error_user_choice[addr] = None
    hardware_error_message[addr] = ""

    # ClientState object for handlers that use the new pattern
    client = ClientState(addr)
    acquisition_thread = None

    # Shared kwargs passed to every handler for access to global state.
    # This dict contains everything any handler might need. Handlers
    # pick what they need via kwargs["key"] or kwargs.get("key").
    handler_kwargs = {
        # Identity
        "addr": addr,
        # Server state
        "server_configured": server_configured,
        "shutdown_event": shutdown_event,
        "connection_state_lock": connection_state_lock,
        "active_connection_addr": active_connection_addr,
        "active_connection_config_path": active_connection_config_path,
        # Config manager (for CONFIG handler to reload settings)
        "config_manager": config_manager,
        # AcquisitionState enum (handlers import-free access)
        "AcquisitionState": AcquisitionState,
        # Acquisition workflow function (for ACQUIRE handler)
        "acquisitionWorkflow": acquisitionWorkflow,
        # Session logging control (for CONFIG handler)
        "start_session_logging": _start_session_logging,
        # Per-client state dicts (legacy -- handlers access by addr)
        "acquisition_states": acquisition_states,
        "acquisition_progress": acquisition_progress,
        "acquisition_locks": acquisition_locks,
        "acquisition_cancel_events": acquisition_cancel_events,
        "acquisition_failure_messages": acquisition_failure_messages,
        "acquisition_final_z": acquisition_final_z,
        "acquisition_saturation_summary": acquisition_saturation_summary,
        "manual_focus_request_events": manual_focus_request_events,
        "manual_focus_complete_events": manual_focus_complete_events,
        "manual_focus_user_choice": manual_focus_user_choice,
        "manual_focus_retries_remaining": manual_focus_retries_remaining,
        "hardware_error_request_events": hardware_error_request_events,
        "hardware_error_complete_events": hardware_error_complete_events,
        "hardware_error_user_choice": hardware_error_user_choice,
        "hardware_error_message": hardware_error_message,
        # Acquisition thread tracking
        "acquisition_thread": acquisition_thread,
    }

    try:
        while True:
            data = conn.recv(8)
            if not data:
                logger.info(f"Client {addr} disconnected (no data)")
                break

            logger.debug(f"Received command from {addr}: {data}")

            # Look up handler in dispatch table
            handler = COMMAND_HANDLERS.get(data)
            if handler is None:
                logger.warning(f"Unknown command from {addr}: {data}")
                continue

            # Update kwargs with current mutable state before each dispatch
            handler_kwargs["server_configured"] = server_configured
            handler_kwargs["active_connection_addr"] = active_connection_addr
            handler_kwargs["active_connection_config_path"] = active_connection_config_path
            handler_kwargs["acquisition_thread"] = acquisition_thread

            result = handler(conn, client, hardware, startup_settings, **handler_kwargs)

            # Handle special return values from handlers
            if result == 'DISCONNECT':
                logger.info(f"Client {addr} requested to disconnect")
                break
            elif result == 'SHUTDOWN':
                logger.warning(f"Client {addr} requested server shutdown")
                shutdown_event.set()
                break
            elif isinstance(result, dict):
                # CONFIG handler returns updated server state
                if 'server_configured' in result:
                    server_configured = result['server_configured']
                if 'active_connection_addr' in result:
                    active_connection_addr = result['active_connection_addr']
                if 'active_connection_config_path' in result:
                    active_connection_config_path = result['active_connection_config_path']
                if 'settings' in result:
                    startup_settings = result['settings']
            elif isinstance(result, threading.Thread):
                # ACQUIRE handler returns the spawned acquisition thread
                acquisition_thread = result

            # All commands dispatched via COMMAND_HANDLERS above
            # (old 4,300-line if/elif chain removed -- see handlers/ modules)



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
        if addr in acquisition_saturation_summary:
            del acquisition_saturation_summary[addr]

        # Remove this connection from tracking and check if all connections from
        # the active IP are gone before unconfiguring.  Java uses main + aux sockets,
        # so we must NOT unconfigure when just one of them disconnects.
        with connection_state_lock:
            active_ip_connections.discard(addr)

            if active_connection_addr == addr:
                # The CONFIG-owning connection disconnected.  Check if any other
                # connections from the same IP are still alive.
                active_ip = addr[0]
                remaining = [a for a in active_ip_connections if a[0] == active_ip]
                if remaining:
                    # Hand ownership to the remaining connection
                    active_connection_addr = remaining[0]
                    logger.info(
                        f"Active connection {addr} disconnected - "
                        f"handing ownership to {active_connection_addr} "
                        f"({len(remaining)} connection(s) still active from same IP)"
                    )
                else:
                    # No more connections from this IP -- truly unconfigure
                    logger.info(f"All connections from {active_ip} disconnected - server now UNCONFIGURED")
                    logger.info("Next connection will need to provide CONFIG command")
                    # Stop any orphaned sequence acquisition left running by
                    # the departed client. If the live viewer crashed or the
                    # main connection timed out mid-stream, the camera keeps
                    # streaming into MM's circular buffer indefinitely --
                    # this can hard-lock the Hamamatsu sCMOS driver and
                    # MicroManager itself (observed on OWS3 2026-04-09).
                    try:
                        if hardware is not None and hardware.core.is_sequence_running():
                            logger.warning(
                                "Stopping orphaned sequence acquisition left by disconnected client"
                            )
                            hardware.core.stop_sequence_acquisition()
                    except Exception as stop_err:
                        logger.error(
                            "Failed to stop orphaned sequence acquisition: %s", stop_err
                        )
                    _stop_session_logging()
                    server_configured = False
                    active_connection_addr = None
                    active_connection_config_path = None
            else:
                # Not the active connection -- just remove from tracking
                logger.debug(f"Non-active connection {addr} disconnected")

        conn.close()
        logger.info(f"<<< Client {addr} disconnected and cleaned up")


def main():
    """Main server loop that accepts client connections and spawns handler threads."""
    logger.info("=" * 60)
    logger.info("Microscope Command Server")
    logger.info("=" * 60)
    logger.info(format_log_header())

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
