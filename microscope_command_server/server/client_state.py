"""Per-client state for the microscope command server.

Each connected client gets a ClientState instance that tracks acquisition
progress, manual focus coordination, and hardware error recovery. This
replaces the 17 separate per-client dictionaries that were previously
keyed by client address.
"""

import enum
import logging
from threading import Lock, Event

logger = logging.getLogger(__name__)


class AcquisitionState(enum.Enum):
    """States for the acquisition lifecycle."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ClientState:
    """Per-client state for a connected QuPath session.

    Consolidates all per-client tracking into a single object instead
    of 17 separate dictionaries keyed by client address.
    """

    def __init__(self, addr):
        self.addr = addr

        # Acquisition tracking
        self.lock = Lock()
        self.state = AcquisitionState.IDLE
        self.progress = (0, 0)  # (current_tile, total_tiles)
        self.cancel_event = Event()
        self.failure_message = None
        self.final_z = None  # Final Z position (for tilt model)
        self.saturation_summary = None

        # Manual focus coordination
        self.manual_focus_request = Event()
        self.manual_focus_complete = Event()
        self.manual_focus_choice = None  # "retry", "skip", "cancel"
        self.manual_focus_retries = 0

        # Hardware error recovery
        self.hw_error_request = Event()
        self.hw_error_complete = Event()
        self.hw_error_choice = None  # "retry", "skip", "cancel"
        self.hw_error_message = ""

    def reset_for_acquisition(self):
        """Reset state for a new acquisition."""
        self.state = AcquisitionState.RUNNING
        self.progress = (0, 0)
        self.cancel_event.clear()
        self.failure_message = None
        self.final_z = None
        self.saturation_summary = None
        self.manual_focus_request.clear()
        self.manual_focus_complete.clear()
        self.manual_focus_choice = None
        self.manual_focus_retries = 0
        self.hw_error_request.clear()
        self.hw_error_complete.clear()
        self.hw_error_choice = None
        self.hw_error_message = ""

    def __repr__(self):
        return f"ClientState(addr={self.addr}, state={self.state.value})"
