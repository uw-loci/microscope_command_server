"""Record a real session at the microscope and replay it against changed code.

``recorder`` wraps the server's Core/Studio and every client socket when
``--trace-dir`` (or ``MICROSCOPE_TRACE_DIR``) is given; ``replayer`` feeds the
recorded client commands to the server running on a ``ReplayCore`` and
compares replies and hardware calls stage by stage. The trace format and the
fake core live in ``microscope_control.trace``.

Command line: ``python -m microscope_command_server.trace replay TRACE_DIR``.
"""

from microscope_command_server.trace.recorder import (
    POLL_COMMANDS,
    READ_COMMANDS,
    RecordingConnection,
    SessionRecorder,
    command_class,
    command_name,
    command_scope,
)

__all__ = [
    "POLL_COMMANDS",
    "READ_COMMANDS",
    "RecordingConnection",
    "SessionRecorder",
    "command_class",
    "command_name",
    "command_scope",
]
