"""
Server package - Socket server for remote microscope control.

This package contains the socket-based communication layer between
client applications and the microscope control system.

Modules:
    qp_server: Main socket server implementation
    protocol: Command definitions and wire protocol
    client: Test client utilities
"""

from microscope_command_server.server.protocol import Command, ExtendedCommand, TCP_PORT, END_MARKER

__all__ = ["Command", "ExtendedCommand", "TCP_PORT", "END_MARKER"]
