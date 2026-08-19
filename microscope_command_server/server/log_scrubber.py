"""Redaction applied to the session log before it leaves this process.

Lives outside the handler package (like `server.probe_parsers`) so it can be
imported and unit-tested without pulling in the handler `__init__` chain, which
transitively requires microscope_control / Pycromanager bindings.
"""

import os
import re
from pathlib import Path


def scrub_paths(text):
    """Replace this machine's home directory with ~ so log text carries no username.

    The session-log tail is shipped to the QuPath in-app bug reporter over the
    GETLOG command and ends up in a public GitHub issue. QuPath scrubs what it
    collects, but it can only scrub with its OWN home directory -- when the
    command server runs on the microscope workstation under a different account,
    the server's username is a string QuPath has never seen and cannot redact.
    So the process that owns the path scrubs it, here, before it crosses the wire.

    On Windows the same path reaches a log in several separator forms and a
    literal replace catches only one:

    - single backslash  ``C:\\Users\\alice``     -- the filesystem form
    - doubled backslash ``C:\\\\Users\\\\alice`` -- repr(), so any list/dict/%r
    - forward slash     ``C:/Users/alice``       -- normalized paths and URIs

    Splitting the home path on the native separator and rejoining with a
    permissive separator class matches all three, and IGNORECASE covers the
    case-insensitive filesystem (``C:\\USERS\\ALICE``).

    Only the home directory is redacted. Lab shares, ``C:\\ProgramData`` and
    install directories carry no user identity and are usually the most
    diagnostic part of the log, so they are left intact.
    """
    home = str(Path.home())
    if not text or not home:
        return text
    if os.name == "nt":
        parts = [p for p in home.split("\\") if p]
        if not parts:
            return text
        pattern = r"[\\/]{1,2}".join(re.escape(p) for p in parts)
        return re.sub(pattern, "~", text, flags=re.IGNORECASE)
    # Unix / macOS: no separator ambiguity, so a substring replace is exact.
    return text.replace(home, "~")
