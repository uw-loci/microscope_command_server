"""Tested invariant for the session log shipped to the QuPath bug reporter.

`scrub_paths` must redact the home directory in EVERY separator form a Windows
log emits, not just the single-backslash filesystem form. A scrubber built from
`re.escape(home)` matches only that one form, so a log line that formatted a
path through repr() (any list, dict or %r -- which doubles every backslash) or
normalized it to forward slashes kept the username verbatim, in the same report
where neighbouring lines were redacted correctly.

The Windows branch is exercised by faking `Path.home()` and `os.name`, so these
run on Linux/macOS CI.
"""

import re

import pytest

from microscope_command_server.server import log_scrubber
from microscope_command_server.server.log_scrubber import scrub_paths

WIN_HOME = r"C:\Users\gboyu"


@pytest.fixture
def windows_home(monkeypatch):
    """Make scrub_paths believe it is running on Windows under WIN_HOME."""
    monkeypatch.setattr(log_scrubber.os, "name", "nt")
    monkeypatch.setattr(log_scrubber.Path, "home", staticmethod(lambda: WIN_HOME))


@pytest.mark.parametrize(
    "line,expected",
    [
        # single backslash -- the filesystem form
        (r"cwd: C:\Users\gboyu\Downloads", r"cwd: ~\Downloads"),
        # doubled backslash -- str() over a list of paths; the form that leaked
        (
            r"sys.path[:2] = ['C:\\Users\\gboyu\\AppData', 'C:\\Users\\gboyu\\Documents']",
            r"sys.path[:2] = ['~\\AppData', '~\\Documents']",
        ),
        # forward slash -- normalized paths and URIs
        ("path: C:/Users/gboyu/AppData/Local", "path: ~/AppData/Local"),
        # case variant -- Windows filesystems are case-insensitive
        (r"C:\USERS\GBOYU\Downloads\log.txt", r"~\Downloads\log.txt"),
    ],
)
def test_redacts_every_separator_form(windows_home, line, expected):
    assert scrub_paths(line) == expected


def test_no_username_survives_a_mixed_log(windows_home):
    # The failure was per-line, so a log mixing all four forms is the real guard.
    log = "\n".join(
        [
            r"cwd: C:\Users\gboyu\Downloads",
            r"['C:\\Users\\gboyu\\AppData']",
            "uri file:/C:/Users/gboyu/img.tif",
            r"C:\USERS\GBOYU\x",
        ]
    )
    assert not re.search("gboyu", scrub_paths(log), re.IGNORECASE)


def test_keeps_non_home_paths_intact(windows_home):
    # Only user identity is redacted; lab shares and install dirs are diagnostic.
    line = r"config C:\ProgramData\QPSC\scope.yml from \\lab-nas\share\data"
    assert scrub_paths(line) == line


def test_redacts_home_on_unix(monkeypatch):
    monkeypatch.setattr(log_scrubber.os, "name", "posix")
    monkeypatch.setattr(log_scrubber.Path, "home", staticmethod(lambda: "/home/gboyu"))
    assert scrub_paths("cwd: /home/gboyu/data") == "cwd: ~/data"


@pytest.mark.parametrize("value", ["", None])
def test_empty_input_is_returned_unchanged(windows_home, value):
    # The GETLOG path must never raise: an unreadable log yields "".
    assert scrub_paths(value) == value
