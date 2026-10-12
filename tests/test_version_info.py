"""The version header must say, at startup, when a lab package's install is shadowed."""

import sys
import types

import pytest

from microscope_command_server import version_info


@pytest.fixture
def fake_modules(monkeypatch):
    """One healthy fake package and one assembled as a namespace package."""
    healthy = types.ModuleType("fake_lab_ok")
    healthy.__file__ = "/somewhere/fake_lab_ok/__init__.py"
    shadowed = types.ModuleType("fake_lab_shadowed")
    shadowed.__file__ = None
    shadowed.__path__ = ["/checkouts/fake_lab_shadowed"]
    monkeypatch.setitem(sys.modules, "fake_lab_ok", healthy)
    monkeypatch.setitem(sys.modules, "fake_lab_shadowed", shadowed)
    monkeypatch.setattr(
        version_info,
        "_GIT_MODULES",
        [
            ("fake_lab_ok", "Fake OK"),
            ("fake_lab_shadowed", "Fake Shadowed"),
            ("fake_lab_absent", "Fake Absent"),
        ],
    )


def test_shadowed_modules_names_only_the_namespace_package(fake_modules):
    found = version_info.shadowed_modules()
    assert [name for name, _ in found] == ["Fake Shadowed"]
    message = found[0][1]
    assert "fake_lab_shadowed" in message
    assert "/checkouts/fake_lab_shadowed" in message
    assert "unknown location" in message
    assert "update_env.bat" in message


def test_git_info_distinguishes_shadowed_from_missing(fake_modules):
    assert version_info._git_info("fake_lab_absent") == "unknown (module not found)"
    assert version_info._git_info("fake_lab_shadowed").startswith("unknown (namespace package")
