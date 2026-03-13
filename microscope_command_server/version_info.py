"""Version information collection for provenance tracking.

Collects version strings from all installed Python packages in the
microscope control ecosystem, plus Python and OS versions. Used to
write version headers in session logs and boot logs.
"""

import platform
import sys
from importlib.metadata import version as _get_version


# Package names as registered in pyproject.toml [project].name
_PACKAGES = [
    ("microscope-command-server", "Microscope Command Server"),
    ("microscope-control", "Microscope Control"),
    ("ppm-library", "PPM Library"),
]


def _safe_version(package_name: str) -> str:
    """Return the installed version of a package, or 'not installed'."""
    try:
        return _get_version(package_name)
    except Exception:
        return "not installed"


def collect_versions() -> dict[str, str]:
    """Collect all relevant version strings as an ordered dict.

    Returns:
        Dictionary mapping component names to version strings.
    """
    versions = {}
    for pkg_name, display_name in _PACKAGES:
        versions[display_name] = _safe_version(pkg_name)
    versions["Python"] = platform.python_version()
    versions["OS"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    return versions


def format_log_header() -> str:
    """Format a multi-line version header for log files.

    Returns:
        Newline-separated string suitable for logging.
    """
    versions = collect_versions()
    lines = ["=== Python Server Version Info ==="]
    for name, ver in versions.items():
        lines.append(f"  {name}: {ver}")
    lines.append("=================================")
    return "\n".join(lines)
