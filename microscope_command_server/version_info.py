"""Version information collection for provenance tracking.

Collects version strings from all installed Python packages in the
microscope control ecosystem, plus Python and OS versions, and the
latest git commit for each repo. Used to write version headers in
session logs and boot logs.
"""

import functools
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version as _get_version
from pathlib import Path

# Package names as registered in pyproject.toml [project].name
_PACKAGES = [
    ("microscope-command-server", "Microscope Command Server"),
    ("microscope-control", "Microscope Control"),
    ("ppm-library", "PPM Library"),
]

# Repo directories to check git status for.
# Keys = display name, values = module whose __file__ locates the repo.
_GIT_MODULES = [
    ("microscope_command_server", "Microscope Command Server"),
    ("microscope_control", "Microscope Control"),
    ("ppm_library", "PPM Library"),
]


def _safe_version(package_name: str) -> str:
    """Return the installed version of a package, or 'not installed'."""
    try:
        return _get_version(package_name)
    except Exception:
        return "not installed"


def _git_info(module_name: str) -> str:
    """Get the latest git commit hash and date for a Python module's repo.

    Locates the repo by finding the module's source file, walking up to
    find a .git directory, then running git log.

    Returns:
        String like 'b120cc5 (2026-04-01)' or 'unknown' on failure.
    """
    try:
        mod = sys.modules.get(module_name)
        if mod is None:
            __import__(module_name)
            mod = sys.modules.get(module_name)
        if mod is None or not hasattr(mod, "__file__") or mod.__file__ is None:
            return "unknown (module not found)"

        # Walk up from the module file to find the git repo root
        path = Path(mod.__file__).resolve()
        repo_dir = None
        for parent in [path] + list(path.parents):
            if (parent / ".git").exists():
                repo_dir = parent
                break
        if repo_dir is None:
            return "unknown (no .git found)"

        result = subprocess.run(
            ["git", "log", "-1", "--format=%h (%ai)"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "unknown (git error)"
    except Exception:
        return "unknown"


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


@functools.lru_cache(maxsize=1)
def collect_git_info() -> dict[str, str]:
    """Collect latest git commit info for each repo.

    The three git subprocesses are run in parallel via a thread pool
    (each call is independent), cutting total wall time roughly 3x.
    The result is memoized so subsequent calls are free; repo contents
    rarely change within a single server process, and a stale commit
    string in a log header is harmless.

    Returns:
        Dictionary mapping component names to git commit strings.
    """
    display_names = [display for _, display in _GIT_MODULES]
    module_names = [module for module, _ in _GIT_MODULES]

    with ThreadPoolExecutor(max_workers=len(_GIT_MODULES)) as executor:
        results = list(executor.map(_git_info, module_names))

    return dict(zip(display_names, results))


def format_log_header() -> str:
    """Format a multi-line version header for log files.

    Returns:
        Newline-separated string suitable for logging.
    """
    versions = collect_versions()
    git_info = collect_git_info()

    lines = ["=== Python Server Version Info ==="]
    for name, ver in versions.items():
        lines.append(f"  {name}: {ver}")
    lines.append("")
    lines.append("=== Git Commits (latest) ===")
    for name, commit in git_info.items():
        lines.append(f"  {name}: {commit}")
    lines.append("=================================")
    return "\n".join(lines)
