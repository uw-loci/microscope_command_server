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
    ("microscope-imageprocessing", "Microscope Image Processing"),
    ("ppm-library", "PPM Library"),
    ("polscope-library", "PolScope Library"),
]

# Repo directories to check git status for.
# Keys = display name, values = module whose __file__ locates the repo.
_GIT_MODULES = [
    ("microscope_command_server", "Microscope Command Server"),
    ("microscope_control", "Microscope Control"),
    ("microscope_imageprocessing", "Microscope Image Processing"),
    ("ppm_library", "PPM Library"),
    ("polscope_library", "PolScope Library"),
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
            try:
                __import__(module_name)
            except ImportError:
                return "unknown (module not found)"
            mod = sys.modules.get(module_name)
        if mod is None:
            return "unknown (module not found)"
        if getattr(mod, "__file__", None) is None:
            return "unknown (namespace package -- the install is shadowed, see the ERROR logged at startup)"

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


def shadowed_modules() -> list[tuple[str, str]]:
    """Lab packages that imported as namespace packages instead of their installs.

    A package whose ``__file__`` is None after import never ran its
    ``__init__.py``: Python assembled it from a bare directory of that name
    found on ``sys.path`` -- in practice the package's own repo root, when the
    process started in the folder that holds the checkouts -- and only its
    subpackages resolve through the real install. Every
    ``from <package> import name`` then fails with "(unknown location)". On the
    LC-PolScope rig (2026-10-12) this surfaced twenty seconds into an LC
    calibration, not at startup, which is why this check exists.

    Returns:
        ``(display_name, message)`` per shadowed package; empty when all is well.
    """
    found = []
    for module_name, display_name in _GIT_MODULES:
        mod = sys.modules.get(module_name)
        if mod is None:
            try:
                __import__(module_name)
            except ImportError:
                continue  # optional library not installed on this rig
            mod = sys.modules.get(module_name)
        if mod is None or getattr(mod, "__file__", None) is not None:
            continue
        portions = ", ".join(str(p) for p in (getattr(mod, "__path__", None) or [])) or "?"
        found.append(
            (
                display_name,
                f"{module_name} imported as a NAMESPACE package (its __init__.py never ran) "
                f"from [{portions}]. A bare folder named '{module_name}' on sys.path -- usually "
                f"the repo root, because the server was started from the folder that contains "
                f"the checkouts -- is shadowing the installed package, so every "
                f"'from {module_name} import ...' will fail with '(unknown location)'. "
                f"Fix: pull the checkout and re-run update_env.bat (pip install -e), then restart.",
            )
        )
    return found


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

    The git subprocesses, one per repo, are run in parallel via a thread
    pool (each call is independent).
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
