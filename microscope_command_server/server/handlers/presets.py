"""MMPRESET: read or apply one Micro-Manager ConfigGroup preset.

Used by QPSC to keep Micro-Manager's objective preset in step with the objective chosen
in QPSC, so the pixel-size preset bound to it applies (an LC-PolScope objective is
selected through a DemoCamera "DObjective" labelling device that a hand-turned nosepiece
never changes). Generic on purpose: group and preset names are whatever a lab's
Micro-Manager configuration calls them, so nothing here assumes a name.

Request: 4-byte big-endian length + UTF-8 JSON
    {"action": "get" | "set", "group": str, "preset": str}
Reply: 4-byte big-endian length + UTF-8 JSON
    {"group", "preset", "current", "group_exists", "preset_exists",
     "groups" (when the group is missing), "presets" (the group's presets),
     "applied" (set only), "pixel_size_um" (after a set), "error" (on refusal)}

A set is refused while an acquisition runs or before the connection is configured: a
preset can move a motorized turret.
"""

import json
import logging
import struct

logger = logging.getLogger(__name__)


def _as_list(vec):
    try:
        return [str(v) for v in vec]
    except TypeError:
        return [str(vec.get(i)) for i in range(vec.size())]


def preset_report(core, group: str, preset: str) -> dict:
    """What Micro-Manager knows about ``group`` and ``preset``, changing nothing."""
    report = {"group": group, "preset": preset}
    groups = _as_list(core.get_available_config_groups())
    report["group_exists"] = group in groups
    if not report["group_exists"]:
        report["groups"] = groups
        report["preset_exists"] = False
        report["current"] = None
        return report
    presets = _as_list(core.get_available_configs(group))
    report["presets"] = presets
    report["preset_exists"] = preset in presets
    try:
        report["current"] = str(core.get_current_config(group))
    except Exception as exc:
        logger.debug("MMPRESET: current preset of '%s' unreadable: %s", group, exc)
        report["current"] = None
    return report


def apply_preset(hardware, group: str, preset: str) -> dict:
    """Apply ``preset`` of ``group`` if both exist; report before and after."""
    core = hardware.core
    report = preset_report(core, group, preset)
    if not report["preset_exists"]:
        report["applied"] = False
        report["error"] = (
            f"Micro-Manager has no group '{group}'"
            if not report["group_exists"]
            else f"Micro-Manager group '{group}' has no preset '{preset}'"
        )
        return report
    if report.get("current") == preset:
        report["applied"] = False
        return report
    hardware.apply_config_preset(group, preset)
    report["applied"] = True
    report["previous"] = report.get("current")
    try:
        report["current"] = str(core.get_current_config(group))
    except Exception:
        report["current"] = preset
    try:
        report["pixel_size_um"] = float(core.get_pixel_size_um())
    except Exception:
        report["pixel_size_um"] = None
    return report


def handle_mmpreset(conn, client, hardware, settings, **kwargs):
    """Read or apply one Micro-Manager preset (see the module docstring)."""
    report: dict
    try:
        header = conn.recv(4)
        while len(header) < 4:
            more = conn.recv(4 - len(header))
            if not more:
                break
            header += more
        length = struct.unpack("!I", header)[0] if len(header) == 4 else 0
        body = b""
        while len(body) < length:
            chunk = conn.recv(length - len(body))
            if not chunk:
                break
            body += chunk
        request = json.loads(body.decode("utf-8")) if length else {}
        action = str(request.get("action", "get"))
        group = str(request.get("group", ""))
        preset = str(request.get("preset", ""))
        thread = kwargs.get("acquisition_thread")
        if not group:
            report = {"error": "no group named"}
        elif action == "set" and not kwargs.get("server_configured", False):
            report = {
                "group": group,
                "preset": preset,
                "applied": False,
                "error": "server not configured",
            }
        elif action == "set" and thread is not None and thread.is_alive():
            report = {
                "group": group,
                "preset": preset,
                "applied": False,
                "error": "an acquisition is running; the preset was not changed",
            }
        elif action == "set":
            report = apply_preset(hardware, group, preset)
            if report.get("applied"):
                logger.info(
                    "MMPRESET: [%s] '%s' -> '%s' (pixel size now %s um)",
                    group,
                    report.get("previous"),
                    preset,
                    report.get("pixel_size_um"),
                )
            elif report.get("error"):
                logger.warning("MMPRESET: %s", report["error"])
        else:
            report = preset_report(hardware.core, group, preset)
    except Exception as exc:  # report, never take the socket down
        logger.error("MMPRESET failed: %s", exc)
        report = {"error": str(exc)}
    payload = json.dumps(report).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)))
    conn.sendall(payload)
