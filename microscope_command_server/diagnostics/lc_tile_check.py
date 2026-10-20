"""Find LC-PolScope tiles whose retardance is lifted, and say which state frame did it.

A stitched retardance image from 2026-10-19 (565 tiles, 20x) had about fifteen
tiles scattered across the mosaic that were uniformly brighter than their
neighbours, tissue texture still visible inside them. The reconstruction
treats the five state frames of a tile as one radiometric set, so a whole-tile
lift means one frame of that set did not match the others: the crystals had
not settled, the lamp flickered, or the camera returned a frame taken under a
different state. The server log cannot tell these apart -- the per-tile stats
it records average the five states together -- but the raw state tiles can.

For every tile this reads the state frames and the retardance tile, and
compares each state's mean, as a ratio to State0, with that ratio's median
over the whole acquisition. A tile whose retardance floor (10th percentile) is
far above the run's typical floor is flagged, and the state whose ratio is
furthest from typical is named. The pattern across the flagged tiles points at
the cause:

* always the state right after the switch (State1), off in one direction:
  the LC settle time is too short;
* every state off together against State0, or random states: illumination
  flicker, or a stale or dropped frame;
* nothing off: the frames agree and the lift happens after capture.

Run it on the rig, in the server's Python environment, against one region's
tile folder (the one holding State0..State4 and retardance)::

    python -m microscope_command_server.diagnostics.lc_tile_check <tile folder>

It writes ``lc_tile_check.csv`` beside the tiles and prints a summary. Reading
every frame of a 565-tile run takes a few minutes.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

STATE_DIR = re.compile(r"^State\d+$")
TILECONFIG = re.compile(r"^\s*([^;#]+?)\s*;\s*;\s*\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)")

# A tile is flagged when its retardance floor sits this many robust standard
# deviations (1.4826 * MAD) above the run's median floor.
DEFAULT_Z = 4.0


def _numeric_key(path: Path) -> Tuple[int, str]:
    m = re.match(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else sys.maxsize, path.name)


def find_states(folder: Path) -> List[str]:
    """The State<n> subfolders, in numeric order."""
    names = [p.name for p in folder.iterdir() if p.is_dir() and STATE_DIR.match(p.name)]
    return sorted(names, key=lambda n: int(n[5:]))


def read_positions(folder: Path) -> Dict[str, Tuple[float, float]]:
    """Stage XY per tile file name, from TileConfiguration_Stage.txt when present."""
    out: Dict[str, Tuple[float, float]] = {}
    path = folder / "TileConfiguration_Stage.txt"
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = TILECONFIG.match(line)
        if m:
            out[Path(m.group(1)).name] = (float(m.group(2)), float(m.group(3)))
    return out


def _read(path: Path) -> Optional[np.ndarray]:
    import tifffile

    try:
        img = tifffile.imread(str(path))
    except Exception as exc:  # a truncated or locked file is reported, not fatal
        print(f"  could not read {path}: {exc}")
        return None
    img = np.asarray(img)
    if img.ndim > 2:
        img = img.reshape(-1, img.shape[-2], img.shape[-1])[0]
    return img.astype(np.float64)


def robust_z(values: np.ndarray) -> np.ndarray:
    """(value - median) / (1.4826 * MAD); zeros when the spread is zero."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    med = np.median(finite)
    mad = 1.4826 * np.median(np.abs(finite - med))
    if mad <= 0:
        return np.zeros_like(values)
    return (values - med) / mad


def measure(folder: Path, states: Sequence[str]) -> List[dict]:
    """One row per tile: each state's mean, and the retardance tile's floor and median."""
    first = folder / states[0]
    tiles = sorted(first.glob("*.tif"), key=_numeric_key)
    retardance_dir = folder / "retardance"
    positions = read_positions(folder)
    rows: List[dict] = []
    for n, tile in enumerate(tiles):
        row: dict = {"tile": tile.name}
        xy = positions.get(tile.name)
        if xy:
            row["stage_x_um"], row["stage_y_um"] = xy
        complete = True
        for state in states:
            img = _read(folder / state / tile.name)
            if img is None:
                complete = False
                break
            row[f"mean_{state}"] = float(img.mean())
        if not complete:
            continue
        ret = _read(retardance_dir / tile.name) if retardance_dir.is_dir() else None
        if ret is not None:
            row["ret_p10"] = float(np.percentile(ret, 10))
            row["ret_median"] = float(np.median(ret))
        rows.append(row)
        if (n + 1) % 50 == 0:
            print(f"  read {n + 1}/{len(tiles)} tiles")
    return rows


def analyse(rows: List[dict], states: Sequence[str], z_limit: float = DEFAULT_Z) -> List[dict]:
    """Adds ratio deviations, the worst state and a flag to each row; returns the flagged rows."""
    if not rows:
        return []
    ref = states[0]
    for state in states[1:]:
        ratios = np.array([r[f"mean_{state}"] / r[f"mean_{ref}"] for r in rows])
        med = float(np.median(ratios))
        for r, ratio in zip(rows, ratios):
            r[f"ratio_{state}"] = float(ratio)
            r[f"dev_{state}_pct"] = 100.0 * (float(ratio) / med - 1.0)
    floors = np.array([r.get("ret_p10", np.nan) for r in rows], dtype=float)
    z = robust_z(floors)
    flagged = []
    for r, zv in zip(rows, z):
        devs = {s: r[f"dev_{s}_pct"] for s in states[1:]}
        worst = max(devs, key=lambda s: abs(devs[s])) if devs else ""
        r["worst_state"] = worst
        r["worst_dev_pct"] = devs.get(worst, 0.0)
        r["ret_p10_z"] = float(zv) if np.isfinite(zv) else ""
        r["flagged"] = bool(np.isfinite(zv) and zv > z_limit)
        if r["flagged"]:
            flagged.append(r)
    return flagged


def summarise(rows: List[dict], flagged: List[dict], states: Sequence[str]) -> str:
    lines = [
        f"{len(rows)} tiles read; {len(flagged)} flagged (retardance floor far above typical)."
    ]
    if rows:
        typical = {s: float(np.median([abs(r[f"dev_{s}_pct"]) for r in rows])) for s in states[1:]}
        lines.append(
            "Typical |deviation| of each state's ratio to "
            + states[0]
            + ": "
            + ", ".join(f"{s} {v:.2f}%" for s, v in typical.items())
        )
    if flagged:
        tally: Dict[str, int] = {}
        for r in flagged:
            tally[r["worst_state"]] = tally.get(r["worst_state"], 0) + 1
        lines.append(
            "Furthest-off state in flagged tiles: "
            + ", ".join(f"{s} x{n}" for s, n in sorted(tally.items(), key=lambda kv: -kv[1]))
        )
        lines.append("Flagged tiles (deviation of each state's ratio from the run median):")
        for r in sorted(flagged, key=lambda r: -float(r["ret_p10_z"] or 0)):
            devs = "  ".join(f"{s} {r[f'dev_{s}_pct']:+.2f}%" for s in states[1:])
            lines.append(f"  {r['tile']:>10}  floor z {float(r['ret_p10_z']):5.1f}   {devs}")
    return "\n".join(lines)


def write_csv(rows: List[dict], path: Path) -> None:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="tile folder holding State0..State4 and retardance")
    ap.add_argument("--z", type=float, default=DEFAULT_Z, help="flag threshold (robust z)")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    states = find_states(folder)
    if len(states) < 2:
        print(f"No State<n> folders under {folder}")
        return 2
    print(f"States: {', '.join(states)}")
    rows = measure(folder, states)
    flagged = analyse(rows, states, args.z)
    out = folder / "lc_tile_check.csv"
    write_csv(rows, out)
    print(summarise(rows, flagged, states))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
