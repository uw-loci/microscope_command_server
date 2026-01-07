from __future__ import annotations
import re, shlex, json, math, os
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List
import numpy as np

try:
    import tifffile as tiff
except Exception:
    tiff = None

from PIL import Image

# ---- Mini DSL parsing ----


def _coerce(v: str):
    # try bool -> int -> float -> stripped string
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except:
            pass
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+\.?)([eE][+-]?\d+)?", s):
        try:
            return float(s)
        except:
            pass
    # strip matching quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def parse_steps(text: str) -> List[Dict[str, Any]]:
    # split by comma / newline / semicolon (outside quotes)
    parts = re.split(r"(?:\s*,\s*|\s*;\s*|\n+)", text.strip())
    steps: List[Dict[str, Any]] = []
    for part in parts:
        if not part.strip():
            continue
        tokens = shlex.split(part)
        if not tokens:
            continue
        cmd = tokens[0].lower()
        args: Dict[str, Any] = {}
        # special 2nd token like "read tif" => fmt=tif
        if len(tokens) >= 2 and "=" not in tokens[1]:
            args["fmt"] = tokens[1].lower()
            kv_tokens = tokens[2:]
        else:
            kv_tokens = tokens[1:]
        for tok in kv_tokens:
            if "=" in tok:
                k, v = tok.split("=", 1)
                args[k.lower()] = _coerce(v)
            else:
                # allow bare tokens like "jpg" after output
                if "fmt" not in args:
                    args["fmt"] = tok.lower()
        steps.append({"cmd": cmd, "args": args})
    return steps


# ---- Execution framework ----

OpFunc = Callable[["Context", Dict[str, Any]], None]
REGISTRY: Dict[str, OpFunc] = {}


def op(name: str):
    def deco(fn: OpFunc):
        REGISTRY[name] = fn
        return fn

    return deco


@dataclass
class Context:
    image: np.ndarray | None = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, cmd: str, args: Dict[str, Any]):
        self.history.append({"cmd": cmd, "args": args})


# ---- Utilities ----


def _read_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff") and tiff is not None:
        arr = tiff.imread(path)
    else:
        arr = np.array(Image.open(path))
    return arr


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.integer):
        # map full dtype range to 0..255
        info = np.iinfo(a.dtype)
        return np.clip(
            ((a.astype(np.float32) - info.min) / (info.max - info.min)) * 255, 0, 255
        ).astype(np.uint8)
    # float: assume 0..1 or general -> scale by robust min/max
    if a.size == 0:
        return a.astype(np.uint8)
    amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
    if math.isclose(amin, amax):
        return np.zeros_like(a, dtype=np.uint8)
    norm = (a - amin) / (amax - amin)
    return np.clip(norm * 255, 0, 255).astype(np.uint8)


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        return np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        if a.shape[-1] == 4:
            return a[..., :3]
        return a
    raise ValueError(f"Unsupported image shape {a.shape}")


# ---- Ops ----


@op("read")
def op_read(ctx: Context, args: Dict[str, Any]):
    path = args.get("path") or args.get("file") or args.get("f")
    if not path:
        raise ValueError("read: provide path=<file>")
    page = int(args.get("page", 0))
    arr = _read_any(path)
    # If multi-page TIFF and page requested
    if (
        arr.ndim == 3
        and arr.dtype != np.uint8
        and page is not None
        and arr.shape[0] > 4
        and arr.shape[-1] not in (3, 4)
    ):
        # likely (pages, H, W) stack
        arr = arr[page]
    ctx.image = arr
    ctx.log("read", {"path": path, "page": page})


@op("whitebalance")
def op_whitebalance(ctx: Context, args: Dict[str, Any]):
    if ctx.image is None:
        raise ValueError("whitebalance: no image")
    img = _ensure_rgb(ctx.image).astype(np.float32)
    method = (args.get("method") or "grayworld").lower()
    if method == "gains":
        r = float(args.get("r", 1.0))
        g = float(args.get("g", 1.0))
        b = float(args.get("b", 1.0))
        gains = np.array([r, g, b], dtype=np.float32)
    else:  # grayworld
        means = img.reshape(-1, 3).mean(axis=0)
        gray = means.mean()
        gains = gray / np.maximum(means, 1e-6)
    wb = img * gains
    # simple clipping
    if np.issubdtype(ctx.image.dtype, np.integer):
        info = np.iinfo(ctx.image.dtype)
        wb = np.clip(wb, info.min, info.max).astype(ctx.image.dtype)
    else:
        wb = wb
    ctx.image = wb
    ctx.log("whitebalance", {"method": method, "gains": [float(g) for g in gains.tolist()]})


@op("shading")
def op_shading(ctx: Context, args: Dict[str, Any]):
    if ctx.image is None:
        raise ValueError("shading: no image")
    flat_path = args.get("flat")
    dark_path = args.get("dark")
    method = (args.get("method") or "divide").lower()
    eps = float(args.get("eps", 1e-6))
    img = ctx.image.astype(np.float32)

    flat = _read_any(flat_path).astype(np.float32) if flat_path else None
    dark = _read_any(dark_path).astype(np.float32) if dark_path else None

    if method == "subtract":
        if dark is None:
            raise ValueError("shading subtract: requires dark=<path>")
        corrected = img - dark
    else:  # divide (flat-field)
        if flat is None:
            raise ValueError("shading divide: requires flat=<path> (and optional dark=<path>)")
        if dark is None:
            dark = 0.0
        denom = np.maximum(flat - dark, eps)
        corrected = (img - dark) / denom
        # scale to preserve intensity
        scale = np.nanmean(denom)
        corrected = corrected * scale

    # keep original dtype if it was integer
    if np.issubdtype(ctx.image.dtype, np.integer):
        info = np.iinfo(ctx.image.dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(ctx.image.dtype)
    ctx.image = corrected
    ctx.log("shading", {"method": method, "flat": flat_path, "dark": dark_path, "eps": eps})


@op("resize")
def op_resize(ctx: Context, args: Dict[str, Any]):
    if ctx.image is None:
        raise ValueError("resize: no image")
    w = int(args.get("w") or args.get("width"))
    h = int(args.get("h") or args.get("height"))
    img8 = _to_uint8(ctx.image)
    pil = Image.fromarray(img8 if img8.ndim == 2 else _ensure_rgb(img8))
    out = pil.resize((w, h), Image.BICUBIC)
    ctx.image = np.array(out)
    ctx.log("resize", {"w": w, "h": h})


@op("output")
def op_output(ctx: Context, args: Dict[str, Any]):
    if ctx.image is None:
        raise ValueError("output: no image")
    fmt = (args.get("fmt") or "jpg").lower().replace("jpeg", "jpg")
    path = args.get("path") or f"output.{fmt}"
    quality = int(args.get("quality", 92))
    write_log = bool(args.get("write_log", True))

    arr8 = _to_uint8(_ensure_rgb(ctx.image))
    Image.fromarray(arr8).save(
        path, format="JPEG" if fmt == "jpg" else fmt.upper(), quality=quality
    )
    ctx.log("output", {"fmt": fmt, "path": path, "quality": quality})

    if write_log:
        meta_path = os.path.splitext(path)[0] + ".json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"history": ctx.history}, f, indent=2)


# ---- Runner ----


def run_pipeline(text: str) -> Context:
    ctx = Context()
    for step in parse_steps(text):
        cmd = step["cmd"]
        fn = REGISTRY.get(cmd)
        if not fn:
            raise ValueError(f"Unknown command: {cmd}")
        fn(ctx, step["args"])
    return ctx


if __name__ == "__main__":
    example = """
    read tif path="input.tif",
    whitebalance method=grayworld,
    shading flat="flat.tif" dark="dark.tif" method=divide,
    output jpg path="out.jpg" quality=92
    """
    run_pipeline(example)
