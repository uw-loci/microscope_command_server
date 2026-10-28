"""Autofocus progress for the operator: each search's start, scored Z steps and result.

One process-wide log, fed by every autofocus the server runs: the stop-and-snap
searches report through ``hardware.autofocus_listener`` (set to :data:`AF_EVENTS`
at startup), and streaming autofocus reports from its own handler. QPSC reads the
log with AFTRACE on either socket, asking for the events after the last sequence
number it saw; it is how the Live Viewer and the acquisition progress dialog show
that a search is running and draw its focus curve. While an acquisition runs, the
same events are appended to ``af_trace.ndjson`` in its output folder, so every
curve can be looked at after the run.

Events are plain dicts, one JSON object each::

    {"seq": 41, "t": 1730148000.123, "type": "start", "run": 7, "method": "sweep",
     "context": "Tile 12 of 80", "z0": -312.4, "search_range": 30.0, ...}
    {"seq": 42, "t": ..., "type": "sample", "run": 7, "z": -327.4, "score": 1.8e5,
     "spread": 5120.0}
    {"seq": 60, "t": ..., "type": "end", "run": 7, "status": "ok", "z": -310.9,
     "elapsed_s": 21.4}

``method`` is ``sweep`` (the full search), ``drift_check`` (the short per-tile
search) or ``streaming``. ``status`` is ``ok``, ``failed``, ``aborted``, ``error``
or, for streaming, ``unavailable``. Searches run one at a time on the hardware;
a search started inside another (a fallback) is tracked on a stack so its
samples are not attributed to the outer one.
"""

import contextlib
import json
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Events kept for AFTRACE readers. A 100-step sweep with two edge retries is
#: about 300 events, so this holds the last dozen or so searches.
CAPACITY = 4000

#: Most events returned by one AFTRACE reply; the reader asks again for the rest.
MAX_EVENTS_PER_REPLY = 1000


def _clean(value: Any) -> Any:
    """JSON-safe value: non-finite floats become None, floats are rounded."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 4)
    if isinstance(value, (int, str)):
        return value
    try:
        f = float(value)
        return None if not math.isfinite(f) else round(f, 4)
    except (TypeError, ValueError):
        return str(value)


class AutofocusEventLog:
    """Ring buffer of autofocus events; also an autofocus listener for the hardware."""

    def __init__(self, capacity: int = CAPACITY):
        self._lock = threading.Lock()
        self._events: deque = deque(maxlen=capacity)
        self._seq = 0
        self._run = 0
        self._stack: List[Dict[str, Any]] = []  # start events of the searches running
        self._context: Optional[Dict[str, Any]] = None
        self._file = None

    # -- the hardware listener interface (microscope_control AutofocusListener) --

    def af_started(self, method: str, params: Dict[str, Any]) -> None:
        with self._lock:
            self._run += 1
            event = {"type": "start", "run": self._run, "method": method}
            if self._context:
                event.update(self._context)
            for key, value in (params or {}).items():
                event[key] = _clean(value)
            self._append(event)
            self._stack.append(event)

    def af_sample(self, z: float, score: float, spread: Optional[float] = None) -> None:
        with self._lock:
            if not self._stack:
                return
            event = {
                "type": "sample",
                "run": self._stack[-1]["run"],
                "z": _clean(z),
                "score": _clean(score),
            }
            if spread is not None:
                event["spread"] = _clean(spread)
            self._append(event)

    def af_finished(self, status: str, z: Optional[float], details: Dict[str, Any]) -> None:
        with self._lock:
            if not self._stack:
                return
            start = self._stack.pop()
            event = {
                "type": "end",
                "run": start["run"],
                "method": start["method"],
                "status": status,
                "z": _clean(z),
                "elapsed_s": round(time.time() - start["t"], 2),
            }
            for key, value in (details or {}).items():
                event[key] = _clean(value)
            self._append(event)

    # -- what the caller is focusing for --

    def set_context(self, label: str, **fields) -> None:
        """Labels the searches started from now on, e.g. ``"Tile 12 of 80"``.

        ``fields`` (tile index, X, Y, ...) are copied into each start event. An
        enclosing :meth:`context` block puts its own label back when it ends.
        """
        self._context = {"context": label, **{k: _clean(v) for k, v in fields.items()}}

    @contextlib.contextmanager
    def context(self, label: str, **fields):
        """:meth:`set_context` for the searches started inside the block."""
        previous = self._context
        self.set_context(label, **fields)
        try:
            yield
        finally:
            self._context = previous

    # -- the acquisition's record --

    def attach_file(self, path) -> None:
        """Appends every later event to ``path`` (NDJSON) until :meth:`detach_file`."""
        with self._lock:
            self._close_file()
            try:
                # Held open across the whole acquisition, closed by detach_file().
                self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115
            except OSError as e:
                logger.warning("Could not open the autofocus trace %s: %s", path, e)
                self._file = None

    def detach_file(self) -> None:
        with self._lock:
            self._close_file()

    # -- AFTRACE --

    def since(self, seq: int, limit: int = MAX_EVENTS_PER_REPLY) -> Dict[str, Any]:
        """Events after ``seq``, oldest first.

        ``seq`` is the newest event in the log, ``first`` the oldest still held (a
        reader that fell further behind than that has missed some), ``more`` says
        the reply was cut at ``limit``, and ``active`` is the start event of the
        search running now, or None.
        """
        with self._lock:
            newer = [e for e in self._events if e["seq"] > seq]
            more = len(newer) > limit
            return {
                "seq": self._seq,
                "first": self._events[0]["seq"] if self._events else self._seq + 1,
                "more": more,
                "active": dict(self._stack[-1]) if self._stack else None,
                "events": newer[:limit],
            }

    # -- internals (lock held) --

    def _append(self, event: Dict[str, Any]) -> None:
        self._seq += 1
        event["seq"] = self._seq
        event["t"] = round(time.time(), 3)
        self._events.append(event)
        if self._file is not None:
            try:
                self._file.write(json.dumps(event) + "\n")
                self._file.flush()
            except (OSError, ValueError) as e:
                logger.warning("Autofocus trace write failed; not writing more: %s", e)
                self._close_file()

    def _close_file(self) -> None:
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None


#: The server's one log. qp_server sets it as ``hardware.autofocus_listener``.
AF_EVENTS = AutofocusEventLog()
