"""Timepoint scheduler for time-lapse acquisitions.

Imported by workflow.py's T-outer loop and by acquire_time_lapse in
stack_timelapse.py to give both paths the same drift-bounded timepoint
pacing semantics.

The scheduler anchors all timepoint start times to a fixed t0 captured at
the start of the acquisition so slow iterations do NOT accumulate drift:
timepoint N is scheduled for t0 + N*interval_seconds. A missed target
(acq_time > interval) is logged as a warning and the scheduler returns
immediately; the next target is still anchored to t0, so the drift is
bounded to a single interval.

The sleep-with-cancellation loop is a direct port of stack_timelapse.py
L288-297 -- polls cancel_event every <=0.5s during the wait so the caller
can abort cleanly in the middle of a long interval.

ASCII-only per project policy. Do not use Unicode in code, logging, or
comments -- this module runs on Windows cp1252 as well as Linux/WSL.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional


class TimepointScheduler:
    """Paces a time-lapse so each timepoint starts at t0 + t_idx*interval.

    Thread-safety: not thread-safe. The scheduler is intended to be driven
    from a single acquisition thread.

    Example::

        scheduler = TimepointScheduler(
            t0_monotonic=time.monotonic(),
            interval_seconds=30.0,
            logger=my_logger,
        )
        for t_idx in range(n_timepoints):
            if t_idx > 0:
                scheduler.wait_until(t_idx, cancel_event=is_cancelled)
            # ... acquire timepoint t_idx ...
    """

    # Granularity of the cancellation-poll sleep. Matches stack_timelapse.py
    # L297: each iteration of the wait loop sleeps at most this many seconds
    # so a cancel request is noticed within the next poll.
    _POLL_INTERVAL_SEC = 0.5

    def __init__(
        self,
        t0_monotonic: float,
        interval_seconds: float,
        logger: logging.Logger,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError(f"interval_seconds must be >= 0 (got {interval_seconds})")
        self.t0 = float(t0_monotonic)
        self.interval_seconds = float(interval_seconds)
        self.logger = logger
        self._clock = clock
        self._sleep = sleep
        self.overdue_count = 0

    def wait_until(
        self,
        t_idx: int,
        cancel_event: Optional[Any] = None,
    ) -> float:
        """Block until wall time reaches t0 + t_idx * interval_seconds.

        Args:
            t_idx: timepoint index (zero-based).
            cancel_event: optional cancellation signal. May be a
                ``threading.Event``-like object (checked via ``is_set()``)
                or a zero-arg callable returning a bool. ``None`` disables
                cancellation polling.

        Returns:
            Actual delay (seconds) that was slept. ``0`` if the target
            had already passed (overdue -- acq_time > interval) or if
            cancellation was detected.
        """
        if t_idx < 0:
            raise ValueError(f"t_idx must be >= 0 (got {t_idx})")

        target = self.t0 + t_idx * self.interval_seconds
        now = self._clock()
        delay = target - now
        if delay <= 0:
            if self.interval_seconds > 0 and t_idx > 0:
                self.overdue_count += 1
                self.logger.warning(
                    "TimepointScheduler: timepoint %d overdue by %.3fs "
                    "(acq_time > interval); continuing immediately",
                    t_idx,
                    -delay,
                )
            return 0.0

        sleep_end = now + delay
        check_cancel = _normalize_cancel(cancel_event)
        while True:
            if check_cancel():
                self.logger.info(
                    "TimepointScheduler: cancellation detected while " "waiting for timepoint %d",
                    t_idx,
                )
                return 0.0
            remaining = sleep_end - self._clock()
            if remaining <= 0:
                break
            self._sleep(min(self._POLL_INTERVAL_SEC, remaining))
        return delay


def _normalize_cancel(
    cancel_event: Optional[Any],
) -> Callable[[], bool]:
    """Return a zero-arg predicate for cancellation.

    Accepts None, a threading.Event-like object (is_set()), or any zero-arg
    callable. Keeps the wait loop agnostic to which style the caller uses.
    """
    if cancel_event is None:
        return _never_cancelled
    if hasattr(cancel_event, "is_set"):
        return cancel_event.is_set
    if callable(cancel_event):
        return cancel_event
    raise TypeError(
        "cancel_event must be None, threading.Event-like, or zero-arg callable; "
        f"got {type(cancel_event).__name__}"
    )


def _never_cancelled() -> bool:
    return False


__all__ = ["TimepointScheduler"]
