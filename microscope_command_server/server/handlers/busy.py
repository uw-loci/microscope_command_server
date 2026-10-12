"""Is any client's acquisition running right now?

ACQUIRE has had cross-client admission control since 2026-08-06, but nothing else did.
On 2026-10-12 the alignment gate's STRMAFZ drove the focus stage 30 um under an acquisition
thread that the server still considered running -- the client had abandoned it on a
timeout, the thread was parked at a manual-focus prompt, and the next ACQUIRE was then
refused as BUSY while the stage had already been moved out from under it.

Any handler that moves the stage on its own initiative asks here first. Kept free of
hardware imports so it can be tested anywhere.
"""

from typing import Any, Mapping, Optional

# The states in which the acquisition thread may still command the stage. Compared by
# name so this works with the server's enum and with the string mirrors the tests use.
_IN_PROGRESS = ("RUNNING", "CANCELLING")


def _state_name(state: Any) -> str:
    return str(getattr(state, "value", state))


def acquisition_in_progress(acquisition_states: Optional[Mapping[Any, Any]]) -> Optional[Any]:
    """Return the address whose acquisition is running or cancelling, else None."""
    if not acquisition_states:
        return None
    for addr, state in list(acquisition_states.items()):
        if _state_name(state) in _IN_PROGRESS:
            return addr
    return None


def busy_reply(addr: Any, what: str) -> str:
    """The FAILED reply for a stage-moving command refused while acquiring."""
    return f"FAILED:busy: an acquisition is running for {addr}; refusing to {what} under it"
