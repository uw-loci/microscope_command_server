"""Control flow of the STRMAFZ special modes (profiling, approach-from-safe-Z).

Both modes replace the edge-retry walk rather than seeding it, which they express by
setting ``max_attempts = 0`` so the retry loop's body never runs. That interacts with a
``for``/``else``: a loop whose body never executes still completes *without a break*, so the
``else`` branch fires. It referenced ``result``, which only the loop body binds.

On 2026-08-26 that surfaced as ``FAILED:local variable 'result' referenced before
assignment`` -- after a full 44 s traverse had already been performed, dumped, and thrown
away. The traverse is the expensive and physically consequential part, so losing its result
to a control-flow slip is the failure worth pinning.

These tests exercise the pattern directly rather than the handler, which needs a live
MMCore. The point is the branch, not the hardware.
"""

import pytest


def _run_loop(max_attempts, body_result="from-loop", preset=None):
    """Reproduces the handler's retry-loop control flow.

    Mirrors the real structure: `final_result` may be preset by a special mode, the loop body
    binds `result`, and the for/else assigns it afterwards.
    """
    final_result = preset
    for _ in range(max_attempts):
        result = body_result
        final_result = result
        break
    else:
        # The guard under test.
        if max_attempts > 0:
            final_result = result  # noqa: F821
    return final_result


def test_a_skipped_loop_does_not_touch_the_unbound_variable():
    # max_attempts == 0 is how profiling and approach modes skip the walk.
    assert _run_loop(0, preset="from-profile-mode") == "from-profile-mode"


def test_a_skipped_loop_preserves_the_mode_s_own_result():
    # The traverse already ran and produced this; the else branch must not clobber it.
    assert _run_loop(0, preset="traverse-complete") == "traverse-complete"


def test_the_normal_path_still_takes_the_loop_result():
    assert _run_loop(3, body_result="attempt-3") == "attempt-3"


def test_the_unguarded_form_is_what_failed():
    """Pins the actual defect, so the guard cannot be removed as redundant."""

    def unguarded(max_attempts):
        final_result = "preset"
        for _ in range(max_attempts):
            result = "loop"
            final_result = result
            break
        else:
            final_result = result  # noqa: F821
        return final_result

    with pytest.raises(UnboundLocalError):
        unguarded(0)
    # ...and is fine whenever the loop actually runs, which is why it survived review.
    assert unguarded(2) == "loop"
