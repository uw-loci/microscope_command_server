"""A client cancel stops an acquisition's autofocus mid-search, not after it.

2026-11-11: a cancel during the focus survey went unanswered and the server was killed.
None of the acquisition's searches (pre-acquisition AF, survey, per-tile standard AF, drift
sweep) was given the cancel check, so a cancel waited for the search to finish; a standard
search with edge retries can run for minutes. The searches poll ``should_abort`` between Z
steps, restore Z and raise ``AutofocusAborted``, which must end the acquisition as a cancel.
"""

import pathlib
import re
import sys
from unittest.mock import MagicMock

import pytest

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "acquisition"
    / "workflow.py"
)
SOURCE = WORKFLOW.read_text()


def _calls(name):
    """The text of each call to ``name(`` in the workflow, excluding its definition."""
    out = []
    for m in re.finditer(r"(?<!def )\b" + re.escape(name) + r"\(", SOURCE):
        depth, i = 0, m.end() - 1
        while True:
            c = SOURCE[i]
            depth += c == "("
            depth -= c == ")"
            i += 1
            if depth == 0:
                break
        out.append(SOURCE[m.start() : i])
    return out


def _acquisition_body():
    start = SOURCE.index("def _run_pre_acquisition_autofocus(")
    return SOURCE[start:]


def test_every_acquisition_autofocus_polls_the_cancel():
    calls = [c for c in _calls("autofocus_with_manual_fallback") if c in _acquisition_body()]
    assert len(calls) >= 4, "pre-acquisition (2), survey and per-tile standard AF"
    for call in calls:
        assert "should_abort=ctx.is_cancelled" in call, call[:200]


def test_the_per_tile_drift_sweep_polls_the_cancel():
    tile = SOURCE[SOURCE.index("def _handle_tile_autofocus(") :]
    tile = tile[: tile.index("\ndef ")]
    sweeps = _calls("hardware.autofocus_sweep_drift_check")
    sweeps = [c for c in sweeps if c in tile]
    assert sweeps and all("should_abort=ctx.is_cancelled" in c for c in sweeps)
    assert "except AutofocusAborted as e:" in tile
    assert 'raise _AcquisitionCancelled("Cancelled during the drift check") from e' in tile


def test_the_survey_does_not_swallow_a_cancel():
    survey = SOURCE[SOURCE.index("def _run_focus_survey(") :]
    survey = survey[: survey.index("\ndef ")]
    cancel = survey.index("except _AcquisitionCancelled:")
    generic = survey.index("except Exception as af_err:")
    assert cancel < generic


@pytest.fixture
def workflow_module():
    mocks = {}
    for mod in [
        "pycromanager",
        "microscope_control",
        "microscope_control.hardware",
        "microscope_control.hardware.base",
        "microscope_control.hardware.pycromanager",
        "microscope_control.autofocus",
        "microscope_control.autofocus.core",
        "ppm_library",
        "ppm_library.imaging",
        "ppm_library.imaging.writer",
        "ppm_library.imaging.background",
        "skimage",
        "skimage.filters",
    ]:
        if mod not in sys.modules:
            mocks[mod] = MagicMock()
            sys.modules[mod] = mocks[mod]

    class AutofocusAborted(Exception):
        pass

    base = sys.modules["microscope_control.hardware.base"]
    saved = getattr(base, "AutofocusAborted", None)
    if isinstance(base, MagicMock):
        base.AutofocusAborted = AutofocusAborted
    from microscope_command_server.acquisition import workflow

    yield workflow, base.AutofocusAborted
    if isinstance(base, MagicMock) and saved is not None:
        base.AutofocusAborted = saved
    for mod in mocks:
        del sys.modules[mod]


def test_an_aborted_search_ends_the_acquisition_as_a_cancel(workflow_module):
    workflow, aborted = workflow_module
    hardware = MagicMock()
    hardware.autofocus.side_effect = aborted("client cancel")

    with pytest.raises(workflow._AcquisitionCancelled):
        workflow.autofocus_with_manual_fallback(
            hardware=hardware,
            logger=MagicMock(),
            request_manual_focus=None,
            max_retries=0,
            should_abort=lambda: True,
        )
    assert hardware.autofocus.call_args.kwargs["should_abort"]() is True
