"""``channel_reduction`` reaches the standard AF scan, not just streaming AF.

The key was added to autofocus_PPM.yml on 2026-09-12 to stop achromatic debris
outscoring stained tissue, but only streaming_focus.py read it. Every failure on the
2026-09-24 run came from the standard scan, which was still scoring an equal mean -- so
the configured mitigation was inert exactly where it was needed.

These pin the wiring by reading the source, because exercising the real loader needs
pycromanager and a microscope.
"""

import pathlib
import re

WORKFLOW = (
    pathlib.Path(__file__).resolve().parents[1]
    / "microscope_command_server"
    / "acquisition"
    / "workflow.py"
)
SOURCE = WORKFLOW.read_text()


def test_the_yaml_key_is_read_from_the_per_objective_entry():
    assert 'af_setting.get("channel_reduction"' in SOURCE


def test_it_is_carried_on_the_acquisition_context():
    assert "af_channel_reduction: Optional[str] = None" in SOURCE
    assert "ctx.af_channel_reduction = af_channel_reduction" in SOURCE


def test_every_autofocus_call_passes_it():
    # It must travel with score_metric: any AF call configured with a metric but no
    # reduction is a scan still using the camera default, which is the original bug.
    metric_sites = SOURCE.count("score_metric=ctx.af_score_metric,")
    reduction_sites = SOURCE.count("channel_reduction=ctx.af_channel_reduction,")
    assert metric_sites > 0
    assert reduction_sites == metric_sites


def test_the_resolved_reduction_is_logged_with_the_other_af_settings():
    # Its absence from the settings dump is what let the gap sit unnoticed: the line
    # listed score_metric and the sweep parameters but never the reduction.
    assert re.search(r"channel_reduction=\{af_channel_reduction", SOURCE)
