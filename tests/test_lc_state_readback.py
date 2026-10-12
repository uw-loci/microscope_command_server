"""The acquisition-side half of the LC trace: what the crystals read back.

The calibration records what each state should be; this records what the
device says it is when the state is applied, once per acquisition, and says
where the settings came from. Both go to the log and into the derived tiles'
metadata, so an acquisition can never again be running a calibration nobody
applied without that fact being written down somewhere.
"""

import logging
from types import SimpleNamespace

import pytest

from microscope_command_server.modality.lcpolscope import (
    LC_READBACK_TOLERANCE_WAVES,
    record_state_readback,
)


class _Core:
    def __init__(self, values):
        self.values = values
        self.reads = []

    def get_property(self, device, prop):
        self.reads.append((device, prop))
        return self.values[(device, prop)]


def _ctx(modality="lcpolscope", recon=None):
    return SimpleNamespace(
        modality=modality,
        ppm_settings={"modalities": {"lcpolscope": {"reconstruction": recon or {}}}},
    )


def _entry(ch_id, lca_waves, lcb_waves):
    return {
        "id": ch_id,
        "device_properties": [
            {
                "device": "MeadowlarkLC",
                "property": "Retardance LC-A [in waves]",
                "value": lca_waves,
            },
            {
                "device": "MeadowlarkLC",
                "property": "Retardance LC-B [in waves]",
                "value": lcb_waves,
            },
        ],
    }


LCA = ("MeadowlarkLC", "Retardance LC-A [in waves]")
LCB = ("MeadowlarkLC", "Retardance LC-B [in waves]")


def test_reads_back_each_state_once_and_records_agreement(caplog):
    core = _Core({LCA: "0.2520", LCB: "0.4420"})
    hw = SimpleNamespace(core=core)
    ctx = _ctx(recon={"lc_calibrated_at": "2026-10-12T14:03:00", "lc_extinction_ratio": 267.3})
    log = logging.getLogger("t")

    with caplog.at_level(logging.INFO, logger="t"):
        record_state_readback(ctx, hw, _entry("State0", 0.2520, 0.4420), log)
        record_state_readback(ctx, hw, _entry("State0", 0.2520, 0.4420), log)

    rb = ctx._lc_state_readback
    assert set(rb) == {"State0"}
    assert rb["State0"]["Retardance LC-A [in waves]"] == {
        "commanded": "0.252",
        "read": "0.2520",
        "agrees": True,
    }
    assert len(core.reads) == 2, "a repeat visit to a state is not read again"
    assert "QPSC's LC calibration of 2026-10-12T14:03:00" in caplog.text
    assert "267.3" in caplog.text


def test_disagreement_is_recorded_and_warned(caplog):
    core = _Core({LCA: str(0.2520 + 5 * LC_READBACK_TOLERANCE_WAVES), LCB: "0.4420"})
    ctx = _ctx()
    log = logging.getLogger("t2")
    with caplog.at_level(logging.WARNING, logger="t2"):
        record_state_readback(
            ctx, SimpleNamespace(core=core), _entry("State1", 0.2520, 0.4420), log
        )
    rb = ctx._lc_state_readback["State1"]
    assert rb["Retardance LC-A [in waves]"]["agrees"] is False
    assert rb["Retardance LC-B [in waves]"]["agrees"] is True
    assert "NOT reproducing the configured state" in caplog.text


def test_unapplied_calibration_is_said_out_loud(caplog):
    """A config still on the shipped presets is legitimate, but must not be silent."""
    core = _Core({LCA: "0.25", LCB: "0.44"})
    log = logging.getLogger("t3")
    with caplog.at_level(logging.WARNING, logger="t3"):
        record_state_readback(
            _ctx(recon={}), SimpleNamespace(core=core), _entry("State0", 0.25, 0.44), log
        )
    assert "no QPSC LC calibration has been applied" in caplog.text


@pytest.mark.parametrize("modality", ["ppm", "brightfield", "widefield"])
def test_other_modalities_are_untouched(modality):
    ctx = _ctx(modality=modality)
    record_state_readback(
        ctx, SimpleNamespace(core=_Core({})), _entry("State0", 0.25, 0.44), logging.getLogger("t4")
    )
    assert not hasattr(ctx, "_lc_state_readback")


def test_unreadable_property_is_skipped_not_fatal(caplog):
    class Broken:
        def get_property(self, device, prop):
            raise RuntimeError("serial timeout")

    ctx = _ctx()
    log = logging.getLogger("t5")
    with caplog.at_level(logging.WARNING, logger="t5"):
        record_state_readback(
            ctx, SimpleNamespace(core=Broken()), _entry("State0", 0.25, 0.44), log
        )
    assert ctx._lc_state_readback["State0"] == {}
    assert "could not read back" in caplog.text
