"""Resolution of the tissue/background test from autofocus_<scope>.yml.

The cases that matter are the ones where the per-modality tuning and the flat
per-objective keys DISAGREE -- reading the wrong one is invisible until the search
reports "no tissue" on a field full of it.
"""

from microscope_command_server.server.focus_validity import (
    DEFAULT_VALIDITY_PARAMS,
    autofocus_doc_path,
    load_autofocus_doc,
    resolve_validity,
)

# A miniature of the shipped LC-PolScope document: the modality binding halves the area
# floor and widens the mask band, while the objective entry still carries the old flat
# values. See autofocus_LCPolScope.yml.
DOC = {
    "schema_version": 2,
    "autofocus_settings": [
        {
            "objective": "LCPS_OBJECTIVE_20X_POL_001",
            "texture_threshold": 0.005,
            "tissue_area_threshold": 0.2,
        }
    ],
    "strategies": {
        "dense_texture": {
            "validity_check": "texture_and_area",
            "validity_params": {
                "texture_threshold": 0.01,
                "tissue_area_threshold": 0.2,
                "tissue_mask_range": [0.1, 0.9],
            },
        },
        "manual_only": {"validity_check": "always_false", "validity_params": {}},
    },
    "modalities": {
        "lcpolscope": {
            "strategy": "dense_texture",
            "overrides": {
                "validity_params": {
                    "tissue_mask_range": [0.05, 0.95],
                    "texture_threshold": 0.005,
                    "tissue_area_threshold": 0.1,
                }
            },
        },
        "bf": {"strategy": "dense_texture"},
        "manual": {"strategy": "manual_only"},
    },
}


def test_modality_overrides_beat_the_flat_objective_keys():
    name, params = resolve_validity(DOC, "lcpolscope", "LCPS_OBJECTIVE_20X_POL_001")
    assert name == "texture_and_area"
    # 0.1, not the objective entry's 0.2 -- the config comment says 0.2 rejects valid fields.
    assert params["tissue_area_threshold"] == 0.1
    assert params["texture_threshold"] == 0.005
    # The widened band cannot be expressed by the flat keys at all.
    assert params["tissue_mask_range"] == [0.05, 0.95]


def test_binding_without_overrides_uses_the_strategy_defaults():
    _, params = resolve_validity(DOC, "bf", "LCPS_OBJECTIVE_20X_POL_001")
    assert params["tissue_area_threshold"] == 0.2
    assert params["tissue_mask_range"] == [0.1, 0.9]


def test_longest_prefix_wins():
    # "lcpolscope_20x" must bind to "lcpolscope", not fall through to no binding.
    _, params = resolve_validity(DOC, "lcpolscope_20x", None)
    assert params["tissue_area_threshold"] == 0.1


def test_manual_only_reports_its_check_rather_than_substituting_one():
    name, _ = resolve_validity(DOC, "manual", None)
    assert name == "always_false"


def test_unbound_modality_falls_back_to_the_objective_entry():
    name, params = resolve_validity(DOC, "confocal", "LCPS_OBJECTIVE_20X_POL_001")
    assert name == "texture_and_area"
    assert params["texture_threshold"] == 0.005
    assert params["tissue_area_threshold"] == 0.2
    assert "tissue_mask_range" not in params


def test_no_modality_falls_back_to_the_objective_entry():
    _, params = resolve_validity(DOC, None, "LCPS_OBJECTIVE_20X_POL_001")
    assert params["tissue_area_threshold"] == 0.2


def test_unknown_objective_uses_the_first_entry():
    _, params = resolve_validity(DOC, None, "NO_SUCH_OBJECTIVE")
    assert params["texture_threshold"] == 0.005


def test_empty_document_gives_the_shipped_defaults():
    name, params = resolve_validity({}, "lcpolscope", "whatever")
    assert name == "texture_and_area"
    assert params == DEFAULT_VALIDITY_PARAMS


def test_non_numeric_flat_value_is_ignored_not_propagated():
    doc = {"autofocus_settings": [{"objective": "o", "texture_threshold": "loose"}]}
    _, params = resolve_validity(doc, None, "o")
    assert params["texture_threshold"] == DEFAULT_VALIDITY_PARAMS["texture_threshold"]


def test_autofocus_path_is_the_config_sibling():
    path = autofocus_doc_path("/cfg/config_PPM.yml")
    assert path is not None
    assert path.name == "autofocus_PPM.yml"
    assert str(path.parent) == "/cfg"


def test_missing_file_loads_as_empty():
    assert load_autofocus_doc("/nowhere/config_Nope.yml") == {}
    assert load_autofocus_doc(None) == {}
