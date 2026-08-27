"""Which "is this tissue?" test a given modality and objective actually use.

Pure configuration resolution, no hardware -- lives outside ``server.handlers`` for the
same reason ``server.tissue_search`` and ``server.focus_geometry`` do: importing the
handlers package pulls in ``microscope_control`` -> ``pycromanager``, which a unit test
has no business needing in order to check how a YAML file is read.

Why this is not just ``af_entry.get("texture_threshold")``
---------------------------------------------------------
``autofocus_<scope>.yml`` says what counts as tissue in TWO places, and only one of them
is per-objective:

* ``autofocus_settings[]`` -- per OBJECTIVE. Carries flat ``texture_threshold`` /
  ``tissue_area_threshold`` values alongside the sweep geometry.
* ``strategies.<name>.validity_params`` plus ``modalities.<mod>.overrides.validity_params``
  -- per MODALITY, and this is where the scopes that need it do their real tuning.

Reading only the first set silently discards the second, and the second is not decoration.
On LC-PolScope the modality binding sets ``tissue_area_threshold: 0.1`` with the comment
"a polarization image is mostly dark background even on a bright state, so the standard
20% tissue-area floor rejects valid fields" -- while the per-objective entry still reads
0.2. Both PPM and LC-PolScope also widen ``tissue_mask_range`` to ``[0.05, 0.95]``,
which the flat keys cannot express at all. A caller using the flat keys on those scopes
applies exactly the thresholds the config says reject good fields, and reports "no
tissue" while looking straight at it.

So this resolves the same chain ``acquisition/workflow.py`` resolves for the acquisition
path (schema v2: modality binding -> strategy -> validity_params, merged with the
binding's overrides), and falls back to the flat per-objective keys only when there is no
v2 binding to honour.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: Used when nothing in the YAML says otherwise -- the same defaults
#: ``microscope_imageprocessing.focus.validity.texture_and_area`` declares.
DEFAULT_VALIDITY_CHECK = "texture_and_area"
DEFAULT_VALIDITY_PARAMS: Dict[str, Any] = {
    "texture_threshold": 0.010,
    "tissue_area_threshold": 0.200,
    "rgb_brightness_threshold": 240.0,
}


def autofocus_doc_path(config_path: str) -> Optional[Path]:
    """``config_<scope>.yml`` -> the sibling ``autofocus_<scope>.yml``, or None."""
    if not config_path:
        return None
    try:
        path = Path(config_path)
        scope = path.stem.replace("config_", "")
        return path.parent / f"autofocus_{scope}.yml"
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("could not derive the autofocus yaml path from %r: %s", config_path, e)
        return None


def load_autofocus_doc(config_path: Optional[str]) -> Dict[str, Any]:
    """The whole parsed ``autofocus_<scope>.yml``, or an empty dict.

    The whole document rather than one objective's entry, because the modality bindings
    and the strategy library that carry the real thresholds live at the top level.
    """
    af_path = autofocus_doc_path(config_path) if config_path else None
    if af_path is None or not af_path.exists():
        return {}
    try:
        import yaml

        with open(af_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("could not read %s: %s", af_path, e)
        return {}


def _binding_for_modality(doc: Dict[str, Any], modality: Optional[str]) -> Optional[Dict[str, Any]]:
    """The ``modalities`` binding for this modality, matched longest-prefix-wins.

    Same matching rule as the acquisition path, so a modality named ``ppm_20x`` binds to
    the ``ppm`` entry, and a scope that declares both ``bf`` and ``bf_if`` gets the more
    specific one.
    """
    bindings = doc.get("modalities") or {}
    if not modality or not isinstance(bindings, dict):
        return None
    wanted = str(modality).lower()
    best = None
    best_len = 0
    for key, binding in bindings.items():
        key_str = str(key).lower()
        if wanted.startswith(key_str) and len(key_str) > best_len and isinstance(binding, dict):
            best = binding
            best_len = len(key_str)
    return best


def resolve_validity(
    doc: Dict[str, Any], modality: Optional[str], objective: Optional[str]
) -> Tuple[str, Dict[str, Any]]:
    """The validity check name and its parameters for this modality and objective.

    Order of preference, highest first:

    1. The schema-v2 chain: ``modalities.<mod>.strategy`` -> ``strategies.<name>``, with
       ``modalities.<mod>.overrides.validity_params`` merged over the strategy's own.
       This is what the acquisition path uses, so both agree about what "has content"
       means.
    2. The flat per-objective ``autofocus_settings`` keys, for a v1 document or a
       modality with no binding.
    3. The shipped ``texture_and_area`` defaults.

    Returns the check name even when it is one the caller cannot use (``always_false``
    on a manual-only strategy, say) -- deciding what to do about that belongs to the
    caller, not here.
    """
    if isinstance(doc, dict):
        binding = _binding_for_modality(doc, modality)
        strategies = doc.get("strategies") or {}
        if binding is not None and isinstance(strategies, dict):
            strategy_name = binding.get("strategy") or "dense_texture"
            entry = strategies.get(strategy_name)
            if isinstance(entry, dict):
                name = entry.get("validity_check") or DEFAULT_VALIDITY_CHECK
                params = dict(entry.get("validity_params") or {})
                overrides = binding.get("overrides") or {}
                if isinstance(overrides, dict) and isinstance(
                    overrides.get("validity_params"), dict
                ):
                    params.update(overrides["validity_params"])
                return str(name), params
    return DEFAULT_VALIDITY_CHECK, _flat_objective_params(doc, objective)


def _flat_objective_params(doc: Dict[str, Any], objective: Optional[str]) -> Dict[str, Any]:
    """The v1 fallback: flat threshold keys off the objective's ``autofocus_settings`` entry."""
    entry: Dict[str, Any] = {}
    entries = (doc or {}).get("autofocus_settings") or []
    if isinstance(entries, list):
        if objective:
            for candidate in entries:
                if isinstance(candidate, dict) and candidate.get("objective") == objective:
                    entry = candidate
                    break
        if not entry and entries and isinstance(entries[0], dict):
            entry = entries[0]
    params = dict(DEFAULT_VALIDITY_PARAMS)
    for key in DEFAULT_VALIDITY_PARAMS:
        if key in entry:
            try:
                params[key] = float(entry[key])
            except (TypeError, ValueError):
                logger.warning("ignoring non-numeric %s=%r in autofocus_settings", key, entry[key])
    return params
