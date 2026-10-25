"""Resolve model-selected sources against one shared data-location catalog."""

import json
from pathlib import Path

from .download import sha256_file

DATA_FIELDS = {"prepared_dir", "auxiliary_dir", "auxiliary_dirs", "auxiliary_sources"}
SOURCE_NAMES = {"official_auxiliary", "scigantic_chembl", "scigantic_pubchem"}


def resolve_config(
    config_path: Path | None = None,
    *,
    model_config: Path | None = None,
    data_config: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[dict, dict]:
    """Paths inside JSON remain relative to the invocation directory, as in legacy runs."""
    if config_path is not None:
        if any(value is not None for value in (model_config, data_config, output_dir)):
            raise ValueError("Use either --config or model/data/output arguments, not both")
        return json.loads(config_path.read_text()), {
            "legacy_config": {"path": str(config_path), "sha256": sha256_file(config_path)}
        }
    if model_config is None or data_config is None or output_dir is None:
        raise ValueError("Provide --model-config, --data-config, and --output-dir together")
    model = json.loads(model_config.read_text())
    profile = json.loads(data_config.read_text())
    config, provenance = compose_config(model, profile, output_dir)
    provenance.update(
        {
            "model_recipe": {"path": str(model_config), "sha256": sha256_file(model_config)},
            "data_catalog": {"path": str(data_config), "sha256": sha256_file(data_config)},
        }
    )
    return config, provenance


def compose_config(model: dict, profile: dict, output_dir: Path) -> tuple[dict, dict]:
    """Resolve in memory for previews without creating trial directories."""
    if not isinstance(model, dict) or not isinstance(profile, dict):
        raise ValueError("Model recipe and data catalog must be JSON objects")
    model = dict(model)
    if set(model) & (DATA_FIELDS | {"output_dir"}):
        raise ValueError("Model recipes select data_sources by name, not paths or output_dir")
    if set(profile) - {"prepared_dir", "auxiliary_sources"}:
        raise ValueError("Data catalog may contain only prepared_dir and auxiliary_sources")
    if not isinstance(profile.get("prepared_dir"), str) or not profile["prepared_dir"]:
        raise ValueError("Data catalog requires a nonempty prepared_dir")
    sources = profile.get("auxiliary_sources", {})
    if not isinstance(sources, dict) or set(sources) - SOURCE_NAMES:
        raise ValueError(f"Unknown auxiliary source; choose from {sorted(SOURCE_NAMES)}")
    if any(not isinstance(path, str) or not path for path in sources.values()):
        raise ValueError("Auxiliary source paths must be nonempty strings")
    if len(set(sources.values())) != len(sources):
        raise ValueError("Each auxiliary source needs a distinct snapshot directory")
    selected = model.pop("data_sources", None)
    if (
        not isinstance(selected, list)
        or any(not isinstance(name, str) for name in selected)
        or len(set(selected)) != len(selected)
        or "official_direct" not in selected
    ):
        raise ValueError("Model data_sources must be a unique list including official_direct")
    missing = set(selected) - {"official_direct", *sources}
    if missing:
        raise ValueError(f"Selected data sources unavailable in catalog: {sorted(missing)}")
    selected_auxiliary = {name: sources[name] for name in selected if name != "official_direct"}
    config = {**model, "prepared_dir": profile["prepared_dir"], "output_dir": str(output_dir)}
    if selected_auxiliary:
        config["auxiliary_dirs"] = list(selected_auxiliary.values())
    provenance = {
        "selected_data_sources": selected,
        "auxiliary_sources": selected_auxiliary,
    }
    return config, provenance


def source_hashes() -> dict[str, str]:
    """Include source adapters in nested packages, without basename collisions."""
    root = Path(__file__).parent
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*.py"))}
