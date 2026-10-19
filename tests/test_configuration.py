import json
from pathlib import Path

import pytest

from somerra_openadmet.configuration import resolve_config, source_hashes


def files(tmp_path, model=None, profile=None):
    recipe, data = tmp_path / "model.json", tmp_path / "data.json"
    recipe.write_text(json.dumps(model if model is not None else {"seed": 42}))
    data.write_text(
        json.dumps(
            profile
            if profile is not None
            else {
                "prepared_dir": "data/prepared/official-direct",
                "auxiliary_sources": {"chembl37": "data/external/chembl37"},
            }
        )
    )
    return dict(model_config=recipe, data_config=data, output_dir=tmp_path / "run")


def test_compose_preserves_source_identity_and_hashes(tmp_path):
    kwargs = files(tmp_path)
    config, provenance = resolve_config(**kwargs)
    assert config["seed"] == 42
    assert config["auxiliary_dirs"] == ["data/external/chembl37"]
    assert provenance["auxiliary_sources"] == {"chembl37": "data/external/chembl37"}
    assert len(provenance["model_recipe"]["sha256"]) == 64
    assert len(provenance["data_profile"]["sha256"]) == 64
    assert "auxiliary_sources" not in config


@pytest.mark.parametrize(
    "model,profile",
    [
        ({"prepared_dir": "data"}, None),
        ({"output_dir": "runs/foo"}, None),
        (None, {"prepared_dir": "data", "seed": 1}),
        (None, {"prepared_dir": "data", "auxiliary_sources": {"unknown": "x"}}),
        (None, {"prepared_dir": "data", "auxiliary_sources": {"chembl37": ""}}),
        (
            None,
            {
                "prepared_dir": "data",
                "auxiliary_sources": {"chembl37": "same", "pubchem1851": "same"},
            },
        ),
    ],
)
def test_reject_ambiguous_profiles(tmp_path, model, profile):
    with pytest.raises(ValueError):
        resolve_config(**files(tmp_path, model, profile))


def test_legacy_and_missing_arguments(tmp_path):
    config = tmp_path / "legacy.json"
    config.write_text('{"prepared_dir": "original", "seed": 42}')
    result, provenance = resolve_config(config)
    assert result == {"prepared_dir": "original", "seed": 42}
    assert "legacy_config" in provenance
    with pytest.raises(ValueError):
        resolve_config(config, output_dir=tmp_path)
    with pytest.raises(ValueError):
        resolve_config(model_config=config)


def test_source_hashes_cover_adapters():
    hashes = source_hashes()
    for source in ("official", "chembl", "pubchem", "common"):
        assert f"data_sources/{source}.py" in hashes
    assert "campaign.py" in hashes


def test_cpu_rejects_auxiliary_sources(tmp_path):
    from somerra_openadmet.campaign import evaluate

    with pytest.raises(ValueError, match="official-direct only"):
        evaluate(**files(tmp_path))


def test_maintained_recipes_compose():
    root = Path(__file__).parents[1]
    for recipe in (root / "configs/models").glob("*.json"):
        config, _ = resolve_config(
            model_config=recipe,
            data_config=root / "configs/data/official-direct.json",
            output_dir=Path("runs/test"),
        )
        assert "models" in config
