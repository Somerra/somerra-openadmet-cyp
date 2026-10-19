import json
from pathlib import Path

import pytest

from somerra_openadmet.configuration import resolve_config, source_hashes


def files(tmp_path, model=None, profile=None):
    recipe, data = tmp_path / "model.json", tmp_path / "data.json"
    recipe.write_text(
        json.dumps(
            model
            if model is not None
            else {"seed": 42, "data_sources": ["official_direct", "chembl37"]}
        )
    )
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
    assert len(provenance["data_catalog"]["sha256"]) == 64
    assert provenance["selected_data_sources"] == ["official_direct", "chembl37"]
    assert "auxiliary_sources" not in config
    assert "data_sources" not in config


@pytest.mark.parametrize(
    "model,profile",
    [
        ({"prepared_dir": "data"}, None),
        ({"output_dir": "runs/foo"}, None),
        ({"seed": 42}, None),
        ({"data_sources": "official_direct"}, None),
        ({"data_sources": ["chembl37"]}, None),
        ({"data_sources": ["official_direct", "official_direct"]}, None),
        ({"data_sources": ["official_direct", {}]}, None),
        ({"data_sources": ["official_direct", "unknown"]}, None),
        ({"data_sources": ["official_direct", "pubchem1851"]}, None),
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
            data_config=root / "configs/data.json",
            output_dir=Path("runs/test"),
        )
        assert "models" in config
        assert "auxiliary_dirs" not in config


def test_same_catalog_different_recipe_selections(tmp_path):
    catalog = {
        "prepared_dir": "primary",
        "auxiliary_sources": {
            "official_auxiliary": "not-downloaded-official",
            "chembl37": "not-downloaded-chembl",
            "pubchem1851": "not-downloaded-pubchem",
        },
    }
    for selected in (
        ["official_direct"],
        ["official_direct", "pubchem1851"],
        ["official_direct", "chembl37", "official_auxiliary"],
    ):
        kwargs = files(tmp_path, {"data_sources": selected}, catalog)
        config, provenance = resolve_config(**kwargs)
        expected = {name: catalog["auxiliary_sources"][name] for name in selected[1:]}
        assert config.get("auxiliary_dirs", []) == list(expected.values())
        assert provenance["auxiliary_sources"] == expected
        assert provenance["selected_data_sources"] == selected
