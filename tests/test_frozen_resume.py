import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS  # noqa: E402
from somerra_openadmet.download import sha256_file  # noqa: E402
from somerra_openadmet.frozen_benchmark import PROTOCOL, RECIPE, FrozenMLP  # noqa: E402
from somerra_openadmet.frozen_resume import completed_head, validate_evaluation  # noqa: E402


def write_json(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture
def experiment(tmp_path):
    prepared, cache, head, evaluation = [
        tmp_path / n for n in ("prepared", "features", "head", "evaluation")
    ]
    for folder in (prepared, cache, head, evaluation):
        folder.mkdir()
    pd.DataFrame({"Molecule_Name": ["held", "fit"], "fold": [0, 1]}).to_csv(
        prepared / "training.csv", index=False
    )
    evidence = {"training_sha256": sha256_file(prepared / "training.csv")}
    write_json(prepared / "manifest.json", evidence)
    np.save(cache / "embeddings.npy", np.zeros((2, 10, 768), dtype=np.float32))
    write_json(
        cache / "manifest.json",
        {
            "kind": "atom_jepa_conformer_features",
            "files": {"embeddings.npy": sha256_file(cache / "embeddings.npy")},
        },
    )
    saved = {
        "schema_version": 3,
        "protocol": PROTOCOL,
        "tasks": list(ACTIVITY_ENDPOINTS),
        "recipe": RECIPE,
        "fold": 0,
        "dimension": 768,
        "state_dict": FrozenMLP(768).state_dict(),
        "training_sha256": evidence["training_sha256"],
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "fit_ids": ["fit"],
        "held_ids": ["held"],
        "config": {
            "prepared_dir": str(prepared),
            "cache_dir": str(cache),
            "seed": 42,
            "representation": "invariant",
            "train_conformers": 2,
            "scaling": "standard",
        },
    }
    torch.save(saved, head / "fold-0.pt")
    (head / "oof_predictions.csv").write_text("y_pred\n1.0\n")
    write_json(
        head / "run_manifest.json",
        {
            "complete": False,
            "pilot_complete": True,
            "requested_folds_complete": True,
            "encoder_trainable": False,
            "checkpoint_sha256": sha256_file(head / "fold-0.pt"),
            "oof_sha256": sha256_file(head / "oof_predictions.csv"),
        },
    )
    (evaluation / "scores.csv").write_text("score\n1.0\n")
    write_json(
        evaluation / "manifest.json",
        {
            "complete": True,
            **evidence,
            "cache_manifest_sha256": saved["cache_manifest_sha256"],
            "checkpoints": {str(head / "fold-0.pt"): sha256_file(head / "fold-0.pt")},
            "files": {"scores.csv": sha256_file(evaluation / "scores.csv")},
        },
    )
    return prepared, cache, head, evaluation, evidence


def test_unchanged_experiment_resumes_and_changed_recipe_fails(experiment):
    _, cache, head, evaluation, evidence = experiment
    assert completed_head(head, {"seed": 42, "train_conformers": 2})
    validate_evaluation(evaluation, evidence, cache)
    with pytest.raises(ValueError, match="configuration: seed"):
        completed_head(head, {"seed": 43})


def test_completed_evaluation_rejects_changed_current_inputs(experiment, monkeypatch):
    _, cache, _, evaluation, evidence = experiment
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_atom_jepa_conformer_sensitivity as driver

    monkeypatch.setattr(driver, "ROOT", evaluation.parent)
    before = (evaluation / "scores.csv").read_bytes()
    driver.evaluate(None, None, evidence)
    with pytest.raises(ValueError, match="training data"):
        driver.evaluate(None, None, {"training_sha256": "changed"})
    assert (evaluation / "scores.csv").read_bytes() == before
    manifest = json.loads((cache / "manifest.json").read_text())
    manifest["new_representation"] = True
    write_json(cache / "manifest.json", manifest)
    with pytest.raises(ValueError, match="feature cache"):
        validate_evaluation(evaluation, evidence, cache)


def test_completed_head_rejects_updated_training_data_manifest(experiment):
    prepared, _, head, _, _ = experiment
    frame = pd.read_csv(prepared / "training.csv")
    frame["fold"] = [1, 0]
    frame.to_csv(prepared / "training.csv", index=False)
    write_json(
        prepared / "manifest.json", {"training_sha256": sha256_file(prepared / "training.csv")}
    )
    with pytest.raises(ValueError, match="training data"):
        completed_head(head)


def test_changed_cache_bytes_and_checkpoints_fail(experiment):
    _, cache, head, evaluation, evidence = experiment
    path = cache / "embeddings.npy"
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="Changed conformer artifact"):
        completed_head(head)
    path.write_bytes(original)
    with (head / "fold-0.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        validate_evaluation(evaluation, evidence, cache)
