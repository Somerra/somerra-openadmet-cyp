import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from somerra_openadmet.constants import ACTIVITY_ENDPOINTS, TEST_FILE
from somerra_openadmet.data_sources.common import write_json
from somerra_openadmet.download import sha256_file
from somerra_openadmet.reference_member_data import load_frames, submission_selection_bundle


@pytest.fixture
def original_bundle(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    official = pd.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c"],
            "parent": ["C", "CC", "CCC"],
            "partition": ["train", "validation", "internal_test"],
        }
    )
    official.to_csv(root / "assignments.csv", index=False)
    for ep in ACTIVITY_ENDPOINTS:
        official[ep] = [3.0, 4.0, 5.0]
    official.to_csv(root / "official.csv", index=False)
    pool = pd.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c", "aux_old_val", "aux_test", "aux_blind", "aux_other"],
            "parent": ["C", "CC", "CCC", "CC", "CCC", "CCCC", "CCO"],
        }
    )
    for ep in ACTIVITY_ENDPOINTS:
        pool[ep] = [3.0, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan]
    pool["aux"] = 2
    pool["sample_weight"] = 1.0
    pool.to_csv(root / "pool.csv", index=False)
    pd.DataFrame({"Molecule_Name": ["test"], "parent": ["CCCC"]}).to_csv(
        root / "test.csv", index=False
    )
    manifest = {
        "protocol": "molecular-60-20-20-v1",
        "exclusion_policy": "exact_parent",
        "tasks": list(ACTIVITY_ENDPOINTS) + ["aux"],
        "groups": {},
        "file_sha256": {
            n: sha256_file(root / n)
            for n in ["official.csv", "pool.csv", "test.csv", "assignments.csv"]
        },
    }
    write_json(root / "manifest.json", manifest)
    return root


def test_submission_mapping_restores_validation_but_excludes_new_holdout_everywhere(
    original_bundle, tmp_path
):
    original_hash = sha256_file(original_bundle / "manifest.json")
    out = tmp_path / "new"
    submission_selection_bundle(original_bundle, out)
    train, val, _, manifest = load_frames(out)
    assert set(train.Molecule_Name) == {"a", "b", "aux_old_val", "aux_other"}
    assert list(val.Molecule_Name) == ["c"]
    assert manifest["partition_rows"] == {"train": 2, "validation": 1}
    assert sha256_file(original_bundle / "manifest.json") == original_hash
    assert not set(train.parent) & set(val.parent)
    assert (out / "test.csv").read_bytes() == (original_bundle / "test.csv").read_bytes()
    with pytest.raises(ValueError, match="retain its validation"):
        load_frames(out, final=True)
    official = pd.read_csv(out / "official.csv")
    official.loc[official.Molecule_Name == "c", "partition"] = "train"
    official.to_csv(out / "official.csv", index=False)
    manifest["file_sha256"]["official.csv"] = sha256_file(out / "official.csv")
    write_json(out / "manifest.json", manifest)
    with pytest.raises(ValueError, match="partition mapping"):
        load_frames(out)


def test_export_selected_checkpoint_is_explicit_and_retains_holdout(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from somerra_openadmet import reference_member as member

    run, bundle, raw = [tmp_path / n for n in ["run", "bundle", "raw"]]
    for path in [run, bundle, raw]:
        path.mkdir()
    test = pd.DataFrame({"Molecule_Name": ["x", "y"], "SMILES": ["C", "CC"]})
    test.to_csv(raw / TEST_FILE, index=False)
    test.assign(parent=test.SMILES).to_csv(bundle / "test.csv", index=False)
    write_json(bundle / "manifest.json", {})
    record = {
        "complete": True,
        "final": False,
        "completed_epochs": 100,
        "epoch_limit": 100,
        "best_epoch": 88,
        "best.ckpt_sha256": "best-digest",
        "recipe": "full",
        "tasks": list(ACTIVITY_ENDPOINTS),
        "data_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "data_manifest": {
            "test_sha256": sha256_file(raw / TEST_FILE),
            "protocol": "submission-train-plus-validation-v1",
            "validation_role": "former internal test",
        },
    }
    write_json(run / "run_manifest.json", record)
    calls = []
    monkeypatch.setattr(member, "load_frames", lambda *a, **kw: calls.append(kw["final"]))
    monkeypatch.setattr(
        member, "load_model", lambda *a: SimpleNamespace(eval=lambda: "selected-best")
    )
    monkeypatch.setattr(member, "prediction_loader", lambda x: x)

    def predict(model, *args):
        assert model == "selected-best"
        return pd.DataFrame({ep: [3.0, 4.0] for ep in ACTIVITY_ENDPOINTS})

    monkeypatch.setattr(member, "predict", predict)
    with pytest.raises(ValueError, match="Final training incomplete"):
        member.export(run, bundle, raw, tmp_path / "bad")
    out = tmp_path / "predictions"
    member.export(run, bundle, raw, out, selected_validation=True)
    metadata = json.loads((out / "manifest.json").read_text())
    assert calls == [False]
    assert metadata["checkpoint"] == "best.ckpt" and metadata["selected_epoch"] == 88
    assert metadata["open_source_code"] is False and metadata["model_report_link"] == ""
    assert metadata["validation"]["valid"]
    record["data_manifest"]["protocol"] = "molecular-60-20-20-v1"
    write_json(run / "run_manifest.json", record)
    with pytest.raises(ValueError, match="wrong protocol"):
        member.export(run, bundle, raw, tmp_path / "wrong", selected_validation=True)
