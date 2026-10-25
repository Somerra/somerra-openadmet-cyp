import pytest

from somerra_openadmet.download import sha256_file
from somerra_openadmet.neural_export import checkpoint_paths


def test_export_requires_complete_hashed_five_fold_ensemble(tmp_path):
    manifest = {
        "complete": True,
        "selected_folds": list(range(5)),
        "completed_folds": list(range(5)),
        "checkpoint_sha256_by_fold": {},
    }
    for fold in range(5):
        path = tmp_path / f"fold-{fold}.pt"
        path.write_bytes(f"checkpoint {fold}".encode())
        manifest["checkpoint_sha256_by_fold"][str(fold)] = sha256_file(path)
    assert len(checkpoint_paths(tmp_path, manifest)) == 5
    manifest["complete"] = False
    with pytest.raises(ValueError, match="five completed"):
        checkpoint_paths(tmp_path, manifest)
    manifest["complete"] = True
    (tmp_path / "fold-2.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        checkpoint_paths(tmp_path, manifest)
