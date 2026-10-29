import hashlib
import json
import subprocess
from pathlib import Path

import pytest


def test_setup_verifies_source_and_weights_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import setup_atom_jepa as setup

    source, model = tmp_path / "source", tmp_path / "model"
    (source / "atom_jepa").mkdir(parents=True)
    model.mkdir()
    (source / "atom_jepa/api.py").write_text("# fixture\n")
    (model / "weights").write_bytes(b"fixture weights")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    pin = {
        "source_revision": revision,
        "model_revision": "fixture-revision",
        "source_files": {
            "api.py": hashlib.sha256((source / "atom_jepa/api.py").read_bytes()).hexdigest()
        },
        "files": {"weights": hashlib.sha256((model / "weights").read_bytes()).hexdigest()},
    }
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin))
    (model / "provenance.json").write_text(json.dumps(pin))
    monkeypatch.setattr(setup, "PIN", pin_path)
    assert setup.setup(source, model, verify_only=True)["verified"]
    (model / "weights").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        setup.setup(source, model)
    assert (model / "weights").read_bytes() == b"corrupted"


def test_verify_only_does_not_create_assets(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import setup_atom_jepa as setup

    with pytest.raises(FileNotFoundError):
        setup.setup(tmp_path / "missing-source", tmp_path / "missing-model", verify_only=True)
    assert not (tmp_path / "missing-source").exists()
