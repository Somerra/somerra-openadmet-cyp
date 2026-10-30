"""Bind completed frozen experiments to current data, features, and recipe."""

import json
from pathlib import Path

import pandas as pd
import torch

from .constants import ACTIVITY_ENDPOINTS
from .download import sha256_file
from .frozen_benchmark import PROTOCOL, RECIPE


def completed_head(directory, expected=None):
    directory = Path(directory)
    if not directory.exists():
        return False
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if (
        not manifest.get("pilot_complete")
        or not manifest.get("requested_folds_complete")
        or manifest.get("complete")
        or manifest.get("encoder_trainable") is not False
        or sha256_file(directory / "fold-0.pt") != manifest["checkpoint_sha256"]
        or sha256_file(directory / "oof_predictions.csv") != manifest["oof_sha256"]
    ):
        raise ValueError(f"Incomplete or changed fit: {directory}")
    saved = torch.load(directory / "fold-0.pt", map_location="cpu", weights_only=True)
    config = saved["config"]
    if (
        saved.get("schema_version") != 3
        or saved.get("protocol") != PROTOCOL
        or saved.get("tasks") != list(ACTIVITY_ENDPOINTS)
        or saved.get("recipe") != RECIPE
        or saved.get("fold") != 0
    ):
        raise ValueError("Changed frozen experiment recipe")
    for key, value in (expected or {}).items():
        actual = config.get(key, 1 if key == "train_conformers" else None)
        if key in ("cache_dir", "prepared_dir", "checkpoint"):
            actual, value = Path(actual).resolve(), Path(value).resolve()
        if actual != value:
            raise ValueError(f"Changed frozen fit configuration: {key}")
    prepared = Path(config["prepared_dir"])
    evidence = json.loads((prepared / "manifest.json").read_text())
    training_hash = sha256_file(prepared / "training.csv")
    if training_hash != evidence["training_sha256"] or training_hash != saved["training_sha256"]:
        raise ValueError("Changed training data for completed fit")
    frame = pd.read_csv(prepared / "training.csv", usecols=["Molecule_Name", "fold"])
    if (
        frame.loc[frame.fold != 0, "Molecule_Name"].tolist() != saved["fit_ids"]
        or frame.loc[frame.fold == 0, "Molecule_Name"].tolist() != saved["held_ids"]
    ):
        raise ValueError("Changed fit/holdout assignments")
    if sha256_file(Path(config["cache_dir"]) / "manifest.json") != saved["cache_manifest_sha256"]:
        raise ValueError("Changed feature cache for completed fit")
    # Validate cache contents too, not only their manifest's identity.
    cache_dir = Path(config["cache_dir"])
    root = json.loads((cache_dir / "manifest.json").read_text())
    if root.get("kind") == "atom_jepa_conformer_features":
        from .conformer_sensitivity import verify_files

        verify_files(cache_dir, root)
    else:
        from .frozen_benchmark import load_features

        full_frame = pd.read_csv(prepared / "training.csv")
        load_features(config, full_frame)
    return True


def validate_evaluation(directory, prepared, cache_dir):
    from .conformer_sensitivity import verify_files

    directory, cache_dir = Path(directory), Path(cache_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Incomplete evaluation")
    if manifest.get("training_sha256") != prepared["training_sha256"]:
        raise ValueError("Changed evaluation training data")
    if manifest.get("cache_manifest_sha256") != sha256_file(cache_dir / "manifest.json"):
        raise ValueError("Changed evaluation feature cache")
    verify_files(cache_dir, json.loads((cache_dir / "manifest.json").read_text()))
    verify_files(directory, manifest)
    for path, digest in manifest["checkpoints"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("Changed evaluation checkpoint")
        completed_head(Path(path).parent)
    return manifest
