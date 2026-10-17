"""Validate and combine disjoint screening fragments without modifying either."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import write_json
from .constants import ACTIVITY_ENDPOINTS
from .download import sha256_file
from .metrics import soft_threshold_rae


def validate_fragment(frame: pd.DataFrame, train: pd.DataFrame) -> set[int]:
    keys = ["Molecule_Name", "endpoint"]
    columns = [*keys, "parent", "cluster", "fold", "y_true", "conf_low", "conf_high"]
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError("Empty or duplicate OOF fragment")
    if not np.isfinite(frame[["y_true", "conf_low", "conf_high", "y_pred"]]).all(axis=None):
        raise ValueError("Non-finite OOF fragment")
    folds = set(frame.fold.unique())
    if not folds <= set(train.fold.unique()):
        raise ValueError("Unknown fold in fragment")
    expected = []
    for endpoint in ACTIVITY_ENDPOINTS:
        selected = train.loc[train.fold.isin(folds) & train[endpoint].notna()]
        part = selected[["Molecule_Name", "parent", "cluster", "fold"]].copy()
        part["endpoint"] = endpoint
        part["y_true"] = selected[endpoint]
        part["conf_low"] = selected[endpoint + "_conf_low"]
        part["conf_high"] = selected[endpoint + "_conf_high"]
        expected.append(part)
    try:
        pd.testing.assert_frame_equal(
            frame[columns].sort_values(keys).reset_index(drop=True),
            pd.concat(expected)[columns].sort_values(keys).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-10,
        )
    except AssertionError as exc:
        raise ValueError("Fragment rows, labels, or folds do not match prepared data") from exc
    return {int(fold) for fold in folds}


def merge_recovered(original_config: Path, recovered: Path, output: Path, audit_path: Path) -> dict:
    audit = json.loads(audit_path.read_text())
    if audit.get("prediction_validation", {}).get("status") != "passed":
        raise ValueError("Original checkpoint prediction verification has not passed")
    for path, digest in audit["sha256_before_validation"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"Audited artifact changed: {path}")
    config = json.loads(original_config.read_text())
    original = Path(config["output_dir"])
    evidence = json.loads((recovered / "run_manifest.json").read_text())
    if not evidence.get("requested_folds_complete"):
        raise ValueError("Recovery did not finish its requested folds")
    ignored = {"output_dir", "selected_folds", "max_gpu_hours"}
    if {k: v for k, v in config.items() if k not in ignored} != {
        k: v for k, v in evidence["config"].items() if k not in ignored
    }:
        raise ValueError("Recovery changed experimental parameters")
    prepared = Path(config["prepared_dir"])
    digest = sha256_file(prepared / "training.csv")
    if digest != evidence["prepared_manifest"]["training_sha256"]:
        raise ValueError("Prepared corpus changed during recovery")
    train = pd.read_csv(prepared / "training.csv")
    parts, seen, artifacts = [], set(), {}
    for directory in (original, recovered):
        path = directory / "oof_predictions.csv"
        frame = pd.read_csv(path)
        folds = validate_fragment(frame, train)
        if directory == recovered and folds != set(evidence["completed_folds"]):
            raise ValueError("Recovery manifest and saved fold coverage disagree")
        if seen & folds:
            raise ValueError("Recovery fragments contain overlapping folds")
        seen |= folds
        parts.append(frame)
        artifacts[str(path)] = sha256_file(path)
        for fold in folds:
            checkpoint = directory / f"fold-{fold}.pt"
            artifacts[str(checkpoint)] = sha256_file(checkpoint)
    if seen != set(train.fold.unique()):
        raise ValueError("Recovery does not cover every outer fold")
    oof = pd.concat(parts, ignore_index=True)
    scores = {
        endpoint: soft_threshold_rae(f.y_true, f.y_pred, f.conf_low, f.conf_high)
        for endpoint in ACTIVITY_ENDPOINTS
        for f in [oof.loc[oof.endpoint == endpoint]]
    }
    result = {
        "endpoints": scores,
        "MA-ST-RAE": float(np.mean(list(scores.values()))),
        "purpose": "recovered screening; not nested model selection",
    }
    output.mkdir(parents=True, exist_ok=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    write_json(output / "metrics.json", result)
    write_json(
        output / "run_manifest.json",
        {
            "config": {**config, "output_dir": str(output)},
            "prepared_manifest": evidence["prepared_manifest"],
            "complete": True,
            "gpu_hours": 0,
            "recovery_manifest_sha256": sha256_file(recovered / "run_manifest.json"),
            "recovery_audit_path": str(audit_path),
            "recovery_audit_sha256": sha256_file(audit_path),
            "fragment_sha256": artifacts,
            "oof_sha256": sha256_file(output / "oof_predictions.csv"),
            "promotion_eligible": False,
            "provenance_limitation": "Original run has no startup/final manifest. Original config "
            "is supplied from the retained file, not attested at original run start. "
            "Original elapsed GPU time is unknown; its full reservation remains charged.",
        },
    )
    return result
