"""Persisted molecular folds and genuinely nested regression controls.

No blinded labels or leaderboard values enter training or model selection.
"""

from __future__ import annotations

import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Cluster import Butina
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

from .constants import ACTIVITY_ENDPOINTS, ACTIVITY_TRAIN_FILE, TEST_FILE
from .data import load_activity, load_test
from .download import sha256_file, verify_activity_data
from .ensemble import _metric_weights
from .features import featurize_smiles
from .metrics import soft_threshold_rae
from .models import make_regressor
from .validation import validate_submission


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parent_smiles(value: str, *, stereo: bool = False) -> str:
    mol = Chem.MolFromSmiles(value)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid molecular structure: {value!r}")
    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=stereo)


def cluster_ids(smiles: list[str], distance: float = 0.4) -> np.ndarray:
    """Deterministic Butina groups; input order is canonicalized by the caller."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    distances = []
    for i in range(1, len(fps)):
        distances.extend(1.0 - np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])))
    clusters = Butina.ClusterData(distances, len(fps), distance, isDistData=True)
    result = np.empty(len(fps), dtype=int)
    for group, indices in enumerate(clusters):
        result[list(indices)] = group
    return result


def balanced_folds(groups: np.ndarray, observed: np.ndarray, n_folds: int) -> np.ndarray:
    unique = np.unique(groups)
    if len(unique) < n_folds or n_folds < 2:
        raise ValueError("Need at least two folds and at least one molecular group per fold")
    counts = np.stack([observed[groups == group].sum(axis=0) for group in unique])
    sizes = np.asarray([(groups == group).sum() for group in unique])
    loads = np.zeros((n_folds, observed.shape[1] + 1))
    totals = np.maximum(np.r_[counts.sum(axis=0), sizes.sum()], 1)
    assignment = {}
    order = sorted(range(len(unique)), key=lambda i: (-int(counts[i].sum()), -sizes[i], i))
    for position, index in enumerate(order):
        weight = np.r_[counts[index], sizes[index]]
        costs = []
        for fold in range(n_folds):
            proposed = loads.copy()
            proposed[fold] += weight
            costs.append(float(np.sum((proposed / totals) ** 2)))
        fold = position if position < n_folds else int(np.argmin(costs))
        assignment[unique[index]] = fold
        loads[fold] += weight
    return np.asarray([assignment[group] for group in groups])


def prepare(data_dir: Path, output_dir: Path, n_folds: int = 5) -> dict:
    report = verify_activity_data(data_dir)
    if not report.valid:
        raise ValueError(f"Dataset verification failed: {report.errors}")
    train, test = load_activity(data_dir), load_test(data_dir)
    train["parent"] = train.SMILES.map(parent_smiles)
    test_parents = set(test.SMILES.map(parent_smiles))
    overlap = train.parent.isin(test_parents)
    excluded = train.loc[overlap, ["Molecule_Name", "parent"]]
    train = train.loc[~overlap].copy()
    parents = sorted(train.parent.unique())
    groups = dict(zip(parents, cluster_ids(parents), strict=True))
    train["cluster"] = train.parent.map(groups)
    train = train.sort_values(["parent", "Molecule_Name"]).reset_index(drop=True)
    train["fold"] = balanced_folds(
        train.cluster.to_numpy(), train[list(ACTIVITY_ENDPOINTS)].notna().to_numpy(), n_folds
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    train.to_csv(output_dir / "training.csv", index=False)
    excluded.to_csv(output_dir / "excluded_test_overlap.csv", index=False)
    manifest = {
        "schema_version": 1,
        "method": "ecfp4_2048_butina_distance_0.4",
        "n_folds": n_folds,
        "rows": len(train),
        "parents": len(parents),
        "clusters": len(set(groups.values())),
        "excluded_test_overlap": len(excluded),
        "input_sha256": {
            name: sha256_file(data_dir / name) for name in (ACTIVITY_TRAIN_FILE, TEST_FILE)
        },
        "training_sha256": sha256_file(output_dir / "training.csv"),
        "rdkit_version": version("rdkit"),
        "fold_label_counts": train.groupby("fold")[list(ACTIVITY_ENDPOINTS)].count().to_dict(),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def fit_predict(name, x, y, query, seed, n_jobs):
    if name.endswith(("_r2", "_r3")):
        name, radius = name.rsplit("_r", 1)
        start = 0 if radius == "2" else 2058
        x, query = x[:, start : start + 2058], query[:, start : start + 2058]
    if name == "constant":
        model = DummyRegressor(strategy="mean")
    elif name == "nearest":
        # Descriptors are excluded from fingerprint Jaccard distances.
        x, query = x[:, :2048].astype(bool), query[:, :2048].astype(bool)
        model = KNeighborsRegressor(n_neighbors=min(5, len(y)), metric="jaccard", n_jobs=n_jobs)
    else:
        model = make_regressor(name, seed, n_jobs)
    model.fit(x, y)
    return model, model.predict(query)


def nested_endpoint(frame, x, names, *, seed=42, n_jobs=8, inner_folds=3, search_samples=512):
    y = frame.y_true.to_numpy()
    low, high = frame.conf_low.to_numpy(), frame.conf_high.to_numpy()
    folds, groups = frame.fold.to_numpy(), frame.cluster.to_numpy()
    base = np.full((len(frame), len(names)), np.nan)
    blend = np.full(len(frame), np.nan)
    weights_by_fold = {}
    for fold in sorted(np.unique(folds)):
        fit, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(groups[fit]) & set(groups[val]):
            raise ValueError("Outer molecular groups overlap")
        inner = np.full((len(fit), len(names)), np.nan)
        splitter = GroupKFold(n_splits=inner_folds)
        for inner_fold, (a, b) in enumerate(splitter.split(x[fit], groups=groups[fit])):
            for column, name in enumerate(names):
                _, inner[b, column] = fit_predict(
                    name, x[fit[a]], y[fit[a]], x[fit[b]], seed + inner_fold, n_jobs
                )
        weights, _ = _metric_weights(
            inner, y[fit], low[fit], high[fit], seed=seed, search_samples=search_samples
        )
        weights_by_fold[str(fold)] = weights.tolist()
        for column, name in enumerate(names):
            _, base[val, column] = fit_predict(name, x[fit], y[fit], x[val], seed, n_jobs)
        blend[val] = base[val] @ weights
    if not np.isfinite(base).all() or not np.isfinite(blend).all():
        raise ValueError("Nested evaluation did not produce complete finite predictions")
    return base, blend, weights_by_fold


def evaluate(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    allowed = {
        "prepared_dir",
        "output_dir",
        "models",
        "seed",
        "n_jobs",
        "inner_folds",
        "embeddings_dir",
        "search_samples",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown configuration keys: {set(config) - allowed}")
    prepared = Path(config["prepared_dir"])
    manifest = json.loads((prepared / "manifest.json").read_text())
    if sha256_file(prepared / "training.csv") != manifest["training_sha256"]:
        raise ValueError("Training data or persisted split assignments changed")
    names = config.get("models", ["lightgbm", "extra_trees"])
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Specify at least two distinct models")
    aliases = {"lightgbm_r2", "lightgbm_r3", "extra_trees_r2", "ridge_r2"}
    if set(names) - {"lightgbm", "extra_trees", "ridge", "constant", "nearest"} - aliases:
        raise ValueError("Unsupported regression model")
    if set(names) & aliases and (not set(names) <= aliases or "embeddings_dir" in config):
        raise ValueError("Radius-specific controls must be used alone without embeddings")
    train = pd.read_csv(prepared / "training.csv")
    if train.groupby("cluster").fold.nunique().max() != 1:
        raise ValueError("A molecular cluster spans outer folds")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    x = featurize_smiles(train.SMILES)
    if set(names) & aliases:
        x = np.concatenate([x, featurize_smiles(train.SMILES, radius=3)], axis=1)
    embedding_manifest = None
    if "embeddings_dir" in config:
        embedding_dir = Path(config["embeddings_dir"])
        embedding_manifest = json.loads((embedding_dir / "manifest.json").read_text())
        if embedding_manifest["training_sha256"] != manifest["training_sha256"]:
            raise ValueError("Embeddings do not match training rows and split manifest")
        if sha256_file(embedding_dir / "embeddings.npy") != embedding_manifest["embeddings_sha256"]:
            raise ValueError("Embedding checksum mismatch")
        embeddings = np.load(embedding_dir / "embeddings.npy", allow_pickle=False)
        if len(embeddings) != len(train) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid embedding array")
        x = np.concatenate([x, embeddings], axis=1)
    rows, metrics, weights = [], {}, {}
    for endpoint in ACTIVITY_ENDPOINTS:
        mask = train[endpoint].notna()
        frame = train.loc[mask, ["Molecule_Name", "parent", "cluster", "fold"]].copy()
        frame["endpoint"] = endpoint
        frame["y_true"] = train.loc[mask, endpoint]
        frame["conf_low"] = train.loc[mask, endpoint + "_conf_low"]
        frame["conf_high"] = train.loc[mask, endpoint + "_conf_high"]
        if not np.isfinite(frame[["y_true", "conf_low", "conf_high"]]).all(axis=None):
            raise ValueError(f"Non-finite labels or intervals for {endpoint}")
        base, blend, weights[endpoint] = nested_endpoint(
            frame,
            x[mask],
            names,
            seed=config.get("seed", 42),
            n_jobs=config.get("n_jobs", 8),
            inner_folds=config.get("inner_folds", 3),
            search_samples=config.get("search_samples", 512),
        )
        predictions = {**dict(zip(names, base.T, strict=True)), "nested_ensemble": blend}
        metrics[endpoint] = {}
        for name, prediction in predictions.items():
            frame[name] = prediction
            metrics[endpoint][name] = soft_threshold_rae(
                frame.y_true, prediction, frame.conf_low, frame.conf_high
            )
        rows.append(frame)
        pd.concat(rows).to_csv(output / "oof_predictions.csv", index=False)
        write_json(output / "metrics.json", metrics)
    result = {
        "endpoints": metrics,
        "macro": {
            name: float(np.mean([m[name] for m in metrics.values()]))
            for name in [*names, "nested_ensemble"]
        },
    }
    write_json(output / "metrics.json", result)
    write_json(output / "weights.json", weights)
    write_json(
        output / "run_manifest.json",
        {
            "config": config,
            "prepared_manifest": manifest,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "versions": {
                p: version(p)
                for p in ("numpy", "pandas", "rdkit", "scikit-learn", "lightgbm", "scipy")
            },
            "gpu_hours": 0,
            "source_sha256": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
            "embedding_manifest": embedding_manifest,
            "oof_sha256": sha256_file(output / "oof_predictions.csv"),
        },
    )
    return result


def export(run_dir: Path, data_dir: Path, output_dir: Path, checkpoint: Path | None = None) -> dict:
    """Refit a completed CPU nested run and export a candidate, without promotion."""
    run = json.loads((run_dir / "run_manifest.json").read_text())
    config = run["config"]
    prepared = Path(config["prepared_dir"])
    manifest = run["prepared_manifest"]
    if sha256_file(prepared / "training.csv") != manifest["training_sha256"]:
        raise ValueError("Prepared corpus changed since evaluation")
    if sha256_file(run_dir / "oof_predictions.csv") != run["oof_sha256"]:
        raise ValueError("OOF predictions changed since evaluation")
    for filename, digest in manifest["input_sha256"].items():
        if sha256_file(data_dir / filename) != digest:
            raise ValueError("Official data changed since evaluation")
    train, test = pd.read_csv(prepared / "training.csv"), load_test(data_dir)
    x, query = featurize_smiles(train.SMILES), featurize_smiles(test.SMILES)
    names = config.get("models", ["lightgbm", "extra_trees"])
    if any(n.endswith(("_r2", "_r3")) for n in names):
        x = np.concatenate([x, featurize_smiles(train.SMILES, radius=3)], axis=1)
        query = np.concatenate([query, featurize_smiles(test.SMILES, radius=3)], axis=1)
    if config.get("embeddings_dir"):
        from .neural import encode_smiles

        evidence = run["embedding_manifest"]
        if checkpoint is None or sha256_file(checkpoint) != evidence["checkpoint_sha256"]:
            raise ValueError("Supply the same pretrained checkpoint used for evaluation")
        embedding_path = Path(config["embeddings_dir"]) / "embeddings.npy"
        if sha256_file(embedding_path) != evidence["embeddings_sha256"]:
            raise ValueError("Embeddings changed since evaluation")
        x = np.concatenate([x, np.load(embedding_path, allow_pickle=False)], axis=1)
        query = np.concatenate([query, encode_smiles(test.SMILES, checkpoint)], axis=1)
    oof = pd.read_csv(run_dir / "oof_predictions.csv")
    names = config.get("models", ["lightgbm", "extra_trees"])
    output_dir.mkdir(parents=True, exist_ok=False)
    submission = test[["Molecule_Name", "SMILES"]].copy()
    weights = {}
    for endpoint in ACTIVITY_ENDPOINTS:
        frame = oof.loc[oof.endpoint == endpoint]
        fitted, _ = _metric_weights(
            frame[names].to_numpy(),
            frame.y_true,
            frame.conf_low,
            frame.conf_high,
            seed=config.get("seed", 42),
            search_samples=config.get("search_samples", 512),
        )
        weights[endpoint] = dict(zip(names, fitted.tolist(), strict=True))
        mask = train[endpoint].notna()
        predictions = []
        for name in names:
            model, prediction = fit_predict(
                name,
                x[mask],
                train.loc[mask, endpoint].to_numpy(),
                query,
                config.get("seed", 42),
                config.get("n_jobs", 8),
            )
            joblib.dump(model, output_dir / f"{endpoint}-{name}.joblib")
            predictions.append(prediction)
        submission[endpoint] = np.column_stack(predictions) @ fitted
    path = output_dir / "activity_submission.csv"
    submission.to_csv(path, index=False)
    report = validate_submission(path, test, "activity")
    if not report.valid:
        raise ValueError(f"Invalid final prediction file: {report.errors}")
    write_json(output_dir / "weights.json", weights)
    result = {
        "evaluation_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "submission_sha256": sha256_file(path),
        "validation": report.to_dict(),
        "status": "candidate_only_not_promoted_or_uploaded",
        "config": config,
    }
    write_json(output_dir / "run_manifest.json", result)
    return result
