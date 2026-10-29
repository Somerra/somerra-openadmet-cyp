"""Immutable conformer views for the private, frozen Atom-JEPA follow-up."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .atom_jepa import (
    GEOMETRY,
    MODEL_REVISION,
    SOURCE_REVISION,
    WEIGHTS_SHA256,
    align_rows,
    encode,
    generate_conformer,
    load_foundation,
    load_view,
)
from .download import sha256_file
from .frozen_benchmark import predict_head
from .representations import Run, atomic_json, identities, prepared_data

CONFORMER_SEEDS = tuple(range(42, 52))
COUNTS = (1, 2, 10)


def cache_stage(config_path):
    from .representation_budget import maybe_supervise

    supervised = maybe_supervise("conformer-cache", config_path)
    if supervised is not None:
        return supervised
    config = json.loads(Path(config_path).read_text())
    return extract_extra(config, config["base_cache"], config["conformer_cache"])


def verify_files(directory, manifest):
    for name, digest in manifest["files"].items():
        if sha256_file(Path(directory) / name) != digest:
            raise ValueError(f"Changed conformer artifact: {name}")


def generate_extra(directory, prepared_dir, workers=8):
    """Nine independent ETKDG seeds; slot zero is reused exactly from the pilot."""
    from joblib import Parallel, delayed

    frame, prepared = prepared_data(Path(prepared_dir))
    rows = identities(frame)
    config = {"output_dir": str(directory), "device": "cpu", "max_gpu_hours": 2, "threads": 1}
    with Run(config) as run:
        rows.to_csv(run.output / "rows.csv", index=False)
        diagnostics = {}
        for seed in CONFORMER_SEEDS[1:]:
            results = []
            for start in range(0, len(rows), 128):
                run.check()
                results.extend(
                    Parallel(n_jobs=workers)(
                        delayed(generate_conformer)(s, seed)
                        for s in rows.SMILES.iloc[start : start + 128]
                    )
                )
            offsets = np.r_[0, np.cumsum([len(r[0]) for r in results])]
            np.savez(
                run.output / f"seed-{seed}.npz",
                atomic_numbers=np.concatenate([r[0] for r in results]),
                coordinates=np.concatenate([r[1] for r in results]),
                offsets=offsets,
            )
            diagnostics[str(seed)] = [r[2] for r in results]
            run.record.update(completed_seeds=list(map(int, diagnostics)))
            run.save()
            print(f"Completed conformer seed {seed}: {len(rows)} molecules", flush=True)
        atomic_json(run.output / "relaxation.json", diagnostics)
        names = ["rows.csv", "relaxation.json"] + [f"seed-{s}.npz" for s in CONFORMER_SEEDS[1:]]
        manifest = {
            "schema_version": 4,
            "kind": "atom_jepa_extra_conformers",
            "complete": True,
            "seeds": list(CONFORMER_SEEDS[1:]),
            "geometry": {k: v for k, v in GEOMETRY.items() if k != "seed"},
            "training_sha256": prepared["training_sha256"],
            "rows": len(rows),
            "nonconverged": sum(
                r["mmff_status"] != 0 for part in diagnostics.values() for r in part
            ),
            "retried": sum(
                r["random_coordinate_retry"] for part in diagnostics.values() for r in part
            ),
            "files": {name: sha256_file(run.output / name) for name in names},
        }
        atomic_json(run.output / "manifest.json", manifest)
        run.record.update(complete=True)
    return manifest


def extract_extra(config, base_cache, geometry_dir):
    """Extract all views once; no labels or fitted preprocessing enter this cache."""
    frame, prepared = prepared_data(Path(config["prepared_dir"]))
    base, baseline = load_view(base_cache, frame, "invariant")
    geometry_dir = Path(geometry_dir)
    geometry = json.loads((geometry_dir / "manifest.json").read_text())
    if (
        not geometry.get("complete")
        or geometry.get("seeds") != list(CONFORMER_SEEDS[1:])
        or geometry.get("training_sha256") != prepared["training_sha256"]
        or geometry.get("schema_version") != 4
    ):
        raise ValueError("Incompatible extra conformer geometry")
    verify_files(geometry_dir, geometry)
    rows = pd.read_csv(geometry_dir / "rows.csv")
    indices = align_rows(rows, frame)
    with Run(config) as run:
        model, _ = load_foundation(config["checkpoint"], run.device)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        matrices = [base]
        parity = []
        with torch.inference_mode():
            for seed in CONFORMER_SEEDS[1:]:
                with np.load(geometry_dir / f"seed-{seed}.npz", allow_pickle=False) as data:
                    z, xyz, offsets = (
                        data[k] for k in ("atomic_numbers", "coordinates", "offsets")
                    )
                if (
                    offsets.shape != (len(rows) + 1,)
                    or offsets[0] != 0
                    or offsets[-1] != len(z)
                    or np.any(np.diff(offsets) < 2)
                    or xyz.shape != (len(z), 3)
                    or not np.isfinite(xyz).all()
                    or np.any((z < 1) | (z >= 128))
                ):
                    raise ValueError("Invalid conformer coordinates")
                samples = [
                    {
                        "atomic_numbers": torch.tensor(z[offsets[i] : offsets[i + 1]]),
                        "node_coordinates": torch.tensor(xyz[offsets[i] : offsets[i + 1]]),
                    }
                    for i in indices
                ]
                parts = []
                for start in range(0, len(samples), 32):
                    run.check()
                    parts.append(
                        encode(model, samples[start : start + 32])["invariant"].cpu().numpy()
                    )
                matrix = np.concatenate(parts).astype(np.float32)
                online = encode(model, samples[:3])["invariant"].cpu().numpy()
                if not np.allclose(matrix[:3], online, atol=1e-4, rtol=1e-4):
                    raise ValueError("Extra conformer cache/online parity failed")
                parity.append(float(np.abs(matrix[:3] - online).max()))
                matrices.append(matrix)
                run.record.update(extracted_seeds=list(CONFORMER_SEEDS[: len(matrices)]))
                run.save()
                print(f"Extracted conformer seed {seed}", flush=True)
        if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
            raise ValueError("Encoder state changed")
        matrix = np.stack(matrices, axis=1)
        np.save(run.output / "embeddings.npy", matrix, allow_pickle=False)
        identities(frame).to_csv(run.output / "rows.csv", index=False)
        manifest = {
            "schema_version": 4,
            "kind": "atom_jepa_conformer_features",
            "complete": True,
            "shape": list(matrix.shape),
            "conformer_seeds": list(CONFORMER_SEEDS),
            "checkpoint_sha256": WEIGHTS_SHA256,
            "source_revision": SOURCE_REVISION,
            "model_revision": MODEL_REVISION,
            "base_manifest_sha256": sha256_file(Path(base_cache) / "manifest.json"),
            "geometry_manifest_sha256": sha256_file(geometry_dir / "manifest.json"),
            "training_sha256": prepared["training_sha256"],
            "pooling": baseline["pooling"],
            "files": {
                name: sha256_file(run.output / name) for name in ("embeddings.npy", "rows.csv")
            },
        }
        atomic_json(run.output / "manifest.json", manifest)
        run.record.update(complete=True, frozen_state_unchanged=True, cached_online_max_abs=parity)
    return manifest


def load_conformer_features(directory, frame):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 4
        or manifest.get("kind") != "atom_jepa_conformer_features"
        or not manifest.get("complete")
        or manifest.get("conformer_seeds") != list(CONFORMER_SEEDS)
        or manifest.get("checkpoint_sha256") != WEIGHTS_SHA256
        or manifest.get("source_revision") != SOURCE_REVISION
        or manifest.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("Incompatible conformer feature cache")
    verify_files(directory, manifest)
    rows = pd.read_csv(directory / "rows.csv")
    indices = align_rows(rows, frame)
    matrix = np.load(directory / "embeddings.npy", allow_pickle=False, mmap_mode="r")
    if (
        matrix.shape != (len(rows), 10, 768)
        or matrix.dtype != np.float32
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Invalid conformer features")
    return np.array(matrix[indices]), manifest


def predict_conformers(model, features):
    """Return predictions for each conformer, never average nonlinear-head inputs."""
    if features.ndim != 3 or features.shape[1] < 1:
        raise ValueError("Expected [molecule, conformer, feature]")
    n, k, d = features.shape
    return predict_head(model, features.reshape(n * k, d)).reshape(n, k, -1)


def fit_stage(config_path):
    from .constants import ACTIVITY_ENDPOINTS
    from .frozen_benchmark import PROTOCOL, RECIPE, fit_head, restore_head, scores
    from .representation_budget import maybe_supervise
    from .representation_training import oof_rows

    config = json.loads(Path(config_path).read_text())
    if (
        config.get("protocol") != PROTOCOL
        or config.get("adaptation") != "frozen"
        or config.get("precision") != "float32"
        or config.get("selected_folds") != [0]
        or config.get("scaling") != "standard"
        or config.get("scaler_conformer") != 0
    ):
        raise ValueError("Conformer sensitivity requires the fixed frozen float32 recipe")
    view, count, seed = config["representation"], config["train_conformers"], config["seed"]
    if view not in ("scalar", "invariant") or count not in (2, 10) or seed not in (42, 43, 44):
        raise ValueError("Unsupported conformer sensitivity arm")
    supervised = maybe_supervise("conformer-fit", config_path)
    if supervised is not None:
        return supervised
    output = Path(config["output_dir"])
    frame, prepared = prepared_data(Path(config["prepared_dir"]))
    features, cache = load_conformer_features(config["cache_dir"], frame)
    fit = np.flatnonzero(frame.fold.to_numpy() != 0)
    held = np.flatnonzero(frame.fold.to_numpy() == 0)
    x = features[:, :count, :256] if view == "scalar" else features[:, :count]
    targets = frame[list(ACTIVITY_ENDPOINTS)].to_numpy(dtype=np.float32)
    with Run(config) as run:
        run.record.update(
            protocol=PROTOCOL,
            recipe=RECIPE,
            prepared_manifest=prepared,
            cache_manifest=cache,
            encoder_trainable=False,
            selected_folds=[0],
            requested_folds_complete=False,
            completed_folds=[],
            promoted=False,
            purpose="single_fold_frozen_conformer_sensitivity",
        )
        run.save()
        model, training = fit_head(x, targets, fit, seed, "standard", run)
        prediction = predict_conformers(model, x[held])[:, 0]
        saved = {
            "schema_version": 3,
            "protocol": PROTOCOL,
            "tasks": list(ACTIVITY_ENDPOINTS),
            "dimension": x.shape[-1],
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": config,
            "recipe": RECIPE,
            "fold": 0,
            "seed": seed,
            "training_sha256": prepared["training_sha256"],
            "cache_manifest_sha256": sha256_file(Path(config["cache_dir"]) / "manifest.json"),
            "fit_ids": frame.iloc[fit].Molecule_Name.tolist(),
            "held_ids": frame.iloc[held].Molecule_Name.tolist(),
        }
        torch.save(saved, output / "fold-0.pt")
        restored, _ = restore_head(output / "fold-0.pt")
        replay = predict_conformers(restored, x[held])[:, 0]
        delta = float(np.abs(prediction - replay).max())
        if delta > 1e-4:
            raise ValueError(f"CPU checkpoint replay failed: {delta}")
        rows = oof_rows(frame, held, prediction)
        rows.to_csv(output / "oof_predictions.csv", index=False)
        run.record.update(
            training=training,
            metrics=scores(rows),
            input_dimension=x.shape[-1],
            completed_folds=[0],
            requested_folds_complete=True,
            complete=False,
            pilot_complete=True,
            cpu_replay_max_abs=delta,
            checkpoint_sha256=sha256_file(output / "fold-0.pt"),
            oof_sha256=sha256_file(output / "oof_predictions.csv"),
        )

    return run.record
