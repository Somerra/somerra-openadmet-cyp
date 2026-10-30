"""Versioned, inner-selected probes of frozen caches; no encoder is loaded."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .constants import ACTIVITY_ENDPOINTS
from .download import sha256_file
from .frozen_benchmark import load_features, predict_head, scores
from .metrics import soft_threshold_rae
from .representation_training import Inputs, inner_split, make_model, oof_rows, step
from .representations import Run, atomic_json, batch_indices, prepared_data, read_config

PROTOCOL = "matched_frozen_v2"
RECIPE = {
    "max_epochs": 300,
    "patience": 30,
    "min_delta": 1e-5,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "head_dropout": 0.2,
    "head": "D->512->BatchNorm->ReLU->Dropout(0.2)->4",
    "standardize_features": True,
    "standardize_targets": True,
    "loss": "endpoint-balanced masked MAE in standardized targets",
    "selection": "inner MA-ST-RAE; no outer-training refit",
    "inner_split_seed": 42,
    "ridge_alphas": [1.0, 10.0, 100.0],
    "tree_leaves": [15, 31],
    "tree_max_estimators": 500,
    "tree_learning_rate": 0.05,
    "tree_min_child_samples": 20,
    "tree_reg_lambda": 1.0,
}


def implementation_hashes():
    return {
        name: sha256_file(Path(__file__).with_name(name))
        for name in (
            "frozen_probe.py",
            "frozen_benchmark.py",
            "representation_training.py",
            "representations.py",
            "representation_budget.py",
            "metrics.py",
        )
    }


def split_indices(frame):
    """One persisted inner split for every feature set, learner and head seed."""
    outer_train = np.flatnonzero(frame.fold.to_numpy() != 0)
    test = np.flatnonzero(frame.fold.to_numpy() == 0)
    train, validation = inner_split(frame, outer_train, RECIPE["inner_split_seed"])
    groups = [set(frame.iloc[i].cluster) for i in (train, validation, test)]
    if not len(test) or any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Probe partitions must be nonempty and cluster-disjoint")
    return train, validation, test


def split_record(frame, partitions):
    return {
        name: frame.iloc[indices].Molecule_Name.tolist()
        for name, indices in zip(("train", "validation", "test"), partitions, strict=True)
    }


def metric(targets, prediction, bounds):
    return float(
        np.mean(
            [
                soft_threshold_rae(
                    targets[:, i], prediction[:, i], bounds[:, i, 0], bounds[:, i, 1]
                )
                for i in range(len(ACTIVITY_ENDPOINTS))
            ]
        )
    )


def target_statistics(targets):
    means = np.nanmean(targets, axis=0).astype(np.float32)
    scales = np.maximum(np.nanstd(targets, axis=0), 1e-6).astype(np.float32)
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise ValueError("Every endpoint needs finite inner-training labels")
    return means, scales


def probe_model(dimension, means, scales, dropout):
    # Reuse the shared frozen MLP and target-normalization buffers. The factory's
    # default encoder name selects its generic dense head; frozen mode loads no encoder.
    return make_model(
        "frozen",
        input_dim=dimension,
        head_dropout=dropout,
        target_mean=means,
        target_scale=scales,
    )


def fit_mlp(train_x, train_y, val_x, val_y, val_bounds, seed, run, recipe=None):
    """The fitting API has no outer-test features or labels; override is test-only."""
    recipe = RECIPE if recipe is None else recipe
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x).astype(np.float32)
    val_x = scaler.transform(val_x).astype(np.float32)
    means, scales = target_statistics(train_y)
    torch.manual_seed(seed)
    model = probe_model(train_x.shape[1], means, scales, recipe["head_dropout"]).to(run.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]
    )
    inputs = Inputs(None, "frozen", run.device, train_x)
    rng = np.random.default_rng(seed)
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    updates = 0
    for epoch in range(1, recipe["max_epochs"] + 1):
        model.train()
        losses = []
        for batch in batch_indices(
            rng.permutation(len(train_x)), recipe["batch_size"], merge_singleton=True
        ):
            run.check()
            loss, _ = step(model, optimizer, inputs, batch, train_y, "float32")
            if loss is not None:
                losses.append(loss)
                updates += 1
        if not losses:
            raise ValueError("No observed training labels")
        run.check()
        inner_score = metric(val_y, predict_head(model, val_x), val_bounds)
        history.append(
            {
                "epoch": epoch,
                "training_standardized_mae": float(np.mean(losses)),
                "inner_MA-ST-RAE": inner_score,
                "updates": updates,
            }
        )
        if inner_score < best - recipe["min_delta"]:
            best, best_epoch = inner_score, epoch
            best_state = copy.deepcopy(model.state_dict())
        atomic_json(run.output / "training.json", history)
        if epoch - best_epoch >= recipe["patience"]:
            break
    model.load_state_dict(best_state)
    return (
        model.eval(),
        scaler,
        {
            "selected_epoch": best_epoch,
            "epochs_run": len(history),
            "inner_MA-ST-RAE": best,
            "successful_updates": updates,
            "head_parameters": sum(p.numel() for p in model.parameters()),
            "target_mean": means.tolist(),
            "target_scale": scales.tolist(),
        },
    )


def tree_validation_metric(mean, scale, truth, bounds):
    # LightGBM interprets a callable's arity as labels/predictions[/weights/group],
    # so bind endpoint data in this factory rather than default arguments.
    def evaluate(y_true, prediction):
        score = soft_threshold_rae(truth, prediction * scale + mean, bounds[:, 0], bounds[:, 1])
        return "ST-RAE", score, False

    return evaluate


def fit_controls(train_x, train_y, val_x, val_y, val_bounds, run, recipe=None):
    """Select CPU controls exclusively on the same inner partitions as the MLP."""
    import lightgbm as lgb

    recipe = RECIPE if recipe is None else recipe
    scaler = StandardScaler().fit(train_x)
    train_x, val_x = scaler.transform(train_x), scaler.transform(val_x)
    means, scales = target_statistics(train_y)
    normalized = (train_y - means) / scales
    result = {"mean": {"models": None, "selection": {"target_mean": means.tolist()}}}
    for learner in ("ridge", "lightgbm"):
        candidates = recipe["ridge_alphas"] if learner == "ridge" else recipe["tree_leaves"]
        best = None
        trials = []
        for value in candidates:
            models, val_prediction = [], np.empty_like(val_y)
            for i in range(len(ACTIVITY_ENDPOINTS)):
                run.check()
                observed = np.isfinite(train_y[:, i])
                valid = np.isfinite(val_y[:, i])
                if learner == "ridge":
                    model = Ridge(alpha=value).fit(train_x[observed], normalized[observed, i])
                else:
                    model = lgb.LGBMRegressor(
                        n_estimators=recipe["tree_max_estimators"],
                        num_leaves=value,
                        learning_rate=recipe["tree_learning_rate"],
                        min_child_samples=recipe["tree_min_child_samples"],
                        reg_lambda=recipe["tree_reg_lambda"],
                        objective="regression_l1",
                        metric="None",
                        random_state=42,
                        n_jobs=8,
                        verbosity=-1,
                        deterministic=True,
                        force_col_wise=True,
                    )

                    model.fit(
                        train_x[observed],
                        normalized[observed, i],
                        eval_set=[(val_x[valid], (val_y[valid, i] - means[i]) / scales[i])],
                        eval_metric=tree_validation_metric(
                            means[i], scales[i], val_y[valid, i], val_bounds[valid, i]
                        ),
                        callbacks=[lgb.early_stopping(recipe["patience"], verbose=False)],
                    )
                models.append(model)
                val_prediction[:, i] = model.predict(val_x) * scales[i] + means[i]
            score = metric(val_y, val_prediction, val_bounds)
            trial = {"value": value, "inner_MA-ST-RAE": score}
            if learner == "lightgbm":
                trial["selected_iterations"] = [int(m.best_iteration_) for m in models]
            trials.append(trial)
            if best is None or score < best["selection"]["inner_MA-ST-RAE"]:
                best = {"models": models, "selection": trial}
        best["trials"] = trials
        result[learner] = best
    return result, scaler, means, scales


def control_prediction(control, features, scaler, means, scales):
    if control["models"] is None:
        return np.tile(means, (len(features), 1))
    values = scaler.transform(features)
    return np.column_stack([m.predict(values) for m in control["models"]]) * scales + means


def restore_probe(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved.get("protocol") != PROTOCOL
        or saved.get("recipe") != RECIPE
        or saved.get("tasks") != list(ACTIVITY_ENDPOINTS)
    ):
        raise ValueError("Incompatible v2 frozen probe")
    model = probe_model(
        saved["dimension"], saved["target_mean"], saved["target_scale"], RECIPE["head_dropout"]
    )
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved


def completed(config):
    """Resume only when current inputs, split, recipe and every artifact still match."""
    directory = Path(config["output_dir"])
    if not directory.exists():
        return False
    record = json.loads((directory / "run_manifest.json").read_text())
    if (
        not record.get("pilot_complete")
        or record.get("error")
        or record.get("config") != config
        or record.get("recipe") != RECIPE
        or record.get("implementation_sha256") != implementation_hashes()
    ):
        raise ValueError("Incomplete or changed v2 probe; refusing resume")
    frame, prepared = prepared_data(Path(config["prepared_dir"]))
    load_features(config, frame)  # Includes current cache bytes, row identities and source pins.
    if (
        record["training_sha256"] != prepared["training_sha256"]
        or record["cache_sha256"] != sha256_file(Path(config["cache_dir"]) / "manifest.json")
        or record["split"] != split_record(frame, split_indices(frame))
    ):
        raise ValueError("Changed v2 probe inputs")
    for name, digest in record["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Changed v2 probe artifact: {name}")
    return True


def screen(config_path):
    config = read_config(
        Path(config_path),
        {
            "protocol",
            "encoder",
            "representation",
            "prepared_dir",
            "cache_dir",
            "checkpoint",
            "learner",
            "ledger",
        },
        {
            "protocol",
            "encoder",
            "representation",
            "prepared_dir",
            "cache_dir",
            "learner",
            "output_dir",
            "max_gpu_hours",
            "seed",
        },
    )
    if config["protocol"] != PROTOCOL or config["learner"] not in {"mlp", "controls"}:
        raise ValueError("Expected the versioned frozen v2 probe")
    if config.get("precision", "float32") != "float32":
        raise ValueError("Frozen v2 probes require float32")
    if config["learner"] == "controls" and config.get("device") != "cpu":
        raise ValueError("Controls must run on CPU")
    from .representation_budget import maybe_supervise

    supervised = maybe_supervise("frozen-probe", config_path)
    if supervised is not None:
        return supervised
    frame, prepared = prepared_data(Path(config["prepared_dir"]))
    if prepared["n_folds"] != 5:
        raise ValueError("Expected the shared five-fold manifest")
    features, _ = load_features(config, frame)
    train, val, test = split_indices(frame)
    targets = frame[list(ACTIVITY_ENDPOINTS)].to_numpy(dtype=np.float32)
    bounds = np.stack(
        [frame[[e + "_conf_low", e + "_conf_high"]].to_numpy() for e in ACTIVITY_ENDPOINTS], axis=1
    )
    with Run(config) as run:
        run.record.update(
            protocol=PROTOCOL,
            recipe=RECIPE,
            implementation_sha256=implementation_hashes(),
            encoder_trainable=False,
            promoted=False,
            training_sha256=prepared["training_sha256"],
            cache_sha256=sha256_file(Path(config["cache_dir"]) / "manifest.json"),
            split=split_record(frame, (train, val, test)),
        )
        run.save()
        outputs = {}
        if config["learner"] == "mlp":
            model, scaler, selection = fit_mlp(
                features[train],
                targets[train],
                features[val],
                targets[val],
                bounds[val],
                config["seed"],
                run,
            )
            saved = {
                "protocol": PROTOCOL,
                "recipe": RECIPE,
                "tasks": list(ACTIVITY_ENDPOINTS),
                "dimension": features.shape[1],
                **selection,
                "feature_mean": scaler.mean_.tolist(),
                "feature_scale": scaler.scale_.tolist(),
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            }
            checkpoint = run.output / "head.pt"
            torch.save(saved, checkpoint)
            test_x = scaler.transform(features[test]).astype(np.float32)
            prediction = predict_head(model, test_x)
            restored, _ = restore_probe(checkpoint)
            delta = float(np.max(np.abs(predict_head(restored, test_x) - prediction)))
            if delta > 1e-4:
                raise ValueError(f"CPU probe replay exceeds tolerance: {delta}")
            run.record.update(cpu_replay_max_abs=delta, selection=selection)
            outputs["mlp"] = prediction
        else:
            import joblib

            controls, scaler, means, scales = fit_controls(
                features[train], targets[train], features[val], targets[val], bounds[val], run
            )
            artifact = run.output / "controls.joblib"
            joblib.dump((controls, scaler, means, scales), artifact)
            replay_controls, replay_scaler, replay_means, replay_scales = joblib.load(artifact)
            for name, control in controls.items():
                outputs[name] = control_prediction(control, features[test], scaler, means, scales)
                replay = control_prediction(
                    replay_controls[name],
                    features[test],
                    replay_scaler,
                    replay_means,
                    replay_scales,
                )
                if not np.allclose(outputs[name], replay, rtol=0, atol=1e-8):
                    raise ValueError("Control checkpoint replay failed")
            run.record["selection"] = {
                name: {k: v for k, v in control.items() if k != "models"}
                for name, control in controls.items()
            }
        # Outer labels enter only after checkpoint/hyperparameter selection is finished.
        metrics = {}
        for name, prediction in outputs.items():
            rows = oof_rows(frame, test, prediction)
            rows.to_csv(run.output / f"{name}-predictions.csv", index=False)
            metrics[name] = scores(rows)
        atomic_json(run.output / "metrics.json", metrics)
        atomic_json(run.output / "split.json", run.record["split"])
        files = {
            p.name: sha256_file(p)
            for p in run.output.iterdir()
            if p.name != "run_manifest.json" and p.is_file()
        }
        run.record.update(
            metrics=metrics,
            files=files,
            completed_folds=[0],
            complete=False,
            requested_folds_complete=True,
            pilot_complete=True,
        )
        return run.record
