"""Single Scigantic armB member, shared challenge scoring and single-model inference."""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .budget import GPUBudget, budget_ledger_path
from .constants import ACTIVITY_ENDPOINTS
from .data_sources.common import write_json
from .download import sha256_file
from .metrics import soft_threshold_rae
from .reference_member_data import load_frames


class Plateau:
    def __init__(self, patience=30, relative=0.005):
        self.patience, self.relative = patience, relative
        self.anchor, self.last = float("inf"), 0

    def update(self, epoch, score):
        if not np.isfinite(score):
            raise ValueError("Nonfinite validation score")
        if score < self.anchor * (1 - self.relative):
            self.anchor, self.last = score, epoch
        return epoch >= 30 and epoch - self.last >= self.patience


def score_predictions(truth, predictions):
    if list(predictions.columns) != list(ACTIVITY_ENDPOINTS) or len(predictions) != len(truth):
        raise ValueError("Prediction shape/order mismatch")
    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("Nonfinite predictions")
    endpoints, components = {}, {}
    for ep in ACTIVITY_ENDPOINTS:
        observed = truth[ep].notna()
        y = truth.loc[observed, ep].to_numpy()
        lo = truth.loc[observed, ep + "_conf_low"].to_numpy()
        hi = truth.loc[observed, ep + "_conf_high"].to_numpy()
        pred = predictions.loc[observed, ep].to_numpy()
        if not np.isfinite(np.column_stack([y, lo, hi])).all():
            raise ValueError("Observed label has invalid bounds")
        endpoints[ep] = soft_threshold_rae(y, pred, lo, hi)
        components[ep] = {
            "labels": len(y),
            "numerator": float((np.maximum(pred - hi, 0) + np.maximum(lo - pred, 0)).sum()),
            "denominator": float(
                (np.maximum(y.mean() - hi, 0) + np.maximum(lo - y.mean(), 0)).sum()
            ),
        }
    return {
        "MA-ST-RAE": float(np.mean(list(endpoints.values()))),
        "endpoints": endpoints,
        "components": components,
    }


def make_model(scaler, tasks, loss_kind="mae", auxiliary_weight=1.0):
    from chemprop import nn

    from .hillclimb_model import StudyMPNN
    from .neural import validate_task_order

    validate_task_order(tasks)
    mp = nn.BondMessagePassing(d_h=2048, depth=4, dropout=0.2)
    ffn = nn.RegressionFFN(
        n_tasks=len(tasks),
        input_dim=2048,
        hidden_dim=1024,
        n_layers=2,
        dropout=0.2,
        criterion=nn.metrics.MAE(),
        output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
    )
    return StudyMPNN(
        mp,
        nn.NormAggregation(),
        ffn,
        batch_norm=True,
        metrics=[nn.metrics.MAE()],
        warmup_epochs=2,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-4,
        loss_kind=loss_kind,
        auxiliary_weight=auxiliary_weight,
    )


def prediction_loader(smiles):
    from chemprop import data

    ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles])
    ds.cache = True
    return data.build_dataloader(ds, batch_size=128, shuffle=False, num_workers=0, drop_last=False)


def predict(model, batches, tasks):
    import torch

    was_training = model.training
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in batches:
            batch.bmg.to(model.device)
            values.append(model(batch.bmg).detach().cpu().numpy())
    model.train(was_training)
    result = np.concatenate(values)
    if result.shape[1] != len(tasks) or not np.isfinite(result).all():
        raise ValueError("Invalid model output")
    return pd.DataFrame(result, columns=tasks)[list(ACTIVITY_ENDPOINTS)]


def train(
    bundle,
    output,
    recipe="full",
    epochs=200,
    hours=2,
    pilot=False,
    final=False,
    smoke=False,
    resume_from=None,
    loss_kind="mae",
    auxiliary_weight=1.0,
):
    import lightning as L
    import torch
    from chemprop import data
    from lightning.pytorch.callbacks import Callback

    from .configuration import source_hashes

    output, bundle = Path(output), Path(bundle)
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not 1 <= epochs <= 200 or not 0 < hours <= 2:
        raise ValueError("Invalid finite run allocation")
    if loss_kind not in {"mae", "interval"} or auxiliary_weight not in {1.0, 0.5}:
        raise ValueError("Unregistered experiment settings")
    pool, val, tasks, manifest = load_frames(bundle, recipe, final)
    output.mkdir(parents=True)
    budget = GPUBudget(budget_ledger_path(), output, hours)
    record = {
        "status": "running",
        "recipe": recipe,
        "loss_kind": loss_kind,
        "auxiliary_weight": auxiliary_weight,
        "matmul_precision": "high",
        "batch_size": 128,
        "graph_cache": True,
        "final": final,
        "seed": 42,
        "tasks": tasks,
        "scheduler_horizon": 200,
        "epoch_limit": epochs,
        "training_records": len(pool),
        "training_parents": pool.parent.nunique(),
        "validation_rows": 0 if final else len(val),
        "data_manifest": manifest,
        "data_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "source_sha256": source_hashes(),
        "best_epoch": None,
        "best_score": None,
        "completed_epochs": 0,
        "smoke": smoke,
    }
    resume_checkpoint = None
    if resume_from is not None:
        previous = Path(resume_from)
        saved = json.loads((previous / "run_manifest.json").read_text())
        for key in [
            "recipe",
            "tasks",
            "data_manifest_sha256",
            "final",
            "scheduler_horizon",
            "loss_kind",
            "auxiliary_weight",
        ]:
            if saved[key] != record[key]:
                raise ValueError(f"Resume contract mismatch: {key}")
        recovery = previous / "recovery.json"
        if recovery.exists():
            snapshot = json.loads(recovery.read_text())
            checkpoint = previous / snapshot["checkpoint"]
            digest = snapshot["sha256"]
            saved = snapshot["record"]
        else:
            checkpoint = previous / "best.ckpt"
            digest = saved["best.ckpt_sha256"]
        if sha256_file(checkpoint) != digest:
            raise ValueError("Resume checkpoint changed")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        progress = state["loops"]["fit_loop"]
        if not progress["epoch_loop.batch_progress"]["is_last_batch"]:
            raise ValueError("Resume requires a completed epoch")
        # Our epoch-end callback checkpoint precedes Lightning's completed counter update.
        for scope in ["total", "current"]:
            ep = progress["epoch_progress"][scope]
            ep["completed"] = ep["processed"]
        resume_checkpoint = output / "resume.ckpt"
        torch.save(state, resume_checkpoint)
        record.update(
            resumed_from=str(previous),
            resumed_epoch=int(state["epoch"]) + 1,
            resumed_optimizer_steps=int(state["global_step"]),
            resume_note="Optimizer/scheduler restored; old checkpoints lack RNG state",
        )
        for key in ["best_epoch", "best_score", "best_metrics"]:
            if key in saved:
                record[key] = saved[key]
        for name in ["best.ckpt", "best-validation.csv"]:
            if (previous / name).exists():
                shutil.copy2(previous / name, output / name)
    budget.reserve()
    started = time.monotonic()
    write_json(output / "run_manifest.json", record)
    try:
        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()
        L.seed_everything(42, workers=True)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(pool[tasks].to_numpy(dtype=np.float32))
        if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
            raise ValueError("Selected task has no finite training scale")
        targets = scaler.transform(pool[tasks].to_numpy(dtype=np.float32))
        if loss_kind == "interval":
            lo = pool[[ep + "_conf_low" for ep in ACTIVITY_ENDPOINTS]].to_numpy(dtype=np.float32)
            hi = pool[[ep + "_conf_high" for ep in ACTIVITY_ENDPOINTS]].to_numpy(dtype=np.float32)
            observed = np.isfinite(targets[:, :4])
            if not ((np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)) | ~observed).all():
                raise ValueError("Missing or reversed direct training bounds")
            targets = np.c_[
                targets,
                (lo - scaler.mean_[:4]) / scaler.scale_[:4],
                (hi - scaler.mean_[:4]) / scaler.scale_[:4],
            ].astype(np.float32)
        points = [
            data.MoleculeDatapoint.from_smi(s, y, weight=float(w))
            for s, y, w in zip(pool.parent, targets, pool.sample_weight, strict=True)
        ]
        dataset = data.MoleculeDataset(points)
        dataset.cache = True
        train_loader = data.build_dataloader(
            dataset, batch_size=128, num_workers=4, persistent_workers=True
        )
        val_loader = None if final else prediction_loader(val.parent)
        model = make_model(scaler, tasks, loss_kind, auxiliary_weight)
        plateau = Plateau()
        curves = []

        class Monitor(Callback):
            def on_train_epoch_start(self, trainer, pl_module):
                self.epoch_start = time.monotonic()
                self.losses = []

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                value = outputs["loss"] if isinstance(outputs, dict) else outputs
                if value is not None:
                    value = float(value.detach().cpu())
                    if not np.isfinite(value):
                        raise ValueError("Nonfinite training loss")
                    self.losses.append(value)
                if time.monotonic() - started > hours * 3600 - 30:
                    record["status"] = "budget_censored"
                    trainer.should_stop = True

            def on_train_epoch_end(self, trainer, pl_module):
                if record["status"] == "budget_censored":
                    return  # never score a partially completed epoch
                epoch = trainer.current_epoch + 1
                row = {
                    "epoch": epoch,
                    "training_loss": float(np.mean(self.losses)),
                    "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
                    "optimizer_steps": trainer.global_step,
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                }
                if not final:
                    predictions = predict(pl_module, val_loader, tasks)
                    scores = score_predictions(val, predictions)
                    row.update(scores)
                    if record["best_score"] is None or scores["MA-ST-RAE"] < record["best_score"]:
                        record.update(
                            best_score=scores["MA-ST-RAE"], best_epoch=epoch, best_metrics=scores
                        )
                        trainer.save_checkpoint(output / "best.ckpt")
                        pd.concat(
                            [val[["Molecule_Name", "parent"]].reset_index(drop=True), predictions],
                            axis=1,
                        ).to_csv(output / "best-validation.csv", index=False)
                    if pilot and plateau.update(epoch, scores["MA-ST-RAE"]):
                        record["status"] = "plateau"
                        trainer.should_stop = True
                row["epoch_seconds"] = time.monotonic() - self.epoch_start
                curves.append(row)
                record.update(completed_epochs=epoch, optimizer_steps=trainer.global_step)
                record["last_metrics"] = row
                with (output / "curve.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                write_json(output / "run_manifest.json", record)
                # Publish a recovery pointer only after its checkpoint is fully written.
                checkpoint_name = f"recovery-{epoch % 2}.ckpt"
                temporary = output / (checkpoint_name + ".tmp")
                trainer.save_checkpoint(temporary)
                temporary.replace(output / checkpoint_name)
                write_json(
                    output / "recovery.json",
                    {
                        "checkpoint": checkpoint_name,
                        "sha256": sha256_file(output / checkpoint_name),
                        "record": record,
                    },
                )
                print(json.dumps(row), flush=True)
                if epoch >= epochs:
                    record["status"] = "epoch_limit"
                    trainer.should_stop = True
                if final and epoch >= epochs:
                    trainer.save_checkpoint(output / "final.ckpt")

        trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=200,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_train_batches=2 if smoke else 1.0,
            callbacks=[Monitor()],
        )
        trainer.fit(model, train_loader, ckpt_path=resume_checkpoint, weights_only=False)
        if smoke:
            record["status"] = "smoke_complete"
        elif record["status"] == "running":
            record["status"] = "epoch_limit"
        for filename in ["best.ckpt", "final.ckpt"]:
            path = output / filename
            if path.exists():
                record[filename + "_sha256"] = sha256_file(path)
        record["complete"] = record["status"] in ["plateau", "epoch_limit"]
        write_json(output / "run_manifest.json", record)
        if not final and record["best_epoch"] is not None:
            # Reload through the same export path; verify against saved validation predictions.
            loaded = (
                load_model(output).to("cuda") if (output / "run_manifest.json").exists() else None
            )
            if loaded is not None:
                actual = predict(loaded, val_loader, tasks).to_numpy()
                expected = pd.read_csv(output / "best-validation.csv")[
                    list(ACTIVITY_ENDPOINTS)
                ].to_numpy()
                error = float(np.max(np.abs(actual - expected)))
                if error > 1e-4:
                    raise ValueError("Saved checkpoint prediction parity failed")
                record["reload_max_abs_error"] = error
                if manifest.get("protocol") == "molecular-60-20-20-v1" and record["complete"]:
                    from .bootstrap_activity import bootstrap_activity

                    official = pd.read_csv(bundle / "official.csv")
                    held = official.loc[official.partition == "internal_test"].reset_index(
                        drop=True
                    )
                    test_predictions = predict(loaded, prediction_loader(held.parent), tasks)
                    pd.concat([held[["Molecule_Name", "parent"]], test_predictions], axis=1).to_csv(
                        output / "internal-test.csv", index=False
                    )
                    record["internal_test_metrics"] = score_predictions(held, test_predictions)
                    record["internal_test_bootstrap"] = bootstrap_activity(held, test_predictions)
                    record["validation_bootstrap"] = bootstrap_activity(
                        val, pd.DataFrame(actual, columns=ACTIVITY_ENDPOINTS)
                    )
    except BaseException as exc:
        record.update(status="failed", complete=False, error=repr(exc))
        raise
    finally:
        record["gpu_hours"] = budget.finish()
        write_json(output / "run_manifest.json", record)
    return record


def load_model(run):
    from .hillclimb_model import StudyMPNN

    run = Path(run)
    record = json.loads((run / "run_manifest.json").read_text())
    filename = "final.ckpt" if record["final"] else "best.ckpt"
    digest = record.get(filename + "_sha256")
    if digest is None or sha256_file(run / filename) != digest:
        raise ValueError("Checkpoint hash missing or changed")
    # Own hash-verified Lightning checkpoint, never an arbitrary downloaded pickle.
    return StudyMPNN.load_from_checkpoint(run / filename, map_location="cpu")


def export(run, bundle, raw, output):
    from .constants import TEST_FILE
    from .data import load_test
    from .validation import validate_submission

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    record = json.loads((Path(run) / "run_manifest.json").read_text())
    if not record.get("complete") or not record["final"]:
        raise ValueError("Final training incomplete")
    if sha256_file(Path(raw) / TEST_FILE) != record["data_manifest"]["test_sha256"]:
        raise ValueError("Test snapshot changed")
    load_frames(bundle, record["recipe"], final=True)
    if sha256_file(Path(bundle) / "manifest.json") != record["data_manifest_sha256"]:
        raise ValueError("Data manifest changed")
    import torch

    torch.set_num_threads(8)
    test = load_test(raw)
    prepared_test = pd.read_csv(Path(bundle) / "test.csv")
    if list(prepared_test.Molecule_Name) != list(test.Molecule_Name):
        raise ValueError("Test identity/order mismatch")
    model = load_model(run).eval()  # CPU export requires no further GPU allocation
    predictions = predict(model, prediction_loader(prepared_test.parent), record["tasks"])
    result = pd.concat(
        [test[["Molecule_Name", "SMILES"]].reset_index(drop=True), predictions], axis=1
    )
    output.mkdir(parents=True)
    result.to_csv(output / "activity_submission.csv", index=False)
    report = validate_submission(output / "activity_submission.csv", test, "activity")
    if not report.valid or report.warnings:
        raise ValueError(report.to_dict())
    write_json(
        output / "manifest.json",
        {
            "status": "prepared_not_submitted",
            "validation": report.to_dict(),
            "source_run": str(run),
            "submission_sha256": sha256_file(output / "activity_submission.csv"),
            "open_source_code": False,
            "model_report_link": "",
        },
    )
