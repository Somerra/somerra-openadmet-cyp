"""Export predictions from a complete current-format neural fold ensemble."""

import json
import time
from pathlib import Path

import numpy as np

from .budget import GPUBudget, budget_ledger_path
from .constants import ACTIVITY_ENDPOINTS, TEST_FILE
from .data import load_test
from .data_sources.common import parent_smiles, write_json
from .download import sha256_file
from .validation import validate_submission


def checkpoint_paths(run_dir, manifest):
    folds = manifest.get("selected_folds", [])
    if (
        manifest.get("complete") is not True
        or len(folds) != 5
        or len(set(folds)) != 5
        or manifest.get("completed_folds") != folds
    ):
        raise ValueError("Export requires all five completed folds")
    hashes = manifest.get("checkpoint_sha256_by_fold", {})
    paths = []
    for fold in folds:
        path = run_dir / f"fold-{fold}.pt"
        if str(fold) not in hashes or sha256_file(path) != hashes[str(fold)]:
            raise ValueError("Missing or changed fold checkpoint")
        paths.append(path)
    return paths


def export(run_dir: Path, data_dir: Path, output_dir: Path, max_gpu_hours=0.5):
    import torch

    from .neural import loader
    from .neural_models import build_model

    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    paths = checkpoint_paths(run_dir, manifest)
    if (
        sha256_file(data_dir / TEST_FILE)
        != manifest["prepared_manifest"]["input_sha256"][TEST_FILE]
    ):
        raise ValueError("Blinded test snapshot changed")
    if manifest["config"].get("model_family") != "scigantic_mt_aux8":
        raise ValueError("Export supports the current eight-head model family")
    if not 0 < max_gpu_hours <= 0.5:
        raise ValueError("Export allocation must be positive and at most 0.5 GPU-hours")
    if not torch.cuda.is_available():
        raise RuntimeError("Neural export requires CUDA")
    test = load_test(data_dir)
    smiles = test.SMILES.map(parent_smiles).tolist()
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.25)
    output_dir.mkdir(parents=True, exist_ok=False)
    allocation = GPUBudget(budget_ledger_path(), output_dir, max_gpu_hours)
    record = {
        "status": "running",
        "evaluation_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "test_sha256": sha256_file(data_dir / TEST_FILE),
        "run_dir": str(run_dir),
        "method": "equal mean of five completed outer-fold models",
        "promoted": False,
        "open_source_code": False,
        "model_report_link": "",
    }
    allocation.reserve()
    start = time.monotonic()
    write_json(output_dir / "run_manifest.json", record)
    try:
        predictions = []
        for path in paths:
            saved = torch.load(path, weights_only=True, map_location="cpu")
            if saved.get("model_family") != "scigantic_mt_aux8" or saved["tasks"][:4] != list(
                ACTIVITY_ENDPOINTS
            ):
                raise ValueError("Unsupported checkpoint family or endpoint order")
            means = np.asarray(saved["means"])
            scales = np.asarray(saved["scales"])
            mp, agg, head = build_model(saved["model_family"], len(saved["tasks"]))
            mp.load_state_dict(saved["encoder_state"])
            head.load_state_dict(saved["head_state"])
            mp, agg, head = mp.cuda().eval(), agg.cuda().eval(), head.cuda().eval()
            rows = []
            with torch.inference_mode():
                for batch in loader(smiles):
                    if time.monotonic() - start >= max_gpu_hours * 3600:
                        raise TimeoutError("Neural export exhausted its allocation")
                    batch.bmg.to("cuda")
                    rows.append(head(agg(mp(batch.bmg), batch.bmg.batch)).cpu().numpy())
            values = (np.concatenate(rows) * scales + means)[:, :4]
            if values.shape != (len(test), 4) or not np.isfinite(values).all():
                raise ValueError("Invalid fold predictions")
            predictions.append(values)
            del mp, agg, head, saved
        result = test[["Molecule_Name", "SMILES"]].copy()
        result[list(ACTIVITY_ENDPOINTS)] = np.mean(predictions, axis=0)
        output = output_dir / "activity_submission.csv"
        result.to_csv(output, index=False)
        validation = validate_submission(output, test, "activity")
        if not validation.valid:
            raise ValueError(validation.errors)
        record.update(
            status="completed",
            submission_sha256=sha256_file(output),
            validation=validation.to_dict(),
        )
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record["gpu_hours"] = allocation.finish()
        write_json(output_dir / "run_manifest.json", record)
    return record
