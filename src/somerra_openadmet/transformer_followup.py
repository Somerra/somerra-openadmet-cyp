"""Frozen inner-validation-only follow-up; no outer scoring or automatic submission."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .budget import GPUBudget
from .campaign import write_json
from .download import sha256_file
from .smiles_transformer import ReferenceMember, count_parameters, matched_architecture, tokenize
from .transformer_pilot import Inputs, prepare, train_one


class CachedInputs(Inputs):
    """Cache canonical and fit-only enumerated tokens on the selected device."""

    def __init__(self, smiles, fit, *, device="cuda", augment=False):
        super().__init__(smiles)
        self.device = device
        self.canonical = tokenize(self.smiles).to(device)
        self.lengths = np.array([len(s) for s in self.smiles])
        self.variants = None
        self.fit = set(map(int, fit))
        self.mapping = {int(index): row for row, index in enumerate(fit)}
        if augment:
            variants = []
            for index in fit:
                molecule = Chem.MolFromSmiles(self.smiles[index])
                values = Chem.MolToRandomSmilesVect(
                    molecule, 8, randomSeed=42 + int(index), isomericSmiles=True
                )
                # Ensure enumeration preserves the original molecule and stereochemistry.
                if any(
                    Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
                    != self.smiles[index]
                    for s in values
                ):
                    raise ValueError("Enumeration changed molecular identity")
                variants.extend(values)
            self.variant_lengths = np.array([len(s) for s in variants]).reshape(len(fit), 8)
            tokens = tokenize(variants).to(device)
            self.variants = tokens.reshape(len(fit), 8, -1)

    def batch(self, indices, family, device):
        if family != "smiles_transformer":
            return super().batch(indices, family, device)
        if device != self.device:
            raise ValueError("Cached input device mismatch")
        width = int(self.lengths[indices].max())
        return self.canonical[torch.as_tensor(indices, device=device), :width]

    def training_batch(self, indices, family, device, epoch):
        if self.variants is None or family != "smiles_transformer":
            return self.batch(indices, family, device)
        if not set(map(int, indices)).issubset(self.fit):
            raise ValueError("Augmentation is restricted to the fit partition")
        rows = np.array([self.mapping[int(i)] for i in indices])
        variant = (epoch - 1) % 8
        width = int(self.variant_lengths[rows, variant].max())
        return self.variants[torch.as_tensor(rows, device=device), variant, :width]


def run(config_path):
    config = json.loads(config_path.read_text())
    output = Path(config["output_dir"])
    frame, targets, means, scales, splits, hashes = prepare(config)
    prior = json.loads(Path(config["prior_manifest"]).read_text())
    if hashes != prior["input_hashes"]:
        raise ValueError("Pilot input hashes differ")
    previous_split = Path(config["prior_manifest"]).parent / "splits.csv"
    import pandas as pd

    previous = pd.read_csv(previous_split)
    expected = {"fit": splits[0], "inner_validation": splits[1], "outer_test": splits[2]}
    for name, indices in expected.items():
        if list(previous.index[previous.partition == name]) != list(indices):
            raise ValueError("Split differs from frozen pilot")
    hours = float(config["max_gpu_hours"])
    if not 0 < hours <= 0.75:
        raise ValueError("Follow-up cap exceeds 45 minutes")
    ledger_path = Path(config["ledger"])
    ledger = json.loads(ledger_path.read_text())
    charged = ledger[str(Path(config["prior_manifest"]).parent.resolve())]["used_hours"]
    for path in prior["config"].get("prior_attempts", []):
        charged += ledger[str(Path(path).resolve())]["used_hours"]
    if hours + charged > 1:
        raise ValueError("Cumulative one-hour allocation exceeded")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    architecture = matched_architecture(count_parameters(ReferenceMember()))
    record = {
        "config": config,
        "input_hashes": hashes,
        "architecture": architecture,
        "source_hashes": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
        "split_sha256": sha256_file(previous_split),
        "prior_gpu_hours": charged,
        "purpose": "adaptive exploratory inner-validation tuning only",
        "selection": "minimum inner weighted masked MAE; tie broken by trial order",
        "outer_evaluated": False,
        "complete": False,
        "results": {},
        "open_source_code": False,
        "model_report_link": "",
    }
    write_json(output / "run_manifest.json", record)
    torch.cuda.set_per_process_memory_fraction(0.7)
    allocation = GPUBudget(ledger_path, output, hours)
    allocation.reserve()
    deadline = time.monotonic() + hours * 3600
    try:
        for index, trial in enumerate(config["trials"]):
            trial_output = output / trial["name"]
            trial_output.mkdir()
            remaining = deadline - time.monotonic()
            if remaining < 60:
                raise TimeoutError("Follow-up allocation exhausted")
            trial_deadline = time.monotonic() + remaining / (len(config["trials"]) - index) - 10
            inputs = CachedInputs(
                frame.canonical_smiles.tolist(), splits[0], augment=trial["augment"]
            )
            model = (
                {"family": "mt_aux8", "n_tasks": 8}
                if trial["family"] == "mt_aux8"
                else architecture
            )
            result = train_one(
                model,
                inputs,
                targets,
                means,
                scales,
                splits,
                frame,
                trial_output,
                trial_deadline,
                training=trial["training"],
            )
            curves = json.loads((trial_output / f"{trial['family']}-curves.json").read_text())
            selected = next(row for row in curves if row["epoch"] == result["best_epoch"])
            result["selected_inner"] = selected
            record["results"][trial["name"]] = result
            write_json(output / "run_manifest.json", record)
            del inputs
            torch.cuda.empty_cache()
        record["selected_trial"] = min(
            record["results"],
            key=lambda k: record["results"][k]["selected_inner"]["validation_loss"],
        )
        record["complete"] = True
    except BaseException as exc:
        record["error"] = repr(exc)
        raise
    finally:
        record["gpu_hours"] = allocation.finish()
        write_json(output / "run_manifest.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
