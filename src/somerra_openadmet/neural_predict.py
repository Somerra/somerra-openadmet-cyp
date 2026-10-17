"""Reload a screening checkpoint for reproducibility checks, not promotion."""

from pathlib import Path

import numpy as np
import pandas as pd


def predict_checkpoint(checkpoint: Path, smiles: list[str], *, device: str = "cpu"):
    import torch
    from chemprop import nn
    from chemprop.nn.utils import Activation

    from .neural import loader

    torch.set_num_threads(8)
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.25)
    # Older local runs stored Lightning's dict subclass; reconstruct as plain data.
    with torch.serialization.safe_globals(
        [(dict, "lightning.fabric.utilities.data.AttributeDict"), nn.BondMessagePassing, Activation]
    ):
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    parameters = {k: v for k, v in saved["encoder_parameters"].items() if k != "cls"}
    mp = nn.BondMessagePassing(**parameters)
    mp.load_state_dict(saved["encoder_state"])
    mp = mp.to(device).eval()
    tasks = saved["tasks"]
    head = torch.nn.Sequential(
        torch.nn.Linear(mp.output_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, len(tasks)),
    )
    head.load_state_dict(saved["head_state"])
    head = head.to(device).eval()
    aggregate = nn.MeanAggregation()
    rows = []
    with torch.inference_mode():
        for batch in loader(smiles):
            batch.bmg.to(device)
            rows.append(head(aggregate(mp(batch.bmg), batch.bmg.batch)).cpu().numpy())
    if not rows:
        return pd.DataFrame(columns=tasks, dtype=float)
    values = np.concatenate(rows) * np.asarray(saved["scales"]) + np.asarray(saved["means"])
    for index, task in enumerate(tasks):
        if task.startswith("PubChem_"):
            values[:, index] = 1 / (1 + np.exp(-np.clip(values[:, index], -80, 80)))
    if not np.isfinite(values).all():
        raise ValueError("Non-finite checkpoint predictions")
    return pd.DataFrame(values, columns=tasks)
