"""Pinned Mol-JEPA loading and differentiable downstream adapters.

Only locally reviewed, checksum-bound Python is imported. No remote-code loader.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

from .download import sha256_file

SOURCE_REVISION = "27f0065b69c73d18eec7a32ecc413b6dc3f33d90"
MODEL_REVISION = "4c912b450175f31b5ba913a5dc921c03b27b985a"
FILES = (
    "configuration_moljepa.py",
    "modeling_moljepa.py",
    "collator.py",
    "data.py",
    "config.json",
    "model.safetensors",
)


def load_foundation(directory):
    directory = Path(directory).resolve()
    evidence = json.loads((directory / "provenance.json").read_text())
    if (
        evidence.get("model_revision") != MODEL_REVISION
        or evidence.get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("Unreviewed Mol-JEPA revision")
    if evidence.get("permitted_use") != "user-confirmed noncommercial research":
        raise ValueError("Missing permitted-use evidence")
    for name in FILES:
        if sha256_file(directory / name) != evidence["sha256"][name]:
            raise ValueError(f"Mol-JEPA checksum mismatch: {name}")
    package = "_pinned_moljepa_" + evidence["sha256"]["modeling_moljepa.py"][:16]
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(directory)]
        sys.modules[package] = module
    configuration = importlib.import_module(package + ".configuration_moljepa")
    implementation = importlib.import_module(package + ".modeling_moljepa")
    config = configuration.MolJEPAConfig.from_dict(
        json.loads((directory / "config.json").read_text())
    )
    model = implementation.MolJEPAModel(config)
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    return model, evidence


def encode(model, smiles):
    """Authors' downstream path, intentionally without no_grad/detach."""
    batch = model.smiles_to_batch(list(smiles)).to(next(model.parameters()).device)
    modalities = [
        f"{m['name']}_x_ptr" if f"{m['name']}_x_ptr" in batch else None
        for m in model.model.modalities_spec
    ]
    device = next(model.parameters()).device
    active = (
        torch.stack(
            [
                batch[k][1:] - batch[k][:-1]
                if k is not None
                else torch.zeros(len(smiles), device=device)
                for k in modalities
            ]
        ).T
        > 0
    )
    full = model.model.encode(batch, modalities, active)
    predictions, cls, embeddings = model.model.predict(full, active.clone(), active)
    result = {"cls": cls, "predictions": predictions, "embeddings": embeddings}
    expected = {
        "cls": (len(smiles), 512),
        "predictions": (len(smiles), 12, 512),
        "embeddings": (len(smiles), 13, 512),
    }
    for name, value in result.items():
        if tuple(value.shape) != expected[name] or not torch.isfinite(value).all():
            raise ValueError(f"Invalid Mol-JEPA representation: {name}")
    return result


class TokenHead(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(512, 4, 2048, 0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(512, 4)

    def forward(self, value):
        if value.ndim != 3 or value.shape[1:] != (12, 512):
            raise ValueError("Expected 12 predicted modality tokens of width 512")
        return self.output(self.transformer(value).mean(dim=1))


class AttentionWithLoRA(nn.Module):
    """Preserve MHA math but call the PEFT output module, including its dropout.

    PyTorch MHA otherwise reads out_proj.weight directly, bypassing LoRA.forward.
    """

    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.batch_first = True

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        import torch.nn.functional as F

        a = self.attention
        width = a.embed_dim
        identity = torch.eye(width, device=query.device, dtype=query.dtype)
        output, weights = F.multi_head_attention_forward(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            width,
            a.num_heads,
            a.in_proj_weight,
            a.in_proj_bias,
            a.bias_k,
            a.bias_v,
            a.add_zero_attn,
            a.dropout,
            identity,
            None,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        return a.out_proj(output.transpose(0, 1)), weights


def install_lora(model):
    from peft import LoraConfig, get_peft_model

    model.requires_grad_(False)
    core = model.model
    core.transformer_head = get_peft_model(
        core.transformer_head,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["out_proj", "linear1", "linear2"],
            bias="none",
        ),
    )
    # Use explicit layer calls even in eval to avoid fused fastpaths reading raw weights.
    for layer in core.transformer_head.transformer.layers:
        layer.self_attn = AttentionWithLoRA(layer.self_attn)

        def forward(src, src_mask=None, src_key_padding_mask=None, is_causal=False, layer=layer):
            x = src
            normalized = layer.norm1(x)
            attended = layer.self_attn(
                normalized,
                normalized,
                normalized,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
            x = x + layer.dropout1(attended)
            x = x + layer.dropout2(
                layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
            )
            return x

        layer.forward = forward


class Regressor(nn.Module):
    def __init__(self, arm, model_dir=None):
        super().__init__()
        if arm not in {"frozen", "tokens", "full", "lora"}:
            raise ValueError("Unknown Mol-JEPA arm")
        self.arm = arm
        self.encoder = None
        if arm in {"full", "lora"}:
            self.encoder, self.provenance = load_foundation(model_dir)
            if arm == "lora":
                install_lora(self.encoder)
        self.inactive_parameter_names = set()
        if arm == "full":
            # SMILES supplies only graph input; other expert encoders have no forward path.
            self.inactive_parameter_names = {
                "encoder." + n
                for n, _ in self.encoder.named_parameters()
                if n.startswith("model.encoders.") and not n.startswith("model.encoders.graph.")
            }
        self.head = (
            TokenHead()
            if arm == "tokens"
            else nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 4)
            )
        )

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None and self.arm == "lora":
            self.encoder.eval()
            self.encoder.model.transformer_head.train(mode)
        return self

    def forward(self, value):
        if self.encoder is not None:
            value = encode(self.encoder, value)["cls"]
        return self.head(value)


def write_views(output, rows, views, metadata):
    """Publish all views together; the root manifest is the completion marker."""
    import numpy as np

    from .representations import atomic_json, write_cache

    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Frozen cache is immutable")
    shapes = {
        "cls": (len(rows), 512),
        "predictions": (len(rows), 12, 512),
        "embeddings": (len(rows), 13, 512),
    }
    if set(views) != set(shapes):
        raise ValueError("All three representation views are required")
    for name, shape in shapes.items():
        value = views[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid representation view: {name}")
    manifests = {}
    for name, value in views.items():
        folder = output / name
        folder.mkdir(parents=True, exist_ok=False)
        manifests[name] = write_cache(folder, rows, value, {**metadata, "representation": name})
    result = {**metadata, "schema_version": 2, "complete": True, "views": manifests}
    atomic_json(output / "manifest.json", result)
    return result


def load_view(directory, frame, view, expected_provenance):
    import numpy as np
    import pandas as pd

    from .representations import identities

    directory = Path(directory)
    root = json.loads((directory / "manifest.json").read_text())
    if root.get("schema_version") != 2 or not root.get("complete"):
        raise ValueError("Incomplete Mol-JEPA cache")
    for key, value in expected_provenance.items():
        if root.get(key) != value:
            raise ValueError(f"Incompatible Mol-JEPA cache: {key}")
    folder = directory / view
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest != root["views"][view]:
        raise ValueError("View manifest changed")
    for name, field in [("rows.csv", "rows_sha256"), ("embeddings.npy", "embeddings_sha256")]:
        if sha256_file(folder / name) != manifest[field]:
            raise ValueError("Cache checksum mismatch")
    rows = pd.read_csv(folder / "rows.csv")
    if not identities(rows).equals(rows):
        raise ValueError("Invalid cache identities")
    query = identities(frame)
    positions = pd.Series(range(len(rows)), index=rows.Molecule_Name)
    if not set(query.Molecule_Name).issubset(positions.index):
        raise ValueError("Cache missing molecule IDs")
    indices = positions.loc[query.Molecule_Name].to_numpy()
    if not np.array_equal(rows.iloc[indices].smiles_sha256, query.smiles_sha256):
        raise ValueError("Cache SMILES mismatch")
    array = np.load(folder / "embeddings.npy", allow_pickle=False, mmap_mode="r")
    shape = {
        "cls": (len(rows), 512),
        "predictions": (len(rows), 12, 512),
        "embeddings": (len(rows), 13, 512),
    }[view]
    if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Invalid cached representation")
    return np.asarray(array[indices]), root


class Inputs:
    def __init__(self, frame, mode, device, features=None):
        self.device = device
        self.smiles = frame.SMILES.to_numpy()
        self.features = None if features is None else torch.as_tensor(features, device=device)

    def batch(self, indices):
        return (
            self.features[indices] if self.features is not None else self.smiles[indices].tolist()
        )


def profile_extraction(model, rows, run):
    import time

    import numpy as np

    from .representations import atomic_json

    order = np.argsort(-rows.SMILES.str.len().to_numpy())
    records = []
    for size in sorted({min(n, len(rows)) for n in (64, 128, 256, 512, 1024)}):
        result = {"batch_size": size, "status": "oom"}
        value = None
        try:
            with torch.no_grad():
                for iteration in range(7):
                    run.check()
                    if iteration == 2:
                        run.sync()
                        start = time.monotonic()
                    value = encode(model, rows.iloc[order[:size]].SMILES)
                run.sync()
                result.update(
                    status="ok", molecules_per_second=5 * size / (time.monotonic() - start)
                )
        except torch.cuda.OutOfMemoryError:
            pass
        finally:
            del value
            if run.device == "cuda":
                torch.cuda.empty_cache()
        records.append(result)
        atomic_json(run.output / "extraction-profile.json", records)
    good = [r for r in records if r["status"] == "ok"]
    if not good:
        raise RuntimeError("No Mol-JEPA extraction batch fits")
    best = max(r["molecules_per_second"] for r in good)
    return min(r["batch_size"] for r in good if r["molecules_per_second"] >= best * 0.95)


def cache_representations(config_path):
    import numpy as np
    import pandas as pd

    from .constants import TEST_FILE
    from .representations import Run, batch_indices, identities, prepared_data, read_config

    config = read_config(
        config_path,
        {"encoder", "prepared_dir", "blind_csv", "checkpoint", "batch_size"},
        {"prepared_dir", "blind_csv", "checkpoint", "output_dir", "max_gpu_hours"},
    )
    frame, evidence = prepared_data(Path(config["prepared_dir"]))
    blind_path = Path(config["blind_csv"])
    if sha256_file(blind_path) != evidence["input_sha256"][TEST_FILE]:
        raise ValueError("Blind input changed")
    rows = identities(pd.concat([frame, pd.read_csv(blind_path)], ignore_index=True))
    if set(rows.parent[: len(frame)]) & set(rows.parent[len(frame) :]):
        raise ValueError("Blind parents overlap training")
    model, pin = load_foundation(config["checkpoint"])
    with Run(config) as run:
        model.to(run.device).eval()
        batch_size = config.get("batch_size", "auto")
        if batch_size == "auto":
            batch_size = profile_extraction(model, rows, run)
        run.record["batch_size"] = batch_size
        run.save()
        values = {k: [] for k in ("cls", "predictions", "embeddings")}
        with torch.no_grad():
            for indices in batch_indices(np.arange(len(rows)), batch_size):
                run.check()
                for name, value in encode(model, rows.iloc[indices].SMILES).items():
                    values[name].append(value.float().cpu().numpy())
        names = [m["name"] for m in model.model.modalities_spec]
        metadata = {
            "encoder": "mol_jepa",
            "model_revision": MODEL_REVISION,
            "source_revision": SOURCE_REVISION,
            "checkpoint_sha256": pin["sha256"]["model.safetensors"],
            "model_provenance": pin,
            "precision": "float32",
            "token_names": names,
            "embedding_token_names": ["CLS"] + names,
            "active_mask": [name == "graph" for name in names],
            "featurizer": "pinned GraphFeaturizer; explicit H; canonical atom order",
            "input_sha256": {
                "training": evidence["training_sha256"],
                "blind": sha256_file(blind_path),
            },
        }
        write_views(run.output, rows, {k: np.concatenate(v) for k, v in values.items()}, metadata)
        run.record.update(complete=True, rows=len(rows))
    return run.record
