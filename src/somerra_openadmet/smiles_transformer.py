"""Small, parameter-matched SMILES encoder; optional neural dependency."""

from __future__ import annotations

import math

import torch
from torch import nn

TOKENIZER = {"kind": "ascii-character", "padding_id": 0, "vocabulary_size": 129}


def tokenize(smiles: list[str]) -> torch.Tensor:
    if not smiles or any(not s or not s.isascii() for s in smiles):
        raise ValueError("Expected nonempty ASCII SMILES")
    result = torch.zeros(len(smiles), max(map(len, smiles)), dtype=torch.long)
    for i, value in enumerate(smiles):
        result[i, : len(value)] = torch.tensor([ord(c) + 1 for c in value])
    return result


def decode(tokens):
    return "".join(chr(int(t) - 1) for t in tokens if int(t))


class SmilesTransformer(nn.Module):
    def __init__(self, feedforward: int, n_tasks: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(129, 192, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            192,
            6,
            feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        # TransformerEncoder clones layers: initialize each layer independently.
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.norm = nn.LayerNorm(192)
        self.head = nn.Linear(192, n_tasks)

    def forward(self, tokens):
        mask = tokens.eq(0)
        if mask.all(1).any():
            raise ValueError("Empty molecule")
        positions = torch.arange(tokens.shape[1], device=tokens.device).float()[:, None]
        rates = torch.exp(torch.arange(0, 192, 2, device=tokens.device) * (-math.log(10000) / 192))
        pe = torch.zeros(tokens.shape[1], 192, device=tokens.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * rates), torch.cos(positions * rates)
        encoded = self.norm(self.encoder(self.embedding(tokens) + pe, src_key_padding_mask=mask))
        pooled = encoded.masked_fill(mask[..., None], 0).sum(1) / (~mask).sum(1)[:, None]
        return self.head(pooled)


class ReferenceMember(nn.Module):
    def __init__(self, n_tasks=8):
        super().__init__()
        from chemprop import nn as cp

        self.encoder = cp.BondMessagePassing(d_h=700, depth=5, dropout=0.1)
        self.aggregate = cp.NormAggregation(norm=100)
        self.head = nn.Sequential(
            nn.BatchNorm1d(700),
            nn.Dropout(0.1),
            nn.Linear(700, 300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, n_tasks),
        )

    def forward(self, graph):
        return self.head(self.aggregate(self.encoder(graph), graph.batch))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_architecture(reference_count):
    base = count_parameters(SmilesTransformer(8))
    increment = count_parameters(SmilesTransformer(16)) - base
    k = max(0, round((reference_count - base) / increment))
    width = 8 + 8 * k
    count = count_parameters(SmilesTransformer(width))
    if abs(count / reference_count - 1) > 0.05:
        raise ValueError("Cannot match reference size within 5%")
    return {"family": "smiles_transformer", "feedforward": width, "n_tasks": 8}


def build_model(architecture):
    if architecture["family"] == "mt_aux8":
        return ReferenceMember(architecture.get("n_tasks", 8))
    if architecture["family"] == "smiles_transformer":
        return SmilesTransformer(architecture["feedforward"], architecture.get("n_tasks", 8))
    raise ValueError("Unknown model family")


def masked_mae(prediction, target, weights):
    observed = torch.isfinite(target)
    counts = observed.sum(0)
    active = counts > 0
    if not active.any():
        raise ValueError("No observed targets")
    safe = torch.where(observed, target, 0)
    per_head = ((prediction - safe).abs() * observed).sum(0) / counts.clamp_min(1)
    return (per_head[active] * weights[active]).sum() / weights[active].sum()


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["tokenizer"] != TOKENIZER:
        raise ValueError("Unsupported tokenizer")
    model = build_model(saved["architecture"])
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), saved
