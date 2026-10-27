"""Editable model settings; data partitions and evaluation settings are not recipe fields."""

import math

DEFAULT_MODEL = {
    "family": "chemprop_bond_mpnn",
    "hidden_dim": 2048,
    "depth": 4,
    "dropout": 0.2,
    "ffn_hidden_dim": 1024,
    "ffn_layers": 2,
    "init_lr": 1e-4,
    "max_lr": 1e-3,
    "final_lr": 1e-4,
    "loss_kind": "mae",
    "auxiliary_weight": 1.0,
    "loss_normalization": "observations",
}


def resolve_model(settings):
    unknown = set(settings) - set(DEFAULT_MODEL)
    if unknown:
        raise ValueError(f"Unknown model settings (evaluation is fixed): {sorted(unknown)}")
    model = {**DEFAULT_MODEL, **settings}
    if model["family"] != "chemprop_bond_mpnn":
        raise ValueError("Unsupported model family")
    if model["loss_kind"] not in {"mae", "interval"}:
        raise ValueError("Unknown model loss")
    if model["loss_normalization"] not in {"observations", "tasks"}:
        raise ValueError("Unknown loss normalization")
    for key in ["hidden_dim", "depth", "ffn_hidden_dim", "ffn_layers"]:
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"Invalid model setting: {key}")
    for key in ["dropout", "init_lr", "max_lr", "final_lr", "auxiliary_weight"]:
        if (
            isinstance(model[key], bool)
            or not isinstance(model[key], (int, float))
            or not math.isfinite(model[key])
        ):
            raise ValueError(f"Invalid model setting: {key}")
    if not 0 <= model["dropout"] < 1 or not 0 <= model["auxiliary_weight"] <= 16:
        raise ValueError("Invalid dropout or auxiliary weight")
    if not 0 < model["init_lr"] <= model["max_lr"] or not 0 < model["final_lr"] <= model["max_lr"]:
        raise ValueError("Invalid learning rates")
    return model
