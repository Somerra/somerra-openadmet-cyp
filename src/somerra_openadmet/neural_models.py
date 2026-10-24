"""Replaceable neural recipes; data selection and scoring live in shared modules."""


def build_model(family, n_tasks, checkpoint=None):
    import torch
    from chemprop import nn

    from .neural import encoder

    if family == "legacy":
        mp = encoder(checkpoint)
        return (
            mp,
            nn.MeanAggregation(),
            torch.nn.Sequential(
                torch.nn.Linear(mp.output_dim, 256), torch.nn.ReLU(), torch.nn.Linear(256, n_tasks)
            ),
        )
    if family != "scigantic_mt_aux8" or checkpoint is not None:
        raise ValueError("Unsupported neural model recipe")
    mp = nn.BondMessagePassing(d_h=700, depth=5, dropout=0.1)
    head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(700),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(700, 300),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(300, 300),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(300, n_tasks),
    )
    return mp, nn.NormAggregation(norm=100), head


def masked_mae(prediction, target, task_weights, *, bounds=None):
    """MAE for observed heads; optional pure interval hinge on the first four heads."""
    import torch

    observed = torch.isfinite(target)
    target = torch.where(observed, target, torch.zeros_like(target))
    losses = (prediction - target).abs()
    if bounds is not None:
        lower, upper = bounds
        active = observed[:, :4]
        lower = torch.where(active, lower, torch.zeros_like(lower))
        upper = torch.where(active, upper, torch.zeros_like(upper))
        if not (torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("Observed intervals require finite bounds")
        if ((lower > upper) & active).any():
            raise ValueError("Reversed intervals")
        interval = torch.relu(lower - prediction[:, :4]) + torch.relu(prediction[:, :4] - upper)
        losses = torch.cat([interval, losses[:, 4:]], dim=1)
    counts = observed.sum(0)
    active_heads = counts > 0
    if not active_heads.any():
        raise ValueError("Batch has no observed targets")
    if not torch.isfinite(task_weights).all() or (task_weights <= 0).any():
        raise ValueError("Task weights must be finite and positive")
    per_head = (losses * observed).sum(0) / counts.clamp_min(1)
    return (per_head[active_heads] * task_weights[active_heads]).sum() / task_weights[
        active_heads
    ].sum()
