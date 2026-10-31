"""Starting recipe, copied independently for each CYP. All hooks are editable."""

STRATEGY = {
    "name": "CYP2C9 Workbench graph baseline",
    "model": "Chemprop D-MPNN with one shared RegressionFFN and distinct task outputs",
    "loss": "masked weighted MAE on training-standardized labels / observed label count",
    "data": "available training parents, all 22 auxiliary tasks",
}

SMOKE_SETTINGS = {
    "hidden_dim": 32,
    "depth": 2,
    "ffn_hidden_dim": 32,
    "batch_size": 32,
    "warmup_epochs": 1,
}

SETTINGS = {
    "hidden_dim": 700,
    "depth": 5,
    "dropout": 0.1,
    "ffn_hidden_dim": 300,
    "ffn_num_layers": 2,
    "batch_size": 64,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "init_lr": 1e-4,
    "max_lr": 1e-3,
    "final_lr": 1e-4,
    "warmup_epochs": 2,
    "max_epochs": 400,
    "patience": 40,
    "primary_weight": 3.0,
    "auxiliary_weight": 0.3,
}


def build_model(n_tasks, settings):
    """Return standardized predictions in the exact task order supplied by the runner."""
    from chemprop import models, nn

    return models.MPNN(
        nn.BondMessagePassing(
            d_h=settings["hidden_dim"], depth=settings["depth"], dropout=settings["dropout"]
        ),
        nn.NormAggregation(norm=100),
        nn.RegressionFFN(
            input_dim=settings["hidden_dim"],
            hidden_dim=settings["ffn_hidden_dim"],
            n_layers=settings["ffn_num_layers"],
            dropout=settings["dropout"],
            n_tasks=n_tasks,
        ),
        batch_norm=True,
    )


def select_tasks(cyp, available):
    """Keep the scored CYP first; select any subset of available auxiliaries."""
    return available


def compute_loss(prediction, target, weights, context):
    """Editable differentiable objective; the challenge metric stays independent.

    context supplies settings, epoch, phase, task names, train-fitted mean/scale,
    and raw primary_low/primary_high intervals for these rows. Missing labels are
    NaN: mask them BEFORE nonlinear operations to avoid NaN gradients.
    """
    import torch

    observed = torch.isfinite(target)
    errors = (prediction - torch.nan_to_num(target, nan=0.0)).abs()
    return (errors * observed * weights).sum() / observed.sum().clamp_min(1)


def build_optimizer(model, settings):
    import torch

    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    return optimizers[settings["optimizer"]](
        model.parameters(), lr=settings["init_lr"], weight_decay=settings["weight_decay"]
    )


def build_scheduler(optimizer, settings, steps_per_epoch):
    from chemprop.schedulers import build_NoamLike_LRSched

    return build_NoamLike_LRSched(
        optimizer,
        settings["warmup_epochs"] * steps_per_epoch,
        (settings["max_epochs"] - settings["warmup_epochs"]) * steps_per_epoch,
        settings["init_lr"],
        settings["max_lr"],
        settings["final_lr"],
    )
