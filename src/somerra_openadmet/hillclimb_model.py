"""Optional GPU model: matched weighted MAE and interval-loss experiments."""

import torch
from chemprop import models, nn

from .model_recipe import resolve_model


def training_loss(
    predictions,
    targets,
    weights,
    loss_kind="mae",
    auxiliary_weight=1.0,
    loss_normalization="observations",
    task_scales=None,
    scored_endpoint_weight=1.0,
):
    tasks = predictions.shape[1]
    values = targets[:, :tasks]
    mask = values.isfinite()
    safe = values.nan_to_num(nan=0.0)
    losses = (predictions - safe).abs()
    if loss_kind == "interval":
        if targets.shape[1] != tasks + 8:
            raise ValueError("Interval loss requires four lower and upper bounds")
        lo, hi = targets[:, tasks : tasks + 4], targets[:, tasks + 4 :]
        observed = mask[:, :4]
        if not ((lo.isfinite() & hi.isfinite() & (lo <= hi)) | ~observed).all():
            raise ValueError("Invalid observed training intervals")
        distance = torch.relu(lo.nan_to_num() - predictions[:, :4]) + torch.relu(
            predictions[:, :4] - hi.nan_to_num()
        )
        losses = torch.cat([distance, losses[:, 4:]], dim=1)
    elif loss_kind != "mae":
        raise ValueError("Unknown loss")
    task_weights = torch.ones(tasks, device=predictions.device)
    task_weights[:4] = scored_endpoint_weight
    task_weights[4:] = auxiliary_weight
    weighted = losses * mask * weights.view(-1, 1) * task_weights
    if loss_normalization == "tasks":
        if task_scales is None or task_scales.shape != (tasks,):
            raise ValueError("Task normalization requires training-only label frequencies")
        return (weighted * task_scales).sum() / predictions.shape[0]
    if loss_normalization != "observations":
        raise ValueError("Unknown loss normalization")
    return weighted.sum() / mask.sum().clamp_min(1)


class StudyMPNN(models.MPNN):
    def __init__(
        self,
        *args,
        loss_kind="mae",
        auxiliary_weight=1.0,
        loss_normalization="observations",
        task_scales=None,
        scored_endpoint_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_kind = loss_kind
        self.auxiliary_weight = auxiliary_weight
        self.scored_endpoint_weight = scored_endpoint_weight
        self.loss_normalization = loss_normalization
        # Legacy observation-normalized checkpoints have no additional buffer.
        if task_scales is not None:
            self.register_buffer("task_scales", torch.tensor(task_scales, dtype=torch.float32))
        self.save_hyperparameters(
            {
                "loss_kind": loss_kind,
                "auxiliary_weight": auxiliary_weight,
                "scored_endpoint_weight": scored_endpoint_weight,
                "loss_normalization": loss_normalization,
                "task_scales": task_scales,
            }
        )

    def training_step(self, batch, batch_idx):
        bmg, descriptors, features, targets, weights, _, _ = batch
        predictions = self.predictor.train_step(self.fingerprint(bmg, descriptors, features))
        loss = training_loss(
            predictions,
            targets,
            weights,
            self.loss_kind,
            self.auxiliary_weight,
            self.loss_normalization,
            getattr(self, "task_scales", None),
            scored_endpoint_weight=self.scored_endpoint_weight,
        )
        self.log("train_loss", loss, batch_size=self.get_batch_size(batch), on_epoch=True)
        return loss


def build_model(scaler, tasks, settings, task_scales=None):
    from .neural import validate_task_order

    validate_task_order(tasks)
    config = resolve_model(settings)
    return StudyMPNN(
        nn.BondMessagePassing(
            d_h=config["hidden_dim"], depth=config["depth"], dropout=config["dropout"]
        ),
        nn.NormAggregation(),
        nn.RegressionFFN(
            n_tasks=len(tasks),
            input_dim=config["hidden_dim"],
            hidden_dim=config["ffn_hidden_dim"],
            n_layers=config["ffn_layers"],
            dropout=config["dropout"],
            criterion=nn.metrics.MAE(),
            output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
        ),
        batch_norm=True,
        metrics=[nn.metrics.MAE()],
        warmup_epochs=2,
        init_lr=config["init_lr"],
        max_lr=config["max_lr"],
        final_lr=config["final_lr"],
        loss_kind=config["loss_kind"],
        auxiliary_weight=config["auxiliary_weight"],
        scored_endpoint_weight=config["scored_endpoint_weight"],
        loss_normalization=config["loss_normalization"],
        task_scales=task_scales,
    )
