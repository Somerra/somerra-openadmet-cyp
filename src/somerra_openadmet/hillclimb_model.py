"""Optional GPU model: matched weighted MAE and interval-loss experiments."""

import torch
from chemprop import models


def training_loss(predictions, targets, weights, loss_kind="mae", auxiliary_weight=1.0):
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
    task_weights[4:] = auxiliary_weight
    return (losses * mask * weights.view(-1, 1) * task_weights).sum() / mask.sum().clamp_min(1)


class StudyMPNN(models.MPNN):
    def __init__(self, *args, loss_kind="mae", auxiliary_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_kind = loss_kind
        self.auxiliary_weight = auxiliary_weight
        self.save_hyperparameters({"loss_kind": loss_kind, "auxiliary_weight": auxiliary_weight})

    def training_step(self, batch, batch_idx):
        bmg, descriptors, features, targets, weights, _, _ = batch
        predictions = self.predictor.train_step(self.fingerprint(bmg, descriptors, features))
        loss = training_loss(predictions, targets, weights, self.loss_kind, self.auxiliary_weight)
        self.log("train_loss", loss, batch_size=self.get_batch_size(batch), on_epoch=True)
        return loss
