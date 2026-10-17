"""Shared PyTorch Lightning module for every segmentation model.

The module owns the training recipe (loss, optimizer, schedule, metrics) and
delegates the architecture to :func:`rsss.models.build_model`. Adding a
backbone (DINO and friends) therefore needs a registry entry and a config, not
a change here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import segmentation_models_pytorch as smp
import torch
from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import (
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

from rsss.models import build_model
from rsss.training.metrics import ClassSupport, StageMetric

# Splits whose metrics are accumulated and logged at the end of their epoch.
METRIC_STAGES = ("val", "test")
# Metrics accumulated for every stage in METRIC_STAGES.
METRIC_NAMES = ("iou", "f1", "support")


def _format_per_class(values: Mapping[str, float], metric: str) -> str:
    """Render the per-class metrics of one split as a compact line.

    Classes no pixel was seen for are reported as ``n/a`` rather than dropped,
    so a reader can tell "absent" from "not measured".
    """
    parts = []
    for key, value in values.items():
        if f"/{metric}/class_" not in key:
            continue
        index = key.rsplit("_", 1)[1]
        parts.append(f"{index}=n/a" if math.isnan(value) else f"{index}={value:.3f}")
    return " ".join(parts)


class SegmentationModule(LightningModule):
    """Train any registered segmentation model with a shared recipe.

    Parameters
    ----------
    model : Mapping[str, Any]
        Config forwarded to :func:`rsss.models.build_model`: a ``"name"`` key
        plus the builder's keyword arguments. The class count comes from
        ``num_classes``, so ``"classes"`` must not be set here.
    num_classes : int
        Number of classes, shared by the model head and the metrics.
    loss : Mapping[str, float], optional
        Weights of the two loss terms, e.g. ``{"ce": 0.5, "dice": 0.5}``.
        Cross-entropy is unweighted unless ``class_weights`` is given; the Dice
        term never ignores a class.
    optimizer : Mapping[str, Any], optional
        ``{"lr": ..., "weight_decay": ...}`` for AdamW.
    scheduler : Mapping[str, Any], optional
        ``{"warmup_epochs": ..., "t_max": ...}``. ``t_max`` defaults to the
        Trainer's ``max_epochs``, so the cosine horizon follows the run.
    class_weights : Sequence[float], optional
        Per-class weights for the cross-entropy term, ordered by class index.
        Use it to counter the heavy class imbalance; default is uniform.

    Notes
    -----
    Masks are expected as class indices of shape ``(batch, height, width)``,
    as produced by the fit datasets.
    """

    def __init__(
        self,
        model: Mapping[str, Any],
        num_classes: int,
        loss: Mapping[str, float] | None = None,
        optimizer: Mapping[str, Any] | None = None,
        scheduler: Mapping[str, Any] | None = None,
        class_weights: Sequence[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if "classes" in model:
            raise ValueError(
                "'classes' must be set through num_classes, not in the model "
                "config: the model head and the metrics have to agree."
            )

        self.num_classes = num_classes
        self.net = build_model({**model, "classes": num_classes})
        self.optimizer_cfg: dict[str, Any] = dict(optimizer or {})
        self.scheduler_cfg: dict[str, Any] = dict(scheduler or {})

        loss_cfg = dict(loss) if loss else {"ce": 0.5, "dice": 0.5}
        self.ce_weight = float(loss_cfg.get("ce", 0.5))
        self.dice_weight = float(loss_cfg.get("dice", 0.5))
        if self.ce_weight == 0 and self.dice_weight == 0:
            raise ValueError("Set a non-zero weight for at least one loss term.")

        if class_weights is not None and len(class_weights) != num_classes:
            raise ValueError(
                f"class_weights has {len(class_weights)} entries but the model "
                f"has {num_classes} classes."
            )
        # Both losses keep their weights as buffers, so moving the module to a
        # device moves the loss state with it.
        self.ce_loss = nn.CrossEntropyLoss(
            weight=None
            if class_weights is None
            else torch.tensor(list(class_weights), dtype=torch.float),
        )
        # No ignore index: all 13 FLAIR classes are real, so none of them is
        # dropped from the soft Dice term.
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=None)

        self.metrics = nn.ModuleDict(
            {
                stage: nn.ModuleDict(
                    {
                        "iou": MulticlassJaccardIndex(
                            num_classes=num_classes, average="none"
                        ),
                        "f1": MulticlassF1Score(
                            num_classes=num_classes, average="none"
                        ),
                        "support": ClassSupport(num_classes=num_classes),
                    }
                )
                for stage in METRIC_STAGES
            }
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the per-class logits of ``x``."""
        return self.net(x)

    def _metric(self, stage: str, name: str) -> StageMetric:
        """Return one metric of one split.

        ``nn.ModuleDict`` lookups are typed as plain modules, so this keeps the
        torchmetrics interface (``update``, ``compute``, ``reset``) in view.
        """
        stage_metrics = cast(nn.ModuleDict, self.metrics[stage])
        return cast(StageMetric, stage_metrics[name])

    def _compute_loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """Weighted cross-entropy plus Dice, both over the full class set."""
        return self.ce_weight * self.ce_loss(
            logits, target
        ) + self.dice_weight * self.dice_loss(logits, target)

    def _update_metrics(self, stage: str, logits: Tensor, target: Tensor) -> None:
        """Feed one batch to the metrics of a reporting split."""
        for name in METRIC_NAMES:
            self._metric(stage, name).update(logits, target)

    def _shared_step(self, batch: Mapping[str, Tensor], stage: str) -> Tensor:
        """Score one batch, accumulating metrics for the splits that report them."""
        logits = self(batch["img"])
        target = batch["msk"].long()
        loss = self._compute_loss(logits, target)
        if stage in METRIC_STAGES:
            self._update_metrics(stage, logits, target)
        return loss

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, "train")
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["img"].shape[0],
        )
        return loss

    def validation_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, "val")
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["img"].shape[0],
        )
        return loss

    def test_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, "test")
        self.log(
            "test/loss",
            loss,
            on_epoch=True,
            batch_size=batch["img"].shape[0],
        )
        return loss

    def _epoch_metrics(self, stage: str) -> dict[str, float]:
        """Per-class IoU/F1 and their means over the classes seen in ``stage``.

        A class absent from the split has no defined IoU or F1: it is reported
        as ``nan`` and left out of the mean, so absence is never mistaken for a
        score of zero. The per-class values are returned alongside the means.
        """
        present = self._metric(stage, "support").compute() > 0
        if not bool(present.any()):
            logger.warning(
                "No labelled pixel seen during '{}': metrics are undefined.", stage
            )
        values: dict[str, float] = {}
        for name, aggregate in (("iou", "mIoU"), ("f1", "mF1")):
            per_class = self._metric(stage, name).compute()
            values[f"{stage}/{aggregate}"] = (
                float(per_class[present].mean())
                if bool(present.any())
                else float("nan")
            )
            values.update(
                {
                    f"{stage}/{name}/class_{index}": float(value)
                    if bool(present[index])
                    else float("nan")
                    for index, value in enumerate(per_class)
                }
            )
        return values

    def _log_stage(self, stage: str) -> None:
        """Log a stage's metrics to Lightning and to loguru, then reset them."""
        values = self._epoch_metrics(stage)
        self.log_dict(values)
        logger.info(
            "{} mIoU={:.4f} mF1={:.4f} | IoU per class: {}",
            stage,
            values[f"{stage}/mIoU"],
            values[f"{stage}/mF1"],
            _format_per_class(values, "iou"),
        )
        for name in METRIC_NAMES:
            self._metric(stage, name).reset()

    def on_validation_epoch_end(self) -> None:
        self._log_stage("val")

    def on_test_epoch_end(self) -> None:
        self._log_stage("test")

    def _scheduler_horizon(self) -> int:
        """Total number of epochs the cosine schedule spans."""
        t_max = self.scheduler_cfg.get("t_max")
        if t_max is not None:
            return int(t_max)
        try:
            max_epochs = self.trainer.max_epochs
        except RuntimeError as exc:
            raise ValueError(
                "scheduler.t_max is unset and the module is not attached to a "
                "Trainer, so the cosine horizon is unknown. Set scheduler.t_max "
                "or call configure_optimizers() from a Trainer."
            ) from exc
        if max_epochs is None:
            raise ValueError("The Trainer has no max_epochs, cannot size the schedule.")
        return int(max_epochs)

    def _build_scheduler(self, optimizer: AdamW) -> LRScheduler:
        """Cosine annealing over the run, preceded by a linear warmup."""
        warmup = int(self.scheduler_cfg.get("warmup_epochs") or 0)
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(self._scheduler_horizon() - warmup, 1)
        )
        if warmup <= 0:
            return cosine
        warmup_schedule = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        return SequentialLR(
            optimizer, schedulers=[warmup_schedule, cosine], milestones=[warmup]
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """AdamW on the whole network, with a warmup-then-cosine schedule."""
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.optimizer_cfg.get("lr", 3e-4)),
            weight_decay=float(self.optimizer_cfg.get("weight_decay", 1e-4)),
        )
        config: OptimizerLRSchedulerConfig = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self._build_scheduler(optimizer),
                "interval": "epoch",
            },
        }
        return config
