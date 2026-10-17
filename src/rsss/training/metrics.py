"""Metrics shared by the training module."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor
from torchmetrics import Metric


class StageMetric(Protocol):
    """The slice of the torchmetrics API the training module relies on.

    torchmetrics' own annotations describe ``update``/``compute`` in a way type
    checkers cannot call, so the module and its tests go through this protocol
    instead: any torchmetrics ``Metric`` implements it.
    """

    def update(self, preds: Tensor | None, target: Tensor) -> None:
        """Accumulate one batch."""
        ...

    def compute(self) -> Tensor:
        """Return the metric over everything accumulated so far."""
        ...

    def reset(self) -> None:
        """Drop everything accumulated so far."""
        ...


class ClassSupport(Metric):
    """Count the target pixels of each class over a split.

    Used to tell a class that is *absent* from a split — its IoU and F1 are
    undefined and must not be reported as 0 — from a class that simply scores
    badly. Counts are summed across processes by torchmetrics, so the verdict
    holds under distributed training.
    """

    counts: Tensor

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.add_state(
            "counts",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor | None, target: Tensor) -> None:
        """Accumulate how many pixels of each class the labels contain.

        ``preds`` is accepted because every torchmetrics metric takes it, and
        otherwise ignored: only the labels say which classes the split covers.
        """
        del preds
        self.counts += torch.bincount(target.flatten(), minlength=self.num_classes)

    def compute(self) -> Tensor:
        """Return the per-class target pixel counts, zero for absent classes."""
        return self.counts
