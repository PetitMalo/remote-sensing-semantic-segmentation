"""Unit tests for the shared ``SegmentationModule``.

The network is a UNet with random weights, so nothing is downloaded and the
tests stay offline. Loss arithmetic, metric aggregation and the optimizer
contract are checked directly; one ``fast_dev_run`` fit covers the Lightning
plumbing end to end.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRSchedulerConfig,
)
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset

from rsss.training.module import SegmentationModule, _format_per_class

NUM_CLASSES = 3
IN_CHANNELS = 3
SIZE = 64


def make_module(**overrides: Any) -> SegmentationModule:
    """Return a tiny UNet module, with test-friendly defaults."""
    kwargs: dict[str, Any] = {
        "model": {
            "name": "unet",
            "encoder_weights": None,
            "in_channels": IN_CHANNELS,
        },
        "num_classes": NUM_CLASSES,
        "scheduler": {"warmup_epochs": 0, "t_max": 4},
    }
    kwargs.update(overrides)
    return SegmentationModule(**kwargs)


@pytest.fixture
def batch() -> dict[str, Tensor]:
    """A two-sample batch shaped like the fit datasets' output."""
    torch.manual_seed(0)
    return {
        "img": torch.randn(2, IN_CHANNELS, SIZE, SIZE),
        "msk": torch.randint(0, NUM_CLASSES, (2, SIZE, SIZE)),
    }


class _SyntheticSegData(Dataset):
    """In-memory dict batches, enough for a Lightning smoke fit."""

    def __init__(self, size: int = 2) -> None:
        torch.manual_seed(0)
        self.img = torch.randn(size, IN_CHANNELS, SIZE, SIZE)
        self.msk = torch.randint(0, NUM_CLASSES, (size, SIZE, SIZE))

    def __len__(self) -> int:
        return self.img.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {"img": self.img[index], "msk": self.msk[index]}


# --------------------------------------------------------------------------- #
# Construction                                                                #
# --------------------------------------------------------------------------- #


def test_default_loss_is_half_cross_entropy_half_dice():
    module = make_module()
    assert (module.ce_weight, module.dice_weight) == (0.5, 0.5)


def test_forward_returns_one_logit_map_per_class(batch):
    assert make_module()(batch["img"]).shape == (
        2,
        NUM_CLASSES,
        SIZE,
        SIZE,
    )


def test_classes_belong_to_num_classes_not_the_model_config():
    with pytest.raises(ValueError, match="num_classes"):
        make_module(model={"name": "unet", "encoder_weights": None, "classes": 3})


def test_rejecting_a_loss_with_no_active_term():
    with pytest.raises(ValueError, match="non-zero weight"):
        make_module(loss={"ce": 0.0, "dice": 0.0})


def test_class_weights_must_match_the_class_count():
    with pytest.raises(ValueError, match="2 entries but the model has 3"):
        make_module(class_weights=[1.0, 1.0])


# --------------------------------------------------------------------------- #
# Loss                                                                        #
# --------------------------------------------------------------------------- #


def test_cross_entropy_only_matches_torch(batch):
    module = make_module(loss={"ce": 1.0, "dice": 0.0})
    logits = module(batch["img"])
    expected = F.cross_entropy(logits, batch["msk"])
    assert torch.allclose(module._compute_loss(logits, batch["msk"]), expected)


def test_dice_only_matches_segmentation_models_pytorch(batch):
    module = make_module(loss={"ce": 0.0, "dice": 1.0})
    logits = module(batch["img"])
    expected = smp.losses.DiceLoss(mode="multiclass", ignore_index=None)(
        logits, batch["msk"]
    )
    assert torch.allclose(module._compute_loss(logits, batch["msk"]), expected)


def test_loss_terms_are_weighted(batch):
    module = make_module(loss={"ce": 0.25, "dice": 0.75})
    logits = module(batch["img"])
    ce = F.cross_entropy(logits, batch["msk"])
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=None)(
        logits, batch["msk"]
    )
    expected = 0.25 * ce + 0.75 * dice
    assert torch.allclose(module._compute_loss(logits, batch["msk"]), expected)


def test_class_weights_reach_the_cross_entropy_term(batch):
    module = make_module(loss={"ce": 1.0, "dice": 0.0}, class_weights=[1.0, 2.0, 3.0])
    logits = module(batch["img"])
    expected = F.cross_entropy(
        logits, batch["msk"], weight=torch.tensor([1.0, 2.0, 3.0])
    )
    assert torch.allclose(module._compute_loss(logits, batch["msk"]), expected)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #


def test_metrics_are_registered_per_stage():
    module = make_module()
    assert sorted(module.metrics.keys()) == ["test", "val"]
    for name in ("iou", "f1", "support"):
        assert module._metric("val", name) is not None


def test_epoch_metrics_leave_absent_classes_out_of_the_mean(batch):
    module = make_module()
    # Only two classes are present, so the third one is undefined, not zero.
    target = batch["msk"].clone()
    target[target == 2] = 0
    module._update_metrics("val", module(batch["img"]), target)

    values = module._epoch_metrics("val")

    assert math.isnan(values["val/iou/class_2"])
    defined = [values["val/iou/class_0"], values["val/iou/class_1"]]
    assert not any(math.isnan(value) for value in defined)
    assert values["val/mIoU"] == pytest.approx(sum(defined) / 2)


def test_epoch_metrics_cover_every_class_and_both_metrics(batch):
    module = make_module()
    module._update_metrics("test", module(batch["img"]), batch["msk"])

    values = module._epoch_metrics("test")

    for index in range(NUM_CLASSES):
        assert f"test/iou/class_{index}" in values
        assert f"test/f1/class_{index}" in values
    assert "test/mIoU" in values
    assert "test/mF1" in values


@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_epoch_metrics_are_nan_when_nothing_was_seen():
    module = make_module(scheduler={"t_max": 4})

    values = module._epoch_metrics("val")

    assert math.isnan(values["val/mIoU"])
    assert math.isnan(values["val/f1/class_0"])


def test_format_per_class_marks_absent_classes():
    rendered = _format_per_class(
        {"val/iou/class_0": 0.5, "val/iou/class_1": float("nan")}, "iou"
    )
    assert rendered == "0=0.500 1=n/a"


# --------------------------------------------------------------------------- #
# Optimizer and schedule                                                      #
# --------------------------------------------------------------------------- #


def configured(module: SegmentationModule) -> OptimizerLRSchedulerConfig:
    """Return the optimizer/scheduler config of ``module``, with its type pinned.

    ``LightningModule.configure_optimizers`` is typed as a wide union; the
    module always returns the single-optimizer mapping.
    """
    return cast(OptimizerLRSchedulerConfig, module.configure_optimizers())


def scheduler_of(config: OptimizerLRSchedulerConfig) -> LRScheduler:
    """Return the learning-rate schedule of an optimizer config."""
    schedule = cast(LRSchedulerConfigType, config["lr_scheduler"])
    return schedule["scheduler"]


def test_configure_optimizers_returns_adamw_and_an_epoch_schedule():
    config = configured(make_module(optimizer={"lr": 1e-3, "weight_decay": 1e-2}))
    optimizer = config["optimizer"]
    schedule = cast(LRSchedulerConfigType, config["lr_scheduler"])

    assert isinstance(optimizer, AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-2
    assert schedule["interval"] == "epoch"


def test_scheduler_matches_the_number_of_epochs():
    config = configured(make_module(scheduler={"warmup_epochs": 0, "t_max": 7}))

    scheduler = scheduler_of(config)

    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == 7


def test_warmup_precedes_the_cosine_decay():
    config = configured(
        make_module(optimizer={"lr": 1.0}, scheduler={"warmup_epochs": 2, "t_max": 10})
    )
    optimizer = config["optimizer"]
    scheduler = scheduler_of(config)

    assert isinstance(scheduler, SequentialLR)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)  # warmup start

    learning_rates = []
    for _ in range(3):
        optimizer.step()  # the schedule expects the optimizer step first
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] < 1.0  # still ramping up
    assert learning_rates[1] == pytest.approx(1.0)  # warmup done
    assert learning_rates[2] < 1.0  # cosine decay started


def test_unset_t_max_requires_an_attached_trainer():
    with pytest.raises(ValueError, match="not attached to a Trainer"):
        make_module(scheduler={"warmup_epochs": 0}).configure_optimizers()


# --------------------------------------------------------------------------- #
# End-to-end                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_fast_dev_run_logs_val_metrics():
    module = make_module()
    loader = DataLoader(_SyntheticSegData(), batch_size=2)
    trainer = Trainer(
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    assert "val/mIoU" in trainer.callback_metrics
    assert "train/loss" in trainer.callback_metrics
    # Metrics were reset after the validation epoch: nothing left to report.
    assert math.isnan(module._epoch_metrics("val")["val/mIoU"])


def test_module_is_an_nn_module_with_a_registered_network():
    module = make_module()
    assert isinstance(module, nn.Module)
    assert any(parameter.requires_grad for parameter in module.parameters())
