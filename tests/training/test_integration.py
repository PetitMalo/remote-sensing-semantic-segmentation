"""End-to-end check that the training module consumes real ``FitDataset`` batches.

The datamodule one-hot encodes its masks while the losses want class indices, so
the contract between ``rsss.data`` and ``rsss.training`` is only exercised when
the two run together. This drives a ``fast_dev_run`` fit over a synthetic FLAIR
tree (no download, no network) to catch a drift between them.
"""

from __future__ import annotations

import pytest
from pytorch_lightning import Trainer

from rsss.data.data_module import FlairDatamodule
from rsss.training.module import SegmentationModule

# FLAIR imagery: 5 bands (RGB + NIR + nDSM) and the 13-class nomenclature.
IN_CHANNELS = 5
NUM_CLASSES = 13
# Large enough for the UNet's five downsampling stages.
TILE_SIZE = 64


def make_module() -> SegmentationModule:
    """Return a small UNet module, with the datamodule's band/class counts."""
    return SegmentationModule(
        model={
            "name": "unet",
            "encoder_weights": None,
            "in_channels": IN_CHANNELS,
        },
        num_classes=NUM_CLASSES,
        scheduler={"warmup_epochs": 0, "t_max": 1},
    )


@pytest.fixture
def datamodule(tmp_path, write_flair_dataset) -> FlairDatamodule:
    """A datamodule over a one-tile-per-split synthetic FLAIR tree."""
    root = write_flair_dataset(tmp_path / "flair_1_toy_dataset", size=TILE_SIZE)
    return FlairDatamodule(data_dir=str(root), batch_size=1, drop_last=False)


@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_fast_dev_run_over_a_real_datamodule(datamodule):
    trainer = Trainer(
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(make_module(), datamodule=datamodule)

    assert "train/loss" in trainer.callback_metrics
    assert "val/mIoU" in trainer.callback_metrics
