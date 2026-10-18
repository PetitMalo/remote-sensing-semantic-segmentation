"""Unit tests for the training entry point.

The end-to-end cases build a synthetic FLAIR tree and small configs, so a whole
fit runs on CPU inside the sandbox: no download, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rsss.data.download import TOY_DATASET
from rsss.train import (
    build_arg_parser,
    build_datamodule,
    build_module,
    build_trainer,
    main,
    run,
)
from rsss.training.config import TrainConfig

# Large enough for the UNet's five downsampling stages.
TILE_SIZE = 64
TINY_MODEL = {"name": "unet", "encoder_weights": None, "in_channels": 5}


def parse(argv: list[str]):
    """Parse ``argv`` with the entry point's own parser."""
    return build_arg_parser().parse_args(argv)


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def test_parser_defaults_to_the_shipped_configs():
    args = parse([])

    assert args.config == "configs/train.yaml"
    assert args.data_config == "configs/data.yaml"
    assert args.output_dir == "outputs"
    assert args.data_dir is None
    assert args.max_epochs is None
    assert args.limit_train_batches is None
    assert args.fast_dev_run is False
    assert args.seed is None


def test_parser_reads_every_override():
    args = parse(
        [
            "--config",
            "recipe.yaml",
            "--data-config",
            "source.yaml",
            "--data-dir",
            "/scratch/me/data",
            "--output-dir",
            "/scratch/me/runs",
            "--max-epochs",
            "2",
            "--limit-train-batches",
            "5",
            "--fast-dev-run",
            "--seed",
            "7",
        ]
    )

    assert args.config == "recipe.yaml"
    assert args.data_config == "source.yaml"
    assert args.data_dir == "/scratch/me/data"
    assert args.output_dir == "/scratch/me/runs"
    assert args.max_epochs == 2
    assert args.limit_train_batches == 5
    assert args.fast_dev_run is True
    assert args.seed == 7


# --------------------------------------------------------------------------- #
# Wiring                                                                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def data_config(tmp_path) -> Path:
    """A data config whose data_dir is ``data/``, relative to the repo root."""
    path = tmp_path / "data.yaml"
    path.write_text("data_dir: data\ndataset: flair_1_toy\n")
    return path


def test_data_dir_flag_replaces_the_data_config_data_dir(tmp_path, data_config):
    args = parse(
        ["--data-config", str(data_config), "--data-dir", str(tmp_path / "scratch")]
    )
    config = TrainConfig(model=TINY_MODEL, num_classes=3)

    datamodule = build_datamodule(config, args)

    assert datamodule.data_dir == str(tmp_path / "scratch" / TOY_DATASET.name)


def test_without_the_flag_the_data_config_data_dir_is_kept(data_config):
    config = TrainConfig(model=TINY_MODEL, num_classes=3)

    datamodule = build_datamodule(config, parse(["--data-config", str(data_config)]))

    assert datamodule.data_dir == str(Path("data") / TOY_DATASET.name)


def test_build_module_forwards_the_recipe():
    config = TrainConfig(
        model=TINY_MODEL,
        num_classes=3,
        loss={"ce": 1.0, "dice": 0.0},
        optimizer={"lr": 1e-3},
        scheduler={"warmup_epochs": 0, "t_max": 2},
        class_weights=[1.0, 2.0, 3.0],
    )

    module = build_module(config)

    assert module.num_classes == 3
    assert (module.ce_weight, module.dice_weight) == (1.0, 0.0)
    class_weights = module.ce_loss.weight
    assert class_weights is not None
    assert class_weights.tolist() == [1.0, 2.0, 3.0]
    assert module.optimizer_cfg["lr"] == 1e-3


def test_trainer_honours_the_recipe_and_the_flags(tmp_path):
    config = TrainConfig(model=TINY_MODEL, num_classes=3, trainer={"max_epochs": 10})
    args = parse(
        [
            "--output-dir",
            str(tmp_path),
            "--max-epochs",
            "2",
            "--limit-train-batches",
            "5",
        ]
    )

    trainer = build_trainer(config, args)

    assert trainer.max_epochs == 2
    assert trainer.limit_train_batches == 5


# --------------------------------------------------------------------------- #
# End to end                                                                  #
# --------------------------------------------------------------------------- #


@pytest.fixture
def smoke_argv(tmp_path, write_flair_dataset) -> list[str]:
    """Arguments for a one-epoch CPU fit over a synthetic FLAIR tree."""
    write_flair_dataset(tmp_path / "data" / TOY_DATASET.name, size=TILE_SIZE)

    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        yaml.safe_dump(
            {
                "model": TINY_MODEL,
                "num_classes": 13,
                "scheduler": {"warmup_epochs": 0},
                "trainer": {
                    "max_epochs": 1,
                    "accelerator": "cpu",
                    "precision": 32,
                    "log_every_n_steps": 1,
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                },
                "datamodule": {"batch_size": 1, "num_workers": 0, "drop_last": False},
            }
        )
    )
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "dataset": "flair_1_toy",
                "download": False,
            }
        )
    )
    return [
        "--config",
        str(train_config),
        "--data-config",
        str(data_config),
        "--output-dir",
        str(tmp_path / "outputs"),
    ]


@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_run_fits_and_writes_its_artefacts(smoke_argv, tmp_path):
    trainer = run(parse(smoke_argv))

    assert "train/loss" in trainer.callback_metrics
    assert "val/mIoU" in trainer.callback_metrics
    assert list((tmp_path / "outputs" / "checkpoints").glob("*.ckpt"))
    assert list((tmp_path / "outputs" / "csv").rglob("metrics.csv"))


@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_main_reports_success(smoke_argv):
    assert main(smoke_argv) == 0
