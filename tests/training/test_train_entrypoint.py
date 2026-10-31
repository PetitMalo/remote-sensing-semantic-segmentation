"""Unit tests for the training entry point.

The end-to-end cases build a synthetic FLAIR tree and small configs, so a whole
fit runs on CPU inside the sandbox: no download, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from rsss.data.download import TOY_DATASET
from rsss.models import register_model
from rsss.train import (
    CHECKPOINT_MONITOR,
    build_arg_parser,
    build_checkpoint_callbacks,
    build_datamodule,
    build_module,
    build_trainer,
    main,
    resolve_resume_checkpoint,
    run,
    seed_everything,
)
from rsss.training.config import TrainConfig

TILE_SIZE = 64


# These tests check the entry point's wiring, not an architecture: a tiny model
# keeps each fit and each of its two checkpoints small, where the real U-Net's
# are ~300 MB apiece.
@register_model("tiny_seg")
def _build_tiny_seg(in_channels: int = 5, classes: int = 13) -> nn.Module:
    """Return a two-convolution segmentation model for the smoke fits."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, classes, kernel_size=1),
    )


TINY_MODEL = {"name": "tiny_seg", "in_channels": 5}


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
    assert args.skip_test is False
    assert args.seed is None
    assert args.resume is None


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
            "--skip-test",
            "--seed",
            "7",
            "--resume",
            "last",
        ]
    )

    assert args.config == "recipe.yaml"
    assert args.data_config == "source.yaml"
    assert args.data_dir == "/scratch/me/data"
    assert args.output_dir == "/scratch/me/runs"
    assert args.max_epochs == 2
    assert args.limit_train_batches == 5
    assert args.fast_dev_run is True
    assert args.skip_test is True
    assert args.seed == 7
    assert args.resume == "last"


def test_resume_without_a_value_means_last(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"state")

    resolved = resolve_resume_checkpoint(
        parse(["--output-dir", str(tmp_path), "--resume"])
    )

    assert resolved == checkpoint


def test_resume_accepts_an_explicit_checkpoint(tmp_path):
    checkpoint = tmp_path / "epoch-3.ckpt"
    checkpoint.write_bytes(b"state")

    resolved = resolve_resume_checkpoint(parse(["--resume", str(checkpoint)]))

    assert resolved == checkpoint


def test_resume_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="No checkpoint to resume from"):
        resolve_resume_checkpoint(
            parse(["--output-dir", str(tmp_path), "--resume", "last"])
        )


def test_fast_dev_run_and_resume_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(parse(["--fast-dev-run", "--resume"]))


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


def test_build_datamodule_forwards_num_classes(data_config):
    config = TrainConfig(model=TINY_MODEL, num_classes=3)

    datamodule = build_datamodule(config, parse(["--data-config", str(data_config)]))

    assert datamodule.num_classes == 3


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


def test_seed_everything_forwards_to_lightning(monkeypatch):
    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        "rsss.train.pl.seed_everything",
        lambda seed, workers: calls.append((seed, workers)),
    )

    seed_everything(7)
    seed_everything(None)

    assert calls == [(7, True)]


def test_run_seeds_before_building_the_module(monkeypatch):
    calls: list[str] = []

    class _StubTrainer:
        def fit(self, *args, **kwargs):
            return None

        def test(self, *args, **kwargs):
            return None

    monkeypatch.setattr(
        "rsss.train.pl.seed_everything",
        lambda seed, workers: calls.append("seed"),
    )
    monkeypatch.setattr(
        "rsss.train.load_train_config",
        lambda path: TrainConfig(model=TINY_MODEL, num_classes=3),
    )
    monkeypatch.setattr("rsss.train.build_datamodule", lambda config, args: None)
    monkeypatch.setattr(
        "rsss.train.build_module", lambda config: calls.append("module")
    )
    monkeypatch.setattr("rsss.train.build_trainer", lambda config, args: _StubTrainer())

    run(parse(["--seed", "3", "--skip-test"]))

    assert calls == ["seed", "module"]


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


def test_checkpoint_callbacks_keep_only_best_and_last(tmp_path):
    best, last = build_checkpoint_callbacks(tmp_path)
    assert isinstance(best, ModelCheckpoint)
    assert isinstance(last, ModelCheckpoint)

    # The best checkpoint is chosen on the metric the module logs at validation
    # end, so the score and the weights belong to the same epoch.
    assert CHECKPOINT_MONITOR == "val/mIoU"
    assert best.monitor == CHECKPOINT_MONITOR
    assert best.mode == "max"
    assert best.save_top_k == 1
    # A separate callback, because save_last alone would duplicate the best
    # checkpoint instead of writing the final epoch's weights.
    assert last.save_top_k == 0
    assert last.save_last is True

    assert best.dirpath == last.dirpath
    assert Path(str(best.dirpath)) == tmp_path / "checkpoints"
    assert best._save_on_train_epoch_end is False
    assert last._save_on_train_epoch_end is False


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
    run(parse(smoke_argv))

    # ``callback_metrics`` holds only the test pass once the fit is over, so the
    # train/val metrics are asserted through the CSVLogger artefacts instead.
    checkpoints = sorted(
        path.name for path in (tmp_path / "outputs" / "checkpoints").glob("*.ckpt")
    )
    assert "last.ckpt" in checkpoints
    assert len(checkpoints) == 2
    csv_files = list((tmp_path / "outputs" / "csv").rglob("metrics.csv"))
    assert csv_files
    logged = "\n".join(path.read_text() for path in csv_files)
    assert "train/loss" in logged
    assert "val/mIoU" in logged


@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_more_epochs_do_not_accumulate_checkpoints(smoke_argv, tmp_path):
    trainer = run(parse([*smoke_argv, "--max-epochs", "2"]))

    # Beyond the best and the last checkpoint, later epochs must not leave a
    # file behind: they are what fills the run directory and the bucket.
    checkpoints = list((tmp_path / "outputs" / "checkpoints").glob("*.ckpt"))
    assert len(checkpoints) == 2

    # ``last.ckpt`` is the final epoch's state, not a copy of the best one.
    last = torch.load(
        tmp_path / "outputs" / "checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert last["global_step"] == trainer.global_step


@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_run_scores_the_test_split_after_the_fit(smoke_argv):
    trainer = run(parse(smoke_argv))

    assert "test/mIoU" in trainer.callback_metrics


@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_skip_test_leaves_the_test_split_unscored(smoke_argv):
    trainer = run(parse([*smoke_argv, "--skip-test"]))

    assert "test/mIoU" not in trainer.callback_metrics


@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_resume_continues_the_fit_from_last(smoke_argv):
    first = run(parse([*smoke_argv, "--skip-test"]))

    resumed = run(
        parse([*smoke_argv, "--max-epochs", "2", "--resume", "last", "--skip-test"])
    )

    # Lightning increments current_epoch after the last epoch, so a run resumed
    # from epoch 1 and stopped at max_epochs=2 reports 2 here.
    assert resumed.current_epoch == 2
    assert resumed.global_step > first.global_step


@pytest.mark.filterwarnings("ignore:.*does not have many workers")
@pytest.mark.filterwarnings("ignore:.*compute.*called before the.*update")
def test_main_reports_success(smoke_argv):
    assert main(smoke_argv) == 0
