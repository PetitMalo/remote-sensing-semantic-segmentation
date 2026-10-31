"""Unit tests for the prediction and plotting entry point.

The figures need matplotlib, which the ``dev`` group provides; the backend is
forced to Agg so nothing needs a display. Models are tiny (``encoder_weights``
None) and the datasets are the shared synthetic FLAIR#1 trees, so the whole file
runs on CPU without a download.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
import yaml

from rsss.data.data_module import FlairDatamodule
from rsss.predict import (
    TilePrediction,
    build_arg_parser,
    load_module,
    plot_results,
    plot_tiles,
    predict_tiles,
    read_metrics,
    run,
)
from rsss.training.config import TrainConfig
from rsss.training.module import SegmentationModule

# FLAIR imagery: 5 bands and the 13-class nomenclature.
IN_CHANNELS = 5
NUM_CLASSES = 13
# Large enough for the UNet's five downsampling stages.
TILE_SIZE = 64

MODEL = {"name": "unet", "encoder_weights": None, "in_channels": IN_CHANNELS}

# A DINOv3 ViT-S/16: the cheap test stand-in for the SAT-493M ViT-L, as in
# ``tests/models/test_dinov3.py``. ``pretrained: True`` on purpose: load_module
# must override it, predicting never downloads the 1.2 GB backbone.
DINO_MODEL = {
    "name": "dinov3_linear_probe",
    "model_name": "vit_small_patch16_dinov3.lvd1689m",
    "pretrained": True,
    "in_channels": IN_CHANNELS,
}
SIZE = 64


def parse(argv: list[str]):
    """Parse ``argv`` with the entry point's own parser."""
    return build_arg_parser().parse_args(argv)


def tiny_module() -> SegmentationModule:
    """Return a small UNet module, matching :data:`MODEL`."""
    return SegmentationModule(
        model=MODEL,
        num_classes=NUM_CLASSES,
        scheduler={"warmup_epochs": 0, "t_max": 1},
    )


def tiny_config() -> TrainConfig:
    """Return a recipe matching :func:`tiny_module`."""
    return TrainConfig(
        model=MODEL,
        num_classes=NUM_CLASSES,
        scheduler={"warmup_epochs": 0, "t_max": 1},
    )


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def test_parser_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        parse([])


def test_parser_defaults_to_the_shipped_configs():
    args = parse(["--checkpoint", "model.ckpt"])

    assert args.config == "configs/train.yaml"
    assert args.data_config == "configs/data.yaml"
    assert args.output_dir == "outputs/predict"
    assert args.num_tiles == 4
    assert args.metrics is None
    assert args.device is None


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #


def test_load_module_restores_the_checkpointed_weights(tmp_path):
    module = tiny_module()
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": module.state_dict()}, checkpoint)
    module.eval()

    loaded = load_module(checkpoint, tiny_config(), device="cpu")

    sample = torch.randn(1, IN_CHANNELS, TILE_SIZE, TILE_SIZE)
    with torch.no_grad():
        assert torch.allclose(module(sample), loaded(sample))


def test_load_module_builds_without_pretrained_weights(tmp_path, monkeypatch):
    """The load must not fetch ImageNet weights the checkpoint already holds."""
    real = tiny_module()
    built: list[dict] = []

    def _build(cfg):
        built.append(dict(cfg))
        return real.net

    monkeypatch.setattr("rsss.training.module.build_model", _build)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": real.state_dict()}, checkpoint)

    load_module(checkpoint, tiny_config(), device="cpu")

    assert built[0]["encoder_weights"] is None


def test_load_module_builds_the_probe_without_pretrained_weights(tmp_path):
    """The DINOv3 builder takes ``pretrained``, not ``encoder_weights``.

    A recipe shipped with the download enabled (the training one) turns into
    an offline build, and the checkpoint's weights end up in the network.
    """
    module = SegmentationModule(
        model=DINO_MODEL,
        num_classes=NUM_CLASSES,
        scheduler={"warmup_epochs": 0, "t_max": 1},
    )
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": module.state_dict()}, checkpoint)

    loaded = load_module(
        checkpoint,
        TrainConfig(
            model=DINO_MODEL,
            num_classes=NUM_CLASSES,
            scheduler={"warmup_epochs": 0, "t_max": 1},
        ),
        device="cpu",
    )

    sample = torch.randn(1, IN_CHANNELS, SIZE, SIZE)
    with torch.no_grad():
        assert torch.allclose(module(sample), loaded(sample))


# --------------------------------------------------------------------------- #
# Prediction                                                                  #
# --------------------------------------------------------------------------- #


def test_predict_tiles_returns_one_entry_per_tile(tmp_path, write_flair_dataset):
    root = write_flair_dataset(tmp_path / "flair_1_toy_dataset", size=TILE_SIZE)
    datamodule = FlairDatamodule(data_dir=str(root), batch_size=1, num_workers=0)
    datamodule.setup("test")

    tiles = predict_tiles(tiny_module(), datamodule.test_dataloader(), num_tiles=1)

    assert len(tiles) == 1
    assert tiles[0].image.shape == (TILE_SIZE, TILE_SIZE, IN_CHANNELS)
    assert tiles[0].target.shape == (TILE_SIZE, TILE_SIZE)
    assert tiles[0].prediction.shape == (TILE_SIZE, TILE_SIZE)


def test_predict_tiles_stops_at_the_requested_count(
    tmp_path, write_flair_dataset, write_flair_split
):
    root = write_flair_dataset(tmp_path / "flair_1_toy_dataset", size=TILE_SIZE)
    write_flair_split(
        root,
        "test",
        [("D012_2019", "Z3_UU", "000001"), ("D012_2019", "Z3_UU", "000002")],
        size=TILE_SIZE,
    )
    datamodule = FlairDatamodule(data_dir=str(root), batch_size=2, num_workers=0)
    datamodule.setup("test")

    tiles = predict_tiles(tiny_module(), datamodule.test_dataloader(), num_tiles=1)

    assert len(tiles) == 1


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #


def test_plot_tiles_writes_a_png(tmp_path):
    tile = TilePrediction(
        image=np.zeros((TILE_SIZE, TILE_SIZE, IN_CHANNELS), dtype=np.float32),
        target=np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int64),
        prediction=np.ones((TILE_SIZE, TILE_SIZE), dtype=np.int64),
    )

    path = plot_tiles([tile], tmp_path / "predictions.png", NUM_CLASSES)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_tiles_rejects_no_tiles(tmp_path):
    with pytest.raises(ValueError, match="No tiles to plot"):
        plot_tiles([], tmp_path / "predictions.png", NUM_CLASSES)


def test_read_metrics_reads_rows(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("epoch,step,val/loss\n0,1,1.5\n")

    assert read_metrics(metrics) == [{"epoch": "0", "step": "1", "val/loss": "1.5"}]


def test_plot_results_writes_a_png(tmp_path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "epoch,step,train/loss_epoch,val/loss,val/mIoU,val/iou/class_0,val/iou/class_1\n"
        "0,10,1.5,1.2,0.3,0.5,0.1\n"
        "1,20,1.0,0.9,0.4,0.6,0.2\n"
    )

    path = plot_results(metrics, tmp_path / "results.png")

    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_results_accepts_the_csv_directory(tmp_path):
    version = tmp_path / "csv" / "version_0"
    version.mkdir(parents=True)
    (version / "metrics.csv").write_text("epoch,val/mIoU\n0,0.3\n")

    path = plot_results(tmp_path / "csv", tmp_path / "results.png")

    assert path.is_file()


def test_plot_results_without_metrics_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No metrics.csv"):
        plot_results(tmp_path / "absent", tmp_path / "results.png")


# --------------------------------------------------------------------------- #
# End to end                                                                  #
# --------------------------------------------------------------------------- #


def test_run_writes_predictions_from_a_checkpoint(tmp_path, write_flair_dataset):
    data_root = tmp_path / "data"
    write_flair_dataset(data_root / "flair_1_toy_dataset", size=TILE_SIZE)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": tiny_module().state_dict()}, checkpoint)

    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        yaml.safe_dump(
            {
                "model": MODEL,
                "num_classes": NUM_CLASSES,
                "scheduler": {"warmup_epochs": 0, "t_max": 1},
                "datamodule": {"batch_size": 1, "num_workers": 0, "drop_last": False},
            }
        )
    )
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(data_root),
                "dataset": "flair_1_toy",
                "download": False,
            }
        )
    )

    output = run(
        parse(
            [
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(train_config),
                "--data-config",
                str(data_config),
                "--output-dir",
                str(tmp_path / "out"),
                "--num-tiles",
                "1",
            ]
        )
    )

    assert (output / "predictions.png").is_file()


def test_run_draws_the_results_when_given_metrics(tmp_path, write_flair_dataset):
    data_root = tmp_path / "data"
    write_flair_dataset(data_root / "flair_1_toy_dataset", size=TILE_SIZE)
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": tiny_module().state_dict()}, checkpoint)
    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        yaml.safe_dump(
            {
                "model": MODEL,
                "num_classes": NUM_CLASSES,
                "scheduler": {"warmup_epochs": 0, "t_max": 1},
                "datamodule": {"batch_size": 1, "num_workers": 0, "drop_last": False},
            }
        )
    )
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(data_root),
                "dataset": "flair_1_toy",
                "download": False,
            }
        )
    )
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("epoch,val/mIoU\n0,0.3\n")

    output = run(
        parse(
            [
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(train_config),
                "--data-config",
                str(data_config),
                "--output-dir",
                str(tmp_path / "out"),
                "--num-tiles",
                "1",
                "--metrics",
                str(metrics),
            ]
        )
    )

    assert (output / "predictions.png").is_file()
    assert (output / "results.png").is_file()
