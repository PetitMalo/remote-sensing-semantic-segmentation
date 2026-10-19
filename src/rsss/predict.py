"""Predict the test split from a training checkpoint and plot the results.

This is a local analysis step: it loads a checkpoint written by ``rsss.train``,
runs the model over the test tiles of a configured FLAIR#1 dataset, draws one
grid of image / ground truth / prediction, and, when given a ``metrics.csv``,
draws the training curves and the per-class IoU.

The plots need matplotlib, which ships in the ``dev`` dependency group: the
default local environment has it, the trimmed ``pando`` stack used on the
cluster does not. It is imported lazily so the rest of the module loads either
way -- the visualisation itself is meant to run locally, on exported results.

Run it with ``uv run python -m rsss.predict`` (``-h`` for the options).
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rsss.train import build_datamodule
from rsss.training.config import TrainConfig, load_train_config
from rsss.training.module import SegmentationModule, _to_class_indices

DEFAULT_TRAIN_CONFIG = "configs/train.yaml"
DEFAULT_DATA_CONFIG = "configs/data.yaml"
DEFAULT_OUTPUT_DIR = "outputs/predict"
DEFAULT_NUM_TILES = 4
# Columns of ``metrics.csv`` the results figure draws.
LOSS_COLUMNS = ("train/loss_epoch", "val/loss")
MIOU_COLUMNS = ("val/mIoU", "test/mIoU")
PER_CLASS_PREFIX = "val/iou/class_"


@dataclass(frozen=True)
class TilePrediction:
    """One test tile, its labels and the model's prediction.

    Parameters
    ----------
    image : np.ndarray
        Imagery as ``(height, width, bands)``, roughly in ``[0, 1]``.
    target : np.ndarray
        Ground-truth labels as ``(height, width)`` class indices.
    prediction : np.ndarray
        Predicted labels as ``(height, width)`` class indices.
    """

    image: np.ndarray
    target: np.ndarray
    prediction: np.ndarray


def default_device() -> str:
    """Return ``"cuda"`` when the machine has a GPU, ``"cpu"`` otherwise."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_module(
    checkpoint: str | Path,
    config: TrainConfig,
    device: str | None = None,
) -> SegmentationModule:
    """Rebuild the recipe's model and load ``checkpoint``'s weights into it.

    The encoder is built without pretrained weights: the checkpoint carries the
    trained ones, and predicting must not need the download a compute node
    cannot make. The architecture comes from the same recipe that trained it,
    so it matches the checkpoint's ``state_dict``.

    Parameters
    ----------
    checkpoint : str | Path
        A ``.ckpt`` written by ``rsss.train``.
    config : TrainConfig
        The recipe the checkpoint was trained with.
    device : str, optional
        Device to load onto; defaults to CUDA when available, CPU otherwise.

    Returns
    -------
    SegmentationModule
        The module in eval mode, on ``device``.
    """
    module = SegmentationModule(
        model={**dict(config.model), "encoder_weights": None},
        num_classes=config.num_classes,
        loss=config.loss,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        class_weights=config.class_weights,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    module.load_state_dict(state["state_dict"])
    module.eval()
    return module.to(device or default_device())


@torch.no_grad()
def predict_tiles(
    module: SegmentationModule,
    dataloader: DataLoader,
    num_tiles: int = DEFAULT_NUM_TILES,
) -> list[TilePrediction]:
    """Predict the first ``num_tiles`` tiles of ``dataloader``.

    Parameters
    ----------
    module : SegmentationModule
        A module in eval mode, on the device the batches are moved to.
    dataloader : DataLoader
        Yields the ``img``/``msk`` batches the training module expects.
    num_tiles : int
        Maximum number of tiles to keep.

    Returns
    -------
    list[TilePrediction]
        One entry per tile, in dataloader order.
    """
    device = next(module.parameters()).device
    tiles: list[TilePrediction] = []
    for batch in dataloader:
        images = batch["img"].to(device)
        predictions = module(images).argmax(dim=1)
        targets = _to_class_indices(batch["msk"])
        for index in range(images.shape[0]):
            tiles.append(
                TilePrediction(
                    image=images[index].permute(1, 2, 0).cpu().numpy(),
                    target=targets[index].cpu().numpy(),
                    prediction=predictions[index].cpu().numpy(),
                )
            )
            if len(tiles) >= num_tiles:
                return tiles
    return tiles


def _class_cmap(num_classes: int):
    """A discrete colormap with one distinct colour per class."""
    import matplotlib.pyplot as plt

    return plt.get_cmap("tab20", num_classes)


def plot_tiles(tiles: list[TilePrediction], path: str | Path, num_classes: int) -> Path:
    """Draw a grid of image / ground truth / prediction, one row per tile."""
    import matplotlib.pyplot as plt

    if not tiles:
        raise ValueError("No tiles to plot; predict at least one tile.")
    cmap = _class_cmap(num_classes)
    limit = num_classes - 0.5
    fig, axes = plt.subplots(len(tiles), 3, figsize=(9, 3 * len(tiles)), squeeze=False)
    for row, tile in enumerate(tiles):
        panels = (
            (np.clip(tile.image[:, :, :3], 0.0, 1.0), "image"),
            (tile.target, "ground truth"),
            (tile.prediction, "prediction"),
        )
        for column, (array, title) in enumerate(panels):
            axis = axes[row, column]
            if column == 0:
                axis.imshow(array)
            else:
                axis.imshow(
                    array, cmap=cmap, vmin=-0.5, vmax=limit, interpolation="nearest"
                )
            if row == 0:
                axis.set_title(title)
            axis.set_axis_off()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def read_metrics(metrics_csv: str | Path) -> list[dict[str, str]]:
    """Read one CSVLogger ``metrics.csv`` into its raw rows."""
    with Path(metrics_csv).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _find_metrics_csvs(metrics: Path) -> list[Path]:
    """Return ``metrics`` when it is a file, else every one below it."""
    return [metrics] if metrics.is_file() else sorted(metrics.rglob("metrics.csv"))


def _series(rows: list[dict[str, str]], column: str) -> tuple[list[int], list[float]]:
    """Return the (epoch, value) points of ``column``, skipping blank and NaN."""
    epochs: list[int] = []
    values: list[float] = []
    for row in rows:
        raw, epoch = row.get(column), row.get("epoch")
        if raw in (None, "") or epoch in (None, ""):
            continue
        value = float(raw)
        if math.isnan(value):  # a class absent from the split
            continue
        epochs.append(int(float(epoch)))
        values.append(value)
    return epochs, values


def _per_class_columns(rows: list[dict[str, str]]) -> list[str]:
    """Return the per-class IoU columns present, ordered by class index."""
    keys = {key for row in rows for key in row if key.startswith(PER_CLASS_PREFIX)}
    return sorted(keys, key=lambda key: int(key.rsplit("_", 1)[1]))


def _last_per_class(rows: list[dict[str, str]], columns: list[str]) -> list[float]:
    """Return the most recent row that filled the per-class columns."""
    for row in reversed(rows):
        if any(row.get(column) not in (None, "") for column in columns):
            return [
                float(row[column])
                if row.get(column) not in (None, "")
                else float("nan")
                for column in columns
            ]
    return [float("nan")] * len(columns)


def plot_results(metrics: str | Path, path: str | Path) -> Path:
    """Draw loss, mIoU and per-class IoU curves from a CSVLogger ``metrics.csv``.

    Parameters
    ----------
    metrics : str | Path
        A ``metrics.csv``, or a directory holding one or more (the ``csv/``
        directory of a run, whose ``version_*`` folders each hold one).
    path : str | Path
        Where to write the figure.
    """
    import matplotlib.pyplot as plt

    metrics = Path(metrics)
    csvs = _find_metrics_csvs(metrics)
    if not csvs:
        raise FileNotFoundError(f"No metrics.csv under {metrics}.")
    rows = [row for csv_path in csvs for row in read_metrics(csv_path)]

    per_class_columns = _per_class_columns(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for column, label in ((LOSS_COLUMNS[0], "train"), (LOSS_COLUMNS[1], "val")):
        epochs, values = _series(rows, column)
        axes[0].plot(epochs, values, marker="o", label=label)
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    for column in MIOU_COLUMNS:
        epochs, values = _series(rows, column)
        if values:
            axes[1].plot(epochs, values, marker="o", label=column.split("/")[0])
    axes[1].set_title("mIoU")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    if per_class_columns:
        values = _last_per_class(rows, per_class_columns)
        positions = range(len(per_class_columns))
        axes[2].bar(positions, values)
        axes[2].set_xticks(list(positions))
        axes[2].set_xticklabels(
            [column.rsplit("_", 1)[1] for column in per_class_columns]
        )
    axes[2].set_title("per-class IoU")
    axes[2].set_xlabel("class")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of the prediction entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="A .ckpt written by rsss.train.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_TRAIN_CONFIG,
        help="Training recipe the checkpoint was trained with (default: %(default)s).",
    )
    parser.add_argument(
        "--data-config",
        default=DEFAULT_DATA_CONFIG,
        help="Data source (default: %(default)s).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the data config's data_dir.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Where the figures are written (default: %(default)s).",
    )
    parser.add_argument(
        "--num-tiles",
        type=int,
        default=DEFAULT_NUM_TILES,
        help="How many test tiles to draw (default: %(default)s).",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="A metrics.csv or the run's csv directory, to draw the results figure.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device to predict on (default: CUDA when available).",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    """Predict some test tiles, draw them, and return the output directory."""
    config = load_train_config(args.config)
    module = load_module(args.checkpoint, config, args.device)
    datamodule = build_datamodule(config, args)
    datamodule.setup("test")
    if datamodule.test_dataset is None:
        raise RuntimeError("The datamodule has no test split after setup('test').")
    # One tile per batch and no workers: visualising a few tiles does not need
    # the recipe's throughput settings.
    loader = DataLoader(
        datamodule.test_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    tiles = predict_tiles(module, loader, args.num_tiles)

    output_dir = Path(args.output_dir)
    plot_tiles(tiles, output_dir / "predictions.png", config.num_classes)
    if args.metrics:
        plot_results(args.metrics, output_dir / "results.png")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, predict and plot, and report success."""
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
