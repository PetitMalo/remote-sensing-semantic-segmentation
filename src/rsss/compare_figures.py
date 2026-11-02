"""Draw every compared run on the same test tiles.

``rsss.compare`` ranks runs on their best-validation checkpoint, but a table of
mIoU values says nothing about *where* the models disagree. This module loads
the same checkpoint for each run, predicts a fixed set of test tiles, and draws
them as one grid: raw RGB, ground truth, then one prediction column per run.

The recipe a run was trained with is not stored in its ``hparams.yaml`` (only
the model section is), so the data pipeline -- notably the per-backbone
normalisation -- is recovered by mapping the recipe's model name to the
``configs/train*.yaml`` that builds it. The reference tiles are read once from
the test split's path CSV, so every model is scored on the same pixels.

The figures need matplotlib, which ships in the ``dev`` group: this step runs
locally, on exported runs, not on the cluster. Importing the module stays cheap;
matplotlib is imported inside the plotting function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

from rsss.data.config import DataConfig, load_data_config
from rsss.data.data_loader import FitDataset
from rsss.data.splits import build_flair_splits
from rsss.predict import load_module, predict_tiles
from rsss.train import datamodule_from_config
from rsss.training.config import TrainConfig, load_train_config
from rsss.viz.classes import CLASS_COLORS, class_label

if TYPE_CHECKING:
    from rsss.compare import RunComparison

DEFAULT_CONFIGS_DIR = "configs"
DEFAULT_DATA_CONFIG = "configs/data_subset.yaml"
DEFAULT_NUM_TILES = 3
FIGURE_NAME = "inference.png"


@dataclass(frozen=True)
class ReferenceTile:
    """One test tile every run is asked to predict.

    Parameters
    ----------
    image : np.ndarray
        Raw imagery as ``(height, width, 3)``, band order Red/Green/Blue.
    target : np.ndarray
        Ground-truth labels as ``(height, width)`` class indices.
    name : str
        Tile name, e.g. ``IMG_062613``, used as the row label.
    """

    image: np.ndarray
    target: np.ndarray
    name: str


@dataclass(frozen=True)
class InferenceFigure:
    """One written inference grid.

    Parameters
    ----------
    path : Path
        PNG file holding the grid.
    runs : tuple[str, ...]
        Runs that have a prediction column, in column order.
    num_tiles : int
        Number of test tiles drawn, one per row.
    """

    path: Path
    runs: tuple[str, ...]
    num_tiles: int


def discover_recipes(
    configs_dir: str | Path = DEFAULT_CONFIGS_DIR,
) -> dict[str, TrainConfig]:
    """Map every recipe's model name to the config that builds it.

    A run records only its model name, so this is what recovers the recipe's
    ``datamodule`` section, where the backbone-specific normalisation lives.
    The first recipe wins when two configs build the same model, since a run's
    weights cannot say which of them trained it.
    """
    recipes: dict[str, TrainConfig] = {}
    for path in sorted(Path(configs_dir).glob("train*.yaml")):
        config = load_train_config(path)
        name = str(config.model.get("name", ""))
        if not name:
            continue
        if name in recipes:
            logger.warning("{} builds {} too; keeping the earlier recipe.", path, name)
            continue
        recipes[name] = config
    return recipes


def stretch_rgb(
    image: np.ndarray, lower: float = 2.0, upper: float = 98.0
) -> np.ndarray:
    """Stretch a raw RGB tile to ``[0, 1]`` on its per-band percentiles.

    The percentiles clip the sensor's bright and dark outliers, which otherwise
    dominate a plain ``/255`` and hide most of the tile. A constant band has no
    span to stretch, so it falls back to the plain scale instead of dividing by
    zero.

    Parameters
    ----------
    image : np.ndarray
        Raw RGB tile as ``(height, width, 3)``.
    lower, upper : float
        Percentiles to map to 0 and 1.

    Returns
    -------
    np.ndarray
        The stretched tile as floats in ``[0, 1]``.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a (height, width, bands) tile, got {array.shape}.")
    low = np.percentile(array, lower, axis=(0, 1), keepdims=True)
    high = np.percentile(array, upper, axis=(0, 1), keepdims=True)
    span = high - low
    stretched = (array - low) / np.where(span > 0, span, 1.0)
    plain = array / 255.0
    return np.clip(np.where(span > 0, stretched, plain), 0.0, 1.0)


def read_reference_tiles(
    data_config: DataConfig,
    num_classes: int,
    num_tiles: int = DEFAULT_NUM_TILES,
) -> list[ReferenceTile]:
    """Read the first test tiles as raw RGB and label indices.

    ``FitDataset`` owns the label encoding (MSK values 1..13 become indices
    0..12, and an out-of-range value becomes class 0), so reusing its readers
    keeps the ground truth identical to what every model was scored against.

    Raises
    ------
    ValueError
        If ``num_tiles`` is not positive or the test split holds no tiles.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}.")
    files = build_flair_splits(data_config.dataset_dir)["test"]
    dataset = FitDataset(
        dict_files=files,
        channels=[1, 2, 3],
        num_classes=num_classes,
        start_one=True,
    )
    if not len(dataset):
        raise ValueError(f"The test split at {data_config.dataset_dir} holds no tiles.")
    tiles: list[ReferenceTile] = []
    for image_path, mask_path in zip(
        dataset.list_imgs[:num_tiles], dataset.list_msks[:num_tiles], strict=True
    ):
        rgb = dataset.read_img(str(image_path))
        labels = dataset.read_msk(str(mask_path), start_one=True).argmax(axis=0)
        tiles.append(
            ReferenceTile(
                image=np.transpose(rgb, (1, 2, 0)),
                target=labels,
                name=Path(str(image_path)).stem,
            )
        )
    return tiles


def predict_run(
    row: RunComparison,
    config: TrainConfig,
    data_config: DataConfig,
    tile_names: Sequence[str],
    device: str | None = None,
) -> list[np.ndarray]:
    """Predict the reference tiles from one run's best checkpoint.

    ``tile_names`` guards the figure's premise: a run whose test split is
    ordered differently would otherwise be drawn beside the wrong ground truth.
    """
    if row.checkpoint is None:
        raise FileNotFoundError(f"{row.run} has no downloaded checkpoint.")
    module = load_module(row.checkpoint, config, device)
    datamodule = datamodule_from_config(config, data_config)
    datamodule.setup("test")
    dataset = datamodule.test_dataset
    if dataset is None:
        raise RuntimeError(f"{row.run}: the recipe has no test dataset.")
    found = [Path(str(path)).stem for path in dataset.list_imgs[: len(tile_names)]]
    if found != list(tile_names):
        raise ValueError(
            f"{row.run}: test tiles {found} differ from the reference {tile_names}."
        )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    predictions = predict_tiles(module, loader, len(tile_names))
    if len(predictions) != len(tile_names):
        raise ValueError(
            f"{row.run}: predicted {len(predictions)} tiles, expected "
            f"{len(tile_names)}."
        )
    return [tile.prediction for tile in predictions]


def plot_inference_grid(
    tiles: Sequence[ReferenceTile],
    predictions: Mapping[str, Sequence[np.ndarray]],
    path: str | Path,
) -> Path:
    """Draw tiles as rows and the runs as prediction columns, with a legend."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    if not tiles:
        raise ValueError("No tiles to plot; read at least one test tile.")
    if not predictions:
        raise ValueError("No predictions to plot; pass at least one run.")
    runs = list(predictions)
    for run in runs:
        if len(predictions[run]) != len(tiles):
            raise ValueError(
                f"{run}: {len(predictions[run])} predictions for {len(tiles)} tiles."
            )

    cmap = ListedColormap(CLASS_COLORS)
    columns = ("RGB", "ground truth", *runs)
    fig, axes = plt.subplots(
        len(tiles),
        len(columns),
        figsize=(2.3 * len(columns), 2.3 * len(tiles)),
        squeeze=False,
    )
    for row_index, tile in enumerate(tiles):
        panels = [
            stretch_rgb(tile.image),
            np.asarray(tile.target),
            *(np.asarray(predictions[run][row_index]) for run in runs),
        ]
        for column_index, panel in enumerate(panels):
            axis = axes[row_index, column_index]
            if column_index == 0:
                axis.imshow(panel)
            else:
                axis.imshow(
                    panel,
                    cmap=cmap,
                    vmin=-0.5,
                    vmax=len(CLASS_COLORS) - 0.5,
                    interpolation="nearest",
                )
            if row_index == 0:
                axis.set_title(columns[column_index], fontsize=9)
            if column_index == 0:
                axis.set_ylabel(tile.name, fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    patches = [
        Patch(facecolor=color, label=class_label(index))
        for index, color in enumerate(CLASS_COLORS)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncols=7,
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate(
    rows: Sequence[RunComparison],
    *,
    configs_dir: str | Path = DEFAULT_CONFIGS_DIR,
    data_config_path: str | Path = DEFAULT_DATA_CONFIG,
    data_dir: str | Path | None = None,
    num_tiles: int = DEFAULT_NUM_TILES,
    device: str | None = None,
    output_dir: str | Path = ".",
) -> InferenceFigure | None:
    """Predict every run's best checkpoint on the same tiles and draw the grid.

    Runs without a downloaded checkpoint or without a recipe matching their
    model name are skipped with a warning, so a partially pulled run root still
    produces a figure from the runs that can be read.
    """
    recipes = discover_recipes(configs_dir)
    data_config = load_data_config(data_config_path)
    if data_dir is not None:
        data_config = replace(data_config, data_dir=Path(data_dir))
    usable = [
        row for row in rows if row.checkpoint is not None and row.recipe in recipes
    ]
    if not usable:
        logger.warning(
            "No run has both a known recipe and a downloaded checkpoint; "
            "skipping the inference figure."
        )
        return None

    num_classes = recipes[usable[0].recipe].num_classes
    tiles = read_reference_tiles(data_config, num_classes, num_tiles)
    tile_names = [tile.name for tile in tiles]
    predictions: dict[str, list[np.ndarray]] = {}
    for row in usable:
        try:
            predictions[row.run] = predict_run(
                row, recipes[row.recipe], data_config, tile_names, device
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            logger.warning("Skipping the inference column for {}: {}", row.run, error)
    if not predictions:
        logger.warning("No run could be predicted; skipping the inference figure.")
        return None

    path = Path(output_dir) / FIGURE_NAME
    plot_inference_grid(tiles, predictions, path)
    return InferenceFigure(path=path, runs=tuple(predictions), num_tiles=len(tiles))
