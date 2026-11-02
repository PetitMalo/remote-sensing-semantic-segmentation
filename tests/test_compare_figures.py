"""Unit tests for the inference grid embedded in the run comparison.

The figures need matplotlib and a model; the backend is forced to Agg and the
models are tiny UNets (``encoder_weights`` None) over the shared synthetic
FLAIR#1 trees, so the file runs on CPU with no download.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
import yaml

from rsss.compare import RunComparison, build_rows, main, render_markdown
from rsss.compare_figures import (
    InferenceFigure,
    ReferenceTile,
    discover_recipes,
    generate,
    plot_inference_grid,
    read_reference_tiles,
    stretch_rgb,
)
from rsss.data.config import DataConfig
from rsss.data.splits import build_flair_splits
from rsss.training.module import SegmentationModule

NUM_CLASSES = 13
# Large enough for the UNet's five downsampling stages.
TILE_SIZE = 64
MODEL = {"name": "unet", "encoder_weights": None, "in_channels": 5}


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> Path:
    """Write a CSVLogger-shaped file, header included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_recipe(path: Path, model: Mapping[str, Any]) -> Path:
    """Write a minimal recipe the comparison can map a run back to."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "model": dict(model),
                "num_classes": NUM_CLASSES,
                "scheduler": {"warmup_epochs": 0, "t_max": 1},
                "datamodule": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "norm_type": "scaling",
                },
            }
        )
    )
    return path


def tiny_state() -> dict[str, torch.Tensor]:
    """Return the weights of the tiny UNet the recipes build."""
    module = SegmentationModule(
        model=MODEL,
        num_classes=NUM_CLASSES,
        scheduler={"warmup_epochs": 0, "t_max": 1},
    )
    return module.state_dict()


def write_run(root: Path, model: Mapping[str, Any]) -> Path:
    """Write one synthetic run with a real checkpoint and both metric logs."""
    run_dir = root / "tiny-01"
    (run_dir / "checkpoints").mkdir(parents=True)
    torch.save(
        {"state_dict": tiny_state()},
        run_dir / "checkpoints" / "epoch=1-step=10.ckpt",
    )
    version = run_dir / "csv" / "version_0"
    version.mkdir(parents=True)
    (version / "hparams.yaml").write_text(
        yaml.safe_dump({"model": dict(model), "num_classes": NUM_CLASSES})
    )
    write_csv(
        version / "metrics.csv",
        ("epoch", "step", "val/mIoU", "val/mF1"),
        [
            {"epoch": 0, "step": 0, "val/mIoU": 0.3, "val/mF1": 0.2},
            {"epoch": 1, "step": 10, "val/mIoU": 0.5, "val/mF1": 0.4},
        ],
    )
    write_csv(
        run_dir / "eval" / "csv" / "version_1" / "metrics.csv",
        ("test/mIoU", "test/mF1", "test/loss"),
        [{"test/mIoU": 0.4, "test/mF1": 0.3, "test/loss": 0.6}],
    )
    return run_dir


def write_data_config(path: Path, data_root: Path) -> Path:
    """Write a data config pointing at the synthetic toy dataset."""
    path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(data_root),
                "dataset": "flair_1_toy",
                "download": False,
            }
        )
    )
    return path


# Two tiles in each of three test domains, so a sample of three tiles can spread
# over all of them.
TEST_TILES = [
    ("D012_2019", "Z1_UU", "000001"),
    ("D012_2019", "Z1_UU", "000002"),
    ("D013_2019", "Z1_UU", "000003"),
    ("D013_2019", "Z1_UU", "000004"),
    ("D014_2019", "Z1_UU", "000005"),
    ("D014_2019", "Z1_UU", "000006"),
]


def write_multidomain_dataset(
    tmp_path: Path, write_flair_dataset, write_flair_split
) -> Path:
    """Write a toy dataset whose test split holds three geographical domains."""
    data_root = tmp_path / "data"
    dataset_root = data_root / "flair_1_toy_dataset"
    write_flair_dataset(dataset_root, size=TILE_SIZE)
    write_flair_split(dataset_root, "test", TEST_TILES, size=TILE_SIZE)
    return data_root


def comparison_row(run: str = "run-01") -> RunComparison:
    """Return one hand-built comparison row for rendering tests."""
    return RunComparison(
        run=run,
        recipe="unet",
        trainable_params=None,
        total_params=None,
        epochs=2,
        best_epoch=1,
        train_metrics=Path("csv/version_0/metrics.csv"),
        test_metrics=Path("eval/csv/version_1/metrics.csv"),
        checkpoint=None,
        val_miou=0.5,
        val_mf1=0.4,
        test_miou=0.4,
        test_mf1=0.3,
        test_loss=0.6,
        test_iou={0: 0.5, 1: 0.4},
        test_f1={0: 0.5, 1: 0.4},
    )


# --------------------------------------------------------------------------- #
# Recipes and tiles                                                           #
# --------------------------------------------------------------------------- #


def test_discover_recipes_indexes_by_model_name(tmp_path: Path) -> None:
    write_recipe(tmp_path / "train_unet.yaml", MODEL)
    write_recipe(tmp_path / "train_other.yaml", {"name": "dinov3_linear_probe"})

    recipes = discover_recipes(tmp_path)

    assert set(recipes) == {"unet", "dinov3_linear_probe"}
    assert recipes["unet"].model["encoder_weights"] is None


def test_discover_recipes_keeps_the_first_of_two_identical_models(
    tmp_path: Path,
) -> None:
    first = write_recipe(tmp_path / "train_a.yaml", MODEL)
    write_recipe(tmp_path / "train_b.yaml", MODEL)

    recipes = discover_recipes(tmp_path)

    assert recipes["unet"].datamodule == yaml.safe_load(first.read_text())["datamodule"]


def test_read_reference_tiles_returns_raw_rgb_and_label_indices(
    tmp_path: Path, write_flair_dataset
) -> None:
    data_root = tmp_path / "data"
    write_flair_dataset(data_root / "flair_1_toy_dataset", size=TILE_SIZE)

    tiles = read_reference_tiles(
        DataConfig(data_dir=data_root, dataset="flair_1_toy"), NUM_CLASSES, 1
    )

    assert len(tiles) == 1
    assert tiles[0].image.shape == (TILE_SIZE, TILE_SIZE, 3)
    assert tiles[0].target.shape == (TILE_SIZE, TILE_SIZE)
    # The fixture's masks are all MSK value 1, i.e. class index 0.
    assert set(np.unique(tiles[0].target)) == {0}
    assert tiles[0].name == "IMG_061946"


def test_read_reference_tiles_samples_across_test_domains(
    tmp_path: Path, write_flair_dataset, write_flair_split
) -> None:
    data_root = write_multidomain_dataset(
        tmp_path, write_flair_dataset, write_flair_split
    )
    domain_by_name = {f"IMG_{stem}": area for area, _, stem in TEST_TILES}

    tiles = read_reference_tiles(
        DataConfig(data_dir=data_root, dataset="flair_1_toy"),
        NUM_CLASSES,
        3,
        seed=0,
    )

    assert len(tiles) == 3
    assert len({domain_by_name[tile.name] for tile in tiles}) == 3


def test_read_reference_tiles_maps_out_of_range_labels_to_class_zero(
    tmp_path: Path, write_flair_dataset, write_msk
) -> None:
    data_root = tmp_path / "data"
    dataset_root = data_root / "flair_1_toy_dataset"
    write_flair_dataset(dataset_root, size=8)
    mask = Path(build_flair_splits(dataset_root)["test"]["MSK"][0])
    write_msk(mask, np.full((8, 8), 0))

    [tile] = read_reference_tiles(
        DataConfig(data_dir=data_root, dataset="flair_1_toy"), NUM_CLASSES, 1
    )

    # ``read_msk`` one-hot encodes 1..13, so an unknown value all-false'd to 0.
    assert set(np.unique(tile.target)) == {0}


def test_read_reference_tiles_rejects_no_tiles(
    tmp_path: Path, write_flair_dataset
) -> None:
    data_root = tmp_path / "data"
    write_flair_dataset(data_root / "flair_1_toy_dataset", size=8)

    with pytest.raises(ValueError, match="positive"):
        read_reference_tiles(
            DataConfig(data_dir=data_root, dataset="flair_1_toy"), NUM_CLASSES, 0
        )


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #


def test_stretch_rgb_maps_a_tile_to_the_unit_range() -> None:
    image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)

    stretched = stretch_rgb(image)

    assert stretched.shape == image.shape
    assert stretched.dtype == np.float32
    assert 0.0 <= stretched.min() and stretched.max() <= 1.0
    assert stretched.min() < 0.1 and stretched.max() > 0.9


def test_stretch_rgb_falls_back_to_the_plain_scale_for_a_constant_band() -> None:
    image = np.full((4, 4, 3), 128, dtype=np.uint8)

    stretched = stretch_rgb(image)

    assert np.allclose(stretched, 128 / 255)


def test_stretch_rgb_rejects_a_non_image() -> None:
    with pytest.raises(ValueError, match="height, width, bands"):
        stretch_rgb(np.zeros((4, 4)))


def test_plot_inference_grid_writes_a_png(tmp_path: Path) -> None:
    tile = ReferenceTile(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        target=np.zeros((8, 8), dtype=np.int64),
        name="IMG_000001",
    )
    predictions = {"a-01": [np.zeros((8, 8), dtype=np.int64)]}

    path = plot_inference_grid([tile], predictions, tmp_path / "inference.png")

    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_inference_grid_rejects_mismatched_predictions(tmp_path: Path) -> None:
    tile = ReferenceTile(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        target=np.zeros((8, 8), dtype=np.int64),
        name="IMG_000001",
    )

    with pytest.raises(ValueError, match="predictions for"):
        plot_inference_grid([tile], {"a-01": []}, tmp_path / "inference.png")


def test_render_markdown_embeds_the_inference_figure() -> None:
    figure = InferenceFigure(
        path=Path("figures/inference.png"), runs=("run-01",), num_tiles=1
    )

    text = render_markdown(
        [comparison_row()],
        figure=figure,
        figure_href="figures/inference.png",
    )

    assert "## Inference on test tiles" in text
    assert "![Inference on test tiles](figures/inference.png)" in text
    assert "`run-01`" in text


def test_render_markdown_without_a_figure_has_no_inference_section() -> None:
    assert "## Inference on test tiles" not in render_markdown([comparison_row()])


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #


def test_generate_skips_a_root_without_usable_runs(tmp_path: Path) -> None:
    data_config = write_data_config(tmp_path / "data.yaml", tmp_path / "data")

    assert generate([], data_config_path=data_config) is None


def test_run_rejects_figures_with_no_params(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--no-params"):
        main(["--runs-root", str(tmp_path), "--figures", "--no-params"])


def test_main_figures_writes_the_grid_and_embeds_it(
    tmp_path: Path, write_flair_dataset
) -> None:
    data_root = tmp_path / "data"
    write_flair_dataset(data_root / "flair_1_toy_dataset", size=TILE_SIZE)
    data_config = write_data_config(tmp_path / "data.yaml", data_root)
    configs = tmp_path / "configs"
    write_recipe(configs / "train.yaml", MODEL)
    runs_root = tmp_path / "runs"
    write_run(runs_root, MODEL)

    assert (
        main(
            [
                "--runs-root",
                str(runs_root),
                "--figures",
                "--num-figures",
                "1",
                "--configs-dir",
                str(configs),
                "--data-config",
                str(data_config),
                "--data-dir",
                str(data_root),
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    figure = runs_root / "figures" / "inference.png"
    assert figure.is_file() and figure.stat().st_size > 0
    text = (runs_root / "comparison.md").read_text()
    assert "## Inference on test tiles" in text
    assert "![Inference on test tiles](figures/inference.png)" in text
    # The synthetic run is the only one, and it was drawn.
    assert "`tiny-01`" in text


def test_generate_reports_the_runs_it_drew(tmp_path: Path, write_flair_dataset) -> None:
    data_root = tmp_path / "data"
    write_flair_dataset(data_root / "flair_1_toy_dataset", size=TILE_SIZE)
    data_config = write_data_config(tmp_path / "data.yaml", data_root)
    configs = tmp_path / "configs"
    write_recipe(configs / "train.yaml", MODEL)
    runs_root = tmp_path / "runs"
    write_run(runs_root, MODEL)
    [row] = build_rows(runs_root, read_checkpoints=False)

    figure = generate(
        [row],
        configs_dir=configs,
        data_config_path=data_config,
        data_dir=data_root,
        num_tiles=1,
        device="cpu",
        output_dir=tmp_path / "figures",
    )

    assert figure is not None
    assert figure.runs == ("tiny-01",)
    assert figure.num_tiles == 1
    assert figure.path.is_file()


def test_generate_predicts_the_same_random_tiles_as_the_reference(
    tmp_path: Path, write_flair_dataset, write_flair_split
) -> None:
    data_root = write_multidomain_dataset(
        tmp_path, write_flair_dataset, write_flair_split
    )
    data_config = write_data_config(tmp_path / "data.yaml", data_root)
    configs = tmp_path / "configs"
    write_recipe(configs / "train.yaml", MODEL)
    runs_root = tmp_path / "runs"
    write_run(runs_root, MODEL)
    [row] = build_rows(runs_root, read_checkpoints=False)

    figure = generate(
        [row],
        configs_dir=configs,
        data_config_path=data_config,
        data_dir=data_root,
        num_tiles=3,
        seed=0,
        device="cpu",
        output_dir=tmp_path / "figures",
    )

    assert figure is not None
    assert figure.num_tiles == 3
    assert figure.path.is_file()
