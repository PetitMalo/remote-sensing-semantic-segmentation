"""Unit tests for the run-comparison entry point.

Every case builds a synthetic run tree under ``pytest``'s ``tmp_path``: three
CSVLogger files and a tiny checkpoint whose optimizer state holds the moments
the parameter count is read from. No dataset, no model, no download.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml

from rsss.compare import (
    RunComparison,
    best_checkpoint,
    best_validation_row,
    build_arg_parser,
    build_rows,
    build_run,
    checkpoint_sizes,
    class_columns,
    latest_test_row,
    main,
    render_markdown,
)

NUM_CLASSES = 13
# Written in an order that is not numeric, to catch a lexicographic sort.
CLASS_ORDER = (0, 10, 1, 11, 2, 12, 3, 4, 5, 6, 7, 8, 9)

TRAIN_COLUMNS = ("epoch", "step", "val/mIoU", "val/mF1", "test/mIoU")


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> Path:
    """Write a CSVLogger-shaped file, header included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_training_log(
    run_dir: Path,
    val_mious: tuple[float, ...],
    model: str,
    *,
    test_miou: float = 0.1,
) -> Path:
    """Write ``csv/version_0``: one validation row per epoch and a test row.

    The trailing ``test/*`` row mimics an old-code fit, which scored the last
    epoch; :mod:`rsss.compare` must ignore it in favour of ``eval/``.
    """
    rows = [
        {"epoch": epoch, "step": epoch, "val/mIoU": miou, "val/mF1": miou}
        for epoch, miou in enumerate(val_mious)
    ]
    rows.append(
        {
            "epoch": len(val_mious),
            "step": len(val_mious),
            "test/mIoU": test_miou,
        }
    )
    path = write_csv(run_dir / "csv" / "version_0" / "metrics.csv", TRAIN_COLUMNS, rows)
    (run_dir / "csv" / "version_0" / "hparams.yaml").write_text(
        yaml.safe_dump({"model": {"name": model}, "num_classes": NUM_CLASSES})
    )
    return path


def write_test_log(
    run_dir: Path,
    version: int,
    miou: float,
    *,
    columns: tuple[int, ...] = CLASS_ORDER,
    nan_class: int | None = None,
) -> Path:
    """Write one ``--test-checkpoint`` run's ``eval/csv/version_N``."""
    names = tuple(
        [f"test/iou/class_{index}" for index in columns]
        + [f"test/f1/class_{index}" for index in columns]
        + ["test/loss", "test/mF1", "test/mIoU"]
    )
    row: dict[str, float] = {}
    for index in range(NUM_CLASSES):
        row[f"test/iou/class_{index}"] = 0.01 * index
        row[f"test/f1/class_{index}"] = 0.02 * index
    if nan_class is not None:
        row[f"test/iou/class_{nan_class}"] = float("nan")
    row.update({"test/loss": 0.5, "test/mF1": miou / 2, "test/mIoU": miou})
    return write_csv(
        run_dir / "eval" / "csv" / f"version_{version}" / "metrics.csv", names, [row]
    )


def write_checkpoint(path: Path, trainable: int = 12, total: int = 20) -> Path:
    """Write a checkpoint whose Adam moments cover ``trainable`` parameters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {"weight": torch.zeros(total)},
            "optimizer_states": [
                {
                    "state": {
                        0: {
                            "step": torch.tensor(1.0),
                            "exp_avg": torch.zeros(trainable),
                            "exp_avg_sq": torch.zeros(trainable),
                        }
                    }
                }
            ],
        },
        path,
    )
    return path


def make_run(
    root: Path,
    name: str,
    *,
    model: str = "dinov3_bitfit",
    val_mious: tuple[float, ...] = (0.2, 0.5, 0.3),
    test_miou: float = 0.6,
    version: int = 1,
    with_test: bool = True,
    with_checkpoint: bool = True,
) -> Path:
    """Build a complete synthetic run directory and return it."""
    run_dir = root / name
    write_training_log(run_dir, val_mious, model, test_miou=0.1)
    if with_test:
        write_test_log(run_dir, version, test_miou)
    if with_checkpoint:
        # The best epoch of ``val_mious`` is 1, with step 10.
        write_checkpoint(run_dir / "checkpoints" / "epoch=1-step=10.ckpt")
    return run_dir


# --------------------------------------------------------------------------- #
# Reading one run                                                             #
# --------------------------------------------------------------------------- #


def test_best_validation_row_picks_the_peak(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, "run-01", val_mious=(0.2, 0.5, 0.3))

    row = best_validation_row(run_dir / "csv" / "version_0" / "metrics.csv")

    assert row["epoch"] == "1"
    assert row["val/mIoU"] == "0.5"


def test_best_validation_row_without_scores_raises(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "metrics.csv", ("epoch", "step"), [{"epoch": 0, "step": 1}]
    )

    with pytest.raises(ValueError, match="val/mIoU"):
        best_validation_row(path)


def test_best_validation_row_ignores_a_nan_peak(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "metrics.csv",
        ("epoch", "val/mIoU"),
        [
            {"epoch": 0, "val/mIoU": 0.3},
            {"epoch": 1, "val/mIoU": float("nan")},
        ],
    )

    assert best_validation_row(path)["epoch"] == "0"


def test_latest_test_row_skips_a_version_without_test_scores(
    tmp_path: Path,
) -> None:
    run_dir = make_run(tmp_path, "run-01")
    # An aborted fit in the same directory only wrote validation rows.
    write_csv(
        run_dir / "eval" / "csv" / "version_0" / "metrics.csv",
        ("epoch", "val/mIoU"),
        [{"epoch": 0, "val/mIoU": 0.28}],
    )

    found = latest_test_row(run_dir)

    assert found is not None
    row, path = found
    assert path.parent.name == "version_1"
    assert float(row["test/mIoU"]) == pytest.approx(0.6)


def test_best_checkpoint_matches_the_best_epoch(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, "run-01")
    write_checkpoint(run_dir / "checkpoints" / "epoch=9-step=90.ckpt")
    (run_dir / "checkpoints" / "last.ckpt").write_text("last")

    checkpoint = best_checkpoint(run_dir, epoch=1)

    assert checkpoint is not None
    assert checkpoint.name == "epoch=1-step=10.ckpt"


def test_checkpoint_sizes_counts_adam_moments(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "tiny.ckpt", trainable=12, total=20)

    assert checkpoint_sizes(checkpoint) == (12, 20)


def test_checkpoint_sizes_without_optimizer_state_returns_none(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "weights-only.ckpt"
    torch.save({"state_dict": {"weight": torch.zeros(5)}}, checkpoint)

    assert checkpoint_sizes(checkpoint) == (None, 5)


def test_checkpoint_sizes_does_not_count_batch_norm_buffers(
    tmp_path: Path,
) -> None:
    checkpoint = write_checkpoint(tmp_path / "tiny.ckpt", trainable=12, total=20)
    # A BatchNorm layer adds statistics a parameter count must skip.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state["state_dict"].update(
        {
            "bn.running_mean": torch.zeros(4),
            "bn.running_var": torch.zeros(4),
            "bn.num_batches_tracked": torch.zeros(1),
        }
    )
    torch.save(state, checkpoint)

    assert checkpoint_sizes(checkpoint) == (12, 20)


def test_class_columns_sort_numerically() -> None:
    row = {f"test/iou/class_{index}": "0.0" for index in CLASS_ORDER}
    row["test/loss"] = "1.0"

    columns = class_columns(row, "test/iou/class_")

    assert columns == [f"test/iou/class_{index}" for index in range(NUM_CLASSES)]


def test_build_run_reads_the_best_epoch_and_the_eval_row(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, "run-01", test_miou=0.6)

    row = build_run(run_dir)

    assert row.run == "run-01"
    assert row.recipe == "dinov3_bitfit"
    assert row.epochs == 3
    assert row.best_epoch == 1
    assert row.val_miou == pytest.approx(0.5)
    assert row.test_miou == pytest.approx(0.6)
    assert row.trainable_params == 12
    assert row.total_params == 20
    assert row.test_metrics.parent.name == "version_1"
    assert row.class_indices == list(range(NUM_CLASSES))


def test_build_run_without_eval_raises(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, "run-01", with_test=False)

    with pytest.raises(ValueError, match="eval/"):
        build_run(run_dir)


def test_build_run_drops_a_class_absent_from_the_test_split(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, "run-01")
    write_test_log(run_dir, 2, 0.6, nan_class=12)

    row = build_run(run_dir)

    assert 12 not in row.test_iou
    assert row.test_metrics.parent.name == "version_2"


# --------------------------------------------------------------------------- #
# Collecting a root                                                           #
# --------------------------------------------------------------------------- #


def test_build_rows_ranks_runs_and_deltas_the_unet(tmp_path: Path) -> None:
    make_run(tmp_path, "unet-01", model="unet", test_miou=0.4)
    make_run(tmp_path, "probe-01", model="dinov3_linear_probe", test_miou=0.6)

    rows = build_rows(tmp_path)

    assert [row.run for row in rows] == ["probe-01", "unet-01"]
    by_name = {row.run: row for row in rows}
    assert by_name["probe-01"].delta_miou == pytest.approx(0.2)
    # The baseline is the reference, so it carries no delta of its own.
    assert by_name["unet-01"].delta_miou is None


def test_build_rows_honours_an_explicit_baseline(tmp_path: Path) -> None:
    make_run(tmp_path, "unet-01", model="unet", test_miou=0.4)
    make_run(tmp_path, "probe-01", model="dinov3_linear_probe", test_miou=0.6)

    rows = build_rows(tmp_path, baseline="probe-01")

    by_name = {row.run: row for row in rows}
    assert by_name["probe-01"].delta_miou is None
    assert by_name["unet-01"].delta_miou == pytest.approx(-0.2)


def test_build_rows_skips_a_run_without_eval(tmp_path: Path) -> None:
    make_run(tmp_path, "probe-01", test_miou=0.6)
    make_run(tmp_path, "unfinished-01", with_test=False)

    rows = build_rows(tmp_path)

    assert [row.run for row in rows] == ["probe-01"]


def test_build_rows_without_checkpoints_leaves_params_blank(tmp_path: Path) -> None:
    make_run(tmp_path, "probe-01")

    [row] = build_rows(tmp_path, read_checkpoints=False)

    assert row.trainable_params is None
    assert row.total_params is None


def test_build_rows_rejects_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No run directory"):
        build_rows(tmp_path / "absent")


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #


def test_render_markdown_lists_classes_in_order_and_names_the_baseline(
    tmp_path: Path,
) -> None:
    make_run(tmp_path, "unet-01", model="unet", test_miou=0.4)
    make_run(tmp_path, "probe-01", model="dinov3_linear_probe", test_miou=0.6)

    text = render_markdown(build_rows(tmp_path))

    assert "Δ is test mIoU against `unet-01`" in text
    assert "+0.2000" in text
    positions = [text.index(f"| class_{index} |") for index in range(NUM_CLASSES)]
    assert positions == sorted(positions)


def test_render_markdown_without_runs_says_so() -> None:
    assert "No run under this root" in render_markdown([])


def test_render_markdown_without_a_baseline_has_no_delta(tmp_path: Path) -> None:
    make_run(tmp_path, "probe-01", model="dinov3_linear_probe", test_miou=0.6)

    text = render_markdown(build_rows(tmp_path))

    assert "against the baseline" in text
    assert "+0.0000" not in text


def test_main_writes_markdown_and_csv(tmp_path: Path) -> None:
    make_run(tmp_path, "unet-01", model="unet", test_miou=0.4)
    make_run(tmp_path, "probe-01", model="dinov3_linear_probe", test_miou=0.6)

    assert main(["--runs-root", str(tmp_path)]) == 0

    markdown = tmp_path / "comparison.md"
    csv_path = tmp_path / "comparison.csv"
    assert markdown.is_file() and csv_path.is_file()
    assert "`probe-01`" in markdown.read_text()

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    # Per-class columns come in class order, IoU before F1.
    assert columns.index("test/iou/class_9") < columns.index("test/iou/class_10")
    assert columns.index("test/f1/class_12") > columns.index("test/iou/class_12")
    assert columns[-1] == "test/f1/class_12"
    assert [row["run"] for row in rows] == ["probe-01", "unet-01"]
    assert float(rows[0]["delta_mIoU"]) == pytest.approx(0.2)
    assert rows[0]["trainable_params"] == "12"
    assert rows[0]["test/iou/class_3"] == str(0.03)


def test_build_arg_parser_defaults() -> None:
    args = build_arg_parser().parse_args([])

    assert args.runs_root == "outputs/pando"
    assert args.output is None
    assert args.csv is None
    assert args.baseline is None
    assert args.no_params is False
    assert args.figures is False
    assert args.num_figures == 3
    assert args.seed == 0
    assert args.configs_dir == "configs"
    assert args.data_config == "configs/data_subset.yaml"
    assert args.data_dir is None
    assert args.device is None


def test_run_comparison_keeps_its_class_indices() -> None:
    row = RunComparison(
        run="run-01",
        recipe="unet",
        trainable_params=None,
        total_params=None,
        epochs=1,
        best_epoch=0,
        train_metrics=Path("csv/version_0/metrics.csv"),
        test_metrics=Path("eval/csv/version_1/metrics.csv"),
        checkpoint=None,
        val_miou=0.1,
        val_mf1=0.2,
        test_miou=0.3,
        test_mf1=0.4,
        test_loss=0.5,
        test_iou={7: 0.7, 2: 0.2},
        test_f1={7: 0.7, 2: 0.2},
    )

    assert row.class_indices == [2, 7]
