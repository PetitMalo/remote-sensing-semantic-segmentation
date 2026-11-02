"""Compare runs on the test scores of their best-validation checkpoint.

A run directory holds three sources this module stitches together: the training
log (``csv/version_0/metrics.csv``), whose ``val/mIoU`` peak names the epoch a
fair comparison should score; the test-only run written by
``rsss.train --test-checkpoint`` (``eval/csv/version_*/metrics.csv``); and the
checkpoint itself, whose optimizer moments give the number of parameters that
actually trained. The result is a leaderboard with a 13-class IoU table and the
provenance of every number, written as Markdown and CSV.

Auto-logged training logs also contain a ``test/*`` row when the fit ended with
a test call. That row is not read here: before ``cad1919`` it scored the last
epoch's weights, and the ``eval/`` run is the one that scored the best epoch.

Run it with ``uv run python -m rsss.compare`` (``-h`` for the options). It reads
local artefacts only, so it runs on a laptop after
``scripts/pando/upload_run.sh`` and ``rclone copy`` have brought a run back.
``--figures`` additionally loads each best checkpoint, predicts a random,
domain-spread sample of test tiles and embeds one comparison grid in the
document; that step needs the dataset and matplotlib, so it is opt-in.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml
from loguru import logger

from rsss.predict import DEFAULT_SEED, read_metrics

if TYPE_CHECKING:
    from rsss.compare_figures import InferenceFigure

DEFAULT_RUNS_ROOT = "outputs/pando"
DEFAULT_MARKDOWN_NAME = "comparison.md"
DEFAULT_CSV_NAME = "comparison.csv"
DEFAULT_CONFIGS_DIR = "configs"
DEFAULT_DATA_CONFIG = "configs/data_subset.yaml"
DEFAULT_NUM_FIGURES = 3
# The monitor ``rsss.train`` keeps its best checkpoint on.
VALIDATION_COLUMN = "val/mIoU"
# Columns of the CSVLogger rows this module reads.
TRAINING_METRICS = Path("csv") / "version_0" / "metrics.csv"
EVAL_METRICS_GLOB = "eval/csv/version_*/metrics.csv"
# Recipe whose run is the classic baseline the delta is measured against.
BASELINE_MODEL = "unet"


@dataclass(frozen=True)
class RunComparison:
    """One run's validation peak and test scores, with where they came from.

    Parameters
    ----------
    run : str
        Run directory name, e.g. ``dinov3-bitfit-01``.
    recipe : str
        Model name the recipe builds, e.g. ``dinov3_bitfit``.
    trainable_params : int | None
        Parameters the optimizer updated, or None when the checkpoint was not
        read.
    total_params : int | None
        Parameters the model holds, or None when the checkpoint was not read.
    epochs : int
        Epochs the fit completed.
    best_epoch : int
        Epoch whose weights scored the best ``val/mIoU``.
    train_metrics : Path
        The training ``metrics.csv`` the validation peak comes from.
    test_metrics : Path
        The ``--test-checkpoint`` run's ``metrics.csv``.
    checkpoint : Path | None
        Best-validation weights, or None when the file is missing.
    val_miou : float
        Best-validation mean IoU.
    val_mf1 : float
        Mean F1 at that epoch.
    test_miou : float
        Test-split mean IoU, scored from ``checkpoint``.
    test_mf1 : float
        Test-split mean F1.
    test_loss : float
        Test-split loss.
    test_iou : dict[int, float]
        Per-class test IoU, keyed by class index.
    test_f1 : dict[int, float]
        Per-class test F1, keyed by class index.
    delta_miou : float | None
        Test mIoU minus the baseline's, or None when there is no baseline.
    """

    run: str
    recipe: str
    trainable_params: int | None
    total_params: int | None
    epochs: int
    best_epoch: int
    train_metrics: Path
    test_metrics: Path
    checkpoint: Path | None
    val_miou: float
    val_mf1: float
    test_miou: float
    test_mf1: float
    test_loss: float
    test_iou: dict[int, float]
    test_f1: dict[int, float]
    delta_miou: float | None = None

    @property
    def class_indices(self) -> list[int]:
        """Class indices the run reports, in order."""
        return sorted(self.test_iou)


def _metric(row: dict[str, str], column: str) -> float | None:
    """Return ``column`` of a CSVLogger row as a finite float, else None.

    A class absent from a split logs ``nan``; treating it as missing keeps it
    out of the peak search and out of the tables.
    """
    raw = row.get(column)
    if raw in (None, ""):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def best_validation_row(train_metrics: str | Path) -> dict[str, str]:
    """Return the row of the training log with the highest ``val/mIoU``.

    Raises
    ------
    FileNotFoundError
        If ``train_metrics`` does not exist.
    ValueError
        If the run logged no validation metric, e.g. a ``--fast-dev-run``.
    """
    path = Path(train_metrics)
    if not path.is_file():
        raise FileNotFoundError(f"No training metrics at {path}.")
    scored = [
        row for row in read_metrics(path) if _metric(row, VALIDATION_COLUMN) is not None
    ]
    if not scored:
        raise ValueError(f"{path} holds no {VALIDATION_COLUMN} row.")
    return max(scored, key=lambda row: _metric(row, VALIDATION_COLUMN) or 0.0)


def _version(path: Path) -> int:
    """Return the ``N`` of a CSVLogger ``version_N`` directory."""
    return int(path.parent.name.removeprefix("version_"))


def latest_test_row(run_dir: str | Path) -> tuple[dict[str, str], Path] | None:
    """Return the newest test row under ``run_dir/eval`` and its CSV path.

    Version directories without a test score are skipped: a fit aborted in the
    same output directory writes validation rows there but no test row, and it
    must not hide the later ``--test-checkpoint`` run.
    """
    files = sorted(Path(run_dir).glob(EVAL_METRICS_GLOB), key=_version, reverse=True)
    for path in files:
        scored = [
            row for row in read_metrics(path) if _metric(row, "test/mIoU") is not None
        ]
        if scored:
            return scored[-1], path
    return None


def best_checkpoint(run_dir: str | Path, epoch: int | None = None) -> Path | None:
    """Return the best-validation checkpoint of a run, if it was downloaded.

    Parameters
    ----------
    run_dir : str | Path
        Run directory holding ``checkpoints/``.
    epoch : int, optional
        Best epoch, used to pick between several files; without it the first
        non-``last.ckpt`` file is returned.
    """
    candidates = [
        path
        for path in sorted((Path(run_dir) / "checkpoints").glob("*.ckpt"))
        if path.name != "last.ckpt"
    ]
    if epoch is not None:
        tagged = [path for path in candidates if _checkpoint_epoch(path) == epoch]
        if tagged:
            return tagged[0]
    return candidates[0] if candidates else None


def _checkpoint_epoch(path: Path) -> int | None:
    """Return the epoch encoded in a ``epoch=<n>-step=<m>.ckpt`` filename."""
    head = path.stem.split("-", 1)[0]
    try:
        return int(head.removeprefix("epoch="))
    except ValueError:
        return None


# Buffers a state dict carries besides parameters: BatchNorm statistics, which
# are tensors but not weights the optimizer updates or a size count should add.
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def checkpoint_sizes(checkpoint: str | Path) -> tuple[int | None, int | None]:
    """Return ``(trainable, total)`` parameter counts of a checkpoint.

    The optimizer state only holds the parameters that took a step, so summing
    its Adam moments gives the trainable count without rebuilding the model.
    Both entries are None when the checkpoint kept no optimizer state.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("state_dict") or {}
    total = sum(
        tensor.numel()
        for name, tensor in weights.items()
        if torch.is_tensor(tensor) and not name.endswith(BUFFER_SUFFIXES)
    )
    optimizers = state.get("optimizer_states") or []
    if not optimizers:
        return None, total or None
    moments = optimizers[0].get("state") or {}
    trainable = sum(
        entry["exp_avg"].numel()
        for entry in moments.values()
        if torch.is_tensor(entry.get("exp_avg"))
    )
    return (trainable or None), (total or None)


def class_columns(row: dict[str, str], prefix: str) -> list[str]:
    """Return the ``prefix/class_N`` columns of a row, ordered by class index."""
    keys = [key for key in row if key.startswith(prefix)]
    return sorted(keys, key=lambda key: int(key.rsplit("_", 1)[1]))


def _per_class(row: dict[str, str], prefix: str) -> dict[int, float]:
    """Return a row's per-class metrics keyed by class index."""
    values: dict[int, float] = {}
    for column in class_columns(row, prefix):
        value = _metric(row, column)
        if value is not None:
            values[int(column.rsplit("_", 1)[1])] = value
    return values


def _recipe(run_dir: Path) -> str:
    """Return the model name the run's recipe builds."""
    path = run_dir / "csv" / "version_0" / "hparams.yaml"
    if not path.is_file():
        return "unknown"
    hparams = yaml.safe_load(path.read_text()) or {}
    return str(hparams.get("model", {}).get("name", "unknown"))


def _epochs_trained(rows: list[dict[str, str]]) -> int:
    """Return how many epochs the validation rows cover.

    The final row of a training log is the test one, whose epoch is already
    one past the last fitted epoch, so only validation rows count.
    """
    epochs = [
        int(float(row["epoch"]))
        for row in rows
        if row.get("epoch") and _metric(row, VALIDATION_COLUMN) is not None
    ]
    return max(epochs) + 1 if epochs else 0


def build_run(run_dir: str | Path, *, read_checkpoints: bool = True) -> RunComparison:
    """Read one run directory into a comparison row.

    Parameters
    ----------
    run_dir : str | Path
        Run directory holding ``csv/``, ``eval/`` and ``checkpoints/``.
    read_checkpoints : bool
        Read the best checkpoint for its parameter counts; False skips the
        download-scale load.

    Raises
    ------
    ValueError
        If the run has no test-split evaluation to compare.
    """
    path = Path(run_dir)
    train_metrics = path / TRAINING_METRICS
    rows = read_metrics(train_metrics)
    best = best_validation_row(train_metrics)
    found = latest_test_row(path)
    if found is None:
        raise ValueError(f"{path} holds no eval/ test row; score it first.")
    test, test_metrics = found

    best_epoch = int(float(best["epoch"]))
    checkpoint = best_checkpoint(path, best_epoch)
    trainable = total = None
    if read_checkpoints and checkpoint is not None:
        trainable, total = checkpoint_sizes(checkpoint)

    return RunComparison(
        run=path.name,
        recipe=_recipe(path),
        trainable_params=trainable,
        total_params=total,
        epochs=_epochs_trained(rows),
        best_epoch=best_epoch,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        checkpoint=checkpoint,
        val_miou=_metric(best, VALIDATION_COLUMN) or 0.0,
        val_mf1=_metric(best, "val/mF1") or 0.0,
        test_miou=_metric(test, "test/mIoU") or 0.0,
        test_mf1=_metric(test, "test/mF1") or 0.0,
        test_loss=_metric(test, "test/loss") or 0.0,
        test_iou=_per_class(test, "test/iou/class_"),
        test_f1=_per_class(test, "test/f1/class_"),
    )


def build_rows(
    runs_root: str | Path,
    *,
    baseline: str | None = None,
    read_checkpoints: bool = True,
) -> list[RunComparison]:
    """Read every run under ``runs_root``, ordered by test mIoU.

    Runs without an ``eval/`` test row are skipped with a warning, so a root can
    hold partially downloaded runs. ``delta_miou`` is computed against
    ``baseline`` when named, else against the first U-Net recipe found.
    """
    root = Path(runs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"No run directory at {root}.")
    rows: list[RunComparison] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (run_dir / TRAINING_METRICS).is_file():
            continue
        try:
            rows.append(build_run(run_dir, read_checkpoints=read_checkpoints))
        except ValueError as error:
            logger.warning("Skipping {}: {}", run_dir.name, error)
    rows.sort(key=lambda row: row.test_miou, reverse=True)
    return _with_delta(rows, baseline)


def _with_delta(rows: list[RunComparison], baseline: str | None) -> list[RunComparison]:
    """Set each row's ``delta_miou`` against the baseline run."""
    if not rows:
        return rows
    reference = next(
        (row for row in rows if row.run == baseline),
        None,
    )
    if reference is None and baseline is None:
        reference = next(
            (row for row in rows if row.recipe == BASELINE_MODEL),
            None,
        )
    if reference is None:
        return rows
    return [
        row
        if row is reference
        else replace(row, delta_miou=row.test_miou - reference.test_miou)
        for row in rows
    ]


def _format_params(row: RunComparison) -> str:
    """Format the trainable and total parameter counts for the Markdown table."""
    if row.trainable_params is None and row.total_params is None:
        return "—"
    trainable = (
        "—" if row.trainable_params is None else _format_millions(row.trainable_params)
    )
    total = "—" if row.total_params is None else _format_millions(row.total_params)
    return f"{trainable} / {total}"


def _format_millions(count: int) -> str:
    """Format a parameter count in millions with an ``M`` suffix."""
    return f"{count / 1e6:.2f} M"


def _format_value(value: float | None) -> str:
    """Format a metric to four decimals, or an em dash when missing."""
    return "—" if value is None else f"{value:.4f}"


def _format_delta(value: float | None) -> str:
    """Format a signed metric difference, or an em dash when missing."""
    return "—" if value is None else f"{value:+.4f}"


def _baseline_name(rows: list[RunComparison]) -> str | None:
    """Return the run the delta column is measured against."""
    if all(row.delta_miou is None for row in rows):
        return None
    reference = next(row for row in rows if row.delta_miou is None)
    return reference.run


def render_markdown(
    rows: list[RunComparison],
    *,
    figure: InferenceFigure | None = None,
    figure_href: str | None = None,
) -> str:
    """Render the comparison as a Markdown document.

    ``figure`` and ``figure_href`` are the inference grid and its path relative
    to the Markdown file; passing neither leaves the document table-only.
    """
    if not rows:
        return (
            "# Test-split comparison\n\n"
            "No run under this root holds both a validation peak and an "
            "`eval/` test score.\n"
        )
    baseline = _baseline_name(rows)
    reference = "the baseline" if baseline is None else f"`{baseline}`"
    intro = (
        f"{len(rows)} run(s), each scored on the checkpoint that peaked on "
        f"`val/mIoU`; Δ is test mIoU against {reference}."
    )
    header = (
        "| Run | Recipe | Trainable / total | Epochs | Best epoch | Val mIoU "
        "| Val mF1 | Test mIoU | Test mF1 | Test loss | Δ mIoU |"
    )
    lines = [
        "# Test-split comparison",
        "",
        intro,
        "",
        header,
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.run}` | `{row.recipe}` | {_format_params(row)} | "
            f"{row.epochs} | {row.best_epoch} | {_format_value(row.val_miou)} | "
            f"{_format_value(row.val_mf1)} | {_format_value(row.test_miou)} | "
            f"{_format_value(row.test_mf1)} | {_format_value(row.test_loss)} | "
            f"{_format_delta(row.delta_miou)} |"
        )

    lines += ["", "## Per-class test IoU", ""]
    run_names = " | ".join(f"`{row.run}`" for row in rows)
    lines += [
        f"| Class | {run_names} |",
        "|---|" + "---:|" * len(rows),
    ]
    classes = sorted({index for row in rows for index in row.class_indices})
    for index in classes:
        cells = " | ".join(_format_value(row.test_iou.get(index)) for row in rows)
        lines.append(f"| class_{index} | {cells} |")

    if figure is not None and figure_href is not None:
        run_list = ", ".join(f"`{run}`" for run in figure.runs)
        tiles = "tile" if figure.num_tiles == 1 else "tiles"
        lines += [
            "",
            "## Inference on test tiles",
            "",
            (
                f"{figure.num_tiles} test {tiles}, top to bottom, predicted by each "
                f"run's best-validation checkpoint: {run_list}. Every model sees the "
                "same tiles and the colours are the FLAIR#1 nomenclature of the "
                "per-class table (`0` is `Building`)."
            ),
            "",
            f"![Inference on test tiles]({figure_href})",
        ]

    lines += ["", "## Provenance", ""]
    for row in rows:
        checkpoint = "—" if row.checkpoint is None else f"`{row.checkpoint}`"
        lines.append(
            f"- `{row.run}`: val peak from `{row.train_metrics}` "
            f"(epoch {row.best_epoch}), test scores from `{row.test_metrics}`, "
            f"weights {checkpoint}."
        )
    used_row_note = (
        "The `test/*` row inside a training `metrics.csv` is not used: runs "
        "finished before the best-checkpoint fix scored the last epoch's "
        "weights there. A missing parameter count means the checkpoint was not "
        "downloaded or the run was read with `--no-params`."
    )
    caveats = (
        "Caveats: one seed and one run per arm, on the subset split (5,275 "
        "train / 3,550 val / 2,250 test tiles), so the deltas are single draws "
        "rather than confidence intervals."
    )
    lines += ["", used_row_note, "", caveats, ""]
    return "\n".join(lines)


CSV_COLUMNS = (
    "run",
    "recipe",
    "trainable_params",
    "total_params",
    "epochs",
    "best_epoch",
    "val_mIoU",
    "val_mF1",
    "test_mIoU",
    "test_mF1",
    "test_loss",
    "delta_mIoU",
    "checkpoint",
    "train_metrics",
    "test_metrics",
)


def write_csv(rows: list[RunComparison], path: str | Path) -> Path:
    """Write the comparison as one CSV row per run, per-class columns included."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = sorted({index for row in rows for index in row.class_indices})
    columns = (
        list(CSV_COLUMNS)
        + [f"test/iou/class_{index}" for index in classes]
        + [f"test/f1/class_{index}" for index in classes]
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_row(row, classes))
    return path


def _csv_row(row: RunComparison, classes: list[int]) -> dict[str, Any]:
    """Flatten a comparison row into CSV cells."""
    values: dict[str, Any] = {
        "run": row.run,
        "recipe": row.recipe,
        "trainable_params": row.trainable_params,
        "total_params": row.total_params,
        "epochs": row.epochs,
        "best_epoch": row.best_epoch,
        "val_mIoU": row.val_miou,
        "val_mF1": row.val_mf1,
        "test_mIoU": row.test_miou,
        "test_mF1": row.test_mf1,
        "test_loss": row.test_loss,
        "delta_mIoU": row.delta_miou,
        "checkpoint": None if row.checkpoint is None else str(row.checkpoint),
        "train_metrics": str(row.train_metrics),
        "test_metrics": str(row.test_metrics),
    }
    for index in classes:
        values[f"test/iou/class_{index}"] = row.test_iou.get(index)
        values[f"test/f1/class_{index}"] = row.test_f1.get(index)
    return values


def write_markdown(
    rows: list[RunComparison],
    path: str | Path,
    *,
    figure: InferenceFigure | None = None,
    figure_href: str | None = None,
) -> Path:
    """Write the comparison as a Markdown document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(rows, figure=figure, figure_href=figure_href))
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of the comparison entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help="Directory holding one sub-directory per run (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Markdown file to write (default: <runs-root>/{DEFAULT_MARKDOWN_NAME}).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=f"CSV file to write (default: <runs-root>/{DEFAULT_CSV_NAME}).",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Run the Δ mIoU column is measured against (default: the U-Net "
        "recipe's run).",
    )
    parser.add_argument(
        "--no-params",
        action="store_true",
        help="Skip the checkpoints, leaving the parameter counts blank.",
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Load each best checkpoint, predict a random sample of test tiles "
        "and embed one comparison grid in the Markdown (needs the dataset and "
        "matplotlib).",
    )
    parser.add_argument(
        "--num-figures",
        type=int,
        default=DEFAULT_NUM_FIGURES,
        help="Test tiles to draw when --figures is set (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed of the random tile selection for --figures (default: %(default)s).",
    )
    parser.add_argument(
        "--configs-dir",
        default=DEFAULT_CONFIGS_DIR,
        help="Directory whose train*.yaml recipes recover each run's data "
        "pipeline (default: %(default)s).",
    )
    parser.add_argument(
        "--data-config",
        default=DEFAULT_DATA_CONFIG,
        help="Data source the runs were trained with (default: %(default)s).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the data config's data_dir for --figures.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Device to predict on for --figures (default: CUDA when available).",
    )
    return parser


def _figure_href(figure: InferenceFigure, markdown: Path) -> str:
    """Return the figure path relative to the Markdown file, with slashes."""
    return os.path.relpath(figure.path, markdown.parent).replace(os.sep, "/")


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    """Build the comparison and return the two written paths."""
    if args.figures and args.no_params:
        raise ValueError("--figures needs the checkpoints; drop --no-params.")
    rows = build_rows(
        args.runs_root,
        baseline=args.baseline,
        read_checkpoints=not args.no_params,
    )
    root = Path(args.runs_root)
    markdown = Path(args.output) if args.output else root / DEFAULT_MARKDOWN_NAME
    csv_path = Path(args.csv) if args.csv else root / DEFAULT_CSV_NAME
    figure = None
    if args.figures:
        from rsss.compare_figures import generate

        figure = generate(
            rows,
            configs_dir=args.configs_dir,
            data_config_path=args.data_config,
            data_dir=args.data_dir,
            num_tiles=args.num_figures,
            seed=args.seed,
            device=args.device,
            output_dir=root / "figures",
        )
    return (
        write_markdown(
            rows,
            markdown,
            figure=figure,
            figure_href=None if figure is None else _figure_href(figure, markdown),
        ),
        write_csv(rows, csv_path),
    )


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, write the comparison and report the paths."""
    args = build_arg_parser().parse_args(argv)
    markdown, csv_path = run(args)
    logger.info("Wrote {} and {}.", markdown, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
