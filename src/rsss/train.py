"""Train a registered segmentation model on the configured FLAIR dataset.

Run it with ``uv run python -m rsss.train`` (``-h`` for the options). The
architecture, the recipe and the data source come from ``configs/train.yaml``
and ``configs/data.yaml``; the flags only override what has to change per run:
the dataset location and the output directory on a cluster, or a quick smoke
fit. A fit is followed by an evaluation on the test split, scored on the
best-validation checkpoint rather than the last epoch's weights, whose
per-class IoU is reported alongside the validation metrics; ``--skip-test``
turns that off. ``--test-checkpoint`` scores an existing checkpoint on the test
split without training, which is how runs are compared on equal footing.
``--resume`` continues a fit from ``<output-dir>/checkpoints/last.ckpt``.
Example: ``uv run python -m rsss.train --fast-dev-run --output-dir
/scratch/me/runs/smoke``.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from rsss.data.config import load_data_config
from rsss.data.data_module import FlairDatamodule
from rsss.training.config import TrainConfig, load_train_config
from rsss.training.module import SegmentationModule

DEFAULT_TRAIN_CONFIG = "configs/train.yaml"
DEFAULT_DATA_CONFIG = "configs/data.yaml"
DEFAULT_OUTPUT_DIR = "outputs"
# Metric the best checkpoint tracks, as logged by the module's
# on_validation_epoch_end. Keeping it here and in the tests avoids a silent
# mismatch between the module's metric names and the callback's monitor.
CHECKPOINT_MONITOR = "val/mIoU"


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of the training entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_TRAIN_CONFIG,
        help="Training recipe (default: %(default)s).",
    )
    parser.add_argument(
        "--data-config",
        default=DEFAULT_DATA_CONFIG,
        help="Data source (default: %(default)s).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the data config's data_dir, the directory datasets are "
        "extracted into (use an absolute path on a cluster).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Root for the logs and the checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the recipe's trainer.max_epochs.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=None,
        help="Cap the training batches per epoch, e.g. 20 for a smoke run.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run one training and one validation batch, then stop.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the test-split evaluation that follows a fit.",
    )
    parser.add_argument(
        "--test-checkpoint",
        default=None,
        metavar="CHECKPOINT",
        help="Skip training and score CHECKPOINT on the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed Python, NumPy and torch (and the dataloader workers).",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from CHECKPOINT, or from "
        "<output-dir>/checkpoints/last.ckpt when the value is omitted. "
        "--max-epochs stays the total, not the number of extra epochs.",
    )
    return parser


def resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    """Return the checkpoint ``--resume`` selects, or None to start fresh.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed entry-point arguments.

    Returns
    -------
    Path | None
        The checkpoint to pass to ``Trainer.fit``, or None for a fresh fit.

    Raises
    ------
    FileNotFoundError
        If the requested checkpoint does not exist.
    """
    if args.resume is None:
        return None
    if args.resume == "last":
        checkpoint = Path(args.output_dir) / "checkpoints" / "last.ckpt"
    else:
        checkpoint = Path(args.resume)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No checkpoint to resume from at {checkpoint}. Run without "
            "--resume first, or pass a checkpoint that exists."
        )
    return checkpoint


def resolve_test_checkpoint(args: argparse.Namespace) -> Path | None:
    """Return the checkpoint ``--test-checkpoint`` names, or None to train.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed entry-point arguments.

    Returns
    -------
    Path | None
        The checkpoint to pass to ``Trainer.test``, or None for a normal fit.

    Raises
    ------
    FileNotFoundError
        If the requested checkpoint does not exist.
    """
    if args.test_checkpoint is None:
        return None
    checkpoint = Path(args.test_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No checkpoint to test at {checkpoint}. Pass a .ckpt written by "
            "rsss.train, or leave --test-checkpoint out to fit a model."
        )
    return checkpoint


def seed_everything(seed: int | None) -> None:
    """Seed Python, NumPy, torch and the dataloader workers, when a seed is set."""
    if seed is not None:
        pl.seed_everything(seed, workers=True)


def build_datamodule(config: TrainConfig, args: argparse.Namespace) -> FlairDatamodule:
    """Build the data module from the data config and the recipe's settings."""
    data_config = load_data_config(args.data_config)
    if args.data_dir is not None:
        data_config = dataclasses.replace(data_config, data_dir=Path(args.data_dir))
    return FlairDatamodule(
        config=data_config,
        num_classes=config.num_classes,
        **dict(config.datamodule),
    )


def build_module(config: TrainConfig) -> SegmentationModule:
    """Build the shared training module from the recipe."""
    return SegmentationModule(
        model=config.model,
        num_classes=config.num_classes,
        loss=config.loss,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        class_weights=config.class_weights,
    )


def _trainer_kwargs(config: TrainConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Recipe settings, plus the flags that override them for one run."""
    kwargs: dict[str, Any] = dict(config.trainer)
    if args.max_epochs is not None:
        kwargs["max_epochs"] = args.max_epochs
    if args.limit_train_batches is not None:
        kwargs["limit_train_batches"] = args.limit_train_batches
    if args.fast_dev_run:
        kwargs["fast_dev_run"] = True
    return kwargs


def build_checkpoint_callbacks(output_dir: str | Path) -> list[Callback]:
    """Return the callbacks that keep the best and the last epoch's weights.

    Two callbacks, not one with ``save_last=True``: Lightning only refreshes
    ``last.ckpt`` when another checkpoint is written, so on a run whose final
    epoch is not a new best it would duplicate the best file instead of holding
    the final weights.

    ``save_on_train_epoch_end=False`` is required as well: with the default, the
    checkpoint is written at the end of the training epoch, before the
    validation that produces ``val/mIoU``. Its filename and score would then
    describe the previous epoch while its weights come from the current one.
    """
    dirpath = Path(output_dir) / "checkpoints"
    return [
        ModelCheckpoint(
            dirpath=dirpath,
            filename="{epoch}-{step}",
            monitor=CHECKPOINT_MONITOR,
            mode="max",
            save_top_k=1,
            save_on_train_epoch_end=False,
        ),
        ModelCheckpoint(
            dirpath=dirpath,
            filename="{epoch}-{step}",
            save_top_k=0,
            save_last=True,
            save_on_train_epoch_end=False,
        ),
    ]


def best_checkpoint_path(trainer: pl.Trainer) -> str | None:
    """Return the best-validation checkpoint the trainer wrote, or None.

    ``build_checkpoint_callbacks`` returns two callbacks: the first monitors
    ``val/mIoU`` and keeps the best epoch, the second only keeps ``last.ckpt``.
    Reading the monitored callback's path is more explicit than Lightning's
    ``ckpt_path="best"``, which warns when several callbacks are configured and
    raises when none wrote a file (``--fast-dev-run``).
    """
    for callback in trainer.checkpoint_callbacks:
        if (
            isinstance(callback, ModelCheckpoint)
            and callback.monitor == CHECKPOINT_MONITOR
            and callback.best_model_path
        ):
            return callback.best_model_path
    return None


def build_trainer(
    config: TrainConfig, args: argparse.Namespace, *, checkpointing: bool = True
) -> pl.Trainer:
    """Build the Trainer, logging and checkpointing under ``--output-dir``.

    ``checkpointing=False`` is for ``--test-checkpoint``: a test-only run writes
    no checkpoint, and restoring a fit's callback state into a second run
    directory would only warn about the changed ``dirpath``.
    """
    output_dir = Path(args.output_dir)
    callbacks = build_checkpoint_callbacks(output_dir) if checkpointing else []
    return pl.Trainer(
        **_trainer_kwargs(config, args),
        logger=CSVLogger(save_dir=str(output_dir), name="csv"),
        callbacks=callbacks,
        enable_checkpointing=checkpointing,
    )


def run(args: argparse.Namespace) -> pl.Trainer:
    """Fit the configured model, evaluate it, and return the Trainer that ran it.

    The test split is scored right after the fit, on the best-validation
    checkpoint rather than the weights the last epoch left, unless
    ``--skip-test`` is given. ``--test-checkpoint`` skips the fit and scores the
    named checkpoint instead, so runs can be compared on their best epoch.
    Returning the Trainer keeps the run inspectable from a caller (and from the
    tests), instead of hiding the run behind the exit code. With ``--resume`` the
    fit continues from a checkpoint, which is how a run longer than one Slurm
    allocation finishes.
    """
    if args.fast_dev_run and args.resume is not None:
        raise ValueError("--fast-dev-run and --resume are mutually exclusive.")
    if args.test_checkpoint is not None and (
        args.resume is not None or args.fast_dev_run or args.skip_test
    ):
        raise ValueError(
            "--test-checkpoint cannot be combined with --resume, "
            "--fast-dev-run or --skip-test."
        )
    seed_everything(args.seed)
    test_checkpoint = resolve_test_checkpoint(args)
    config = load_train_config(args.config)
    datamodule = build_datamodule(config, args)
    module = build_module(config)
    trainer = build_trainer(config, args, checkpointing=test_checkpoint is None)
    if test_checkpoint is not None:
        logger.info(
            "Scoring checkpoint {} on the test split; output under {}.",
            test_checkpoint,
            args.output_dir,
        )
        trainer.test(module, datamodule=datamodule, ckpt_path=test_checkpoint)
        return trainer
    checkpoint = resolve_resume_checkpoint(args)
    logger.info(
        "Training {} on {} classes; output under {}.",
        config.model["name"],
        config.num_classes,
        args.output_dir,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=checkpoint)
    if not args.skip_test:
        best = best_checkpoint_path(trainer)
        if best is None:
            logger.warning(
                "No best-validation checkpoint was written; scoring the final "
                "epoch's weights on the test split."
            )
        else:
            logger.info(
                "Scoring the best-validation checkpoint {} on the test split.", best
            )
        trainer.test(module, datamodule=datamodule, ckpt_path=best)
    return trainer


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, fit the model and report success."""
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
