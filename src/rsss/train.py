"""Train a registered segmentation model on the configured FLAIR dataset.

Run it with ``uv run python -m rsss.train`` (``-h`` for the options). The
architecture, the recipe and the data source come from ``configs/train.yaml``
and ``configs/data.yaml``; the flags only override what has to change per run:
the dataset location and the output directory on a cluster, or a quick smoke
fit. A fit is followed by an evaluation on the test split, whose per-class IoU
is reported alongside the validation metrics; ``--skip-test`` turns that off.
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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from rsss.data.config import load_data_config
from rsss.data.data_module import FlairDatamodule
from rsss.training.config import TrainConfig, load_train_config
from rsss.training.module import SegmentationModule

DEFAULT_TRAIN_CONFIG = "configs/train.yaml"
DEFAULT_DATA_CONFIG = "configs/data.yaml"
DEFAULT_OUTPUT_DIR = "outputs"


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
        "--seed",
        type=int,
        default=None,
        help="Seed Python, NumPy and torch (and the dataloader workers).",
    )
    return parser


def build_datamodule(config: TrainConfig, args: argparse.Namespace) -> FlairDatamodule:
    """Build the data module from the data config and the recipe's settings."""
    data_config = load_data_config(args.data_config)
    if args.data_dir is not None:
        data_config = dataclasses.replace(data_config, data_dir=Path(args.data_dir))
    return FlairDatamodule(config=data_config, **dict(config.datamodule))


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


def build_trainer(config: TrainConfig, args: argparse.Namespace) -> pl.Trainer:
    """Build the Trainer, logging and checkpointing under ``--output-dir``."""
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
    output_dir = Path(args.output_dir)
    return pl.Trainer(
        **_trainer_kwargs(config, args),
        logger=CSVLogger(save_dir=str(output_dir), name="csv"),
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename="{epoch}-{step}",
                save_top_k=-1,
            )
        ],
    )


def run(args: argparse.Namespace) -> pl.Trainer:
    """Fit the configured model, evaluate it, and return the Trainer that ran it.

    The test split is scored right after the fit, on the weights the last epoch
    left, unless ``--skip-test`` is given. Returning the Trainer keeps the run
    inspectable from a caller (and from the tests), instead of hiding the run
    behind the exit code.
    """
    config = load_train_config(args.config)
    datamodule = build_datamodule(config, args)
    module = build_module(config)
    trainer = build_trainer(config, args)
    logger.info(
        "Training {} on {} classes; output under {}.",
        config.model["name"],
        config.num_classes,
        args.output_dir,
    )
    trainer.fit(module, datamodule=datamodule)
    if not args.skip_test:
        trainer.test(module, datamodule=datamodule)
    return trainer


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, fit the model and report success."""
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
