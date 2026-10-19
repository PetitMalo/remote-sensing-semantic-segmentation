"""Prepare and train one run on a GCP VM.

This is the on-instance entry point, started by the rendered startup script.
It stages the dataset (from GCS, or from IGN on the first run), pulls the last
checkpoint when asked to resume, runs :mod:`rsss.train`, and copies the run
directory to GCS while training. Checkpoints are uploaded as ``.partial`` and
then renamed, so a preempted upload never leaves a truncated ``last.ckpt`` for
the next attempt to resume from.

Run it with ``uv run python -m rsss.gcp.job`` on the VM, or with ``--dry-run``
to print the training command without touching GCS.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from loguru import logger

from rsss.data.config import load_data_config
from rsss.data.download import Dataset, ensure_dataset
from rsss.gcp.shell import Command, Runner, format_command, run
from rsss.gcp.storage import data_uri, object_exists, run_uri

DEFAULT_DATA_DIR = "/mnt/data"
DEFAULT_RUN_ROOT = "/mnt/run"
DEFAULT_TRAIN_CONFIG = "configs/train.yaml"
DEFAULT_DATA_CONFIG = "configs/data_full.yaml"
DEFAULT_SYNC_INTERVAL = 600
# Lightning writes a checkpoint in place, so a sync that lands mid-write would
# publish a truncated file. Background syncs wait this long after a checkpoint's
# last modification; the sync after training does not.
CHECKPOINT_GRACE_SECONDS = 10.0


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the parser of the on-instance entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket", required=True, help="Bucket holding the data and runs."
    )
    parser.add_argument("--run", required=True, help="Run name, also its GCS prefix.")
    parser.add_argument("--sha", default="", help="Commit under training, for the log.")
    parser.add_argument(
        "--train-config",
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
        default=DEFAULT_DATA_DIR,
        help="Root datasets are staged under (default: %(default)s).",
    )
    parser.add_argument(
        "--run-root",
        default=DEFAULT_RUN_ROOT,
        help="Root the run directories live under (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total epoch count; the recipe's value applies when omitted.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from <run>/checkpoints/last.ckpt when GCS holds one.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=None,
        help="Cap the training batches per epoch, for a smoke run.",
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=DEFAULT_SYNC_INTERVAL,
        help="Seconds between two artefact uploads (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training command without staging data or training.",
    )
    return parser


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """Return the run directory ``args`` selects."""
    return Path(args.run_root) / args.run


def build_train_args(
    args: argparse.Namespace, run_dir: str | Path, *, resume: bool
) -> list[str]:
    """Return the :mod:`rsss.train` command for this run."""
    command = [
        sys.executable,
        "-m",
        "rsss.train",
        "--config",
        args.train_config,
        "--data-config",
        args.data_config,
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(run_dir),
        "--seed",
        "0",
    ]
    if args.epochs is not None:
        command += ["--max-epochs", str(args.epochs)]
    if resume:
        command += ["--resume", "last"]
    if args.limit_train_batches is not None:
        command += ["--limit-train-batches", str(args.limit_train_batches)]
    return command


def last_checkpoint_uri(bucket: str, run_name: str) -> str:
    """Return the object holding the run's final checkpoint."""
    return f"{run_uri(bucket, run_name)}/checkpoints/last.ckpt"


def pull_last_checkpoint(
    bucket: str,
    run_name: str,
    run_dir: str | Path,
    *,
    runner: Runner = run,
) -> bool:
    """Copy the run's last checkpoint from GCS, if there is one.

    Returns
    -------
    bool
        Whether a checkpoint was found and copied.
    """
    uri = last_checkpoint_uri(bucket, run_name)
    if not object_exists(uri, runner=runner):
        logger.info("No checkpoint at {}; starting from scratch.", uri)
        return False
    destination = Path(run_dir) / "checkpoints" / "last.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Resuming from {}.", uri)
    runner(["gcloud", "storage", "cp", uri, str(destination)])
    return True


def ensure_data(
    bucket: str,
    dataset: Dataset,
    data_root: str | Path,
    *,
    runner: Runner = run,
) -> Path:
    """Stage ``dataset`` locally, mirroring GCS on the first run.

    GCS is the source of truth once it holds a copy. On the first run the
    dataset is downloaded from IGN, verified and uploaded, so later VMs only
    pull it from the bucket.
    """
    root = Path(data_root) / dataset.name
    prefix = data_uri(bucket, dataset.name)
    Path(data_root).mkdir(parents=True, exist_ok=True)

    if object_exists(f"{prefix}/CSVs", runner=runner):
        logger.info("Syncing {} -> {}.", prefix, root)
        runner(["gcloud", "storage", "rsync", "--recursive", prefix, str(root)])
        # The bucket copy could be truncated; verify the tiles before training.
        ensure_dataset(dataset, data_root)
    else:
        logger.info("{} is not in GCS yet; staging it locally.", dataset.name)
        ensure_dataset(dataset, data_root)
        logger.info("Uploading {} -> {}.", root, prefix)
        runner(["gcloud", "storage", "rsync", "--recursive", str(root), prefix])
    return root


def upload_checkpoint(path: str | Path, uri: str, *, runner: Runner = run) -> None:
    """Upload ``path`` to ``uri`` without ever publishing a partial object."""
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="rsss-ckpt-") as temporary:
        # Snapshot the file so the upload reads one version even if Lightning
        # is still writing the checkpoint.
        snapshot = Path(temporary) / path.name
        shutil.copy2(path, snapshot)
        runner(["gcloud", "storage", "cp", str(snapshot), f"{uri}.partial"])
    runner(["gcloud", "storage", "mv", f"{uri}.partial", uri])


def sync_artifacts(
    run_dir: str | Path,
    bucket: str,
    run_name: str,
    *,
    min_checkpoint_age: float = 0.0,
    runner: Runner = run,
) -> None:
    """Copy the CSVs, the training log and the checkpoints to GCS.

    Parameters
    ----------
    min_checkpoint_age : float
        Skip a checkpoint modified less than this many seconds ago, for a
        background sync racing the trainer's write.
    """
    run_dir = Path(run_dir)
    destination = run_uri(bucket, run_name)

    csv_dir = run_dir / "csv"
    if csv_dir.is_dir():
        runner(
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                str(csv_dir),
                f"{destination}/csv",
            ]
        )

    log_path = run_dir / "train.log"
    if log_path.is_file():
        runner(["gcloud", "storage", "cp", str(log_path), f"{destination}/train.log"])

    checkpoints = run_dir / "checkpoints"
    if checkpoints.is_dir():
        for checkpoint in sorted(checkpoints.glob("*.ckpt")):
            if min_checkpoint_age and (
                time.time() - checkpoint.stat().st_mtime < min_checkpoint_age
            ):
                logger.debug("{} is being written; skipping it.", checkpoint.name)
                continue
            upload_checkpoint(
                checkpoint,
                f"{destination}/checkpoints/{checkpoint.name}",
                runner=runner,
            )


def start_sync_thread(
    run_dir: str | Path,
    bucket: str,
    run_name: str,
    interval: int,
    *,
    runner: Runner = run,
) -> tuple[threading.Event, threading.Thread]:
    """Start a daemon thread that uploads artefacts every ``interval`` seconds."""
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(interval):
            try:
                sync_artifacts(
                    run_dir,
                    bucket,
                    run_name,
                    min_checkpoint_age=CHECKPOINT_GRACE_SECONDS,
                    runner=runner,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                # A transient GCS or disk failure must not stop training.
                logger.warning("Artefact sync failed, retrying later: {}", error)

    thread = threading.Thread(target=_loop, name="rsss-gcs-sync", daemon=True)
    thread.start()
    return stop, thread


def train(command: Command, run_dir: str | Path) -> int:
    """Run training, copying its output to ``<run_dir>/train.log`` and stdout."""
    log_path = Path(run_dir) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("$ {}", format_command(command))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def run(args: argparse.Namespace, *, runner: Runner = run) -> int:
    """Prepare the run, train it, and return the training exit status."""
    run_dir = resolve_run_dir(args)

    if args.dry_run:
        command = build_train_args(args, run_dir, resume=args.resume)
        logger.info("$ {}", format_command(command))
        return 0

    data_config = load_data_config(args.data_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run {} from commit {}.", args.run, args.sha or "unknown")
    resumed = (
        pull_last_checkpoint(args.bucket, args.run, run_dir, runner=runner)
        if args.resume
        else False
    )
    ensure_data(
        args.bucket,
        data_config.remote,
        Path(args.data_dir),
        runner=runner,
    )

    stop, thread = start_sync_thread(
        run_dir, args.bucket, args.run, args.sync_interval, runner=runner
    )
    status = 1
    try:
        status = train(build_train_args(args, run_dir, resume=resumed), run_dir)
    finally:
        stop.set()
        thread.join()
        try:
            sync_artifacts(run_dir, args.bucket, args.run, runner=runner)
        except (OSError, subprocess.CalledProcessError) as error:
            # Report it, but keep the training exit status.
            logger.error("Final artefact sync failed: {}", error)
    logger.info("Training finished with status {}.", status)
    return status


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and run one training job."""
    args = build_arg_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
