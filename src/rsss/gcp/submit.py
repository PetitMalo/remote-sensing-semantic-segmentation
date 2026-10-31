"""Submit one training run to a spot GPU VM on GCP.

Run it with ``uv run python -m rsss.gcp.submit`` (``-h`` for the options), or
through ``scripts/gcp/submit.sh``. The launcher archives the current commit,
uploads it once per commit, renders the VM's startup script and creates a spot
L4 instance. It never uploads the dataset: the first run stages it into GCS.

The worktree has to be clean: the VM runs the archived commit, not whatever is
uncommitted locally. ``--allow-dirty`` archives HEAD anyway, for a deliberate
test of an older commit.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from rsss.gcp.config import DEFAULT_CONFIG, GcpConfig, load_gcp_config
from rsss.gcp.shell import Runner, format_command, run
from rsss.gcp.startup import render_startup_script
from rsss.gcp.storage import code_uri, object_exists

# ``src/rsss/gcp/submit.py`` -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_CONFIG = "configs/train.yaml"
DEFAULT_DATA_CONFIG = "configs/data_full.yaml"
DEFAULT_SYNC_INTERVAL = 600
_RUN_NAME = re.compile(r"[a-z][a-z0-9-]{0,40}")


@dataclass(frozen=True)
class Submission:
    """Everything one ``gcloud compute instances create`` call needs."""

    config: GcpConfig
    run_name: str
    sha: str
    archive: Path
    startup_script: Path
    upload_command: list[str]
    create_command: list[str]


def default_run_name(now: datetime | None = None) -> str:
    """Return a timestamped run name, ``unet-YYYYmmdd-HHMMSS``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"unet-{stamp}"


def validate_run_name(name: str) -> str:
    """Return ``name`` if it can be a GCS prefix and a VM label.

    Raises
    ------
    ValueError
        If the name holds anything but lowercase letters, digits and hyphens,
        or is too long.
    """
    if not _RUN_NAME.fullmatch(name):
        raise ValueError(
            f"Invalid run name {name!r}: use a lowercase letter followed by "
            "lowercase letters, digits or hyphens (at most 41 characters)."
        )
    return name


def instance_name(run_name: str) -> str:
    """Return the instance name for ``run_name``."""
    return f"rsss-{run_name}"


def git_head(repo_root: str | Path, *, runner: Runner = run) -> str:
    """Return the commit the VM will run."""
    result = runner(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
    )
    return result.stdout.strip()


def worktree_is_clean(repo_root: str | Path, *, runner: Runner = run) -> bool:
    """Return whether ``git status --porcelain`` is empty."""
    result = runner(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
    )
    return not result.stdout.strip()


def archive_code(
    repo_root: str | Path, sha: str, destination: str | Path, *, runner: Runner = run
) -> Path:
    """Write ``sha``'s tree to ``destination`` as a gzipped tarball."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    runner(
        ["git", "archive", "--format=tar.gz", f"--output={destination}", sha],
        cwd=str(repo_root),
    )
    return destination


def build_create_command(
    config: GcpConfig,
    run_name: str,
    sha: str,
    startup_script: str | Path,
    *,
    zone: str | None = None,
    machine_type: str | None = None,
) -> list[str]:
    """Return the ``gcloud compute instances create`` command for one run."""
    metadata = ",".join(
        (
            f"rsss-bucket={config.bucket}",
            f"rsss-run={run_name}",
            f"rsss-sha={sha}",
            # So a boot that dies before the job runs is still diagnosable, via
            # `gcloud compute instances get-serial-port-output`.
            "serial-port-enable=TRUE",
        )
    )
    return [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance_name(run_name),
        "--project",
        config.project,
        "--zone",
        zone or config.zone,
        "--machine-type",
        machine_type or config.machine_type,
        "--maintenance-policy",
        "TERMINATE",
        "--provisioning-model",
        "SPOT",
        "--image-family",
        config.image_family,
        "--image-project",
        config.image_project,
        "--boot-disk-size",
        f"{config.boot_disk_size_gb}GB",
        "--boot-disk-type",
        config.boot_disk_type,
        "--service-account",
        config.vm_service_account,
        "--scopes",
        "cloud-platform",
        "--metadata-from-file",
        f"startup-script={startup_script}",
        "--metadata",
        metadata,
        "--labels",
        f"run={run_name},sha={sha[:12]}",
        "--max-run-duration",
        config.max_run_duration,
        "--instance-termination-action",
        "DELETE",
        "--quiet",
    ]


def plan_submission(
    config: GcpConfig,
    run_name: str,
    sha: str,
    *,
    train_config: str = DEFAULT_TRAIN_CONFIG,
    data_config: str = DEFAULT_DATA_CONFIG,
    epochs: int | None = None,
    resume: bool = False,
    limit_train_batches: float | None = None,
    sync_interval: int = DEFAULT_SYNC_INTERVAL,
    zone: str | None = None,
    machine_type: str | None = None,
    repo_root: str | Path = REPO_ROOT,
    workdir: str | Path | None = None,
) -> Submission:
    """Build the local archive, the startup script and the gcloud command."""
    validate_run_name(run_name)
    directory = Path(workdir or tempfile.mkdtemp(prefix="rsss-submit-"))
    archive = archive_code(repo_root, sha, directory / f"rsss-{sha[:12]}.tar.gz")
    startup_script = directory / "startup.sh"
    startup_script.write_text(
        render_startup_script(
            bucket=config.bucket,
            run_name=run_name,
            sha=sha,
            train_config=train_config,
            data_config=data_config,
            epochs=epochs,
            resume=resume,
            limit_train_batches=limit_train_batches,
            sync_interval=sync_interval,
        )
    )
    startup_script.chmod(0o755)
    return Submission(
        config=config,
        run_name=run_name,
        sha=sha,
        archive=archive,
        startup_script=startup_script,
        upload_command=[
            "gcloud",
            "storage",
            "cp",
            str(archive),
            code_uri(config.bucket, sha),
        ],
        create_command=build_create_command(
            config,
            run_name,
            sha,
            startup_script,
            zone=zone,
            machine_type=machine_type,
        ),
    )


def execute(
    submission: Submission, *, dry_run: bool = False, runner: Runner = run
) -> None:
    """Upload the code archive when needed, then create the VM."""
    if dry_run:
        print(format_command(submission.upload_command))
        print(format_command(submission.create_command))
        return
    if object_exists(code_uri(submission.config.bucket, submission.sha), runner=runner):
        logger.info(
            "Code for {} is already in GCS; skipping the upload.",
            submission.sha[:12],
        )
    else:
        runner(submission.upload_command)
    runner(submission.create_command)


def print_follow_up(config: GcpConfig, run_name: str, sha: str) -> None:
    """Print the commands that follow a successful submission."""
    instance = instance_name(run_name)
    print(
        f"\nSubmitted {instance} in {config.zone} (commit {sha[:12]}).\n"
        f"  gcloud compute ssh {instance} --project {config.project} "
        f"--zone {config.zone} -- tail -f /var/log/rsss-startup.log\n"
        f"  gcloud storage ls -r gs://{config.bucket}/runs/{run_name}/\n"
        f"  scripts/gcp/pull_run.sh {run_name}\n"
        f"  gcloud compute instances delete {instance} --project "
        f"{config.project} --zone {config.zone}\n"
        "\nThe VM stops itself after the run; delete it to stop paying for the "
        "boot disk.\n"
    )


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def _env_bool(name: str) -> bool:
    """Read a boolean environment variable, rejecting ambiguous spellings."""
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return False
    # ``scripts/pando/job.sh`` spells resume as ``RESUME=last``; accept it so
    # one environment convention works for both targets.
    if value in {"1", "true", "yes", "on", "last"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean (1/true/yes/on or 0/false/no/off), got {value!r}."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the parser of the submission entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.environ.get("GCP_CONFIG", DEFAULT_CONFIG),
        help="GCP host config (default: %(default)s).",
    )
    parser.add_argument(
        "--run",
        default=os.environ.get("RUN"),
        help="Run name and GCS prefix (default: unet-<timestamp>).",
    )
    parser.add_argument(
        "--train-config",
        default=DEFAULT_TRAIN_CONFIG,
        help="Training recipe inside the archive (default: %(default)s).",
    )
    parser.add_argument(
        "--data-config",
        default=DEFAULT_DATA_CONFIG,
        help="Data source inside the archive (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_env_int("EPOCHS"),
        help="Total epoch count; the recipe's value applies when omitted.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=_env_bool("RESUME"),
        help="Continue from the run's last checkpoint when GCS holds one.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=_env_float("LIMIT_TRAIN_BATCHES"),
        help="Cap the training batches per epoch, for a smoke run.",
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=DEFAULT_SYNC_INTERVAL,
        help="Seconds between two artefact uploads (default: %(default)s).",
    )
    parser.add_argument(
        "--zone",
        default=None,
        help="Override the config's zone, e.g. when L4 capacity is short.",
    )
    parser.add_argument(
        "--machine-type",
        default=None,
        help="Override the config's machine type.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Submit HEAD even when the worktree has uncommitted changes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gcloud commands without uploading or creating anything.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments, plan the run and create the VM."""
    args = build_arg_parser().parse_args(argv)
    config = load_gcp_config(args.config)
    if args.zone:
        config = replace(config, zone=args.zone)
    if args.machine_type:
        config = replace(config, machine_type=args.machine_type)

    run_name = validate_run_name(args.run or default_run_name())

    if not args.allow_dirty and not worktree_is_clean(REPO_ROOT):
        raise SystemExit(
            "The worktree has uncommitted changes, and the VM runs HEAD. Commit "
            "them first, or pass --allow-dirty to submit the last commit anyway."
        )

    sha = git_head(REPO_ROOT)
    submission = plan_submission(
        config,
        run_name,
        sha,
        train_config=args.train_config,
        data_config=args.data_config,
        epochs=args.epochs,
        resume=args.resume,
        limit_train_batches=args.limit_train_batches,
        sync_interval=args.sync_interval,
        zone=args.zone,
        machine_type=args.machine_type,
    )
    execute(submission, dry_run=args.dry_run)
    if not args.dry_run:
        print_follow_up(config, run_name, sha)
    else:
        logger.info("Startup script: {}", submission.startup_script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
