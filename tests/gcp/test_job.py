"""Unit tests for the on-VM job: staging, resuming and artefact sync.

Nothing here trains or reaches GCS: the fake runner records what the job would
run, and ``ensure_dataset`` is replaced where downloading is possible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rsss.data.download import DATASETS
from rsss.gcp import job
from rsss.gcp.job import (
    build_arg_parser,
    build_train_args,
    ensure_data,
    last_checkpoint_uri,
    pull_last_checkpoint,
    resolve_run_dir,
    run,
    sync_artifacts,
    upload_checkpoint,
)

FULL = DATASETS["flair_1_full"]
BUCKET = "rsss-data-bucket"
RUN = "unet-01"


def parse(*argv: str):
    """Parse the job's own arguments, with the required two filled in."""
    return build_arg_parser().parse_args(["--bucket", BUCKET, "--run", RUN, *argv])


def test_resolve_run_dir_joins_the_root_and_the_run():
    assert resolve_run_dir(parse()) == Path("/mnt/run") / RUN


def test_the_training_command_defers_to_the_recipe_by_default():
    args = parse("--run-root", "/mnt/run")
    command = build_train_args(args, resolve_run_dir(args), resume=False)

    assert command[:3] == [sys.executable, "-m", "rsss.train"]
    assert command[command.index("--output-dir") + 1] == "/mnt/run/unet-01"
    assert command[command.index("--seed") + 1] == "0"
    assert "--max-epochs" not in command
    assert "--resume" not in command
    assert "--limit-train-batches" not in command


def test_the_training_command_carries_the_optional_flags():
    args = parse("--epochs", "30", "--limit-train-batches", "2")
    command = build_train_args(args, "/mnt/run/unet-01", resume=True)

    assert command[command.index("--max-epochs") + 1] == "30"
    assert command[command.index("--resume") + 1] == "last"
    assert command[command.index("--limit-train-batches") + 1] == "2.0"


def test_last_checkpoint_uri_points_at_the_run():
    assert (
        last_checkpoint_uri(BUCKET, RUN)
        == f"gs://{BUCKET}/runs/{RUN}/checkpoints/last.ckpt"
    )


def test_pull_last_checkpoint_does_nothing_without_one(tmp_path, make_runner):
    runner = make_runner(missing=[last_checkpoint_uri(BUCKET, RUN)])

    assert not pull_last_checkpoint(BUCKET, RUN, tmp_path, runner=runner)
    assert runner.commands("gcloud", "storage", "cp") == []


def test_pull_last_checkpoint_copies_it_down(tmp_path, fake_runner):
    assert pull_last_checkpoint(BUCKET, RUN, tmp_path, runner=fake_runner)

    destination = tmp_path / "checkpoints" / "last.ckpt"
    assert fake_runner.commands("gcloud", "storage", "cp") == [
        [
            "gcloud",
            "storage",
            "cp",
            last_checkpoint_uri(BUCKET, RUN),
            str(destination),
        ]
    ]


def test_ensure_data_pulls_from_gcs_when_the_bucket_holds_it(
    tmp_path, fake_runner, monkeypatch
):
    verified: list[Path] = []
    monkeypatch.setattr(
        job, "ensure_dataset", lambda dataset, root: verified.append(Path(root))
    )

    root = ensure_data(BUCKET, FULL, tmp_path, runner=fake_runner)

    assert root == tmp_path / FULL.name
    assert verified == [tmp_path]
    assert [
        "gcloud",
        "storage",
        "rsync",
        "--recursive",
        f"gs://{BUCKET}/data/{FULL.name}",
        str(root),
    ] in fake_runner.calls
    # Nothing was staged locally, so nothing is uploaded.
    assert not fake_runner.commands(
        "gcloud", "storage", "rsync", "--recursive", str(root)
    )


def test_ensure_data_stages_and_uploads_on_the_first_run(
    tmp_path, make_runner, monkeypatch
):
    runner = make_runner(missing=[f"gs://{BUCKET}/data/{FULL.name}/CSVs"])
    staged: list[str] = []

    def fake_ensure(dataset, data_dir):
        staged.append(dataset.name)
        root = Path(data_dir) / dataset.name
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(job, "ensure_dataset", fake_ensure)

    root = ensure_data(BUCKET, FULL, tmp_path, runner=runner)

    assert staged == [FULL.name]
    assert [
        "gcloud",
        "storage",
        "rsync",
        "--recursive",
        str(root),
        f"gs://{BUCKET}/data/{FULL.name}",
    ] in runner.calls


def test_upload_checkpoint_publishes_partial_then_renames(tmp_path, fake_runner):
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.write_bytes(b"weights")
    uri = f"gs://{BUCKET}/runs/{RUN}/checkpoints/last.ckpt"

    upload_checkpoint(checkpoint, uri, runner=fake_runner)

    upload = fake_runner.commands("gcloud", "storage", "cp")[0]
    assert upload[-1] == f"{uri}.partial"
    assert fake_runner.calls[-1] == [
        "gcloud",
        "storage",
        "mv",
        f"{uri}.partial",
        uri,
    ]


def test_sync_artifacts_sends_csvs_the_log_and_the_checkpoints(tmp_path, fake_runner):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.ckpt").write_bytes(b"weights")
    (run_dir / "csv" / "version_0").mkdir(parents=True)
    (run_dir / "csv" / "version_0" / "metrics.csv").write_text("epoch\n")
    (run_dir / "train.log").write_text("[rsss] training\n")

    sync_artifacts(run_dir, BUCKET, RUN, runner=fake_runner)

    assert [
        "gcloud",
        "storage",
        "rsync",
        "--recursive",
        str(run_dir / "csv"),
        f"gs://{BUCKET}/runs/{RUN}/csv",
    ] in fake_runner.calls
    assert [
        "gcloud",
        "storage",
        "cp",
        str(run_dir / "train.log"),
        f"gs://{BUCKET}/runs/{RUN}/train.log",
    ] in fake_runner.calls
    assert len(fake_runner.commands("gcloud", "storage", "mv")) == 1


@pytest.mark.parametrize("name", ["last.ckpt", "3-1200.ckpt"])
def test_sync_artifacts_uploads_every_checkpoint(tmp_path, fake_runner, name):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / name).write_bytes(b"weights")

    sync_artifacts(run_dir, BUCKET, RUN, runner=fake_runner)

    uploads = [call[-1] for call in fake_runner.commands("gcloud", "storage", "cp")]
    assert uploads == [f"gs://{BUCKET}/runs/{RUN}/checkpoints/{name}.partial"]


def test_sync_artifacts_leaves_a_checkpoint_that_is_still_being_written(
    tmp_path, fake_runner
):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.ckpt").write_bytes(b"weights")

    sync_artifacts(run_dir, BUCKET, RUN, min_checkpoint_age=60.0, runner=fake_runner)

    assert fake_runner.commands("gcloud", "storage", "cp") == []


def test_dry_run_only_prints_the_training_command(make_runner):
    runner = make_runner()

    assert run(parse("--dry-run"), runner=runner) == 0
    assert runner.calls == []
