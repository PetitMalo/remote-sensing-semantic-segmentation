"""Unit tests for the local submission launcher."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rsss.gcp import submit
from rsss.gcp.config import GcpConfig
from rsss.gcp.storage import code_uri
from rsss.gcp.submit import (
    Submission,
    build_arg_parser,
    build_create_command,
    default_run_name,
    execute,
    git_head,
    instance_name,
    plan_submission,
    print_follow_up,
    validate_run_name,
    worktree_is_clean,
)

CONFIG = GcpConfig(
    project="rsss-project",
    zone="europe-west4-a",
    bucket="rsss-data-bucket",
    service_account="rsss-vm@rsss-project.iam.gserviceaccount.com",
)
SHA = "a" * 40
RUN = "unet-01"


def submission() -> Submission:
    """Return a submission with dummy commands, for the execute tests."""
    return Submission(
        config=CONFIG,
        run_name=RUN,
        sha=SHA,
        archive=Path("/tmp/rsss.tar.gz"),
        startup_script=Path("/tmp/startup.sh"),
        upload_command=["gcloud", "storage", "cp", "/tmp/rsss.tar.gz", "gs://x"],
        create_command=["gcloud", "compute", "instances", "create", "rsss-unet-01"],
    )


def test_the_default_run_name_is_timestamped():
    now = datetime(2026, 10, 19, 12, 30, 45, tzinfo=UTC)

    assert default_run_name(now) == "unet-20261019-123045"


@pytest.mark.parametrize("name", ["unet-01", "a", "unet-20261019-120000"])
def test_valid_run_names_are_accepted(name):
    assert validate_run_name(name) == name


@pytest.mark.parametrize("name", ["Unet-01", "unet_01", "-unet", "", "a" * 42])
def test_invalid_run_names_are_rejected(name):
    with pytest.raises(ValueError, match="Invalid run name"):
        validate_run_name(name)


def test_instance_names_are_prefixed():
    assert instance_name(RUN) == "rsss-unet-01"


def test_the_create_command_requests_a_spot_l4_vm_for_the_commit():
    command = build_create_command(CONFIG, RUN, SHA, "/tmp/startup.sh")

    assert command[:5] == ["gcloud", "compute", "instances", "create", "rsss-unet-01"]
    assert command[command.index("--provisioning-model") + 1] == "SPOT"
    assert command[command.index("--machine-type") + 1] == "g2-standard-8"
    assert command[command.index("--image-family") + 1] == "common-cu130-ubuntu-2204"
    assert command[command.index("--boot-disk-size") + 1] == "200GB"
    assert (
        command[command.index("--service-account") + 1]
        == "rsss-vm@rsss-project.iam.gserviceaccount.com"
    )
    assert command[command.index("--metadata-from-file") + 1] == (
        "startup-script=/tmp/startup.sh"
    )
    metadata = command[command.index("--metadata") + 1]
    assert f"rsss-bucket={CONFIG.bucket}" in metadata
    assert f"rsss-sha={SHA}" in metadata
    assert "serial-port-enable=TRUE" in metadata
    assert command[command.index("--labels") + 1] == f"run={RUN},sha={SHA[:12]}"
    # The cost guard must be paired with a termination action gcloud accepts.
    assert command[command.index("--max-run-duration") + 1] == "24h"
    assert command[command.index("--instance-termination-action") + 1] == "DELETE"


def test_the_zone_and_machine_type_can_be_overridden():
    command = build_create_command(
        CONFIG,
        RUN,
        SHA,
        "/tmp/startup.sh",
        zone="europe-west1-b",
        machine_type="n1-standard-8",
    )

    assert command[command.index("--zone") + 1] == "europe-west1-b"
    assert command[command.index("--machine-type") + 1] == "n1-standard-8"


def test_plan_submission_writes_the_startup_script_and_the_upload(
    tmp_path, monkeypatch
):
    def fake_archive(repo_root, sha, destination, *, runner=None):
        destination = Path(destination)
        destination.write_bytes(b"code")
        return destination

    monkeypatch.setattr(submit, "archive_code", fake_archive)

    planned = plan_submission(CONFIG, RUN, SHA, workdir=tmp_path)

    assert planned.archive.read_bytes() == b"code"
    assert planned.startup_script.is_file()
    assert "--run unet-01" in planned.startup_script.read_text()
    assert planned.upload_command == [
        "gcloud",
        "storage",
        "cp",
        str(planned.archive),
        code_uri(CONFIG.bucket, SHA),
    ]


def test_dry_run_prints_both_commands_without_calling_them(capsys, make_runner):
    runner = make_runner()

    execute(submission(), dry_run=True, runner=runner)

    printed = capsys.readouterr().out
    assert "gcloud storage cp /tmp/rsss.tar.gz gs://x" in printed
    assert "gcloud compute instances create rsss-unet-01" in printed
    assert runner.calls == []


def test_execute_uploads_the_code_when_the_bucket_lacks_it(make_runner):
    runner = make_runner(missing=[code_uri(CONFIG.bucket, SHA)])
    planned = submission()

    execute(planned, runner=runner)

    assert runner.calls[0] == ["gcloud", "storage", "ls", code_uri(CONFIG.bucket, SHA)]
    assert runner.calls[1] == planned.upload_command
    assert runner.calls[2] == planned.create_command


def test_execute_skips_the_upload_when_the_code_is_already_in_gcs(fake_runner):
    planned = submission()

    execute(planned, runner=fake_runner)

    assert fake_runner.calls == [
        ["gcloud", "storage", "ls", code_uri(CONFIG.bucket, SHA)],
        planned.create_command,
    ]


def test_git_head_strips_the_trailing_newline(make_runner):
    runner = make_runner(stdout=f"{SHA}\n")

    assert git_head("/repo", runner=runner) == SHA
    assert runner.calls == [["git", "rev-parse", "HEAD"]]


def test_worktree_is_clean_reads_git_status(make_runner):
    assert worktree_is_clean("/repo", runner=make_runner(stdout="")) is True
    assert worktree_is_clean("/repo", runner=make_runner(stdout=" M a.py\n")) is False


def test_the_follow_up_names_the_logs_the_run_and_the_cleanup(capsys):
    print_follow_up(CONFIG, RUN, SHA)

    printed = capsys.readouterr().out
    assert "gcloud compute ssh rsss-unet-01" in printed
    assert f"gs://{CONFIG.bucket}/runs/{RUN}/" in printed
    assert "scripts/gcp/pull_run.sh unet-01" in printed
    assert "gcloud compute instances delete rsss-unet-01" in printed


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("last", True),
    ],
)
def test_resume_reads_the_environment_as_a_boolean(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("RESUME", raising=False)
    else:
        monkeypatch.setenv("RESUME", value)

    args = build_arg_parser().parse_args([])

    assert args.resume is expected


def test_resume_rejects_an_ambiguous_environment_value(monkeypatch):
    monkeypatch.setenv("RESUME", "maybe")

    with pytest.raises(ValueError, match="RESUME must be a boolean"):
        build_arg_parser()


def test_resume_flag_still_works(monkeypatch):
    monkeypatch.delenv("RESUME", raising=False)

    assert build_arg_parser().parse_args(["--resume"]).resume is True
