"""Unit tests for the rendered VM startup script."""

from __future__ import annotations

from rsss.gcp.startup import render_startup_script


def script(
    *,
    epochs: int | None = None,
    resume: bool = False,
    limit_train_batches: float | None = None,
    data_config: str = "configs/data_full.yaml",
) -> str:
    """Render a startup script for a fixed run, with the options a test varies."""
    return render_startup_script(
        bucket="rsss-data-bucket",
        run_name="unet-20261019-120000",
        sha="0" * 40,
        train_config="configs/train.yaml",
        data_config=data_config,
        epochs=epochs,
        resume=resume,
        limit_train_batches=limit_train_batches,
    )


def test_the_script_runs_the_job_with_the_run_arguments():
    text = script(epochs=30)

    assert "python -m rsss.gcp.job" in text
    assert "--bucket rsss-data-bucket" in text
    assert "--run unet-20261019-120000" in text
    assert "--epochs 30" in text
    assert "gs://$BUCKET/code/$SHA.tar.gz" in text


def test_the_script_uploads_the_boot_log_and_shuts_down_even_on_failure():
    text = script()

    assert "trap cleanup EXIT" in text
    assert "shutdown -h now" in text
    assert "runs/$RUN/logs/startup.log" in text


def test_the_optional_flags_are_omitted_by_default():
    text = script()

    assert "--epochs" not in text
    assert "--resume" not in text
    assert "--limit-train-batches" not in text


def test_the_optional_flags_are_added_when_asked_for():
    text = script(resume=True, limit_train_batches=2.0)

    assert "    --resume\n" in text
    assert "    --limit-train-batches 2.0\n" in text


def test_the_cuda_check_runs_before_training():
    text = script()

    assert "torch.cuda.is_available()" in text
    assert text.index("torch.cuda.is_available()") < text.index("rsss.gcp.job")


def test_paths_with_spaces_are_shell_quoted():
    text = script(data_config="my configs/data.yaml")

    assert "--data-config 'my configs/data.yaml'" in text
