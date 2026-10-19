"""Unit tests for the GCS layout and the ``gcloud storage ls`` probe."""

from __future__ import annotations

from rsss.gcp.storage import code_uri, data_uri, object_exists, run_uri


def test_uris_lay_everything_under_one_bucket():
    assert run_uri("b", "unet-01") == "gs://b/runs/unet-01"
    assert data_uri("b", "flair_1_full_dataset") == "gs://b/data/flair_1_full_dataset"
    assert code_uri("b", "abc123") == "gs://b/code/abc123.tar.gz"


def test_object_exists_is_true_when_gcloud_succeeds(fake_runner):
    assert object_exists(
        "gs://b/runs/unet-01/checkpoints/last.ckpt", runner=fake_runner
    )
    assert fake_runner.calls == [
        ["gcloud", "storage", "ls", "gs://b/runs/unet-01/checkpoints/last.ckpt"]
    ]


def test_object_exists_is_false_when_gcloud_matches_nothing(make_runner):
    uri = "gs://b/data/flair_1_full_dataset/CSVs"
    runner = make_runner(missing=[uri])

    assert not object_exists(uri, runner=runner)
