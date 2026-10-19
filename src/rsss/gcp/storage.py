"""GCS object layout for the dataset, the code archive and the runs.

Keeping the URI construction here means the launcher and the on-VM job cannot
disagree about where a run lives.
"""

from __future__ import annotations

from rsss.gcp.shell import Runner, run


def run_uri(bucket: str, run_name: str) -> str:
    """Return the prefix holding one run's artefacts."""
    return f"gs://{bucket}/runs/{run_name}"


def data_uri(bucket: str, dataset_name: str) -> str:
    """Return the prefix holding one extracted dataset."""
    return f"gs://{bucket}/data/{dataset_name}"


def code_uri(bucket: str, sha: str) -> str:
    """Return the object holding the code archive of one commit."""
    return f"gs://{bucket}/code/{sha}.tar.gz"


def object_exists(uri: str, *, runner: Runner = run) -> bool:
    """Return whether at least one object lives under ``uri``."""
    result = runner(["gcloud", "storage", "ls", uri], check=False, capture_output=True)
    return result.returncode == 0
