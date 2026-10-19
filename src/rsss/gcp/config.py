"""GCP host configuration for a training run.

``configs/gcp.yaml`` names the project, zone, bucket, image and disk, so the
launcher holds no cloud settings of its own. The training recipe and the data
source stay in ``configs/train.yaml`` and ``configs/data.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/gcp.yaml"

_ALLOWED_KEYS = frozenset(
    {
        "project",
        "zone",
        "bucket",
        "machine_type",
        "image_family",
        "image_project",
        "boot_disk_size_gb",
        "boot_disk_type",
        "service_account",
        "max_run_duration",
    }
)
_REQUIRED_KEYS = frozenset({"project", "zone", "bucket"})


@dataclass(frozen=True)
class GcpConfig:
    """Where and how a training VM is created.

    Parameters
    ----------
    project : str
        GCP project that owns the VM and the bucket.
    zone : str
        Zone the VM runs in, e.g. ``europe-west4-a``.
    bucket : str
        Bucket that holds the dataset, the code archive and the runs.
    machine_type : str
        Machine type. ``g2-standard-8`` bundles one NVIDIA L4.
    image_family, image_project : str
        Deep Learning VM image. The ``gcp`` dependency group installs CUDA 13
        wheels, so the image must ship a 580+ driver.
    boot_disk_size_gb : int
        Boot disk size, large enough for the extracted dataset.
    boot_disk_type : str
        Boot disk type, e.g. ``pd-balanced``.
    service_account : str, optional
        Account attached to the VM; defaults to ``rsss-vm@<project>``.
    max_run_duration : str
        Cost guard, passed to ``gcloud compute instances create``.
    """

    project: str
    zone: str
    bucket: str
    machine_type: str = "g2-standard-8"
    image_family: str = "common-cu130-ubuntu-2204"
    image_project: str = "deeplearning-platform-release"
    boot_disk_size_gb: int = 200
    boot_disk_type: str = "pd-balanced"
    service_account: str | None = None
    max_run_duration: str = "24h"

    @property
    def region(self) -> str:
        """Region of :attr:`zone`, where the bucket should live."""
        return self.zone.rsplit("-", 1)[0]

    @property
    def vm_service_account(self) -> str:
        """Service account the VM runs as."""
        if self.service_account:
            return self.service_account
        return f"rsss-vm@{self.project}.iam.gserviceaccount.com"


def load_gcp_config(path: str | Path = DEFAULT_CONFIG) -> GcpConfig:
    """Read a GCP YAML into a :class:`GcpConfig`.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file.

    Returns
    -------
    GcpConfig
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If the file is not a YAML mapping or a value has the wrong type.
    ValueError
        If the file holds unknown keys or misses a required one.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GCP config not found: {path}")
    raw = yaml.safe_load(path.read_text())

    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}.")
    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown}; allowed: {sorted(_ALLOWED_KEYS)}."
        )
    missing = sorted(_REQUIRED_KEYS - set(raw))
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}.")

    for key in _REQUIRED_KEYS:
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise TypeError(f"{path}: '{key}' must be a non-empty string.")

    disk_size = raw.get("boot_disk_size_gb", 200)
    if not isinstance(disk_size, int) or isinstance(disk_size, bool):
        raise TypeError(
            f"{path}: 'boot_disk_size_gb' must be an int, got {disk_size!r}."
        )
    if disk_size <= 0:
        raise ValueError(f"{path}: 'boot_disk_size_gb' must be positive.")

    return GcpConfig(
        project=raw["project"],
        zone=raw["zone"],
        bucket=raw["bucket"],
        machine_type=raw.get("machine_type", "g2-standard-8"),
        image_family=raw.get("image_family", "common-cu130-ubuntu-2204"),
        image_project=raw.get("image_project", "deeplearning-platform-release"),
        boot_disk_size_gb=disk_size,
        boot_disk_type=raw.get("boot_disk_type", "pd-balanced"),
        service_account=raw.get("service_account"),
        max_run_duration=raw.get("max_run_duration", "24h"),
    )
