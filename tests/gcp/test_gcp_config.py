"""Unit tests for the GCP host configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rsss.gcp.config import load_gcp_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "gcp.yaml"

MINIMAL = {"project": "rsss-project", "zone": "europe-west4-a", "bucket": "b"}


@pytest.fixture
def write_config(tmp_path):
    """Return a writer for a YAML GCP config, from a mapping."""

    def _write(config: dict, name: str = "gcp.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(config))
        return path

    return _write


def test_minimal_config_uses_the_defaults(write_config):
    config = load_gcp_config(write_config(MINIMAL))

    assert config.project == "rsss-project"
    assert config.zone == "europe-west4-a"
    assert config.bucket == "b"
    assert config.machine_type == "g2-standard-8"
    assert config.image_family == "common-cu130-ubuntu-2204"
    assert config.boot_disk_size_gb == 200
    assert config.max_run_duration == "24h"


def test_the_shipped_config_loads():
    config = load_gcp_config(SHIPPED_CONFIG)

    assert config.project == "rsss-project"
    assert config.bucket == "rsss-data-bucket"
    # The bucket has to live in the zone's region for free, local transfers.
    assert config.region == "europe-west4"
    # CUDA 13 wheels need the 580+ driver the cu130 image family ships.
    assert "cu130" in config.image_family


def test_the_service_account_defaults_to_the_project(write_config):
    config = load_gcp_config(write_config(MINIMAL))

    assert config.vm_service_account == "rsss-vm@rsss-project.iam.gserviceaccount.com"


def test_an_explicit_service_account_is_kept(write_config):
    path = write_config({**MINIMAL, "service_account": "custom@example.com"})

    assert load_gcp_config(path).vm_service_account == "custom@example.com"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="GCP config not found"):
        load_gcp_config(tmp_path / "absent.yaml")


def test_a_non_mapping_file_raises(tmp_path):
    path = tmp_path / "gcp.yaml"
    path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(TypeError, match="expected a YAML mapping"):
        load_gcp_config(path)


def test_unknown_key_is_rejected(write_config):
    path = write_config({**MINIMAL, "region": "europe-west4"})

    with pytest.raises(ValueError, match="unknown key"):
        load_gcp_config(path)


@pytest.mark.parametrize("missing", ["project", "zone", "bucket"])
def test_missing_required_key_is_rejected(write_config, missing):
    config = {key: value for key, value in MINIMAL.items() if key != missing}

    with pytest.raises(ValueError, match="missing required key"):
        load_gcp_config(write_config(config))


def test_a_required_key_must_be_a_non_empty_string(write_config):
    with pytest.raises(TypeError, match="'bucket' must be a non-empty string"):
        load_gcp_config(write_config({**MINIMAL, "bucket": ""}))


def test_the_boot_disk_size_must_be_a_positive_integer(write_config):
    with pytest.raises(TypeError, match="must be an int"):
        load_gcp_config(write_config({**MINIMAL, "boot_disk_size_gb": "200"}))

    with pytest.raises(ValueError, match="must be positive"):
        load_gcp_config(write_config({**MINIMAL, "boot_disk_size_gb": 0}))
