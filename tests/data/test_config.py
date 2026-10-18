"""Unit tests for ``rsss.data.config``.

Everything runs against small YAML files written to ``tmp_path``; no real
config or dataset is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rsss.data.config import DataConfig, load_data_config
from rsss.data.download import TOY_DATASET


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "data.yaml"
    path.write_text(text)
    return path


def test_load_data_config_reads_the_file(tmp_path):
    path = _write(
        tmp_path,
        "data_dir: scratch\ndataset: flair_1_toy\ndownload: true\n",
    )

    config = load_data_config(path)

    assert config == DataConfig(
        data_dir=Path("scratch"), dataset="flair_1_toy", download=True
    )


def test_load_data_config_defaults_download_to_false(tmp_path):
    path = _write(tmp_path, "data_dir: data\ndataset: flair_1_toy\n")

    config = load_data_config(path)

    assert config.download is False


def test_config_exposes_the_remote_and_dataset_dir(tmp_path):
    config = DataConfig(data_dir=Path("data"), dataset="flair_1_toy")

    assert config.remote is TOY_DATASET
    assert config.dataset_dir == Path("data") / TOY_DATASET.name


def test_load_data_config_rejects_unknown_keys(tmp_path):
    path = _write(tmp_path, "data_dir: data\ndataset: flair_1_toy\ndatadir: oops\n")

    with pytest.raises(ValueError, match="unknown key"):
        load_data_config(path)


def test_load_data_config_rejects_missing_keys(tmp_path):
    path = _write(tmp_path, "dataset: flair_1_toy\n")

    with pytest.raises(ValueError, match="missing required key"):
        load_data_config(path)


def test_load_data_config_rejects_unknown_datasets(tmp_path):
    path = _write(tmp_path, "data_dir: data\ndataset: nope\n")

    with pytest.raises(ValueError, match="unknown dataset"):
        load_data_config(path)


def test_load_data_config_rejects_non_mappings(tmp_path):
    path = _write(tmp_path, "- data_dir\n- dataset\n")

    with pytest.raises(TypeError, match="expected a YAML mapping"):
        load_data_config(path)


def test_load_data_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Data config not found"):
        load_data_config(tmp_path / "absent.yaml")


def test_shipped_config_loads():
    """The committed ``configs/data.yaml`` must stay loadable."""
    config = load_data_config(Path(__file__).parents[2] / "configs/data.yaml")

    assert config.dataset in {"flair_1_toy"}
    assert config.dataset_dir.name == TOY_DATASET.name
