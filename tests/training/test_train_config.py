"""Unit tests for ``rsss.training.config``.

Every case writes its own YAML to ``tmp_path``; one test loads the shipped
``configs/train.yaml`` so the committed recipe cannot rot unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rsss.training.config import load_train_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "train.yaml"
SHIPPED_DINOV3_CONFIG = REPO_ROOT / "configs" / "train_dinov3.yaml"

MINIMAL: dict = {"model": {"name": "unet", "encoder_weights": None}, "num_classes": 3}


@pytest.fixture
def write_config(tmp_path):
    """Return a writer for a YAML training config, from a mapping."""

    def _write(config: dict, name: str = "train.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(config))
        return path

    return _write


def test_minimal_config_leaves_the_optional_sections_empty(write_config):
    config = load_train_config(write_config(MINIMAL))

    assert config.model == MINIMAL["model"]
    assert config.num_classes == 3
    assert config.loss is None
    assert config.optimizer is None
    assert config.scheduler is None
    assert config.class_weights is None
    assert config.trainer == {}
    assert config.datamodule == {}


def test_sections_are_read_one_by_one(write_config):
    path = write_config(
        {
            **MINIMAL,
            "loss": {"ce": 0.25, "dice": 0.75},
            "optimizer": {"lr": 1e-3},
            "scheduler": {"warmup_epochs": 2},
            "class_weights": [1.0, 2.0, 3.0],
            "trainer": {"max_epochs": 5},
            "datamodule": {"batch_size": 8},
        }
    )

    config = load_train_config(path)

    assert config.loss == {"ce": 0.25, "dice": 0.75}
    assert config.optimizer == {"lr": 1e-3}
    assert config.scheduler == {"warmup_epochs": 2}
    assert config.class_weights == [1.0, 2.0, 3.0]
    assert config.trainer == {"max_epochs": 5}
    assert config.datamodule == {"batch_size": 8}


def test_the_shipped_config_loads():
    config = load_train_config(SHIPPED_CONFIG)

    assert config.num_classes == 13
    assert config.model["name"] == "unet"
    assert config.model["in_channels"] == 5
    # FLAIR#1's official per-band normalization, over the raw 0..255 values.
    assert config.datamodule["norm_type"] == "custom"
    assert len(config.datamodule["means"]) == 5
    assert len(config.datamodule["stds"]) == 5


def test_the_shipped_dinov3_config_loads():
    config = load_train_config(SHIPPED_DINOV3_CONFIG)

    assert config.num_classes == 13
    assert config.model["name"] == "dinov3_linear_probe"
    assert config.model["freeze_backbone"] is True
    assert config.model["train_stem"] is True
    assert config.optimizer is not None
    assert config.optimizer["stem_lr"] < config.optimizer["lr"]
    assert config.datamodule["norm_type"] == "custom"
    assert len(config.datamodule["means"]) == 5
    assert len(config.datamodule["stds"]) == 5


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Training config not found"):
        load_train_config(tmp_path / "absent.yaml")


def test_a_non_mapping_file_raises(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text("- not\n- a\n- mapping\n")

    with pytest.raises(TypeError, match="expected a YAML mapping"):
        load_train_config(path)


def test_unknown_key_is_rejected(write_config):
    path = write_config({**MINIMAL, "datadir": "data"})

    with pytest.raises(ValueError, match="unknown key"):
        load_train_config(path)


@pytest.mark.parametrize("missing", ["model", "num_classes"])
def test_missing_required_key_is_rejected(write_config, missing):
    config = {key: value for key, value in MINIMAL.items() if key != missing}

    with pytest.raises(ValueError, match="missing required key"):
        load_train_config(write_config(config))


def test_a_section_that_is_not_a_mapping_is_rejected(write_config):
    path = write_config({**MINIMAL, "loss": 0.5})

    with pytest.raises(TypeError, match="'loss' must be a mapping"):
        load_train_config(path)


def test_num_classes_must_be_an_integer(write_config):
    path = write_config({**MINIMAL, "num_classes": "3"})

    with pytest.raises(TypeError, match="'num_classes' must be an int"):
        load_train_config(path)


@pytest.mark.parametrize(
    "key", ["data_dir", "dataset", "download", "config", "num_classes"]
)
def test_the_datamodule_section_rejects_reserved_keys(write_config, key):
    path = write_config({**MINIMAL, "datamodule": {key: "whatever"}})

    with pytest.raises(ValueError, match="must not set"):
        load_train_config(path)
