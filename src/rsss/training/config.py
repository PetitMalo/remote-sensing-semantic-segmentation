"""YAML training configuration for the segmentation pipeline.

A ``configs/train.yaml`` names the model, the loss, the optimizer, the schedule
and the Trainer/DataModule settings, so no hyperparameter lives as a default
scattered across the training script.

Only the top level is validated. The nested mappings are forwarded as they are
to :func:`rsss.models.build_model`, :class:`~rsss.training.module.SegmentationModule`
and the Lightning ``Trainer``, which already reject what they cannot use and
accept far more options than a schema here should mirror.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Sections forwarded to a component, plus the two scalar keys.
_NESTED_SECTIONS = ("model", "loss", "optimizer", "scheduler", "trainer", "datamodule")
_ALLOWED_KEYS = frozenset(_NESTED_SECTIONS) | {"num_classes", "class_weights"}
_REQUIRED_KEYS = frozenset({"model", "num_classes"})
# Keys owned by the data config: ``FlairDatamodule`` refuses them alongside a
# ``config``, so a ``datamodule`` section carrying one could never work.
_DATA_KEYS = frozenset({"config", "data_dir", "dataset", "download"})


@dataclass(frozen=True)
class TrainConfig:
    """The training recipe, as read from a ``configs/train.yaml``.

    Parameters
    ----------
    model : Mapping[str, Any]
        Forwarded to :func:`rsss.models.build_model`: a ``"name"`` key plus the
        builder's keyword arguments.
    num_classes : int
        Number of classes, shared by the model head and the metrics.
    loss, optimizer, scheduler : Mapping, optional
        Forwarded to :class:`~rsss.training.module.SegmentationModule`.
    class_weights : Sequence[float], optional
        Per-class weights of the cross-entropy term, to counter the imbalance.
    trainer : Mapping[str, Any]
        Keyword arguments for the Lightning ``Trainer``.
    datamodule : Mapping[str, Any]
        Keyword arguments for ``FlairDatamodule``, minus the data-config keys.
    """

    model: Mapping[str, Any]
    num_classes: int
    loss: Mapping[str, float] | None = None
    optimizer: Mapping[str, Any] | None = None
    scheduler: Mapping[str, Any] | None = None
    class_weights: Sequence[float] | None = None
    trainer: Mapping[str, Any] = field(default_factory=dict)
    datamodule: Mapping[str, Any] = field(default_factory=dict)


def load_train_config(path: str | Path) -> TrainConfig:
    """Read a training YAML into a :class:`TrainConfig`.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file.

    Returns
    -------
    TrainConfig
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If the file or one of its sections is not a YAML mapping, or if
        ``num_classes`` is not an integer.
    ValueError
        If the file holds unknown keys, is missing a required key, or sets one
        of the data-config keys in its ``datamodule`` section.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
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

    for section in _NESTED_SECTIONS:
        if section in raw and not isinstance(raw[section], dict):
            raise TypeError(
                f"{path}: '{section}' must be a mapping, got "
                f"{type(raw[section]).__name__}."
            )

    num_classes = raw["num_classes"]
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError(f"{path}: 'num_classes' must be an int, got {num_classes!r}.")

    datamodule = raw.get("datamodule", {})
    data_keys = sorted(_DATA_KEYS & set(datamodule))
    if data_keys:
        raise ValueError(
            f"{path}: the 'datamodule' section must not set {data_keys}; they come "
            "from the data config, and FlairDatamodule refuses both at once."
        )

    return TrainConfig(
        model=raw["model"],
        num_classes=num_classes,
        loss=raw.get("loss"),
        optimizer=raw.get("optimizer"),
        scheduler=raw.get("scheduler"),
        class_weights=raw.get("class_weights"),
        trainer=raw.get("trainer", {}),
        datamodule=datamodule,
    )
