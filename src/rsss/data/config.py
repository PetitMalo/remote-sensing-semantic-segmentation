"""YAML data configuration for the segmentation pipeline.

A small ``configs/data.yaml`` names where datasets live, which one to use, and
whether it should be fetched. :class:`FlairDatamodule` reads it, so those
choices stop living as constructor defaults scattered across notebooks and
training scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from rsss.data.download import DATASETS, Dataset

_CONFIG_KEYS = frozenset({"data_dir", "dataset", "download"})
_REQUIRED_KEYS = frozenset({"data_dir", "dataset"})


@dataclass(frozen=True)
class DataConfig:
    """Where datasets live, which one to use, and whether to fetch it.

    Parameters
    ----------
    data_dir : Path
        Directory datasets are extracted into (the repo's ``data/``).
    dataset : str
        Key into :data:`rsss.data.download.DATASETS`.
    download : bool
        Whether to fetch ``dataset`` when it is missing.
    """

    data_dir: Path
    dataset: str
    download: bool = False

    @property
    def remote(self) -> Dataset:
        """The downloadable dataset ``dataset`` names."""
        return DATASETS[self.dataset]

    @property
    def dataset_dir(self) -> Path:
        """Root of the extracted dataset, as the datamodule reads it."""
        return self.data_dir / self.remote.name


def load_data_config(path: str | Path) -> DataConfig:
    """Read a data YAML into a :class:`DataConfig`.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file.

    Returns
    -------
    DataConfig
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If the file is not a YAML mapping.
    ValueError
        If the file holds unknown keys, is missing a required key, or names an
        unknown dataset. Unknown keys are rejected so a typo (``datadir``)
        fails loudly rather than being silently ignored.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data config not found: {path}")
    raw = yaml.safe_load(path.read_text())

    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}.")
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown}; allowed: {sorted(_CONFIG_KEYS)}."
        )
    missing = sorted(_REQUIRED_KEYS - set(raw))
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}.")

    dataset = raw["dataset"]
    if dataset not in DATASETS:
        raise ValueError(
            f"{path}: unknown dataset {dataset!r}; known: {sorted(DATASETS)}."
        )
    return DataConfig(
        data_dir=Path(raw["data_dir"]),
        dataset=dataset,
        download=bool(raw.get("download", False)),
    )
