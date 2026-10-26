"""FLAIR#1 train/validation/test split, read from the dataset's path CSVs.

Both the toy and the full FLAIR#1 dataset describe their split through path
lists shipped under ``CSVs/``: one CSV per split, each row an ``IMG,MSK`` pair
whose paths are relative to that directory. Reading those lists is what the
official baseline does (see the ``csv_toy/`` and ``csv_full/`` folders of
``IGNF/FLAIR-1``), so the same code ingests either dataset unchanged — only the
dataset root differs.

The lists are authoritative, not a convenience: the validation tiles live inside
the train collection directory (``flair_1_toy_aerial_train`` for the toy,
``flair_aerial_train`` for the full dataset), so no directory scan can tell them
apart from the train tiles.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CSV_SUBDIR = "CSVs"

# A CSV's split is the last "-"-separated token of its stem, so both
# "flair-1-paths-train.csv" and "flair-1-paths-toy-train.csv" mean train. "val"
# is exposed as "validation" to match the Lightning stage names.
_SPLIT_NAMES: dict[str, str] = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


def split_of_csv(csv_path: str | Path) -> str | None:
    """Return the split a FLAIR path CSV describes.

    Parameters
    ----------
    csv_path : str | Path
        A ``flair-1-paths-*.csv`` file, e.g. ``...-train.csv`` or
        ``...-toy-val.csv``.

    Returns
    -------
    str | None
        ``"train"``, ``"validation"`` or ``"test"``, or ``None`` when the file
        name does not end in a known split token.
    """
    return _SPLIT_NAMES.get(Path(csv_path).stem.rsplit("-", 1)[-1].lower())


def read_pairs_rows(csv_path: str | Path) -> list[tuple[str, str]]:
    """Read one paths CSV into its raw ``(IMG, MSK)`` rows.

    The rows keep the paths exactly as written, relative to the CSV's own
    directory; :func:`read_pairs_csv` is what resolves them. Filtering a split
    rewrites these rows (see :mod:`rsss.data.subset`), so they have to survive a
    read/write round trip unchanged.

    Parameters
    ----------
    csv_path : str | Path
        A FLAIR path CSV.

    Returns
    -------
    list[tuple[str, str]]
        One ``(image, mask)`` pair per non-empty row, in file order.

    Raises
    ------
    ValueError
        If a non-empty row does not hold exactly two fields.
    """
    path = Path(csv_path)
    rows: list[tuple[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(
                    f"{path}: expected 'IMG,MSK' rows, got {len(row)} fields: {row!r}."
                )
            rows.append((row[0].strip(), row[1].strip()))
    return rows


def read_pairs_csv(csv_path: str | Path) -> dict[str, list[str]]:
    """Read one paths CSV into index-aligned IMG and MSK lists.

    Paths are resolved relative to the CSV's own directory, which is where the
    datasets' ``../`` prefixes point.

    Parameters
    ----------
    csv_path : str | Path
        A FLAIR path CSV.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys "IMG" and "MSK"; ``IMG[i]`` and ``MSK[i]`` are the
        same tile.
    """
    base = Path(csv_path).parent
    rows = read_pairs_rows(csv_path)
    return {
        "IMG": [str((base / image).resolve()) for image, _ in rows],
        "MSK": [str((base / mask).resolve()) for _, mask in rows],
    }


def build_flair_splits(dataset_root: str | Path) -> dict[str, dict[str, list[str]]]:
    """Return the dataset's train/validation/test IMG/MSK path lists.

    Every CSV under ``<dataset_root>/CSVs`` is bucketed by the split its name
    ends in, so the toy and the full FLAIR#1 dataset go through the same call —
    only ``dataset_root`` changes.

    Parameters
    ----------
    dataset_root : str | Path
        Root of an extracted FLAIR#1 dataset (toy or full), holding ``CSVs/``
        and the ``flair*_aerial_*`` / ``flair*_labels_*`` collections the CSVs
        point at.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        One entry per split (``"train"``, ``"validation"``, ``"test"``), each
        mapping ``"IMG"`` and ``"MSK"`` to the file paths of that split.

    Raises
    ------
    FileNotFoundError
        If ``CSVs/`` is missing, or one of the three splits has no CSV.
    ValueError
        If two CSVs in ``CSVs/`` describe the same split.
    """
    csv_dir = Path(dataset_root) / CSV_SUBDIR
    if not csv_dir.is_dir():
        raise FileNotFoundError(
            f"No {CSV_SUBDIR!r} directory under {str(dataset_root)!r}: expected an "
            "extracted FLAIR#1 dataset, whose CSVs hold the split."
        )

    splits: dict[str, dict[str, list[str]]] = {}
    for csv_path in sorted(csv_dir.glob("*.csv")):
        split = split_of_csv(csv_path)
        if split is None:
            logger.warning(
                "Ignoring %s: its name does not end in -train/-val/-test.",
                csv_path.name,
            )
            continue
        if split in splits:
            raise ValueError(
                f"Two CSVs describe the '{split}' split under {str(csv_dir)!r}: "
                "keep one CSV per split."
            )
        splits[split] = read_pairs_csv(csv_path)

    missing = sorted(set(_SPLIT_NAMES.values()) - splits.keys())
    if missing:
        raise FileNotFoundError(
            f"No CSV for the {missing} split(s) under {str(csv_dir)!r}."
        )
    return splits
