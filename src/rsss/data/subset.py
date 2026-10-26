"""Whole-domain subsets of FLAIR#1: pick domains, filter the path CSVs.

A full FLAIR#1 tree is ~95 GiB, more than some cluster accounts hold, so the
project stages a subset instead: entire spatial domains are kept, per split.
Domains are ordered alphabetically, so ``SubsetSpec(train=3)`` always means the
same 3 domains, on any machine and in any run — U-Net and DINO comparisons then
score the identical tiles. :func:`build_subset_csvs` writes the filtered path
CSVs and :func:`write_manifest` records the choice under the subset root.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsss.data.splits import CSV_SUBDIR, read_pairs_rows

# Directory the subset is assembled into, and the file recording its selection.
SUBSET_ROOT_NAME = "flair_1_subset_dataset"
MANIFEST_NAME = "subset_manifest.json"
# The spec's field names double as the split names :func:`build_flair_splits`
# returns, so the two stay in step without a translation table.
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SubsetSpec:
    """How many domains of each split a subset keeps.

    The defaults are the recipe that fits a 25 GiB staging budget: the first 3
    of the 32 train domains (5,275 tiles), the first 2 of the 8 validation
    domains (3,550) and the first 2 of the 10 test domains (2,250), 11,075 tiles
    and 13.6 GiB in total. A split with fewer domains than its limit keeps all
    of them.

    Parameters
    ----------
    train, validation, test : int
        Domains kept in each split, alphabetically first.
    """

    train: int = 3
    validation: int = 2
    test: int = 2

    def __post_init__(self) -> None:
        for split in SPLIT_NAMES:
            if getattr(self, split) < 1:
                raise ValueError(
                    f"SubsetSpec.{split} must be at least 1, "
                    f"got {getattr(self, split)}."
                )

    def as_dict(self) -> dict[str, int]:
        """Return the per-split domain counts, keyed by split name."""
        return {split: getattr(self, split) for split in SPLIT_NAMES}

    @classmethod
    def from_text(cls, text: str) -> SubsetSpec:
        """Parse ``TRAIN,VALIDATION,TEST`` domain counts, e.g. ``"3,2,2"``.

        Parameters
        ----------
        text : str
            Three comma-separated positive integers.

        Raises
        ------
        ValueError
            If the text does not hold exactly three positive integers.
        """
        fields = [field.strip() for field in text.split(",")]
        if len(fields) != len(SPLIT_NAMES) or not all(
            field.isdigit() for field in fields
        ):
            raise ValueError(
                f"Expected {len(SPLIT_NAMES)} comma-separated domain counts "
                f"(train,validation,test), got {text!r}."
            )
        train, validation, test = (int(field) for field in fields)
        return cls(train=train, validation=validation, test=test)

    def as_text(self) -> str:
        """Return the spec as ``TRAIN,VALIDATION,TEST`` for logs and the CLI."""
        return ",".join(str(count) for count in self.as_dict().values())


@dataclass(frozen=True)
class SplitSelection:
    """The rows and domains a spec kept for one split.

    Parameters
    ----------
    rows : list[tuple[str, str]]
        Kept ``(IMG, MSK)`` rows, as written in the source CSV.
    domains : list[str]
        Domains kept, alphabetically sorted.
    available : list[str]
        Every domain the source CSV offered, alphabetically sorted; it tells a
        report how much room a larger spec has.
    """

    rows: list[tuple[str, str]]
    domains: list[str]
    available: list[str]


def domain_of(path: str | Path) -> str:
    """Return the FLAIR domain a tile path belongs to.

    Both layouts store it four components from the end: the train and
    validation collections as ``flair_*_aerial_train/<domain>/<zone>/img/IMG``,
    the test collection as ``data/flair_1_aerial_test/<domain>/...``.

    Parameters
    ----------
    path : str | Path
        A tile path, relative or absolute.

    Raises
    ------
    ValueError
        If the path is too short to hold a domain component.
    """
    parts = Path(path).parts
    if len(parts) < 4:
        raise ValueError(f"{path}: too few components to hold a domain.")
    return parts[-4]


def kept_domains_by_collection(
    paths: Iterable[str | Path], root: str | Path
) -> dict[Path, set[str]]:
    """Group the domains a set of tile paths names by their collection directory.

    FLAIR#1 lays tiles out as ``<collection>/<domain>/<zone>/<kind>/<file>``, so
    the collection is everything before the domain component. Reading the
    collections off the paths keeps pruning (see
    :func:`rsss.data.download.prune_out_of_spec_domains`) from hard-coding the
    train and test layout differences.

    Parameters
    ----------
    paths : Iterable[str | Path]
        Tile paths, absolute or relative to ``root``.
    root : str | Path
        Directory the paths are relative to (the subset root).

    Returns
    -------
    dict[Path, set[str]]
        Collection directory, relative to ``root``, to the domains kept in it.

    Raises
    ------
    ValueError
        If a path is outside ``root`` or too short to hold a collection.
    """
    base = Path(root).resolve()
    kept: dict[Path, set[str]] = {}
    for path in paths:
        relative = Path(path).resolve().relative_to(base)
        if len(relative.parts) < 5:
            raise ValueError(f"{path}: too few components to hold a collection.")
        kept.setdefault(Path(*relative.parts[:-4]), set()).add(relative.parts[-4])
    return kept


def select_rows(rows: list[tuple[str, str]], limit: int) -> SplitSelection:
    """Keep the rows of the alphabetically first ``limit`` domains.

    Parameters
    ----------
    rows : list[tuple[str, str]]
        Raw ``(IMG, MSK)`` rows of one split.
    limit : int
        Domains to keep; above the available count, everything is kept.

    Raises
    ------
    ValueError
        If ``limit`` is below 1.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}.")
    available = sorted({domain_of(image) for image, _ in rows})
    kept = set(available[:limit])
    return SplitSelection(
        rows=[row for row in rows if domain_of(row[0]) in kept],
        domains=sorted(kept),
        available=available,
    )


def write_split_csv(dest: str | Path, rows: list[tuple[str, str]]) -> Path:
    """Write ``rows`` as a FLAIR path CSV, creating parent directories."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return dest


def build_subset_csvs(
    source_csvs: Mapping[str, Path],
    root: str | Path,
    spec: SubsetSpec,
) -> dict[str, SplitSelection]:
    """Filter each split's official CSV into ``<root>/CSVs``.

    The filtered files keep the official names (``flair-1-paths-train.csv``...),
    so :func:`rsss.data.splits.build_flair_splits` and the verification only see
    a smaller FLAIR#1 tree, with nothing to special-case.

    Parameters
    ----------
    source_csvs : Mapping[str, Path]
        Split name (``train``, ``validation``, ``test``) to its official CSV.
    root : str | Path
        Subset root to write the CSVs under.
    spec : SubsetSpec
        Domains to keep per split.

    Returns
    -------
    dict[str, SplitSelection]
        What was kept, per split.

    Raises
    ------
    ValueError
        If ``source_csvs`` names a split the spec does not describe.
    """
    counts = spec.as_dict()
    csv_dir = Path(root) / CSV_SUBDIR
    selections: dict[str, SplitSelection] = {}
    for split, source in source_csvs.items():
        if split not in counts:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}.")
        selection = select_rows(read_pairs_rows(source), counts[split])
        write_split_csv(csv_dir / Path(source).name, selection.rows)
        selections[split] = selection
    return selections


def write_manifest(
    root: str | Path,
    spec: SubsetSpec,
    selections: Mapping[str, SplitSelection],
) -> Path:
    """Record the strategy, the spec and what each split kept under ``root``.

    The manifest is what lets later runs tell a subset staged with one spec from
    one staged with another; the CSVs alone cannot say how many domains the
    source offered.
    """
    payload = {
        "strategy": "domains",
        "spec": spec.as_dict(),
        "splits": {
            split: {
                "domains": selection.domains,
                "available": selection.available,
                "tiles": len(selection.rows),
            }
            for split, selection in selections.items()
        },
    }
    path = Path(root) / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(root: str | Path) -> dict[str, Any] | None:
    """Return the subset manifest under ``root``, or None when there is none."""
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def manifest_matches(manifest: Mapping[str, Any], spec: SubsetSpec) -> bool:
    """Return whether a manifest records ``spec`` (strategy and counts)."""
    return (
        manifest.get("strategy") == "domains" and manifest.get("spec") == spec.as_dict()
    )
