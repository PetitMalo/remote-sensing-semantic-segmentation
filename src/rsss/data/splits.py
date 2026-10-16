"""Official FLAIR#1 train/validation/test domain split.

The FLAIR dataset (French Land cover from Aerial Imagery) is partitioned by
spatial domain, aligned with French departement boundaries. The lists below
come from the FLAIR datapaper (NeurIPS 2023 Datasets & Benchmarks, appendix
A.6.1) and are matched on the domain code (the ``D0xx`` prefix of an area name
such as ``D004_2021``), so the mapping does not depend on the capture year.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rsss.data.data_loader import discover_data

logger = logging.getLogger(__name__)

FLAIR_1_SPLIT: dict[str, set[str]] = {
    "train": {
        "D006",
        "D007",
        "D008",
        "D009",
        "D013",
        "D016",
        "D017",
        "D021",
        "D023",
        "D030",
        "D032",
        "D033",
        "D034",
        "D035",
        "D038",
        "D041",
        "D044",
        "D046",
        "D049",
        "D051",
        "D052",
        "D055",
        "D060",
        "D063",
        "D070",
        "D072",
        "D074",
        "D078",
        "D080",
        "D081",
        "D086",
        "D091",
    },
    "validation": {
        "D004",
        "D014",
        "D029",
        "D031",
        "D058",
        "D066",
        "D067",
        "D077",
    },
    "test": {
        "D012",
        "D022",
        "D026",
        "D064",
        "D068",
        "D071",
        "D075",
        "D076",
        "D083",
        "D085",
    },
}

_DOMAIN_TO_SPLIT: dict[str, str] = {
    domain: split for split, domains in FLAIR_1_SPLIT.items() for domain in domains
}


def domain_of(name: str) -> str:
    """Return the FLAIR domain code of an area name.

    Parameters
    ----------
    name : str
        Area name, e.g. ``"D004_2021"``.

    Returns
    -------
    str
        Domain code, e.g. ``"D004"``.
    """
    return name.split("_", 1)[0]


def split_of_domain(domain: str) -> str | None:
    """Return the official split name for a domain code.

    Parameters
    ----------
    domain : str
        Domain code, e.g. ``"D004"``.

    Returns
    -------
    str | None
        One of ``"train"``, ``"validation"``, ``"test"``, or ``None`` when the
        domain is not part of the official FLAIR#1 split.
    """
    return _DOMAIN_TO_SPLIT.get(domain)


def build_flair_splits(data_dir: str) -> dict[str, dict[str, list[str]]]:
    """Bucket the locally available tiles into the official FLAIR#1 splits.

    Parameters
    ----------
    data_dir : str
        Root directory containing area subdirectories (e.g. "data/").

    Returns
    -------
    dict[str, dict[str, list[str]]]
        One entry per split (``"train"``, ``"validation"``, ``"test"``), each
        mapping ``"IMG"`` and ``"MSK"`` to the sorted file paths of that split.

    Notes
    -----
    Only a subset of the official domains may be present locally. A warning is
    logged for every local domain the official mapping does not know (its tiles
    are dropped, never mixed into a split) and for every official split that
    has no local data.
    """
    root = Path(data_dir)
    files = discover_data(data_dir)

    splits: dict[str, dict[str, list[str]]] = {
        split: {"IMG": [], "MSK": []} for split in FLAIR_1_SPLIT
    }

    unknown_areas: set[str] = set()
    for kind in ("IMG", "MSK"):
        for path in files[kind]:
            area = Path(path).relative_to(root).parts[0]
            split = split_of_domain(domain_of(area))
            if split is None:
                unknown_areas.add(area)
                continue
            splits[split][kind].append(path)

    for area in sorted(unknown_areas):
        logger.warning(
            "Area %s (domain %s) is not part of the official FLAIR#1 split; "
            "its tiles are ignored.",
            area,
            domain_of(area),
        )

    for split, contents in splits.items():
        if not contents["IMG"]:
            logger.warning("No local data found for the '%s' split.", split)

    return splits
