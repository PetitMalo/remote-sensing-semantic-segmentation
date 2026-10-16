"""Unit tests for ``rsss.data.splits``.

The official FLAIR#1 domain lists are part of the module's contract, so a few
tests assert invariants over ``FLAIR_1_SPLIT`` itself; the rest run against a
synthetic tree written to ``tmp_path`` via the shared ``write_tile`` fixture.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from rsss.data.splits import (
    FLAIR_1_SPLIT,
    build_flair_splits,
    domain_of,
    split_of_domain,
)

# --------------------------------------------------------------------------- #
# domain_of                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("D004_2021", "D004"),
        ("D006_2020", "D006"),
        ("D012", "D012"),
        ("D004_2021_extra", "D004"),
    ],
)
def test_domain_of_returns_prefix_before_year(name, expected):
    assert domain_of(name) == expected


# --------------------------------------------------------------------------- #
# split_of_domain                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        (domain, split)
        for split, domains in FLAIR_1_SPLIT.items()
        for domain in sorted(domains)
    ],
)
def test_split_of_domain_maps_every_official_domain(domain, expected):
    assert split_of_domain(domain) == expected


def test_split_of_domain_returns_none_for_unknown_domain():
    assert split_of_domain("D999") is None


# --------------------------------------------------------------------------- #
# FLAIR_1_SPLIT invariants                                                    #
# --------------------------------------------------------------------------- #


def test_official_splits_use_the_expected_names():
    assert set(FLAIR_1_SPLIT) == {"train", "validation", "test"}


def test_official_splits_do_not_share_domains():
    all_domains = [domain for domains in FLAIR_1_SPLIT.values() for domain in domains]

    assert len(all_domains) == len(set(all_domains))


# --------------------------------------------------------------------------- #
# build_flair_splits                                                          #
# --------------------------------------------------------------------------- #


def _areas_of(paths: list[str]) -> list[str]:
    """Return the area directory (4th path component from the end) of each path."""
    return [Path(path).parts[-4] for path in paths]


def test_build_flair_splits_buckets_areas_by_official_split(tmp_path, write_tile):
    root = tmp_path / "data"
    write_tile(root, "D006_2020", "Z1", ["000001"], ["000001"])  # train
    write_tile(root, "D004_2021", "Z1", ["000001"], ["000001"])  # validation
    write_tile(root, "D012_2019", "Z1", ["000001"], ["000001"])  # test

    splits = build_flair_splits(str(root))

    assert _areas_of(splits["train"]["IMG"]) == ["D006_2020"]
    assert _areas_of(splits["validation"]["IMG"]) == ["D004_2021"]
    assert _areas_of(splits["test"]["IMG"]) == ["D012_2019"]
    for contents in splits.values():
        assert _areas_of(contents["IMG"]) == _areas_of(contents["MSK"])


def test_build_flair_splits_keeps_img_and_msk_aligned(tmp_path, write_tile):
    root = tmp_path / "data"
    write_tile(root, "D006_2020", "Z1", ["000002", "000001"], ["000001", "000002"])

    train = build_flair_splits(str(root))["train"]

    assert len(train["IMG"]) == 2
    for img, msk in zip(train["IMG"], train["MSK"], strict=True):
        assert Path(img).name[len("IMG_") :] == Path(msk).name[len("MSK_") :]


def test_build_flair_splits_returns_empty_lists_when_no_data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    splits = build_flair_splits(str(root))

    assert splits == {
        "train": {"IMG": [], "MSK": []},
        "validation": {"IMG": [], "MSK": []},
        "test": {"IMG": [], "MSK": []},
    }


def test_build_flair_splits_drops_unknown_domain_with_warning(
    tmp_path, write_tile, caplog
):
    root = tmp_path / "data"
    write_tile(root, "D999_2020", "Z1", ["000001"], ["000001"])

    with caplog.at_level(logging.WARNING):
        splits = build_flair_splits(str(root))

    assert all(not contents["IMG"] for contents in splits.values())
    assert "D999_2020" in caplog.text
    assert "not part of the official FLAIR#1 split" in caplog.text


def test_build_flair_splits_warns_when_a_split_has_no_local_data(
    tmp_path, write_tile, caplog
):
    root = tmp_path / "data"
    write_tile(root, "D006_2020", "Z1", ["000001"], ["000001"])

    with caplog.at_level(logging.WARNING):
        build_flair_splits(str(root))

    assert "No local data found for the 'validation' split." in caplog.text
    assert "No local data found for the 'test' split." in caplog.text
