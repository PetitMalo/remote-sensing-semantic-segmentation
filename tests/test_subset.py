"""Tests for whole-domain subset selection, its CSVs and its manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsss.data.splits import build_flair_splits, read_pairs_rows
from rsss.data.subset import (
    MANIFEST_NAME,
    SplitSelection,
    SubsetSpec,
    build_subset_csvs,
    domain_of,
    manifest_matches,
    read_manifest,
    select_rows,
    write_manifest,
    write_split_csv,
)


def _row(domain: str, stem: str = "000001") -> tuple[str, str]:
    """Return a raw CSV row of the train collection for ``domain``."""
    return (
        f"../flair_aerial_train/{domain}/Z1_UU/img/IMG_{stem}.tif",
        f"../flair_labels_train/{domain}/Z1_UU/msk/MSK_{stem}.tif",
    )


def test_domain_of_reads_the_train_and_the_test_layouts() -> None:
    assert (
        domain_of("../flair_aerial_train/D006_2020/Z10_UU/img/IMG_002926.tif")
        == "D006_2020"
    )
    assert (
        domain_of("../data/flair_1_aerial_test/D012_2019/Z10_UU/img/IMG_062613.tif")
        == "D012_2019"
    )
    with pytest.raises(ValueError):
        domain_of("IMG_000001.tif")


def test_select_rows_keeps_the_alphabetically_first_domains() -> None:
    rows = [_row("D006_2020"), _row("D001_2020"), _row("D003_2021")]

    selection = select_rows(rows, 2)

    assert selection.domains == ["D001_2020", "D003_2021"]
    assert [domain_of(image) for image, _ in selection.rows] == [
        "D001_2020",
        "D003_2021",
    ]
    assert selection.available == ["D001_2020", "D003_2021", "D006_2020"]


def test_select_rows_keeps_everything_when_the_limit_exceeds_the_domains() -> None:
    selection = select_rows([_row("D006_2020"), _row("D001_2020")], 5)

    assert selection.domains == ["D001_2020", "D006_2020"]
    assert len(selection.rows) == 2


def test_select_rows_rejects_a_zero_limit() -> None:
    with pytest.raises(ValueError):
        select_rows([_row("D001_2020")], 0)


def test_subset_spec_round_trips_through_text() -> None:
    spec = SubsetSpec.from_text(" 21, 8,10 ")

    assert (spec.train, spec.validation, spec.test) == (21, 8, 10)
    assert spec.as_text() == "21,8,10"
    assert spec.as_dict() == {"train": 21, "validation": 8, "test": 10}


@pytest.mark.parametrize("text", ["21,8", "21,8,10,1", "a,b,c", "21,0,10"])
def test_subset_spec_rejects_bad_text(text: str) -> None:
    with pytest.raises(ValueError):
        SubsetSpec.from_text(text)


def test_write_split_csv_round_trips_the_raw_rows(tmp_path: Path) -> None:
    rows = [("../img/IMG_1.tif", "../msk/MSK_1.tif")]

    path = write_split_csv(tmp_path / "CSVs" / "flair-1-paths-train.csv", rows)

    assert read_pairs_rows(path) == rows


def test_build_subset_csvs_filters_each_split_into_the_subset_root(
    write_flair_split, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    write_flair_split(
        source,
        "train",
        [("D006_2020", "Z1_UU", "000001"), ("D001_2020", "Z1_UU", "000002")],
        csv_stem="flair-1-paths",
    )
    write_flair_split(
        source,
        "validation",
        [("D004_2021", "Z1_UU", "000004"), ("D002_2021", "Z1_UU", "000005")],
        csv_stem="flair-1-paths",
    )
    write_flair_split(
        source,
        "test",
        [("D012_2019", "Z1_UU", "000006")],
        csv_stem="flair-1-paths",
    )
    csvs = {
        "train": source / "CSVs" / "flair-1-paths-train.csv",
        "validation": source / "CSVs" / "flair-1-paths-val.csv",
        "test": source / "CSVs" / "flair-1-paths-test.csv",
    }

    root = tmp_path / "subset"
    selections = build_subset_csvs(
        csvs, root, SubsetSpec(train=2, validation=1, test=3)
    )

    assert selections["train"].domains == ["D001_2020", "D006_2020"]
    assert selections["validation"].domains == ["D002_2021"]
    assert selections["test"].domains == ["D012_2019"]
    splits = build_flair_splits(root)
    assert {split: len(pair["IMG"]) for split, pair in splits.items()} == {
        "train": 2,
        "validation": 1,
        "test": 1,
    }


def test_build_subset_csvs_rejects_an_unknown_split(tmp_path: Path) -> None:
    source = write_split_csv(tmp_path / "a-train.csv", [_row("D001_2020")])

    with pytest.raises(ValueError, match="Unknown split"):
        build_subset_csvs({"holdout": source}, tmp_path / "subset", SubsetSpec())


def test_manifest_round_trips_and_matches_its_spec(tmp_path: Path) -> None:
    spec = SubsetSpec(train=1, validation=1, test=1)
    selections = {
        "train": SplitSelection(
            rows=[_row("D001_2020")], domains=["D001_2020"], available=["D001_2020"]
        )
    }

    path = write_manifest(tmp_path, spec, selections)

    assert path.name == MANIFEST_NAME
    manifest = read_manifest(tmp_path)
    assert manifest is not None
    assert manifest["strategy"] == "domains"
    assert manifest_matches(manifest, spec)
    assert not manifest_matches(manifest, SubsetSpec(train=2, validation=1, test=1))
    assert read_manifest(tmp_path / "absent") is None
