"""Unit tests for ``rsss.data.splits``.

The FLAIR dataset defines its split through the path CSVs it ships, so these
tests cover CSV parsing, split classification and path resolution against
synthetic trees written to ``tmp_path``. Both the toy and the full dataset's
naming schemes are exercised, since the same code has to ingest either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rsss.data.splits import build_flair_splits, read_pairs_csv, split_of_csv

# (collection prefix, CSV stem) for the toy and the full dataset.
NAMING = [
    pytest.param("flair_1_toy", "flair-1-paths-toy", id="toy"),
    pytest.param("flair", "flair-1-paths", id="full"),
]


# --------------------------------------------------------------------------- #
# split_of_csv                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("flair-1-paths-train.csv", "train"),
        ("flair-1-paths-toy-train.csv", "train"),
        ("flair-1-paths-val.csv", "validation"),
        ("flair-1-paths-toy-val.csv", "validation"),
        ("flair-1-paths-test.csv", "test"),
        ("flair-1-paths-toy-test.csv", "test"),
        ("metadata.csv", None),
        ("flair-1-paths-toy.csv", None),
    ],
)
def test_split_of_csv_reads_the_trailing_token(file_name, expected):
    assert split_of_csv(file_name) == expected


# --------------------------------------------------------------------------- #
# read_pairs_csv                                                              #
# --------------------------------------------------------------------------- #


def test_read_pairs_csv_resolves_paths_relative_to_the_csv(tmp_path: Path):
    csv_dir = tmp_path / "CSVs"
    csv_dir.mkdir()
    (csv_dir / "paths.csv").write_text(
        "../aerial/D006_2020/Z1_UU/img/IMG_1.tif,"
        "../labels/D006_2020/Z1_UU/msk/MSK_1.tif\n"
    )

    files = read_pairs_csv(csv_dir / "paths.csv")

    assert files["IMG"] == [
        str((tmp_path / "aerial/D006_2020/Z1_UU/img/IMG_1.tif").resolve())
    ]
    assert files["MSK"] == [
        str((tmp_path / "labels/D006_2020/Z1_UU/msk/MSK_1.tif").resolve())
    ]


def test_read_pairs_csv_skips_blank_lines(tmp_path: Path):
    csv_dir = tmp_path / "CSVs"
    csv_dir.mkdir()
    (csv_dir / "paths.csv").write_text(
        "../a/img/IMG_1.tif,../b/msk/MSK_1.tif\n"
        "\n"
        "../a/img/IMG_2.tif,../b/msk/MSK_2.tif\n"
    )

    files = read_pairs_csv(csv_dir / "paths.csv")

    assert len(files["IMG"]) == len(files["MSK"]) == 2


def test_read_pairs_csv_rejects_rows_without_two_fields(tmp_path: Path):
    csv_dir = tmp_path / "CSVs"
    csv_dir.mkdir()
    (csv_dir / "paths.csv").write_text("../a/img/IMG_1.tif\n")

    with pytest.raises(ValueError, match="expected 'IMG,MSK' rows"):
        read_pairs_csv(csv_dir / "paths.csv")


# --------------------------------------------------------------------------- #
# build_flair_splits                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("collection", "csv_stem"), NAMING)
def test_build_flair_splits_reads_the_three_splits(
    tmp_path, write_flair_dataset, collection, csv_stem
):
    root = write_flair_dataset(
        tmp_path / "dataset", collection=collection, csv_stem=csv_stem
    )

    splits = build_flair_splits(root)

    assert set(splits) == {"train", "validation", "test"}
    for contents in splits.values():
        assert len(contents["IMG"]) == len(contents["MSK"]) == 1
    assert f"{collection}_aerial_train" in splits["train"]["IMG"][0]


def test_build_flair_splits_maps_val_to_validation_outside_train(
    tmp_path, write_flair_dataset
):
    root = write_flair_dataset(tmp_path / "dataset")

    splits = build_flair_splits(root)

    # Both live under the train collection directory, so only the CSVs separate
    # them: the validation tile must not leak into train.
    assert splits["train"]["IMG"] == [
        str(
            (
                root / "flair_1_toy_aerial_train/D006_2020/Z1_UU/img/IMG_000001.tif"
            ).resolve()
        )
    ]
    assert splits["validation"]["IMG"] == [
        str(
            (
                root / "flair_1_toy_aerial_train/D004_2021/Z9_UN/img/IMG_000734.tif"
            ).resolve()
        )
    ]
    assert splits["test"]["IMG"] == [
        str(
            (
                root / "flair_1_toy_aerial_test/D012_2019/Z3_UU/img/IMG_061946.tif"
            ).resolve()
        )
    ]


def test_build_flair_splits_ignores_csvs_without_a_split_token(
    tmp_path, write_flair_dataset
):
    root = write_flair_dataset(tmp_path / "dataset")
    (root / "CSVs" / "metadata.csv").write_text("a,b\n")

    splits = build_flair_splits(root)

    assert set(splits) == {"train", "validation", "test"}


def test_build_flair_splits_raises_without_csv_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="'CSVs' directory"):
        build_flair_splits(tmp_path / "dataset")


def test_build_flair_splits_raises_when_a_split_is_missing(tmp_path, write_flair_split):
    root = tmp_path / "dataset"
    write_flair_split(root, "train", [("D006_2020", "Z1_UU", "000001")])

    with pytest.raises(FileNotFoundError, match="validation"):
        build_flair_splits(root)


def test_build_flair_splits_rejects_two_csvs_for_one_split(
    tmp_path, write_flair_dataset
):
    root = write_flair_dataset(tmp_path / "dataset")
    (root / "CSVs" / "flair-1-paths-toy-alt-train.csv").write_text("")

    with pytest.raises(ValueError, match="Two CSVs describe the 'train' split"):
        build_flair_splits(root)
