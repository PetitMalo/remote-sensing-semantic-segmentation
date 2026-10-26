"""Tests for staging a whole-domain FLAIR#1 subset from remote archives."""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from rsss.data.config import load_data_config
from rsss.data.download import (
    FullDataset,
    RemoteArchive,
    SubsetDataset,
    ensure_subset_dataset,
    main,
    members_for_archive,
    subset_split_sizes,
    verify_flair_dataset,
)
from rsss.data.subset import SubsetSpec, domain_of, read_manifest

if TYPE_CHECKING:
    from tests.conftest import RangeServer

# Three train tiles across two domains, one val tile and three test tiles across
# two domains; the (1, 1, 1) spec below keeps D001, D003 and D010.
_TILES = {
    "train": [("D001", "000001"), ("D001", "000002"), ("D002", "000003")],
    "validation": [("D003", "000004")],
    "test": [("D010", "000005"), ("D010", "000006"), ("D011", "000007")],
}
_KEPT = {"train": "D001", "validation": "D003", "test": "D010"}
_CSV_SUFFIX = {"train": "train", "validation": "val", "test": "test"}
_TILE_SIZE = 512


@dataclass(frozen=True)
class FakeFlair:
    """A served four-archive FLAIR#1 stand-in and what its subset costs."""

    full: FullDataset
    bytes_per_split: dict[str, int]


def _zip_members(path: Path, members: dict[str, Path]) -> int:
    """Write ``members`` (archive name -> file) as a zip; return its size."""
    with zipfile.ZipFile(path, "w") as archive:
        for member, file in members.items():
            archive.write(file, arcname=member)
    return path.stat().st_size


def _spec() -> SubsetSpec:
    """Return the one-domain-per-split spec the tests stage."""
    return SubsetSpec(train=1, validation=1, test=1)


@pytest.fixture
def fake_flair(
    range_server: RangeServer, write_img, write_msk, tmp_path: Path
) -> FakeFlair:
    """Serve a four-archive FLAIR#1-shaped dataset from the local server."""
    build = tmp_path / "build"
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    test_images: dict[str, Path] = {}
    test_masks: dict[str, Path] = {}
    rows: dict[str, list[tuple[str, str]]] = {split: [] for split in _TILES}

    for split, tiles in _TILES.items():
        for domain, stem in tiles:
            if split == "test":
                image = (
                    build / f"data/flair_1_aerial_test/{domain}/Z1/img/IMG_{stem}.tif"
                )
                mask = (
                    build / f"data/flair_1_labels_test/{domain}/Z1/msk/MSK_{stem}.tif"
                )
                test_images[f"flair_1_aerial_test/{domain}/Z1/img/IMG_{stem}.tif"] = (
                    write_img(image, size=_TILE_SIZE)
                )
                test_masks[f"flair_1_labels_test/{domain}/Z1/msk/MSK_{stem}.tif"] = (
                    write_msk(mask, np.full((_TILE_SIZE, _TILE_SIZE), 1))
                )
            else:
                image = build / f"flair_aerial_train/{domain}/Z1/img/IMG_{stem}.tif"
                mask = build / f"flair_labels_train/{domain}/Z1/msk/MSK_{stem}.tif"
                images[f"flair_aerial_train/{domain}/Z1/img/IMG_{stem}.tif"] = (
                    write_img(image, size=_TILE_SIZE)
                )
                masks[f"flair_labels_train/{domain}/Z1/msk/MSK_{stem}.tif"] = write_msk(
                    mask, np.full((_TILE_SIZE, _TILE_SIZE), 1)
                )
            rows[split].append(
                (f"../{image.relative_to(build)}", f"../{mask.relative_to(build)}")
            )

    archives = {
        "train": _zip_members(range_server.root / "train.zip", images),
        "labels": _zip_members(range_server.root / "labels.zip", masks),
        "test": _zip_members(range_server.root / "test.zip", test_images),
        "test_labels": _zip_members(range_server.root / "test_labels.zip", test_masks),
    }

    csv_urls = {}
    csv_dir = range_server.root / "csv"
    csv_dir.mkdir()
    for split, split_rows in rows.items():
        name = f"flair-1-paths-{_CSV_SUFFIX[split]}.csv"
        with (csv_dir / name).open("w", newline="") as handle:
            csv.writer(handle).writerows(split_rows)
        csv_urls[name] = range_server.route(f"csv/{name}")

    full = FullDataset(
        name="fake_full",
        archives=(
            RemoteArchive(
                url=range_server.route("train.zip"),
                size=archives["train"],
                extract_to=Path("fake_full"),
                provides=Path("flair_aerial_train"),
            ),
            RemoteArchive(
                url=range_server.route("labels.zip"),
                size=archives["labels"],
                extract_to=Path("fake_full"),
                provides=Path("flair_labels_train"),
            ),
            RemoteArchive(
                url=range_server.route("test.zip"),
                size=archives["test"],
                extract_to=Path("fake_full") / "data",
                provides=Path("data/flair_1_aerial_test"),
            ),
            RemoteArchive(
                url=range_server.route("test_labels.zip"),
                size=archives["test_labels"],
                extract_to=Path("fake_full") / "data",
                provides=Path("data/flair_1_labels_test"),
            ),
        ),
        csv_urls=csv_urls,
    )
    bytes_per_split = {
        split: sum(
            (build / relative.removeprefix("../")).stat().st_size
            for row in split_rows
            if domain_of(row[0]) == _KEPT[split]
            for relative in row
        )
        for split, split_rows in rows.items()
    }
    return FakeFlair(full=full, bytes_per_split=bytes_per_split)


def test_members_for_archive_maps_the_train_and_test_layouts(tmp_path: Path) -> None:
    root = tmp_path / "subset"
    train = RemoteArchive(
        url="u", size=1, extract_to=Path("full"), provides=Path("flair_aerial_train")
    )
    wanted = [
        root / "flair_aerial_train/D001/img/IMG_1.tif",
        root / "flair_labels_train/D001/msk/MSK_1.tif",
    ]
    test = RemoteArchive(
        url="u",
        size=1,
        extract_to=Path("full") / "data",
        provides=Path("data/flair_1_aerial_test"),
    )
    wanted_test = [root / "data/flair_1_aerial_test/D010/img/IMG_1.tif"]

    assert members_for_archive(train, "full", root, wanted) == {
        "flair_aerial_train/D001/img/IMG_1.tif": wanted[0]
    }
    assert members_for_archive(test, "full", root, wanted_test) == {
        "flair_1_aerial_test/D010/img/IMG_1.tif": wanted_test[0]
    }


def test_ensure_subset_dataset_stages_only_the_kept_domains(
    fake_flair: FakeFlair, tmp_path: Path
) -> None:
    subset = SubsetDataset(name="fake_subset", full=fake_flair.full, spec=_spec())

    root = ensure_subset_dataset(subset, tmp_path / "data", workers=2)

    report = verify_flair_dataset(root)
    assert report.tiles_per_split == {"train": 2, "validation": 1, "test": 2}
    assert (root / "flair_aerial_train/D001").is_dir()
    assert not (root / "flair_aerial_train/D002").exists()
    assert not (root / "data/flair_1_aerial_test/D011").exists()
    assert (root / "source_csvs/flair-1-paths-train.csv").is_file()
    manifest = read_manifest(root)
    assert manifest is not None
    assert manifest["spec"] == {"train": 1, "validation": 1, "test": 1}
    assert manifest["splits"]["train"]["available"] == ["D001", "D002"]
    assert manifest["splits"]["test"]["available"] == ["D010", "D011"]


def test_ensure_subset_dataset_skips_tiles_already_staged(
    fake_flair: FakeFlair, tmp_path: Path
) -> None:
    subset = SubsetDataset(name="fake_subset", full=fake_flair.full, spec=_spec())
    root = ensure_subset_dataset(subset, tmp_path / "data")
    staged = {path: path.stat().st_mtime_ns for path in root.rglob("*.tif")}

    ensure_subset_dataset(subset, tmp_path / "data")

    assert {path: path.stat().st_mtime_ns for path in root.rglob("*.tif")} == staged


def test_ensure_subset_dataset_refuses_another_spec(
    fake_flair: FakeFlair, tmp_path: Path
) -> None:
    ensure_subset_dataset(
        SubsetDataset(name="fake_subset", full=fake_flair.full, spec=_spec()),
        tmp_path / "data",
    )
    wider = SubsetDataset(
        name="fake_subset", full=fake_flair.full, spec=SubsetSpec(2, 1, 1)
    )

    with pytest.raises(ValueError, match="staged with spec"):
        ensure_subset_dataset(wider, tmp_path / "data")


def test_report_only_sizes_the_subset_without_fetching_tiles(
    fake_flair: FakeFlair, tmp_path: Path
) -> None:
    subset = SubsetDataset(name="fake_subset", full=fake_flair.full, spec=_spec())

    root = ensure_subset_dataset(subset, tmp_path / "data", report_only=True)

    assert not (root / "flair_aerial_train").exists()
    assert (root / "CSVs/flair-1-paths-train.csv").is_file()
    assert subset_split_sizes(subset, root) == fake_flair.bytes_per_split


def test_data_subset_config_selects_the_subset_dataset() -> None:
    config = load_data_config(
        Path(__file__).parents[1] / "configs" / "data_subset.yaml"
    )

    assert config.dataset == "flair_1_subset"
    assert config.dataset_dir.name == "flair_1_subset_dataset"
    assert config.download


def test_main_rejects_subset_flags_for_the_full_dataset() -> None:
    with pytest.raises(SystemExit):
        main(["--dataset", "flair_1_full", "--num-domains", "1,1,1"])


def test_main_rejects_report_only_with_force() -> None:
    with pytest.raises(SystemExit):
        main(["--dataset", "flair_1_subset", "--report-only", "--force"])
