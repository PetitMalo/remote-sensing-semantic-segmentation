"""Unit tests for ``rsss.data.data_loader``.

Everything runs against tiny synthetic GeoTIFFs written to ``tmp_path``, so the
tests never touch the real FLAIR dataset.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from rsss.data.data_loader import (
    FitDataset,
    PredictDataset,
    _assert_paired,
    discover_data,
    norm,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _write_tif(path: Path, array: np.ndarray) -> Path:
    """Write a ``(bands, height, width)`` array as a GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype.name,
        transform=from_origin(100.0, 200.0, 0.5, 0.5),
    ) as dst:
        dst.write(array)
    return path


def _write_img(path: Path, num_bands: int = 5, size: int = 4) -> Path:
    """Write an image whose band ``b`` is filled with the value ``b``."""
    array = np.stack(
        [np.full((size, size), b, dtype=np.uint8) for b in range(num_bands)]
    )
    return _write_tif(path, array)


def _write_msk(path: Path, values: np.ndarray) -> Path:
    """Write a single-band label raster from a ``(height, width)`` array."""
    return _write_tif(path, values.astype(np.uint8)[None, :, :])


def _write_tile(
    root: Path,
    area: str,
    tile: str,
    img_ids: list[str],
    msk_ids: list[str],
) -> None:
    """Populate one ``<area>/{aerial,labels}/<tile>/`` directory tree."""
    for img_id in img_ids:
        _write_img(root / area / "aerial" / tile / f"IMG_{img_id}.tif")
    for msk_id in msk_ids:
        _write_msk(
            root / area / "labels" / tile / f"MSK_{msk_id}.tif",
            np.full((4, 4), 1),
        )


@pytest.fixture
def tile_files(tmp_path: Path) -> dict[str, list[str]]:
    """A single tile with predictable image values and labels."""
    root = tmp_path / "data"
    img = _write_img(root / "D006_2020" / "aerial" / "Z1" / "IMG_000001.tif")
    msk = _write_msk(
        root / "D006_2020" / "labels" / "Z1" / "MSK_000001.tif",
        np.array(
            [
                [1, 1, 2, 2],
                [1, 3, 3, 2],
                [0, 0, 2, 2],
                [3, 3, 3, 3],
            ]
        ),
    )
    return {"IMG": [str(img)], "MSK": [str(msk)]}


# --------------------------------------------------------------------------- #
# norm                                                                        #
# --------------------------------------------------------------------------- #


def test_norm_scaling_maps_uint8_to_unit_range():
    img = np.array([[[0, 128]], [[255, 255]]], dtype=np.uint8)

    out = norm(img, norm_type="scaling")

    assert out.dtype == np.float64
    np.testing.assert_allclose(out, img / 255.0)


def test_norm_without_returns_input_unchanged():
    img = np.array([[[0, 1]], [[2, 3]]], dtype=np.uint8)

    out = norm(img, norm_type="without")

    assert out is img


def test_norm_custom_standardizes_each_band():
    img = np.array([[[0.0, 2.0]], [[10.0, 20.0]]])

    out = norm(img, norm_type="custom", means=[1.0, 15.0], stds=[1.0, 5.0])

    np.testing.assert_allclose(out, [[[-1.0, 1.0]], [[-1.0, 1.0]]])


def test_norm_custom_rejects_mismatched_means_and_stds():
    img = np.zeros((2, 2, 2))

    with pytest.raises(ValueError, match="same length"):
        norm(img, norm_type="custom", means=[0.0, 0.0], stds=[1.0])


def test_norm_rejects_unknown_type():
    img = np.zeros((1, 2, 2))

    with pytest.raises(ValueError, match="'scaling', 'custom' or 'without'"):
        norm(img, norm_type="minmax")


# --------------------------------------------------------------------------- #
# _assert_paired                                                             #
# --------------------------------------------------------------------------- #


def test_assert_paired_accepts_matching_tiles():
    imgs = [Path("IMG_000001.tif"), Path("IMG_000002.tif")]
    msks = [Path("MSK_000002.tif"), Path("MSK_000001.tif")]

    _assert_paired("D006_2020", "Z1", imgs, msks)  # does not raise


def test_assert_paired_reports_missing_mask():
    imgs = [Path("IMG_000001.tif"), Path("IMG_000002.tif")]

    with pytest.raises(ValueError, match="missing MSK for \\['000002.tif'\\]"):
        _assert_paired("D006_2020", "Z1", imgs, [Path("MSK_000001.tif")])


def test_assert_paired_reports_missing_image():
    msks = [Path("MSK_000001.tif"), Path("MSK_000002.tif")]

    with pytest.raises(ValueError, match="missing IMG for \\['000002.tif'\\]"):
        _assert_paired("D006_2020", "Z1", [Path("IMG_000001.tif")], msks)


# --------------------------------------------------------------------------- #
# discover_data                                                              #
# --------------------------------------------------------------------------- #


def test_discover_data_returns_index_aligned_sorted_paths(tmp_path: Path):
    root = tmp_path / "data"
    _write_tile(root, "D006_2020", "Z2", ["000002"], ["000002"])
    _write_tile(root, "D006_2020", "Z1", ["000002", "000001"], ["000001", "000002"])
    _write_tile(root, "D004_2021", "Z1", ["000001"], ["000001"])

    files = discover_data(str(root))

    assert files["IMG"] == [str(p) for p in sorted(root.rglob("IMG_*.tif"))]
    assert files["MSK"] == [str(p) for p in sorted(root.rglob("MSK_*.tif"))]
    assert all(isinstance(path, str) for path in files["IMG"])
    for img, msk in zip(files["IMG"], files["MSK"], strict=True):
        assert Path(img).name[len("IMG_") :] == Path(msk).name[len("MSK_") :]


def test_discover_data_missing_directory_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        discover_data(str(tmp_path / "does_not_exist"))


def test_discover_data_skips_areas_without_aerial_or_labels(tmp_path: Path):
    root = tmp_path / "data"
    _write_img(root / "D006_2020" / "aerial" / "Z1" / "IMG_000001.tif")
    _write_msk(
        root / "D004_2021" / "labels" / "Z1" / "MSK_000001.tif", np.full((4, 4), 1)
    )
    _write_tile(root, "D012_2019", "Z1", ["000001"], ["000001"])

    files = discover_data(str(root))

    assert [Path(p).parts[-4] for p in files["IMG"]] == ["D012_2019"]


def test_discover_data_skips_non_directory_entries_at_root(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "README.txt").write_text("not an area")
    _write_tile(root, "D006_2020", "Z1", ["000001"], ["000001"])

    files = discover_data(str(root))

    assert len(files["IMG"]) == 1


def test_discover_data_ignores_sentinel_npy_files(tmp_path: Path):
    root = tmp_path / "data"
    _write_tile(root, "D006_2020", "Z1", ["000001"], ["000001"])
    (root / "D006_2020" / "aerial" / "Z1" / "IMG_000001.npy").write_bytes(b"")

    files = discover_data(str(root))

    assert all(path.endswith(".tif") for path in files["IMG"])
    assert len(files["IMG"]) == 1


def test_discover_data_raises_on_tile_missing_mask(tmp_path: Path):
    root = tmp_path / "data"
    _write_tile(root, "D006_2020", "Z1", ["000001", "000002"], ["000001"])

    with pytest.raises(ValueError, match="missing MSK"):
        discover_data(str(root))


def test_discover_data_raises_on_tile_missing_image(tmp_path: Path):
    root = tmp_path / "data"
    _write_tile(root, "D006_2020", "Z1", ["000001"], ["000001", "000002"])

    with pytest.raises(ValueError, match="missing IMG"):
        discover_data(str(root))


# --------------------------------------------------------------------------- #
# FitDataset                                                                  #
# --------------------------------------------------------------------------- #


def test_fit_dataset_len(tile_files):
    assert len(FitDataset(tile_files)) == 1


def test_fit_dataset_read_img_selects_requested_channels(tile_files):
    dataset = FitDataset(tile_files, channels=[1, 3])

    img = dataset.read_img(tile_files["IMG"][0])

    assert img.shape == (2, 4, 4)
    assert img[0].min() == img[0].max() == 0
    assert img[1].min() == img[1].max() == 2


def test_fit_dataset_read_msk_one_hot_encodes_labels(tile_files):
    dataset = FitDataset(tile_files, num_classes=4)

    msk = dataset.read_msk(tile_files["MSK"][0], start_one=True)

    assert msk.shape == (4, 4, 4)
    # start_one shifts labels down by one, so class 0 is the original value 1...
    assert msk[0].sum() == 3
    # ...and the original background value 0 falls outside every class.
    assert msk.sum() == 14
    # One-hot channels are mutually exclusive.
    assert np.all(msk.sum(axis=0) <= 1)


def test_fit_dataset_read_msk_without_start_one(tile_files):
    dataset = FitDataset(tile_files, num_classes=4)

    msk = dataset.read_msk(tile_files["MSK"][0], start_one=False)

    # Without the shift, the original background (0) lights up channel 0 and the
    # original value 1 lights up channel 1.
    assert msk[0].sum() == 2
    assert msk[1].sum() == 3


def test_fit_dataset_getitem_returns_normalized_tensors(tile_files):
    dataset = FitDataset(tile_files, channels=[1, 2, 3], num_classes=4)

    sample = dataset[0]

    assert set(sample) == {"img", "msk"}
    assert sample["img"].shape == (3, 4, 4)
    assert sample["img"].dtype == torch.float
    assert sample["msk"].shape == (4, 4, 4)
    assert sample["msk"].dtype == torch.float
    # 'scaling' rescales the uint8 band 2 (fill value 1) to 1 / 255.
    torch.testing.assert_close(sample["img"][1], torch.full((4, 4), 1 / 255))


def test_fit_dataset_getitem_with_metadata(tile_files):
    files = {**tile_files, "MTD": [np.array([1.0, 2.0])]}
    dataset = FitDataset(files, use_metadata=True)

    sample = dataset[0]

    assert set(sample) == {"img", "mtd", "msk"}
    torch.testing.assert_close(sample["mtd"], torch.tensor([1.0, 2.0]))


def test_fit_dataset_getitem_without_metadata_omits_key(tile_files):
    dataset = FitDataset(tile_files, use_metadata=True)

    sample = dataset[0]

    assert "mtd" not in sample


def test_fit_dataset_getitem_applies_augmentations(tile_files):
    calls: list[tuple[int, ...]] = []

    def augment(image, mask):
        calls.append(image.shape)
        return {"image": image + 1, "mask": mask}

    dataset = FitDataset(tile_files, channels=[1], use_augmentations=augment)

    sample = dataset[0]

    assert calls == [(4, 4, 1)]
    # The augmentation added 1 to band 1 (1-based) before normalization; that
    # band starts at fill value 0, so it becomes 1 / 255.
    torch.testing.assert_close(sample["img"], torch.full((1, 4, 4), 1 / 255))


def test_fit_dataset_getitem_custom_norm(tile_files):
    dataset = FitDataset(
        tile_files,
        channels=[1],
        norm_type="custom",
        means=[1.0],
        stds=[2.0],
    )

    sample = dataset[0]

    torch.testing.assert_close(sample["img"], torch.full((1, 4, 4), -0.5))


# --------------------------------------------------------------------------- #
# PredictDataset                                                              #
# --------------------------------------------------------------------------- #


def test_predict_dataset_len(tile_files):
    assert len(PredictDataset(tile_files)) == 1


def test_predict_dataset_defaults_to_five_channels(tile_files):
    dataset = PredictDataset(tile_files)

    img = dataset.read_img(tile_files["IMG"][0])

    assert img.shape == (5, 4, 4)
    for band in range(5):
        assert img[band].min() == img[band].max() == band


def test_predict_dataset_getitem_returns_image_and_id(tile_files):
    dataset = PredictDataset(tile_files, channels=[1], use_metadata=False)

    sample = dataset[0]

    assert set(sample) == {"img", "id"}
    assert sample["id"] == tile_files["IMG"][0]
    assert sample["img"].shape == (1, 4, 4)


def test_predict_dataset_getitem_with_metadata(tile_files):
    files = {**tile_files, "MTD": [np.array([1.0, 2.0])]}
    dataset = PredictDataset(files, channels=[1], use_metadata=True)

    sample = dataset[0]

    assert set(sample) == {"img", "mtd", "id"}
    torch.testing.assert_close(sample["mtd"], torch.tensor([1.0, 2.0]))
