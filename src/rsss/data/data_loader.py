from pathlib import Path

import numpy as np
import rasterio
import torch
from skimage.util import img_as_float
from torch.utils.data import Dataset


def norm(
    in_img: np.ndarray,
    norm_type: str | None = None,
    means: list | None = None,
    stds: list | None = None,
) -> np.ndarray:
    """Normalize an image array.

    Parameters
    ----------
    in_img : np.ndarray
        Input image array, shaped (bands, height, width).
    norm_type : str, optional
        Normalization type. One of 'scaling', 'custom', 'without'.
    means : list, optional
        Per-band means, used by 'custom'.
    stds : list, optional
        Per-band stds, used by 'custom'.

    Returns
    -------
    np.ndarray
        Normalized image.

    Notes
    -----
    'scaling' only rescales by the range of the input dtype: a uint8 array is
    divided by 255 and mapped to [0, 1]. It does not standardize to mean 0 /
    std 1. Backbones pre-trained with specific statistics (DINOv2, DINOv3)
    need 'custom' with the statistics they were trained with. 'without'
    returns the input unchanged.
    """
    if norm_type not in ["scaling", "custom", "without"]:
        raise ValueError(
            "Normalization argument should be 'scaling', 'custom' or 'without'."
        )
    if means is None:
        means = []
    if stds is None:
        stds = []
    if norm_type == "custom":
        if len(means) != len(stds):
            raise ValueError(
                "If custom, provided normalization means and stds should be of same length."
            )
        else:
            in_img = in_img.astype(np.float64)
            for i in range(in_img.shape[0]):
                in_img[i] -= means[i]
                in_img[i] /= stds[i]
    elif norm_type == "scaling":
        in_img = img_as_float(in_img)
    return in_img


def _assert_paired(area: str, tile: str, imgs: list[Path], msks: list[Path]) -> None:
    """Ensure IMG and MSK tiles correspond, so the two file lists stay aligned.

    IMG and MSK are collected by two independent globs. Without this check an
    unpaired tile would silently shift every following pair out of sync.
    """
    img_keys = {path.name[len("IMG_") :] for path in imgs}
    msk_keys = {path.name[len("MSK_") :] for path in msks}
    if img_keys != msk_keys:
        missing_msk = sorted(img_keys - msk_keys)
        missing_img = sorted(msk_keys - img_keys)
        raise ValueError(
            f"Unpaired tiles in {area}/{tile}: "
            f"missing MSK for {missing_msk}, missing IMG for {missing_img}."
        )


def discover_data(data_dir: str) -> dict[str, list[str]]:
    """Discover IMG and MSK file paths from the data directory structure.

    Expected structure under data_dir:
        <area>/
            aerial/
                <tile>/
                    IMG_*.tif
            labels/
                <tile>/
                    MSK_*.tif

    Areas missing either an ``aerial/`` or a ``labels/`` directory are skipped,
    as are any non-directory entries at the root. Sentinel ``.npy`` files are
    deliberately not discovered: nothing in the datasets can consume them yet,
    so returning them would only advertise metadata that is silently dropped.

    Parameters
    ----------
    data_dir : str
        Root directory containing area subdirectories (e.g. "data/").

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys "IMG" and "MSK", each a sorted list of paths. The
        two lists are index-aligned: ``IMG[i]`` and ``MSK[i]`` are the same
        tile.

    Raises
    ------
    FileNotFoundError
        If data_dir does not exist.
    ValueError
        If an ``IMG_*.tif`` has no matching ``MSK_*.tif`` (or vice versa) within
        a tile directory.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    img_files: list[str] = []
    msk_files: list[str] = []

    for area in sorted(path for path in root.iterdir() if path.is_dir()):
        aerial_dir = area / "aerial"
        labels_dir = area / "labels"

        if not aerial_dir.is_dir() or not labels_dir.is_dir():
            continue

        for tile_aerial in sorted(
            path for path in aerial_dir.iterdir() if path.is_dir()
        ):
            imgs = sorted(tile_aerial.glob("IMG_*.tif"))

            tile_labels = labels_dir / tile_aerial.name
            msks = sorted(tile_labels.glob("MSK_*.tif")) if tile_labels.is_dir() else []

            _assert_paired(area.name, tile_aerial.name, imgs, msks)
            img_files.extend(str(path) for path in imgs)
            msk_files.extend(str(path) for path in msks)

    return {"IMG": img_files, "MSK": msk_files}


class FitDataset(Dataset):
    def __init__(
        self,
        dict_files: dict,
        channels: list | None = None,
        num_classes: int = 13,
        start_one: bool = True,
        use_metadata: bool = False,
        use_augmentations=None,
        norm_type: str = "scaling",
        means: list | None = None,
        stds: list | None = None,
    ):

        self.list_imgs = np.array(dict_files["IMG"])
        self.list_msks = np.array(dict_files["MSK"])
        self.start_one = start_one
        self.use_metadata = use_metadata
        if use_metadata == True:
            self.list_metadata = np.array(dict_files.get("MTD", []))
        else:
            self.list_metadata = []
        self.use_augmentations = use_augmentations
        self.channels = channels
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.means = means
        self.stds = stds

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read(self.channels)
            return array

    def read_msk(self, raster_file: str, start_one) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            array = src_msk.read()[0]
            if start_one:
                array = array - 1
            array = np.stack([array == i for i in range(self.num_classes)], axis=0)
            return array

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)

        mask_file = self.list_msks[index]
        msk = self.read_msk(raster_file=mask_file, start_one=self.start_one)

        if self.use_augmentations:
            sample = {
                "image": img.swapaxes(0, 2).swapaxes(0, 1),
                "mask": msk.swapaxes(0, 2).swapaxes(0, 1),
            }
            transformed_sample = self.use_augmentations(**sample)
            img, msk = (
                transformed_sample["image"].swapaxes(0, 2).swapaxes(1, 2).copy(),
                transformed_sample["mask"].swapaxes(0, 2).swapaxes(1, 2).copy(),
            )

        img = norm(img, norm_type=self.norm_type, means=self.means, stds=self.stds)

        if self.use_metadata and len(self.list_metadata) > 0:
            mtd = self.list_metadata[index]
            return {
                "img": torch.as_tensor(img, dtype=torch.float),
                "mtd": torch.as_tensor(mtd, dtype=torch.float),
                "msk": torch.as_tensor(msk, dtype=torch.float),
            }
        else:
            return {
                "img": torch.as_tensor(img, dtype=torch.float),
                "msk": torch.as_tensor(msk, dtype=torch.float),
            }


class PredictDataset(Dataset):
    def __init__(
        self,
        dict_files: dict,
        channels: list | None = None,
        num_classes: int = 13,
        use_metadata: bool = True,
        norm_type: str = "scaling",
        means: list | None = None,
        stds: list | None = None,
    ):

        if channels is None:
            channels = [1, 2, 3, 4, 5]

        self.list_imgs = np.array(dict_files["IMG"])
        self.num_classes = num_classes
        self.use_metadata = use_metadata
        if use_metadata == True:
            self.list_metadata = np.array(dict_files.get("MTD", []))
        self.channels = channels
        self.norm_type = norm_type
        self.means = means
        self.stds = stds

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read(self.channels)
        return array

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(raster_file=image_file)

        img = norm(img, norm_type=self.norm_type, means=self.means, stds=self.stds)

        if self.use_metadata and len(self.list_metadata) > 0:
            mtd = self.list_metadata[index]
            return {
                "img": torch.as_tensor(img, dtype=torch.float),
                "mtd": torch.as_tensor(mtd, dtype=torch.float),
                "id": image_file,
            }
        else:
            return {"img": torch.as_tensor(img, dtype=torch.float), "id": image_file}
