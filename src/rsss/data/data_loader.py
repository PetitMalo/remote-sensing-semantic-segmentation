from collections.abc import Callable

import numpy as np
import rasterio
import torch
from skimage.util import img_as_float
from torch.utils.data import Dataset

# Albumentations-style augmentation: called with the sample as keyword
# arguments ("image" and "mask") and returns a mapping with the same keys.
Augmentation = Callable[..., dict]


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


class FitDataset(Dataset):
    def __init__(
        self,
        dict_files: dict,
        channels: list | None = None,
        num_classes: int = 13,
        start_one: bool = True,
        use_metadata: bool = False,
        use_augmentations: Augmentation | None = None,
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
