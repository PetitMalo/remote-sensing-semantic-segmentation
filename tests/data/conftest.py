"""Fixtures shared by the ``rsss.data`` unit tests.

Everything is written to ``tmp_path`` as tiny synthetic GeoTIFFs, so the tests
never touch the real FLAIR dataset. The fixtures return callables so each test
can build exactly the tree it needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def write_tif():
    """Return a writer for a ``(bands, height, width)`` array as a GeoTIFF."""

    def _write_tif(path: Path, array: np.ndarray) -> Path:
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

    return _write_tif


@pytest.fixture
def write_img(write_tif):
    """Return a writer for an image whose band ``b`` is filled with ``b``."""

    def _write_img(path: Path, num_bands: int = 5, size: int = 4) -> Path:
        array = np.stack(
            [np.full((size, size), b, dtype=np.uint8) for b in range(num_bands)]
        )
        return write_tif(path, array)

    return _write_img


@pytest.fixture
def write_msk(write_tif):
    """Return a writer for a single-band label raster from a 2D array."""

    def _write_msk(path: Path, values: np.ndarray) -> Path:
        return write_tif(path, np.asarray(values, dtype=np.uint8)[None, :, :])

    return _write_msk


@pytest.fixture
def write_tile(write_img, write_msk):
    """Return a writer populating one ``<area>/{aerial,labels}/<tile>/`` tree."""

    def _write_tile(
        root: Path,
        area: str,
        tile: str,
        img_ids: list[str],
        msk_ids: list[str],
    ) -> None:
        for img_id in img_ids:
            write_img(root / area / "aerial" / tile / f"IMG_{img_id}.tif")
        for msk_id in msk_ids:
            write_msk(
                root / area / "labels" / tile / f"MSK_{msk_id}.tif",
                np.full((4, 4), 1),
            )

    return _write_tile
