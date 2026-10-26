"""Fixtures shared by the unit tests.

Everything is written to ``tmp_path`` as tiny synthetic GeoTIFFs, so the tests
never touch the real FLAIR dataset. The fixtures return callables so each test
can build exactly the tree it needs. They sit at the ``tests/`` root, not under
``tests/data/``, because the training tests reuse them to run a model over real
``FitDataset`` output.
"""

from __future__ import annotations

import csv
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

# FLAIR ships its split as path CSVs whose last "-"-separated token names the
# split; "val" is the toy's and the full dataset's spelling for validation.
_CSV_SUFFIXES = {"train": "train", "validation": "val", "test": "test"}
# Validation tiles live in the train collection, for both datasets.
_COLLECTION_ROOTS = {"train": "train", "validation": "train", "test": "test"}


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
def write_flair_split(write_img, write_msk):
    """Return a writer for one split of a synthetic FLAIR#1 dataset.

    It reproduces the real layout: one CSV per split under ``<root>/CSVs``, and
    the validation tiles inside the train collection directory. ``collection``
    and ``csv_stem`` select the naming scheme, so both the toy
    (``flair_1_toy`` / ``flair-1-paths-toy``) and the full dataset
    (``flair`` / ``flair-1-paths``) can be built.
    """

    def _write_flair_split(
        root: Path,
        split: str,
        tiles: list[tuple[str, str, str]],
        collection: str = "flair_1_toy",
        csv_stem: str = "flair-1-paths-toy",
        size: int = 4,
    ) -> Path:
        """Write the split's CSV and IMG/MSK files; ``tiles`` are (area, zone, stem)."""
        collection_root = _COLLECTION_ROOTS[split]
        rows: list[tuple[str, str]] = []
        for area, zone, stem in tiles:
            img_rel = (
                f"../{collection}_aerial_{collection_root}/"
                f"{area}/{zone}/img/IMG_{stem}.tif"
            )
            msk_rel = (
                f"../{collection}_labels_{collection_root}/"
                f"{area}/{zone}/msk/MSK_{stem}.tif"
            )
            write_img((root / "CSVs" / img_rel).resolve(), size=size)
            write_msk((root / "CSVs" / msk_rel).resolve(), np.full((size, size), 1))
            rows.append((img_rel, msk_rel))

        csv_path = root / "CSVs" / f"{csv_stem}-{_CSV_SUFFIXES[split]}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        return csv_path

    return _write_flair_split


@pytest.fixture
def write_flair_dataset(write_flair_split):
    """Return a writer for a complete three-split synthetic FLAIR#1 dataset."""

    def _write_flair_dataset(
        root: Path,
        collection: str = "flair_1_toy",
        csv_stem: str = "flair-1-paths-toy",
        size: int = 4,
    ) -> Path:
        """Write one tile per split and return ``root``."""
        tiles = {
            "train": [("D006_2020", "Z1_UU", "000001")],
            "validation": [("D004_2021", "Z9_UN", "000734")],
            "test": [("D012_2019", "Z3_UU", "061946")],
        }
        for split, split_tiles in tiles.items():
            write_flair_split(
                root,
                split,
                split_tiles,
                collection=collection,
                csv_stem=csv_stem,
                size=size,
            )
        return root

    return _write_flair_dataset


class RangeServer:
    """A local HTTP server serving ``root`` and honouring ``Range`` requests.

    It stands in for IGN's OVH storage: the remote-zip and subset-staging tests
    write their synthetic archives and CSVs under ``root`` and point the code at
    ``url``, so no test ever touches the network. ``ignore_range`` makes it
    answer ``200`` to a ranged GET (to test that the reader refuses such a
    server), ``failures`` returns that many ``503`` responses first (to test the
    retry path), and ``requests`` records every ``(path, Range)`` received.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.url = ""
        self.ignore_range = False
        self.failures = 0
        self.requests: list[tuple[str, str | None]] = []
        self._server: _RangeHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def route(self, relative: str) -> str:
        """Return the absolute URL of ``relative`` under the served root."""
        return f"{self.url}/{relative}"

    def start(self) -> RangeServer:
        """Bind an ephemeral port and serve in a background thread."""
        self._server = _RangeHTTPServer(("127.0.0.1", 0), _RangeHandler)
        self._server.state = self
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the server and release its socket."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


class _RangeHTTPServer(ThreadingHTTPServer):
    """The HTTP server carrying the shared :class:`RangeServer` state."""

    state: RangeServer


class _RangeHandler(BaseHTTPRequestHandler):
    """Serve files from ``RangeServer.root``, slicing the requested byte range."""

    def do_GET(self) -> None:
        server = cast(_RangeHTTPServer, self.server)
        state = server.state
        path = state.root / self.path.split("?", 1)[0].lstrip("/")
        if not path.is_file():
            self.send_error(404)
            return
        state.requests.append((self.path, self.headers.get("Range")))
        if state.failures:
            state.failures -= 1
            self.send_error(503)
            return
        data = path.read_bytes()
        range_header = self.headers.get("Range")
        if range_header and not state.ignore_range:
            span = range_header.partition("=")[2]
            start_text, _, end_text = span.partition("-")
            start = int(start_text)
            end = int(end_text) if end_text else len(data) - 1
            chunk = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """Stay quiet: pytest captures what it needs."""


@pytest.fixture
def range_server(tmp_path: Path) -> Iterator[RangeServer]:
    """Serve ``<tmp_path>/served`` over HTTP with Range support, for one test."""
    root = tmp_path / "served"
    root.mkdir()
    server = RangeServer(root).start()
    yield server
    server.stop()
