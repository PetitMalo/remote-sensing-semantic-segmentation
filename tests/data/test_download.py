"""Unit tests for ``rsss.data.download``.

Everything runs offline: downloads hit a throwaway HTTP server on localhost, and
datasets are the tiny synthetic trees from ``conftest``. Nothing here touches the
real OVH archive or ``data/``.
"""

from __future__ import annotations

import http.server
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rsss.data.download import (
    TOY_DATASET,
    RemoteDataset,
    download_file,
    ensure_dataset,
    extract_zip,
    verify_flair_dataset,
)


@pytest.fixture
def http_server(tmp_path: Path):
    """Serve ``tmp_path/served`` over HTTP, honouring a single ``Range`` header.

    Yields a namespace with the served ``directory``, the ``base`` URL, the
    ``ranges`` header seen on each request, and a ``state`` dict whose
    ``ignore_range`` flag forces the server to answer a Range request with a
    full ``200`` body (so the restart path can be exercised).
    """
    directory = tmp_path / "served"
    directory.mkdir()
    ranges: list[str | None] = []
    state = {"ignore_range": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            ranges.append(self.headers.get("Range"))
            payload = (directory / self.path.lstrip("/")).read_bytes()
            start = 0
            status = 200
            if self.headers.get("Range") and not state["ignore_range"]:
                start = int(self.headers["Range"].split("=", 1)[1].split("-", 1)[0])
                status = 206
            body = payload[start:]
            self.send_response(status)
            if status == 206:
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{len(payload) - 1}/{len(payload)}",
                )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(
            directory=directory,
            base=f"http://127.0.0.1:{server.server_address[1]}",
            ranges=ranges,
            state=state,
        )
    finally:
        server.shutdown()
        thread.join()


# --------------------------------------------------------------------------- #
# RemoteDataset                                                               #
# --------------------------------------------------------------------------- #


def test_toy_dataset_points_at_the_ovh_archive():
    assert TOY_DATASET.name == "flair_1_toy_dataset"
    assert TOY_DATASET.url.endswith("/flair_1_toy_dataset.zip")
    assert "storage.gra.cloud.ovh.net" in TOY_DATASET.url
    assert TOY_DATASET.size == 226_168_911


# --------------------------------------------------------------------------- #
# download_file                                                               #
# --------------------------------------------------------------------------- #


def test_download_file_streams_the_payload(tmp_path, http_server):
    payload = b"flair" * 1000
    (http_server.directory / "blob.bin").write_bytes(payload)
    dest = tmp_path / "blob.bin"

    download_file(f"{http_server.base}/blob.bin", dest, size=len(payload))

    assert dest.read_bytes() == payload
    assert not (tmp_path / "blob.bin.part").exists()
    assert http_server.ranges == [None]


def test_download_file_resumes_a_partial_file(tmp_path, http_server):
    payload = bytes(range(256)) * 40
    (http_server.directory / "blob.bin").write_bytes(payload)
    dest = tmp_path / "blob.bin"
    (tmp_path / "blob.bin.part").write_bytes(payload[:100])

    download_file(f"{http_server.base}/blob.bin", dest, size=len(payload))

    assert dest.read_bytes() == payload
    assert http_server.ranges[-1] == "bytes=100-"


def test_download_file_restarts_when_the_server_ignores_range(tmp_path, http_server):
    payload = b"x" * 500
    (http_server.directory / "blob.bin").write_bytes(payload)
    dest = tmp_path / "blob.bin"
    (tmp_path / "blob.bin.part").write_bytes(payload[:100])
    http_server.state["ignore_range"] = True

    download_file(f"{http_server.base}/blob.bin", dest, size=len(payload))

    # A Range header was sent, but the server ignored it, so the file must be
    # rewritten whole rather than appended to.
    assert http_server.ranges[-1] == "bytes=100-"
    assert dest.read_bytes() == payload
    assert not (tmp_path / "blob.bin.part").exists()


def test_download_file_rejects_a_size_mismatch(tmp_path, http_server):
    (http_server.directory / "blob.bin").write_bytes(b"short")

    with pytest.raises(ValueError, match="expected 99 bytes, got 5"):
        download_file(f"{http_server.base}/blob.bin", tmp_path / "blob.bin", size=99)

    assert (tmp_path / "blob.bin.part").is_file()
    assert not (tmp_path / "blob.bin").exists()


# --------------------------------------------------------------------------- #
# extract_zip                                                                 #
# --------------------------------------------------------------------------- #


def test_extract_zip_writes_the_tree(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("root/CSVs/paths.csv", "a,b\n")
        handle.writestr("root/file.txt", "hi")

    extract_zip(archive, tmp_path / "out")

    assert (tmp_path / "out/root/CSVs/paths.csv").read_text() == "a,b\n"
    assert (tmp_path / "out/root/file.txt").read_text() == "hi"


def test_extract_zip_rejects_a_member_escaping_the_destination(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../evil.txt", "boom")

    with pytest.raises(ValueError, match="escapes"):
        extract_zip(archive, tmp_path / "out")

    assert not (tmp_path / "evil.txt").exists()


# --------------------------------------------------------------------------- #
# verify_flair_dataset                                                        #
# --------------------------------------------------------------------------- #


def test_verify_flair_dataset_reports_counts_and_domains(tmp_path, write_flair_dataset):
    root = write_flair_dataset(tmp_path / "dataset")

    report = verify_flair_dataset(root, tile_size=4)

    assert report.tiles_per_split == {"train": 1, "validation": 1, "test": 1}
    assert report.domains_per_split == {
        "train": ["D006_2020"],
        "validation": ["D004_2021"],
        "test": ["D012_2019"],
    }
    assert report.summary() == ("train: 1 tiles, validation: 1 tiles, test: 1 tiles")


def test_verify_flair_dataset_raises_when_a_file_is_missing(
    tmp_path, write_flair_dataset
):
    root = write_flair_dataset(tmp_path / "dataset")
    next((root / "flair_1_toy_aerial_test").rglob("IMG_*.tif")).unlink()

    with pytest.raises(FileNotFoundError, match="missing"):
        verify_flair_dataset(root, tile_size=4)


def test_verify_flair_dataset_rejects_a_mismatched_pair(
    tmp_path, write_flair_dataset, write_msk
):
    root = write_flair_dataset(tmp_path / "dataset")
    write_msk(
        root / "flair_1_toy_labels_train/D006_2020/Z1_UU/msk/MSK_999999.tif",
        np.ones((4, 4)),
    )
    (root / "CSVs/flair-1-paths-toy-train.csv").write_text(
        "../flair_1_toy_aerial_train/D006_2020/Z1_UU/img/IMG_000001.tif,"
        "../flair_1_toy_labels_train/D006_2020/Z1_UU/msk/MSK_999999.tif\n"
    )

    with pytest.raises(ValueError, match="pair mismatch"):
        verify_flair_dataset(root, tile_size=4)


def test_verify_flair_dataset_rejects_a_wrong_band_count(
    tmp_path, write_flair_dataset, write_tif
):
    root = write_flair_dataset(tmp_path / "dataset")
    write_tif(
        root / "flair_1_toy_aerial_test/D012_2019/Z3_UU/img/IMG_061946.tif",
        np.zeros((3, 4, 4), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="expected 5 band"):
        verify_flair_dataset(root, tile_size=4)


def test_verify_flair_dataset_rejects_out_of_range_labels(
    tmp_path, write_flair_dataset, write_msk
):
    root = write_flair_dataset(tmp_path / "dataset")
    write_msk(
        root / "flair_1_toy_labels_test/D012_2019/Z3_UU/msk/MSK_061946.tif",
        np.full((4, 4), 14),
    )

    with pytest.raises(ValueError, match=r"outside 1\.\.13"):
        verify_flair_dataset(root, tile_size=4)


# --------------------------------------------------------------------------- #
# ensure_dataset                                                              #
# --------------------------------------------------------------------------- #


def test_ensure_dataset_reuses_a_verified_tree(tmp_path, write_flair_dataset):
    data_dir = tmp_path / "data"
    write_flair_dataset(data_dir / "toy", size=512)
    # An unreachable URL proves the existing tree is reused, not re-downloaded.
    dataset = RemoteDataset(name="toy", url="http://127.0.0.1:1/never", size=1)

    root = ensure_dataset(dataset, data_dir)

    assert root == data_dir / "toy"


def test_ensure_dataset_downloads_extracts_verifies_and_drops_the_zip(
    tmp_path, http_server, write_flair_dataset
):
    staging = write_flair_dataset(tmp_path / "staging", size=512)
    archive = tmp_path / "toy.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                rel = (Path("toy") / path.relative_to(staging)).as_posix()
                handle.write(path, arcname=rel)
    (http_server.directory / "toy.zip").write_bytes(archive.read_bytes())
    dataset = RemoteDataset(
        name="toy", url=f"{http_server.base}/toy.zip", size=archive.stat().st_size
    )
    data_dir = tmp_path / "data"

    root = ensure_dataset(dataset, data_dir)

    assert root == data_dir / "toy"
    assert (root / "CSVs").is_dir()
    assert not (data_dir / "toy.zip").exists()
