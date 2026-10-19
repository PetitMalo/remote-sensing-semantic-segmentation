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
    FULL_DATASET,
    TOY_DATASET,
    FullDataset,
    RemoteArchive,
    RemoteDataset,
    archive_filename,
    download_file,
    ensure_dataset,
    extract_zip,
    main,
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


def test_full_dataset_points_at_four_archives_and_three_csvs():
    assert isinstance(FULL_DATASET, FullDataset)
    assert FULL_DATASET.name == "flair_1_full_dataset"

    archives = {archive_filename(a.url): a for a in FULL_DATASET.archives}
    assert set(archives) == {
        "flair_aerial_train.zip",
        "flair_labels_train.zip",
        "flair_1_aerial_test.zip",
        "flair_1_labels_test.zip",
    }
    assert archives["flair_aerial_train.zip"].size == 54_458_900_252
    assert archives["flair_labels_train.zip"].size == 509_091_073
    assert archives["flair_1_aerial_test.zip"].size == 14_463_094_631
    assert archives["flair_1_labels_test.zip"].size == 130_481_211
    # The upstream test CSV points at ../data/..., so those archives must land
    # there rather than at the dataset root.
    assert archives["flair_1_aerial_test.zip"].extract_to == Path(
        "flair_1_full_dataset/data"
    )
    assert archives["flair_1_aerial_test.zip"].provides == Path(
        "data/flair_1_aerial_test"
    )
    assert set(FULL_DATASET.csv_urls) == {
        "flair-1-paths-train.csv",
        "flair-1-paths-val.csv",
        "flair-1-paths-test.csv",
    }
    assert all("/csv_full/" in url for url in FULL_DATASET.csv_urls.values())
    assert FULL_DATASET.size == sum(a.size for a in FULL_DATASET.archives)


def test_archive_filename_reads_the_last_url_segment():
    assert (
        archive_filename("https://host.example/defi-ia/flair_data_1/a.zip?x=1")
        == "a.zip"
    )


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


# --------------------------------------------------------------------------- #
# ensure_dataset: full dataset                                                #
# --------------------------------------------------------------------------- #

# Archive name -> (extract_to, provides) mirroring FULL_DATASET.
FULL_LOCAL_LAYOUT = {
    "flair_aerial_train.zip": (
        Path("flair_1_full_dataset"),
        Path("flair_aerial_train"),
    ),
    "flair_labels_train.zip": (
        Path("flair_1_full_dataset"),
        Path("flair_labels_train"),
    ),
    "flair_1_aerial_test.zip": (
        Path("flair_1_full_dataset/data"),
        Path("data/flair_1_aerial_test"),
    ),
    "flair_1_labels_test.zip": (
        Path("flair_1_full_dataset/data"),
        Path("data/flair_1_labels_test"),
    ),
}

FULL_CSV_NAMES = (
    "flair-1-paths-train.csv",
    "flair-1-paths-val.csv",
    "flair-1-paths-test.csv",
)


def _write_full_staging(staging: Path, write_img, write_msk) -> None:
    """Write the four collections and three CSVs of a synthetic full dataset."""
    write_img(
        staging / "flair_aerial_train/D006_2020/Z1_UU/img/IMG_000001.tif", size=512
    )
    write_msk(
        staging / "flair_labels_train/D006_2020/Z1_UU/msk/MSK_000001.tif",
        np.ones((512, 512)),
    )
    write_img(
        staging / "flair_aerial_train/D004_2021/Z9_UN/img/IMG_000734.tif", size=512
    )
    write_msk(
        staging / "flair_labels_train/D004_2021/Z9_UN/msk/MSK_000734.tif",
        np.ones((512, 512)),
    )
    write_img(
        staging / "flair_1_aerial_test/D012_2019/Z3_UU/img/IMG_061946.tif", size=512
    )
    write_msk(
        staging / "flair_1_labels_test/D012_2019/Z3_UU/msk/MSK_061946.tif",
        np.ones((512, 512)),
    )

    csv_dir = staging / "csv"
    csv_dir.mkdir(parents=True)
    (csv_dir / "flair-1-paths-train.csv").write_text(
        "../flair_aerial_train/D006_2020/Z1_UU/img/IMG_000001.tif,"
        "../flair_labels_train/D006_2020/Z1_UU/msk/MSK_000001.tif\n"
    )
    (csv_dir / "flair-1-paths-val.csv").write_text(
        "../flair_aerial_train/D004_2021/Z9_UN/img/IMG_000734.tif,"
        "../flair_labels_train/D004_2021/Z9_UN/msk/MSK_000734.tif\n"
    )
    (csv_dir / "flair-1-paths-test.csv").write_text(
        "../data/flair_1_aerial_test/D012_2019/Z3_UU/img/IMG_061946.tif,"
        "../data/flair_1_labels_test/D012_2019/Z3_UU/msk/MSK_061946.tif\n"
    )


def _zip_tree(source: Path, archive: Path) -> Path:
    """Zip ``source`` so its own name is the archive's top-level directory."""
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(
                    path, arcname=(source.name / path.relative_to(source)).as_posix()
                )
    return archive


def _local_full_dataset(http_server) -> FullDataset:
    """Build FULL_DATASET's shape with local URLs and real archive sizes."""
    return FullDataset(
        name="flair_1_full_dataset",
        archives=tuple(
            RemoteArchive(
                url=f"{http_server.base}/{name}",
                size=(http_server.directory / name).stat().st_size,
                extract_to=extract_to,
                provides=provides,
            )
            for name, (extract_to, provides) in FULL_LOCAL_LAYOUT.items()
        ),
        csv_urls={name: f"{http_server.base}/csv/{name}" for name in FULL_CSV_NAMES},
    )


@pytest.fixture
def served_full_dataset(tmp_path, http_server, write_img, write_msk):
    """Serve a synthetic full dataset and return its local spec."""
    staging = tmp_path / "staging"
    _write_full_staging(staging, write_img, write_msk)
    for name in FULL_LOCAL_LAYOUT:
        _zip_tree(staging / name.removesuffix(".zip"), http_server.directory / name)
    (http_server.directory / "csv").mkdir()
    for name in FULL_CSV_NAMES:
        (http_server.directory / "csv" / name).write_bytes(
            (staging / "csv" / name).read_bytes()
        )
    return _local_full_dataset(http_server)


def test_ensure_full_dataset_stages_archives_and_csvs(tmp_path, served_full_dataset):
    data_dir = tmp_path / "data"

    root = ensure_dataset(served_full_dataset, data_dir)

    assert root == data_dir / "flair_1_full_dataset"
    assert (root / "CSVs/flair-1-paths-train.csv").is_file()
    assert (root / "CSVs/flair-1-paths-val.csv").is_file()
    assert (root / "CSVs/flair-1-paths-test.csv").is_file()
    assert (root / "flair_aerial_train/D006_2020/Z1_UU/img/IMG_000001.tif").is_file()
    assert (
        root / "data/flair_1_aerial_test/D012_2019/Z3_UU/img/IMG_061946.tif"
    ).is_file()
    assert not list(data_dir.glob("*.zip"))
    assert verify_flair_dataset(root).tiles_per_split == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_ensure_full_dataset_reuses_a_verified_tree(
    tmp_path, served_full_dataset, http_server
):
    data_dir = tmp_path / "data"
    root = ensure_dataset(served_full_dataset, data_dir)
    requests_after_staging = len(http_server.ranges)

    reused = ensure_dataset(served_full_dataset, data_dir)

    assert reused == root
    assert len(http_server.ranges) == requests_after_staging


def test_ensure_full_dataset_keeps_an_invalid_tree(
    tmp_path, served_full_dataset, write_msk
):
    data_dir = tmp_path / "data"
    root = ensure_dataset(served_full_dataset, data_dir)
    write_msk(
        root / "data/flair_1_labels_test/D012_2019/Z3_UU/msk/MSK_061946.tif",
        np.full((512, 512), 14),
    )

    with pytest.raises(ValueError, match="--force"):
        ensure_dataset(served_full_dataset, data_dir)

    # Rebuilding costs a full re-download, so the tree must survive the error.
    assert (root / "flair_aerial_train/D006_2020/Z1_UU/img/IMG_000001.tif").is_file()


def test_main_selects_the_full_dataset(tmp_path, monkeypatch):
    calls: list[tuple] = []

    def _ensure(dataset, data_dir, **kwargs):
        calls.append((dataset, Path(data_dir)))
        return Path(data_dir) / dataset.name

    monkeypatch.setattr("rsss.data.download.ensure_dataset", _ensure)

    main(["--dataset", "flair_1_full", "--data-dir", str(tmp_path)])

    assert calls == [(FULL_DATASET, tmp_path)]
