"""Tests for the remote zip reader, against a local Range-capable HTTP server."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rsss.data.remote_zip import HTTPRangeFile, fetch_members, member_infos

if TYPE_CHECKING:
    from tests.conftest import RangeServer

# The three members of the synthetic archive, one of each kind the staging code
# requests: an image, its mask, and a tile the tests leave behind.
_ZIP_MEMBERS = {
    "flair_aerial_train/D001/z1/img/IMG_000001.tif": b"image-one" * 1000,
    "flair_labels_train/D001/z1/msk/MSK_000001.tif": b"mask-one",
    "flair_aerial_train/D002/z1/img/IMG_000002.tif": b"image-two",
}
_WANTED = [
    "flair_aerial_train/D001/z1/img/IMG_000001.tif",
    "flair_labels_train/D001/z1/msk/MSK_000001.tif",
]


def _write_zip(path: Path, members: dict[str, bytes]) -> int:
    """Write ``members`` as a zip, deflating the larger ones; return its size."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            method = zipfile.ZIP_DEFLATED if len(data) > 1000 else zipfile.ZIP_STORED
            archive.writestr(name, data, compress_type=method)
    return path.stat().st_size


@pytest.fixture
def served_zip(range_server: RangeServer) -> tuple[str, int, dict[str, bytes]]:
    """Serve a synthetic FLAIR-shaped archive; return its URL, size and members."""
    path = range_server.root / "dataset.zip"
    size = _write_zip(path, _ZIP_MEMBERS)
    return range_server.route("dataset.zip"), size, _ZIP_MEMBERS


def _destinations(tmp_path: Path, names: list[str]) -> dict[str, Path]:
    """Map member names to paths under ``tmp_path/out``, keeping their tree."""
    return {name: tmp_path / "out" / name for name in names}


def test_http_range_file_reads_and_seeks_like_a_local_file(
    range_server: RangeServer,
) -> None:
    payload = bytes(range(256)) * (4 << 10)  # 1 MiB
    (range_server.root / "blob.bin").write_bytes(payload)
    url = range_server.route("blob.bin")

    with HTTPRangeFile(url, len(payload), block_size=256 << 10) as remote:
        assert remote.seekable() and remote.readable() and not remote.writable()
        assert remote.read(1000) == payload[:1000]
        assert remote.tell() == 1000
        buffer = bytearray(10)
        assert remote.readinto(buffer) == 10
        assert buffer == payload[1000:1010]

        assert remote.seek(0) == 0
        assert remote.read() == payload
        assert remote.seek(-16, 2) == len(payload) - 16
        assert remote.read(4) == payload[-16:-12]
        assert remote.read(0) == b""
        assert remote.bytes_downloaded >= len(payload)

    with pytest.raises(ValueError):
        remote.seek(-1)


def test_http_range_file_refuses_a_short_file(range_server: RangeServer) -> None:
    (range_server.root / "short.bin").write_bytes(b"only-100-bytes" * 7)
    url = range_server.route("short.bin")

    with HTTPRangeFile(url, 10_000) as remote, pytest.raises(RuntimeError):
        remote.read()


def test_http_range_file_refuses_a_server_ignoring_ranges(
    range_server: RangeServer,
) -> None:
    (range_server.root / "blob.bin").write_bytes(b"data" * 100)
    range_server.ignore_range = True

    with (
        HTTPRangeFile(range_server.route("blob.bin"), 400) as remote,
        pytest.raises(RuntimeError, match="range"),
    ):
        remote.read()


def test_member_infos_sizes_the_archive_without_extracting_it(
    served_zip: tuple[str, int, dict[str, bytes]],
) -> None:
    url, size, members = served_zip

    infos = member_infos(url, size)

    assert infos == {name: len(data) for name, data in members.items()}


def test_fetch_members_extracts_only_the_requested_members(
    served_zip: tuple[str, int, dict[str, bytes]], tmp_path: Path
) -> None:
    url, size, members = served_zip
    destinations = _destinations(tmp_path, _WANTED)

    report = fetch_members(url, size, destinations, workers=2)

    assert (report.extracted, report.skipped) == (2, 0)
    assert report.bytes_downloaded > 0
    for name, dest in destinations.items():
        assert dest.read_bytes() == members[name]
    # The unrequested tile's tree must not be created, not just left empty.
    assert not (tmp_path / "out" / "flair_aerial_train" / "D002").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_members_skips_members_already_extracted(
    served_zip: tuple[str, int, dict[str, bytes]], tmp_path: Path
) -> None:
    url, size, members = served_zip
    destinations = _destinations(tmp_path, _WANTED)
    fetch_members(url, size, destinations)

    report = fetch_members(url, size, destinations)

    assert (report.extracted, report.skipped) == (0, len(destinations))
    for name, dest in destinations.items():
        assert dest.read_bytes() == members[name]


def test_fetch_members_rejects_an_unknown_member(
    served_zip: tuple[str, int, dict[str, bytes]], tmp_path: Path
) -> None:
    url, size, _ = served_zip

    with pytest.raises(ValueError, match="missing from the archive"):
        fetch_members(
            url, size, {"flair_aerial_train/D404/img/IMG_404.tif": tmp_path / "x"}
        )
    assert not (tmp_path / "x").exists()


def test_fetch_members_retries_a_transient_failure(
    served_zip: tuple[str, int, dict[str, bytes]],
    range_server: RangeServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Do not actually wait out the backoff in a unit test.
    monkeypatch.setattr("rsss.data.remote_zip.time.sleep", lambda _seconds: None)
    range_server.failures = 1
    url, size, _ = served_zip

    report = fetch_members(url, size, _destinations(tmp_path, _WANTED[:1]))

    assert report.extracted == 1


def test_fetch_members_without_members_touches_no_server() -> None:
    report = fetch_members("http://127.0.0.1:9/never.zip", 10, {})

    assert (report.extracted, report.skipped, report.bytes_downloaded) == (0, 0, 0)
