"""Extract selected members of a remote zip over HTTP Range requests.

FLAIR#1 ships as multi-gigabyte archives on IGN's OVH Swift storage, and a
cluster login node does not have room for them next to their extracted tree.
This module opens such an archive remotely: the stdlib :mod:`zipfile` parses the
central directory and decompresses, while a seekable file object backed by
ranged ``GET``s fetches only the byte spans the requested members occupy. The
disk cost of staging a subset is then proportional to the subset, not to the
archive.
"""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from loguru import logger

# Bytes fetched per ranged GET, unless a caller reads across a larger span.
_BLOCK_SIZE = 1 << 20  # 1 MiB
# Seconds before a stalled ranged GET is abandoned, and attempts made on
# transient failures (5xx, 429, dropped connections) before giving up.
_TIMEOUT = 60.0
_MAX_RETRIES = 5
# Bytes per copy when writing a member out.
_COPY_CHUNK = 1 << 20
# Members fetched between two progress log lines.
_LOG_EVERY = 500


def _is_transient(error: urllib.error.URLError | TimeoutError) -> bool:
    """Return whether retrying the request that raised ``error`` may work."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or error.code >= 500
    return True


class HTTPRangeFile:
    """A read-only, seekable file object backed by ranged HTTP GETs.

    It is what lets :class:`zipfile.ZipFile` open an archive that lives on a
    remote server: reads are served from a block cache filled by ``Range``
    requests, and seeking outside the cached block simply refills it. The server
    must honour ``Range`` (answering ``206 Partial Content``); the IGN archives
    do, since the downloader's resume logic already relies on it.

    Parameters
    ----------
    url : str
        URL of the file.
    size : int
        Exact file size in bytes, used to clamp reads.
    block_size : int
        Bytes fetched per ranged GET.
    timeout : float
        Seconds before a single request is abandoned.
    max_retries : int
        Attempts made on transient failures before raising.

    Raises
    ------
    ValueError
        If ``size`` is not positive.
    """

    def __init__(
        self,
        url: str,
        size: int,
        *,
        block_size: int = _BLOCK_SIZE,
        timeout: float = _TIMEOUT,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        if size <= 0:
            raise ValueError(f"{url}: size must be positive, got {size}.")
        self.url = url
        self.size = size
        self.block_size = block_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.bytes_downloaded = 0
        self._pos = 0
        self._block = b""
        self._block_start = 0

    def readable(self) -> bool:
        """Return True: the object only ever reads."""
        return True

    def writable(self) -> bool:
        """Return False: the remote file is never written to."""
        return False

    def seekable(self) -> bool:
        """Return True: :class:`zipfile.ZipFile` needs random access."""
        return True

    def tell(self) -> int:
        """Return the current byte offset."""
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move to ``offset``, resolving ``whence`` like a regular file.

        Parameters
        ----------
        offset : int
            Target offset, relative to ``whence``.
        whence : int
            ``0`` (start), ``1`` (current position) or ``2`` (end).

        Raises
        ------
        ValueError
            If ``whence`` is unknown or the target lands before the start.
        """
        if whence == 0:
            target = offset
        elif whence == 1:
            target = self._pos + offset
        elif whence == 2:
            target = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if target < 0:
            raise ValueError(f"cannot seek to {target} in {self.url}.")
        self._pos = target
        return target

    def read(self, n: int = -1) -> bytes:
        """Read up to ``n`` bytes, or everything left when ``n`` is negative."""
        if n < 0:
            parts: list[bytes] = []
            while chunk := self._read_at_most(self.block_size):
                parts.append(chunk)
            return b"".join(parts)
        return self._read_at_most(n)

    def readinto(self, buffer: bytearray) -> int:
        """Fill ``buffer`` with bytes from the current position; return the count."""
        chunk = self.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)

    def close(self) -> None:
        """Drop the cached block; there is no connection to release."""
        self._block = b""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _read_at_most(self, n: int) -> bytes:
        """Return at most ``n`` bytes from the current position."""
        end = min(self.size, self._pos + n)
        if end <= self._pos:
            return b""
        if not (
            self._block_start <= self._pos < self._block_start + len(self._block)
            and end <= self._block_start + len(self._block)
        ):
            stop = min(self.size, max(end, self._pos + self.block_size))
            self._block = self._fetch(self._pos, stop - 1)
            self._block_start = self._pos
        start = self._pos - self._block_start
        chunk = self._block[start : start + (end - self._pos)]
        self._pos += len(chunk)
        return chunk

    def _fetch(self, start: int, end: int) -> bytes:
        """Fetch the inclusive byte span ``start..end``, retrying transient errors.

        Raises
        ------
        RuntimeError
            If the server ignores ranges, returns a short body, or keeps failing.
        """
        length = end - start + 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"}
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"{self.url} answered {response.status} to a ranged GET; "
                            "it must honour Range requests (206 Partial Content)."
                        )
                    data = response.read()
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if not _is_transient(error) or attempt + 1 == self.max_retries:
                    raise RuntimeError(
                        f"Could not read bytes {start}-{end} of {self.url}: {error}"
                    ) from error
                time.sleep(2**attempt)
            else:
                if len(data) != length:
                    raise RuntimeError(
                        f"{self.url}: asked for {length} bytes at {start}, got "
                        f"{len(data)}; the file may be shorter than stated."
                    )
                self.bytes_downloaded += len(data)
                return data
        raise RuntimeError(
            f"Could not read bytes {start}-{end} of {self.url}: {last_error}"
        )


@dataclass(frozen=True)
class FetchReport:
    """What one :func:`fetch_members` call extracted and read."""

    url: str
    extracted: int
    skipped: int
    bytes_downloaded: int

    def summary(self) -> str:
        """Return a one-line, human-readable recap of the transfer."""
        return (
            f"{self.extracted} extracted, {self.skipped} already present, "
            f"{self.bytes_downloaded / 2**20:.1f} MiB read"
        )


def member_infos(
    url: str,
    size: int,
    *,
    block_size: int = _BLOCK_SIZE,
    timeout: float = _TIMEOUT,
    max_retries: int = _MAX_RETRIES,
) -> dict[str, int]:
    """Return every member's size in bytes, read from the remote central directory.

    Only the archive tables are fetched, never a member, so this is what sizes a
    staging run before any tile is read.

    Parameters
    ----------
    url : str
        URL of the remote zip archive.
    size : int
        Exact archive size in bytes.
    block_size, timeout, max_retries
        Passed to :class:`HTTPRangeFile`.

    Returns
    -------
    dict[str, int]
        Member name to its uncompressed size in bytes.
    """
    remote = HTTPRangeFile(
        url, size, block_size=block_size, timeout=timeout, max_retries=max_retries
    )
    try:
        with zipfile.ZipFile(remote) as archive:
            return {info.filename: info.file_size for info in archive.infolist()}
    finally:
        remote.close()


def _fetch_group(
    url: str,
    size: int,
    members: Mapping[str, Path],
    *,
    block_size: int,
    timeout: float,
    max_retries: int,
) -> tuple[int, int, int]:
    """Extract one group of members with its own reader; return counts and bytes.

    Each worker gets its own :class:`HTTPRangeFile` because it caches a block and
    a position, which threads cannot share.
    """
    extracted = 0
    skipped = 0
    remote = HTTPRangeFile(
        url, size, block_size=block_size, timeout=timeout, max_retries=max_retries
    )
    try:
        with zipfile.ZipFile(remote) as archive:
            infos = {info.filename: info for info in archive.infolist()}
            for done, (name, dest) in enumerate(members.items(), start=1):
                info = infos[name]
                if dest.is_file() and dest.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_name(dest.name + ".part")
                with archive.open(info) as source, part.open("wb") as sink:
                    shutil.copyfileobj(source, sink, _COPY_CHUNK)
                part.replace(dest)
                extracted += 1
                if done % _LOG_EVERY == 0:
                    logger.info("{}: {}/{} members", url, done, len(members))
    finally:
        remote.close()
    return extracted, skipped, remote.bytes_downloaded


def fetch_members(
    url: str,
    size: int,
    members: Mapping[str, str | Path],
    *,
    workers: int = 8,
    block_size: int = _BLOCK_SIZE,
    timeout: float = _TIMEOUT,
    max_retries: int = _MAX_RETRIES,
) -> FetchReport:
    """Extract named members of a remote zip, fetching only their byte spans.

    Members already present at their destination with the archived size are
    skipped, so an interrupted staging run resumes where it stopped. Each file
    lands through a ``<name>.part`` temporary, so a failed read never leaves a
    truncated tile behind.

    Parameters
    ----------
    url : str
        URL of the remote zip archive.
    size : int
        Exact archive size in bytes.
    members : Mapping[str, str | Path]
        Archive member name mapped to the path it should be written to.
    workers : int
        Number of concurrent connections; each worker reads its own share of
        the members, since one archive reader cannot be shared across threads.
    block_size, timeout, max_retries
        Passed to :class:`HTTPRangeFile`.

    Returns
    -------
    FetchReport
        Extracted/skipped member counts and bytes read.

    Raises
    ------
    ValueError
        If a requested member is not in the archive.
    RuntimeError
        If the archive cannot be read.
    """
    destinations = {name: Path(dest) for name, dest in members.items()}
    if not destinations:
        return FetchReport(url=url, extracted=0, skipped=0, bytes_downloaded=0)

    present = member_infos(
        url, size, block_size=block_size, timeout=timeout, max_retries=max_retries
    )
    missing = sorted(name for name in destinations if name not in present)
    if missing:
        shown = ", ".join(missing[:3]) + (", ..." if len(missing) > 3 else "")
        raise ValueError(
            f"{url}: {len(missing)} requested member(s) missing from the "
            f"archive, e.g. {shown}."
        )

    workers = max(1, min(workers, len(destinations)))
    groups = [
        dict(list(destinations.items())[index::workers]) for index in range(workers)
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reports = list(
            pool.map(
                lambda group: _fetch_group(
                    url,
                    size,
                    group,
                    block_size=block_size,
                    timeout=timeout,
                    max_retries=max_retries,
                ),
                (group for group in groups if group),
            )
        )
    extracted = sum(report[0] for report in reports)
    skipped = sum(report[1] for report in reports)
    downloaded = sum(report[2] for report in reports)
    return FetchReport(
        url=url,
        extracted=extracted,
        skipped=skipped,
        bytes_downloaded=downloaded,
    )
