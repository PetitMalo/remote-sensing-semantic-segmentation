"""Download and verify a FLAIR#1 dataset (the toy, by default).

The archives are hosted on IGN's OVH Swift storage, linked from the FLAIR
website. The toy is a single self-contained archive; the full dataset is four
archives plus the split CSVs hosted in the FLAIR-1 repository. Either way, this
module assembles the tree and checks it against those path CSVs, so a truncated
download or a bad archive fails here rather than midway through a training job
on the cluster.

Run it with ``uv run python -m rsss.data.download`` (``-h`` for the options).
"""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import rasterio
from loguru import logger

from rsss.data.splits import CSV_SUBDIR, build_flair_splits

# IGN's OVH Swift container holding the official FLAIR#1 archives.
_FLAIR_1_BASE_URL = (
    "https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461"
    "/defi-ia/flair_data_1"
)

# FLAIR tiles are 512x512 patches: 5-band uint8 imagery (RGB, NIR, nDSM) and
# 1-band uint8 labels encoding the 13 baseline classes as values 1..13.
TILE_SIZE = 512
NUM_BANDS = 5
NUM_CLASSES = 13

# Bytes read per chunk while streaming a download.
_CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class RemoteDataset:
    """A FLAIR#1 archive to fetch and the directory it extracts to.

    Parameters
    ----------
    name : str
        Directory the archive extracts to, under ``data_dir``.
    url : str
        Direct download URL of the zip archive.
    size : int
        Exact archive size in bytes. A download of the wrong size is left as
        ``<name>.zip.part`` for resuming instead of being promoted.
    """

    name: str
    url: str
    size: int


TOY_DATASET = RemoteDataset(
    name="flair_1_toy_dataset",
    url=f"{_FLAIR_1_BASE_URL}/flair_1_toy_dataset.zip",
    size=226_168_911,
)


@dataclass(frozen=True)
class RemoteArchive:
    """One archive of a dataset that is split across several downloads.

    Parameters
    ----------
    url : str
        Direct download URL of the zip archive.
    size : int
        Exact archive size in bytes.
    extract_to : Path
        Directory the archive extracts into, relative to ``data_dir``.
    provides : Path
        Top-level directory the archive creates, relative to the dataset root.
        It tells a resumed staging run which archives are already extracted.
    """

    url: str
    size: int
    extract_to: Path
    provides: Path


@dataclass(frozen=True)
class FullDataset:
    """A dataset assembled from several archives plus split CSVs.

    FLAIR#1 full ships four archives and keeps the train/validation/test paths
    in CSVs hosted in the FLAIR-1 repository, so no single archive can be
    extracted and verified on its own.

    Parameters
    ----------
    name : str
        Directory the dataset is assembled into, under ``data_dir``.
    archives : tuple[RemoteArchive, ...]
        Archives to download and extract, in order.
    csv_urls : Mapping[str, str]
        Split CSVs to fetch into ``<root>/CSVs``, keyed by their file name.
    """

    name: str
    archives: tuple[RemoteArchive, ...]
    csv_urls: Mapping[str, str]

    @property
    def size(self) -> int:
        """Total size of the archives, in bytes."""
        return sum(archive.size for archive in self.archives)


# The three split CSVs live in the FLAIR-1 repository, not in any archive.
_FLAIR_1_CSV_BASE_URL = "https://raw.githubusercontent.com/IGNF/FLAIR-1/main/csv_full"
_FLAIR_1_FULL_ROOT = "flair_1_full_dataset"

# FLAIR#1 full. The test split sits under ``data/`` because that is where the
# upstream test CSV's ``../data/flair_1_aerial_test`` paths point; the train
# archives extract at the dataset root, like their CSV rows expect.
FULL_DATASET = FullDataset(
    name=_FLAIR_1_FULL_ROOT,
    archives=(
        RemoteArchive(
            url=f"{_FLAIR_1_BASE_URL}/flair_aerial_train.zip",
            size=54_458_900_252,
            extract_to=Path(_FLAIR_1_FULL_ROOT),
            provides=Path("flair_aerial_train"),
        ),
        RemoteArchive(
            url=f"{_FLAIR_1_BASE_URL}/flair_labels_train.zip",
            size=509_091_073,
            extract_to=Path(_FLAIR_1_FULL_ROOT),
            provides=Path("flair_labels_train"),
        ),
        RemoteArchive(
            url=f"{_FLAIR_1_BASE_URL}/flair_1_aerial_test.zip",
            size=14_463_094_631,
            extract_to=Path(_FLAIR_1_FULL_ROOT) / "data",
            provides=Path("data/flair_1_aerial_test"),
        ),
        RemoteArchive(
            url=f"{_FLAIR_1_BASE_URL}/flair_1_labels_test.zip",
            size=130_481_211,
            extract_to=Path(_FLAIR_1_FULL_ROOT) / "data",
            provides=Path("data/flair_1_labels_test"),
        ),
    ),
    csv_urls={
        f"flair-1-paths-{split}.csv": f"{_FLAIR_1_CSV_BASE_URL}/flair-1-paths-{split}.csv"
        for split in ("train", "val", "test")
    },
)

Dataset = RemoteDataset | FullDataset

# Downloadable datasets, keyed by the name a data config may use.
DATASETS: dict[str, Dataset] = {
    "flair_1_toy": TOY_DATASET,
    "flair_1_full": FULL_DATASET,
}


def archive_filename(url: str) -> str:
    """Return the file name an archive URL downloads to."""
    return Path(urlparse(url).path).name


def download_file(url: str, dest: str | Path, *, size: int | None = None) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial ``<dest>.part`` file.

    The body goes to ``<dest>.part`` and is only moved into place once complete
    (and ``size``-checked), so an interrupted run never leaves a half-written
    archive that looks finished.

    Parameters
    ----------
    url : str
        Direct download URL.
    dest : str | Path
        Final path of the downloaded file.
    size : int, optional
        Expected size in bytes; a mismatch raises instead of promoting the file.

    Returns
    -------
    Path
        ``dest``, once complete.

    Raises
    ------
    RuntimeError
        If the download fails.
    ValueError
        If the downloaded size does not match ``size``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.is_file() else 0

    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with urllib.request.urlopen(request) as response:
            if offset and response.status != 206:
                # The server ignored the Range header: restart from scratch.
                logger.warning("{} ignored the range request; restarting.", url)
                offset = 0
            with part.open("ab" if offset else "wb") as sink:
                while chunk := response.read(_CHUNK_SIZE):
                    sink.write(chunk)
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not download {url}: {error}") from error

    downloaded = part.stat().st_size
    if size is not None and downloaded != size:
        raise ValueError(
            f"{url}: expected {size} bytes, got {downloaded}. The partial file "
            f"{part.name} is kept for resuming."
        )
    if offset:
        logger.info("Resumed {} from {} bytes.", dest.name, offset)
    part.replace(dest)
    return dest


def extract_zip(zip_path: str | Path, dest_dir: str | Path) -> None:
    """Extract ``zip_path`` into ``dest_dir``, rejecting escaping members.

    Parameters
    ----------
    zip_path : str | Path
        Archive to extract.
    dest_dir : str | Path
        Directory to extract into, created if needed.

    Raises
    ------
    ValueError
        If an archive member would land outside ``dest_dir`` (zip-slip).
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if not (dest_dir / member.filename).resolve().is_relative_to(root):
                raise ValueError(
                    f"{zip_path.name}: member {member.filename!r} escapes {dest_dir}."
                )
        archive.extractall(dest_dir)


@dataclass(frozen=True)
class DatasetVerification:
    """What a successful :func:`verify_flair_dataset` found."""

    root: Path
    tiles_per_split: dict[str, int]
    domains_per_split: dict[str, list[str]]

    def summary(self) -> str:
        """Return a one-line, human-readable recap of the splits."""
        return ", ".join(
            f"{split}: {self.tiles_per_split[split]} tiles"
            for split in ("train", "validation", "test")
        )


def _check_tile(
    path: Path,
    tile_size: int,
    expected_bands: int,
    num_classes: int,
    *,
    labels: bool,
) -> None:
    """Raise unless ``path`` matches the FLAIR tile spec."""
    with rasterio.open(path) as src:
        if src.count != expected_bands:
            raise ValueError(
                f"{path}: expected {expected_bands} band(s), got {src.count}."
            )
        if (src.width, src.height) != (tile_size, tile_size):
            raise ValueError(
                f"{path}: expected {tile_size}x{tile_size}, "
                f"got {src.width}x{src.height}."
            )
        if src.dtypes[0] != "uint8":
            raise ValueError(f"{path}: expected uint8, got {src.dtypes[0]}.")
        if labels:
            values = src.read(1)
            if values.min() < 1 or values.max() > num_classes:
                raise ValueError(
                    f"{path}: label values outside 1..{num_classes} "
                    f"({values.min()}..{values.max()})."
                )


def verify_flair_dataset(
    dataset_root: str | Path,
    *,
    tile_size: int = TILE_SIZE,
    num_bands: int = NUM_BANDS,
    num_classes: int = NUM_CLASSES,
) -> DatasetVerification:
    """Check an extracted FLAIR#1 dataset against its path CSVs.

    The CSVs define the split, so the checks start there: every ``IMG,MSK`` row
    must resolve to an existing file, the two lists must pair positionally, and
    the first tile of each split must match the FLAIR raster spec
    (``tile_size`` square, ``num_bands``-band uint8 imagery, 1-band uint8
    labels in ``1..num_classes``).

    Parameters
    ----------
    dataset_root : str | Path
        Root of an extracted FLAIR#1 dataset (toy or full).
    tile_size : int
        Expected tile side, in pixels.
    num_bands : int
        Expected number of image bands.
    num_classes : int
        Highest valid label value (labels are encoded 1..``num_classes``).

    Returns
    -------
    DatasetVerification
        Per-split tile counts and domains.

    Raises
    ------
    FileNotFoundError
        If a CSV-referenced image or label is missing.
    ValueError
        If a pair is mismatched or a sampled raster breaks the tile spec.
    """
    root = Path(dataset_root)
    splits = build_flair_splits(root)

    tiles_per_split: dict[str, int] = {}
    domains_per_split: dict[str, list[str]] = {}
    for split, pair in splits.items():
        imgs, msks = pair["IMG"], pair["MSK"]
        missing = [path for path in (*imgs, *msks) if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{split}: {len(missing)} CSV-referenced file(s) missing, "
                f"e.g. {missing[0]}"
            )
        for image, mask in zip(imgs, msks, strict=True):
            if Path(image).stem.replace("IMG_", "MSK_") != Path(mask).stem:
                raise ValueError(f"{split}: IMG/MSK pair mismatch: {image} vs {mask}.")
        tiles_per_split[split] = len(imgs)
        domains_per_split[split] = sorted({Path(path).parts[-4] for path in imgs})
        _check_tile(Path(imgs[0]), tile_size, num_bands, num_classes, labels=False)
        _check_tile(Path(msks[0]), tile_size, 1, num_classes, labels=True)

    return DatasetVerification(
        root=root,
        tiles_per_split=tiles_per_split,
        domains_per_split=domains_per_split,
    )


def _stage_archive(
    archive: RemoteArchive,
    data_dir: Path,
    root: Path,
    *,
    keep_zip: bool,
) -> None:
    """Download and extract one archive, deleting it unless ``keep_zip``.

    A failed extraction leaves a partial tree behind; the directory the archive
    provides is removed so the next run stages it again instead of skipping it.
    """
    zip_path = data_dir / archive_filename(archive.url)
    if zip_path.is_file() and zip_path.stat().st_size == archive.size:
        logger.info("Reusing the existing archive {}.", zip_path)
    else:
        logger.info("Downloading {} -> {}", archive.url, zip_path)
        download_file(archive.url, zip_path, size=archive.size)

    target = data_dir / archive.extract_to
    logger.info("Extracting {} into {}.", zip_path.name, target)
    try:
        extract_zip(zip_path, target)
    except Exception:
        shutil.rmtree(root / archive.provides, ignore_errors=True)
        raise

    if keep_zip:
        logger.info("Kept the archive {}.", zip_path.name)
    else:
        zip_path.unlink()


def _stage_csvs(csv_urls: Mapping[str, str], root: Path) -> None:
    """Download any missing split CSV into ``<root>/CSVs``."""
    csv_dir = root / CSV_SUBDIR
    for name, url in csv_urls.items():
        dest = csv_dir / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        logger.info("Downloading {} -> {}", url, dest)
        download_file(url, dest)


def ensure_full_dataset(
    dataset: FullDataset,
    data_dir: str | Path = "data",
    *,
    force: bool = False,
    keep_zip: bool = False,
) -> Path:
    """Assemble a multi-archive dataset and verify the result.

    Archives already extracted under the dataset root are skipped, so an
    interrupted staging run resumes instead of restarting. Unlike the toy
    dataset, an invalid existing tree is never deleted automatically: rebuilding
    it costs tens of gigabytes of downloads, so the caller has to pass
    ``force=True``.

    Parameters
    ----------
    dataset : FullDataset
        Dataset specification to stage.
    data_dir : str | Path
        Directory the dataset is assembled under.
    force : bool
        Delete the dataset root and stage every archive again.
    keep_zip : bool
        Keep the archives after extracting them.

    Returns
    -------
    Path
        Root of the assembled, verified dataset.

    Raises
    ------
    ValueError
        If the staged tree still fails verification.
    """
    data_dir = Path(data_dir)
    root = data_dir / dataset.name
    data_dir.mkdir(parents=True, exist_ok=True)

    if force:
        shutil.rmtree(root, ignore_errors=True)

    if root.is_dir():
        try:
            report = verify_flair_dataset(root)
        except (FileNotFoundError, ValueError) as error:
            logger.warning(
                "Existing {} is incomplete ({}); staging the missing parts.",
                root,
                error,
            )
        else:
            logger.info("{} is already available ({}).", dataset.name, report.summary())
            return root

    for archive in dataset.archives:
        if (root / archive.provides).is_dir():
            logger.info(
                "{} is already extracted; skipping {}.",
                archive.provides,
                archive_filename(archive.url),
            )
            continue
        _stage_archive(archive, data_dir, root, keep_zip=keep_zip)

    _stage_csvs(dataset.csv_urls, root)

    try:
        report = verify_flair_dataset(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            f"{dataset.name}: staging finished but verification failed ({error}). "
            f"Rerun with --force to rebuild {root}."
        ) from error

    logger.info("Verified {} ({}).", dataset.name, report.summary())
    return root


def ensure_dataset(
    dataset: Dataset = TOY_DATASET,
    data_dir: str | Path = "data",
    *,
    force: bool = False,
    keep_zip: bool = False,
) -> Path:
    """Make sure ``dataset`` is present, extracted and verified under data_dir.

    An already-extracted dataset that passes :func:`verify_flair_dataset` is
    reused as is. Otherwise the archive is downloaded (or a complete local one
    reused) and extracted, the result is verified, and the archive is deleted
    unless ``keep_zip``. A bad toy tree is dropped and rebuilt; a bad full tree
    is kept, since rebuilding it costs a full re-download.

    Parameters
    ----------
    dataset : RemoteDataset | FullDataset
        Dataset to stage; defaults to the FLAIR#1 toy.
    data_dir : str | Path
        Directory the dataset is assembled under (the repo's ``data/``).
    force : bool
        Re-download even when a verified copy already exists.
    keep_zip : bool
        Keep the archive(s) after a successful verify.

    Returns
    -------
    Path
        Root of the extracted, verified dataset.
    """
    if isinstance(dataset, FullDataset):
        return ensure_full_dataset(dataset, data_dir, force=force, keep_zip=keep_zip)

    data_dir = Path(data_dir)
    root = data_dir / dataset.name
    archive = data_dir / f"{dataset.name}.zip"

    if force:
        shutil.rmtree(root, ignore_errors=True)
        archive.unlink(missing_ok=True)

    if root.is_dir():
        try:
            report = verify_flair_dataset(root)
        except (FileNotFoundError, ValueError) as error:
            logger.warning("Existing {} is invalid ({}); re-downloading.", root, error)
            shutil.rmtree(root)
        else:
            logger.info("{} is already available ({}).", dataset.name, report.summary())
            return root

    if archive.is_file() and archive.stat().st_size == dataset.size:
        logger.info("Reusing the existing archive {}.", archive)
    else:
        logger.info("Downloading {} -> {}", dataset.url, archive)
        download_file(dataset.url, archive, size=dataset.size)

    logger.info("Extracting {} into {}.", archive.name, data_dir)
    extract_zip(archive, data_dir)

    report = verify_flair_dataset(root)
    logger.info("Verified {} ({}).", dataset.name, report.summary())

    if keep_zip:
        logger.info("Kept the archive {}.", archive.name)
    else:
        archive.unlink()
        logger.info("Removed the archive {}.", archive.name)
    return root


def main(argv: list[str] | None = None) -> int:
    """Download or verify a FLAIR#1 dataset."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--dataset",
        default="flair_1_toy",
        choices=sorted(DATASETS),
        help="Dataset to stage (default: %(default)s).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory the dataset lives under (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a verified copy already exists.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the archive(s) after a successful verify.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify an already-extracted dataset; never download.",
    )
    args = parser.parse_args(argv)
    dataset = DATASETS[args.dataset]

    if args.verify_only:
        report = verify_flair_dataset(Path(args.data_dir) / dataset.name)
        logger.info(report.summary())
    else:
        ensure_dataset(dataset, args.data_dir, force=args.force, keep_zip=args.keep_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
