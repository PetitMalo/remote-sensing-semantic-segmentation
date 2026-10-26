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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

import rasterio
from loguru import logger

from rsss.data.remote_zip import fetch_members, member_infos
from rsss.data.splits import CSV_SUBDIR, build_flair_splits, split_of_csv
from rsss.data.subset import (
    SUBSET_ROOT_NAME,
    SubsetSpec,
    build_subset_csvs,
    manifest_matches,
    read_manifest,
    write_manifest,
)

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


@dataclass(frozen=True)
class SubsetDataset:
    """A whole-domain subset of a :class:`FullDataset`'s archives.

    The full dataset is never staged: the subset keeps whole spatial domains per
    split (see :class:`rsss.data.subset.SubsetSpec`), fetches only the members
    the filtered CSVs name, directly out of the remote archives, and verifies
    the result like any other FLAIR#1 tree.

    Parameters
    ----------
    name : str
        Directory the subset is assembled into, under ``data_dir``.
    full : FullDataset
        Dataset to take the domains from; its archives are read remotely.
    spec : SubsetSpec
        Domains to keep per split.
    """

    name: str
    full: FullDataset
    spec: SubsetSpec


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

Dataset = RemoteDataset | FullDataset | SubsetDataset

# FLAIR#1 subset: the recipe that fits a 75 GiB staging budget, measured from the
# remote central directories (58,912 tiles, 72.6 GiB). Domains are taken
# alphabetically, so this selects the same tiles on every machine: the first 21
# train domains, all 8 validation domains, all 10 test.
SUBSET_DATASET = SubsetDataset(
    name=SUBSET_ROOT_NAME,
    full=FULL_DATASET,
    spec=SubsetSpec(),
)

# Downloadable datasets, keyed by the name a data config may use.
DATASETS: dict[str, Dataset] = {
    "flair_1_toy": TOY_DATASET,
    "flair_1_full": FULL_DATASET,
    "flair_1_subset": SUBSET_DATASET,
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


def _stage_csvs(
    csv_urls: Mapping[str, str], root: Path, *, subdir: str = CSV_SUBDIR
) -> None:
    """Download any missing split CSV into ``<root>/<subdir>``."""
    csv_dir = root / subdir
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


# Official CSVs are kept pristine under this subdirectory, so a rerun always
# filters from the full lists rather than from an already-filtered subset.
_SOURCE_CSV_SUBDIR = "source_csvs"


def members_for_archive(
    archive: RemoteArchive,
    full_name: str,
    dataset_root: str | Path,
    wanted: Iterable[Path],
) -> dict[str, Path]:
    """Map the wanted files ``archive`` provides to their member names.

    ``archive.extract_to`` is relative to the data directory while
    ``archive.provides`` is relative to the dataset root; the subset mirrors the
    full layout under its own root, so the member name is the file's path
    relative to the archive's extraction target.

    Parameters
    ----------
    archive : RemoteArchive
        Archive to map the files onto.
    full_name : str
        Directory name of the full dataset the archive paths are under.
    dataset_root : str | Path
        Root of the subset tree being staged.
    wanted : Iterable[Path]
        Every file the filtered CSVs name, images and masks alike.

    Returns
    -------
    dict[str, Path]
        Member name to destination path, for the files this archive provides.
    """
    extract_rel = archive.extract_to.relative_to(full_name)
    target_root = (Path(dataset_root) / extract_rel).resolve()
    prefix = archive.provides.relative_to(extract_rel)
    members: dict[str, Path] = {}
    for path in wanted:
        path = Path(path)
        if not path.is_relative_to(target_root):
            continue
        rel = path.relative_to(target_root)
        if prefix.parts and rel.parts[: len(prefix.parts)] != prefix.parts:
            continue
        members[str(rel)] = path
    return members


def _wanted_files(root: Path) -> list[Path]:
    """Return every image and mask path the subset's filtered CSVs name."""
    splits = build_flair_splits(root)
    return [
        Path(path) for pair in splits.values() for path in (*pair["IMG"], *pair["MSK"])
    ]


def subset_split_sizes(dataset: SubsetDataset, root: str | Path) -> dict[str, int]:
    """Return the bytes each split's tiles and masks would take on disk.

    Sizes come from the remote archives' central directories: only the archive
    tables are read, never a member. It is what ``--report-only`` prints before
    a spec is committed to disk.

    Parameters
    ----------
    dataset : SubsetDataset
        Subset whose archives hold the files.
    root : str | Path
        Subset root, holding the filtered CSVs.

    Returns
    -------
    dict[str, int]
        Bytes per split.

    Raises
    ------
    ValueError
        If a file named by the CSVs belongs to none of the archives.
    """
    root = Path(root)
    splits = build_flair_splits(root)
    wanted = [
        Path(path) for pair in splits.values() for path in (*pair["IMG"], *pair["MSK"])
    ]
    sizes: dict[str, int] = {}
    for archive in dataset.full.archives:
        members = members_for_archive(archive, dataset.full.name, root, wanted)
        if not members:
            continue
        infos = member_infos(archive.url, archive.size)
        for member, dest in members.items():
            if member not in infos:
                raise ValueError(
                    f"{archive_filename(archive.url)}: {member} is not in the archive."
                )
            sizes[str(dest)] = infos[member]
    missing = [path for path in wanted if str(path) not in sizes]
    if missing:
        raise ValueError(
            f"{len(missing)} subset file(s) belong to no archive, e.g. {missing[0]}."
        )
    return {
        split: sum(sizes[str(Path(path))] for path in (*pair["IMG"], *pair["MSK"]))
        for split, pair in splits.items()
    }


def ensure_subset_dataset(
    dataset: SubsetDataset,
    data_dir: str | Path = "data",
    *,
    force: bool = False,
    report_only: bool = False,
    workers: int = 8,
) -> Path:
    """Stage and verify a whole-domain subset, fetching only its tiles.

    The official path CSVs are downloaded once into ``source_csvs/``, filtered
    into ``CSVs/`` for ``dataset.spec``, and recorded in
    ``subset_manifest.json``. The tiles the filtered CSVs name are then fetched
    one member at a time out of the remote archives, so the full dataset never
    touches the disk, and the tree is verified like any other FLAIR#1 dataset.
    Members already present are skipped, which makes an interrupted staging run
    resumable.

    Parameters
    ----------
    dataset : SubsetDataset
        Subset to stage.
    data_dir : str | Path
        Directory the subset is assembled under.
    force : bool
        Delete the subset root and stage it again.
    report_only : bool
        Stop after the filtered CSVs and log what each split would cost.
    workers : int
        Concurrent downloads per archive.

    Returns
    -------
    Path
        Root of the filtered, verified subset.

    Raises
    ------
    ValueError
        If ``root`` holds a subset staged with another spec, or if the staged
        tree fails verification.
    """
    data_dir = Path(data_dir)
    root = data_dir / dataset.name
    if force:
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(root)
    if manifest is not None and not manifest_matches(manifest, dataset.spec):
        raise ValueError(
            f"{root} holds a subset staged with spec {manifest.get('spec')}; "
            f"requested {dataset.spec.as_text()}. Rebuild it with --force, or "
            f"point --data-dir elsewhere."
        )
    if manifest is None:
        _stage_csvs(dataset.full.csv_urls, root, subdir=_SOURCE_CSV_SUBDIR)
        sources: dict[str, Path] = {}
        for name in dataset.full.csv_urls:
            source = root / _SOURCE_CSV_SUBDIR / name
            split = split_of_csv(source)
            if split is None:
                raise ValueError(f"{source}: name does not end in a FLAIR split.")
            sources[split] = source
        selections = build_subset_csvs(sources, root, dataset.spec)
        write_manifest(root, dataset.spec, selections)
        manifest = read_manifest(root)
    if manifest is None:  # pragma: no cover - write_manifest just wrote it
        raise RuntimeError(f"{root}: subset manifest could not be written.")
    for split, info in sorted(manifest["splits"].items()):
        logger.info(
            "{}: {} tiles over {}/{} domains kept",
            split,
            info["tiles"],
            len(info["domains"]),
            len(info["available"]),
        )

    if report_only:
        sizes = subset_split_sizes(dataset, root)
        for split, size in sizes.items():
            logger.info("{}: {:.1f} MiB to fetch", split, size / 2**20)
        logger.info(
            "{}: {:.1f} MiB in total", dataset.name, sum(sizes.values()) / 2**20
        )
        return root

    try:
        report = verify_flair_dataset(root)
    except (FileNotFoundError, ValueError):
        pass
    else:
        logger.info("{} is already available ({}).", dataset.name, report.summary())
        return root

    wanted = _wanted_files(root)
    fetched = 0
    for archive in dataset.full.archives:
        members = members_for_archive(archive, dataset.full.name, root, wanted)
        if not members:
            continue
        logger.info(
            "Fetching {} member(s) from {}",
            len(members),
            archive_filename(archive.url),
        )
        fetch_report = fetch_members(
            archive.url, archive.size, members, workers=workers
        )
        logger.info("{}: {}", archive_filename(archive.url), fetch_report.summary())
        fetched += fetch_report.bytes_downloaded

    try:
        report = verify_flair_dataset(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            f"{dataset.name}: staging finished but verification failed ({error}). "
            f"Rerun with --force to rebuild {root}."
        ) from error

    logger.info(
        "Verified {} ({}, {:.1f} MiB fetched).",
        dataset.name,
        report.summary(),
        fetched / 2**20,
    )
    return root


def ensure_dataset(
    dataset: Dataset = TOY_DATASET,
    data_dir: str | Path = "data",
    *,
    force: bool = False,
    keep_zip: bool = False,
    report_only: bool = False,
    workers: int = 8,
) -> Path:
    """Make sure ``dataset`` is present, extracted and verified under data_dir.

    An already-extracted dataset that passes :func:`verify_flair_dataset` is
    reused as is. Otherwise the archive is downloaded (or a complete local one
    reused) and extracted, the result is verified, and the archive is deleted
    unless ``keep_zip``. A bad toy tree is dropped and rebuilt; a bad full tree
    is kept, since rebuilding it costs a full re-download. A
    :class:`SubsetDataset` is delegated to :func:`ensure_subset_dataset`.

    Parameters
    ----------
    dataset : Dataset
        Dataset to stage; defaults to the FLAIR#1 toy.
    data_dir : str | Path
        Directory the dataset is assembled under (the repo's ``data/``).
    force : bool
        Re-download even when a verified copy already exists.
    keep_zip : bool
        Keep the archive(s) after a successful verify; ignored by subsets,
        which never stage a full archive.
    report_only : bool
        For a subset, stop after the filtered CSVs and report what it would
        cost, without fetching any tile.
    workers : int
        Concurrent downloads for a subset.

    Returns
    -------
    Path
        Root of the extracted, verified dataset.
    """
    if isinstance(dataset, SubsetDataset):
        return ensure_subset_dataset(
            dataset,
            data_dir,
            force=force,
            report_only=report_only,
            workers=workers,
        )
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
    parser.add_argument(
        "--num-domains",
        default=None,
        metavar="TRAIN,VALIDATION,TEST",
        help="Domains kept per split for --dataset flair_1_subset, "
        "alphabetically first (default: 21,8,10).",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="For --dataset flair_1_subset: stage the filtered CSVs and report "
        "what each split would cost, without fetching any tile.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent downloads per archive for flair_1_subset "
        "(default: %(default)s).",
    )
    args = parser.parse_args(argv)
    dataset = DATASETS[args.dataset]

    if args.num_domains is not None:
        if not isinstance(dataset, SubsetDataset):
            parser.error("--num-domains only applies to --dataset flair_1_subset.")
        try:
            dataset = replace(dataset, spec=SubsetSpec.from_text(args.num_domains))
        except ValueError as error:
            parser.error(str(error))
    if args.report_only:
        if not isinstance(dataset, SubsetDataset):
            parser.error("--report-only only applies to --dataset flair_1_subset.")
        if args.force or args.verify_only:
            parser.error("--report-only cannot be combined with --force/--verify-only.")

    if args.verify_only:
        report = verify_flair_dataset(Path(args.data_dir) / dataset.name)
        logger.info(report.summary())
    else:
        ensure_dataset(
            dataset,
            args.data_dir,
            force=args.force,
            keep_zip=args.keep_zip,
            report_only=args.report_only,
            workers=args.workers,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
