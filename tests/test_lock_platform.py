"""Guard the PANDO runtime stack against wheels glibc 2.17 cannot load.

PANDO runs RHEL 7.8, so the `pando` dependency group must resolve to wheels
tagged manylinux2014 or older. A modern host cannot notice when that breaks --
this laptop happily installs manylinux_2_28 wheels -- and the cluster would only
report it after a failed `uv sync`. The property is therefore asserted against
`uv.lock` here, which the login node consumes verbatim through `uv sync --frozen`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = REPO_ROOT / "uv.lock"

# RHEL 7.8 (glibc 2.17), the interpreter pinned by .python-version.
MAX_GLIBC = (2, 17)
CPYTHON = (3, 12)

# Pre-PEP 600 platform tags, which manylinux_<major>_<minor> replaced. The floor
# of a wheel is the oldest tag it carries, so both spellings have to be read.
LEGACY_TAGS = (
    ("manylinux2014", (2, 17)),
    ("manylinux2010", (2, 12)),
    ("manylinux1_", (2, 5)),
)


def glibc_floor(wheel: str) -> tuple[int, int] | None:
    """Return the oldest glibc a manylinux wheel needs, or None if never pinned."""
    floors = [
        (int(major), int(minor))
        for major, minor in re.findall(r"manylinux_(\d+)_(\d+)_", wheel)
    ]
    floors += [floor for tag, floor in LEGACY_TAGS if tag in wheel]
    return min(floors) if floors else None


def is_cpython_312_compatible(python_tag: str, abi_tag: str) -> bool:
    """Whether a wheel targets the interpreter version running on PANDO."""
    if python_tag == "py3":
        return True
    if abi_tag == "abi3":
        match = re.match(r"cp(\d)(\d+)$", python_tag)
        return (
            match is not None and (int(match.group(1)), int(match.group(2))) <= CPYTHON
        )
    return python_tag == f"cp{CPYTHON[0]}{CPYTHON[1]}"


def wheel_names(entry: dict) -> list[str]:
    """Filenames of every wheel uv recorded for a locked package."""
    return [wheel["url"].rsplit("/", 1)[-1] for wheel in entry.get("wheels", [])]


def installable_on_pando(wheels: list[str]) -> bool:
    """Whether at least one wheel runs on glibc 2.17 with CPython 3.12."""
    if any(wheel.endswith("py3-none-any.whl") for wheel in wheels):
        return True
    for wheel in wheels:
        if "linux" not in wheel or "x86_64" not in wheel or "musl" in wheel:
            continue
        *_, python_tag, abi_tag, _platform = wheel[: -len(".whl")].rsplit("-", 3)
        if not is_cpython_312_compatible(python_tag, abi_tag):
            continue
        floor = glibc_floor(wheel)
        if floor is None or floor <= MAX_GLIBC:
            return True
    return False


def locked_packages() -> dict[tuple[str, str], dict]:
    """Every entry of uv.lock, keyed by lowercased name and version."""
    with LOCK_PATH.open("rb") as handle:
        lock = tomllib.load(handle)
    return {
        (package["name"].lower(), package["version"]): package
        for package in lock["package"]
    }


def selected_versions(group: str | None) -> dict[str, str]:
    """Resolved versions of a group's fork, read from the lockfile by uv."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")
    command = [
        "uv",
        "export",
        "--frozen",
        "--no-emit-project",
        "--no-dev",
        "--format",
        "requirements-txt",
        "--no-hashes",
    ]
    command += ["--no-default-groups", "--group", group] if group else []
    result = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return dict(re.findall(r"^([\w.-]+)==([^\s;]+)", result.stdout, re.MULTILINE))


def major_minor(version: str) -> tuple[int, int]:
    """First two components of a version, for ordering comparisons."""
    parts = version.split("+")[0].split(".")
    return int(parts[0]), int(parts[1])


@pytest.mark.parametrize(
    ("wheel", "expected"),
    [
        ("torch-2.6.0-cp312-cp312-manylinux1_x86_64.whl", (2, 5)),
        ("numpy-2.2.6-cp312-cp312-manylinux_2_17_x86_64.whl", (2, 17)),
        ("rasterio-1.4.4-cp312-cp312-manylinux_2_28_x86_64.whl", (2, 28)),
        ("torch-2.14.1-cp312-cp312-manylinux1_x86_64.whl", (2, 5)),
        (
            "psutil-7.2.2-cp36-abi3-manylinux2010_x86_64.manylinux_2_28_x86_64.whl",
            (2, 12),
        ),
        ("numpy-2.5.4-cp312-cp312-macosx_14_0_arm64.whl", None),
    ],
)
def test_glibc_floor(wheel: str, expected: tuple[int, int] | None) -> None:
    assert glibc_floor(wheel) == expected


@pytest.mark.parametrize(
    ("wheels", "expected"),
    [
        (["loguru-0.7.3-py3-none-any.whl"], True),
        (
            [
                "hf_xet-1.7.0-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
                "hf_xet-1.7.0-cp38-abi3-musllinux_1_2_x86_64.whl",
            ],
            True,
        ),
        (["rasterio-1.4.4-cp312-cp312-manylinux_2_28_x86_64.whl"], False),
        (["numpy-2.5.4-cp312-cp312-macosx_14_0_arm64.whl"], False),
        (["numpy-2.5.4-cp314-cp314-manylinux_2_17_x86_64.whl"], False),
    ],
)
def test_installable_on_pando(wheels: list[str], expected: bool) -> None:
    assert installable_on_pando(wheels) is expected


def test_major_minor() -> None:
    assert major_minor("2.6.0+cu124") == (2, 6)


def test_pando_group_only_selects_wheels_glibc_2_17_can_load() -> None:
    packages = locked_packages()
    offenders = []
    for name, version in sorted(selected_versions("pando").items()):
        entry = packages.get((name, version))
        assert entry is not None, f"{name}=={version} is missing from uv.lock"
        if not installable_on_pando(wheel_names(entry)):
            offenders.append(f"{name}=={version}")
    assert not offenders, (
        "the pando group resolves to packages that need glibc > 2.17 "
        "(or a source build): " + ", ".join(offenders)
    )


def test_pando_group_pins_an_older_stack_than_the_default() -> None:
    pando = selected_versions("pando")
    default = selected_versions(None)
    assert major_minor(pando["torch"]) < major_minor(default["torch"])
    assert major_minor(pando["numpy"]) < major_minor(default["numpy"])
