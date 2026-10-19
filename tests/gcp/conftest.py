"""Shared fixtures for the GCP tests.

Every test replaces the command runner, so nothing here touches GCP, GCS or a
network: :class:`FakeRunner` records the commands a module would issue and can
report selected arguments as missing, which is how ``gcloud storage ls``
failures are simulated.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest


class FakeRunner:
    """Record commands, and exit non-zero when one holds a ``missing`` value."""

    def __init__(self, missing: Sequence[str] = (), stdout: str = "") -> None:
        self.calls: list[list[str]] = []
        self.missing = set(missing)
        self.stdout = stdout

    def __call__(
        self,
        command: Sequence[str],
        *,
        check: bool = True,
        capture_output: bool = False,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        recorded = [str(part) for part in command]
        self.calls.append(recorded)
        returncode = 1 if self.missing & set(recorded) else 0
        result = subprocess.CompletedProcess(
            recorded, returncode, stdout=self.stdout, stderr=""
        )
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, recorded)
        return result

    def commands(self, *prefix: str) -> list[list[str]]:
        """Return the recorded commands that start with ``prefix``."""
        return [call for call in self.calls if call[: len(prefix)] == list(prefix)]


@pytest.fixture
def fake_runner() -> FakeRunner:
    """Return a runner that records commands and always succeeds."""
    return FakeRunner()


@pytest.fixture
def make_runner() -> type[FakeRunner]:
    """Return the runner class, to build one with specific ``missing`` values."""
    return FakeRunner
