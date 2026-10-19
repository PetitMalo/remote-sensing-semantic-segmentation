"""Run external commands through an injectable seam.

Every GCP module takes a ``runner`` argument and defaults to :func:`run`, so a
test can record the commands a run would issue without a cloud account.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Sequence

Command = Sequence[str]
Runner = Callable[..., subprocess.CompletedProcess[str]]


def run(
    command: Command,
    *,
    check: bool = True,
    capture_output: bool = False,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` and return the completed process.

    Parameters
    ----------
    command : Sequence[str]
        Program and arguments, without a shell.
    check : bool
        Raise ``CalledProcessError`` on a non-zero exit status.
    capture_output : bool
        Capture stdout and stderr instead of inheriting them.
    cwd : str, optional
        Directory to run the command in.
    """
    return subprocess.run(
        list(command),
        check=check,
        capture_output=capture_output,
        text=True,
        cwd=cwd,
    )


def format_command(command: Command) -> str:
    """Return ``command`` as a copy-pasteable shell string."""
    return shlex.join(str(part) for part in command)
