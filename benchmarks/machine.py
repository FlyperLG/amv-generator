import os
import platform
from typing import Any

from amv.config import FFMPEG
from amv.io.ffmpeg import run_subprocess


def get_machine_information() -> dict[str, Any]:
    """Describe the machine a run was measured on, so stored numbers can be compared later.

    Returns:
        dict[str, Any]: platform, processor, cpu_count (logical cores), python version and
            ffmpeg (the first line of `ffmpeg -version`).

    Raises:
        RuntimeError: If ffmpeg fails or takes longer than 5 s."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "ffmpeg": _get_ffmpeg_version(),
    }


def _get_ffmpeg_version() -> str:
    result = run_subprocess(
        [FFMPEG, "-version"],
        5,
    )
    return result.stdout.splitlines()[0]
