import json
import os
import subprocess
from pathlib import Path
from typing import Any

from amv.config import FFPROBE
from amv.io.media import MediaInfo

TIMEOUT: int = 30  # in seconds


def probe(path: str | os.PathLike) -> MediaInfo:
    """Probe a media file and return its metadata as a MediaInfo instance.
    Args:
        path (str | os.PathLike): The path to the media file."""
    path = Path(path)
    data = _run_ffprobe(path)
    return MediaInfo.from_ffprobe(path, data)


def _run_ffprobe(path: str | os.PathLike) -> dict[str, Any]:
    """Probe a media file using ffprobe.

    Args:
        path (str | os.PathLike): The path to the media file.

    Returns:
        dict: A dictionary containing the metadata of the media file."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    command = [
        str(FFPROBE),
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-loglevel",
        "error",
        "-i",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=TIMEOUT)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed with error: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffprobe timed out on {path}") from e
