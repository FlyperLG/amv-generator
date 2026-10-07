import json
import logging
import os
import subprocess
import uuid
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

from amv.config import FFMPEG, FFPROBE, PROBE_TIMEOUT, PROXY_TRANSCODE_TIMEOUT
from amv.io.media import MediaInfo


def _file_exists(path: str | os.PathLike | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _run_subprocess(command: list[str], timeout: float | None) -> CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"command failed with error: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("command timed out") from e


def probe(path: str | os.PathLike) -> MediaInfo:
    """Probe a media file and return its metadata as a MediaInfo instance.
    Args:
        path (str | os.PathLike): The path to the media file."""
    data = _run_ffprobe(path)
    return MediaInfo.from_ffprobe(Path(path), data)


def _run_ffprobe(path: str | os.PathLike) -> dict[str, Any]:
    """Probe a media file using ffprobe.

    Args:
        path (str | os.PathLike): The path to the media file.

    Returns:
        dict: A dictionary containing the metadata of the media file."""
    path = _file_exists(path)

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
        result = _run_subprocess(command, PROBE_TIMEOUT)
        return json.loads(result.stdout)
    except RuntimeError as e:
        raise RuntimeError(f"ffprobe on path {path} failed with error: {e}") from e


def _generate_file_uuid(filename: str) -> str:
    if not filename.strip():
        raise ValueError("Filename for uuid generation cannot be empty.")

    namespace = uuid.NAMESPACE_URL
    file_uuid = uuid.uuid5(namespace, filename)
    return str(file_uuid)


def run_proxy_transcode(in_path: str | os.PathLike | Path, out_dir: str | os.PathLike | Path) -> Path:
    in_path = _file_exists(in_path).resolve()
    out_dir = Path(out_dir)
    if out_dir.is_file():
        raise NotADirectoryError(f"out_dir needs to be an directory was {out_dir}")
    out_dir.mkdir(exist_ok=True, parents=True)

    file_uuid = _generate_file_uuid(str(in_path))
    base_out_path = out_dir / Path(file_uuid)
    tmp_out_path = base_out_path.with_suffix(".tmp.mp4")
    final_out_path = base_out_path.with_suffix(".mp4")

    if final_out_path.is_file() and in_path.stat().st_mtime_ns < final_out_path.stat().st_mtime_ns:
        logging.info("Skipped %s, cause it already exists.", in_path)
        return final_out_path

    command = [
        str(FFMPEG),
        "-i",
        str(in_path),
        "-vf",
        "scale=640:-2",
        "-fps_mode",
        "cfr",
        "-preset",
        "ultrafast",
        "-y",
        "-threads",
        "0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        str(tmp_out_path),
    ]
    try:
        _run_subprocess(command, PROXY_TRANSCODE_TIMEOUT)
        return tmp_out_path.replace(final_out_path)
    except RuntimeError as e:
        raise RuntimeError(f"ffmpeg proxy transcode on path {in_path} failed with error: {e}") from e
    finally:
        tmp_out_path.unlink(missing_ok=True)
