import json
import logging
import os
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import numpy as np

from amv.config import FFMPEG, FFPROBE, PROBE_TIMEOUT, PROXY_TRANSCODE_TIMEOUT
from amv.io.media import MediaInfo


def _file_exists(path: str | os.PathLike | Path) -> Path:
    """Check that a path points to an existing file.

    Args:
        path (str | os.PathLike | Path): The path to check.

    Returns:
        Path: The path as a Path object.

    Raises:
        FileNotFoundError: If the path does not exist or is not a regular file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _run_subprocess(command: list[str], timeout: float | None) -> CompletedProcess[str]:
    """Run a command to completion and capture its output as text.

    Args:
        command (list[str]): The executable followed by its arguments.
        timeout (float | None): Seconds before the command is killed, or None to wait forever.

    Returns:
        CompletedProcess[str]: The finished process, with stdout and stderr as strings.

    Raises:
        RuntimeError: If the command exits with a non-zero code (the message contains its
            stderr) or exceeds the timeout."""
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"command failed with error: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("command timed out") from e


def probe(path: str | os.PathLike) -> MediaInfo:
    """Probe a media file and return its metadata as a MediaInfo instance.

    Args:
        path (str | os.PathLike): The path to the media file.

    Returns:
        MediaInfo: The container, video and audio metadata of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe fails or times out, e.g. because the file is not a media file."""
    data = _run_ffprobe(path)
    return MediaInfo.from_ffprobe(Path(path), data)


def _run_ffprobe(path: str | os.PathLike) -> dict[str, Any]:
    """Probe a media file using ffprobe.

    Args:
        path (str | os.PathLike): The path to the media file.

    Returns:
        dict: The parsed ffprobe JSON, with the keys "streams" and "format".

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe fails or times out."""
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
    """Derive a deterministic UUID (version 5) from a file name.

    The same input always gives the same UUID, so it can be used as a stable cache key.

    Args:
        filename (str): The name to hash, typically an absolute, resolved path.

    Returns:
        str: The UUID in its canonical 36-character string form.

    Raises:
        ValueError: If filename is empty or only whitespace."""
    if not filename.strip():
        raise ValueError("Filename for uuid generation cannot be empty.")

    namespace = uuid.NAMESPACE_URL
    file_uuid = uuid.uuid5(namespace, filename)
    return str(file_uuid)


def run_proxy_transcode(in_path: str | os.PathLike | Path, out_dir: str | os.PathLike | Path) -> Path:
    """Transcode a video into a small, fast-to-decode proxy for analysis.

    The proxy is 640 px wide (height keeps the aspect ratio, rounded to an even number),
    constant frame rate, H.264 (ultrafast, CRF 28) and has no audio. It is named after a
    UUID of the resolved input path. If a proxy newer than the input already exists, it
    is reused and nothing is transcoded. ffmpeg writes to a temporary file that is only
    renamed on success, so a failed or interrupted run never leaves a broken proxy behind.

    Args:
        in_path (str | os.PathLike | Path): The source video.
        out_dir (str | os.PathLike | Path): The directory for proxies; created if missing.

    Returns:
        Path: The path of the proxy, out_dir / "<uuid>.mp4".

    Raises:
        FileNotFoundError: If in_path does not exist.
        NotADirectoryError: If out_dir exists but is a file.
        RuntimeError: If ffmpeg fails or exceeds PROXY_TRANSCODE_TIMEOUT."""
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


def iter_frames(
    path: str | os.PathLike | Path,
    fps: float,
    width: int,
    height: int,
    start: float = 0,
    end: float | None = None,
    gray: bool = False,
) -> Iterator[tuple[float, np.ndarray]]:
    """Decode a video into frames, resampled to a fixed frame rate and size.

    This is a generator: ffmpeg starts and the arguments are validated on the first
    next(), so errors are raised there and not when the function is called. ffmpeg is
    stopped as soon as the consumer stops iterating (break or close()).

    Args:
        path (str | os.PathLike | Path): The video to decode.
        fps (float): The output frame rate; frames are dropped or duplicated to match it.
        width (int): The output frame width in pixels.
        height (int): The output frame height in pixels. The aspect ratio is not kept.
        start (float): Time in the source video, in seconds, of the first frame. Defaults to 0.
        end (float | None): Time in seconds at which to stop; that frame itself is not
            included, so the range is [start, end). None reads to the end of the video.
        gray (bool): If True, yield single-channel luma frames instead of RGB.

    Yields:
        tuple[float, np.ndarray]: (timestamp, frame) where
            timestamp is the time of the frame in the source video in seconds,
                start + i / fps for the i-th frame;
            frame is a writable, C-contiguous uint8 array that no other frame shares, of shape
                (height, width, 3) in RGB order, or (height, width) if gray is True.
        A start past the end of the video yields nothing.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If fps, width or height is <= 0, start < 0, or end <= start.
        RuntimeError: If ffmpeg fails, e.g. the file is not a video or has no video stream.
            The message contains ffmpeg's stderr."""
    path = _file_exists(path)

    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid input for fps, width, or height: ({fps}, {width}, {height}).")

    if start < 0 or end is not None and start >= end:
        raise ValueError(f"Invalid start, and end configuration received: ({start}, {end}).")

    seek_args: list[str] = ["-ss", str(start)]
    if end is not None:
        seek_args += ["-to", str(end)]

    pix_fmt: str = "rgb24" if not gray else "gray"

    command = [
        str(FFMPEG),
        *seek_args,
        "-i",
        str(path),
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "pipe:1",
    ]

    channels = 1 if gray else 3
    frame_size = width * height * channels
    frame_shape = (height, width, channels) if not gray else (height, width)
    with tempfile.TemporaryFile() as err:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=err,
            bufsize=frame_size,
            text=False,
        )
        assert process.stdout is not None
        try:
            i: int = 0
            while True:
                frame = process.stdout.read(
                    frame_size,
                )
                if not frame:
                    break
                if len(frame) != frame_size:
                    logging.warning("Incomplete frame read from file %s", path)
                    break
                frame_array = np.frombuffer(frame, dtype=np.uint8).reshape(frame_shape).copy()
                timestamp = start + i / fps
                yield (timestamp, frame_array)
                i += 1

            if process.wait() != 0:
                err.seek(0)
                message = err.read().decode(errors="replace").strip()
                raise RuntimeError(f"ffmpeg iter frames on path {path} failed: {message}")
        finally:
            if process.poll() is None:
                process.kill()
            process.stdout.close()
            process.wait()


def iter_frame_batches(
    path: str | os.PathLike | Path,
    fps: float,
    width: int,
    height: int,
    batch_size: int,
    start: float = 0,
    end: float | None = None,
    gray: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Decode a video like iter_frames, but yield the frames in batches.

    Gives the same frames and timestamps as iter_frames, stacked into arrays so they can
    be fed to a model in one call.

    Args:
        path (str | os.PathLike | Path): The video to decode.
        fps (float): The output frame rate.
        width (int): The output frame width in pixels.
        height (int): The output frame height in pixels.
        batch_size (int): The number of frames per batch.
        start (float): Time in seconds of the first frame. Defaults to 0.
        end (float | None): Time in seconds at which to stop, exclusive. None reads to the end.
        gray (bool): If True, yield single-channel luma frames instead of RGB.

    Yields:
        tuple[np.ndarray, np.ndarray]: (timestamps, frames) where
            timestamps is a float64 array of shape (B,), in seconds in the source video;
            frames is a uint8 array of shape (B, height, width, 3) in RGB order,
                or (B, height, width) if gray is True.
        B equals batch_size for every batch except the last, which may be smaller but is
        never empty.

    Raises:
        ValueError: If batch_size <= 0, or for the invalid arguments listed in iter_frames.
        FileNotFoundError, RuntimeError: As in iter_frames."""
    if batch_size <= 0:
        raise ValueError(f"Batch size needs to be greater than 0 was {batch_size}")

    frames_gen = iter_frames(
        path=path,
        fps=fps,
        width=width,
        height=height,
        start=start,
        end=end,
        gray=gray,
    )

    batched_frames: list[np.ndarray] = []
    timestamps: list[float] = []
    for timestamp, frame in frames_gen:
        batched_frames.append(frame)
        timestamps.append(timestamp)

        if len(batched_frames) == batch_size:
            yield (np.array(timestamps), np.array(batched_frames))
            batched_frames = []
            timestamps = []
    if batched_frames:
        yield (np.array(timestamps), np.array(batched_frames))
