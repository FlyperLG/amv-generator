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

from amv.config import (
    AUDIO_CHANNELS,
    AUDIO_EXTRACTION_TIMEOUT,
    AUDIO_SAMPLE_RATE,
    FFMPEG,
    FFPROBE,
    PROBE_TIMEOUT,
    PROXY_CRF,
    PROXY_PRESET,
    PROXY_TRANSCODE_TIMEOUT,
    PROXY_WIDTH,
)
from amv.io.media import MediaInfo

logger = logging.getLogger(__name__)


# --- Shared helpers -----------------------------------------------------------


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


def run_subprocess(command: list[str], timeout: float | None) -> CompletedProcess[str]:
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
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"command failed with error: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("command timed out") from e


def _generate_uuid(name: str) -> str:
    """Derive a deterministic UUID (version 5) from a string.

    The same input always gives the same UUID, so it can be used as a stable cache key.

    Args:
        name (str): The string to hash, e.g. a resolved path or a joined list of ffmpeg arguments.

    Returns:
        str: The UUID in its canonical 36-character string form.

    Raises:
        ValueError: If name is empty or only whitespace."""
    if not name.strip():
        raise ValueError("Name for uuid generation cannot be empty.")

    namespace = uuid.NAMESPACE_URL
    file_uuid = uuid.uuid5(namespace, name)
    return str(file_uuid)


# --- Probing ------------------------------------------------------------------


def probe(path: str | os.PathLike) -> MediaInfo:
    """Probe a media file and return its metadata as a MediaInfo instance.

    Args:
        path (str | os.PathLike): The path to the media file.

    Returns:
        MediaInfo: The container, video and audio metadata of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffprobe fails or times out, e.g. because the file is not a media file."""
    path = Path(path)
    data = _run_ffprobe(path)
    return MediaInfo.from_ffprobe(path, data)


def _run_ffprobe(path: Path) -> dict[str, Any]:
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
        result = run_subprocess(command, PROBE_TIMEOUT)
        return json.loads(result.stdout)
    except RuntimeError as e:
        raise RuntimeError(f"ffprobe on path {path} failed with error: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe on path {path} returned invalid JSON") from e


# --- Cached proxies (video and audio) -----------------------------------------


def generate_file_id(path: str | os.PathLike | Path) -> str:
    """Return the video_id of a source file: a UUID derived from its resolved path.

    The same file always gets the same id, and two files with the same name in different
    folders get different ones. It is the "<video_uuid>" prefix of every proxy name and
    the video_id column of videos.parquet. Moving the file (or the data/ folder) changes it.

    Args:
        path (str | os.PathLike | Path): The source media file.

    Returns:
        str: The UUID in its canonical 36-character string form.

    Raises:
        FileNotFoundError: If path does not exist."""
    path = _file_exists(path).resolve()
    return _generate_uuid(str(path))


def _output_path(in_path: Path, out_dir: Path, suffix: str, ffmpeg_args: list[str]) -> Path:
    """Compute the cached output path for a conversion, without creating anything.

    The name is "<video_uuid>_<settings_uuid><suffix>": the first UUID comes from the
    resolved input path, the second from ffmpeg_args, so changing any setting gives a
    new name instead of reusing a stale file. All outputs of one source share the prefix.

    Args:
        in_path (Path): The source media file.
        out_dir (Path): The output directory. It does not have to exist yet.
        suffix (str): The output extension including the dot, e.g. ".mp4".
        ffmpeg_args (list[str]): The arguments that describe the conversion.

    Returns:
        Path: out_dir / "<video_uuid>_<settings_uuid><suffix>".

    Raises:
        FileNotFoundError: If in_path does not exist.
        NotADirectoryError: If out_dir exists but is a file."""
    if out_dir.is_file():
        raise NotADirectoryError(f"out_dir needs to be an directory was {out_dir}")

    file_uuid = generate_file_id(in_path)
    config_uuid = _generate_uuid(" ".join(ffmpeg_args))
    return out_dir / Path(f"{file_uuid}_{config_uuid}{suffix}")


def _ffmpeg_to_file(
    in_path: Path,
    out_dir: Path,
    suffix: str,
    ffmpeg_args: list[str],
    timeout: float | None,
    task: str,
) -> Path:
    """Run one ffmpeg conversion of in_path into a cached output file in out_dir.

    The output path comes from _output_path. If that file already exists and is newer
    than the input, it is returned without running ffmpeg. ffmpeg writes to a temporary
    "<name>.tmp<suffix>" file that is only renamed on success, so a failed or interrupted
    run never leaves a broken output behind. Next to the output, a "<name>.json" sidecar
    records the resolved source path, the task and the ffmpeg_args, so a human can tell
    what a hash stands for.

    Args:
        in_path (Path): The source media file.
        out_dir (Path): The output directory; created if missing.
        suffix (str): The output extension including the dot, e.g. ".mp4". ffmpeg picks
            the output format from it.
        ffmpeg_args (list[str]): The arguments between the input and the output file. They
            are part of the cache key, so they must fully describe the output.
        timeout (float | None): Seconds before ffmpeg is killed, or None to wait forever.
        task (str): A short name for the conversion, used in log and error messages.

    Returns:
        Path: The path of the output file.

    Raises:
        FileNotFoundError: If in_path does not exist.
        NotADirectoryError: If out_dir exists but is a file.
        RuntimeError: If ffmpeg fails or exceeds the timeout. The message contains the task
            and ffmpeg's stderr."""
    in_path = in_path.resolve()
    final_out_path = _output_path(in_path, out_dir, suffix, ffmpeg_args)
    tmp_out_path = final_out_path.with_suffix(f".tmp{suffix}")
    out_dir.mkdir(exist_ok=True, parents=True)

    if final_out_path.is_file() and in_path.stat().st_mtime_ns < final_out_path.stat().st_mtime_ns:
        logger.info("%s: Skipping existing file: %s", task, in_path)
        return final_out_path

    command = [
        FFMPEG,
        "-i",
        str(in_path),
        *ffmpeg_args,
        "-y",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        str(tmp_out_path),
    ]
    try:
        run_subprocess(command, timeout)
        sidecar_data = {
            "source": str(in_path),
            "task": task,
            "ffmpeg_args": ffmpeg_args,
        }
        with open(final_out_path.with_suffix(".json"), "w", encoding="utf-8") as sidecar_json:
            json.dump(sidecar_data, sidecar_json, indent=4)

        final_out_path = tmp_out_path.replace(final_out_path)
        return final_out_path
    except RuntimeError as e:
        raise RuntimeError(f"ffmpeg {task} on path {in_path} failed with error: {e}") from e
    finally:
        tmp_out_path.unlink(missing_ok=True)


def _video_args() -> list[str]:
    """The ffmpeg arguments of a video proxy, built from the PROXY_* settings at call time."""
    return [
        "-vf",
        f"scale={PROXY_WIDTH}:-2",
        "-fps_mode",
        "cfr",
        "-preset",
        PROXY_PRESET,
        "-threads",
        "0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(PROXY_CRF),
    ]


def video_proxy_path(in_path: str | os.PathLike | Path, out_dir: str | os.PathLike | Path) -> Path:
    """Return the path make_video_proxy uses for in_path with the current settings.

    Runs no ffmpeg and creates nothing, so it is a cheap way to ask whether a proxy
    already exists, or which files in out_dir are still current.

    Args:
        in_path (str | os.PathLike | Path): The source video.
        out_dir (str | os.PathLike | Path): The directory for proxies. It does not have to exist.

    Returns:
        Path: out_dir / "<video_uuid>_<settings_uuid>.mp4". The file may not exist yet.

    Raises:
        FileNotFoundError: If in_path does not exist.
        NotADirectoryError: If out_dir exists but is a file."""
    return _output_path(Path(in_path), Path(out_dir), ".mp4", _video_args())


def make_video_proxy(in_path: str | os.PathLike | Path, out_dir: str | os.PathLike | Path) -> Path:
    """Transcode a video into a small, fast-to-decode proxy for analysis.

    The proxy is PROXY_WIDTH px wide (height keeps the aspect ratio, rounded to an even
    number), constant frame rate, H.264 with PROXY_PRESET and PROXY_CRF, and has no audio.
    It is cached in out_dir, see _ffmpeg_to_file for the naming and skip rules.

    Args:
        in_path (str | os.PathLike | Path): The source video.
        out_dir (str | os.PathLike | Path): The directory for proxies; created if missing.

    Returns:
        Path: The path of the proxy, out_dir / "<video_uuid>_<settings_uuid>.mp4".

    Raises:
        FileNotFoundError: If in_path does not exist.
        NotADirectoryError: If out_dir exists but is a file.
        RuntimeError: If ffmpeg fails or exceeds PROXY_TRANSCODE_TIMEOUT."""
    return _ffmpeg_to_file(
        in_path=Path(in_path),
        out_dir=Path(out_dir),
        suffix=".mp4",
        ffmpeg_args=_video_args(),
        timeout=PROXY_TRANSCODE_TIMEOUT,
        task="video proxy",
    )


def _audio_args() -> list[str]:
    """The ffmpeg arguments of an audio proxy, built from the AUDIO_* settings at call time."""
    return ["-vn", "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_SAMPLE_RATE)]


def audio_proxy_path(in_path: str | os.PathLike | Path, out_dir: str | os.PathLike | Path) -> Path:
    """Return the path make_audio_proxy uses for in_path with the current settings.

    Runs no ffmpeg and creates nothing, so it is a cheap way to ask whether an audio
    proxy already exists, or which files in out_dir are still current.

    Args:
        in_path (str | os.PathLike | Path): The source video or audio file.
        out_dir (str | os.PathLike | Path): The directory for the wav. It does not have to exist.

    Returns:
        Path: out_dir / "<video_uuid>_<settings_uuid>.wav". The file may not exist yet.

    Raises:
        FileNotFoundError: If in_path does not exist.
        NotADirectoryError: If out_dir exists but is a file."""
    return _output_path(Path(in_path), Path(out_dir), ".wav", _audio_args())


def make_audio_proxy(in_path: str | os.PathLike | Path, out_dir: str | os.PathLike | Path) -> Path:
    """Extract the audio track of a media file as an uncompressed wav for music analysis.

    The wav is 16-bit PCM, resampled to AUDIO_SAMPLE_RATE Hz and mixed down to
    AUDIO_CHANNELS channels. The input can be a video (e.g. an AMV) or an audio file
    (e.g. a song). It is cached in out_dir, see _ffmpeg_to_file for the naming and skip rules.

    Args:
        in_path (str | os.PathLike | Path): The source video or audio file.
        out_dir (str | os.PathLike | Path): The directory for the wav; created if missing.

    Returns:
        Path: The path of the wav, out_dir / "<video_uuid>_<settings_uuid>.wav".

    Raises:
        FileNotFoundError: If in_path does not exist.
        NotADirectoryError: If out_dir exists but is a file.
        RuntimeError: If ffmpeg fails (e.g. the file has no audio stream) or exceeds
            AUDIO_EXTRACTION_TIMEOUT."""
    return _ffmpeg_to_file(
        in_path=Path(in_path),
        out_dir=Path(out_dir),
        suffix=".wav",
        ffmpeg_args=_audio_args(),
        timeout=AUDIO_EXTRACTION_TIMEOUT,
        task="audio proxy",
    )


# --- Frame reading ------------------------------------------------------------


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
                    logger.warning("Incomplete frame read from file %s", path)
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
