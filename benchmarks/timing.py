"""Measuring: time a function over several repeats and wrap the result in a BenchResult."""

import time
from collections.abc import Callable

from tqdm import tqdm

from amv.io.media import MediaInfo
from benchmarks.results import BenchResult


def time_it(
    fn: Callable[[], int],
    warmup: int = 1,
    repeat: int = 5,
    name: str | None = None,
) -> tuple[list[float], int]:
    """Call fn warmup + repeat times and measure each call's wall-clock time.

    Wall clock (time.perf_counter) rather than CPU time, so work done in child
    processes such as ffmpeg is included. The warmup calls run first and are discarded;
    they absorb one-off costs such as a cold disk cache.

    Args:
        fn (Callable[[], int]): The work to measure. Takes no arguments (bind them with
            functools.partial) and returns the number of frames it processed.
        warmup (int): Unmeasured calls before the measured ones. Defaults to 1.
        repeat (int): Measured calls. Defaults to 5.
        name (str | None): Label for the progress bar.

    Returns:
        tuple[list[float], int]: (seconds, n_frames) where
            seconds holds one wall-clock duration per measured call, in call order;
            n_frames is the frame count every measured call returned.

    Raises:
        ValueError: If repeat < 1 or warmup < 0; fn is not called.
        RuntimeError: If the measured calls returned different frame counts. The warmup
            counts are not checked."""
    if repeat < 1:
        raise ValueError(f"Repeat must be >= 1, got {repeat}.")
    if warmup < 0:
        raise ValueError(f"Warmup must be >= 0, got {warmup}.")

    timings: list[float] = []
    processed_frames: list[int] = []
    for _ in tqdm(range(warmup + repeat), desc=name, dynamic_ncols=True):
        start = time.perf_counter()
        frame_counter = fn()
        end = time.perf_counter()
        processed_frames.append(frame_counter)
        timings.append(end - start)

    if len(set(processed_frames[warmup:])) != 1:
        raise RuntimeError(f"frame count differs between repeats: {processed_frames}")
    return (timings[warmup:], processed_frames[-1])


def benchmark(
    fn: Callable[[], int],
    name: str,
    media_info: MediaInfo,
    warmup: int = 1,
    repeat: int = 5,
) -> BenchResult:
    """Measure fn with time_it and bundle the timings with the media they ran on.

    Args:
        fn (Callable[[], int]): The work to measure; see time_it.
        name (str): The benchmark's name, used in the table, the JSON and the plots.
        media_info (MediaInfo): The file fn processes. Its video duration is the
            reference for "x real time".
        warmup (int): Unmeasured calls before the measured ones. Defaults to 1.
        repeat (int): Measured calls. Defaults to 5.

    Returns:
        BenchResult: The measured timings, the frame count and the media duration.

    Raises:
        ValueError: If media_info has no video stream.
        RuntimeError: If the frame count differs between repeats (see time_it)."""
    if media_info.video is None:
        raise ValueError(f"Provided media was not a valid video: {media_info.path}")

    seconds, n_frames = time_it(fn, warmup, repeat, name=name)
    return BenchResult(
        name=name,
        input=media_info,
        seconds=seconds,
        n_frames=n_frames,
        media_seconds=media_info.video.duration,
    )
