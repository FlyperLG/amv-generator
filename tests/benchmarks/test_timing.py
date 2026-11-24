"""Unit tests for benchmarks.timing: time_it, benchmark.

No ffmpeg, no media files: the measured functions are fakes that count their calls.

SPECIFICATION (decisions these tests encode)
--------------------------------------------
time_it(fn, warmup=1, repeat=5, name=None) -> (seconds, n_frames)
  - fn is called warmup + repeat times; the warmup timings are dropped, so len(seconds) == repeat.
  - fn returns the number of frames it processed. Every measured repeat must return the same
    count, otherwise RuntimeError (a decode that varies between runs is a bug). Warmup runs
    are not checked.
  - warmup=0 is allowed. repeat < 1 or warmup < 0 -> ValueError (time_it is shared code
    now, so it cannot rely on a script's argparse check).
  - Timings are wall-clock seconds (perf_counter), so a child process like ffmpeg is counted.

benchmark(fn, name, media_info, stream, warmup=1, repeat=5) -> BenchResult
  - name, input and the timings come straight through.
  - stream ("video" | "audio", required) names the stream the timed work is about; its
    duration is media_seconds. The caller says it, because only the caller knows: an AMV
    mp4 has both streams, and their durations differ by tens to hundreds of ms. Not
    container_duration: step 01's end-card cut can make it differ.
  - The requested stream missing -> ValueError before fn is called, even if the other
    stream exists (no silent fallback).
"""

import dataclasses
import time
from pathlib import Path

import pytest

from amv.io.media import AudioInfo, MediaInfo
from benchmarks.timing import benchmark, time_it
from tests.benchmarks.helpers import make_media_info


class CountingFn:
    """Fake measured function: returns frame_counts[i] on call i (the last one repeats)."""

    def __init__(self, *frame_counts: int, sleep: float = 0.0) -> None:
        self.frame_counts = frame_counts
        self.sleep = sleep
        self.calls = 0

    def __call__(self) -> int:
        count = self.frame_counts[min(self.calls, len(self.frame_counts) - 1)]
        self.calls += 1
        if self.sleep:
            time.sleep(self.sleep)
        return count


# =============================================================================
# time_it
# =============================================================================


class TestTimeIt:
    def test_calls_fn_warmup_plus_repeat_times(self):
        fn = CountingFn(100)

        time_it(fn, warmup=2, repeat=3)

        assert fn.calls == 5

    def test_drops_warmup_timings(self):
        seconds, _ = time_it(CountingFn(100), warmup=2, repeat=3)

        assert len(seconds) == 3

    def test_warmup_zero_is_allowed(self):
        fn = CountingFn(100)

        seconds, _ = time_it(fn, warmup=0, repeat=4)

        assert fn.calls == 4
        assert len(seconds) == 4

    @pytest.mark.parametrize(("warmup", "repeat"), [(1, 0), (0, 0), (-1, 3)])
    def test_invalid_counts_raise_before_calling_fn(self, warmup, repeat):
        fn = CountingFn(100)

        with pytest.raises(ValueError):
            time_it(fn, warmup=warmup, repeat=repeat)
        assert fn.calls == 0

    def test_returns_frame_count(self):
        _, n_frames = time_it(CountingFn(1087), warmup=1, repeat=3)

        assert n_frames == 1087

    def test_differing_frame_counts_raise(self):
        # warmup, then 100, 100, 99: one repeat decoded a frame less.
        fn = CountingFn(100, 100, 100, 99)

        with pytest.raises(RuntimeError, match="frame count"):
            time_it(fn, warmup=1, repeat=3)

    def test_warmup_frame_count_is_not_checked(self):
        # The warmup may differ (e.g. a cold cache); only the measured repeats must agree.
        _, n_frames = time_it(CountingFn(1, 100), warmup=1, repeat=3)

        assert n_frames == 100

    def test_timings_measure_the_call(self):
        seconds, _ = time_it(CountingFn(1, sleep=0.02), warmup=0, repeat=2)

        assert all(0.015 < s < 1.0 for s in seconds)


# =============================================================================
# benchmark
# =============================================================================


class TestBenchmark:
    def test_builds_result_from_timings_and_media(self):
        media = make_media_info(duration=135.84)

        result = benchmark(CountingFn(1087), name="read", media_info=media, stream="video", warmup=1, repeat=3)

        assert result.name == "read"
        assert result.input is media
        assert len(result.seconds) == 3
        assert result.n_frames == 1087
        assert result.media_seconds == pytest.approx(135.84)

    def test_audio_stream_gives_audio_duration(self):
        song = MediaInfo(
            path=Path("song.wav"),
            container_duration=134.0,
            video=None,
            audio=AudioInfo(duration=133.5, sample_rate=22050, channels=1, codec="pcm_s16le"),
        )

        result = benchmark(CountingFn(240), name="beats", media_info=song, stream="audio", warmup=0, repeat=1)

        assert result.media_seconds == pytest.approx(133.5)
        assert result.input is song

    @pytest.mark.parametrize(("stream", "expected"), [("video", 30.0), ("audio", 29.9)])
    def test_stream_picks_the_duration_when_both_exist(self, stream, expected):
        amv = dataclasses.replace(
            make_media_info(duration=30.0),
            audio=AudioInfo(duration=29.9, sample_rate=48000, channels=2, codec="aac"),
        )

        result = benchmark(CountingFn(1), name="read", media_info=amv, stream=stream, warmup=0, repeat=1)

        assert result.media_seconds == pytest.approx(expected)

    @pytest.mark.parametrize("stream", ["video", "audio"])
    def test_missing_stream_raises_before_calling_fn(self, stream):
        video_only = make_media_info()
        audio_only = MediaInfo(
            path=Path("song.wav"),
            container_duration=10.0,
            video=None,
            audio=AudioInfo(duration=10.0, sample_rate=22050, channels=1, codec="pcm_s16le"),
        )
        media = audio_only if stream == "video" else video_only
        fn = CountingFn(1)

        with pytest.raises(ValueError, match=stream):
            benchmark(fn, name="read", media_info=media, stream=stream, warmup=0, repeat=1)
        assert fn.calls == 0

    def test_stream_is_required(self):
        with pytest.raises(TypeError):
            benchmark(CountingFn(1), name="read", media_info=make_media_info())  # type: ignore[call-arg]
