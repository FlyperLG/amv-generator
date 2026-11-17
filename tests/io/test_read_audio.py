"""Tests for amv.io.ffmpeg.read_audio (docs/01-decoding-and-proxies, task 3.8).

The end-card refinement reads a few seconds of an AMV's audio straight from the original,
the audio counterpart of iter_frames. Whole songs keep going through the cached wav.

SPECIFICATION
  read_audio(path, sample_rate, start=0, end=None) -> np.ndarray
  - Decodes [start, end) of the file's audio (start/end are input options before -i, as in
    iter_frames; end=None reads to the end), resampled to sample_rate and mixed to mono.
  - Returns float32, 1-D, values in [-1, 1], about (end - start) * sample_rate samples.
    A start past the end of the audio gives an empty array.
  - Works for videos with audio (.mp4) and audio files (.mp3).
  - Missing file -> FileNotFoundError, before ffmpeg runs.
  - sample_rate <= 0, start < 0 or end <= start -> ValueError.
  - ffmpeg fails (no audio stream, not a media file) -> RuntimeError mentioning "ffmpeg".
    (Decision: like make_audio_proxy, a file without audio is an error, not silence. The
    caller knows from probe() whether there is audio.)
  - Binary output: read stdout as bytes (run_subprocess decodes text, so it can't be used
    as is) and np.frombuffer(..., np.float32) with "-f f32le".
"""

import numpy as np
import pytest

from amv.io.ffmpeg import read_audio

pytestmark = pytest.mark.integration  # every test here runs the real ffmpeg

RATE = 8000


def dominant_frequency(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    return float(np.fft.rfftfreq(len(samples), 1 / rate)[spectrum.argmax()])


def test_returns_mono_float32_samples(make_media):
    src = make_media("amv.mp4", video=True, audio=True, duration=2.0)

    samples = read_audio(src, RATE)

    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert len(samples) == pytest.approx(2.0 * RATE, rel=0.05)


def test_keeps_the_audio_content(make_media):
    src = make_media("amv.mp4", video=True, audio=True, duration=2.0)

    samples = read_audio(src, RATE)

    assert np.abs(samples).max() <= 1.0
    assert np.abs(samples).max() > 0.1  # not silence
    assert dominant_frequency(samples, RATE) == pytest.approx(440, abs=5)


def test_stereo_is_mixed_to_mono(make_media):
    src = make_media("song.mp3", video=False, audio=True, duration=1.0, channels=2)

    assert read_audio(src, RATE).ndim == 1


def test_start_and_end_select_a_range(make_media):
    src = make_media("amv.mp4", video=True, audio=True, duration=3.0)

    samples = read_audio(src, RATE, start=0.5, end=1.5)

    assert len(samples) == pytest.approx(1.0 * RATE, rel=0.02)


def test_end_none_reads_to_the_end(make_media):
    src = make_media("amv.mp4", video=True, audio=True, duration=3.0)

    samples = read_audio(src, RATE, start=2.0)

    assert len(samples) == pytest.approx(1.0 * RATE, rel=0.05)


def test_a_start_past_the_end_gives_an_empty_array(make_media):
    src = make_media("amv.mp4", video=True, audio=True, duration=1.0)

    samples = read_audio(src, RATE, start=5.0)

    assert samples.shape == (0,)
    assert samples.dtype == np.float32


def test_a_file_without_audio_raises(make_media):
    src = make_media("silent.mp4", video=True, audio=False, duration=1.0)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        read_audio(src, RATE)


def test_not_a_media_file_raises(tmp_path):
    fake = tmp_path / "fake.mp4"
    fake.write_text("not media")

    with pytest.raises(RuntimeError, match="ffmpeg"):
        read_audio(fake, RATE)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_audio(tmp_path / "missing.mp4", RATE)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"sample_rate": RATE, "start": -1.0},
        {"sample_rate": RATE, "start": 1.0, "end": 1.0},
    ],
)
def test_bad_arguments_raise(make_media, kwargs):
    src = make_media("amv.mp4", video=True, audio=True, duration=1.0)

    with pytest.raises(ValueError):
        read_audio(src, **kwargs)
