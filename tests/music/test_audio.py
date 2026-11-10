"""Tests for amv.music.audio.load_audio (docs/02-music-features, task 3.1).

The wavs are written with soundfile in tmp_path, so most tests need no ffmpeg. Only
the last test runs the real make_audio_proxy to check that both ends fit together.

SPECIFICATION
  load_audio(path, sample_rate) -> tuple[np.ndarray, int]
  - path accepts str or PathLike. Reads the file with soundfile.
  - sample_rate is the rate the file must have, a required argument (Decision: audio.py
    does not read amv.config; the entry point passes MUSIC.sample_rate in).
  - Returns (y, sr): y is a 1-D float32 array with one value per sample in the file,
    sr is sample_rate.
  - Peak-normalised: y = x / max|x|, so max|y| == 1.0 and the waveform's shape is
    unchanged (a negative peak becomes -1.0). Normalising happens in memory; the file
    is never modified.

  Only audio proxies are accepted (Decision: resampling and down-mixing are step 01's
  job; doing it here a second time would hide a missing proxy and cost time)
  - file's rate != sample_rate         -> ValueError mentioning the sample rate.
  - more than one channel              -> ValueError mentioning the channels.

  Failures
  - Missing file -> FileNotFoundError.
  - No samples, or only silence (all zeros) -> ValueError. (Decision: nothing in
    step 02 means anything for a silent song, and dividing by a peak of 0 gives NaN.)
"""

import numpy as np
import pytest
import soundfile as sf

from amv.config import DECODING, MUSIC
from amv.music.audio import load_audio

SR = MUSIC.sample_rate


def write_wav(path, samples, sr=SR, subtype="PCM_16"):
    sf.write(str(path), np.asarray(samples), sr, subtype=subtype)
    return path


def sine(freq=440.0, seconds=1.0, amplitude=0.25, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


# --- Output ---------------------------------------------------------------------


def test_returns_1d_float32_and_sample_rate(tmp_path):
    wav = write_wav(tmp_path / "song.wav", sine())

    y, sr = load_audio(wav, SR)

    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert y.dtype == np.float32
    assert sr == SR


def test_keeps_every_sample(tmp_path):
    wav = write_wav(tmp_path / "song.wav", sine(seconds=1.5))

    y, _ = load_audio(wav, SR)

    assert len(y) == int(1.5 * SR)


def test_accepts_str_path(tmp_path):
    wav = write_wav(tmp_path / "song.wav", sine())

    y, _ = load_audio(str(wav), SR)

    assert len(y) == SR


# --- Peak normalisation ---------------------------------------------------------


def test_quiet_audio_is_scaled_to_peak_one(tmp_path):
    wav = write_wav(tmp_path / "quiet.wav", sine(amplitude=0.05))

    y, _ = load_audio(wav, SR)

    assert np.abs(y).max() == pytest.approx(1.0, abs=1e-6)


def test_waveform_shape_is_unchanged(tmp_path):
    x = sine(amplitude=0.1) + sine(freq=1000, amplitude=0.05)
    wav = write_wav(tmp_path / "song.wav", x, subtype="FLOAT")

    y, _ = load_audio(wav, SR)

    np.testing.assert_allclose(y, x / np.abs(x).max(), atol=1e-6)


def test_negative_peak_becomes_minus_one(tmp_path):
    x = np.full(SR, 0.1)
    x[100] = -0.5  # the largest magnitude is negative
    wav = write_wav(tmp_path / "song.wav", x, subtype="FLOAT")

    y, _ = load_audio(wav, SR)

    assert y.min() == pytest.approx(-1.0)
    assert y.max() == pytest.approx(0.2)


def test_file_is_not_modified(tmp_path):
    wav = write_wav(tmp_path / "song.wav", sine(amplitude=0.05))
    before = (wav.read_bytes(), wav.stat().st_mtime_ns)

    load_audio(wav, SR)

    assert (wav.read_bytes(), wav.stat().st_mtime_ns) == before


# --- Only audio proxies are accepted --------------------------------------------


def test_sample_rate_has_no_default(tmp_path):
    wav = write_wav(tmp_path / "song.wav", sine())

    with pytest.raises(TypeError):
        load_audio(wav)  # type: ignore[call-arg]


def test_expected_sample_rate_is_a_parameter(tmp_path):
    wav = write_wav(tmp_path / "song_44k.wav", sine(sr=44100), sr=44100)

    y, sr = load_audio(wav, 44100)

    assert sr == 44100
    assert len(y) == 44100


def test_other_sample_rate_raises(tmp_path):
    wav = write_wav(tmp_path / "song_44k.wav", sine(sr=44100), sr=44100)

    with pytest.raises(ValueError, match="(?i)sample.?rate"):
        load_audio(wav, SR)


def test_stereo_raises(tmp_path):
    x = sine()
    wav = write_wav(tmp_path / "stereo.wav", np.stack([x, x], axis=1))

    with pytest.raises(ValueError, match="(?i)channel"):
        load_audio(wav, SR)


# --- Failures -------------------------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "does_not_exist.wav", SR)


def test_silence_raises(tmp_path):
    wav = write_wav(tmp_path / "silence.wav", np.zeros(SR))

    with pytest.raises(ValueError):
        load_audio(wav, SR)


def test_empty_file_raises(tmp_path):
    wav = write_wav(tmp_path / "empty.wav", np.zeros(0))

    with pytest.raises(ValueError):
        load_audio(wav, SR)


# --- Together with step 01 ------------------------------------------------------


@pytest.mark.integration
def test_loads_an_audio_proxy(make_media, tmp_path):
    from amv.io.ffmpeg import make_audio_proxy

    src = make_media("song.mp3", video=False, audio=True, channels=2)  # 44.1 kHz stereo, 440 Hz

    y, sr = load_audio(make_audio_proxy(src, tmp_path / "proxies", DECODING.audio_proxy), SR)

    assert sr == SR
    assert np.abs(y).max() == pytest.approx(1.0, abs=1e-6)
    spectrum = np.abs(np.fft.rfft(y))
    assert np.fft.rfftfreq(len(y), 1 / sr)[spectrum.argmax()] == pytest.approx(440, abs=5)
