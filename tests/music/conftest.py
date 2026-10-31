"""Fixtures for amv.music tests: synthetic drum loops, the beat model, the Blue Lock song."""

import shutil

import numpy as np
import pytest

from tests.helpers import DATA_DIR

SR = 22050


@pytest.fixture
def drum_loop():
    """Factory: a synthetic drum loop and the times of its beats.

    Usage:  y, beats = drum_loop(bpm=120, bars=8)

    Kick on every beat (louder on the first of each bar), snare on beats 2 and 4,
    hi-hat on every off-beat. The first beat is at 0.5 s and the loop ends 1 s after
    the last beat. y is float32 at 22050 Hz, peak 1.0; beats is float64 in seconds.
    """

    def _make(bpm: float = 120, bars: int = 8, beats_per_bar: int = 4) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        ibi = 60 / bpm
        beats = 0.5 + np.arange(bars * beats_per_bar) * ibi
        y = np.zeros(int((beats[-1] + 1.0) * SR), np.float32)

        t = np.arange(int(0.15 * SR)) / SR
        kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 40)) * t) * np.exp(-t * 25)
        t_snare = np.arange(int(0.1 * SR)) / SR
        snare = 0.6 * rng.standard_normal(len(t_snare)) * np.exp(-t_snare * 40)
        t_hat = np.arange(int(0.03 * SR)) / SR
        hat = 0.3 * rng.standard_normal(len(t_hat)) * np.exp(-t_hat * 150)

        for i, b in enumerate(beats):
            s = int(b * SR)
            y[s : s + len(kick)] += (1.0 if i % beats_per_bar == 0 else 0.7) * kick
            if i % 2 == 1:
                y[s : s + len(snare)] += snare
            h = int((b + ibi / 2) * SR)
            y[h : h + len(hat)] += hat
        return y / np.abs(y).max(), beats

    return _make


@pytest.fixture(scope="session")
def beat_model():
    """The beat_this model, loaded once per test session (downloads ~77 MB the first time)."""
    from amv.config import BEAT_THIS_CHECKPOINT
    from amv.music.beats import load_beat_model

    return load_beat_model(BEAT_THIS_CHECKPOINT)


@pytest.fixture(scope="session")
def bluelock_wav(tmp_path_factory):
    """The Blue Lock AMV's audio as a 22050 Hz mono wav (made once per session); skips if absent."""
    matches = sorted((DATA_DIR / "amvs").glob("*Blue Lock*.mp4"))
    if not matches:
        pytest.skip("Blue Lock AMV not found in data/amvs/")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    from amv.io.ffmpeg import make_audio_proxy

    return make_audio_proxy(matches[0], tmp_path_factory.mktemp("proxies"))
