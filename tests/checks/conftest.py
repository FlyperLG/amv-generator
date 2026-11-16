"""Fixtures for the checks/ tests: a synthetic kick loop with exactly known beat times.

No beat model and no real media: the beat table for the loop is built from its beat times
(tests/checks/helpers.py) instead of being detected.
"""

import numpy as np
import pytest

from tests.checks.helpers import SR


@pytest.fixture
def kick_loop():
    """Factory: a kick on every beat (louder on the first of each bar), snare on beats 2 and 4.

    Usage:  y, beats = kick_loop(bpm=120, bars=8, shift=0.0)

    The first beat is at 0.5 s and the loop ends 1 s after the last beat. shift moves every
    hit later by that many seconds while beats stays put, to fake a late beat grid. y is
    float32 at 22050 Hz with a peak of 1.0.
    """

    def _make(bpm: float = 120, bars: int = 8, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        beats = 0.5 + np.arange(bars * 4) * 60 / bpm
        y = np.zeros(int((beats[-1] + 1.0) * SR), np.float32)
        t = np.arange(int(0.15 * SR)) / SR
        kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 40)) * t) * np.exp(-t * 25)
        t_snare = np.arange(int(0.1 * SR)) / SR
        snare = 0.6 * rng.standard_normal(len(t_snare)) * np.exp(-t_snare * 40)
        for i, b in enumerate(beats):
            s = int((b + shift) * SR)
            y[s : s + len(kick)] += (1.0 if i % 4 == 0 else 0.7) * kick
            if i % 2 == 1:
                y[s : s + len(snare)] += snare
        return y / np.abs(y).max(), beats

    return _make
