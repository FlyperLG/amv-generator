"""Plain helpers for the checks/ tests: a fake beat table and analysis (import, don't request)."""

import numpy as np
import pandas as pd

from amv.music.analysis import SongAnalysis
from amv.music.dense import DenseFeatures

SR = 22050
DENSE_NAMES = ("rms", "onset", "flux", "low", "mid", "high")


def make_beats(beat_times: np.ndarray, sections: list[str] | None = None) -> pd.DataFrame:
    """A beat table like build_beat_table's, for 4/4 starting on a downbeat.

    sections gives one label per bar; without it every beat is "drop". The curve columns are
    simple ramps, enough for plotting."""
    n = len(beat_times)
    beat_in_bar = np.arange(n) % 4
    bar_idx = np.arange(n) // 4
    ibi = np.diff(beat_times)
    labels = np.array(sections)[bar_idx] if sections is not None else np.full(n, "drop")
    ramp = np.linspace(-1, 1, n)
    return pd.DataFrame(
        {
            "t": beat_times.astype(np.float64),
            "is_downbeat": beat_in_bar == 0,
            "beat_in_bar": beat_in_bar.astype(np.int64),
            "bar_idx": bar_idx.astype(np.int64),
            "ibi": np.append(ibi, ibi[-1]),
            "rms": ramp,
            "rms_s": ramp,
            "rms_d1": np.gradient(ramp),
            "rms_d2": np.zeros(n),
            "section": labels.astype(str),
        }
    )


def make_analysis(beats: pd.DataFrame, duration: float) -> SongAnalysis:
    """A SongAnalysis with the beat table and 12 dense curves (the 6 names, then their _s), ~43 Hz."""
    times = np.arange(0, duration, 512 / SR)
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((len(times), len(DENSE_NAMES)))
    return SongAnalysis(
        beats=beats,
        dense=DenseFeatures(
            times=times,
            values=np.hstack([raw, raw]),
            names=DENSE_NAMES + tuple(f"{n}_s" for n in DENSE_NAMES),
        ),
    )
