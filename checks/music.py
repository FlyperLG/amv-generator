"""What the step 02 check (task 3.7) computes and draws; checks/plot_music.py runs it per song.

Nothing here is part of the pipeline. It reads the step 02 cache and recomputes only what the
cache leaves out (waveform envelope, section novelty), measures the beat offset and kick
pattern, builds the click track, and draws the five panels. Functions take plain arguments;
only plot_music.py reads the settings.
"""

from dataclasses import dataclass

import librosa
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.signal import find_peaks, peak_prominences

from amv.config import MusicSettings
from amv.music.analysis import SongAnalysis
from amv.music.dense import DenseFeatures, compute_timbre_features
from amv.music.dynamics import pool_to_beats
from amv.music.sections import SECTION_LABELS, find_section_boundaries, section_novelty


def waveform_envelope(y: np.ndarray, sample_rate: int, hop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lowest and highest sample of every hop: a waveform small enough to plot.

    Only whole hops are used; the < hop samples left at the end are dropped.

    Args:
        y (np.ndarray): The mono signal. Not modified.
        sample_rate (int): The sample rate of y in Hz.
        hop (int): Samples per bucket.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (times, lo, hi), each of shape
            [len(y) // hop]; times is the centre of each bucket in seconds."""
    hops = y[: len(y) // hop * hop].reshape(-1, hop)
    lo = hops.min(axis=1)
    hi = hops.max(axis=1)
    times = (np.arange(hops.shape[0]) + 0.5) * hop / sample_rate
    return times, lo, hi


@dataclass(frozen=True, slots=True)
class SectionCheck:
    """The section novelty of a song, recomputed for the novelty panel (it isn't cached)."""

    bar_times: np.ndarray  # [n_bars] downbeat times, bar k = k-th downbeat
    novelty: np.ndarray  # [n_bars]
    boundaries: np.ndarray  # bar indices from find_section_boundaries, starts with 0
    peaks: np.ndarray  # bar indices of every local maximum of novelty
    prominences: np.ndarray  # [len(peaks)] prominence of each


def recompute_sections(
    y: np.ndarray,
    sample_rate: int,
    beats: pd.DataFrame,
    settings: MusicSettings,
) -> SectionCheck:
    """Recompute the section novelty and boundaries the way _analyse does, plus every candidate peak.

    The timbre is pooled per bar on the cached beat table's downbeats, so the result matches
    the cached labels unless the code or the wav changed since (see stale_boundaries).

    Args:
        y (np.ndarray): The mono signal from load_audio. Not modified.
        sample_rate (int): The sample rate of y in Hz.
        beats (pd.DataFrame): The cached beat table; t and is_downbeat are used. Not modified.
        settings (MusicSettings): The step 02 settings the cache was made with.

    Returns:
        SectionCheck: Bar times, novelty, the chosen boundaries, and every local maximum of
            the novelty with its prominence (scipy.signal.peak_prominences)."""
    timbre = compute_timbre_features(
        y=y,
        sample_rate=sample_rate,
        hop_length=settings.hop_length,
        n_fft=settings.n_fft,
        n_mels=settings.n_mels,
        n_mfcc=settings.n_mfcc,
    )
    bar_times = beats["t"][beats["is_downbeat"]].to_numpy()
    bars = pool_to_beats(dense=timbre, beat_times=bar_times)
    novelty = section_novelty(bar_features=bars, kernel_bars=settings.section_kernel_bars)
    boundaries = find_section_boundaries(
        novelty=novelty,
        min_section_bars=settings.min_section_bars,
        edge_bars=settings.edge_bars,
        prominence=settings.novelty_prominence,
    )
    peaks, _ = find_peaks(novelty)
    prominences = peak_prominences(x=novelty, peaks=peaks)[0]
    return SectionCheck(
        bar_times=bar_times,
        novelty=novelty,
        boundaries=boundaries,
        peaks=peaks,
        prominences=prominences,
    )


def stale_boundaries(beats: pd.DataFrame, boundaries: np.ndarray) -> np.ndarray:
    """Find cached label changes that the recomputed boundaries don't explain.

    Only that direction: a boundary without a label change is normal, since two adjacent
    segments can both be drops. The first row never counts as a change, so a pickup
    (bar_idx -1) is never reported. A non-empty result means the cache is stale.

    Args:
        beats (pd.DataFrame): The cached beat table; section and bar_idx are used. Not modified.
        boundaries (np.ndarray): The bar indices from recompute_sections.

    Returns:
        np.ndarray: The sorted bar indices where a label changes but no boundary is; empty
            if the cache matches."""
    mask = beats["section"] != beats["section"].shift()
    mask.iloc[0] = False
    borders = beats.loc[mask, "bar_idx"]
    return np.setdiff1d(borders, boundaries)


def make_click_track(
    beat_times: np.ndarray,
    is_downbeat: np.ndarray,
    n_samples: int,
    sample_rate: int,
    click_freq: float,
    downbeat_click_freq: float,
    click_duration: float,
    beat_gain: float,
    downbeat_gain: float,
) -> np.ndarray:
    """A track of short clicks on the beats: higher and louder on downbeats.

    Each beat gets exactly one click (a downbeat only the downbeat click), so the loudest
    sample is about 0.94 times the larger gain and nothing clips at gains <= 1.

    Args:
        beat_times (np.ndarray): The beat times in seconds, shape [B]. Not modified.
        is_downbeat (np.ndarray): bool, shape [B]; which beats are downbeats.
        n_samples (int): The length of the track: len(y), so it lines up with the music.
        sample_rate (int): Hz.
        click_freq (float): Pitch of the beat clicks in Hz.
        downbeat_click_freq (float): Pitch of the downbeat clicks in Hz.
        click_duration (float): Seconds per click.
        beat_gain (float): Peak level of the beat clicks.
        downbeat_gain (float): Peak level of the downbeat clicks.

    Returns:
        np.ndarray: float32, shape [n_samples]; silence where there is no click."""
    beat_clicks = (
        librosa.clicks(
            times=beat_times[~is_downbeat],
            sr=sample_rate,
            click_freq=click_freq,
            click_duration=click_duration,
            length=n_samples,
        )
        * beat_gain
    )
    downbeat_clicks = (
        librosa.clicks(
            times=beat_times[is_downbeat],
            sr=sample_rate,
            click_freq=downbeat_click_freq,
            click_duration=click_duration,
            length=n_samples,
        )
        * downbeat_gain
    )

    return beat_clicks + downbeat_clicks


def stereo_click_mix(y: np.ndarray, clicks: np.ndarray) -> np.ndarray:  # float32 [n, 2]
    """Put the music on the left channel and the clicks on the right, for listening on headphones.

    Both are already within -1..1, so there is no mixing and no gain to balance.

    Args:
        y (np.ndarray): The mono music, shape [n]. Not modified.
        clicks (np.ndarray): The click track from make_click_track, shape [n]. Not modified.

    Returns:
        np.ndarray: float32, shape [n, 2].

    Raises:
        ValueError: If y and clicks differ in length."""
    if len(y) != len(clicks):
        raise ValueError(
            f"Both soundtracks need to have the same lengths: y({len(y)}), clicks({len(clicks)})"
        )
    return np.column_stack((y, clicks)).astype(np.float32, copy=False)


def beat_offsets(
    y: np.ndarray,
    sample_rate: int,
    beat_times: np.ndarray,
    hop_length: int,
    n_fft: int,
    max_offset: float,
    peak_percentile: float,
) -> np.ndarray:
    """Measure how far each beat is from the nearest onset in the audio: the click-track check by numbers.

    Onsets are the peaks of librosa's onset strength at a fine hop (64 samples ~ 2.9 ms, against
    23 ms in the pipeline). A peak counts if its prominence is at least the peak_percentile-th
    percentile of the curve. The frames are centred, so frame k is at k * hop_length / sample_rate.
    Calibrated on a synthetic kick loop with exact beat times at 90-176 BPM: +4 to +5 ms,
    because the onset strength peaks a few ms after a kick's attack; kicks moved 30 ms late
    read +34 ms. A consistent offset well under ~20 ms is inaudible.

    Args:
        y (np.ndarray): The mono signal from load_audio. Not modified.
        sample_rate (int): The sample rate of y in Hz.
        beat_times (np.ndarray): The beat times in seconds, shape [B]. Not modified.
        hop_length (int): Samples between onset frames; sets the time resolution.
        n_fft (int): Samples per onset analysis window.
        max_offset (float): Seconds; a beat with no onset this close gets NaN.
        peak_percentile (float): Percentile of the onset curve a peak's prominence must reach.

    Returns:
        np.ndarray: Per beat, nearest onset time minus beat time in seconds (positive = the
            onset comes after the beat), shape [B]; NaN where no onset is within max_offset."""
    offsets = np.full(len(beat_times), np.nan)
    if len(beat_times) == 0:
        return offsets
    envelope = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)
    peaks, _ = find_peaks(envelope, prominence=np.percentile(envelope, peak_percentile))
    if len(peaks) == 0:
        return offsets

    onset_times = librosa.frames_to_time(peaks, sr=sample_rate, hop_length=hop_length)
    after = np.searchsorted(onset_times, beat_times)  # first onset at or after each beat
    d_after = onset_times[np.minimum(after, len(onset_times) - 1)] - beat_times
    d_before = onset_times[np.maximum(after - 1, 0)] - beat_times
    nearest = np.where(np.abs(d_before) <= np.abs(d_after), d_before, d_after)
    return np.where(np.abs(nearest) <= max_offset, nearest, np.nan)


def kick_by_position(
    y: np.ndarray,
    sample_rate: int,
    beat_times: np.ndarray,
    beat_in_bar: np.ndarray,
    beats_per_bar: int,
    hop_length: int,
    n_fft: int,
    max_hz: float,
    window: float,
) -> np.ndarray:
    """How hard the kick hits on each position in the bar: the "downbeat is on 1" check by numbers.

    The kick of a beat is the strongest rise of the energy below max_hz in the frames from the
    beat to window seconds after it. In most songs the kick is strongest on "1", so position 0
    should score 1.0. A song with a kick on every beat (four-on-the-floor) scores about equal
    everywhere and can't be judged this way.

    Args:
        y (np.ndarray): The mono signal from load_audio. Not modified.
        sample_rate (int): The sample rate of y in Hz.
        beat_times (np.ndarray): The beat times in seconds, shape [B]. Not modified.
        beat_in_bar (np.ndarray): Each beat's position in its bar, 0..beats_per_bar - 1, shape [B].
        beats_per_bar (int): The number of positions.
        hop_length (int): Samples between STFT frames.
        n_fft (int): Samples per STFT window.
        max_hz (float): The highest frequency that counts as kick, in Hz.
        window (float): Seconds after each beat searched for its kick; at least one frame.

    Returns:
        np.ndarray: The mean kick per position divided by the largest, shape [beats_per_bar];
            all zeros if there is no kick at all. A position with no beats scores 0."""
    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    low = spectrum[librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft) <= max_hz].sum(axis=0)
    rise = np.maximum(0.0, np.diff(low, prepend=low[0]))

    n_frames = max(1, round(window * sample_rate / hop_length))
    first = librosa.time_to_frames(beat_times, sr=sample_rate, hop_length=hop_length)
    frames = np.clip(first[:, None] + np.arange(n_frames), 0, len(rise) - 1)
    kick = rise[frames].max(axis=1)

    total = np.bincount(beat_in_bar, weights=kick, minlength=beats_per_bar)
    count = np.bincount(beat_in_bar, minlength=beats_per_bar)
    mean = np.divide(total, count, out=np.zeros(beats_per_bar), where=count > 0)
    top = mean.max()
    return mean / top if top > 0 else mean


def window_mask(times: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:  # bool
    """Which times are inside the plot window, both edges included; None keeps all."""
    if window is None:
        return np.ones(len(times), dtype=bool)

    start, end = window
    return (times >= start) & (times <= end)


SECTION_COLORS: dict[str, str] = {  # one per SECTION_LABELS; a fixed vocabulary, not a setting
    "intro": "tab:gray",
    "build": "tab:orange",
    "drop": "tab:red",
    "break": "tab:blue",
    "outro": "tab:purple",
}


def _column(dense: DenseFeatures, name: str) -> np.ndarray:
    """The dense curve called name, shape [T]."""
    return dense.values[:, dense.names.index(name)]


def _beat_mask(beats: pd.DataFrame, window: tuple[float, float] | None) -> np.ndarray:
    """Which beats to draw: those whose interval [t, t + ibi) overlaps the window; None keeps all.

    Unlike window_mask, this keeps the beat starting just before the window, so a step plot
    reaches the window's left edge."""
    times = beats["t"].to_numpy()
    ibi = beats["ibi"].to_numpy()

    if window is None:
        return np.ones(len(times), dtype=bool)

    start, end = window
    return (times + ibi >= start) & (times <= end)


def _beat_steps(beats: pd.DataFrame, column: str, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x and y for ax.step(where="post") of a beat column: each value held until the next beat.

    One point is added at the end of the last beat (t + ibi) with the last value, so the
    last beat is drawn too. No beats in mask gives two empty arrays."""
    t = beats["t"].to_numpy()[mask]
    if len(t) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    ibi = beats["ibi"].to_numpy()[mask]
    v = beats[column].to_numpy()[mask]
    x = np.append(t, t[-1] + ibi[-1])
    y = np.append(v, v[-1])
    return x, y


def draw_sections(ax: Axes, beats: pd.DataFrame) -> None:
    """Colour the background of ax by section: one band per run of equal labels, behind everything.

    A run goes from its first beat to the next run's first beat; the last ends at the last
    t + ibi."""
    t = beats["t"].to_numpy()
    ibi = beats["ibi"].to_numpy()
    section = beats["section"].to_numpy()
    starts = np.r_[0, np.flatnonzero(section[1:] != section[:-1]) + 1]

    for start, end, label in zip(
        t[starts], np.r_[t[starts[1:]], t[-1] + ibi[-1]], section[starts], strict=True
    ):
        ax.axvspan(start, end, color=SECTION_COLORS[label], alpha=0.15, lw=0, zorder=0)


def draw_beat_ticks(ax: Axes, beats: pd.DataFrame) -> None:
    """Draw a full-height line at every beat: thin and light for beats, bold and dark for downbeats.

    x is in seconds and y in panel height, so the ticks never change the y range."""
    t = beats["t"].to_numpy()
    is_downbeat = beats["is_downbeat"].to_numpy()
    ax.vlines(t[~is_downbeat], 0, 1, transform=ax.get_xaxis_transform(), colors="0.75", lw=0.5, zorder=1)
    ax.vlines(t[is_downbeat], 0, 1, transform=ax.get_xaxis_transform(), colors="0.2", lw=1.2, zorder=1)


def plot_waveform(
    ax: Axes, times: np.ndarray, lo: np.ndarray, hi: np.ndarray, window: tuple[float, float] | None
) -> None:
    """Draw the waveform envelope from waveform_envelope on a fixed -1..1 scale.

    The scale is fixed, so a quiet passage also looks quiet in a window."""
    m = window_mask(times=times, window=window)
    ax.fill_between(times[m], lo[m], hi[m], color="tab:blue", lw=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("waveform")


def plot_onset(ax: Axes, dense: DenseFeatures, window: tuple[float, float] | None) -> None:
    """Draw the raw dense onset curve; its peaks should sit on the beat ticks.

    Raw, not onset_s: smoothing would smear exactly the peak to compare with the tick."""
    m = window_mask(dense.times, window)
    ax.plot(dense.times[m], _column(dense, "onset")[m], color="tab:green", lw=0.8)
    ax.set_ylabel("onset")


def plot_rms(ax: Axes, dense: DenseFeatures, beats: pd.DataFrame, window: tuple[float, float] | None) -> None:
    """Draw the loudness three ways: dense rms (light), dense rms_s, and the beat rms_s as steps."""
    times = dense.times
    m = window_mask(dense.times, window)
    bm = _beat_mask(beats, window)
    ax.plot(times[m], _column(dense, "rms")[m], color="tab:blue", lw=0.5, alpha=0.4, label="rms")
    ax.plot(times[m], _column(dense, "rms_s")[m], color="tab:blue", lw=1.2, label="rms_s (dense)")
    ax.step(*_beat_steps(beats, "rms_s", bm), where="post", color="black", lw=1, label="rms_s (beat)")
    ax.set_ylabel("rms")
    ax.legend(loc="upper right", fontsize=7)


def plot_dynamics(ax: Axes, beats: pd.DataFrame, window: tuple[float, float] | None) -> None:
    """Draw rms_d1 and rms_d2 per beat as steps around a zero line; the units are per beat."""
    bm = _beat_mask(beats, window)
    ax.step(*_beat_steps(beats, "rms_d1", bm), where="post", color="tab:orange", label="rms_d1")
    ax.step(*_beat_steps(beats, "rms_d2", bm), where="post", color="tab:red", label="rms_d2")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Δ per beat")
    ax.legend(loc="upper right", fontsize=7)


def plot_novelty(
    ax: Axes,
    check: SectionCheck,
    prominence: float,
    edge_bars: int,
    window: tuple[float, float] | None,
) -> None:
    """Draw the section novelty and why each peak was or wasn't chosen as a boundary.

    Every candidate peak in the window gets a bar from its contour line up to the peak (its
    prominence) and a label with the value: chosen ones in colour with a marker, rejected
    ones in grey. The first and last edge_bars bars are hatched, since peaks there
    are always dropped. The threshold is a prominence, not a height, so it is in the legend,
    not a line. Reading it: a grey peak with a short bar was too weak, one in the hatched
    zone too close to an end, one with a long bar lost to a taller peak less than
    MUSIC.min_section_bars away.

    Args:
        ax (Axes): The panel to draw into.
        check (SectionCheck): From recompute_sections.
        prominence (float): MUSIC.novelty_prominence, for the legend.
        edge_bars (int): MUSIC.edge_bars, for the hatched zones.
        window (tuple[float, float] | None): The plot window in seconds, or None."""
    x = check.bar_times
    nov = check.novelty
    p = check.peaks
    prom = check.prominences

    m = window_mask(x, window)
    ax.plot(x[m], nov[m], color="tab:purple", lw=1, label="novelty")
    picked = np.isin(p, check.boundaries)
    px = x[p]
    py = nov[p]
    base = py - prom
    pm = window_mask(px, window)
    sel = pm & picked
    rej = pm & ~picked
    ax.vlines(px[sel], base[sel], py[sel], colors="tab:purple", lw=2)
    ax.vlines(px[rej], base[rej], py[rej], colors="0.6", lw=1.5)
    ax.plot(px[sel], py[sel], "v", color="tab:purple", label="boundary")

    for px_i, py_i, prom_i, is_picked in zip(px[pm], py[pm], prom[pm], picked[pm], strict=True):
        ax.annotate(
            f"{prom_i:.2f}",
            (px_i, py_i),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color="tab:purple" if is_picked else "0.4",
        )

    n = len(x)
    if n > 2 * edge_bars:
        ax.axvspan(x[0], x[edge_bars], fill=False, hatch="//", edgecolor="0.7", lw=0)
        ax.axvspan(x[n - edge_bars], x[-1], fill=False, hatch="//", edgecolor="0.7", lw=0)

    ax.plot([], [], " ", label=f"prominence ≥ {prominence:.2f}")
    ax.margins(y=0.15)
    ax.set_ylabel("novelty")
    ax.legend(loc="upper right", fontsize=7)


def plot_song(
    title: str,
    analysis: SongAnalysis,
    envelope: tuple[np.ndarray, np.ndarray, np.ndarray],
    check: SectionCheck,
    prominence: float,
    edge_bars: int,
    window: tuple[float, float] | None,
    fig_width: float,
    panel_height: float,
) -> Figure:
    """Lay out one song's five panels on a shared time axis: waveform, onset, rms, dynamics, novelty.

    Built as a plain Figure, not with pyplot: no window opens, nothing has to be closed, and
    the caller saves it with fig.savefig. Every panel gets the section bands, the first two
    the beat ticks. x runs from 0 to the song end, or over the window (an open end stops at
    the song end). The top panel's legend lists only the sections the song has.

    Args:
        title (str): The figure title.
        analysis (SongAnalysis): The cached analysis from load_analysis.
        envelope (tuple[np.ndarray, np.ndarray, np.ndarray]): From waveform_envelope.
        check (SectionCheck): From recompute_sections.
        prominence (float): MUSIC.novelty_prominence.
        edge_bars (int): MUSIC.edge_bars, for the hatched zones of the novelty panel.
        window (tuple[float, float] | None): The plot window in seconds, or None.
        fig_width (float): Inches.
        panel_height (float): Inches per panel.

    Returns:
        Figure: The figure, not yet saved."""
    fig = Figure(figsize=(fig_width, 5 * panel_height), layout="constrained")
    axes = fig.subplots(5, 1, sharex=True)
    env_times, lo, hi = envelope

    plot_waveform(axes[0], env_times, lo, hi, window)
    plot_onset(axes[1], analysis.dense, window)
    plot_rms(axes[2], analysis.dense, analysis.beats, window)
    plot_dynamics(axes[3], analysis.beats, window)
    plot_novelty(axes[4], check, prominence, edge_bars, window)

    for ax in axes:
        draw_sections(ax, analysis.beats)
    draw_beat_ticks(axes[0], analysis.beats)
    draw_beat_ticks(axes[1], analysis.beats)

    song_end = analysis.dense.times[-1]
    if window is None:
        axes[0].set_xlim(0, song_end)
    else:
        start, end = window
        axes[0].set_xlim(start, min(end, song_end))

    axes[-1].set_xlabel("time (s)")

    handles = [
        Patch(color=SECTION_COLORS[label], alpha=0.3, label=label)
        for label in SECTION_LABELS
        if label in set(analysis.beats["section"])
    ]

    axes[0].legend(handles=handles, loc="upper right", ncols=len(handles), fontsize=7)

    fig.suptitle(title)
    return fig
