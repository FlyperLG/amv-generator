"""End-card detection for AMVs (step 01, task 3.8).

Many AMVs end with a few seconds that are not part of the edit: a title card, the editor's
logo, a "subscribe" screen. These are (nearly) frozen, while AMV footage changes almost every
frame. So the end card is the static stretch that runs up to the end of the file.

The video alone cuts a little late: a title card or a logo fading in still moves, so it only
counts as static once it has settled, while its jingle has already started. The audio fixes
that: the song usually ends in a short digital silence before the outro's sound, so the cut
moves back to the start of that silence. Without silence, a deep valley between a loud song
and a loud outro marks the song's end instead.

The tables (amv.io.videos, amv.io.songs) call detect_end_card on every AMV and cut both
proxies at the returned time, so every later step sees the AMV without its end card. The
original file is never changed.
"""

import os
from pathlib import Path

import numpy as np

from amv.config import EndCardSettings
from amv.io.ffmpeg import iter_frames, read_audio


def motion_curve(frames: np.ndarray) -> np.ndarray:
    """Mean absolute luma change between each pair of consecutive frames.

    Args:
        frames (np.ndarray): Gray frames, shape (N, height, width), any numeric dtype
            (uint8 from iter_frames(..., gray=True)).

    Returns:
        np.ndarray: float64, shape (N - 1,). Element i is mean(|frames[i + 1] - frames[i]|),
            on the frames' own scale (0..255 for uint8). Empty for fewer than 2 frames.

    Raises:
        ValueError: If frames is not 3-D."""
    if frames.ndim != 3:
        raise ValueError(f"Frames needs to have 3 dimensions, got {frames.ndim}")

    if len(frames) < 2:
        return np.empty(0, dtype=np.float64)

    frames = np.diff(frames.astype(np.float64), axis=0)
    return np.mean(np.abs(frames), axis=(1, 2))


def find_end_card_start(
    motion: np.ndarray,
    fps: float,
    threshold: float,
    smooth: float,
    min_tail: float,
) -> float | None:
    """Find where the static stretch at the end of a motion curve starts.

    The curve is first smoothed with a running median over smooth seconds, so a single blip
    (a sparkle on the logo, the cut between a title card and a logo) does not break the
    stretch. The static tail is then the longest run at the END of the smoothed curve whose
    values are all below threshold. If it starts at motion index j, frame j is the first
    frame of the card, and the tail lasts (len(motion) + 1 - j) / fps seconds (frame j up to
    the end of the last frame).

    Args:
        motion (np.ndarray): 1-D, from motion_curve; frames sampled at fps.
        fps (float): The rate the frames were sampled at, > 0.
        threshold (float): A smoothed value below this counts as static, > 0.
        smooth (float): Seconds of the running-median window, >= 0. The window is
            round(smooth * fps) samples, made odd by adding one if even; 0 or 1 sample
            means no smoothing.
        min_tail (float): Seconds; a shorter static tail is not an end card, >= 0.

    Returns:
        float | None: j / fps, the time of the card's first frame relative to the first
            frame of the curve; 0.0 if the whole curve is static. None if the curve is
            empty, the last value is not static, or the tail is shorter than min_tail.

    Raises:
        ValueError: If motion is not 1-D, or fps, threshold, smooth or min_tail is out of
            range."""
    if motion.ndim != 1:
        raise ValueError(f"Motion needs to have 1 dimension, got {motion.ndim}")
    if fps <= 0 or threshold <= 0:
        raise ValueError(f"fps and threshold need to be > 0, got fps={fps}, threshold={threshold}")
    if smooth < 0 or min_tail < 0:
        raise ValueError(f"smooth and min_tail need to be >= 0, got smooth={smooth}, min_tail={min_tail}")

    if len(motion) == 0:
        return None

    w = round(smooth * fps)
    w = w + 1 if w % 2 == 0 else w

    smoothed = np.array([np.median(motion[max(0, i - w // 2) : i + w // 2 + 1]) for i in range(len(motion))])
    static = smoothed < threshold
    if not static[-1]:
        return None

    j = np.flatnonzero(~static)
    j = 0 if j.size == 0 else j[-1] + 1
    tail_length = (len(motion) + 1 - j) / fps
    if tail_length < min_tail:
        return None
    return float(j / fps)


def rms_db(samples: np.ndarray, sample_rate: int, window: float) -> np.ndarray:
    """Loudness of consecutive, non-overlapping windows, in dB relative to full scale.

    Args:
        samples (np.ndarray): 1-D mono samples in [-1, 1], from read_audio.
        sample_rate (int): Their rate in Hz, > 0.
        window (float): Seconds per window, > 0; round(window * sample_rate) samples each.

    Returns:
        np.ndarray: float64, one value per complete window (a partial last window is
            dropped): 20 * log10(rms), with rms floored at 1e-12, so digital silence is
            -240 dB and never -inf or NaN. A full-scale sine is about -3 dB.

    Raises:
        ValueError: If samples is not 1-D, or sample_rate or window is <= 0."""
    if samples.ndim != 1:
        raise ValueError(f"Samples need to have 1 dimension, got {samples.ndim}")
    if sample_rate <= 0 or window <= 0:
        raise ValueError(
            f"sample_rate and window need to be > 0, got sample_rate={sample_rate}, window={window}"
        )

    n = round(window * sample_rate)
    if n == 0:
        raise ValueError(f"The audio window is too small: round({window} * {sample_rate}) = 0 samples")

    k = len(samples) // n

    x = samples[: k * n].reshape(k, n).astype(np.float64)
    rms = np.sqrt(np.mean(x**2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-12))  # floor: silence is -240 dB, not -inf


def find_song_end(db: np.ndarray, window: float, silence_db: float, min_gap: float) -> float | None:
    """Find where the music ends: the start of the last silent gap in a loudness curve.

    A window is silent if its value is below silence_db. A gap is a run of consecutive silent
    windows lasting at least min_gap seconds (count * window >= min_gap); shorter runs are
    pauses inside the music and are ignored.

    Args:
        db (np.ndarray): 1-D, from rms_db.
        window (float): Seconds per value of db, > 0.
        silence_db (float): Values below this are silent.
        min_gap (float): Seconds; the shortest gap that counts, >= 0.

    Returns:
        float | None: index * window of the first window of the LAST gap, relative to the
            start of the curve; 0.0 if that gap starts at the first window. None if there is
            no gap.

    Raises:
        ValueError: If db is not 1-D, window <= 0 or min_gap < 0."""
    if db.ndim != 1:
        raise ValueError(f"db needs to have 1 dimension, got {db.ndim}")
    if window <= 0:
        raise ValueError(f"window needs to be > 0, got {window}")
    if min_gap < 0:
        raise ValueError(f"min_gap needs to be >= 0, got {min_gap}")

    silent = np.concatenate(([0], db < silence_db, [0])).astype(np.int8)
    edges = np.diff(silent)
    starts = np.flatnonzero(edges == 1)  # first silent window of each run
    ends = np.flatnonzero(edges == -1)  # one past its last
    gaps = starts[(ends - starts) * window >= min_gap - 1e-9]  # tolerance: 5 * 0.05 must count as 0.25
    if gaps.size == 0:
        return None
    return float(gaps[-1] * window)


def find_valley(db: np.ndarray, window: float, min_dip: float) -> float | None:
    """Find the quiet point between a song and the outro that follows it.

    The fallback of find_song_end for outros without digital silence: the song fades out (or
    drops), and the outro's sound comes back up before the video cut. The valley is the
    quietest window (the first one if several are equally quiet). It only counts if it lies
    at least min_dip dB below the loudest window before it (the song) AND the loudest window
    after it (the outro), both ranges including the valley itself. A plain fade-out, with
    nothing coming back up, is not a valley.

    Args:
        db (np.ndarray): 1-D, from rms_db.
        window (float): Seconds per value of db, > 0.
        min_dip (float): dB the valley must lie below both sides, > 0.

    Returns:
        float | None: index * window of the valley, relative to the start of the curve, or
            None if the curve is empty or the dip is too shallow on either side.

    Raises:
        ValueError: If db is not 1-D, window <= 0 or min_dip <= 0."""
    if db.ndim != 1:
        raise ValueError(f"db needs to have 1 dimension, got {db.ndim}")
    if window <= 0:
        raise ValueError(f"window needs to be > 0, got {window}")
    if min_dip <= 0:
        raise ValueError(f"min_dip needs to be > 0, got {min_dip}")

    if db.size == 0:
        return None

    i = int(np.argmin(db))  # first of equal minima
    below_song = db[: i + 1].max() - db[i]
    below_outro = db[i:].max() - db[i]
    if below_song < min_dip or below_outro < min_dip:
        return None
    return float(i * window)


def detect_end_card(
    path: str | os.PathLike | Path,
    duration: float,
    settings: EndCardSettings,
    has_audio: bool,
) -> float | None:
    """Find the start of a video's end card, in seconds from the start of the video.

    Two passes, both cheap:
    1. Video: only the last settings.scan seconds are decoded (from max(0, duration - scan)),
       as width x height gray frames at settings.fps, and the static tail is found with
       motion_curve and find_end_card_start. No tail: None, and the audio is not read.
    2. Audio (if has_audio and settings.lookback > 0): the lookback seconds before that cut,
       [max(0, cut - lookback), cut), are decoded with read_audio at settings.audio_rate,
       and find_song_end looks for the silence between song and outro (rms_db over
       audio_window seconds, silence_db, min_gap). A gap moves the cut back to the gap's
       start. Without a gap, find_valley (min_dip) looks for a deep valley between song and
       outro and moves the cut to it. Neither keeps the video cut. The cut never moves later.
    Deterministic: the same file and settings always give the same float, which both tables
    rely on to cut the video and the wav at the same time.

    Args:
        path (str | os.PathLike | Path): The original video.
        duration (float): Its video stream's duration, probe(path).video.duration.
        settings (EndCardSettings): The detection settings, DECODING.end_card.
        has_audio (bool): Whether the file has an audio stream (probe(path).audio is not
            None). False skips pass 2, since read_audio fails on a file without audio.

    Returns:
        float | None: The time the AMV ends (the start of the song's final silence, else
            its valley before the outro, else the card's first frame), or None if the video
            has no end card.

    Raises:
        FileNotFoundError: If path does not exist.
        RuntimeError: If ffmpeg fails to decode it."""
    start = max(0.0, duration - settings.scan)
    frames = []
    for _, frame in iter_frames(path, settings.fps, settings.width, settings.height, start=start, gray=True):
        frames.append(frame)
    if len(frames) == 0:
        return None
    frames = np.stack(frames)
    motion = motion_curve(frames)
    result = find_end_card_start(motion, settings.fps, settings.threshold, settings.smooth, settings.min_tail)
    if result is None:
        return None

    cut = start + result
    if not has_audio or settings.lookback <= 0:
        return cut

    a0 = max(0.0, cut - settings.lookback)
    samples = read_audio(path, settings.audio_rate, start=a0, end=cut)
    db = rms_db(samples, settings.audio_rate, settings.audio_window)
    rel = find_song_end(db, settings.audio_window, settings.silence_db, settings.min_gap)
    if rel is None:
        rel = find_valley(db, settings.audio_window, settings.min_dip)  # no silence: try a valley
    return cut if rel is None else a0 + rel
