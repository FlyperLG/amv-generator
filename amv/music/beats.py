import functools
import logging
from dataclasses import dataclass

import numpy as np
from beat_this.inference import Audio2Beats

logger = logging.getLogger(__name__)

_OFF_TEMPO_TOLERANCE = 0.10  # a split part this far from the median doesn't fit the tempo
_OFF_TEMPO_WARN_FRACTION = 0.05  # share of intervals before warning


@functools.cache
def load_beat_model(checkpoint: str) -> Audio2Beats:
    """Load the beat_this model on CPU, once per checkpoint.

    Loading takes seconds, so the result is cached: another call with the same
    checkpoint returns the same object.

    Args:
        checkpoint (str): The beat_this checkpoint name or path.

    Returns:
        Audio2Beats: The model, ready to be passed to detect_beats."""
    return Audio2Beats(checkpoint_path=checkpoint, device="cpu", float16=False, dbn=False)


def detect_beats(y: np.ndarray, sample_rate: int, model: Audio2Beats) -> tuple[np.ndarray, np.ndarray]:
    """Run the beat tracker on a signal, without any cleaning (see clean_beats).

    Args:
        y (np.ndarray): The mono signal.
        sample_rate (int): The sample rate of y in Hz.
        model (Audio2Beats): The model from load_beat_model.

    Returns:
        tuple[np.ndarray, np.ndarray]: The beat times and the downbeat times in seconds,
            both 1-D float64 arrays."""
    beats, downbeat = model(y, sample_rate)
    return np.array(beats, dtype=np.float64), np.array(downbeat, dtype=np.float64)


def clean_beats(beats, min_ibi: float) -> np.ndarray:
    """Drop beats that are too close together and fill missed or half-tempo beats.

    A beat is kept only if it is at least min_ibi after the last kept beat. Every interval
    of about n times the median interval (n >= 2) is then split into n equal parts, with n
    capped so that no part is shorter than min_ibi. The first and last kept beat stay the
    ends.

    The median is song-wide, so a real tempo change can be misread as half tempo. A warning
    is logged if over 5% of the intervals were split into parts more than 10% away from the
    median. A change of exactly 2x looks like half tempo and goes undetected.

    Args:
        beats (ArrayLike): The sorted beat times in seconds. Not modified.
        min_ibi (float): The minimum interval between two beats in seconds.

    Returns:
        np.ndarray: The cleaned beat times as a new, strictly increasing float64 array.
            Fewer than 2 input beats are returned unchanged."""
    if len(beats) < 2:
        return np.array(beats, dtype=np.float64)

    current_beat = beats[0]
    cleaned_beats: list[float] = [current_beat]
    intervals: list[float] = []
    for i in range(1, len(beats)):
        if current_beat + min_ibi > beats[i]:
            continue

        cleaned_beats.append(beats[i])
        intervals.append(beats[i] - current_beat)
        current_beat = beats[i]

    if len(intervals) == 0:
        return np.array(cleaned_beats, dtype=np.float64)

    median_beats = np.median(intervals)

    off_tempo_splits = 0
    filled_beats: list[float] = []
    for i in range(len(intervals)):
        filled_beats.append(cleaned_beats[i])
        n = min(round(intervals[i] / median_beats), int(intervals[i] // min_ibi))
        if n >= 2:
            gap_fill = np.linspace(cleaned_beats[i], cleaned_beats[i + 1], n + 1)[1:-1]
            for fill in gap_fill:
                filled_beats.append(fill)
            if abs(intervals[i] / n - median_beats) / median_beats > _OFF_TEMPO_TOLERANCE:
                off_tempo_splits += 1

    off_tempo_fraction = off_tempo_splits / len(intervals)
    if off_tempo_fraction > _OFF_TEMPO_WARN_FRACTION:
        logger.warning(
            "%.0f%% of beat intervals don't fit the median of %.3f s, the song may change tempo.",
            off_tempo_fraction * 100,
            median_beats,
        )

    filled_beats.append(cleaned_beats[-1])
    return np.array(filled_beats, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BeatGrid:
    """Every beat of a song and its place in the bars, as 1-D arrays of the same length."""

    times: np.ndarray  # float64, beat times in seconds
    is_downbeat: np.ndarray  # bool, exactly where beat_in_bar == 0
    beat_in_bar: np.ndarray  # int, 0 .. beats_per_bar - 1
    bar_idx: np.ndarray  # int, increases at every downbeat; -1 before the first one (a pickup)


def build_beat_grid(
    beats: np.ndarray,
    downbeats: np.ndarray,
    beats_per_bar: int,
    tol: float,
    window_bars: int,
) -> BeatGrid:
    """Assign every beat its place in the bars, using the tracker's downbeats as votes.

    The downbeats are too noisy to count beats from directly. Each one is matched to its
    nearest beat, and for each beat the matched beats within window_bars bars vote on the
    bar offset (matched index % beats_per_bar). On a tie, or without votes, the beat keeps
    the previous beat's offset; beat 0 takes the smallest tied offset. Where the offset
    changes, one bar is shorter or longer than beats_per_bar.

    Args:
        beats (np.ndarray): The cleaned beat times in seconds. Not modified.
        downbeats (np.ndarray): The tracker's downbeat times in seconds. Not modified.
        beats_per_bar (int): The number of beats in a bar.
        tol (float): The maximum distance in seconds for matching a downbeat to a beat.
            Downbeats further away are ignored.
        window_bars (int): How many bars on each side of a beat get a vote.

    Returns:
        BeatGrid: The grid, empty if beats is empty."""
    if beats.size == 0:
        return BeatGrid(
            times=np.empty(0, dtype=np.float64),
            is_downbeat=np.empty(0, dtype=np.bool),
            beat_in_bar=np.empty(0, dtype=np.int64),
            bar_idx=np.empty(0, dtype=np.int64),
        )

    marked_beat_indices: set[int] = set()
    for downbeat in downbeats:
        index = np.argmin(np.abs(beats - downbeat))
        if abs(beats[index] - downbeat) <= tol:
            marked_beat_indices.add(int(index))

    vote_radius = window_bars * beats_per_bar
    previous_bar_offset = 0

    beat_in_bar = np.zeros(len(beats), dtype=int)
    for beat_index in range(len(beats)):
        votes = np.zeros(beats_per_bar, dtype=int)

        for marked_index in marked_beat_indices:
            if abs(marked_index - beat_index) <= vote_radius:
                voted_offset = marked_index % beats_per_bar
                votes[voted_offset] += 1

        number_of_winners = np.sum(votes == votes.max())
        if number_of_winners == 1 or beat_index == 0:
            bar_offset = np.argmax(votes)
        else:
            bar_offset = previous_bar_offset

        beat_in_bar[beat_index] = (beat_index - bar_offset) % beats_per_bar
        previous_bar_offset = bar_offset

    is_downbeat = beat_in_bar == 0

    return BeatGrid(
        times=beats.copy(),
        is_downbeat=is_downbeat,
        beat_in_bar=beat_in_bar,
        bar_idx=np.cumsum(is_downbeat) - 1,
    )
