import numpy as np
from scipy.signal import savgol_filter

from amv.music.dense import DenseFeatures


def _odd_window(length: float, polyorder: int) -> int:
    """Round length to the nearest odd int, at least the smallest odd int > polyorder."""
    odd_number = 2 * round((length - 1) / 2) + 1
    if odd_number <= polyorder:
        return polyorder + 1 if polyorder % 2 == 0 else polyorder + 2
    return odd_number


def smooth_dense(
    dense: DenseFeatures,
    beat_period: float,
    window_beats: float,
    polyorder: int,
) -> DenseFeatures:
    """Smooth every dense curve with a Savitzky–Golay filter about window_beats beats wide.

    The window is the odd number of frames nearest to window_beats * beat_period, using the
    frame period of dense.times, and at least the smallest odd number above polyorder. The
    edges are fitted (mode="interp"), so a polynomial of degree <= polyorder passes through
    unchanged, first and last frames included.

    Args:
        dense (DenseFeatures): The curves to smooth, with at least 2 frames. Not modified.
        beat_period (float): The typical beat interval in seconds, e.g. the median IBI.
        window_beats (float): The window width in beats.
        polyorder (int): The degree of the polynomial fitted in each window.

    Returns:
        DenseFeatures: New features with the same times, the smoothed values and every name
            suffixed with "_s" ("rms" -> "rms_s")."""
    frame_period = dense.times[1] - dense.times[0]
    window = _odd_window(window_beats * beat_period / frame_period, polyorder)
    return DenseFeatures(
        times=dense.times,
        values=np.asarray(
            savgol_filter(
                dense.values,
                window,
                polyorder,
                axis=0,
                mode="interp",
            )
        ),
        names=tuple(f"{n}_s" for n in dense.names),
    )


def pool_to_beats(
    dense: DenseFeatures,
    beat_times: np.ndarray,
) -> np.ndarray:
    """Pool the dense curves to one row per beat: the mean of each beat's frames, the max for onset.

    Beat i owns the frames at times t with beat_times[i] <= t < beat_times[i + 1]. The last
    beat owns one more interval as long as the one before it, clipped to the frames that
    exist; frames before the first beat belong to no beat. A beat that owns no frame takes
    the frame nearest to its time, so the result never contains NaN. Passing the downbeat
    times instead pools per bar, which is how the timbre features are pooled for sections.

    Args:
        dense (DenseFeatures): The raw (unsmoothed) curves. An "onset" column, if there is
            one, takes the max; every other column the mean. Not modified.
        beat_times (np.ndarray): The sorted beat (or downbeat) times in seconds, shape [B].
            Not modified.

    Returns:
        np.ndarray: The pooled features, shape [B, F], columns in dense.names order.

    Raises:
        ValueError: If there are fewer than 2 beats."""
    if beat_times.size < 2:
        raise ValueError(f"Pooling beats requires at least two beats, got {beat_times.size}")

    ends = beat_times[1:]
    ends = np.append(ends, beat_times[-1] + (beat_times[-1] - beat_times[-2]))
    start = np.searchsorted(dense.times, beat_times)  # beat i owns values[start[i]:stop[i]]
    stop = np.searchsorted(dense.times, ends)
    counts = stop - start

    # zero row: makes index T valid for reduceat
    padded = np.vstack([dense.values, np.zeros((1, dense.values.shape[1]))])
    # [start0, stop0, start1, ...]: even entries reduce start[i]:stop[i], odd ones are dropped
    idx = np.empty(2 * start.size, dtype=np.intp)
    idx[0::2] = start
    idx[1::2] = stop

    sums = np.add.reduceat(padded, idx, axis=0)[0::2]
    pooled = sums / np.maximum(counts, 1)[:, None]  # empty rows are fixed below
    if "onset" in dense.names:
        onset_col = dense.names.index("onset")
        pooled[:, onset_col] = np.maximum.reduceat(padded[:, onset_col], idx)[0::2]

    # empty beat -> nearest frame, which is start - 1 or start
    empty = counts == 0
    if empty.any():
        right = np.clip(start[empty], 1, len(dense.times) - 1)
        left = right - 1
        bt = beat_times[empty]
        nearest = np.where(bt - dense.times[left] <= dense.times[right] - bt, left, right)
        pooled[empty] = dense.values[nearest]
    return pooled


def beat_dynamics(
    pooled: np.ndarray,
    names: tuple[str, ...],
    window: int,
    polyorder: int,
) -> dict[str, np.ndarray]:
    """Smooth the pooled beat series and take its first and second derivative, per beat.

    A Savitzky–Golay filter fits a polynomial over each window of beats and reads off its
    value (_s), slope (_d1) and curvature (_d2). The derivatives are per beat, not per
    second, so they don't depend on the tempo.

    Args:
        pooled (np.ndarray): The pooled features from pool_to_beats, shape [B, F]. Not modified.
        names (tuple[str, ...]): The feature names, one per column of pooled.
        window (int): The window width in beats. Odd, larger than polyorder, at most B.
        polyorder (int): The degree of the polynomial fitted in each window.

    Returns:
        dict[str, np.ndarray]: 4 * F arrays of shape [B], keyed in this order: every name
            (the raw pooled column), then every name + "_s", + "_d1" and + "_d2".

    Raises:
        ValueError: If names doesn't have one entry per column, or there are fewer than
            window beats."""
    if len(names) != pooled.shape[1]:
        raise ValueError("Names need to match the length of the pooled vector.")

    result: dict[str, np.ndarray] = {}
    s = np.asarray(savgol_filter(pooled, window, polyorder, deriv=0, delta=1.0, axis=0, mode="interp"))
    d1 = np.asarray(savgol_filter(pooled, window, polyorder, deriv=1, delta=1.0, axis=0, mode="interp"))
    d2 = np.asarray(savgol_filter(pooled, window, polyorder, deriv=2, delta=1.0, axis=0, mode="interp"))
    for i, name in enumerate(names):
        result[name] = pooled[:, i].copy()
    for i, name in enumerate(names):
        result[f"{name}_s"] = s[:, i]
    for i, name in enumerate(names):
        result[f"{name}_d1"] = d1[:, i]
    for i, name in enumerate(names):
        result[f"{name}_d2"] = d2[:, i]
    return result
