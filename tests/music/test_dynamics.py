"""Tests for amv.music.dynamics (docs/02-music-features, task 3.4).

Almost everything here uses hand-made DenseFeatures and pooled arrays whose answer can be
worked out on paper: a frame index as the value, a ramp, a parabola. The window rounding is
only checked through smooth_dense, so these tests don't care whether odd_window is public
or private. One integration test runs the whole chain on the Blue Lock song.

SPECIFICATION
  Settings are required arguments (Decision: dynamics.py does not read amv.config; the entry
  point reads the values from config and passes them in).
  (config: SG_POLYORDER = 2, DENSE_SG_WINDOW_BEATS = 1.0, BEAT_SG_WINDOW = 5)

  The order of operations (Decision: pool the RAW dense curves, not the smoothed ones.
  Averaging over a beat is already a 1-beat smoothing, so pooling the dense _s curves would
  smooth twice; and the schema's rms column is "pooled", rms_s "smoothed"):
      dense ──smooth_dense──> dense _s curves         (for the npz, the 3.6 plot, step 07)
      dense ──pool_to_beats──> [B, F] ──beat_dynamics──> raw, _s, _d1, _d2 per beat

  smooth_dense(dense, beat_period, window_beats, polyorder) -> DenseFeatures
  - A NEW DenseFeatures; the input is not modified (it is frozen, so don't assign to it).
    times are the same, values have the same shape, names get "_s": "rms" -> "rms_s".
  - Every column is Savitzky-Golay filtered: savgol_filter(values, window, polyorder,
    axis=0, mode="interp").
  - window = the odd number nearest to window_beats * beat_period / frame_period frames,
    where frame_period = times[1] - times[0] (derived from the data, not a setting).
    At least the smallest odd number > polyorder: 3 for polyorder 2, 5 for polyorder 3.
    Example: 1 beat of 0.29 s at 0.02 s per frame = 14.5 frames -> 15.
  - A polynomial of degree <= polyorder passes through unchanged, edges included
    (that's what mode="interp" buys over the default edge handling of a plain filter).

  pool_to_beats(dense, beat_times) -> np.ndarray [B, F], float64
  - Row i is beat i, columns in dense.names order.
  - Beat i owns the frames whose time t satisfies beat_times[i] <= t < beat_times[i + 1].
    The last beat has no next beat, so it owns one previous interval:
    beat_times[-1] <= t < beat_times[-1] + (beat_times[-1] - beat_times[-2])
    (Decision: not the song end, or a long outro after the last beat would be averaged
    into it). Frames past the last frame simply don't exist, so the interval is clipped.
  - Frames before the first beat belong to no beat.
  - Each beat gets the MEAN of its frames, except "onset", which gets the MAX (a beat's
    attack is its strongest moment, and averaging would wash it out). onset is found by
    name, not by column position.
  - A beat that owns no frame (two beats closer than one frame, or a beat after the last
    frame) gets the values of the frame nearest to its beat time, so the result never
    contains NaN. With MIN_IBI = 0.25 s and ~23 ms frames this doesn't happen in practice.
  - Fewer than 2 beats: ValueError (there is no interval to measure).
  - Inputs are not modified.

  beat_dynamics(pooled, names, window, polyorder) -> dict[str, np.ndarray]
  - Keys, in this order: every name, then every name + "_s", then every name + "_d1",
    then every name + "_d2". So table.py can do pd.DataFrame({...}) and get the schema's
    column order. Each value is a 1-D float64 array of length B.
  - name       = pooled[:, j], the raw pooled column.
  - name_s     = savgol_filter(pooled, window, polyorder, deriv=0, axis=0, mode="interp")
  - name_d1/d2 = the same with deriv=1 / deriv=2 and delta=1.0.
  - Derivatives are PER BEAT, not per second (Decision: tempo-invariant; a build looks the
    same at 120 and 176 BPM). So no beat times go in: a ramp that rises 0.5 per beat has
    d1 = 0.5 everywhere, at any tempo.
  - window is used as given (the caller passes an odd number from config).
  - Fewer beats than window: ValueError. len(names) != number of columns: ValueError.
  - Inputs are not modified.
"""

import time

import numpy as np
import pytest
from scipy.signal import savgol_filter

from amv.music.dense import DenseFeatures
from amv.music.dynamics import beat_dynamics, pool_to_beats, smooth_dense

NAMES = ("rms", "onset", "flux", "low", "mid", "high")


def make_dense(values, frame_period=0.1, names=NAMES):
    """DenseFeatures with frames at 0, frame_period, 2 * frame_period, ..."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = np.repeat(values[:, None], len(names), axis=1)
    return DenseFeatures(times=np.arange(len(values)) * frame_period, values=values, names=names)


def frame_index_dense(n_frames=20, frame_period=0.1, names=NAMES):
    """Every column holds its frame index, so a mean or max over frames is easy to work out."""
    return make_dense(np.arange(n_frames, dtype=np.float64), frame_period, names)


def col(pooled, name, names=NAMES):
    return pooled[:, names.index(name)]


def reference_pool(dense, beat_times):
    """The spec of pool_to_beats as a slow, obvious loop."""
    times, values = dense.times, dense.values
    ends = np.append(beat_times[1:], 2 * beat_times[-1] - beat_times[-2])
    out = np.empty((len(beat_times), values.shape[1]))
    for i, (start, end) in enumerate(zip(beat_times, ends)):
        inside = (times >= start) & (times < end)
        frames = values[inside] if inside.any() else values[[np.argmin(np.abs(times - start))]]
        for j, name in enumerate(dense.names):
            out[i, j] = frames[:, j].max() if name == "onset" else frames[:, j].mean()
    return out


# =================================================================================
# smooth_dense
# =================================================================================


@pytest.fixture
def noisy_dense():
    rng = np.random.default_rng(0)
    return make_dense(rng.standard_normal((500, len(NAMES))), frame_period=0.02)


def test_smooth_returns_new_dense_features(noisy_dense):
    smoothed = smooth_dense(noisy_dense, 0.3, 1.0, 2)

    assert isinstance(smoothed, DenseFeatures)
    assert smoothed is not noisy_dense


def test_smooth_does_not_modify_the_input(noisy_dense):
    original = noisy_dense.values.copy()

    smooth_dense(noisy_dense, 0.3, 1.0, 2)

    np.testing.assert_array_equal(noisy_dense.values, original)


def test_smooth_keeps_times_and_shape_and_suffixes_names(noisy_dense):
    smoothed = smooth_dense(noisy_dense, 0.3, 1.0, 2)

    np.testing.assert_array_equal(smoothed.times, noisy_dense.times)
    assert smoothed.values.shape == noisy_dense.values.shape
    assert tuple(smoothed.names) == tuple(f"{name}_s" for name in NAMES)


def test_smooth_reduces_noise(noisy_dense):
    smoothed = smooth_dense(noisy_dense, 0.3, 1.0, 2)

    assert (smoothed.values.std(axis=0) < 0.5 * noisy_dense.values.std(axis=0)).all()


def test_smooth_leaves_a_parabola_unchanged_including_the_edges():
    """SG fits a degree-2 polynomial, so a parabola is reproduced exactly. With mode="interp"
    that holds at the first and last frames too."""
    i = np.arange(300, dtype=np.float64)
    dense = make_dense(0.001 * i**2 - 0.2 * i + 3, frame_period=0.02)

    smoothed = smooth_dense(dense, 0.3, 1.0, 2)

    np.testing.assert_allclose(smoothed.values, dense.values, atol=1e-8)


@pytest.mark.parametrize(
    ("frame_period", "beat_period", "window_beats", "polyorder", "window"),
    [
        (0.02, 0.305, 1.0, 2, 15),  # 15.25 frames
        (0.02, 0.29, 1.0, 2, 15),  # 14.5 frames: 15 is the nearest odd
        (0.02, 0.33, 1.0, 2, 17),  # 16.5 frames: 17 is the nearest odd
        (0.01, 0.305, 1.0, 2, 31),  # the frame period comes from times: 30.5 frames
        (0.02, 0.305, 2.0, 2, 31),  # window_beats scales the window: 30.5 frames
        (0.02, 0.01, 1.0, 2, 3),  # 0.5 frames: raised to the minimum for polyorder 2
        (0.02, 0.01, 1.0, 3, 5),  # 0.5 frames: raised to the minimum for polyorder 3
        (0.02, 0.07, 1.0, 3, 5),  # 3.5 frames: nearest odd is 3, but 3 is not > polyorder 3
    ],
)
def test_smooth_window_is_about_window_beats_in_frames(frame_period, beat_period, window_beats, polyorder, window):
    rng = np.random.default_rng(1)
    dense = make_dense(rng.standard_normal((500, len(NAMES))), frame_period=frame_period)

    smoothed = smooth_dense(dense, beat_period, window_beats, polyorder)

    expected = savgol_filter(dense.values, window, polyorder, axis=0, mode="interp")
    np.testing.assert_allclose(smoothed.values, expected, atol=1e-10)


# =================================================================================
# pool_to_beats
# =================================================================================
# Frames are at 0.0, 0.1, ..., 1.9 s and every value is its frame index, unless a test says
# otherwise. Beats sit between frames (x.x5) so no frame lands exactly on a boundary.


def test_pool_returns_one_float_row_per_beat():
    pooled = pool_to_beats(frame_index_dense(), np.array([0.25, 0.55, 1.05]))

    assert pooled.shape == (3, len(NAMES))
    assert pooled.dtype == np.float64


def test_pool_takes_the_mean_of_each_beat_interval():
    """Beat 0 owns frames 3-5, beat 1 frames 6-10, and the last beat one more interval of
    0.5 s: frames 11-15. Frames 0-2 come before the first beat and belong to nobody."""
    pooled = pool_to_beats(frame_index_dense(), np.array([0.25, 0.55, 1.05]))

    np.testing.assert_allclose(col(pooled, "rms"), [4.0, 8.0, 13.0])
    for name in ("flux", "low", "mid", "high"):
        np.testing.assert_allclose(col(pooled, name), [4.0, 8.0, 13.0], err_msg=name)


def test_pool_takes_the_max_for_onset():
    pooled = pool_to_beats(frame_index_dense(), np.array([0.25, 0.55, 1.05]))

    np.testing.assert_allclose(col(pooled, "onset"), [5.0, 10.0, 15.0])


def test_pool_finds_onset_by_name_not_by_position():
    names = ("rms", "flux", "low", "onset", "mid", "high")

    pooled = pool_to_beats(frame_index_dense(names=names), np.array([0.25, 0.55, 1.05]))

    np.testing.assert_allclose(col(pooled, "onset", names), [5.0, 10.0, 15.0])
    np.testing.assert_allclose(col(pooled, "rms", names), [4.0, 8.0, 13.0])


def test_pool_keeps_each_column_to_itself():
    """Column j holds frame index + 100 * j, so a mixed-up column gives the wrong hundreds."""
    values = np.arange(20, dtype=np.float64)[:, None] + 100 * np.arange(len(NAMES))
    dense = make_dense(values)

    pooled = pool_to_beats(dense, np.array([0.25, 0.55, 1.05]))

    for j, name in enumerate(NAMES):
        expected = np.array([5.0, 10.0, 15.0]) if name == "onset" else np.array([4.0, 8.0, 13.0])
        np.testing.assert_allclose(pooled[:, j], expected + 100 * j, err_msg=name)


def test_pool_interval_includes_its_start_and_excludes_its_end():
    """Frames every 0.125 s (exact in binary). Beats on frames 2 and 4: beat 0 owns frames
    2 and 3, not 4; the last beat owns frames 4 and 5, not 6."""
    dense = frame_index_dense(n_frames=10, frame_period=0.125)

    pooled = pool_to_beats(dense, np.array([0.25, 0.5]))

    np.testing.assert_allclose(col(pooled, "rms"), [2.5, 4.5])
    np.testing.assert_allclose(col(pooled, "onset"), [3.0, 5.0])


def test_pool_last_interval_is_clipped_to_the_frames_that_exist():
    """The last beat would own 1.85-3.45 s, but only the frame at 1.9 s exists."""
    pooled = pool_to_beats(frame_index_dense(), np.array([0.25, 1.85]))

    np.testing.assert_allclose(col(pooled, "rms"), [10.5, 19.0])  # beat 0: frames 3-18


def test_pool_beat_with_no_frame_takes_the_nearest_frame():
    """Beat 0 owns 0.31-0.35 s, which holds no frame: the nearest frame to 0.31 s is frame 3."""
    pooled = pool_to_beats(frame_index_dense(), np.array([0.31, 0.35, 0.75]))

    np.testing.assert_allclose(col(pooled, "rms"), [3.0, 5.5, 9.5])
    np.testing.assert_allclose(col(pooled, "onset"), [3.0, 7.0, 11.0])


def test_pool_beat_after_the_last_frame_takes_the_last_frame():
    pooled = pool_to_beats(frame_index_dense(), np.array([1.05, 2.5]))

    np.testing.assert_allclose(col(pooled, "rms"), [15.0, 19.0])  # beat 0: frames 11-19


def test_pool_matches_the_reference_on_random_input():
    """Random frames and an irregular beat grid, against the slow loop at the top of this file."""
    rng = np.random.default_rng(2)
    dense = make_dense(rng.standard_normal((3000, len(NAMES))), frame_period=512 / 22050)
    beat_times = np.cumsum(rng.uniform(0.01, 0.6, size=200)) + 0.3  # some intervals hold no frame

    pooled = pool_to_beats(dense, beat_times)

    np.testing.assert_allclose(pooled, reference_pool(dense, beat_times), atol=1e-10)


@pytest.mark.parametrize("beat_times", [np.array([]), np.array([0.5])])
def test_pool_needs_at_least_two_beats(beat_times):
    with pytest.raises(ValueError):
        pool_to_beats(frame_index_dense(), beat_times)


def test_pool_does_not_modify_inputs():
    dense = frame_index_dense()
    beat_times = np.array([0.25, 0.55, 1.05])
    original_values, original_beats = dense.values.copy(), beat_times.copy()

    pool_to_beats(dense, beat_times)

    np.testing.assert_array_equal(dense.values, original_values)
    np.testing.assert_array_equal(beat_times, original_beats)


# =================================================================================
# beat_dynamics
# =================================================================================


def pooled_from(column, n_features=len(NAMES)):
    """The same 1-D series in every column."""
    return np.repeat(np.asarray(column, dtype=np.float64)[:, None], n_features, axis=1)


def test_dynamics_keys_follow_the_schema_order():
    out = beat_dynamics(pooled_from(np.zeros(20)), NAMES, 5, 2)

    expected = [*NAMES, *(f"{n}_s" for n in NAMES), *(f"{n}_d1" for n in NAMES), *(f"{n}_d2" for n in NAMES)]
    assert list(out) == expected


def test_dynamics_values_are_float_arrays_of_length_b():
    out = beat_dynamics(pooled_from(np.arange(20)), NAMES, 5, 2)

    for key, value in out.items():
        assert value.shape == (20,), key
        assert value.dtype == np.float64, key


def test_dynamics_raw_columns_are_the_pooled_columns():
    rng = np.random.default_rng(3)
    pooled = rng.standard_normal((30, len(NAMES)))

    out = beat_dynamics(pooled, NAMES, 5, 2)

    for j, name in enumerate(NAMES):
        np.testing.assert_array_equal(out[name], pooled[:, j], err_msg=name)


def test_dynamics_of_a_constant_has_zero_derivatives():
    out = beat_dynamics(pooled_from(np.full(20, 2.5)), NAMES, 5, 2)

    np.testing.assert_allclose(out["rms_s"], 2.5, atol=1e-10)
    np.testing.assert_allclose(out["rms_d1"], 0.0, atol=1e-10)
    np.testing.assert_allclose(out["rms_d2"], 0.0, atol=1e-10)


def test_dynamics_of_a_ramp_has_d1_in_units_per_beat():
    """Rising 0.5 per beat: d1 = 0.5 at every beat, edges included. No times go in, so this is
    per beat whatever the tempo."""
    ramp = 1.0 + 0.5 * np.arange(20)

    out = beat_dynamics(pooled_from(ramp), NAMES, 5, 2)

    np.testing.assert_allclose(out["rms_s"], ramp, atol=1e-10)
    np.testing.assert_allclose(out["rms_d1"], 0.5, atol=1e-10)
    np.testing.assert_allclose(out["rms_d2"], 0.0, atol=1e-10)


def test_dynamics_of_a_parabola_has_constant_d2():
    """x = 0.1 * i^2: d1 = 0.2 * i and d2 = 0.2, the textbook derivatives."""
    i = np.arange(20, dtype=np.float64)

    out = beat_dynamics(pooled_from(0.1 * i**2), NAMES, 5, 2)

    np.testing.assert_allclose(out["rms_d1"], 0.2 * i, atol=1e-10)
    np.testing.assert_allclose(out["rms_d2"], 0.2, atol=1e-10)


def test_dynamics_s_is_smoother_than_the_raw_column():
    rng = np.random.default_rng(4)
    pooled = rng.standard_normal((200, len(NAMES)))

    out = beat_dynamics(pooled, NAMES, 5, 2)

    for name in NAMES:
        assert out[f"{name}_s"].std() < 0.8 * out[name].std(), name


def test_dynamics_keeps_each_column_to_itself():
    """Each column a different ramp: column j rises j per beat."""
    pooled = np.arange(20, dtype=np.float64)[:, None] * np.arange(len(NAMES))

    out = beat_dynamics(pooled, NAMES, 5, 2)

    for j, name in enumerate(NAMES):
        np.testing.assert_allclose(out[f"{name}_d1"], float(j), atol=1e-10, err_msg=name)


@pytest.mark.parametrize(("window", "polyorder"), [(5, 2), (7, 2), (7, 3)])
def test_dynamics_uses_the_window_and_polyorder_given(window, polyorder):
    rng = np.random.default_rng(5)
    pooled = rng.standard_normal((50, len(NAMES)))

    out = beat_dynamics(pooled, NAMES, window, polyorder)

    for deriv, suffix in [(0, "_s"), (1, "_d1"), (2, "_d2")]:
        expected = savgol_filter(pooled, window, polyorder, deriv=deriv, delta=1.0, axis=0, mode="interp")
        for j, name in enumerate(NAMES):
            np.testing.assert_allclose(out[name + suffix], expected[:, j], atol=1e-10, err_msg=name + suffix)


def test_dynamics_accepts_exactly_window_beats():
    out = beat_dynamics(pooled_from(np.arange(5)), NAMES, 5, 2)

    assert out["rms_s"].shape == (5,)


def test_dynamics_needs_at_least_window_beats():
    with pytest.raises(ValueError):
        beat_dynamics(pooled_from(np.arange(4)), NAMES, 5, 2)


def test_dynamics_needs_one_name_per_column():
    with pytest.raises(ValueError):
        beat_dynamics(pooled_from(np.arange(20)), NAMES[:-1], 5, 2)


def test_dynamics_does_not_modify_the_input():
    pooled = np.random.default_rng(6).standard_normal((20, len(NAMES)))
    original = pooled.copy()

    beat_dynamics(pooled, NAMES, 5, 2)

    np.testing.assert_array_equal(pooled, original)


# =================================================================================
# Speed, config
# =================================================================================


def test_the_chain_is_fast_on_a_song_sized_input():
    """A 3-min song: ~7750 frames at 43 Hz and ~530 beats at 176 BPM. All of 3.4 should be a
    rounding error next to beat_this (~3 s). Best of 3, so a busy machine doesn't fail it."""
    rng = np.random.default_rng(7)
    dense = make_dense(rng.standard_normal((7750, len(NAMES))), frame_period=512 / 22050)
    beat_times = 0.3 + np.arange(530) * (60 / 176)

    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        smooth_dense(dense, 60 / 176, 1.0, 2)
        beat_dynamics(pool_to_beats(dense, beat_times), NAMES, 5, 2)
        best = min(best, time.perf_counter() - start)

    assert best < 0.05, f"{best * 1000:.1f} ms"


def test_config_has_the_planned_settings():
    import amv.config as config

    assert config.SG_POLYORDER == 2
    assert config.DENSE_SG_WINDOW_BEATS == 1.0
    assert config.BEAT_SG_WINDOW == 5


# =================================================================================
# Real song (integration)
# =================================================================================


@pytest.mark.integration
def test_bluelock_chain_gives_one_finite_row_per_beat(beat_model, bluelock_wav):
    import amv.config as config
    from amv.music.audio import load_audio
    from amv.music.beats import clean_beats, detect_beats
    from amv.music.dense import compute_dense_features

    y, sr = load_audio(bluelock_wav, config.AUDIO_SAMPLE_RATE)
    beats, _ = detect_beats(y, sr, beat_model)
    beats = clean_beats(beats, config.MIN_IBI)
    dense = compute_dense_features(
        y, sr, config.HOP_LENGTH, config.N_FFT, config.N_MELS, config.LOW_BAND_HZ, config.HIGH_BAND_HZ
    )

    smoothed = smooth_dense(dense, float(np.median(np.diff(beats))), config.DENSE_SG_WINDOW_BEATS, config.SG_POLYORDER)
    out = beat_dynamics(pool_to_beats(dense, beats), dense.names, config.BEAT_SG_WINDOW, config.SG_POLYORDER)

    assert smoothed.values.shape == dense.values.shape
    assert np.isfinite(smoothed.values).all()
    assert len(out) == 4 * len(dense.names)
    for key, value in out.items():
        assert value.shape == beats.shape, key
        assert np.isfinite(value).all(), key
