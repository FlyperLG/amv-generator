"""Tests for amv.music.beats (docs/02-music-features, task 3.2).

clean_beats and build_beat_grid are pure numpy and are tested with hand-made beat
times. detect_beats is tested with a fake model, and against the real beat_this
model on synthetic drum loops and on the Blue Lock song (integration).

Why the rules below look the way they do: on the Blue Lock song (~176 BPM), beat_this
switches to half tempo in the 60-90 s breakdown (0.68 s intervals), and its
downbeats give bars of 1 to 7 beats. The doc's "fill gaps > 1.5 s" rule catches
neither, so the rules are relative to the song's own median interval instead.

SPECIFICATION
  Settings are required arguments (Decision: beats.py does not read amv.config; the
  entry point reads the values from config and passes them in).

  load_beat_model(checkpoint) -> beat_this.inference.Audio2Beats
  - CPU model. Loading takes seconds, so the result is cached: a second call with the
    same checkpoint returns the SAME object (functools.cache).

  detect_beats(y, sr, model) -> tuple[np.ndarray, np.ndarray]
  - Calls model(y, sr) exactly once and returns (beats, downbeats) in seconds, both
    1-D float64 arrays. No cleaning here; that is clean_beats' job.

  clean_beats(beats, min_ibi) -> np.ndarray          (config: MIN_IBI = 0.25 s)
  - beats: sorted times in seconds, any array-like. Returns a new float64 array,
    strictly increasing; the input is not modified.
  - Step 1, drop: walk through the beats in order and keep a beat only if it is at
    least min_ibi after the last KEPT beat.
  - Step 2, reference: ref = median interval of the kept beats.
  - Step 3, fill: every interval with n = round(interval / ref) >= 2 is split into n
    equal parts (n - 1 beats inserted). This fills missed beats and undoes
    half-tempo sections. Intervals below 1.5 * ref stay as they are. n is capped at
    floor(interval / min_ibi), so a fill never creates beats closer than min_ibi.
  - Never extrapolates: the first and last kept beat stay the ends.
  - Warns (logger amv.music.beats) if over 5% of the intervals were split into parts more
    than 10% away from ref: the song may really change tempo. The result is the same
    either way. A change of exactly 2x looks like half tempo and goes undetected.
  - Fewer than 2 beats: returned unchanged (as a float64 copy).
  - Replaces the doc's max_ibi rule (Decision: a fixed 1.5 s misses half-tempo
    sections, see above).

  BeatGrid: frozen dataclass with four 1-D arrays of the same length
    times        float64  beat times (s), a copy of the input beats
    is_downbeat  bool     exactly where beat_in_bar == 0
    beat_in_bar  int      0 .. beats_per_bar - 1
    bar_idx      int      the number of downbeats so far, minus 1. Beats before the first
                          downbeat (a pickup) get -1; the bar index increases by 1 at
                          every downbeat and never decreases.

  build_beat_grid(beats, downbeats, beats_per_bar, tol, window_bars) -> BeatGrid
    (config: 4 beats per bar, DOWNBEAT_TOL = 0.07 s, DOWNBEAT_WINDOW_BARS = 8)
  - The tracker's downbeats are VOTES, not the answer (Decision: they are too noisy
    to count "beats since the last downbeat", see above):
      1. Each downbeat is matched to its nearest beat, if within tol seconds;
         otherwise it is ignored. A beat counts at most once, however many
         downbeats match it.
      2. For each beat i, look at the matched beats j with |j - i| <= window_bars *
         beats_per_bar (indices, not seconds) and count them per residue j % beats_per_bar.
         phase_i = the residue with the most votes. On a tie (also: no votes at all)
         phase_i = phase_{i-1}; for beat 0 the smallest tied residue.
      3. beat_in_bar_i = (i - phase_i) % beats_per_bar.
  - Consequences: the grid is extended backwards into a pickup (two beats before the
    first downbeat get beat_in_bar 2, 3 in 4/4); a stretch without downbeats
    continues the bar grid of the beats before it; no downbeats at all gives
    beat 0 as the first downbeat; where the phase changes, one bar is shorter or
    longer than beats_per_bar.
  - Empty beats give an empty grid.
"""

import dataclasses
import logging
import time

import numpy as np
import pytest

from amv.config import AUDIO_SAMPLE_RATE, BEAT_THIS_CHECKPOINT
from amv.music.beats import BeatGrid, build_beat_grid, clean_beats, detect_beats, load_beat_model

IBI = 0.5  # 120 BPM, used for hand-made beat grids


def grid_times(n, ibi=IBI, start=0.0):
    return start + np.arange(n) * ibi


# Fixed here, not read from config: these tests must not shift when the config is tuned.
def clean(beats, min_ibi=0.25):
    return clean_beats(beats, min_ibi)


def make_grid(beats, downbeats, beats_per_bar=4, tol=0.07, window_bars=8):
    return build_beat_grid(beats, downbeats, beats_per_bar, tol, window_bars)


# --- Settings are required ------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: load_beat_model(),
        lambda: clean_beats([0.0, 0.5]),
        lambda: build_beat_grid(np.array([0.0]), np.array([0.0])),
    ],
    ids=["load_beat_model", "clean_beats", "build_beat_grid"],
)
def test_settings_have_no_defaults(call):
    with pytest.raises(TypeError):
        call()


# --- detect_beats (fake model) --------------------------------------------------


class FakeModel:
    """Stands in for Audio2Beats: records calls and returns fixed lists."""

    def __init__(self, beats, downbeats):
        self.result = (beats, downbeats)
        self.calls = []

    def __call__(self, y, sr):
        self.calls.append((y, sr))
        return self.result


def test_detect_calls_the_model_once_with_signal_and_rate():
    y = np.zeros(22050, np.float32)
    model = FakeModel([0.5, 1.0], [0.5])

    detect_beats(y, 22050, model)

    assert len(model.calls) == 1
    called_y, called_sr = model.calls[0]
    assert called_y is y and called_sr == 22050


def test_detect_returns_float64_arrays():
    model = FakeModel([0.5, 1.0, 1.5], [0.5])

    beats, downbeats = detect_beats(np.zeros(100, np.float32), 22050, model)

    for arr, expected in ((beats, [0.5, 1.0, 1.5]), (downbeats, [0.5])):
        assert isinstance(arr, np.ndarray) and arr.dtype == np.float64 and arr.ndim == 1
        np.testing.assert_array_equal(arr, expected)


# --- clean_beats: drop ----------------------------------------------------------


def test_clean_regular_grid_is_unchanged():
    beats = grid_times(16)

    np.testing.assert_allclose(clean(beats), beats)


def test_clean_drops_beat_closer_than_min_ibi():
    beats = [0.0, 0.5, 0.74, 1.0, 1.5, 2.0]  # 0.74 is 0.24 s after 0.5

    np.testing.assert_allclose(clean(beats), [0.0, 0.5, 1.0, 1.5, 2.0])


def test_clean_drops_against_the_last_kept_beat():
    beats = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]  # 0.1 and 0.2 are both too close to 0.0

    np.testing.assert_allclose(clean(beats), [0.0, 0.5, 1.0, 1.5, 2.0])


def test_clean_min_ibi_is_a_parameter():
    beats = [0.0, 0.5, 0.74, 1.0, 1.5, 2.0]

    np.testing.assert_allclose(clean(beats, min_ibi=0.1), beats)


# --- clean_beats: fill ----------------------------------------------------------


def test_clean_fills_one_missed_beat():
    full = grid_times(12)
    missing = np.delete(full, 4)

    np.testing.assert_allclose(clean(missing), full)


def test_clean_fills_several_missed_beats_evenly():
    full = grid_times(12)
    missing = np.delete(full, [4, 5, 6])

    np.testing.assert_allclose(clean(missing), full)


def test_clean_undoes_a_half_tempo_section():
    ibi = 0.34  # Blue Lock: ~176 BPM, breakdown tracked at 0.68 s
    fast1 = grid_times(40, ibi)
    slow = fast1[-1] + np.arange(1, 11) * 2 * ibi
    fast2 = slow[-1] + np.arange(1, 21) * ibi
    beats = np.concatenate([fast1, slow, fast2])

    cleaned = clean(beats)

    assert len(cleaned) == len(beats) + 10
    np.testing.assert_allclose(np.diff(cleaned), ibi, atol=1e-9)


def test_clean_does_not_split_intervals_below_one_and_a_half_ref():
    beats = [0.0, 0.5, 1.0, 1.7, 2.2, 2.7, 3.2]  # 0.7 s = 1.4 x the 0.5 s median

    np.testing.assert_allclose(clean(beats), beats)


def test_clean_fill_does_not_split_below_min_ibi():
    # ref 0.26: the 0.40 s gap rounds to n = 2, but two 0.20 s parts are below min_ibi.
    beats = [0.0, 0.26, 0.52, 0.78, 1.04, 1.44, 1.70, 1.96, 2.22]

    np.testing.assert_allclose(clean(beats, min_ibi=0.25), beats)


def test_clean_fill_caps_the_parts_at_min_ibi():
    # ref 0.26: the 0.96 s gap rounds to n = 4 (0.24 s parts), capped to 3 parts of 0.32 s.
    beats = [0.0, 0.26, 0.52, 0.78, 1.04, 2.0, 2.26, 2.52, 2.78, 3.04]

    cleaned = clean(beats, min_ibi=0.25)

    np.testing.assert_allclose(cleaned[4:8], [1.04, 1.36, 1.68, 2.0])
    assert len(cleaned) == len(beats) + 2


def test_clean_reference_is_taken_after_dropping():
    # Raw median 0.175 (would split every 0.5 s gap into 3); after the drop it is 0.5.
    beats = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 1.5, 2.0]

    np.testing.assert_allclose(clean(beats), [0.0, 0.5, 1.0, 1.5, 2.0])


def test_clean_one_beat_left_after_dropping(recwarn):
    cleaned = clean([0.0, 0.01])

    np.testing.assert_array_equal(cleaned, [0.0])
    assert len(recwarn) == 0  # no median of an empty list


def test_clean_single_int_beat_is_float64():
    assert clean([1]).dtype == np.float64


def test_clean_does_not_extrapolate():
    beats = np.delete(grid_times(12, start=3.0), 5)

    cleaned = clean(beats)

    assert cleaned[0] == beats[0] and cleaned[-1] == beats[-1]


# --- clean_beats: tempo change warning ------------------------------------------


def test_clean_warns_about_a_tempo_change(caplog):
    fast = grid_times(60, 0.353)
    slow = fast[-1] + np.arange(1, 41) * 0.6  # 1.7x slower: split into 0.3 s parts, 15% off

    with caplog.at_level(logging.WARNING, logger="amv.music.beats"):
        clean(np.concatenate([fast, slow]))

    assert "change tempo" in caplog.text


def test_clean_does_not_warn_about_half_tempo_with_jitter(caplog):
    jitter = np.tile([-0.02, 0.0, 0.02], 7)[:20]  # beat_this has 20 ms frames
    intervals = np.concatenate([np.full(60, 0.34), np.full(20, 0.68) + jitter])

    with caplog.at_level(logging.WARNING, logger="amv.music.beats"):
        clean(np.cumsum(intervals))

    assert caplog.text == ""


def test_clean_does_not_warn_about_a_few_late_beats(caplog):
    # 2 of 40 intervals (exactly 5%, not over it) split into parts 25% off the median
    intervals = np.full(40, 0.34)
    intervals[[10, 30]] = 0.51
    beats = np.concatenate([[0.0], np.cumsum(intervals)])

    with caplog.at_level(logging.WARNING, logger="amv.music.beats"):
        clean(beats)

    assert caplog.text == ""


def test_clean_warning_does_not_change_the_result(caplog):
    fast = grid_times(60, 0.353)
    slow = fast[-1] + np.arange(1, 41) * 0.6

    with caplog.at_level(logging.WARNING, logger="amv.music.beats"):
        cleaned = clean(np.concatenate([fast, slow]))

    assert len(cleaned) == 100 + 40
    np.testing.assert_allclose(np.diff(cleaned)[-3:], 0.3)


# --- clean_beats: output --------------------------------------------------------


def test_clean_returns_strictly_increasing_float64():
    cleaned = clean([0, 1, 2, 3, 4, 6])  # ints in, one gap

    assert cleaned.dtype == np.float64
    assert np.all(np.diff(cleaned) > 0)


def test_clean_does_not_modify_input():
    beats = np.delete(grid_times(12), 4)
    before = beats.copy()

    clean(beats)

    np.testing.assert_array_equal(beats, before)


@pytest.mark.parametrize("beats", [[], [1.0]])
def test_clean_fewer_than_two_beats_are_returned_unchanged(beats):
    cleaned = clean(beats)

    assert cleaned.dtype == np.float64
    np.testing.assert_array_equal(cleaned, beats)


# --- build_beat_grid: structure -------------------------------------------------


def test_grid_is_a_frozen_dataclass_of_equal_length_arrays():
    beats = grid_times(10)

    grid = make_grid(beats, beats[::4])

    assert isinstance(grid, BeatGrid)
    assert grid.times.dtype == np.float64
    assert grid.is_downbeat.dtype == np.bool_
    assert np.issubdtype(grid.beat_in_bar.dtype, np.integer)
    assert np.issubdtype(grid.bar_idx.dtype, np.integer)
    assert {len(grid.times), len(grid.is_downbeat), len(grid.beat_in_bar), len(grid.bar_idx)} == {10}
    np.testing.assert_array_equal(grid.times, beats)
    with pytest.raises(dataclasses.FrozenInstanceError):
        grid.times = beats  # type: ignore[misc]


def test_grid_regular_four_four():
    beats = grid_times(16)

    grid = make_grid(beats, beats[::4])

    np.testing.assert_array_equal(grid.beat_in_bar, np.arange(16) % 4)
    np.testing.assert_array_equal(grid.bar_idx, np.arange(16) // 4)
    np.testing.assert_array_equal(grid.is_downbeat, np.arange(16) % 4 == 0)


def test_grid_pickup_beats_continue_the_grid_backwards():
    beats = grid_times(14)

    grid = make_grid(beats, beats[2::4])  # first downbeat is the third beat

    np.testing.assert_array_equal(grid.beat_in_bar[:7], [2, 3, 0, 1, 2, 3, 0])
    np.testing.assert_array_equal(grid.bar_idx[:7], [-1, -1, 0, 0, 0, 0, 1])


def test_grid_without_downbeats_starts_a_bar_on_the_first_beat():
    grid = make_grid(grid_times(10), np.array([]))

    np.testing.assert_array_equal(grid.beat_in_bar, np.arange(10) % 4)
    np.testing.assert_array_equal(grid.bar_idx, np.arange(10) // 4)


def test_grid_three_four():
    beats = grid_times(12)

    grid = make_grid(beats, beats[1::3], beats_per_bar=3)

    np.testing.assert_array_equal(grid.beat_in_bar, (np.arange(12) - 1) % 3)
    np.testing.assert_array_equal(grid.bar_idx, [-1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])


def test_grid_empty():
    grid = make_grid(np.array([]), np.array([]))

    assert len(grid.times) == len(grid.is_downbeat) == len(grid.beat_in_bar) == len(grid.bar_idx) == 0


def test_grid_empty_beats_with_downbeats_is_an_empty_float64_grid():
    grid = make_grid(np.array([]), np.array([1.0]))

    assert len(grid.times) == 0
    assert grid.times.dtype == np.float64


def test_grid_single_beat():
    grid = make_grid(np.array([1.0]), np.array([1.0]))

    assert grid.beat_in_bar.tolist() == [0]
    assert grid.bar_idx.tolist() == [0]
    assert grid.is_downbeat.tolist() == [True]


# --- build_beat_grid: voting ----------------------------------------------------


def test_grid_majority_wins_over_missing_and_wrong_downbeats():
    beats = grid_times(32)
    down_idx = [0, 4, 8, 16, 20, 24, 28, 9]  # 12 missing, 9 is wrong

    grid = make_grid(beats, beats[down_idx])

    np.testing.assert_array_equal(grid.beat_in_bar, np.arange(32) % 4)


def test_grid_downbeats_are_matched_within_tolerance_only():
    beats = grid_times(24)
    near = beats[[0, 4]] + [0.03, -0.03]  # 2 votes for residue 0
    far = beats[[2, 6, 10, 14, 18, 22]] + 0.2  # 6 would-be votes for residue 2, too far away

    grid = make_grid(beats, np.sort(np.concatenate([near, far])), tol=0.07)

    np.testing.assert_array_equal(grid.beat_in_bar, np.arange(24) % 4)


def test_grid_tol_is_a_parameter():
    beats = grid_times(24)
    near = beats[[0, 4]] + [0.03, -0.03]  # 2 votes for residue 0
    far = beats[[2, 6, 10, 14, 18, 22]] + 0.2  # 6 votes for residue 2 once tol allows them

    grid = make_grid(beats, np.sort(np.concatenate([near, far])), tol=0.25)

    np.testing.assert_array_equal(grid.beat_in_bar, (np.arange(24) - 2) % 4)


def test_grid_a_beat_votes_only_once():
    beats = grid_times(16)
    downbeats = np.sort(np.concatenate([beats[[0, 4, 8]], np.repeat(beats[1], 5)]))

    grid = make_grid(beats, downbeats)

    np.testing.assert_array_equal(grid.beat_in_bar, np.arange(16) % 4)


def test_grid_follows_a_phase_change():
    beats = grid_times(200)
    down_idx = np.concatenate([np.arange(0, 100, 4), np.arange(102, 200, 4)])

    grid = make_grid(beats, beats[down_idx], window_bars=8)  # window = +-32 beats

    i = np.arange(200)
    np.testing.assert_array_equal(grid.beat_in_bar[:68], i[:68] % 4)
    np.testing.assert_array_equal(grid.beat_in_bar[134:], (i[134:] - 2) % 4)


def test_grid_window_bars_is_a_parameter():
    beats = grid_times(200)
    down_idx = np.concatenate([np.arange(0, 100, 4), np.arange(102, 200, 4)])

    grid = make_grid(beats, beats[down_idx], window_bars=1)  # window = +-4 beats

    i = np.arange(200)
    np.testing.assert_array_equal(grid.beat_in_bar[:101], i[:101] % 4)
    np.testing.assert_array_equal(grid.beat_in_bar[101:], (i[101:] - 2) % 4)


def test_grid_stretch_without_downbeats_keeps_the_previous_phase():
    beats = grid_times(200)
    down_idx = np.arange(1, 40, 4)  # downbeats only in the first 40 beats

    grid = make_grid(beats, beats[down_idx])

    np.testing.assert_array_equal(grid.beat_in_bar, (np.arange(200) - 1) % 4)


# --- build_beat_grid: invariants ------------------------------------------------


def test_grid_invariants_on_noisy_downbeats():
    rng = np.random.default_rng(1)
    beats = np.cumsum(rng.uniform(0.3, 0.4, 300))
    downbeats = np.sort(rng.choice(beats, 60, replace=False))

    grid = make_grid(beats, downbeats)

    assert grid.beat_in_bar.min() >= 0 and grid.beat_in_bar.max() <= 3
    np.testing.assert_array_equal(grid.is_downbeat, grid.beat_in_bar == 0)
    steps = np.diff(grid.bar_idx)
    assert set(steps.tolist()) <= {0, 1}
    np.testing.assert_array_equal(steps == 1, grid.is_downbeat[1:])
    assert grid.bar_idx[0] == (0 if grid.is_downbeat[0] else -1)


def test_grid_does_not_modify_inputs():
    beats = grid_times(16)
    downbeats = beats[::4].copy()
    before = (beats.copy(), downbeats.copy())

    make_grid(beats, downbeats)

    np.testing.assert_array_equal(beats, before[0])
    np.testing.assert_array_equal(downbeats, before[1])


# --- Real model (integration) ---------------------------------------------------


@pytest.mark.integration
def test_load_beat_model_is_cached(beat_model):
    assert callable(beat_model)
    assert load_beat_model(BEAT_THIS_CHECKPOINT) is beat_model


@pytest.mark.integration
@pytest.mark.parametrize("bpm", [120, 150])
def test_detect_finds_the_tempo_of_a_drum_loop(beat_model, drum_loop, bpm):
    y, expected = drum_loop(bpm=bpm, bars=8)

    beats, downbeats = detect_beats(y, 22050, beat_model)

    assert len(beats) >= len(expected)
    assert np.median(np.diff(beats)) == pytest.approx(60 / bpm, abs=0.021)  # beat_this: 20 ms frames
    # beat_this puts every downbeat on a beat
    assert np.all(np.min(np.abs(downbeats[:, None] - beats[None, :]), axis=1) < 1e-6)


@pytest.fixture(scope="module")
def bluelock_beats(beat_model, bluelock_wav):
    """(raw beats, raw downbeats, seconds detect_beats took) for the Blue Lock song."""
    from amv.music.audio import load_audio

    y, sr = load_audio(bluelock_wav, AUDIO_SAMPLE_RATE)
    start = time.perf_counter()
    beats, downbeats = detect_beats(y, sr, beat_model)
    return beats, downbeats, time.perf_counter() - start


@pytest.mark.integration
def test_bluelock_detection_is_fast_enough(bluelock_beats):
    _, _, seconds = bluelock_beats

    assert seconds < 15  # doc §4: a 3-min song in < 15 s; this one is 2:13 (measured ~3 s)


@pytest.mark.integration
def test_bluelock_tempo_is_steady_after_cleaning(bluelock_beats):
    beats, _, _ = bluelock_beats

    cleaned = clean(beats)

    ibi = np.diff(cleaned)
    assert 60 / np.median(ibi) == pytest.approx(176, abs=3)
    assert ibi.min() >= 0.25
    assert ibi.max() < 1.5 * np.median(ibi)  # the half-tempo breakdown was filled


@pytest.mark.integration
def test_bluelock_has_no_tempo_change_warning(bluelock_beats, caplog):
    beats, _, _ = bluelock_beats

    with caplog.at_level(logging.WARNING, logger="amv.music.beats"):
        clean(beats)

    assert caplog.text == ""  # its half-tempo breakdown is not a tempo change (measured 2.1%)


@pytest.mark.integration
def test_bluelock_bars_are_four_beats(bluelock_beats):
    beats, downbeats, _ = bluelock_beats

    grid = make_grid(clean(beats), downbeats)

    bar_lengths = np.diff(np.flatnonzero(grid.is_downbeat))
    assert np.mean(bar_lengths == 4) >= 0.9  # measured 89 of 90 with window_bars=8
