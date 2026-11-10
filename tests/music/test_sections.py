"""Tests for amv.music.sections (docs/02-music-features, task 3.5).

Section finding is split into small steps, each tested on hand-made input whose answer can be
worked out on paper: bar features made of blocks, a similarity matrix of +1/-1 blocks, a
beat grid with a known number of bars. The private helpers are tested directly, because a
wrong kernel or an off-by-one in the novelty is very hard to find from the final labels. One
integration test runs the whole chain on the Blue Lock song.

SPECIFICATION
  Settings are required arguments (Decision: sections.py does not read amv.config; the entry
  point reads the values from config and passes them in).
  (config: SECTION_KERNEL_BARS = 8, MIN_SECTION_BARS = 4, NOVELTY_PROMINENCE = 0.1,
           DROP_LEVEL = 0.75, BUILD_RISE = 0.5)
  SECTION_LABELS = ("intro", "build", "drop", "break", "outro") is a fixed vocabulary, not config.

  The chain (the caller does the pooling, with functions that already exist):
      timbre  = compute_timbre_features(y, ...)                       [T, D] raw
      bars    = pool_to_beats(timbre, grid.times[grid.is_downbeat])   [n_bars, D]  bar k = bar_idx k
      novelty = section_novelty(bars, kernel_bars)                    [n_bars]
      bounds  = find_section_boundaries(novelty, min_section_bars, prominence)
      labels  = label_sections(grid, dyn["rms_s"], dyn["rms_d1"], bounds, drop_level, build_rise)
  Boundaries are bar indices, so they are snapped to downbeats by construction.
  (Decision, task 3.7: the novelty is public and computed once, so the diagnostic plot can draw
  the same curve the boundaries were picked from. It is not cached; the plot recomputes it.)

  _self_similarity(bar_features) -> [n_bars, n_bars]
  - z-score each column over the bars: (x - mean) / std. A column with std 0 becomes all
    zeros (Decision: it says nothing about where sections change). So the scale of a column
    and a constant column have no effect.
  - Then the cosine similarity of every pair of rows. A row that is all zeros has similarity
    0 with every row, itself included. Never NaN, no numpy warning.
  - Symmetric, values in [-1, 1], 1 on the diagonal (for non-zero rows).

  _checkerboard_kernel(kernel_bars) -> [k, k], k = kernel_bars (even)
  - Sign +1 where row and column are on the same side of the middle (both < k/2 or both
    >= k/2), -1 where they are on opposite sides.
  - Times a Gaussian taper exp(-(di^2 + dj^2) / (2 sigma^2)), di = i - (k - 1) / 2, likewise
    dj, sigma = k / 4 (Decision: part of the kernel's shape, not a setting). Bars close to
    the middle count most.
  - Scaled so the sum of |K| is 1 (Decision, replacing "novelty scaled to max 1": then the
    novelty is in similarity units, a perfect change gives 1, and the prominence means "how
    much more alike the bars are within each side than across", the same in every song.
    Measured on the 7 AMV songs: median novelty 0.08-0.13, 90th percentile 0.22-0.42.)
  - The sum of K is then 0: a matrix that is the same everywhere gives novelty 0.

  _novelty(ssm, kernel_bars) -> [n_bars]
  - novelty[b] = sum(K * window), where window is the k x k block of ssm over the bars
    b - k/2 .. b + k/2 - 1, zero-padded outside the song. So the top-left quadrant compares
    the k/2 bars before b with each other, the bottom-right the k/2 bars from b on, and the
    peak at bar b means "the section changes between bar b - 1 and bar b".
  - Not rescaled. A perfect change (+1 within each side, -1 across) gives exactly 1.

  section_novelty(bar_features, kernel_bars) -> np.ndarray [n_bars], float
  - _novelty(_self_similarity(bar_features), kernel_bars).
  - Odd kernel_bars: ValueError. Fewer than 2 bars: all zeros, shape [n_bars], no numpy
    warning (Decision: a 0- or 1-bar song has no change anywhere). Input not modified.
  - Known limitation: the z-scoring stretches a song's variation to unit size, so a song
    with no real sections still gets novelty peaks at its biggest wobbles. Real songs have
    sections, and imperfect labels are acceptable here (step 13 lets the user fix them).

  find_section_boundaries(novelty, min_section_bars, prominence) -> np.ndarray
  - Sorted int bar indices where a section starts. Always starts with 0. n_bars = len(novelty).
  - The other boundaries are the peaks found by
    scipy.signal.find_peaks(novelty, prominence=prominence, distance=min_section_bars),
    minus any peak closer than min_section_bars to the start or the end
    (min_section_bars <= peak <= n_bars - min_section_bars). So no section is shorter than
    min_section_bars bars.
  - Fewer than 2 bars: [0]. Input not modified.

  label_sections(grid, rms_s, rms_d1, boundaries, drop_level, build_rise) -> np.ndarray [B]
  - One label per beat, every one in SECTION_LABELS.
  - Segment s is the bars boundaries[s] <= bar_idx < boundaries[s + 1]; the last one runs to
    the end. Pickup beats (bar_idx -1) belong to segment 0. Boundaries past the last bar
    are ignored, so every segment has beats.
  - Per segment: energy = the mean of rms_s over its beats, rise = the SUM of rms_d1 over
    its beats (about how much rms_s gained across the segment, in rms_s units, which are
    already the song's IQR units).
  - rms_s and rms_d1 must have one value per beat: ValueError otherwise. Inputs not modified.

  _segment_labels(energy, rise, drop_level, build_rise) -> list[str], one per segment
  - level = (energy - min) / (max - min) over the song's segments, 0 for all if they are
    equal (Decision: rank segments within the song. Measured on the 7 AMV songs, "0.5 IQR
    above the median" almost never fires, because in a loud song the drops ARE the median.)
  - Rules, first match wins:
      1. level >= drop_level                                   -> "drop"
      2. rise >= build_rise and the NEXT segment is a drop     -> "build"
         (Decision: a rising segment that leads nowhere, e.g. a fade-in, is not a build)
      3. first segment -> "intro", last segment -> "outro", any other -> "break"
    So a single segment is "intro", and a loud intro or outro is a "drop".
"""

import time
import warnings

import numpy as np
import pytest

from amv.music.beats import BeatGrid
from amv.music.sections import (
    SECTION_LABELS,
    _checkerboard_kernel,
    _novelty,
    _segment_labels,
    _self_similarity,
    find_section_boundaries,
    label_sections,
    section_novelty,
)

K = 8
MIN_BARS = 4
PROMINENCE = 0.1
DROP_LEVEL = 0.75
BUILD_RISE = 0.5
BEATS_PER_BAR = 4


def boundaries(bar_features, kernel_bars=K, min_section_bars=MIN_BARS, prominence=PROMINENCE):
    """The whole chain from bar features to boundaries."""
    return find_section_boundaries(section_novelty(bar_features, kernel_bars), min_section_bars, prominence)


def spikes(n_bars, **peaks):
    """A flat zero novelty curve of n_bars values with a spike of the given height at each bar
    (keys are "b<bar>", e.g. spikes(20, b8=0.5))."""
    novelty = np.zeros(n_bars)
    for key, height in peaks.items():
        novelty[int(key[1:])] = height
    return novelty


def blocks(*lengths, dim=24, noise=0.1, seed=0):
    """Bar features made of blocks: every bar of block i is the same random vector plus noise."""
    rng = np.random.default_rng(seed)
    rows = [rng.standard_normal(dim) + noise * rng.standard_normal((n, dim)) for n in lengths]
    return np.vstack(rows)


def block_ssm(*lengths):
    """+1 within a block, -1 across blocks."""
    ids = np.repeat(np.arange(len(lengths)), lengths)
    return np.where(ids[:, None] == ids[None, :], 1.0, -1.0)


def make_grid(n_bars, pickup=0):
    """A steady 120 BPM grid: `pickup` beats before the first downbeat, then n_bars bars of 4."""
    n = pickup + n_bars * BEATS_PER_BAR
    beat_in_bar = (np.arange(n) - pickup) % BEATS_PER_BAR
    is_downbeat = beat_in_bar == 0
    is_downbeat[:pickup] = False
    return BeatGrid(
        times=0.5 + 0.5 * np.arange(n),
        is_downbeat=is_downbeat,
        beat_in_bar=beat_in_bar,
        bar_idx=np.cumsum(is_downbeat) - 1,
    )


def per_bar(grid, values_per_bar):
    """Spread one value per bar over its beats; pickup beats take bar 0's value."""
    return np.asarray(values_per_bar, dtype=np.float64)[np.maximum(grid.bar_idx, 0)]


def labels_for(segment_bars, energy, rise=None, drop_level=DROP_LEVEL, build_rise=BUILD_RISE, pickup=0):
    """Label a grid whose segments are segment_bars bars long, with a flat rms_s per segment and
    rms_d1 spread evenly so each segment's sum is its rise. Returns (grid, labels)."""
    grid = make_grid(sum(segment_bars), pickup)
    starts = np.concatenate([[0], np.cumsum(segment_bars)[:-1]])
    seg_of_bar = np.repeat(np.arange(len(segment_bars)), segment_bars)
    rms_s = per_bar(grid, np.asarray(energy, dtype=np.float64)[seg_of_bar])
    rise = np.zeros(len(segment_bars)) if rise is None else np.asarray(rise, dtype=np.float64)
    seg = seg_of_bar[np.maximum(grid.bar_idx, 0)]
    rms_d1 = rise[seg] / np.bincount(seg)[seg]
    return grid, label_sections(grid, rms_s, rms_d1, starts, drop_level, build_rise)


def segment_of_each_beat_labels(grid, labels, starts):
    """The single label of each segment, asserting it is the same for all its beats."""
    seg = np.searchsorted(starts, np.maximum(grid.bar_idx, 0), side="right") - 1
    out = []
    for s in range(len(starts)):
        assert len(set(labels[seg == s].tolist())) == 1, f"segment {s} has mixed labels"
        out.append(str(labels[seg == s][0]))
    return out


# =================================================================================
# _self_similarity
# =================================================================================


def test_ssm_is_square_symmetric_with_a_diagonal_of_ones():
    ssm = _self_similarity(blocks(5, 7))

    assert ssm.shape == (12, 12)
    np.testing.assert_allclose(ssm, ssm.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(ssm), 1.0)
    assert ssm.min() >= -1 - 1e-12 and ssm.max() <= 1 + 1e-12


def test_ssm_of_two_opposite_blocks_is_plus_one_within_and_minus_one_across():
    """After z-scoring, [1, 0] and [0, 1] become [+1, -1] and [-1, +1]: opposite directions."""
    bars = np.array([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4)

    np.testing.assert_allclose(_self_similarity(bars), block_ssm(4, 4), atol=1e-12)


def test_ssm_ignores_the_scale_of_a_column():
    bars = blocks(6, 6)
    scaled = bars * np.linspace(0.1, 100, bars.shape[1])

    np.testing.assert_allclose(_self_similarity(scaled), _self_similarity(bars), atol=1e-9)


def test_ssm_ignores_a_constant_column():
    bars = blocks(6, 6)
    with_constant = np.hstack([bars, np.full((len(bars), 1), 7.0)])

    np.testing.assert_allclose(_self_similarity(with_constant), _self_similarity(bars), atol=1e-12)


def test_ssm_of_constant_features_is_all_zeros_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        ssm = _self_similarity(np.ones((6, 3)))

    np.testing.assert_array_equal(ssm, np.zeros((6, 6)))


# =================================================================================
# _checkerboard_kernel
# =================================================================================


@pytest.mark.parametrize("k", [2, 4, 8, 16])
def test_kernel_matches_the_formula(k):
    d = np.arange(k) - (k - 1) / 2
    taper = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * (k / 4) ** 2))
    side = np.arange(k) < k // 2
    sign = np.where(side[:, None] == side[None, :], 1.0, -1.0)
    expected = sign * taper / taper.sum()

    np.testing.assert_allclose(_checkerboard_kernel(k), expected, atol=1e-12)


def test_kernel_has_the_checkerboard_signs():
    kernel = _checkerboard_kernel(8)

    assert (kernel[:4, :4] > 0).all() and (kernel[4:, 4:] > 0).all()
    assert (kernel[:4, 4:] < 0).all() and (kernel[4:, :4] < 0).all()


def test_kernel_abs_sums_to_one_and_sums_to_zero():
    kernel = _checkerboard_kernel(8)

    assert np.abs(kernel).sum() == pytest.approx(1.0)
    assert kernel.sum() == pytest.approx(0.0, abs=1e-12)


def test_kernel_weighs_the_middle_most():
    kernel = np.abs(_checkerboard_kernel(8))

    assert kernel[3, 3] == kernel.max()
    assert kernel[0, 0] < kernel[3, 3] / 10


# =================================================================================
# _novelty
# =================================================================================


def test_novelty_has_one_value_per_bar():
    assert _novelty(block_ssm(8, 8), K).shape == (16,)


def test_novelty_peaks_at_the_first_bar_of_the_new_section():
    novelty = _novelty(block_ssm(8, 8), K)

    assert np.argmax(novelty) == 8


def test_novelty_of_a_perfect_change_is_exactly_one():
    """Not rescaled: half as similar inside, half as dissimilar across, gives half the peak."""
    assert _novelty(block_ssm(8, 8), K)[8] == pytest.approx(1.0)
    assert _novelty(0.5 * block_ssm(8, 8), K)[8] == pytest.approx(0.5)


def test_novelty_inside_a_uniform_block_is_zero():
    novelty = _novelty(np.ones((16, 16)), K)

    np.testing.assert_allclose(novelty[4:13], 0.0, atol=1e-12)


def test_novelty_pads_with_zeros_outside_the_song():
    """At bar 0 only the bottom-right quadrant (bars 0..3 vs bars 0..3) is inside the song.
    At the last bar (15), the window covers bars 11..18, of which 11..15 exist."""
    novelty = _novelty(np.ones((16, 16)), K)

    assert novelty[0] == pytest.approx(0.25)
    assert novelty[-1] == pytest.approx(_checkerboard_kernel(K)[:5, :5].sum())


def test_novelty_matches_the_reference_on_random_input():
    rng = np.random.default_rng(3)
    ssm = rng.uniform(-1, 1, (30, 30))
    ssm = (ssm + ssm.T) / 2
    kernel = _checkerboard_kernel(K)
    padded = np.pad(ssm, K // 2)
    expected = [np.sum(kernel * padded[b : b + K, b : b + K]) for b in range(30)]

    np.testing.assert_allclose(_novelty(ssm, K), expected, atol=1e-12)


# =================================================================================
# section_novelty
# =================================================================================


def test_section_novelty_is_the_novelty_of_the_self_similarity():
    bars = blocks(10, 14, 9)

    np.testing.assert_allclose(section_novelty(bars, K), _novelty(_self_similarity(bars), K), atol=1e-12)


def test_section_novelty_has_one_float_per_bar():
    novelty = section_novelty(blocks(12, 12), K)

    assert novelty.shape == (24,)
    assert np.issubdtype(novelty.dtype, np.floating)


def test_section_novelty_peaks_where_the_blocks_change():
    assert np.argmax(section_novelty(blocks(12, 12), K)) == 12


def test_section_novelty_does_not_depend_on_the_scale_of_the_features():
    bars = blocks(12, 12, 12)

    np.testing.assert_allclose(section_novelty(bars * 50, K), section_novelty(bars, K), atol=1e-9)


@pytest.mark.parametrize("n_bars", [0, 1])
def test_section_novelty_of_fewer_than_two_bars_is_zeros_without_warnings(n_bars):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        novelty = section_novelty(np.ones((n_bars, 24)), K)

    np.testing.assert_array_equal(novelty, np.zeros(n_bars))


def test_odd_kernel_is_rejected():
    with pytest.raises(ValueError):
        section_novelty(blocks(12, 12), 7)


def test_section_novelty_does_not_modify_the_input():
    bars = blocks(12, 12)
    original = bars.copy()

    section_novelty(bars, K)

    np.testing.assert_array_equal(bars, original)


# =================================================================================
# find_section_boundaries: on hand-made novelty curves
# =================================================================================


def test_boundaries_are_the_prominent_peaks():
    novelty = spikes(24, b8=0.5, b14=0.05, b18=0.3)

    np.testing.assert_array_equal(find_section_boundaries(novelty, MIN_BARS, 0.1), [0, 8, 18])


def test_boundaries_use_prominence_as_given():
    novelty = spikes(24, b8=0.5, b18=0.3)

    np.testing.assert_array_equal(find_section_boundaries(novelty, MIN_BARS, 0.4), [0, 8])


def test_peaks_closer_than_min_section_bars_to_either_end_are_dropped():
    """Bar 3 is too close to the start, bar 21 to the end (24 - 4 = 20 is the last allowed)."""
    novelty = spikes(24, b3=1.0, b10=0.5, b21=1.0)

    found = find_section_boundaries(novelty, MIN_BARS, PROMINENCE)

    assert 3 not in found and 21 not in found
    assert found[0] == 0 and 10 in found


def test_a_peak_exactly_min_section_bars_from_the_end_is_kept():
    np.testing.assert_array_equal(find_section_boundaries(spikes(24, b20=0.5), MIN_BARS, PROMINENCE), [0, 20])


def test_close_peaks_keep_only_the_higher_one():
    np.testing.assert_array_equal(
        find_section_boundaries(spikes(24, b10=0.5, b12=0.9), MIN_BARS, PROMINENCE), [0, 12]
    )


def test_a_flat_novelty_gives_only_boundary_zero():
    np.testing.assert_array_equal(find_section_boundaries(np.zeros(24), MIN_BARS, PROMINENCE), [0])


@pytest.mark.parametrize("n_bars", [0, 1])
def test_boundaries_of_fewer_than_two_bars_are_only_zero(n_bars):
    """A 1-D [0], not a 0-d array: assert_array_equal would broadcast 0 against [0]."""
    found = find_section_boundaries(np.zeros(n_bars), MIN_BARS, PROMINENCE)

    assert found.shape == (1,)
    assert found[0] == 0


def test_boundaries_do_not_modify_the_novelty():
    novelty = spikes(24, b8=0.5, b18=0.3)
    original = novelty.copy()

    find_section_boundaries(novelty, MIN_BARS, PROMINENCE)

    np.testing.assert_array_equal(novelty, original)


# =================================================================================
# section_novelty -> find_section_boundaries: the chain on block features
# =================================================================================


def test_boundaries_are_found_where_the_blocks_change():
    np.testing.assert_array_equal(boundaries(blocks(12, 12, 12)), [0, 12, 24])


def test_boundaries_are_sorted_ints_starting_at_zero():
    found = boundaries(blocks(10, 14, 9, 11))

    assert np.issubdtype(found.dtype, np.integer)
    assert found[0] == 0
    assert (np.diff(found) > 0).all()


def test_no_section_is_shorter_than_min_section_bars():
    found = boundaries(blocks(12, 2, 12, 3, 12))

    assert (np.diff(found) >= MIN_BARS).all()
    assert len(found) >= 3


def test_no_boundary_close_to_the_end():
    """A 2-bar tail can't be its own section."""
    np.testing.assert_array_equal(boundaries(blocks(16, 2)), [0])


def test_min_section_bars_is_used_as_given():
    np.testing.assert_array_equal(boundaries(blocks(12, 6, 12), min_section_bars=8), [0, 12])


def test_prominence_drops_weak_changes():
    """Block 2 is block 1 slightly shifted: a weak change (novelty prominence ~0.2), while the
    change to block 3 is strong (~0.7) and the noise peaks are under 0.03."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(24)
    strong = rng.standard_normal(24)
    bars = np.vstack(
        [np.tile(base, (12, 1)), np.tile(base + 0.4 * strong, (12, 1)), np.tile(strong, (12, 1))]
    )
    bars += 0.1 * rng.standard_normal(bars.shape)

    np.testing.assert_array_equal(boundaries(bars, prominence=0.1), [0, 12, 24])
    np.testing.assert_array_equal(boundaries(bars, prominence=0.4), [0, 24])


@pytest.mark.parametrize("n_bars", [0, 1])
def test_fewer_than_two_bars_gives_only_boundary_zero(n_bars):
    found = boundaries(np.ones((n_bars, 24)))

    assert found.shape == (1,)
    assert found[0] == 0


def test_a_one_bar_song_can_be_labelled():
    grid = make_grid(1)
    found = boundaries(np.ones((1, 24)))

    labels = label_sections(grid, np.zeros(4), np.zeros(4), found, DROP_LEVEL, BUILD_RISE)

    assert labels.tolist() == ["intro"] * 4


# =================================================================================
# _segment_labels: the rules table
# =================================================================================


@pytest.mark.parametrize(
    ("energy", "rise", "expected"),
    [
        # quiet, loud, quiet: intro / outro by position
        ([0, 1, 0], [0, 0, 0], ["intro", "drop", "outro"]),
        # a quiet middle segment is a break
        ([0, 1, 0, 1, 0], [0, 0, 0, 0, 0], ["intro", "drop", "break", "drop", "outro"]),
        # rising into a drop is a build, wherever it is
        ([0, 0.2, 1, 0], [0, 1, 0, 0], ["intro", "build", "drop", "outro"]),
        ([0, 1, 0], [1, 0, 0], ["build", "drop", "outro"]),
        # rising into something that is not a drop is not a build
        ([0, 0.2, 0.3, 1], [1, 1, 1, 0], ["intro", "break", "build", "drop"]),
        ([0, 1, 0], [0, 0, 5], ["intro", "drop", "outro"]),
        # loud wins over rising, and a loud first or last segment is a drop
        ([1, 0, 1], [5, 5, 5], ["drop", "build", "drop"]),
        # level is relative to the song's own range: only the shift and spread matter
        ([10, 30, 10], [0, 0, 0], ["intro", "drop", "outro"]),
        ([0.0, 0.8, 0.7, 1.0], [0, 0, 0, 0], ["intro", "drop", "break", "drop"]),
        # one segment, or all the same: no drop
        ([3], [0], ["intro"]),
        ([2, 2, 2], [0, 0, 0], ["intro", "break", "outro"]),
    ],
)
def test_segment_rules(energy, rise, expected):
    assert _segment_labels(np.array(energy, float), np.array(rise, float), DROP_LEVEL, BUILD_RISE) == expected


def test_drop_level_and_build_rise_are_used_as_given():
    energy = np.array([0.0, 0.8, 1.0])
    rise = np.array([0.0, 0.6, 0.0])

    assert _segment_labels(energy, rise, 0.75, 0.5) == ["intro", "drop", "drop"]
    assert _segment_labels(energy, rise, 0.9, 0.5) == ["intro", "build", "drop"]
    assert _segment_labels(energy, rise, 0.9, 0.7) == ["intro", "break", "drop"]


# =================================================================================
# label_sections
# =================================================================================


def test_labels_have_one_known_label_per_beat():
    grid, labels = labels_for([4, 8, 4], [0, 1, 0])

    assert labels.shape == grid.times.shape
    assert set(labels.tolist()) <= set(SECTION_LABELS)


def test_labels_change_only_at_boundary_downbeats():
    grid, labels = labels_for([4, 8, 4], [0, 1, 0])

    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    assert grid.is_downbeat[changes].all()
    np.testing.assert_array_equal(grid.bar_idx[changes], [4, 12])


def test_labels_follow_the_segment_rules():
    grid, labels = labels_for([4, 8, 4, 8, 4], [0, 1, 0, 1, 0])

    starts = np.array([0, 4, 12, 16, 24])
    assert segment_of_each_beat_labels(grid, labels, starts) == ["intro", "drop", "break", "drop", "outro"]


def test_pickup_beats_join_the_first_segment():
    grid, labels = labels_for([4, 8, 4], [0, 1, 0], pickup=3)

    assert labels[:3].tolist() == ["intro"] * 3
    assert labels[3] == "intro"


def test_energy_is_the_mean_of_rms_s_over_the_segment():
    """Segment 1 is quiet for half its beats and very loud for the other half: its mean is the highest."""
    grid = make_grid(12)
    rms_s = per_bar(grid, [0, 0, 0, 0, 0, 0, 3, 3, 1, 1, 1, 1])
    labels = label_sections(grid, rms_s, np.zeros_like(rms_s), np.array([0, 4, 8]), DROP_LEVEL, BUILD_RISE)

    assert segment_of_each_beat_labels(grid, labels, np.array([0, 4, 8])) == ["intro", "drop", "outro"]


def test_rise_is_the_sum_of_rms_d1_over_the_segment():
    """0.05 per beat over 16 beats = 0.8 >= 0.5; 0.02 per beat = 0.32 < 0.5."""
    starts = np.array([0, 4, 8])
    for per_beat, middle in [(0.05, "build"), (0.02, "break")]:
        grid = make_grid(12)
        rms_s = per_bar(grid, [0] * 4 + [0.1] * 4 + [1] * 4)
        rms_d1 = np.where(grid.bar_idx // 4 == 1, per_beat, 0.0)
        labels = label_sections(grid, rms_s, rms_d1, starts, DROP_LEVEL, BUILD_RISE)

        assert segment_of_each_beat_labels(grid, labels, starts) == ["intro", middle, "drop"]


def test_boundaries_past_the_last_bar_are_ignored():
    grid = make_grid(12)
    rms_s = per_bar(grid, [0] * 4 + [1] * 4 + [0] * 4)
    labels = label_sections(
        grid, rms_s, np.zeros_like(rms_s), np.array([0, 4, 8, 40]), DROP_LEVEL, BUILD_RISE
    )

    assert segment_of_each_beat_labels(grid, labels, np.array([0, 4, 8])) == ["intro", "drop", "outro"]


def test_a_single_boundary_labels_everything_intro():
    grid, labels = labels_for([12], [0])

    assert set(labels.tolist()) == {"intro"}


@pytest.mark.parametrize("which", ["rms_s", "rms_d1"])
def test_curves_must_have_one_value_per_beat(which):
    grid = make_grid(8)
    curves = {"rms_s": np.zeros(len(grid.times)), "rms_d1": np.zeros(len(grid.times))}
    curves[which] = curves[which][:-1]

    with pytest.raises(ValueError):
        label_sections(grid, curves["rms_s"], curves["rms_d1"], np.array([0, 4]), DROP_LEVEL, BUILD_RISE)


def test_labels_do_not_modify_the_inputs():
    grid = make_grid(12)
    rms_s = per_bar(grid, [0] * 4 + [1] * 4 + [0] * 4)
    rms_d1 = np.linspace(-1, 1, len(rms_s))
    bounds = np.array([0, 4, 8])
    copies = [a.copy() for a in (rms_s, rms_d1, bounds, grid.times, grid.bar_idx)]

    label_sections(grid, rms_s, rms_d1, bounds, DROP_LEVEL, BUILD_RISE)

    for after, before in zip((rms_s, rms_d1, bounds, grid.times, grid.bar_idx), copies, strict=True):
        np.testing.assert_array_equal(after, before)


# =================================================================================
# Speed, config
# =================================================================================


def test_sections_are_fast_on_a_song_sized_input():
    """A 3-min song at 176 BPM: ~130 bars, ~530 beats, 24 timbre columns. Best of 3."""
    rng = np.random.default_rng(9)
    bars = blocks(16, 16, 32, 16, 16, 32, dim=24)
    grid = make_grid(len(bars))
    rms_s = rng.standard_normal(len(grid.times))
    rms_d1 = rng.standard_normal(len(grid.times))

    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        found = boundaries(bars)
        label_sections(grid, rms_s, rms_d1, found, DROP_LEVEL, BUILD_RISE)
        best = min(best, time.perf_counter() - start)

    assert best < 0.02, f"{best * 1000:.1f} ms"


def test_config_has_the_planned_settings():
    import amv.config as config

    assert config.SECTION_KERNEL_BARS == K
    assert config.MIN_SECTION_BARS == MIN_BARS
    assert config.NOVELTY_PROMINENCE == PROMINENCE
    assert config.DROP_LEVEL == DROP_LEVEL
    assert config.BUILD_RISE == BUILD_RISE


# =================================================================================
# Real song (integration)
# =================================================================================


@pytest.mark.integration
def test_bluelock_sections_are_plausible(beat_model, bluelock_wav):
    """Measured with a reference version: 8 sections, starting at 6, 32, 44, 54, 77, 81, 106 and
    118 s; the half-time breakdown (53.7-76.6 s) is a break, and there are drops on either side."""
    import amv.config as c
    from amv.music.audio import load_audio
    from amv.music.beats import build_beat_grid, clean_beats, detect_beats
    from amv.music.dense import compute_dense_features, compute_timbre_features
    from amv.music.dynamics import beat_dynamics, pool_to_beats

    y, sr = load_audio(bluelock_wav, c.AUDIO_SAMPLE_RATE)
    beats, downbeats = detect_beats(y, sr, beat_model)
    grid = build_beat_grid(
        clean_beats(beats, c.MIN_IBI), downbeats, c.BEATS_PER_BAR, c.DOWNBEAT_TOL, c.DOWNBEAT_WINDOW_BARS
    )
    timbre = compute_timbre_features(y, sr, c.HOP_LENGTH, c.N_FFT, c.N_MELS, c.N_MFCC)
    dense = compute_dense_features(y, sr, c.HOP_LENGTH, c.N_FFT, c.N_MELS, c.LOW_BAND_HZ, c.HIGH_BAND_HZ)
    dyn = beat_dynamics(pool_to_beats(dense, grid.times), dense.names, c.BEAT_SG_WINDOW, c.SG_POLYORDER)

    bars = pool_to_beats(timbre, grid.times[grid.is_downbeat])
    novelty = section_novelty(bars, c.SECTION_KERNEL_BARS)
    found = find_section_boundaries(novelty, c.MIN_SECTION_BARS, c.NOVELTY_PROMINENCE)
    labels = label_sections(grid, dyn["rms_s"], dyn["rms_d1"], found, c.DROP_LEVEL, c.BUILD_RISE)

    assert 4 <= len(found) <= 12
    assert (np.diff(found) >= c.MIN_SECTION_BARS).all()
    assert labels.shape == grid.times.shape
    breakdown = (grid.times > 58) & (grid.times < 74)
    assert np.mean(labels[breakdown] == "drop") < 0.2
    assert np.any(labels[grid.times < 50] == "drop")
    assert np.any(labels[grid.times > 90] == "drop")
