"""Tests for amv.shots.metrics (docs/03-shot-detection, task 3.4; reused by step 11).

Detected cuts are scored against the hand labels (amv.shots.labels) with precision, recall
and F1 at a tolerance of ±SHOTS.eval_tol frames. All tests use hand-made frame numbers whose
answer can be worked out on paper.

SPECIFICATION
  Settings are required arguments (Decision: metrics.py does not read amv.config; the entry
  point passes SHOTS.eval_tol in).  (config: SHOTS.eval_tol = 2 frames)

  match_cuts(true_frames, pred_frames, tol) -> np.ndarray, shape [k, 2], int64
  - One-to-one: each label and each detection is used at most once. A pair is allowed when
    |pred - true| <= tol (inclusive).
  - The matching has the MOST pairs possible (Decision: not nearest-first, which loses pairs
    when cuts are closer than 2 * tol; see test_matching_is_maximum_not_nearest_first). Among
    maximum matchings any one is fine. Hint: sort both, then walk them in order once.
  - Rows are (index into true_frames, index into pred_frames), sorted by the first column.
    The inputs may be unsorted; the indices refer to the inputs as given.
  - Any empty input -> shape (0, 2).

  score_cuts(labels, detected, tol, cut_type=None) -> CutScores
  - detected: frame indices, any int array-like, any order. Duplicates are separate
    detections (the second one is a false positive).
  - Only detections with labels.window_start < frame <= labels.window_end are scored; the
    others are dropped first, neither TP nor FP. (Decision: the same rule as for labels. A cut
    at f is the boundary between f - 1 and f, visible only if both frames were reviewed.)
  - Matching runs ONCE, against all labels, with match_cuts.
  - cut_type None: tp = matched pairs, fp = detections in the window - tp, fn = labels - tp.
  - cut_type "hard" or "soft": tp = matched pairs whose label has that type, fn = labels of
    that type left unmatched, fp = detections matched to NO label. Detections matched to a
    label of the other type are ignored. (Decision: a found soft cut is not a false alarm
    in the hard score, and it can't steal a hard label because matching ran on all labels.)
  - Any other cut_type -> ValueError.
  - precision = tp / (tp + fp), recall = tp / (tp + fn), f1 = 2 p r / (p + r).
    0 / 0 -> nan (Decision: "no detections" or "no labels" is undefined, neither perfect nor
    zero). f1 is nan if p or r is nan, and 0.0 if p + r == 0. No numpy warnings.
  - offsets: detection frame - label frame for every TP, int64, ordered by label frame. A
    median offset != 0 shows a systematic off-by-one in the detector.

  CutScores: frozen dataclass: tp, fp, fn (int), precision, recall, f1 (float), offsets.

  OPEN (not decided yet, not tested): a wider tolerance for soft labels. Soft labels mark the
  middle of transitions up to ~12 frames long, so a detector that fires at the start or end
  of a dissolve misses by more than 2. Decide once offsets on real detections are known.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest
from amv.shots.metrics import CutScores, match_cuts, score_cuts

from amv.config import SHOTS
from amv.shots.labels import CutLabels

TOL = 2


def make_labels(frames, cut_types=None, window=(0, 1000)) -> CutLabels:
    """CutLabels built by hand: all cuts hard unless cut_types is given."""
    frames = np.asarray(frames, np.int64)
    if cut_types is None:
        cut_types = ["hard"] * len(frames)
    return CutLabels(
        video_id="vid",
        window_start=window[0],
        window_end=window[1],
        frames=frames,
        cut_types=np.asarray(cut_types, dtype=str),
    )


def pairs(true, pred, tol=TOL) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in row) for row in match_cuts(true, pred, tol)]


# --- match_cuts ---------------------------------------------------------------


def test_exact_matches_pair_up():
    assert pairs([10, 50, 90], [10, 50, 90]) == [(0, 0), (1, 1), (2, 2)]


def test_tolerance_is_inclusive():
    assert pairs([10], [12]) == [(0, 0)]
    assert pairs([10], [8]) == [(0, 0)]
    assert pairs([10], [13]) == []


def test_zero_tolerance_needs_the_exact_frame():
    assert pairs([10, 20], [10, 21], tol=0) == [(0, 0)]


def test_each_label_and_detection_is_used_once():
    assert len(pairs([10], [9, 10, 11])) == 1
    assert len(pairs([9, 10, 11], [10])) == 1


def test_matching_is_maximum_not_nearest_first():
    # nearest-first pairs 11<->10 (distance 1) and then 8 and 13 are 5 apart: 1 pair.
    # 8<->10 and 11<->13 are both within 2: the maximum is 2 pairs.
    assert pairs([8, 11], [10, 13]) == [(0, 0), (1, 1)]


def test_indices_refer_to_unsorted_inputs():
    result = pairs([90, 10, 50], [51, 89, 11])

    assert sorted(result) == [(0, 1), (1, 2), (2, 0)]
    assert result == sorted(result)


def test_result_shape_and_dtype():
    result = match_cuts([10, 20], [10, 20], TOL)
    assert result.shape == (2, 2)
    assert result.dtype == np.int64


@pytest.mark.parametrize(("true", "pred"), [([], []), ([10], []), ([], [10])])
def test_empty_inputs_give_no_pairs(true, pred):
    assert match_cuts(true, pred, TOL).shape == (0, 2)


# --- score_cuts: counts -------------------------------------------------------


def test_perfect_detection():
    s = score_cuts(make_labels([10, 50, 90]), [10, 50, 90], TOL)

    assert isinstance(s, CutScores)
    assert (s.tp, s.fp, s.fn) == (3, 0, 0)
    assert (s.precision, s.recall, s.f1) == (1.0, 1.0, 1.0)


def test_counts_and_rates_worked_out_by_hand():
    # 10 found (+1), 50 found (-2), 90 missed; 70 and 200 are false alarms
    s = score_cuts(make_labels([10, 50, 90]), [11, 48, 70, 200], TOL)

    assert (s.tp, s.fp, s.fn) == (2, 2, 1)
    assert s.precision == pytest.approx(2 / 4)
    assert s.recall == pytest.approx(2 / 3)
    assert s.f1 == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_duplicate_detection_is_a_false_positive():
    s = score_cuts(make_labels([10]), [10, 10], TOL)

    assert (s.tp, s.fp, s.fn) == (1, 1, 0)


def test_unsorted_detections_score_the_same():
    labels = make_labels([10, 50, 90])
    a = score_cuts(labels, [11, 48, 70], TOL)
    b = score_cuts(labels, [70, 11, 48], TOL)

    assert (a.tp, a.fp, a.fn) == (b.tp, b.fp, b.fn)
    np.testing.assert_array_equal(a.offsets, b.offsets)


def test_detections_outside_the_window_are_ignored():
    labels = make_labels([105, 150], window=(100, 200))
    # 50 and 99 are before the window, 100 is the boundary 99|100 (99 not reviewed), 300 after it
    s = score_cuts(labels, [50, 99, 100, 105, 150, 300], TOL)

    assert (s.tp, s.fp, s.fn) == (2, 0, 0)


def test_detection_on_window_end_counts():
    # 200 is the boundary 199|200, both reviewed: inside, and a false alarm here
    s = score_cuts(make_labels([150], window=(100, 200)), [150, 200], TOL)

    assert (s.tp, s.fp, s.fn) == (1, 1, 0)


def test_offsets_are_detection_minus_label_in_label_order():
    s = score_cuts(make_labels([10, 50, 90]), [91, 48, 11], TOL)

    np.testing.assert_array_equal(s.offsets, [1, -2, 1])
    assert s.offsets.dtype == np.int64


def test_tolerance_is_passed_through():
    labels = make_labels([10])

    assert score_cuts(labels, [13], 2).tp == 0
    assert score_cuts(labels, [13], 3).tp == 1


# --- score_cuts: per cut type -------------------------------------------------


@pytest.fixture
def mixed():
    """Hard cuts at 10 and 50, soft cuts at 30 and 70."""
    return make_labels([10, 30, 50, 70], ["hard", "soft", "hard", "soft"])


def test_all_types_by_default(mixed):
    # found: 10, 30;  missed: 50, 70;  false alarm: 90
    s = score_cuts(mixed, [10, 31, 90], TOL)

    assert (s.tp, s.fp, s.fn) == (2, 1, 2)


def test_hard_only_ignores_detections_of_soft_cuts(mixed):
    s = score_cuts(mixed, [10, 31, 90], TOL, cut_type="hard")

    # 31 found the soft cut at 30: neither TP nor FP here
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    np.testing.assert_array_equal(s.offsets, [0])


def test_soft_only_ignores_detections_of_hard_cuts(mixed):
    s = score_cuts(mixed, [10, 31, 90], TOL, cut_type="soft")

    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    np.testing.assert_array_equal(s.offsets, [1])


def test_per_type_counts_add_up_to_the_total(mixed):
    detected = [9, 29, 52, 90, 120]
    total = score_cuts(mixed, detected, TOL)
    hard = score_cuts(mixed, detected, TOL, cut_type="hard")
    soft = score_cuts(mixed, detected, TOL, cut_type="soft")

    assert hard.tp + soft.tp == total.tp
    assert hard.fn + soft.fn == total.fn
    assert hard.fp == soft.fp == total.fp


def test_matching_runs_against_all_labels():
    # Hard cut at 10, soft cut at 14. On all labels the maximum matching is 12<->10 and
    # 15<->14, so 15 found the soft cut and is ignored in the hard score. Matching on the
    # hard labels alone would leave 15 unmatched and wrongly count it as a false positive.
    labels = make_labels([10, 14], ["hard", "soft"])
    s = score_cuts(labels, [12, 15], TOL, cut_type="hard")

    assert (s.tp, s.fp, s.fn) == (1, 0, 0)


def test_unknown_cut_type_raises(mixed):
    with pytest.raises(ValueError):
        score_cuts(mixed, [10], TOL, cut_type="fade")


# --- score_cuts: undefined rates ----------------------------------------------


def test_no_detections_gives_nan_precision_and_zero_recall():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = score_cuts(make_labels([10, 50]), [], TOL)

    assert (s.tp, s.fp, s.fn) == (0, 0, 2)
    assert math.isnan(s.precision)
    assert s.recall == 0.0
    assert math.isnan(s.f1)


def test_no_labels_gives_zero_precision_and_nan_recall():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = score_cuts(make_labels([]), [10], TOL)

    assert (s.tp, s.fp, s.fn) == (0, 1, 0)
    assert s.precision == 0.0
    assert math.isnan(s.recall)
    assert math.isnan(s.f1)


def test_all_wrong_gives_f1_zero_not_nan():
    s = score_cuts(make_labels([10]), [500], TOL)

    assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)


def test_no_soft_labels_gives_nan_soft_recall():
    s = score_cuts(make_labels([10]), [10], TOL, cut_type="soft")

    assert (s.tp, s.fp, s.fn) == (0, 0, 0)
    assert math.isnan(s.recall)


def test_scores_are_frozen():
    s = score_cuts(make_labels([10]), [10], TOL)

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.tp = 0


# --- Config ---------------------------------------------------------------------


def test_eval_tolerance_is_two_frames():
    assert SHOTS.eval_tol == 2
