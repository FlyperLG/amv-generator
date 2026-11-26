"""Cut-detection metric: detected cut frames scored against hand labels (task 3.4).

Precision / recall / F1 within a frame tolerance, for all cuts or per cut type, plus the
detection - label offsets of the hits. Step 03 tunes its detectors with this; step 11
reuses it. See docs/03-shot-detection §3.4 for the decisions behind the rules.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike

from amv.shots.labels import CutLabels, CUT_TYPES


def match_cuts(true_frames: ArrayLike, pred_frames: ArrayLike, tol: int) -> np.ndarray:
    """Pair labels with detections one-to-one, with as many pairs as possible.

    A pair is allowed when |pred - true| <= tol. Both lists are sorted and walked once in
    order; each label takes the earliest unused detection in reach. Because every label's
    reach has the same width, this greedy walk gives a maximum matching, unlike
    nearest-first, which loses pairs when cuts are closer than 2 * tol.

    Args:
        true_frames: label frames, any order.
        pred_frames: detected frames, any order; duplicates are separate detections.
        tol: allowed distance in frames, inclusive.

    Returns:
        int64 array of shape [k, 2]: rows (index into true_frames, index into pred_frames),
        indices into the inputs as given, sorted by the first column. Shape (0, 2) when
        nothing matches or an input is empty.
    """
    label_order = np.argsort(true_frames)
    pred_order = np.argsort(pred_frames)
    labels = np.array(true_frames)[label_order].tolist()
    preds = np.array(pred_frames)[pred_order].tolist()

    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < len(labels) and j < len(preds):
        d = preds[j] - labels[i]
        if d < -tol:
            j += 1
        elif d > tol:
            i += 1
        else:
            pairs.append((label_order[i], pred_order[j]))
            i += 1
            j += 1

    pairs.sort()
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True, slots=True)
class CutScores:
    """Detection scores for one label file.

    Attributes:
        tp: detections matched to a label (of the scored type).
        fp: detections in the window matched to no label at all.
        fn: labels (of the scored type) left unmatched.
        precision: tp / (tp + fp); nan when there are no detections.
        recall: tp / (tp + fn); nan when there are no labels.
        f1: 2 p r / (p + r); 0.0 when p + r == 0, nan when p or r is nan.
        offsets: int64 detection frame - label frame for every TP, ordered by label. A
            median != 0 shows a systematic off-by-one in the detector.
    """

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    offsets: np.ndarray


def score_cuts(labels: CutLabels, detected: ArrayLike, tol: int, cut_type: str | None = None) -> CutScores:
    """Score detected cut frames against one label file.

    Detections outside the reviewed window (window_start < frame <= window_end) are dropped
    first and count as neither TP nor FP. Matching runs once, against all labels, so per-type
    scores stay consistent: with a cut_type, a detection that found a label of the other
    type is ignored (not a false alarm, and it can't steal a label of this type).

    Args:
        labels: the hand labels and their window.
        detected: detected cut frames, any int array-like, any order.
        tol: allowed distance in frames (SHOTS.eval_tol).
        cut_type: None for all cuts, or one of CUT_TYPES to score that type only.

    Returns:
        CutScores; fp is the same for every cut_type.

    Raises:
        ValueError: cut_type is not None and not in CUT_TYPES.
    """
    if cut_type is not None and cut_type not in CUT_TYPES:
        raise ValueError(f"Cut type needs to be one of {CUT_TYPES}, got {cut_type}.")

    detected = np.asarray(detected, dtype=np.int64)
    detected = detected[(labels.window_start < detected) & (detected <= labels.window_end)]
    matches = match_cuts(labels.frames, detected, tol)
    keep = (
        labels.cut_types[matches[:, 0]] == cut_type
        if cut_type is not None
        else np.ones(len(matches), dtype=bool)
    )
    kept = matches[keep]

    tp = len(kept)
    fp = len(detected) - len(matches)
    num_labels = int(np.sum(labels.cut_types == cut_type)) if cut_type is not None else len(labels.frames)
    fn = num_labels - tp

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    elif math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    offsets = detected[kept[:, 1]] - labels.frames[kept[:, 0]]
    return CutScores(tp, fp, fn, precision, recall, f1, offsets)


def _ratio(num: int, den: int) -> float:
    return math.nan if den == 0 else num / den
