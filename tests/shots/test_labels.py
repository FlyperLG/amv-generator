"""Tests for amv.shots.labels (docs/03-shot-detection, task 3.4).

The hand-labelled cut files are the ground truth the shot detector is tuned and scored
against. They are made with the mpv script from docs/03 (one key press per cut) and stored
in git, so the loader is strict: a malformed file fails loudly instead of silently
skewing F1.

SPECIFICATION
  File: PATHS.label_dir / f"cuts_{video_id}.csv". Self-contained (Decision: the reviewed
  window is stored inside the file, not in the name or elsewhere):

      frame,type
      899,window_start
      901,hard
      942,soft
      ...
      2697,window_end

  - frame: 0-based frame index in the PROXY (constant fps), not the original video.
  - type: hard | soft | window_start | window_end.
      hard  the last frame of shot A is directly followed by the first frame of shot B; the
            label is the first frame of B (= frame_in of the new shot). Also when B starts
            blurred, dark or filtered.
      soft  there are frames where neither shot is clearly shown (dissolve, fade through
            black or white, flash, glitch noise); the label is the middle of those frames.
    Effects inside one shot (colour cycles, inverts, strobes, blur-in, zoom) are not labelled.
  - window_start and window_end are the first and last frame that were reviewed (both seen).
    A cut at f is the boundary between f - 1 and f, so it is visible only if both were
    reviewed: window_start < f <= window_end. (Decision: (start, end], not [start, end): a
    cut on window_start would need frame window_start - 1, which nobody looked at.)

  load_cut_labels(path) -> CutLabels
  - video_id = the file stem without the "cuts_" prefix; another file name -> ValueError.
  - The first non-blank line must be the header "frame,type" -> else ValueError.
  - Blank lines are skipped; whitespace around fields is stripped.
  - Exactly one window_start and one window_end, window_start < window_end -> else ValueError.
  - Any other type, a frame that is not a non-negative integer, a row without exactly two
    fields -> ValueError.
  - Every cut lies inside the window, window_start < frame <= window_end -> else ValueError.
    (Decision: a label outside the window means the window is wrong; don't drop it silently.)
  - Two cuts at the same frame -> ValueError, whatever their types.
  - Rows may come in any order (mpv appends in key-press order); cuts are returned sorted.
  - A window without cuts is valid: empty arrays.

  CutLabels: frozen dataclass
    video_id      str
    window_start  int
    window_end    int
    frames        np.ndarray int64, strictly increasing
    cut_types     np.ndarray of str ("hard" / "soft"), aligned with frames
"""

import dataclasses

import numpy as np
import pytest

from amv.config import PATHS
from amv.shots.labels import CutLabels, load_cut_labels

BLUELOCK_ID = "4b5d74f6-0f5d-5d70-baba-c7e8ec42a66c_7653e33e-4305-5c35-9455-c145f48bb433"


@pytest.fixture
def write_labels(tmp_path):
    """Factory: write a label file and return its path.  write_labels("frame,type\\n...")"""

    def _write(text: str, video_id: str = "vid", name: str | None = None):
        path = tmp_path / (name or f"cuts_{video_id}.csv")
        path.write_text(text, encoding="utf-8")
        return path

    return _write


VALID = "frame,type\n100,window_start\n105,hard\n130,soft\n150,hard\n200,window_end\n"


# --- Happy path ---------------------------------------------------------------


def test_loads_window_cuts_and_types(write_labels):
    labels = load_cut_labels(write_labels(VALID, video_id="abc"))

    assert isinstance(labels, CutLabels)
    assert labels.video_id == "abc"
    assert (labels.window_start, labels.window_end) == (100, 200)
    np.testing.assert_array_equal(labels.frames, [105, 130, 150])
    assert labels.frames.dtype == np.int64
    assert list(labels.cut_types) == ["hard", "soft", "hard"]


def test_window_bounds_are_plain_ints(write_labels):
    labels = load_cut_labels(write_labels(VALID))

    assert type(labels.window_start) is int
    assert type(labels.window_end) is int


def test_labels_are_frozen(write_labels):
    labels = load_cut_labels(write_labels(VALID))

    with pytest.raises(dataclasses.FrozenInstanceError):
        labels.window_start = 0


def test_rows_in_any_order_are_sorted_by_frame(write_labels):
    text = "frame,type\n150,hard\n200,window_end\n130,soft\n100,window_start\n105,hard\n"
    labels = load_cut_labels(write_labels(text))

    np.testing.assert_array_equal(labels.frames, [105, 130, 150])
    assert list(labels.cut_types) == ["hard", "soft", "hard"]


def test_blank_lines_and_whitespace_are_tolerated(write_labels):
    text = "\nframe,type\n100,window_start\n\n 105 , hard \n200,window_end\n\n"
    labels = load_cut_labels(write_labels(text))

    np.testing.assert_array_equal(labels.frames, [105])
    assert list(labels.cut_types) == ["hard"]


def test_a_cut_on_window_end_is_inside(write_labels):
    labels = load_cut_labels(write_labels("frame,type\n100,window_start\n200,hard\n200,window_end\n"))

    np.testing.assert_array_equal(labels.frames, [200])


def test_a_window_without_cuts_is_valid(write_labels):
    labels = load_cut_labels(write_labels("frame,type\n100,window_start\n200,window_end\n"))

    assert labels.frames.shape == (0,)
    assert labels.cut_types.shape == (0,)


# --- Malformed files ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("100,window_start\n105,hard\n200,window_end\n", id="no-header"),
        pytest.param("frame,kind\n100,window_start\n200,window_end\n", id="wrong-header"),
        pytest.param("frame,type\n105,hard\n200,window_end\n", id="no-window-start"),
        pytest.param("frame,type\n100,window_start\n105,hard\n", id="no-window-end"),
        pytest.param(
            "frame,type\n100,window_start\n110,window_start\n200,window_end\n", id="two-window-starts"
        ),
        pytest.param("frame,type\n200,window_start\n100,window_end\n", id="start-after-end"),
        pytest.param("frame,type\n100,window_start\n100,window_end\n", id="empty-window"),
        pytest.param("frame,type\n100,window_start\n105,fade\n200,window_end\n", id="unknown-type"),
        pytest.param("frame,type\n100,window_start\n10.5,hard\n200,window_end\n", id="float-frame"),
        pytest.param("frame,type\n100,window_start\nabc,hard\n200,window_end\n", id="text-frame"),
        pytest.param("frame,type\n-5,window_start\n0,hard\n200,window_end\n", id="negative-frame"),
        pytest.param("frame,type\n100,window_start\n105\n200,window_end\n", id="missing-field"),
        pytest.param("frame,type\n100,window_start\n105,hard,x\n200,window_end\n", id="extra-field"),
        pytest.param("frame,type\n100,window_start\n99,hard\n200,window_end\n", id="cut-before-window"),
        pytest.param("frame,type\n100,window_start\n100,hard\n200,window_end\n", id="cut-on-window-start"),
        pytest.param("frame,type\n100,window_start\n201,hard\n200,window_end\n", id="cut-after-window"),
        pytest.param(
            "frame,type\n100,window_start\n105,hard\n105,soft\n200,window_end\n", id="two-cuts-same-frame"
        ),
    ],
)
def test_malformed_files_raise(write_labels, text):
    with pytest.raises(ValueError):
        load_cut_labels(write_labels(text))


def test_file_name_without_cuts_prefix_raises(write_labels):
    with pytest.raises(ValueError):
        load_cut_labels(write_labels(VALID, name="labels_vid.csv"))


# --- The real hand-labelled file ----------------------------------------------


def test_labels_live_in_their_own_data_folder():
    assert PATHS.label_dir == PATHS.amv_dir.parent / "labels"


@pytest.mark.integration
def test_bluelock_labels_load():
    path = PATHS.label_dir / f"cuts_{BLUELOCK_ID}.csv"
    if not path.exists():
        pytest.skip(f"{path.name} not found in data/labels/")

    labels = load_cut_labels(path)

    assert labels.video_id == BLUELOCK_ID
    assert (labels.window_start, labels.window_end) == (899, 2697)  # ~30 s to ~90 s at 29.97 fps
    assert len(labels.frames) == 76
    assert (labels.cut_types == "hard").sum() == 51
    assert (labels.cut_types == "soft").sum() == 25
