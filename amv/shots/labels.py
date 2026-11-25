"""Hand-labelled shot cuts: the ground truth step 03 is tuned and scored against (task 3.4).

One CSV per video in PATHS.label_dir, written with the mpv labelling script; see
docs/03-shot-detection §3.4 for the format and the labelling conventions.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CUT_TYPES = ("hard", "soft")
WINDOW_TYPES = ("window_start", "window_end")


@dataclass(frozen=True, slots=True)
class CutLabels:
    """The cuts of one label file inside its reviewed window.

    Attributes:
        video_id: the video the frames refer to (its proxy).
        window_start: first reviewed frame. A cut on it can't be seen (frame - 1 wasn't).
        window_end: last reviewed frame.
        frames: int64 cut frames, strictly increasing, all in (window_start, window_end].
            A hard cut is the first frame of the new shot, a soft cut the middle of the
            transition.
        cut_types: str array aligned with frames, "hard" or "soft".
    """

    video_id: str
    window_start: int
    window_end: int
    frames: np.ndarray
    cut_types: np.ndarray


def load_cut_labels(path: Path) -> CutLabels:
    """Load and validate one hand-labelled cut file (PATHS.label_dir / "cuts_<video_id>.csv").

    The file has the header "frame,type", one window_start and one window_end row (the
    first and last frame that were reviewed, both inclusive), and one row per cut with type
    "hard" or "soft". Frames are 0-based frame indices of the proxy; a cut at frame f is the
    boundary between f - 1 and f, so it is only visible inside the window when
    window_start < f <= window_end. Rows may come in any order; blank lines and whitespace
    around fields are ignored. The labelling conventions are in docs/03-shot-detection §3.4.

    Args:
        path: the label file; its stem must start with "cuts_", the rest is the video_id.

    Returns:
        CutLabels with the cuts sorted by frame.

    Raises:
        ValueError: on a wrong file name or header, a missing or repeated window row, an
            empty window, an unknown type, a frame that is not a non-negative integer, a
            row without exactly two fields, a cut outside the window, or two cuts on the
            same frame. The message names the file and, where there is one, the line.
    """
    if not path.stem.startswith("cuts_"):
        raise ValueError(f"{path}: a label file must be named cuts_<video_id>.csv")

    video_id = path.stem.removeprefix("cuts_")
    rows = _read_rows(path)
    starts = [f for _, f, t in rows if t == "window_start"]
    ends = [f for _, f, t in rows if t == "window_end"]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(
            f"{path}: expected one window_start and one window_end, found {len(starts)} and {len(ends)}"
        )
    (start,), (end,) = starts, ends
    if start >= end:
        raise ValueError(f"{path}: window_start ({start}) must be before window_end ({end})")

    cuts = sorted((f, t) for _, f, t in rows if t in CUT_TYPES)
    for index, (frame, kind) in enumerate(cuts):
        if not start < frame <= end:
            raise ValueError(f"{path}: {kind} cut at frame {frame} lies outside the window ({start}, {end}]")
        if index != len(cuts) - 1 and frame == cuts[index + 1][0]:  # sorted: duplicates are neighbours
            raise ValueError(f"{path}: more than one cut at frame {frame}")

    frames = np.array([f for f, _ in cuts], dtype=np.int64)
    cut_types = np.array([t for _, t in cuts], dtype=str)

    return CutLabels(
        video_id=video_id,
        window_start=start,
        window_end=end,
        frames=frames,
        cut_types=cut_types,
    )


def _read_rows(path: Path) -> list[tuple[int, int, str]]:
    """Read a label file into (line number, frame, type) rows, in file order.

    Checks everything that concerns one line: the header, two fields per row, a
    non-negative integer frame, a known type. Whole-file rules are load_cut_labels' job.

    Raises:
        ValueError: "<path>:<line>: <what is wrong>".
    """

    def err(n, msg):
        return ValueError(f"{path}:{n}: {msg}")

    lines = [
        (n, line)
        for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
        if line.strip()
    ]
    if not lines or [f.strip() for f in lines[0][1].split(",")] != ["frame", "type"]:
        raise err(lines[0][0] if lines else 1, "expected the header 'frame,type'")

    rows = []
    for n, line in lines[1:]:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 2:
            raise err(n, f"expected 2 fields 'frame,type', got {len(fields)}")
        frame, kind = fields
        if not frame.isdecimal():
            raise err(n, f"frame must be a non-negative integer, got {frame!r}")
        if kind not in CUT_TYPES + WINDOW_TYPES:
            raise err(n, f"unknown type {kind!r}, expected one of {CUT_TYPES + WINDOW_TYPES}")
        rows.append((n, int(frame), kind))
    return rows
