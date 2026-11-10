import numpy as np
import scipy

from amv.music.beats import BeatGrid

SECTION_LABELS = ("intro", "build", "drop", "break", "outro")  # fixed vocabulary, not config


def find_section_boundaries(
    novelty: np.ndarray,
    min_section_bars: int,
    prominence: float,
) -> np.ndarray:
    """Pick the section starts: the novelty peaks that stand out by at least prominence.

    Peaks are at least min_section_bars apart (the higher one wins), and peaks closer than
    min_section_bars to either end are dropped, so no section is shorter than
    min_section_bars, the first and last included.

    Args:
        novelty (np.ndarray): The novelty per bar from section_novelty, shape [n_bars].
            Not modified.
        min_section_bars (int): The shortest section allowed, in bars.
        prominence (float): How far a novelty peak must stand out, in similarity units
            (a perfect change scores 1).

    Returns:
        np.ndarray: The sorted int bar indices where a section starts, 1-D, always
            starting with 0. Just [0] for fewer than 2 bars."""
    n_bars = len(novelty)
    if n_bars < 2:
        return np.array([0])
    peaks, _ = scipy.signal.find_peaks(novelty, prominence=prominence, distance=min_section_bars)
    peaks = peaks[(min_section_bars <= peaks) & (peaks <= (n_bars - min_section_bars))]
    peaks = np.concatenate([[0], peaks])
    return peaks


def _self_similarity(bar_features: np.ndarray) -> np.ndarray:
    """Compare every bar with every other bar by the direction of its feature vector.

    Each column is z-scored over the bars first, so a column's scale doesn't matter (mfcc_1
    is in the hundreds, chroma in [0, 1]) and a constant column, which says nothing about
    where sections change, becomes zeros. The similarity of two bars is then the cosine of
    their rows: 1 = alike, 0 = unrelated, -1 = opposite.

    Args:
        bar_features (np.ndarray): The features per bar, shape [n_bars, D]. Not modified.

    Returns:
        np.ndarray: The symmetric similarity matrix, shape [n_bars, n_bars], values in
            [-1, 1] and 1 on the diagonal. A bar whose row is all zeros after the z-scoring
            has similarity 0 with every bar, itself included. Never NaN."""
    std = np.std(bar_features, axis=0)
    safe_std = np.where(std == 0, 1, std)
    zscores = (bar_features - bar_features.mean(axis=0)) / safe_std
    norms = np.linalg.norm(zscores, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1, norms)
    normalized = zscores / safe_norms
    sim_matrix = normalized @ normalized.T
    return sim_matrix


def _checkerboard_kernel(kernel_bars: int) -> np.ndarray:
    """Build the kernel that scores "alike within each side, different across" at a boundary.

    The first kernel_bars / 2 rows and columns stand for the bars before the boundary, the
    rest for the bars after it. The sign is +1 where row and column are on the same side
    (before vs before, after vs after) and -1 where they are on opposite sides. A Gaussian
    taper exp(-(di^2 + dj^2) / (2 sigma^2)), with di and dj the distance from the kernel's
    centre (k - 1) / 2 and sigma = k / 4, makes the bars next to the boundary count most.
    The kernel is scaled so the sum of |K| is 1, so a perfect change scores exactly 1.
    The four quadrants weigh the same, so K sums to 0 and a stretch without a change
    scores 0.

    Args:
        kernel_bars (int): The kernel width k in bars. Even.

    Returns:
        np.ndarray: The kernel, shape [k, k], symmetric."""
    d = np.arange(kernel_bars) - (kernel_bars - 1) / 2
    taper = np.exp(-((d[:, None] ** 2 + d[None, :] ** 2) / (2 * (kernel_bars / 4) ** 2)))
    side = np.arange(kernel_bars) < kernel_bars // 2
    sign = np.where(side[:, None] == side[None, :], 1.0, -1.0)
    kernel = sign * taper
    kernel = kernel / taper.sum()
    return kernel


def section_novelty(bar_features: np.ndarray, kernel_bars: int) -> np.ndarray:
    """Score every bar for how much the song's timbre and harmony change there.

    Compares every bar with every other (_self_similarity), then slides a checkerboard
    kernel along the diagonal (_novelty). A peak at bar b means a section starts at b. The
    features are z-scored per column, so a song without real sections still gets peaks at
    its largest changes.

    Args:
        bar_features (np.ndarray): The timbre features pooled per bar, shape [n_bars, D],
            row k = bar_idx k. Not modified.
        kernel_bars (int): The kernel width in bars: kernel_bars / 2 before vs after. Even.

    Returns:
        np.ndarray: The novelty per bar, shape [n_bars], in similarity units (a perfect
            change scores 1). All zeros for fewer than 2 bars.

    Raises:
        ValueError: If kernel_bars is odd."""
    if kernel_bars % 2 != 0:
        raise ValueError(
            f"Received uneven number of kernel_bars {kernel_bars}, number of bars needs to be even."
        )

    n_bars = bar_features.shape[0]
    if n_bars < 2:
        return np.zeros(n_bars)

    similarity = _self_similarity(bar_features)
    return _novelty(similarity, kernel_bars)


def _novelty(ssm: np.ndarray, kernel_bars: int) -> np.ndarray:
    """Score every bar for how much the song changes there.

    novelty[b] is the sum of the checkerboard kernel times the k x k block of ssm over the
    bars b - k/2 .. b + k/2 - 1. So it compares the k/2 bars before b with the k/2 bars from
    b on, and a peak at bar b means the section changes between bar b - 1 and bar b. ssm is
    zero-padded by k/2 on every side, so the windows at the song's edges exist and the bars
    outside the song add nothing. The result is not rescaled: a perfect change scores 1.

    Args:
        ssm (np.ndarray): The bar similarity matrix from _self_similarity, shape
            [n_bars, n_bars]. Not modified.
        kernel_bars (int): The kernel width k in bars. Even.

    Returns:
        np.ndarray: The novelty per bar, shape [n_bars]."""
    padded = np.pad(ssm, kernel_bars // 2)
    result = []
    kernel = _checkerboard_kernel(kernel_bars)
    for b in range(ssm.shape[0]):
        window = padded[b : b + kernel_bars, b : b + kernel_bars]
        value = (kernel * window).sum()
        result.append(value)
    return np.array(result)


def label_sections(
    grid: BeatGrid,
    rms_s: np.ndarray,
    rms_d1: np.ndarray,
    boundaries: np.ndarray,
    drop_level: float,
    build_rise: float,
) -> np.ndarray:
    """Label every beat with its section: intro, build, drop, break or outro.

    Segment s is the bars from boundaries[s] up to the next boundary; pickup beats join
    segment 0, and boundaries past the last bar are ignored. Each segment gets the mean
    rms_s of its beats as its energy and the sum of rms_d1 as its rise, then
    _segment_labels applies the rules.

    Args:
        grid (BeatGrid): The beat grid the curves belong to. Not modified.
        rms_s (np.ndarray): The smoothed loudness per beat, shape [B]. Not modified.
        rms_d1 (np.ndarray): Its change per beat, shape [B]. Not modified.
        boundaries (np.ndarray): The section starts from find_section_boundaries. Not modified.
        drop_level (float): The energy level from which a segment is a drop, as a fraction
            of the song's segment-energy range (0 = quietest segment, 1 = loudest).
        build_rise (float): The rise from which a segment leading into a drop is a build,
            in rms_s units.

    Returns:
        np.ndarray: One label per beat, shape [B], every one in SECTION_LABELS.

    Raises:
        ValueError: If rms_s or rms_d1 doesn't have one value per beat."""
    if len(rms_s) != len(grid.times):
        raise ValueError(f"rms_s has {len(rms_s)} values for {len(grid.times)} beats")
    if len(rms_d1) != len(grid.times):
        raise ValueError(f"rms_d1 has {len(rms_d1)} values for {len(grid.times)} beats")

    bar_idx = np.maximum(grid.bar_idx, 0)
    segments = np.searchsorted(boundaries, bar_idx, side="right") - 1
    # boundaries past the last bar get no beats, and bincount stops at the last segment used
    energy = np.bincount(segments, weights=rms_s) / np.bincount(segments)
    rise = np.bincount(segments, weights=rms_d1)
    labels = np.array(_segment_labels(energy, rise, drop_level, build_rise))
    return labels[segments]


def _segment_labels(energy: np.ndarray, rise: np.ndarray, drop_level: float, build_rise: float) -> list[str]:
    """Label each segment from its energy, its rise and its position in the song.

    The energies are ranked within the song: level = (energy - min) / (max - min), so the
    quietest segment is 0 and the loudest 1 (all 0 if they are equal). Rules, first match
    wins:
        1. level >= drop_level                                -> "drop"
        2. rise >= build_rise and the next segment is a drop  -> "build"
        3. the first segment -> "intro", the last -> "outro", any other -> "break"
    So a single segment is "intro", a loud first or last segment is a "drop", and a rising
    segment that doesn't lead into a drop (e.g. a fade-in) is not a build.

    Args:
        energy (np.ndarray): The mean rms_s of each segment, shape [S]. Not modified.
        rise (np.ndarray): The sum of rms_d1 over each segment, shape [S]. Not modified.
        drop_level (float): The level from which a segment is a drop, in [0, 1].
        build_rise (float): The rise from which a segment leading into a drop is a build,
            in rms_s units.

    Returns:
        list[str]: One label per segment, in order, every one in SECTION_LABELS."""
    energy_min = energy.min()
    levels = (energy - energy_min) / ((energy.max() - energy_min) or 1)
    drops = levels >= drop_level
    labels: list[str] = []
    for s in range(len(energy)):
        if drops[s]:
            label = "drop"
        elif rise[s] >= build_rise and s + 1 < len(levels) and drops[s + 1]:
            label = "build"
        elif s == 0:
            label = "intro"
        elif s == len(levels) - 1:
            label = "outro"
        else:
            label = "break"
        labels.append(label)
    return labels
