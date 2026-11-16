"""Settings of the manual checks in checks/ (task 3.7 of step 02).

Same pattern as amv/config.py, kept here because amv/config.py configures the pipeline only.
Nothing here is in MUSIC.settings_id, so changing it never invalidates the music cache.
"""

from dataclasses import dataclass
from pathlib import Path

from amv.config import DATA_DIR


@dataclass(frozen=True, slots=True)
class PlotMusicSettings:
    """checks/plot_music.py: output folder, click track, figure size and the two timing measurements."""

    out_dir: Path = DATA_DIR / "out" / "music"  # pngs and click wavs; not a cache
    click_freq_hz: float = 1000.0  # Hz; beat click pitch
    downbeat_click_freq_hz: float = 1500.0  # Hz; higher, so "1" stands out
    click_duration: float = 0.03  # seconds
    beat_click_gain: float = 0.5  # 0 leaves only the downbeat clicks
    downbeat_click_gain: float = 1.0  # at most 1, or the wav clips
    fig_width: float = 16.0  # inches
    panel_height: float = 2.2  # inches per panel
    dpi: int = 120  # png resolution
    onset_hop_length: int = 64  # samples; ~2.9 ms resolution for beat_offsets
    onset_n_fft: int = 1024  # samples per onset analysis window
    max_onset_offset: float = 0.08  # seconds; a beat with no onset this close has none
    onset_peak_percentile: float = 90  # an onset peak must stand out by this percentile of the curve
    kick_max_hz: float = 100  # Hz; STFT bins up to here count as kick
    kick_window: float = 0.07  # seconds after a beat searched for its kick


PLOT_MUSIC = PlotMusicSettings()
