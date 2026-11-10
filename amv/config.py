"""Central settings for the whole pipeline.

Each section is a frozen dataclass whose defaults are the configured values, plus one
instance of it (PATHS, DECODING, MUSIC). Library functions never import these instances:
entry points (scripts, benchmarks) pass a whole section to the functions that cache
outputs, and single fields (MUSIC.min_ibi) to the others. Variants for tests or
experiments: dataclasses.replace(MUSIC, drop_level=0.5).

Settings that change an output are part of its cache name (the settings_id), so changing
one rebuilds those files instead of reusing stale ones. Old files stay until deleted by
hand. Timeouts don't affect any output.

This module is a leaf: it imports nothing from amv.
"""

import dataclasses
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# --- Paths --------------------------------------------------------------------
# Anchored to the repo root, so scripts work from any working directory.

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@dataclass(frozen=True, slots=True)
class PathSettings:
    """Where inputs, caches and outputs live."""

    amv_dir: Path = DATA_DIR / "amvs"  # reference AMVs (training data)
    library_dir: Path = DATA_DIR / "library"  # source footage, one sub-folder per series
    song_dir: Path = DATA_DIR / "songs"  # input songs
    proxy_dir: Path = DATA_DIR / "proxies"  # step 01 proxies, keyed by video_id
    video_table_path: Path = DATA_DIR / "cache" / "videos.parquet"  # step 01 output, read by every later step
    song_table_path: Path = DATA_DIR / "cache" / "songs.parquet"  # step 01 output, read by every later step
    music_cache_dir: Path = DATA_DIR / "cache" / "music"  # step 02 beat tables and dense curves
    bench_dir: Path = DATA_DIR / "out" / "bench"  # benchmark runs (json + png)


PATHS = PathSettings()


# --- Shared across steps ------------------------------------------------------
# Values more than one step must agree on (e.g. ANALYSIS_FPS from step 03 on).


# --- Step 01: decoding & proxies ----------------------------------------------


@dataclass(frozen=True, slots=True)
class VideoProxySettings:
    """The video proxy: what all analysis reads instead of the 1080p original."""

    width: int = 640  # px; height keeps the aspect ratio, rounded to even
    preset: str = "ultrafast"  # x264 preset: encode speed over file size
    crf: int = 28  # x264 quality, lower = better and bigger
    timeout: float | None = 300  # seconds, ~2.5 h of footage at the measured ~30x


@dataclass(frozen=True, slots=True)
class AudioProxySettings:
    """The audio proxy: the wav that steps 02/07 analyse."""

    sample_rate: int = 22050  # Hz, librosa's default
    channels: int = 1  # mono
    timeout: float | None = 60  # seconds


@dataclass(frozen=True, slots=True)
class DecodingSettings:
    """Step 01: finding, probing and proxying the source media."""

    video_extensions: frozenset[str] = frozenset({".mp4", ".mkv"})  # lowercase; matched case-insensitively
    audio_extensions: frozenset[str] = frozenset({".mp3", ".wav", ".flac"})
    probe_timeout: float | None = 10  # seconds per ffprobe call
    video_proxy: VideoProxySettings = field(default_factory=VideoProxySettings)
    audio_proxy: AudioProxySettings = field(default_factory=AudioProxySettings)


DECODING = DecodingSettings()


# --- Step 02: music features --------------------------------------------------


@dataclass(frozen=True, slots=True)
class MusicSettings:
    """Step 02: every setting that changes the beat table or the dense curves."""

    sample_rate: int = DECODING.audio_proxy.sample_rate  # Hz; the wavs step 01 makes
    beat_checkpoint: str = "final0"  # beat_this checkpoint name or path
    min_ibi: float = 0.25  # seconds; a beat closer than this to the last kept beat is dropped
    beats_per_bar: int = 4  # beats; the songs are assumed to be in 4/4
    downbeat_tol: float = 0.07  # seconds; a detected downbeat further than this from every beat is ignored
    downbeat_window_bars: int = 8  # bars each side of a beat that vote on its bar phase
    hop_length: int = 512  # samples between frames, ~43 frames/s at 22050 Hz
    n_fft: int = 2048  # samples per analysis window
    n_mels: int = 64  # mel bands
    low_band_hz: float = 200  # Hz; mel bands centred below this are "low"
    high_band_hz: float = 4000  # Hz; mel bands centred above this are "high"
    sg_polyorder: int = 2  # Savitzky–Golay polynomial order, dense and beat level
    dense_sg_window_beats: float = 1.0  # beats; dense SG window ≈ this many beats in frames
    beat_sg_window: int = 5  # beats; SG window on the pooled beat series (odd)
    n_mfcc: int = 13  # MFCCs computed; mfcc_0 (loudness) is dropped
    section_kernel_bars: int = 8  # bars; novelty kernel width, half before and half after a boundary (even)
    min_section_bars: int = 4  # bars; the shortest section, first and last included
    novelty_prominence: float = 0.1  # how far a novelty peak must stand out; a perfect change scores 1
    drop_level: float = 0.75  # fraction of the song's segment-energy range (0 quietest, 1 loudest) for a drop
    build_rise: float = 0.5  # rms_s gained across a segment before a drop for it to be a build

    @property
    def settings_id(self) -> str:
        """A stable id of these values for cache file names: uuid5 of the sorted fields as JSON.

        The same in every process (unlike hash()), and it changes if any field changes."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(dataclasses.asdict(self), sort_keys=True)))


MUSIC = MusicSettings()
