"""Central settings for the whole pipeline.

Changing a value that goes into the ffmpeg arguments (PROXY_WIDTH, PROXY_PRESET,
PROXY_CRF, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS) changes the settings_id in the cached
file names, so the next run rebuilds those files instead of reusing stale ones (old
files stay in data/proxies/ until deleted by hand). Timeouts don't affect it.
"""

import shutil
from pathlib import Path

# --- Paths --------------------------------------------------------------------
# Anchored to the repo root, so scripts work from any working directory.

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

AMV_DIR = DATA_DIR / "amvs"  # reference AMVs (training data)
LIBRARY_DIR = DATA_DIR / "library"  # source footage, one sub-folder per series
SONG_DIR = DATA_DIR / "songs"  # source directory for songs
PROXY_DIR = DATA_DIR / "proxies"  # generated cache, keyed by video_id
VIDEO_TABLE_PATH = DATA_DIR / "cache" / "videos.parquet"  # step 01 video output, read by every later step
SONG_TABLE_PATH = DATA_DIR / "cache" / "songs.parquet"  # step 01 song output, read by every later step

# --- External tools -----------------------------------------------------------


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found. Please install ffmpeg and ensure {name} is in your PATH.")
    return path


FFMPEG = _require_tool("ffmpeg")
FFPROBE = _require_tool("ffprobe")


# --- Shared across steps ------------------------------------------------------
# Values more than one step must agree on (e.g. ANALYSIS_FPS from step 03 on).


# --- Step 01: decoding & proxies ----------------------------------------------

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv"})  # lowercase; matched case-insensitively
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac"})

PROBE_TIMEOUT: float | None = 10  # seconds per ffprobe call

# Video proxy: what all analysis reads instead of the 1080p original
PROXY_WIDTH = 640  # px; height keeps the aspect ratio, rounded to even
PROXY_PRESET = "ultrafast"  # x264 preset: encode speed over file size
PROXY_CRF = 28  # x264 quality, lower = better and bigger
PROXY_TRANSCODE_TIMEOUT: float | None = 300  # seconds, ~2.5 h of footage at the measured ~30x

# Audio proxy: the wav that steps 02/07 analyse
AUDIO_SAMPLE_RATE = 22050  # Hz, librosa's default
AUDIO_CHANNELS = 1  # mono
AUDIO_EXTRACTION_TIMEOUT: float | None = 60  # seconds
