import shutil
from pathlib import Path


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found. Please install ffmpeg and ensure {name} is in your PATH.")
    return path


FFMPEG = _require_tool("ffmpeg")
FFPROBE = _require_tool("ffprobe")


PROBE_TIMEOUT: float | None = 10  # in seconds
PROXY_TRANSCODE_TIMEOUT: float | None = 300  # in seconds
AUDIO_EXTRACTION_TIMEOUT: float | None = 10  # in seconds


PROXY_WIDTH = 640
PROXY_PRESET = "ultrafast"
PROXY_CRF = 28

AUDIO_SAMPLE_RATE = 22050
AUDIO_CHANNELS = 1

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv"})

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
AMV_DIR = DATA_DIR / "amvs"
LIBRARY_DIR = DATA_DIR / "library"
PROXY_DIR = DATA_DIR / "proxies"
VIDEO_TABLE_PATH = DATA_DIR / "cache" / "videos.parquet"
