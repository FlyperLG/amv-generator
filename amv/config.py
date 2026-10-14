import shutil


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
