import shutil

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure ffmpeg is in your PATH.")

FFPROBE = shutil.which("ffprobe")
if FFPROBE is None:
    raise RuntimeError("ffprobe not found. Please install ffmpeg and ensure ffprobe is in your PATH.")


PROBE_TIMEOUT: float | None = 10  # in seconds
PROXY_TRANSCODE_TIMEOUT: float | None = 300  # in seconds
