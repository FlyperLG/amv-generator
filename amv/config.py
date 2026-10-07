import shutil

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure ffmpeg is in your PATH.")

FFPROBE = shutil.which("ffprobe")
if FFPROBE is None:
    raise RuntimeError("ffprobe not found. Please install ffmpeg and ensure ffprobe is in your PATH.")
