"""Builders for benchmark tests: small, valid MediaInfo objects without ffprobe."""

from fractions import Fraction
from pathlib import Path

from amv.io.media import MediaInfo, VideoInfo


def make_media_info(path: str = "data/amvs/clip.mp4", duration: float = 30.0, width: int = 1920) -> MediaInfo:
    """A video-only MediaInfo at 30000/1001 fps; enough for every benchmark calculation."""
    return MediaInfo(
        path=Path(path),
        container_duration=duration,
        video=VideoInfo(
            duration=duration,
            fps=Fraction(30000, 1001),
            is_vfr=False,
            width=width,
            height=width * 9 // 16,
            codec="h264",
            n_frames=round(duration * 30000 / 1001),
        ),
        audio=None,
    )
