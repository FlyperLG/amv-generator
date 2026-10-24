from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Self

VFR_TOLERANCE: float = 0.005


def _parse_rate(s: str | None) -> Fraction | None:
    """Converts string to fraction and returns None for missing, "0/0", zero, or garbage"""
    if s is None:
        return s

    try:
        fraction = Fraction(s)
        return fraction if fraction > 0 else None
    except (ZeroDivisionError, ValueError):
        return None


def _parse_duration(stream: dict[str, Any], fallback: float | None) -> float:
    duration = float(stream["duration"]) if stream.get("duration") else fallback
    if duration is None:
        raise ValueError("No duration or fallback_duration was provided.")
    return duration


@dataclass(frozen=True, slots=True)
class VideoInfo:
    duration: float  # seconds
    fps: Fraction
    is_vfr: bool
    width: int
    height: int
    codec: str
    n_frames: int | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            duration=data["duration"],
            fps=Fraction(data["fps"]),
            is_vfr=data["is_vfr"],
            width=data["width"],
            height=data["height"],
            codec=data["codec"],
            n_frames=data["n_frames"],
        )

    @classmethod
    def from_stream(cls, video_stream: dict[str, Any], fallback_duration: float | None = None) -> Self:
        """Create a VideoInfo instance from stream data."""

        try:
            duration = _parse_duration(video_stream, fallback_duration)
            fps = _parse_rate(video_stream.get("r_frame_rate"))
            if fps is None:
                raise ValueError(f"Invalid 'r_frame_rate' provided: {video_stream.get('r_frame_rate')}")
            avg = _parse_rate(video_stream.get("avg_frame_rate"))

            return cls(
                duration=duration,
                fps=fps,
                is_vfr=False if avg is None else abs(fps - avg) / fps > VFR_TOLERANCE,
                width=int(video_stream["width"]),
                height=int(video_stream["height"]),
                codec=video_stream["codec_name"],
                n_frames=int(video_stream["nb_frames"])
                if video_stream.get("nb_frames") is not None
                else None,
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid video stream: {e}") from e


@dataclass(frozen=True, slots=True)
class AudioInfo:
    duration: float  # seconds
    sample_rate: int
    channels: int
    codec: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            duration=data["duration"],
            sample_rate=data["sample_rate"],
            channels=data["channels"],
            codec=data["codec"],
        )

    @classmethod
    def from_stream(cls, audio_stream: dict[str, Any], fallback_duration: float | None = None) -> Self:
        """Create a AudioInfo instance from stream data."""
        try:
            duration = _parse_duration(audio_stream, fallback_duration)

            return cls(
                duration=duration,
                sample_rate=int(audio_stream["sample_rate"]),
                channels=int(audio_stream["channels"]),
                codec=audio_stream["codec_name"],
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid audio stream: {e}") from e


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: Path
    container_duration: float | None
    video: VideoInfo | None
    audio: AudioInfo | None

    @staticmethod
    def _first_stream(data: dict[str, Any], kind: str) -> dict[str, Any] | None:
        """Return the first stream of a given kind from ffprobe data."""
        for stream in data["streams"]:
            if stream.get("codec_type") == kind:
                if kind == "video" and stream.get("disposition", {}).get("attached_pic") == 1:
                    continue
                return stream
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            path=Path(data["path"]),
            container_duration=data["container_duration"],
            video=VideoInfo.from_dict(data["video"]) if data["video"] is not None else None,
            audio=AudioInfo.from_dict(data["audio"]) if data["audio"] is not None else None,
        )

    @classmethod
    def from_ffprobe(cls, path: Path, data: dict[str, Any]) -> Self:
        try:
            container_duration = float(data["format"]["duration"]) if data["format"].get("duration") else None

            video_stream = cls._first_stream(data, "video")
            video_info = (
                VideoInfo.from_stream(video_stream, fallback_duration=container_duration)
                if video_stream is not None
                else None
            )

            audio_stream = cls._first_stream(data, "audio")
            audio_info = (
                AudioInfo.from_stream(audio_stream, fallback_duration=container_duration)
                if audio_stream is not None
                else None
            )

            if video_info is None and audio_info is None:
                raise ValueError("Unsupported file received, no 'Video' or 'Audio' stream was found.")

            return cls(
                path=path,
                container_duration=container_duration,
                video=video_info,
                audio=audio_info,
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid ffprobe data for {path}: {e}") from e
