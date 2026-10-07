"""Unit tests for amv.io.media: VideoInfo, AudioInfo, MediaInfo.

These are pure parsing tests: no ffprobe, no video files, they run in milliseconds.
Inputs are recorded ffprobe outputs (tests/fixtures/*.json), edited per test.

SPECIFICATION (decisions these tests encode)
--------------------------------------------
Structure
  - MediaInfo describes the FILE: path, container_duration, video, audio.
  - VideoInfo / AudioInfo describe ONE STREAM each and know nothing about the path.
  - VideoInfo.from_stream(stream, fallback_duration=None)
    AudioInfo.from_stream(stream, fallback_duration=None)
    MediaInfo.from_ffprobe(path, data)

Stream selection
  - video = first stream with codec_type == "video" that is not cover art
    (disposition.attached_pic == 1 is skipped). None if there is none.
  - audio = first stream with codec_type == "audio". None if there is none.
  - A file with neither video nor audio is invalid -> ValueError.

Durations
  - container_duration = format.duration, or None if missing. Missing is NOT an error:
    format.duration exists for normal files (mp4, mkv, mp3), but can be absent for raw
    streams or broken/live-recorded files. Callers that need it check for None.
  - Stream duration = stream.duration, else the fallback (MediaInfo passes
    container_duration). MKV streams never have stream.duration, so the fallback is
    essential for anime episodes.
  - If neither exists -> ValueError. A stream without a duration is unusable for all
    later frame/time math, so this is where we fail loudly.

Video fields
  - fps = r_frame_rate (nominal rate) as Fraction.
  - is_vfr = relative difference between avg_frame_rate and r_frame_rate > 0.5 %.
    If avg_frame_rate is missing or "0/0", is_vfr = False (cannot tell; assume CFR).
  - r_frame_rate missing or "0/0" -> ValueError (fps is required).
  - n_frames = int(nb_frames) or None (MKV never has it).
  - width, height, codec required.

Audio fields
  - sample_rate, channels, codec required (ffprobe always reports them for audio).

Errors
  - Stream parsers raise ValueError only (never KeyError / ZeroDivisionError / TypeError).
  - MediaInfo.from_ffprobe re-raises with the file path in the message.
"""

import dataclasses
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from amv.io.media import AudioInfo, MediaInfo, VideoInfo

AMV_PATH = Path("data/amvs/bluelock.mp4")
SONG_PATH = Path("data/songs/song.mp3")
MKV_PATH = Path("data/library/episode.mkv")


def first_stream(data: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(s for s in data["streams"] if s["codec_type"] == kind)


# =============================================================================
# VideoInfo.from_stream
# =============================================================================


class TestVideoInfo:
    def test_bluelock_fields(self, bluelock_probe):
        info = VideoInfo.from_stream(first_stream(bluelock_probe, "video"))

        assert info.width == 1920
        assert info.height == 1080
        assert info.codec == "h264"
        assert info.n_frames == 4070
        assert info.duration == pytest.approx(135.840067)

    def test_fps_is_nominal_rate(self, bluelock_probe):
        info = VideoInfo.from_stream(first_stream(bluelock_probe, "video"))

        assert info.fps == Fraction(30000, 1001)

    def test_field_types(self, bluelock_probe):
        # Dataclasses do not enforce type hints at runtime, so check explicitly.
        info = VideoInfo.from_stream(first_stream(bluelock_probe, "video"))

        assert type(info.n_frames) is int
        assert type(info.width) is int
        assert type(info.height) is int
        assert type(info.duration) is float
        assert type(info.is_vfr) is bool
        assert isinstance(info.fps, Fraction)

    def test_has_no_audio_or_path_fields(self):
        names = {f.name for f in dataclasses.fields(VideoInfo)}

        assert "audio_sample_rate" not in names
        assert "path" not in names

    def test_is_immutable(self, bluelock_probe):
        info = VideoInfo.from_stream(first_stream(bluelock_probe, "video"))

        with pytest.raises(dataclasses.FrozenInstanceError):
            info.width = 1280  # type: ignore[misc]

    # --- duration -------------------------------------------------------------

    def test_stream_duration_wins_over_fallback(self, bluelock_probe):
        info = VideoInfo.from_stream(first_stream(bluelock_probe, "video"), fallback_duration=999.0)

        assert info.duration == pytest.approx(135.840067)

    def test_missing_duration_uses_fallback(self, bluelock_probe):
        stream = first_stream(bluelock_probe, "video")
        del stream["duration"]

        info = VideoInfo.from_stream(stream, fallback_duration=135.881905)

        assert info.duration == pytest.approx(135.881905)

    def test_missing_duration_without_fallback_raises(self, bluelock_probe):
        stream = first_stream(bluelock_probe, "video")
        del stream["duration"]

        with pytest.raises(ValueError, match="duration"):
            VideoInfo.from_stream(stream)

    def test_mkv_stream_with_fallback(self, mkv_probe):
        # Real MKV output: no stream duration, no nb_frames.
        info = VideoInfo.from_stream(first_stream(mkv_probe, "video"), fallback_duration=3.003)

        assert info.duration == pytest.approx(3.003)
        assert info.n_frames is None
        assert info.fps == Fraction(24000, 1001)

    # --- frame rate / VFR -----------------------------------------------------

    @pytest.mark.parametrize(
        ("r_frame_rate", "avg_frame_rate", "expected_vfr"),
        [
            pytest.param("30000/1001", "30000/1001", False, id="identical"),
            pytest.param("30000/1001", "61050000/2037601", False, id="one-frame-missing"),
            pytest.param("24000/1001", "23.95", False, id="small-drift-24p"),
            pytest.param("30/1", "25/1", True, id="real-vfr"),
            pytest.param("60/1", "30/1", True, id="half-rate"),
            pytest.param("30000/1001", "0/0", False, id="avg-unknown"),
        ],
    )
    def test_vfr_detection(self, bluelock_probe, r_frame_rate, avg_frame_rate, expected_vfr):
        stream = first_stream(bluelock_probe, "video")
        stream["r_frame_rate"] = r_frame_rate
        stream["avg_frame_rate"] = avg_frame_rate

        info = VideoInfo.from_stream(stream)

        assert info.is_vfr is expected_vfr

    def test_missing_avg_frame_rate_is_not_vfr(self, bluelock_probe):
        stream = first_stream(bluelock_probe, "video")
        del stream["avg_frame_rate"]

        info = VideoInfo.from_stream(stream)

        assert info.is_vfr is False

    @pytest.mark.parametrize("bad_rate", ["0/0", "0/1", "abc", ""])
    def test_invalid_r_frame_rate_raises_value_error(self, bluelock_probe, bad_rate):
        # A zero fps is as unusable as a missing one (and would divide by zero in is_vfr).
        stream = first_stream(bluelock_probe, "video")
        stream["r_frame_rate"] = bad_rate

        with pytest.raises(ValueError):
            VideoInfo.from_stream(stream)

    def test_garbage_duration_raises_value_error(self, bluelock_probe):
        stream = first_stream(bluelock_probe, "video")
        stream["duration"] = "N/A"

        with pytest.raises(ValueError):
            VideoInfo.from_stream(stream)

    # --- optional / required fields -------------------------------------------

    def test_missing_nb_frames_gives_none(self, bluelock_probe):
        stream = first_stream(bluelock_probe, "video")
        del stream["nb_frames"]

        assert VideoInfo.from_stream(stream).n_frames is None

    @pytest.mark.parametrize("missing_key", ["width", "height", "codec_name", "r_frame_rate"])
    def test_missing_required_field_raises_value_error(self, bluelock_probe, missing_key):
        stream = first_stream(bluelock_probe, "video")
        del stream[missing_key]

        with pytest.raises(ValueError):
            VideoInfo.from_stream(stream)

    def test_garbage_number_raises_value_error(self, bluelock_probe):
        stream = first_stream(bluelock_probe, "video")
        stream["width"] = "not-a-number"

        with pytest.raises(ValueError):
            VideoInfo.from_stream(stream)


# =============================================================================
# AudioInfo.from_stream
# =============================================================================


class TestAudioInfo:
    def test_bluelock_aac_fields(self, bluelock_probe):
        info = AudioInfo.from_stream(first_stream(bluelock_probe, "audio"))

        assert info.sample_rate == 44100
        assert info.channels == 2
        assert info.codec == "aac"
        assert info.duration == pytest.approx(135.881905)

    def test_mp3_fields(self, song_probe):
        info = AudioInfo.from_stream(first_stream(song_probe, "audio"))

        assert info.sample_rate == 44100
        assert info.channels == 2
        assert info.codec == "mp3"
        assert info.duration == pytest.approx(3.0)

    def test_field_types(self, bluelock_probe):
        # ffprobe reports sample_rate as a STRING ("44100") but channels as an int.
        info = AudioInfo.from_stream(first_stream(bluelock_probe, "audio"))

        assert type(info.sample_rate) is int
        assert type(info.channels) is int
        assert type(info.duration) is float

    def test_missing_duration_uses_fallback(self, bluelock_probe):
        stream = first_stream(bluelock_probe, "audio")
        del stream["duration"]

        info = AudioInfo.from_stream(stream, fallback_duration=135.881905)

        assert info.duration == pytest.approx(135.881905)

    def test_missing_duration_without_fallback_raises(self, bluelock_probe):
        stream = first_stream(bluelock_probe, "audio")
        del stream["duration"]

        with pytest.raises(ValueError, match="duration"):
            AudioInfo.from_stream(stream)

    @pytest.mark.parametrize("missing_key", ["sample_rate", "channels", "codec_name"])
    def test_missing_required_field_raises_value_error(self, bluelock_probe, missing_key):
        stream = first_stream(bluelock_probe, "audio")
        del stream[missing_key]

        with pytest.raises(ValueError):
            AudioInfo.from_stream(stream)


# =============================================================================
# MediaInfo.from_ffprobe
# =============================================================================


class TestMediaInfo:
    # --- the three real-world file shapes ------------------------------------

    def test_amv_has_video_and_audio(self, bluelock_probe):
        info = MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

        assert info.path == AMV_PATH
        assert info.video is not None
        assert info.audio is not None
        assert info.video.width == 1920
        assert info.audio.sample_rate == 44100

    def test_song_is_audio_only(self, song_probe):
        info = MediaInfo.from_ffprobe(SONG_PATH, song_probe)

        assert info.video is None
        assert info.audio is not None
        assert info.audio.codec == "mp3"

    def test_mkv_episode_is_video_only_and_uses_container_duration(self, mkv_probe):
        info = MediaInfo.from_ffprobe(MKV_PATH, mkv_probe)

        assert info.audio is None
        assert info.video is not None
        assert info.video.duration == pytest.approx(3.003)

    def test_mkv_episode_with_audio_uses_container_duration_for_both(self, mkv_av_probe):
        # Real MKV output: NEITHER stream has a duration, so both need the fallback.
        # This is the shape of a typical anime episode in the library.
        info = MediaInfo.from_ffprobe(MKV_PATH, mkv_av_probe)

        assert info.container_duration is not None
        assert info.video is not None
        assert info.audio is not None
        assert info.video.duration == pytest.approx(info.container_duration)
        assert info.audio.duration == pytest.approx(info.container_duration)

    # --- durations ------------------------------------------------------------

    def test_each_duration_comes_from_its_own_source(self, bluelock_probe):
        # Real file: video 135.840067, audio 135.881905, container 135.881905.
        info = MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

        assert info.container_duration == pytest.approx(135.881905)
        assert info.video is not None
        assert info.video.duration == pytest.approx(135.840067)
        assert info.audio is not None
        assert info.audio.duration == pytest.approx(135.881905)

    def test_container_duration_is_float(self, bluelock_probe):
        # ffprobe reports it as a string; it must be converted.
        info = MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

        assert type(info.container_duration) is float

    def test_missing_container_duration_gives_none(self, bluelock_probe):
        del bluelock_probe["format"]["duration"]

        info = MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

        assert info.container_duration is None
        assert info.video is not None
        assert info.video.duration == pytest.approx(135.840067)  # stream durations still work

    def test_no_duration_anywhere_raises_with_path(self, mkv_probe):
        del mkv_probe["format"]["duration"]  # MKV stream has none either

        with pytest.raises(ValueError, match="episode.mkv"):
            MediaInfo.from_ffprobe(MKV_PATH, mkv_probe)

    # --- stream selection -----------------------------------------------------

    def test_audio_stream_listed_first(self, bluelock_probe):
        bluelock_probe["streams"].reverse()

        info = MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

        assert info.video is not None
        assert info.video.width == 1920
        assert info.audio is not None
        assert info.audio.sample_rate == 44100

    def test_cover_art_is_not_a_video_stream(self, song_probe):
        # MP3s often carry album art, which ffprobe lists as a video stream.
        cover_art = {
            "index": 1,
            "codec_name": "mjpeg",
            "codec_type": "video",
            "width": 600,
            "height": 600,
            "r_frame_rate": "90000/1",
            "avg_frame_rate": "0/0",
            "disposition": {"default": 0, "attached_pic": 1},
        }
        song_probe["streams"].append(cover_art)

        info = MediaInfo.from_ffprobe(SONG_PATH, song_probe)

        assert info.video is None

    def test_cover_art_before_real_video_is_skipped(self, bluelock_probe):
        cover_art = {
            "index": 0,
            "codec_name": "mjpeg",
            "codec_type": "video",
            "width": 600,
            "height": 600,
            "r_frame_rate": "90000/1",
            "avg_frame_rate": "0/0",
            "disposition": {"default": 0, "attached_pic": 1},
        }
        bluelock_probe["streams"].insert(0, cover_art)

        info = MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

        assert info.video is not None
        assert info.video.width == 1920
        assert info.video.codec == "h264"

    def test_stream_without_disposition_is_accepted(self, bluelock_probe):
        # Some containers omit "disposition"; that must not crash stream selection.
        for s in bluelock_probe["streams"]:
            s.pop("disposition", None)

        info = MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

        assert info.video is not None

    def test_subtitle_streams_are_ignored(self, bluelock_probe):
        bluelock_probe["streams"].insert(0, {"index": 2, "codec_type": "subtitle", "codec_name": "ass"})

        info = MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

        assert info.video is not None
        assert info.video.width == 1920
        assert info.audio is not None
        assert info.audio.sample_rate == 44100

    def test_no_streams_raises_with_path(self, bluelock_probe):
        bluelock_probe["streams"] = []

        with pytest.raises(ValueError, match="bluelock.mp4"):
            MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

    def test_only_subtitles_raises_with_path(self, bluelock_probe):
        bluelock_probe["streams"] = [{"index": 0, "codec_type": "subtitle", "codec_name": "ass"}]

        with pytest.raises(ValueError, match="bluelock.mp4"):
            MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

    # --- error reporting ------------------------------------------------------

    def test_broken_video_stream_error_contains_path(self, bluelock_probe):
        del first_stream(bluelock_probe, "video")["width"]

        with pytest.raises(ValueError, match="bluelock.mp4"):
            MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

    def test_broken_audio_stream_error_contains_path(self, bluelock_probe):
        del first_stream(bluelock_probe, "audio")["sample_rate"]

        with pytest.raises(ValueError, match="bluelock.mp4"):
            MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

    def test_is_immutable(self, bluelock_probe):
        info = MediaInfo.from_ffprobe(AMV_PATH, bluelock_probe)

        with pytest.raises(dataclasses.FrozenInstanceError):
            info.path = Path("other.mp4")  # type: ignore[misc]
