"""Tests for amv.io.ffmpeg.

Tests marked `integration` call the real ffmpeg/ffprobe binaries.
Run only the fast unit tests with:  pytest -m "not integration"

SPECIFICATION
  - probe(path) accepts str or PathLike and returns a MediaInfo.
  - Missing file -> FileNotFoundError (checked before calling ffprobe).
  - File ffprobe cannot read -> RuntimeError whose message mentions "ffprobe".
  - probe() has no side effects (writes no files).
  - Non-ASCII file names and metadata work. ffmpeg/ffprobe write UTF-8, so their
    output is decoded as UTF-8, not with the OS default encoding (cp1252 on Windows).
"""

import shutil
import subprocess
from fractions import Fraction

import pytest

from amv.io.ffmpeg import probe
from amv.io.media import MediaInfo


def test_probe_missing_file_raises(tmp_path):
    # tmp_path is a built-in pytest fixture: a fresh empty directory per test.
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "does_not_exist.mp4")


@pytest.mark.integration
def test_probe_non_media_file_raises(tmp_path):
    fake = tmp_path / "not_a_video.mp4"
    fake.write_text("this is not a video")

    with pytest.raises(RuntimeError, match="ffprobe"):
        probe(fake)


@pytest.mark.integration
def test_probe_accepts_str_path(make_media):
    path = make_media("clip.mp4", video=True, audio=True)

    info = probe(str(path))

    assert isinstance(info, MediaInfo)
    assert info.path == path


@pytest.mark.integration
def test_probe_audio_only_file(make_media):
    info = probe(make_media("song.mp3", video=False, audio=True))

    assert info.video is None
    assert info.audio is not None
    assert info.audio.sample_rate == 44100
    assert info.audio.duration == pytest.approx(2.0, abs=0.1)


@pytest.mark.integration
def test_probe_video_only_mkv(make_media):
    info = probe(make_media("episode.mkv", video=True, audio=False))

    assert info.audio is None
    assert info.video is not None
    assert info.video.fps == Fraction(24000, 1001)
    assert info.video.width == 320
    assert info.video.duration == pytest.approx(2.0, abs=0.1)


@pytest.mark.integration
def test_probe_mkv_with_audio(make_media):
    # Typical anime episode container: MKV with video + audio, no per-stream durations.
    info = probe(make_media("episode_av.mkv", video=True, audio=True))

    assert info.video is not None
    assert info.audio is not None
    assert info.audio.duration == pytest.approx(2.0, abs=0.1)


@pytest.mark.integration
def test_probe_writes_no_files(make_media, tmp_path):
    path = make_media("clip.mp4", video=True, audio=True)
    before = set(tmp_path.iterdir())

    probe(path)

    assert set(tmp_path.iterdir()) == before


# "э" is the UTF-8 bytes D1 8D, and 0x8D is undefined in cp1252 (Windows' default
# encoding): decoding ffprobe's UTF-8 output as cp1252 fails on exactly this byte.
NON_ASCII = "Синтэл 進撃 Pokémon"


@pytest.mark.integration
def test_probe_non_ascii_file_name(make_media):
    path = make_media(f"{NON_ASCII}.mp4", video=True, audio=True, duration=1.0)

    info = probe(path)

    assert info.video is not None and info.audio is not None


@pytest.mark.integration
def test_probe_non_ascii_metadata(make_media, tmp_path):
    # Real case: Sintel's mkv carries subtitle tracks in Russian, Vietnamese, ...
    src = make_media("plain.mkv", video=True, audio=True, duration=1.0)
    tagged = tmp_path / "tagged.mkv"
    subprocess.run(
        [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-c", "copy", "-metadata", f"title={NON_ASCII}", "-metadata:s:a:0", f"title={NON_ASCII}",
         str(tagged)],
        check=True,
    )  # fmt: skip

    info = probe(tagged)

    assert info.video is not None and info.audio is not None


@pytest.mark.integration
def test_probe_bluelock_amv(bluelock_amv):
    info = probe(bluelock_amv)

    assert info.video is not None
    assert info.video.width == 1920
    assert info.video.height == 1080
    assert info.video.fps == Fraction(30000, 1001)
    assert info.video.is_vfr is False
    assert info.video.duration == pytest.approx(132.93, abs=0.01)
    assert info.audio is not None
    assert info.audio.sample_rate == 44100


@pytest.mark.integration
def test_probe_is_repeatable(bluelock_amv):
    assert probe(bluelock_amv) == probe(bluelock_amv)
