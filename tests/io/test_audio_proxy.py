"""Tests for amv.io.ffmpeg.make_audio_proxy / audio_proxy_path (docs/01-decoding-and-proxies, task 3.4).

Black-box tests: they call the functions and inspect the files left in out_dir. The wav
is read with the standard library's `wave` module, which only understands plain PCM,
so reading it at all already checks the format.

SPECIFICATION
  make_audio_proxy(in_path, out_dir) -> Path
  - Both arguments accept str or PathLike. Returns the path of a .wav file directly
    inside out_dir. Works for videos with audio (.mp4, .mkv) and audio files (.mp3).

  The wav file
  - 16-bit PCM, AUDIO_SAMPLE_RATE Hz (22050), AUDIO_CHANNELS channels (1): stereo
    sources are mixed down to mono.
  - Same duration as the source audio (within 0.1 s; mp3 adds a little padding).
  - Contains the source audio, not silence (a 440 Hz tone stays a 440 Hz tone).

  The name: "<video_id>_<settings_id>.wav"
  - Both parts are UUIDs. <video_id> depends only on the resolved source path, so the
    audio proxy and the video proxy of the same file share it. <settings_id> depends
    on the settings, so changing AUDIO_SAMPLE_RATE gives a new file and keeps the old one.

  Sidecar: "<same name>.json" next to the wav
  - Written after a successful run. A JSON object with at least
      "source":      the resolved source path, as a str
      "task":        a short name of the conversion, a str
      "ffmpeg_args": the list of ffmpeg arguments the settings_id was made from
  - Lets a human find out what a hash means without recomputing it.

  audio_proxy_path(in_path, out_dir) -> Path
  - Returns the path make_audio_proxy would return (or did return) for the current
    settings, without running ffmpeg and without creating anything: no file, no
    sidecar, not even out_dir. (This is the "path split": the name computation of
    _ffmpeg_to_file in its own function, which _ffmpeg_to_file then calls. 3.6 and a
    cleanup script use it to ask "is this done already?" cheaply.)
    The same function for the video side, video_proxy_path, follows the same rules.

  Skipping and re-doing work
  - An up-to-date wav is returned without re-running ffmpeg (file left untouched).
  - If the source changed since the wav was made, it is rebuilt.

  Failures
  - Missing input -> FileNotFoundError.
  - out_dir is a file -> NotADirectoryError; a missing out_dir is created.
  - ffmpeg fails, e.g. a video without an audio stream -> RuntimeError mentioning
    "ffmpeg". Nothing is left in out_dir: no wav, no temp file, no sidecar.
"""

import json
import re
import wave

import numpy as np
import pytest

import amv.io.ffmpeg as ffmpeg_module
from amv.io.ffmpeg import audio_proxy_path, make_audio_proxy, make_video_proxy

pytestmark = pytest.mark.integration  # every test here runs the real ffmpeg

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


@pytest.fixture
def out_dir(tmp_path):
    """Output folder, kept apart from the source files make_media writes into tmp_path."""
    d = tmp_path / "proxies"
    d.mkdir()
    return d


def read_wav(path):
    """(sample_rate, channels, sample_width_bytes, samples as int16 array)."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return rate, channels, width, samples


def dominant_frequency(samples, rate):
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    return np.fft.rfftfreq(len(samples), 1 / rate)[spectrum.argmax()]


# --- The wav file ---------------------------------------------------------------


def test_returns_wav_inside_out_dir(make_media, out_dir):
    src = make_media("amv.mp4", video=True, audio=True)

    wav = make_audio_proxy(src, out_dir)

    assert wav.is_file()
    assert wav.parent == out_dir
    assert wav.suffix == ".wav"


def test_accepts_str_paths(make_media, out_dir):
    src = make_media("amv.mp4", video=True, audio=True)

    assert make_audio_proxy(str(src), str(out_dir)).is_file()


@pytest.mark.parametrize("name", ["amv.mp4", "episode.mkv", "song.mp3"])
def test_format_is_22050_hz_mono_16_bit(make_media, out_dir, name):
    src = make_media(name, video=not name.endswith(".mp3"), audio=True)

    rate, channels, width, _ = read_wav(make_audio_proxy(src, out_dir))

    assert (rate, channels, width) == (22050, 1, 2)


def test_stereo_is_mixed_down_to_mono(make_media, out_dir):
    src = make_media("stereo.mp3", video=False, audio=True, channels=2)

    _, channels, _, _ = read_wav(make_audio_proxy(src, out_dir))

    assert channels == 1


@pytest.mark.parametrize(("name", "duration"), [("amv.mp4", 2.0), ("song.mp3", 3.0)])
def test_duration_matches_source(make_media, out_dir, name, duration):
    src = make_media(name, video=name.endswith(".mp4"), audio=True, duration=duration)

    rate, _, _, samples = read_wav(make_audio_proxy(src, out_dir))

    assert len(samples) / rate == pytest.approx(duration, abs=0.1)


def test_audio_content_is_kept(make_media, out_dir):
    src = make_media("song.mp3", video=False, audio=True)  # 440 Hz sine

    rate, _, _, samples = read_wav(make_audio_proxy(src, out_dir))

    assert np.abs(samples).max() > 1000  # not silence
    assert dominant_frequency(samples, rate) == pytest.approx(440, abs=5)


def test_source_is_not_modified(make_media, out_dir):
    src = make_media("amv.mp4", video=True, audio=True)
    before = (src.read_bytes(), src.stat().st_mtime_ns)

    make_audio_proxy(src, out_dir)

    assert (src.read_bytes(), src.stat().st_mtime_ns) == before


# --- The name -------------------------------------------------------------------


def test_name_is_video_id_and_settings_id(make_media, out_dir):
    src = make_media("Agony [AMV] (Blue Lock).mp4", video=True, audio=True)

    wav = make_audio_proxy(src, out_dir)

    assert re.fullmatch(f"{UUID}_{UUID}\\.wav", wav.name)


def test_shares_video_id_with_the_video_proxy(make_media, out_dir):
    src = make_media("amv.mp4", video=True, audio=True, duration=1.0)

    wav = make_audio_proxy(src, out_dir)
    mp4 = make_video_proxy(src, out_dir)

    assert wav.name.split("_")[0] == mp4.name.split("_")[0]
    assert wav.stem != mp4.stem  # different settings -> different settings_id


def test_same_file_name_in_different_folders_gives_different_wavs(make_media, out_dir):
    a = make_media("Show A/opening.mp3", video=False, audio=True, duration=1.0)
    b = make_media("Show B/opening.mp3", video=False, audio=True, duration=2.0)

    assert make_audio_proxy(a, out_dir) != make_audio_proxy(b, out_dir)


def test_changed_settings_give_a_new_file_and_keep_the_old(make_media, out_dir, monkeypatch):
    src = make_media("song.mp3", video=False, audio=True, duration=1.0)
    old = make_audio_proxy(src, out_dir)

    monkeypatch.setattr(ffmpeg_module, "AUDIO_SAMPLE_RATE", 16000)
    new = make_audio_proxy(src, out_dir)

    assert new != old
    assert new.name.split("_")[0] == old.name.split("_")[0]  # same video_id
    assert read_wav(new)[0] == 16000
    assert old.is_file() and read_wav(old)[0] == 22050


# --- Sidecar --------------------------------------------------------------------


def test_sidecar_describes_the_wav(make_media, out_dir):
    src = make_media("amv.mp4", video=True, audio=True, duration=1.0)

    wav = make_audio_proxy(src, out_dir)
    sidecar = wav.with_suffix(".json")

    assert sidecar.is_file()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["source"] == str(src.resolve())
    assert isinstance(meta["task"], str) and meta["task"]
    assert isinstance(meta["ffmpeg_args"], list)
    assert "22050" in meta["ffmpeg_args"]


def test_sidecar_source_is_resolved_for_relative_paths(make_media, out_dir, monkeypatch):
    src = make_media("song.mp3", video=False, audio=True, duration=1.0)
    monkeypatch.chdir(src.parent)  # so "song.mp3" is a valid relative path

    wav = make_audio_proxy("song.mp3", out_dir)

    meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["source"] == str(src.resolve())


def test_out_dir_holds_only_wav_and_sidecar(make_media, out_dir):
    src = make_media("amv.mp4", video=True, audio=True, duration=1.0)

    wav = make_audio_proxy(src, out_dir)

    assert set(out_dir.iterdir()) == {wav, wav.with_suffix(".json")}  # no temp file left


# --- audio_proxy_path -----------------------------------------------------------


def test_path_matches_make_audio_proxy(make_media, out_dir):
    src = make_media("amv.mp4", video=True, audio=True, duration=1.0)

    predicted = audio_proxy_path(src, out_dir)

    assert make_audio_proxy(src, out_dir) == predicted


def test_path_creates_nothing(make_media, tmp_path):
    src = make_media("amv.mp4", video=True, audio=True, duration=1.0)
    out_dir = tmp_path / "not_created_yet"

    path = audio_proxy_path(src, out_dir)

    assert path.parent == out_dir
    assert not out_dir.exists()


def test_path_follows_the_settings(make_media, out_dir, monkeypatch):
    src = make_media("song.mp3", video=False, audio=True, duration=1.0)
    before = audio_proxy_path(src, out_dir)

    monkeypatch.setattr(ffmpeg_module, "AUDIO_SAMPLE_RATE", 16000)

    assert audio_proxy_path(src, out_dir) != before
    assert audio_proxy_path(str(src), str(out_dir)) == audio_proxy_path(src, out_dir)


# --- Skipping and re-doing work -------------------------------------------------


def test_second_call_reuses_existing_wav(make_media, out_dir):
    src = make_media("song.mp3", video=False, audio=True)
    first = make_audio_proxy(src, out_dir)
    stamp = first.stat().st_mtime_ns
    files_after_first = set(out_dir.iterdir())

    second = make_audio_proxy(src, out_dir)

    assert second == first
    assert second.stat().st_mtime_ns == stamp  # ffmpeg did not run again
    assert set(out_dir.iterdir()) == files_after_first


def test_changed_source_rebuilds_wav(make_media, out_dir):
    src = make_media("song.mp3", video=False, audio=True, duration=1.0)
    make_audio_proxy(src, out_dir)

    make_media("song.mp3", video=False, audio=True, duration=3.0)  # overwrite the same source path
    rate, _, _, samples = read_wav(make_audio_proxy(src, out_dir))

    assert len(samples) / rate == pytest.approx(3.0, abs=0.1)


# --- Failures -------------------------------------------------------------------


def test_missing_input_raises(tmp_path, out_dir):
    with pytest.raises(FileNotFoundError):
        make_audio_proxy(tmp_path / "does_not_exist.mp4", out_dir)


def test_missing_out_dir_is_created(make_media, tmp_path):
    src = make_media("song.mp3", video=False, audio=True, duration=1.0)
    out_dir = tmp_path / "data" / "proxies"

    wav = make_audio_proxy(src, out_dir)

    assert wav.parent == out_dir and wav.is_file()


def test_out_dir_that_is_a_file_raises(make_media, tmp_path):
    src = make_media("song.mp3", video=False, audio=True, duration=1.0)
    not_a_dir = tmp_path / "proxies.txt"
    not_a_dir.write_text("")

    with pytest.raises(NotADirectoryError):
        make_audio_proxy(src, not_a_dir)


def test_video_without_audio_raises_and_leaves_nothing(make_media, out_dir):
    src = make_media("episode.mkv", video=True, audio=False)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        make_audio_proxy(src, out_dir)

    assert list(out_dir.iterdir()) == []


def test_not_a_media_file_raises_and_leaves_nothing(tmp_path, out_dir):
    fake = tmp_path / "not_audio.mp3"
    fake.write_text("this is not audio")

    with pytest.raises(RuntimeError, match="ffmpeg"):
        make_audio_proxy(fake, out_dir)

    assert list(out_dir.iterdir()) == []
