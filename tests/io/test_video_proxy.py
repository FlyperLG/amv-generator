"""Tests for amv.io.ffmpeg.make_video_proxy / video_proxy_path (docs/01-decoding-and-proxies, 3.2 + 3.4).

These are black-box tests: they only call make_video_proxy() and inspect the files
it leaves behind (using probe(), which has its own tests). How the function decides
names, skips work or stores temp files is up to the implementation.

SPECIFICATION
  Signature
  - make_video_proxy(in_path, out_dir) -> Path
    Both arguments accept str or PathLike. Returns the path of the proxy file, which
    lies directly inside out_dir.

  The proxy file
  - H.264 video in an .mp4 container, whatever the input container (.mkv, .mp4, ...).
  - Exactly 640 px wide. Height keeps the source aspect ratio and is always even
    (H.264 rejects odd heights). E.g. 1920x1080 -> 640x360, 960x720 -> 640x480.
  - Constant frame rate (probe(...).video.is_vfr is False), even for VFR sources,
    at the source's nominal frame rate (r_frame_rate).
  - Duration equals the source duration within one frame.
  - No audio stream. (Decision: analysis reads audio from the separate wav of task
    3.4, so encoding audio into every proxy is wasted CPU time.)

  The proxy name
  - Deterministic: the same source path always maps to the same proxy path.
  - Does not contain the original file name and uses only [A-Za-z0-9._-].
  - Two different sources with the same file name in different folders
    (e.g. "Show A/Episode 01.mkv" and "Show B/Episode 01.mkv") get different proxies.
  - Format "<video_id>_<settings_id>.mp4", both UUIDs. Changing a PROXY_* setting gives
    a new file and keeps the old one. (Same rules as the audio proxy, see
    test_audio_proxy.py.)

  Sidecar: "<same name>.json" next to the proxy
  - A JSON object with at least "source" (resolved source path, str), "task" (str)
    and "ffmpeg_args" (list of str).

  video_proxy_path(in_path, out_dir) -> Path
  - The path make_video_proxy returns for the current settings, computed without
    running ffmpeg or creating anything (not even out_dir).

  Skipping and re-doing work
  - If an up-to-date proxy already exists, the call returns its path without
    re-encoding it (the proxy file is left untouched).
  - If the source changed since the proxy was made (different size or mtime), the
    proxy is rebuilt.

  Failures
  - Missing input file -> FileNotFoundError, raised before ffmpeg is called.
  - out_dir does not exist -> it is created (including parents).
  - out_dir exists but is a file -> NotADirectoryError.
  - ffmpeg fails (e.g. input is not a video) -> RuntimeError whose message mentions
    "ffmpeg". No proxy and no half-written/temp file is left in out_dir.

  Not covered by tests, but required
  - No fixed timeout that a 24-minute episode could hit (a transcode takes minutes).
  - A crash or Ctrl+C mid-transcode never leaves a file at the final proxy path
    (write to a temp name, then rename).
  - The source file is never modified.
"""

import json
import re

import pytest

import amv.io.ffmpeg as ffmpeg_module
from amv.io.ffmpeg import make_video_proxy, probe, video_proxy_path

pytestmark = pytest.mark.integration  # every test here runs the real ffmpeg


@pytest.fixture
def out_dir(tmp_path):
    """Proxy output folder, kept apart from the source files make_media writes into tmp_path."""
    d = tmp_path / "proxies"
    d.mkdir()
    return d


# --- The proxy file -------------------------------------------------------------


def test_returns_existing_mp4_inside_out_dir(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False)

    proxy = make_video_proxy(src, out_dir)

    assert proxy.is_file()
    assert proxy.parent == out_dir
    assert proxy.suffix == ".mp4"


def test_accepts_str_paths(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False)

    proxy = make_video_proxy(str(src), str(out_dir))

    assert proxy.is_file()


def test_mkv_input_gives_mp4_proxy(make_media, out_dir):
    src = make_media("episode.mkv", video=True, audio=False)

    proxy = make_video_proxy(src, out_dir)

    assert proxy.suffix == ".mp4"
    assert probe(proxy).video.codec == "h264"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1280x720", (640, 360)),  # 16:9 downscale
        ("960x720", (640, 480)),  # 4:3
        ("320x180", (640, 360)),  # small sources are scaled to 640 too
    ],
)
def test_scales_to_640_wide_keeping_aspect(make_media, out_dir, size, expected):
    src = make_media("clip.mp4", video=True, audio=False, size=size, duration=1.0)

    video = probe(make_video_proxy(src, out_dir)).video

    assert (video.width, video.height) == expected


def test_height_is_even_when_exact_scaling_would_be_odd(make_media, out_dir):
    # 1000x564 scaled to 640 wide is 640x360.96; "-1" rounds that to 361, which libx264 rejects.
    src = make_media("clip.mp4", video=True, audio=False, size="1000x564", duration=1.0)

    video = probe(make_video_proxy(src, out_dir)).video

    assert video.width == 640
    assert video.height % 2 == 0
    assert abs(video.height - 360.96) <= 2


def test_keeps_frame_rate_and_duration(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False, duration=3.0)
    src_video = probe(src).video

    video = probe(make_video_proxy(src, out_dir)).video

    assert video.fps == src_video.fps
    assert video.is_vfr is False
    assert video.duration == pytest.approx(src_video.duration, abs=1 / float(src_video.fps))


def test_vfr_source_becomes_cfr(make_media, out_dir):
    src = make_media("vfr.mp4", video=True, audio=False, rate="24", vfr=True)
    assert probe(src).video.is_vfr is True  # sanity check of the test input itself

    video = probe(make_video_proxy(src, out_dir)).video

    assert video.is_vfr is False
    assert video.fps == 24


def test_proxy_has_no_audio(make_media, out_dir):
    src = make_media("amv.mp4", video=True, audio=True)

    info = probe(make_video_proxy(src, out_dir))

    assert info.video is not None
    assert info.audio is None


def test_source_is_not_modified(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False)
    before = (src.read_bytes(), src.stat().st_mtime_ns)

    make_video_proxy(src, out_dir)

    assert (src.read_bytes(), src.stat().st_mtime_ns) == before


# --- The proxy name -------------------------------------------------------------


def test_name_is_safe_and_hides_original_name(make_media, out_dir):
    src = make_media("Agony [AMV] (Blue Lock) ブルーロック.mp4", video=True, audio=False, duration=1.0)

    proxy = make_video_proxy(src, out_dir)

    assert re.fullmatch(r"[A-Za-z0-9._-]+", proxy.name)
    assert "Agony" not in proxy.name


def test_same_file_name_in_different_folders_gives_different_proxies(make_media, out_dir):
    a = make_media("Show A/Episode 01.mkv", video=True, audio=False, duration=1.0)
    b = make_media("Show B/Episode 01.mkv", video=True, audio=False, duration=2.0)

    proxy_a = make_video_proxy(a, out_dir)
    proxy_b = make_video_proxy(b, out_dir)

    assert proxy_a != proxy_b
    assert probe(proxy_a).video.duration == pytest.approx(1.0, abs=0.1)
    assert probe(proxy_b).video.duration == pytest.approx(2.0, abs=0.1)


# --- Skipping and re-doing work -------------------------------------------------


def test_second_call_reuses_existing_proxy(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False)
    first = make_video_proxy(src, out_dir)
    stamp = first.stat().st_mtime_ns
    files_after_first = set(out_dir.iterdir())

    second = make_video_proxy(src, out_dir)

    assert second == first
    assert second.stat().st_mtime_ns == stamp  # not re-encoded
    assert set(out_dir.iterdir()) == files_after_first


def test_changed_source_rebuilds_proxy(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    make_video_proxy(src, out_dir)

    make_media("clip.mp4", video=True, audio=False, duration=3.0)  # overwrite the same source path
    proxy = make_video_proxy(src, out_dir)

    assert probe(proxy).video.duration == pytest.approx(3.0, abs=0.1)


# --- Failures -------------------------------------------------------------------


def test_missing_input_raises(tmp_path, out_dir):
    with pytest.raises(FileNotFoundError):
        make_video_proxy(tmp_path / "does_not_exist.mp4", out_dir)


def test_missing_out_dir_is_created(make_media, tmp_path):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    out_dir = tmp_path / "data" / "proxies"

    proxy = make_video_proxy(src, out_dir)

    assert proxy.parent == out_dir
    assert proxy.is_file()


def test_out_dir_that_is_a_file_raises(make_media, tmp_path):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    not_a_dir = tmp_path / "proxies.txt"
    not_a_dir.write_text("")

    with pytest.raises(NotADirectoryError):
        make_video_proxy(src, not_a_dir)


def test_ffmpeg_failure_raises_and_leaves_nothing_behind(tmp_path, out_dir):
    fake = tmp_path / "not_a_video.mp4"
    fake.write_text("this is not a video")

    with pytest.raises(RuntimeError, match="ffmpeg"):
        make_video_proxy(fake, out_dir)

    assert list(out_dir.iterdir()) == []


# --- Name, settings and sidecar (task 3.4) --------------------------------------

UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def test_name_is_video_id_and_settings_id(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)

    proxy = make_video_proxy(src, out_dir)

    assert re.fullmatch(f"{UUID}_{UUID}\.mp4", proxy.name)


def test_changed_settings_give_a_new_file_and_keep_the_old(make_media, out_dir, monkeypatch):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    old = make_video_proxy(src, out_dir)

    monkeypatch.setattr(ffmpeg_module, "PROXY_WIDTH", 320)
    new = make_video_proxy(src, out_dir)

    assert new != old
    assert new.name.split("_")[0] == old.name.split("_")[0]  # same video_id
    assert probe(new).video.width == 320
    assert old.is_file() and probe(old).video.width == 640


def test_sidecar_describes_the_proxy(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)

    proxy = make_video_proxy(src, out_dir)

    meta = json.loads(proxy.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["source"] == str(src.resolve())
    assert isinstance(meta["task"], str) and meta["task"]
    assert isinstance(meta["ffmpeg_args"], list)
    assert any("640" in arg for arg in meta["ffmpeg_args"])  # e.g. "scale=640:-2"


def test_sidecar_source_is_resolved_for_relative_paths(make_media, out_dir, monkeypatch):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    monkeypatch.chdir(src.parent)  # so "clip.mp4" is a valid relative path

    proxy = make_video_proxy("clip.mp4", out_dir)

    meta = json.loads(proxy.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["source"] == str(src.resolve())


def test_out_dir_holds_only_proxy_and_sidecar(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)

    proxy = make_video_proxy(src, out_dir)

    assert set(out_dir.iterdir()) == {proxy, proxy.with_suffix(".json")}  # no temp file left


def test_path_matches_make_video_proxy(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)

    predicted = video_proxy_path(src, out_dir)

    assert make_video_proxy(src, out_dir) == predicted


def test_path_creates_nothing(make_media, tmp_path):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    out_dir = tmp_path / "not_created_yet"

    path = video_proxy_path(src, out_dir)

    assert path.parent == out_dir
    assert not out_dir.exists()


def test_path_follows_the_settings(make_media, out_dir, monkeypatch):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    before = video_proxy_path(src, out_dir)

    monkeypatch.setattr(ffmpeg_module, "PROXY_WIDTH", 320)

    assert video_proxy_path(src, out_dir) != before
    assert video_proxy_path(str(src), str(out_dir)) == video_proxy_path(src, out_dir)
