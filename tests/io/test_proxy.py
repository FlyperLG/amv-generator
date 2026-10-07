"""Tests for amv.io.ffmpeg.run_proxy_transcode (docs/01-decoding-and-proxies, task 3.2).

These are black-box tests: they only call run_proxy_transcode() and inspect the files
it leaves behind (using probe(), which has its own tests). How the function decides
names, skips work or stores temp files is up to the implementation.

SPECIFICATION
  Signature
  - run_proxy_transcode(in_path, out_dir) -> Path
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

import re

import pytest

from amv.io.ffmpeg import probe, run_proxy_transcode

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

    proxy = run_proxy_transcode(src, out_dir)

    assert proxy.is_file()
    assert proxy.parent == out_dir
    assert proxy.suffix == ".mp4"


def test_accepts_str_paths(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False)

    proxy = run_proxy_transcode(str(src), str(out_dir))

    assert proxy.is_file()


def test_mkv_input_gives_mp4_proxy(make_media, out_dir):
    src = make_media("episode.mkv", video=True, audio=False)

    proxy = run_proxy_transcode(src, out_dir)

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

    video = probe(run_proxy_transcode(src, out_dir)).video

    assert (video.width, video.height) == expected


def test_height_is_even_when_exact_scaling_would_be_odd(make_media, out_dir):
    # 1000x564 scaled to 640 wide is 640x360.96; "-1" rounds that to 361, which libx264 rejects.
    src = make_media("clip.mp4", video=True, audio=False, size="1000x564", duration=1.0)

    video = probe(run_proxy_transcode(src, out_dir)).video

    assert video.width == 640
    assert video.height % 2 == 0
    assert abs(video.height - 360.96) <= 2


def test_keeps_frame_rate_and_duration(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False, duration=3.0)
    src_video = probe(src).video

    video = probe(run_proxy_transcode(src, out_dir)).video

    assert video.fps == src_video.fps
    assert video.is_vfr is False
    assert video.duration == pytest.approx(src_video.duration, abs=1 / float(src_video.fps))


def test_vfr_source_becomes_cfr(make_media, out_dir):
    src = make_media("vfr.mp4", video=True, audio=False, rate="24", vfr=True)
    assert probe(src).video.is_vfr is True  # sanity check of the test input itself

    video = probe(run_proxy_transcode(src, out_dir)).video

    assert video.is_vfr is False
    assert video.fps == 24


def test_proxy_has_no_audio(make_media, out_dir):
    src = make_media("amv.mp4", video=True, audio=True)

    info = probe(run_proxy_transcode(src, out_dir))

    assert info.video is not None
    assert info.audio is None


def test_source_is_not_modified(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False)
    before = (src.read_bytes(), src.stat().st_mtime_ns)

    run_proxy_transcode(src, out_dir)

    assert (src.read_bytes(), src.stat().st_mtime_ns) == before


# --- The proxy name -------------------------------------------------------------


def test_name_is_safe_and_hides_original_name(make_media, out_dir):
    src = make_media("Agony [AMV] (Blue Lock) ブルーロック.mp4", video=True, audio=False, duration=1.0)

    proxy = run_proxy_transcode(src, out_dir)

    assert re.fullmatch(r"[A-Za-z0-9._-]+", proxy.name)
    assert "Agony" not in proxy.name


def test_same_file_name_in_different_folders_gives_different_proxies(make_media, out_dir):
    a = make_media("Show A/Episode 01.mkv", video=True, audio=False, duration=1.0)
    b = make_media("Show B/Episode 01.mkv", video=True, audio=False, duration=2.0)

    proxy_a = run_proxy_transcode(a, out_dir)
    proxy_b = run_proxy_transcode(b, out_dir)

    assert proxy_a != proxy_b
    assert probe(proxy_a).video.duration == pytest.approx(1.0, abs=0.1)
    assert probe(proxy_b).video.duration == pytest.approx(2.0, abs=0.1)


# --- Skipping and re-doing work -------------------------------------------------


def test_second_call_reuses_existing_proxy(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False)
    first = run_proxy_transcode(src, out_dir)
    stamp = first.stat().st_mtime_ns
    files_after_first = set(out_dir.iterdir())

    second = run_proxy_transcode(src, out_dir)

    assert second == first
    assert second.stat().st_mtime_ns == stamp  # not re-encoded
    assert set(out_dir.iterdir()) == files_after_first


def test_changed_source_rebuilds_proxy(make_media, out_dir):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    run_proxy_transcode(src, out_dir)

    make_media("clip.mp4", video=True, audio=False, duration=3.0)  # overwrite the same source path
    proxy = run_proxy_transcode(src, out_dir)

    assert probe(proxy).video.duration == pytest.approx(3.0, abs=0.1)


# --- Failures -------------------------------------------------------------------


def test_missing_input_raises(tmp_path, out_dir):
    with pytest.raises(FileNotFoundError):
        run_proxy_transcode(tmp_path / "does_not_exist.mp4", out_dir)


def test_missing_out_dir_is_created(make_media, tmp_path):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    out_dir = tmp_path / "data" / "proxies"

    proxy = run_proxy_transcode(src, out_dir)

    assert proxy.parent == out_dir
    assert proxy.is_file()


def test_out_dir_that_is_a_file_raises(make_media, tmp_path):
    src = make_media("clip.mp4", video=True, audio=False, duration=1.0)
    not_a_dir = tmp_path / "proxies.txt"
    not_a_dir.write_text("")

    with pytest.raises(NotADirectoryError):
        run_proxy_transcode(src, not_a_dir)


def test_ffmpeg_failure_raises_and_leaves_nothing_behind(tmp_path, out_dir):
    fake = tmp_path / "not_a_video.mp4"
    fake.write_text("this is not a video")

    with pytest.raises(RuntimeError, match="ffmpeg"):
        run_proxy_transcode(fake, out_dir)

    assert list(out_dir.iterdir()) == []
