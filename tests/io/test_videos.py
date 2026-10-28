"""Tests for amv.io.videos.build_video_table (docs/01-decoding-and-proxies, task 3.6).

Black-box tests: they lay out a tiny data/ tree of synthetic clips in tmp_path, call
build_video_table() and inspect the returned DataFrame, the parquet file and proxy_dir.
Every test runs the real ffmpeg, so clips are kept to 1 s.

SPECIFICATION
  build_video_table(amv_dir, library_dir, proxy_dir, out_path) -> pd.DataFrame
  - All four arguments accept str or PathLike. out_path is the parquet FILE
    (data/cache/videos.parquet), not a folder; its parent is created if missing.
  - Returns the table and writes the same table to out_path
    (pd.read_parquet(out_path) equals the returned DataFrame).

  Which files are videos
  - Every *.mp4 and *.mkv anywhere below amv_dir / library_dir, at any depth.
    Extensions match case-insensitively ("EP01.MKV" counts).
  - Everything else is ignored (.json, .txt, .wav, ...).
  - Row order is stable: all AMVs, then all library videos, each sorted by path.
  - Empty folders give an empty table that still has every column.
  - A missing amv_dir or library_dir raises FileNotFoundError. (Decision: rglob on a
    missing folder silently finds nothing, which hides a typo in a path.)

  One row per video, these columns (and no others)
    video_id           generate_file_id(original): the uuid prefix of the proxy names
    kind               "amv" or "library", from the folder the file was found in
    series             library: the FIRST folder below library_dir, at any depth
                       ("Blue Lock/Season 1/E01.mkv" -> "Blue Lock"); amv: None
    path               the original, resolved absolute path, as str
    name               the original file name, e.g. "S01E01.mkv"
    video_proxy_path   str, the path make_video_proxy returned; the file exists
    audio_proxy_path   amv: str, the path make_audio_proxy returned; library: None
    duration, is_vfr, width, height, codec
                       of the ORIGINAL (probe(original).video), not of the proxy
    fps                the original's nominal fps as an exact fraction string,
                       e.g. "24000/1001" (Fraction(row.fps) gives it back exactly)
    n_frames           the original's frame count; dtype "Int64" (nullable, mkv
                       often has none)
    proxy_width, proxy_height
                       size of the proxy file, so later steps can call iter_frames

  Rules
  - Library videos get no wav: no audio proxy is made for them, so proxy_dir holds
    only their .mp4 (+ sidecar). (AMVs need audio for steps 02/07, the library doesn't.)
  - A library video directly in library_dir (no series folder) raises ValueError
    naming the file, BEFORE any ffmpeg work (nothing is written to proxy_dir).
    (Decision: step 03 needs a series for every clip; failing at the end of a
    two-hour run would be worse.)
  - A .mp4/.mkv without a video stream raises ValueError naming the file
    (not AssertionError: asserts disappear under python -O).
  - Re-running is cheap: a second call with nothing changed runs no transcode
    (proxy files keep their mtime) and returns an equal table.
  - Progress reporting (tqdm bar, log messages) is up to the implementation and
    not tested. (Decision: it's presentation, not behaviour.)

  Importing amv.io.videos must not do any work: no code at module level besides
  definitions (put quick manual runs under `if __name__ == "__main__":`).
"""

from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from amv.io.ffmpeg import generate_file_id, probe
from amv.io.videos import build_video_table

pytestmark = pytest.mark.integration  # every test here runs the real ffmpeg

COLUMNS = {
    "video_id",
    "kind",
    "series",
    "path",
    "name",
    "video_proxy_path",
    "audio_proxy_path",
    "duration",
    "fps",
    "is_vfr",
    "width",
    "height",
    "n_frames",
    "codec",
    "proxy_width",
    "proxy_height",
}


@pytest.fixture
def dirs(tmp_path):
    """The data/ layout inside tmp_path. amvs/ and library/ exist; proxies/ and cache/ don't."""
    amv_dir, library_dir = tmp_path / "amvs", tmp_path / "library"
    amv_dir.mkdir()
    library_dir.mkdir()
    return {
        "amv_dir": amv_dir,
        "library_dir": library_dir,
        "proxy_dir": tmp_path / "proxies",
        "out_path": tmp_path / "cache" / "videos.parquet",
    }


@pytest.fixture
def amv(make_media):
    return make_media("amvs/My AMV.mp4", video=True, audio=True, duration=1.0, size="320x180")


@pytest.fixture
def episode(make_media):
    return make_media("library/Blue Lock/S01E01.mkv", video=True, audio=True, duration=1.0, size="480x360")


def row_of(table: pd.DataFrame, path: Path) -> pd.Series:
    rows = table[table["path"] == str(path.resolve())]
    assert len(rows) == 1, f"expected one row for {path}, got {len(rows)}"
    return rows.iloc[0]


# --- The table and the parquet file -----------------------------------------------


def test_has_exactly_the_specified_columns(dirs, amv, episode):
    table = build_video_table(**dirs)

    assert set(table.columns) == COLUMNS


def test_one_row_per_video(dirs, amv, episode):
    table = build_video_table(**dirs)

    assert len(table) == 2
    assert table["video_id"].is_unique


def test_writes_the_same_table_to_parquet(dirs, amv, episode):
    table = build_video_table(**dirs)

    assert dirs["out_path"].is_file()  # cache/ did not exist before
    pd.testing.assert_frame_equal(pd.read_parquet(dirs["out_path"]), table)


def test_accepts_str_paths(dirs, amv):
    table = build_video_table(**{k: str(v) for k, v in dirs.items()})

    assert len(table) == 1
    assert Path(dirs["out_path"]).is_file()


def test_empty_folders_give_an_empty_table_with_all_columns(dirs):
    table = build_video_table(**dirs)

    assert len(table) == 0
    assert set(table.columns) == COLUMNS
    assert dirs["out_path"].is_file()


@pytest.mark.parametrize("missing", ["amv_dir", "library_dir"])
def test_missing_input_folder_raises(dirs, missing):
    dirs[missing] = dirs[missing].parent / "typo"

    with pytest.raises(FileNotFoundError):
        build_video_table(**dirs)


# --- Which files are videos --------------------------------------------------------


def test_finds_videos_at_any_depth(dirs, make_media):
    make_media("amvs/sub/deeper/a.mp4", video=True, audio=True, duration=1.0)
    make_media("library/Show/Season 1/Disc 2/e.mkv", video=True, audio=False, duration=1.0)

    table = build_video_table(**dirs)

    assert sorted(table["name"]) == ["a.mp4", "e.mkv"]


def test_extensions_match_case_insensitively(dirs, make_media):
    make_media("library/Show/EP01.MKV", video=True, audio=False, duration=1.0)

    table = build_video_table(**dirs)

    assert list(table["name"]) == ["EP01.MKV"]


def test_ignores_non_video_files(dirs, amv):
    (dirs["amv_dir"] / "notes.txt").write_text("hi")
    (dirs["amv_dir"] / "My AMV.json").write_text("{}")
    (dirs["library_dir"] / "Blue Lock").mkdir()
    (dirs["library_dir"] / "Blue Lock" / "cover.jpg").write_bytes(b"\xff\xd8")

    table = build_video_table(**dirs)

    assert list(table["name"]) == ["My AMV.mp4"]


def test_row_order_is_amvs_then_library_each_sorted_by_path(dirs, make_media):
    for name in ["library/B/2.mkv", "amvs/z.mp4", "library/A/1.mkv", "amvs/a.mp4"]:
        make_media(name, video=True, audio=True, duration=1.0)

    table = build_video_table(**dirs)

    assert list(table["kind"]) == ["amv", "amv", "library", "library"]
    assert list(table["name"]) == ["a.mp4", "z.mp4", "1.mkv", "2.mkv"]


def test_row_order_is_sorted_by_path_across_extensions(dirs, make_media):
    for name in ["library/Show/E02.mp4", "library/Show/E01.mkv", "library/Show/E03.mkv"]:
        make_media(name, video=True, audio=False, duration=1.0)

    table = build_video_table(**dirs)

    assert list(table["name"]) == ["E01.mkv", "E02.mp4", "E03.mkv"]


# --- The columns -------------------------------------------------------------------


def test_kind_comes_from_the_folder(dirs, amv, episode):
    table = build_video_table(**dirs)

    assert row_of(table, amv)["kind"] == "amv"
    assert row_of(table, episode)["kind"] == "library"


def test_video_id_matches_the_proxy_names(dirs, amv, episode):
    table = build_video_table(**dirs)

    for src in (amv, episode):
        row = row_of(table, src)
        assert row["video_id"] == generate_file_id(src)
        assert Path(row["video_proxy_path"]).name.startswith(row["video_id"] + "_")


def test_path_is_resolved_and_name_is_the_file_name(dirs, amv, monkeypatch):
    monkeypatch.chdir(dirs["amv_dir"].parent)  # so relative folder paths are valid

    table = build_video_table(
        amv_dir=Path("amvs"),
        library_dir=Path("library"),
        proxy_dir=Path("proxies"),
        out_path=Path("cache/videos.parquet"),
    )

    row = table.iloc[0]
    assert row["path"] == str(amv.resolve())
    assert Path(row["path"]).is_absolute()
    assert row["name"] == "My AMV.mp4"


@pytest.mark.parametrize(
    ("relative", "series"),
    [
        ("library/Blue Lock/S01E01.mkv", "Blue Lock"),
        ("library/Blue Lock/Season 1/S01E01.mkv", "Blue Lock"),  # first folder, not the parent
    ],
)
def test_series_is_the_first_folder_below_library(dirs, make_media, relative, series):
    src = make_media(relative, video=True, audio=False, duration=1.0)

    table = build_video_table(**dirs)

    assert row_of(table, src)["series"] == series


def test_amvs_have_no_series(dirs, make_media):
    src = make_media("amvs/Blue Lock/edit.mp4", video=True, audio=True, duration=1.0)

    table = build_video_table(**dirs)

    assert row_of(table, src)["series"] is None


def test_metadata_describes_the_original_not_the_proxy(dirs, episode):
    table = build_video_table(**dirs)

    row = row_of(table, episode)
    original = probe(episode).video
    assert (row["width"], row["height"]) == (480, 360)
    assert row["duration"] == pytest.approx(original.duration)
    assert row["codec"] == original.codec
    assert bool(row["is_vfr"]) is False


def test_is_vfr_is_the_originals(dirs, make_media):
    src = make_media("amvs/vfr.mp4", video=True, audio=True, duration=2.0, vfr=True)

    table = build_video_table(**dirs)

    assert bool(row_of(table, src)["is_vfr"]) is True  # the proxy itself is CFR


def test_fps_is_an_exact_fraction_string(dirs, amv):
    table = build_video_table(**dirs)

    fps = row_of(table, amv)["fps"]
    assert isinstance(fps, str)
    assert Fraction(fps) == Fraction(24000, 1001)


def test_n_frames_is_a_nullable_int(dirs, amv, episode):
    table = build_video_table(**dirs)

    assert table["n_frames"].dtype == "Int64"
    assert pd.read_parquet(dirs["out_path"])["n_frames"].dtype == "Int64"
    n = row_of(table, amv)["n_frames"]
    assert pd.isna(n) or n == pytest.approx(24, abs=1)  # 1 s at 23.976 fps


def test_proxy_size_matches_the_proxy_file(dirs, amv, episode):
    table = build_video_table(**dirs)

    for src, expected in ((amv, (640, 360)), (episode, (640, 480))):
        row = row_of(table, src)
        proxy = probe(row["video_proxy_path"]).video
        assert (row["proxy_width"], row["proxy_height"]) == (proxy.width, proxy.height) == expected


# --- Proxies and audio -------------------------------------------------------------


def test_proxy_paths_exist_and_live_in_proxy_dir(dirs, amv, episode):
    table = build_video_table(**dirs)

    for src in (amv, episode):
        path = Path(row_of(table, src)["video_proxy_path"])
        assert path.is_file()
        assert path.parent.resolve() == dirs["proxy_dir"].resolve()


def test_amvs_get_a_wav(dirs, amv):
    table = build_video_table(**dirs)

    wav = row_of(table, amv)["audio_proxy_path"]
    assert isinstance(wav, str)
    assert Path(wav).is_file() and Path(wav).suffix == ".wav"


def test_library_videos_get_no_wav(dirs, episode):
    table = build_video_table(**dirs)

    assert row_of(table, episode)["audio_proxy_path"] is None
    assert list(dirs["proxy_dir"].glob("*.wav")) == []  # not made and then ignored


# --- Failures ----------------------------------------------------------------------


def test_library_video_without_series_folder_raises_before_any_work(dirs, amv, make_media):
    make_media("library/loose.mkv", video=True, audio=False, duration=1.0)

    with pytest.raises(ValueError, match="loose.mkv"):
        build_video_table(**dirs)

    proxy_dir = dirs["proxy_dir"]
    assert not proxy_dir.exists() or list(proxy_dir.iterdir()) == []  # not even the AMV was done


def test_file_without_video_stream_raises(dirs, make_media):
    make_media("amvs/audio only.mp4", video=False, audio=True, duration=1.0)

    with pytest.raises(ValueError, match="audio only.mp4"):
        build_video_table(**dirs)


# --- Re-running -----------------------------------------------------------------


def test_second_run_transcodes_nothing_and_gives_the_same_table(dirs, amv, episode):
    first = build_video_table(**dirs)
    stamps = {p: p.stat().st_mtime_ns for p in dirs["proxy_dir"].iterdir()}

    second = build_video_table(**dirs)

    assert {p: p.stat().st_mtime_ns for p in dirs["proxy_dir"].iterdir()} == stamps
    pd.testing.assert_frame_equal(second, first)
