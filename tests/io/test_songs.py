"""Tests for amv.io.songs.build_song_table (docs/01-decoding-and-proxies, task 3.7).

The song table is to steps 02/07 what videos.parquet is to steps 03/04: the one list
of audio inputs, and the owner of the audio proxies. Step 02 iterates over its rows and never
lists proxy_dir itself (the wav names are hashes, stale versions stay behind after a
settings change, and they don't say whether a wav came from an AMV or a song).

Black-box tests: they lay out a tiny data/ tree in tmp_path, call build_song_table()
and inspect the returned DataFrame, the parquet file and proxy_dir.

SPECIFICATION
  build_song_table(amv_dir, song_dir, proxy_dir, out_path, settings) -> pd.DataFrame
  - All four paths accept str or PathLike. settings is step 01's DecodingSettings
    (the entry point passes DECODING): its extension sets, probe_timeout and
    audio_proxy are used. out_path is the parquet FILE
    (data/cache/songs.parquet); its parent is created if missing.
  - Returns the table and writes the same table to out_path.
  - Independent of the video table: it neither reads nor needs videos.parquet.
    (Decision: each table owns one kind of proxy. Steps that need both join on
    song_id == video_id, which holds because both come from generate_file_id.)

  Which files are inputs
  - kind "amv":  every settings.video_extensions file below amv_dir (same rule as the video
                 table: any depth, case-insensitive).
  - kind "song": every settings.audio_extensions file below song_dir (.wav, .mp3, .flac; any
                 depth, case-insensitive). Video files there are ignored. A .wav gets
                 a proxy too: step 02 only accepts 22050 Hz mono (see
                 tests/music/test_audio.py).
  - Everything else is ignored. Row order: AMVs, then songs, each sorted by path.

  Two phases, as in the video table
  - Discovery probes every input first. A file without an audio stream raises
    ValueError naming the file, and ffprobe failing on a file raises RuntimeError,
    both BEFORE any wav is made (proxy_dir stays empty).
  - Then every input gets make_audio_proxy(file, proxy_dir, settings.audio_proxy).

  Columns (and no others)
    song_id           generate_file_id(original): the uuid prefix of the wav's name,
                      and for an AMV equal to its video_id in videos.parquet
    kind              "amv" or "song" (same column name as in videos.parquet)
    path              the original, resolved, as str
    name              its file name
    audio_proxy_path  str, the path make_audio_proxy returned; the file exists
    duration, sample_rate, channels, codec
                      of the ORIGINAL's audio stream (probe(original).audio),
                      not of the wav (which is always 22050 Hz mono pcm)
    end_time          float64 seconds of the original where the wav stops (task 3.8): an
                      AMV's end-card start, or `duration` when nothing is cut

  End cards (task 3.8), the audio side of the video table's rule
  - Only AMVs, and only if settings.end_card is not None. Songs are never searched.
  - detect_end_card(original, probe(original).video.duration, settings.end_card,
    has_audio=True): the VIDEO duration, exactly as the video table calls it (discovery
    already guarantees audio), so an AMV's wav and video proxy are cut at the same float.
    A found card -> make_audio_proxy(..., end=that time), otherwise end=None (an AMV
    without a card keeps the wav it already had).

  Rules
  - Re-running is cheap: nothing changed -> no ffmpeg runs (wav mtimes unchanged)
    and an equal table.
  - Missing amv_dir or song_dir -> FileNotFoundError. (Decision: as in step 01, a
    typo must not silently give an empty table. Create data/songs/ even if it stays
    empty.)
  - Empty folders give an empty table that still has every column.
"""

import dataclasses
import wave
from pathlib import Path

import pandas as pd
import pytest

from amv.config import DECODING
from amv.io.ffmpeg import audio_proxy_path, generate_file_id, probe
from amv.io.songs import build_song_table
from amv.io.videos import build_video_table

pytestmark = pytest.mark.integration  # every test here runs the real ffmpeg

COLUMNS = {
    "song_id",
    "kind",
    "path",
    "name",
    "audio_proxy_path",
    "duration",
    "sample_rate",
    "channels",
    "codec",
    "end_time",
}


@pytest.fixture
def dirs(tmp_path):
    """The data/ layout inside tmp_path. amvs/ and songs/ exist; proxies/ and cache/ don't."""
    (tmp_path / "amvs").mkdir()
    (tmp_path / "songs").mkdir()
    return {
        "amv_dir": tmp_path / "amvs",
        "song_dir": tmp_path / "songs",
        "proxy_dir": tmp_path / "proxies",
        "out_path": tmp_path / "cache" / "songs.parquet",
    }


@pytest.fixture
def amv(make_media):
    return make_media("amvs/My AMV.mp4", video=True, audio=True, duration=1.0)


@pytest.fixture
def song(make_media):
    return make_media("songs/track.mp3", video=False, audio=True, duration=1.0, channels=2)


def row_of(table: pd.DataFrame, path: Path) -> pd.Series:
    rows = table[table["path"] == str(path.resolve())]
    assert len(rows) == 1, f"expected one row for {path}, got {len(rows)}"
    return rows.iloc[0]


# --- The table and the parquet file -----------------------------------------------


def test_has_exactly_the_specified_columns(dirs, amv, song):
    table = build_song_table(**dirs, settings=DECODING)

    assert set(table.columns) == COLUMNS


def test_one_row_per_input(dirs, amv, song):
    table = build_song_table(**dirs, settings=DECODING)

    assert len(table) == 2
    assert table["song_id"].is_unique


def test_writes_the_same_table_to_parquet(dirs, amv, song):
    table = build_song_table(**dirs, settings=DECODING)

    assert dirs["out_path"].is_file()  # cache/ did not exist before
    pd.testing.assert_frame_equal(pd.read_parquet(dirs["out_path"]), table)


def test_accepts_str_paths(dirs, song):
    table = build_song_table(**{k: str(v) for k, v in dirs.items()}, settings=DECODING)

    assert len(table) == 1


def test_empty_folders_give_an_empty_table_with_all_columns(dirs):
    table = build_song_table(**dirs, settings=DECODING)

    assert len(table) == 0
    assert set(table.columns) == COLUMNS
    assert dirs["out_path"].is_file()


@pytest.mark.parametrize("missing", ["amv_dir", "song_dir"])
def test_missing_input_folder_raises(dirs, missing):
    dirs[missing] = dirs[missing].parent / "typo"

    with pytest.raises(FileNotFoundError):
        build_song_table(**dirs, settings=DECODING)


# --- Which files are inputs -------------------------------------------------------


@pytest.mark.parametrize("name", ["a.mp3", "b.wav", "c.flac", "D.MP3"])
def test_each_audio_format_is_a_song(dirs, make_media, name):
    src = make_media(f"songs/{name}", video=False, audio=True, duration=1.0)

    table = build_song_table(**dirs, settings=DECODING)

    row = row_of(table, src)
    assert row["kind"] == "song"
    assert row["name"] == name


@pytest.mark.parametrize("name", ["a.mp4", "b.mkv", "C.MP4"])
def test_each_video_format_is_an_amv(dirs, make_media, name):
    src = make_media(f"amvs/{name}", video=True, audio=True, duration=1.0)

    table = build_song_table(**dirs, settings=DECODING)

    assert row_of(table, src)["kind"] == "amv"


def test_finds_inputs_at_any_depth_and_ignores_other_files(dirs, make_media):
    make_media("songs/Artist/Album/track.mp3", video=False, audio=True, duration=1.0)
    make_media("amvs/2024/edit.mkv", video=True, audio=True, duration=1.0)
    (dirs["song_dir"] / "cover.jpg").write_bytes(b"\xff\xd8")
    (dirs["amv_dir"] / "notes.txt").write_text("hi")
    (dirs["amv_dir"] / "song.mp3").write_bytes(b"")  # audio in amv_dir is not an AMV
    (dirs["song_dir"] / "clip.mp4").write_bytes(b"")  # video in song_dir is not a song

    table = build_song_table(**dirs, settings=DECODING)

    assert sorted(table["name"]) == ["edit.mkv", "track.mp3"]


def test_row_order_is_amvs_then_songs_each_sorted_by_path(dirs, make_media):
    for name in ["songs/z.mp3", "amvs/b.mp4", "songs/A/b.mp3", "amvs/a.mp4", "songs/a.mp3"]:
        make_media(name, video=name.startswith("amvs"), audio=True, duration=1.0)

    table = build_song_table(**dirs, settings=DECODING)

    assert list(table["kind"]) == ["amv", "amv", "song", "song", "song"]
    for kind in ("amv", "song"):
        paths = table[table["kind"] == kind]["path"].tolist()
        assert paths == sorted(paths)


# --- One row ----------------------------------------------------------------------


def test_identity_columns(dirs, song):
    table = build_song_table(**dirs, settings=DECODING)

    row = row_of(table, song)
    assert row["song_id"] == generate_file_id(song)
    assert row["path"] == str(song.resolve())
    assert row["name"] == "track.mp3"


def test_wav_is_the_audio_proxy_in_proxy_dir(dirs, amv, song):
    table = build_song_table(**dirs, settings=DECODING)

    for src in (amv, song):
        row = row_of(table, src)
        wav = Path(row["audio_proxy_path"])
        assert wav.is_file() and wav.suffix == ".wav"
        assert wav.parent.resolve() == dirs["proxy_dir"].resolve()
        assert wav.name.startswith(row["song_id"])
        audio = probe(wav, DECODING.probe_timeout).audio
        assert audio is not None
        assert (audio.sample_rate, audio.channels) == (22050, 1)


def test_metadata_describes_the_original_not_the_wav(dirs, amv, song):
    table = build_song_table(**dirs, settings=DECODING)

    for src in (amv, song):
        row = row_of(table, src)
        original = probe(src, DECODING.probe_timeout).audio
        assert original is not None
        assert row["duration"] == pytest.approx(original.duration)
        assert row["sample_rate"] == original.sample_rate == 44100
        assert row["channels"] == original.channels
        assert row["codec"] == original.codec
    assert row_of(table, song)["channels"] == 2


def test_amv_song_id_equals_its_video_id(dirs, amv, tmp_path):
    (tmp_path / "library").mkdir()
    videos = build_video_table(
        dirs["amv_dir"], tmp_path / "library", dirs["proxy_dir"], tmp_path / "v.parquet", settings=DECODING
    )

    songs = build_song_table(**dirs, settings=DECODING)

    assert row_of(songs, amv)["song_id"] == row_of(videos, amv)["video_id"]


# --- Failures ---------------------------------------------------------------------


def test_amv_without_audio_raises_before_any_work(dirs, song, make_media):
    make_media("amvs/silent edit.mp4", video=True, audio=False, duration=1.0)

    with pytest.raises(ValueError, match="silent edit.mp4"):
        build_song_table(**dirs, settings=DECODING)

    proxy_dir = dirs["proxy_dir"]
    assert not proxy_dir.exists() or list(proxy_dir.iterdir()) == []  # not even the song was done


def test_undecodable_song_raises_before_any_work(dirs, amv):
    (dirs["song_dir"] / "broken.mp3").write_text("this is not audio")

    with pytest.raises(RuntimeError):
        build_song_table(**dirs, settings=DECODING)

    proxy_dir = dirs["proxy_dir"]
    assert not proxy_dir.exists() or list(proxy_dir.iterdir()) == []


# --- Re-running -------------------------------------------------------------------


def test_second_run_extracts_nothing_and_gives_the_same_table(dirs, amv, song):
    first = build_song_table(**dirs, settings=DECODING)
    stamps = {p: p.stat().st_mtime_ns for p in dirs["proxy_dir"].iterdir()}

    second = build_song_table(**dirs, settings=DECODING)

    assert {p: p.stat().st_mtime_ns for p in dirs["proxy_dir"].iterdir()} == stamps
    pd.testing.assert_frame_equal(second, first)


# --- End cards (task 3.8) ---------------------------------------------------------


@pytest.fixture
def carded_amv(make_amv_with_end_card):
    """4 s of footage, then a 3 s static card, with audio under both."""
    return make_amv_with_end_card("amvs/Carded AMV.mp4", footage=4.0, card=3.0)


def wav_seconds(path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def test_an_amvs_end_card_is_cut_from_the_wav(dirs, carded_amv):
    row = row_of(build_song_table(**dirs, settings=DECODING), carded_amv)

    assert row["end_time"] == pytest.approx(4.0, abs=1 / DECODING.end_card.fps)
    assert wav_seconds(row["audio_proxy_path"]) == pytest.approx(row["end_time"], abs=0.01)
    assert row["end_time"] < row["duration"]  # duration stays the original's


def test_wav_and_video_proxy_are_cut_at_the_same_time(dirs, carded_amv, tmp_path):
    (tmp_path / "library").mkdir()
    videos = build_video_table(
        dirs["amv_dir"], tmp_path / "library", dirs["proxy_dir"], tmp_path / "v.parquet", settings=DECODING
    )

    songs = build_song_table(**dirs, settings=DECODING)

    assert row_of(songs, carded_amv)["end_time"] == row_of(videos, carded_amv)["end_time"]


def test_without_a_card_end_time_is_the_duration_and_the_wav_is_unchanged(dirs, amv):
    row = row_of(build_song_table(**dirs, settings=DECODING), amv)

    assert row["end_time"] == row["duration"]
    assert row["audio_proxy_path"] == str(audio_proxy_path(amv, dirs["proxy_dir"], DECODING.audio_proxy))


def test_songs_are_never_cut(dirs, song):
    row = row_of(build_song_table(**dirs, settings=DECODING), song)

    assert row["end_time"] == row["duration"]
    assert row["audio_proxy_path"] == str(audio_proxy_path(song, dirs["proxy_dir"], DECODING.audio_proxy))


def test_detection_can_be_turned_off(dirs, carded_amv):
    off = dataclasses.replace(DECODING, end_card=None)

    row = row_of(build_song_table(**dirs, settings=off), carded_amv)

    assert row["end_time"] == row["duration"]
    assert row["audio_proxy_path"] == str(
        audio_proxy_path(carded_amv, dirs["proxy_dir"], DECODING.audio_proxy)
    )


def test_end_time_is_float64(dirs, amv, carded_amv, song):
    table = build_song_table(**dirs, settings=DECODING)

    assert table["end_time"].dtype == "float64"


def test_a_cut_amv_is_not_extracted_again(dirs, carded_amv):
    first = build_song_table(**dirs, settings=DECODING)
    wav = Path(row_of(first, carded_amv)["audio_proxy_path"])
    stamp = wav.stat().st_mtime_ns

    second = build_song_table(**dirs, settings=DECODING)

    assert wav.stat().st_mtime_ns == stamp
    pd.testing.assert_frame_equal(second, first)


def test_the_wav_ends_where_the_song_ends(dirs, make_amv_with_end_card):
    src = make_amv_with_end_card("amvs/Outro AMV.mp4", footage=4.0, card=3.0, song_end=2.5)

    row = row_of(build_song_table(**dirs, settings=DECODING), src)

    assert row["end_time"] == pytest.approx(2.5, abs=0.1)
    assert wav_seconds(row["audio_proxy_path"]) == pytest.approx(row["end_time"], abs=0.01)


def test_wav_and_video_proxy_end_at_the_same_song_end(dirs, make_amv_with_end_card, tmp_path):
    src = make_amv_with_end_card("amvs/Outro AMV.mp4", footage=4.0, card=3.0, song_end=2.5)
    (tmp_path / "library").mkdir()
    videos = build_video_table(
        dirs["amv_dir"], tmp_path / "library", dirs["proxy_dir"], tmp_path / "v.parquet", settings=DECODING
    )

    songs = build_song_table(**dirs, settings=DECODING)

    assert row_of(songs, src)["end_time"] == row_of(videos, src)["end_time"]
