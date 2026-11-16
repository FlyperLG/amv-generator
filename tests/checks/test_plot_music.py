"""Tests for checks.plot_music: the command line of the step 02 check (task 3.7).

The end-to-end tests write a synthetic song's wav and a fake analysis into the cache layout
of step 02, so main runs without the beat model.

SPECIFICATION (decisions these tests encode)
--------------------------------------------
make_window(start, end)
  - No --start and no --end -> None (the whole song). Otherwise (start or 0, end or inf).

select_songs(songs, pattern)
  - None -> every row. Otherwise the rows whose name contains pattern, case-insensitive and
    literal (song names contain brackets). No match -> ValueError.

output_stem(name, window)
  - The png's name: the song's file name without extension, plus "_<start>-<end>s" with a
    window, or "_<start>s-end" with an open end; numbers without trailing ".0". A windowed
    plot never overwrites the plot of the whole song.

parse_args(argv)
  - Exits with a usage error if the song table is missing, --start < 0, or --end <= --start.
  - Defaults: PATHS for the song table and music cache, PLOT_MUSIC.out_dir for the output.

main(argv)
  - Per selected song: <name>.png (or the windowed name), and unless --no-click
    <name>_click.wav, always the whole song: stereo, music left, clicks right.
  - Logs per song the name and BPM, then the beat offset and kick by bar position.
  - A pattern that matches nothing raises ValueError before anything is written.
"""

import logging
import math

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from amv.config import MUSIC, PATHS
from amv.music.analysis import _save, cache_paths
from checks.plot_music import main, make_window, output_stem, parse_args, select_songs
from checks.settings import PLOT_MUSIC
from tests.checks.helpers import SR, make_analysis, make_beats

# --- make_window / select_songs / output_stem ------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "window"),
    [
        (None, None, None),
        (30.0, 40.0, (30.0, 40.0)),
        (30.0, None, (30.0, math.inf)),
        (None, 40.0, (0.0, 40.0)),
    ],
)
def test_make_window(start, end, window):
    assert make_window(start, end) == window


@pytest.fixture
def songs():
    return pd.DataFrame({"name": ["Heroes [abc].mp4", "Blue Lock [x1].mp4", "Enemy (remix).mp3"]})


def test_select_songs_without_a_pattern_returns_all(songs):
    assert len(select_songs(songs, None)) == 3


def test_select_songs_is_case_insensitive_and_literal(songs):
    assert list(select_songs(songs, "blue LOCK")["name"]) == ["Blue Lock [x1].mp4"]
    assert list(select_songs(songs, "[abc]")["name"]) == ["Heroes [abc].mp4"]
    assert list(select_songs(songs, "(remix)")["name"]) == ["Enemy (remix).mp3"]


def test_select_songs_without_a_match_raises(songs):
    with pytest.raises(ValueError, match="zzz"):
        select_songs(songs, "zzz")


@pytest.mark.parametrize(
    ("window", "stem"),
    [(None, "Song [x]"), ((30.0, 40.0), "Song [x]_30-40s"), ((2.5, math.inf), "Song [x]_2.5s-end")],
)
def test_output_stem(window, stem):
    assert output_stem("Song [x].mp4", window) == stem


# --- parse_args ------------------------------------------------------------------------


@pytest.fixture
def song_table(tmp_path):
    path = tmp_path / "songs.parquet"
    path.write_bytes(b"")  # parse_args only checks that it exists
    return path


def test_defaults_come_from_the_settings(song_table):
    args = parse_args(["--song-table", str(song_table)])
    assert args.cache_dir == PATHS.music_cache_dir
    assert args.out_dir == PLOT_MUSIC.out_dir
    assert (args.song, args.start, args.end, args.no_click) == (None, None, None, False)


@pytest.mark.parametrize(
    "extra",
    [["--start", "-1"], ["--start", "40", "--end", "30"], ["--start", "30", "--end", "30"]],
    ids=["negative start", "end before start", "empty window"],
)
def test_bad_windows_exit(song_table, extra):
    with pytest.raises(SystemExit):
        parse_args(["--song-table", str(song_table), *extra])


def test_a_missing_song_table_exits(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--song-table", str(tmp_path / "missing.parquet")])


# --- main, end to end on a fake cache --------------------------------------------------


@pytest.fixture
def fake_cache(tmp_path, kick_loop):
    """A song table with one synthetic song, its wav, and its analysis in the cache layout."""
    y, beat_times = kick_loop(bpm=120, bars=16)
    wav = tmp_path / "loop.wav"
    sf.write(wav, y, SR)
    cache_dir = tmp_path / "music"
    analysis = make_analysis(make_beats(beat_times), duration=len(y) / SR)
    _save(analysis, cache_paths("loop01", cache_dir, MUSIC), MUSIC)
    table = tmp_path / "songs.parquet"
    pd.DataFrame(
        {"song_id": ["loop01"], "name": ["Kick Loop [t1].wav"], "audio_proxy_path": [str(wav)]}
    ).to_parquet(table)
    return {"table": table, "cache_dir": cache_dir, "out": tmp_path / "out", "n_samples": len(y)}


def _run(cache, *extra):
    main(
        [
            "--song-table", str(cache["table"]),
            "--cache-dir", str(cache["cache_dir"]),
            "--out-dir", str(cache["out"]),
            *extra,
        ]
    )  # fmt: skip


def test_main_writes_the_plot_and_the_click_track(fake_cache, caplog):
    with caplog.at_level(logging.INFO, logger="checks.plot_music"):
        _run(fake_cache)

    assert (fake_cache["out"] / "Kick Loop [t1].png").exists()
    mix, sr = sf.read(fake_cache["out"] / "Kick Loop [t1]_click.wav")
    assert sr == SR
    assert mix.shape == (fake_cache["n_samples"], 2)
    assert np.abs(mix[:, 1]).max() > 0.5  # clicks on the right

    messages = [r.getMessage() for r in caplog.records]
    assert any("Kick Loop [t1] (120 BPM)" in m for m in messages)
    assert any("beat offset" in m and "kick by bar position [1.00" in m for m in messages)


def test_a_windowed_run_keeps_the_full_plot_and_skips_clicks_on_request(fake_cache):
    _run(fake_cache, "--no-click")
    _run(fake_cache, "--start", "5", "--end", "10", "--no-click")
    assert sorted(p.name for p in fake_cache["out"].iterdir()) == [
        "Kick Loop [t1].png",
        "Kick Loop [t1]_5-10s.png",
    ]


def test_a_pattern_without_a_match_writes_nothing(fake_cache):
    with pytest.raises(ValueError):
        _run(fake_cache, "--song", "zzz")
    assert not fake_cache["out"].exists()
