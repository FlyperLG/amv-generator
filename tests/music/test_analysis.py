"""Tests for amv.music.analysis (docs/02-music-features, task 3.6).

analysis.py is the step 02 counterpart of amv/io/songs.py: it runs tasks 3.1-3.5 on every
song of the song table and caches the two outputs per song (docs section 2). Black-box
tests: a synthetic drum loop is written as a wav into tmp_path and analysed with a FAKE beat
model that returns the loop's true beats, so no download and no real inference is needed.
Everything after the model (librosa, Savitzky-Golay, sections) runs for real. One
integration test runs the real model on the Blue Lock song.

SPECIFICATION
  amv.config.MusicSettings (frozen dataclass): every setting that changes step 02's outputs,
  as fields whose DEFAULTS are the configured values, each with its comment. Then
      MUSIC = MusicSettings()
  in amv/config.py. (Decision: the step 02 settings are one typed config section, like an
  options class, replacing the loose step 02 constants. Entry points pass MUSIC whole to
  analyse_song(s), or single fields to a step function: MUSIC.min_ibi. The step functions
  still take plain arguments. Variants for tests or experiments: dataclasses.replace(MUSIC,
  drop_level=0.5). config.py stays a leaf module: it imports nothing from amv.)
      sample_rate             default DECODING.audio_proxy.sample_rate (the rate of step 01's wavs)
      beat_checkpoint         (was BEAT_THIS_CHECKPOINT)
      min_ibi, beats_per_bar, downbeat_tol, downbeat_window_bars,
      hop_length, n_fft, n_mels, low_band_hz, high_band_hz,
      sg_polyorder, dense_sg_window_beats, beat_sg_window, n_mfcc,
      section_kernel_bars, min_section_bars, novelty_prominence, drop_level, build_rise
                              (was the constant of the same name in upper case)
  - settings_id (property) -> str: the same for equal settings, different if any field
    differs, the same in every Python process (so NOT hash(), which is randomised per
    process for strings), and safe in a file name (only [0-9a-z-]). Like the settings uuid
    in step 01's proxy names.

  SongAnalysis (frozen dataclass): beats: pd.DataFrame, dense: DenseFeatures.

  build_beat_table(grid, dynamics, sections) -> pd.DataFrame, one row per beat
  - Columns, in order: t, is_downbeat, beat_in_bar, bar_idx, ibi, then every key of dynamics
    in its order, then section.
  - t (float64), is_downbeat (bool), beat_in_bar and bar_idx (int64) come from the grid.
  - ibi (float64): t[i + 1] - t[i]; the last beat repeats the one before it (Decision: the
    same rule pool_to_beats uses for the last beat's interval). Never NaN.
  - section: the labels, as strings.
  - At least 2 beats, and every dynamics array and sections must have one value per beat:
    ValueError otherwise. Inputs not modified.

  cache_paths(song_id, cache_dir, settings) -> tuple[Path, Path]
  - (cache_dir / "<song_id>_<settings_id>_beats.parquet",
     cache_dir / "<song_id>_<settings_id>_dense.npz"). Nothing is created.
  - Replaces the doc's "_v1" suffix: changing a setting gives new names, so stale results
    are never reused. Old files stay until deleted by hand. (Known limitation, as in step 01:
    a code change without a settings change does not invalidate the cache.)

  analyse_song(song_id, audio_path, cache_dir, settings, model) -> SongAnalysis
  - If both cache files exist and are newer than audio_path: loaded and returned, the model
    is not called.
  - Otherwise the full chain, with the settings passed through:
        y, sr   = load_audio(audio_path, sample_rate)
        grid    = build_beat_grid(clean_beats(beats), downbeats, ...)  beats from detect_beats(y, sr, model)
        dense   = compute_dense_features(y, sr, ...)
        dense_s = smooth_dense(dense, median(diff(grid.times)), dense_sg_window_beats, sg_polyorder)
        dyn     = beat_dynamics(pool_to_beats(dense, grid.times), dense.names, beat_sg_window, sg_polyorder)
        bars    = pool_to_beats(compute_timbre_features(y, sr, ...), grid.times[grid.is_downbeat])
        bounds  = find_section_boundaries(section_novelty(bars, ...), ...)
        labels  = label_sections(grid, dyn["rms_s"], dyn["rms_d1"], bounds, ...)
        beats   = build_beat_table(grid, dyn, labels)
    The dense output holds the raw curves, then their smoothed versions: names
    (rms, onset, flux, low, mid, high, rms_s, ..., high_s), values [T, 12], same times.
  - Everything is computed before anything is written. Each file is written to a temporary
    name and then renamed, so a crash never leaves a half-written or a stale-looking file.
  - cache_dir is created if missing. Also writes cache_dir / "settings_<settings_id>.json"
    with the settings as a dict, so a human can tell what an id stands for (as the sidecar
    json in step 01).
  - Errors from the steps propagate unchanged, e.g. ValueError for a song with too few beats.
  - The model is an argument (Decision: loaded once by the caller, as detect_beats takes it).

  load_analysis(song_id, cache_dir, settings) -> SongAnalysis
  - Reads the two cache files. FileNotFoundError if either is missing.
  - Gives back exactly what analyse_song computed (same frame, same arrays, names a tuple
    of str). The npz is read without pickle.

  analyse_songs(songs, cache_dir, settings, model) -> None
  - songs is the song table as a DataFrame (Decision: the caller reads the parquet; the
    function needs no file and can take a filtered table). Uses only song_id and
    audio_proxy_path; other columns are ignored.
  - A missing column: ValueError before any song is analysed.
  - Every row in order, with a progress bar. A song whose cache is up to date (analyse_song's
    rule) is skipped WITHOUT loading it (Decision: analyse_songs only makes sure the cache
    is complete; later steps load one song at a time with load_analysis). So a re-run only
    analyses new songs.
  - An error stops the run and propagates (Decision: fail loudly, as step 01). Songs
    finished before it stay cached.
"""

import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from amv.config import DECODING, MUSIC, MusicSettings
from amv.music.analysis import (
    SongAnalysis,
    analyse_song,
    analyse_songs,
    build_beat_table,
    cache_paths,
    load_analysis,
)
from amv.music.beats import BeatGrid
from amv.music.dense import DenseFeatures
from amv.music.sections import SECTION_LABELS

SR = 22050
DENSE_NAMES = ("rms", "onset", "flux", "low", "mid", "high")
BEAT_COLUMNS = [
    "t",
    "is_downbeat",
    "beat_in_bar",
    "bar_idx",
    "ibi",
    *DENSE_NAMES,
    *(f"{n}_s" for n in DENSE_NAMES),
    *(f"{n}_d1" for n in DENSE_NAMES),
    *(f"{n}_d2" for n in DENSE_NAMES),
    "section",
]
SETTING_FIELDS = [
    "sample_rate",
    "beat_checkpoint",
    "min_ibi",
    "beats_per_bar",
    "downbeat_tol",
    "downbeat_window_bars",
    "hop_length",
    "n_fft",
    "n_mels",
    "low_band_hz",
    "high_band_hz",
    "sg_polyorder",
    "dense_sg_window_beats",
    "beat_sg_window",
    "n_mfcc",
    "section_kernel_bars",
    "min_section_bars",
    "novelty_prominence",
    "drop_level",
    "build_rise",
]


# --- Helpers and fixtures ---------------------------------------------------------


class FakeModel:
    """Stands in for beat_this: returns fixed beats, downbeats on every 4th, counts calls."""

    def __init__(self, beats):
        self.beats = np.asarray(beats, dtype=np.float64)
        self.calls = 0

    def __call__(self, y, sample_rate):
        self.calls += 1
        return self.beats.copy(), self.beats[::4].copy()


def exploding_model(y, sample_rate):
    raise AssertionError("the model must not be called")


@pytest.fixture
def settings():
    return MUSIC


@pytest.fixture
def song(tmp_path, drum_loop):
    """A 16-bar 120 BPM drum loop as a 22050 Hz mono wav, and a fake model with its true beats."""
    y, beats = drum_loop(bpm=120, bars=16)
    path = tmp_path / "songs" / "loop.wav"
    path.parent.mkdir()
    sf.write(path, y, SR)
    return path, FakeModel(beats)


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache" / "music"


def make_grid(n_bars, beats_per_bar=4, ibi=0.5):
    n = n_bars * beats_per_bar
    beat_in_bar = np.arange(n) % beats_per_bar
    is_downbeat = beat_in_bar == 0
    return BeatGrid(
        times=0.5 + ibi * np.arange(n),
        is_downbeat=is_downbeat,
        beat_in_bar=beat_in_bar,
        bar_idx=np.cumsum(is_downbeat) - 1,
    )


def table_inputs(n_bars=2):
    grid = make_grid(n_bars)
    n = len(grid.times)
    dynamics = {"rms": np.linspace(0, 1, n), "rms_s": np.linspace(1, 2, n)}
    sections = np.array(["intro"] * (n // 2) + ["drop"] * (n - n // 2))
    return grid, dynamics, sections


def mtimes(paths):
    return [p.stat().st_mtime_ns for p in paths]


# =================================================================================
# MusicSettings
# =================================================================================


def test_settings_have_exactly_the_planned_fields():
    assert [f.name for f in dataclasses.fields(MusicSettings)] == SETTING_FIELDS


def test_settings_are_frozen(settings):
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.drop_level = 0.5  # type: ignore[misc]


def test_music_is_the_default_settings():
    assert MusicSettings() == MUSIC


def test_sample_rate_is_step_01s_wav_rate():
    assert MUSIC.sample_rate == DECODING.audio_proxy.sample_rate


def test_config_has_the_planned_beat_settings():
    """The other fields are checked next to the tests that use them (test_dense, ...)."""
    assert MUSIC.beat_checkpoint == "final0"
    assert MUSIC.min_ibi == 0.25
    assert MUSIC.beats_per_bar == 4
    assert MUSIC.downbeat_tol == 0.07
    assert MUSIC.downbeat_window_bars == 8


def test_the_loose_step_02_constants_are_gone():
    """One source of truth: the values live only in MusicSettings."""
    import amv.config as config

    leftovers = [n.upper() for n in SETTING_FIELDS[2:] if hasattr(config, n.upper())]
    assert leftovers == []
    assert not hasattr(config, "BEAT_THIS_CHECKPOINT")


def test_settings_id_is_the_same_for_equal_settings(settings):
    assert MusicSettings().settings_id == settings.settings_id


@pytest.mark.parametrize("name", SETTING_FIELDS)
def test_settings_id_changes_with_every_field(settings, name):
    value = getattr(settings, name)
    changed = "other" if isinstance(value, str) else value + 1

    assert dataclasses.replace(settings, **{name: changed}).settings_id != settings.settings_id


def test_settings_id_is_safe_in_a_file_name(settings):
    settings_id = settings.settings_id

    assert settings_id
    assert set(settings_id) <= set("0123456789abcdefghijklmnopqrstuvwxyz-")


def test_settings_id_is_the_same_in_another_process(settings):
    """hash() of a str changes per process, so it can't name a cache file."""
    code = "from amv.config import MUSIC; print(MUSIC.settings_id)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == settings.settings_id


# =================================================================================
# build_beat_table
# =================================================================================


def test_beat_table_has_the_columns_in_order():
    grid, dynamics, sections = table_inputs()

    df = build_beat_table(grid, dynamics, sections)

    assert list(df.columns) == [
        "t",
        "is_downbeat",
        "beat_in_bar",
        "bar_idx",
        "ibi",
        "rms",
        "rms_s",
        "section",
    ]
    assert len(df) == len(grid.times)


def test_beat_table_takes_the_grid_and_curves_as_given():
    grid, dynamics, sections = table_inputs()

    df = build_beat_table(grid, dynamics, sections)

    np.testing.assert_array_equal(df["t"], grid.times)
    np.testing.assert_array_equal(df["is_downbeat"], grid.is_downbeat)
    np.testing.assert_array_equal(df["beat_in_bar"], grid.beat_in_bar)
    np.testing.assert_array_equal(df["bar_idx"], grid.bar_idx)
    np.testing.assert_array_equal(df["rms_s"], dynamics["rms_s"])
    assert df["section"].tolist() == sections.tolist()


def test_beat_table_dtypes():
    df = build_beat_table(*table_inputs())

    assert df["t"].dtype == np.float64
    assert df["is_downbeat"].dtype == np.bool_
    assert df["beat_in_bar"].dtype == np.int64
    assert df["bar_idx"].dtype == np.int64
    assert df["ibi"].dtype == np.float64
    assert all(isinstance(s, str) for s in df["section"])


def test_ibi_is_the_gap_to_the_next_beat_and_the_last_repeats_the_one_before():
    grid = make_grid(1)
    grid = dataclasses.replace(grid, times=np.array([1.0, 1.5, 2.25, 2.75]))
    sections = np.array(["intro"] * 4)

    df = build_beat_table(grid, {}, sections)

    np.testing.assert_allclose(df["ibi"], [0.5, 0.75, 0.5, 0.5])


def test_beat_table_rejects_a_single_beat():
    grid = make_grid(1)
    one = BeatGrid(grid.times[:1], grid.is_downbeat[:1], grid.beat_in_bar[:1], grid.bar_idx[:1])

    with pytest.raises(ValueError):
        build_beat_table(one, {}, np.array(["intro"]))


def test_beat_table_rejects_a_curve_of_the_wrong_length():
    grid, dynamics, sections = table_inputs()
    dynamics["rms"] = dynamics["rms"][:-1]

    with pytest.raises(ValueError):
        build_beat_table(grid, dynamics, sections)


def test_beat_table_rejects_sections_of_the_wrong_length():
    grid, dynamics, sections = table_inputs()

    with pytest.raises(ValueError):
        build_beat_table(grid, dynamics, sections[:-1])


def test_beat_table_does_not_modify_the_inputs():
    grid, dynamics, sections = table_inputs()
    before = [a.copy() for a in (grid.times, *dynamics.values(), sections)]

    build_beat_table(grid, dynamics, sections)

    for after, original in zip((grid.times, *dynamics.values(), sections), before, strict=True):
        np.testing.assert_array_equal(after, original)


# =================================================================================
# cache_paths
# =================================================================================


def test_cache_paths_are_named_by_song_and_settings(settings, cache_dir):
    beats, dense = cache_paths("abc", cache_dir, settings)

    assert beats == cache_dir / f"abc_{settings.settings_id}_beats.parquet"
    assert dense == cache_dir / f"abc_{settings.settings_id}_dense.npz"
    assert not cache_dir.exists()


def test_cache_paths_change_with_the_settings(settings, cache_dir):
    other = dataclasses.replace(settings, drop_level=0.5)

    assert set(cache_paths("abc", cache_dir, settings)).isdisjoint(cache_paths("abc", cache_dir, other))


# =================================================================================
# analyse_song
# =================================================================================


def test_analyse_song_writes_both_files_and_the_settings_sidecar(song, settings, cache_dir):
    path, model = song

    analyse_song("loop", path, cache_dir, settings, model)

    beats_path, dense_path = cache_paths("loop", cache_dir, settings)
    sidecar = cache_dir / f"settings_{settings.settings_id}.json"
    assert sorted(p.name for p in cache_dir.iterdir()) == sorted(
        [beats_path.name, dense_path.name, sidecar.name]
    )
    assert json.loads(sidecar.read_text(encoding="utf-8")) == dataclasses.asdict(settings)


def test_analyse_song_returns_the_full_beat_table(song, settings, cache_dir):
    path, model = song

    result = analyse_song("loop", path, cache_dir, settings, model)

    assert isinstance(result, SongAnalysis)
    df = result.beats
    assert list(df.columns) == BEAT_COLUMNS
    np.testing.assert_allclose(df["t"], model.beats)
    assert df["is_downbeat"].sum() == 16
    assert set(df["section"]) <= set(SECTION_LABELS)
    assert not df.isna().any().any()


def test_analyse_song_returns_raw_and_smoothed_dense_curves(song, settings, cache_dir):
    path, model = song

    dense = analyse_song("loop", path, cache_dir, settings, model).dense

    assert isinstance(dense, DenseFeatures)
    assert dense.names == (*DENSE_NAMES, *(f"{n}_s" for n in DENSE_NAMES))
    assert dense.values.shape == (len(dense.times), 12)
    assert dense.times[0] == 0.0
    duration = sf.info(path).duration
    assert dense.times[-1] == pytest.approx(duration, abs=settings.hop_length / SR)


def test_beat_curves_are_the_pooled_dense_curves(song, settings, cache_dir):
    """Beat 3's rms is the mean of the raw dense rms over [t_3, t_4)."""
    path, model = song

    result = analyse_song("loop", path, cache_dir, settings, model)

    t = result.beats["t"].to_numpy()
    frames = (result.dense.times >= t[3]) & (result.dense.times < t[4])
    rms = result.dense.values[:, result.dense.names.index("rms")]
    assert result.beats["rms"].iloc[3] == pytest.approx(rms[frames].mean())


def test_a_cached_song_is_loaded_without_calling_the_model(song, settings, cache_dir):
    path, model = song
    first = analyse_song("loop", path, cache_dir, settings, model)
    before = mtimes(cache_paths("loop", cache_dir, settings))

    second = analyse_song("loop", path, cache_dir, settings, exploding_model)

    assert mtimes(cache_paths("loop", cache_dir, settings)) == before
    pd.testing.assert_frame_equal(second.beats, first.beats)
    np.testing.assert_array_equal(second.dense.values, first.dense.values)


def test_a_song_newer_than_its_cache_is_analysed_again(song, settings, cache_dir):
    path, model = song
    analyse_song("loop", path, cache_dir, settings, model)
    future = cache_paths("loop", cache_dir, settings)[0].stat().st_mtime_ns + 10**10
    os.utime(path, ns=(future, future))

    analyse_song("loop", path, cache_dir, settings, model)

    assert model.calls == 2


def test_a_missing_file_is_analysed_again(song, settings, cache_dir):
    path, model = song
    analyse_song("loop", path, cache_dir, settings, model)
    cache_paths("loop", cache_dir, settings)[1].unlink()

    analyse_song("loop", path, cache_dir, settings, model)

    assert model.calls == 2
    assert all(p.is_file() for p in cache_paths("loop", cache_dir, settings))


def test_other_settings_get_their_own_files(song, settings, cache_dir):
    path, model = song
    other = dataclasses.replace(settings, drop_level=0.5)

    analyse_song("loop", path, cache_dir, settings, model)
    analyse_song("loop", path, cache_dir, other, model)

    assert model.calls == 2
    assert all(
        p.is_file()
        for p in (*cache_paths("loop", cache_dir, settings), *cache_paths("loop", cache_dir, other))
    )


def test_a_failing_song_leaves_no_files(song, settings, cache_dir):
    """3 beats are too few for the 5-beat Savitzky-Golay window."""
    path, model = song
    model.beats = model.beats[:3]

    with pytest.raises(ValueError):
        analyse_song("loop", path, cache_dir, settings, model)

    assert not cache_dir.exists() or not any(cache_dir.glob("loop_*"))


# =================================================================================
# load_analysis
# =================================================================================


def test_load_gives_back_exactly_what_was_computed(song, settings, cache_dir):
    path, model = song
    computed = analyse_song("loop", path, cache_dir, settings, model)

    loaded = load_analysis("loop", cache_dir, settings)

    pd.testing.assert_frame_equal(loaded.beats, computed.beats)
    np.testing.assert_array_equal(loaded.dense.times, computed.dense.times)
    np.testing.assert_array_equal(loaded.dense.values, computed.dense.values)
    assert loaded.dense.names == computed.dense.names
    assert all(type(n) is str for n in loaded.dense.names)


def test_load_needs_both_files(song, settings, cache_dir):
    path, model = song
    analyse_song("loop", path, cache_dir, settings, model)
    cache_paths("loop", cache_dir, settings)[0].unlink()

    with pytest.raises(FileNotFoundError):
        load_analysis("loop", cache_dir, settings)


def test_load_of_an_unknown_song_raises(settings, cache_dir):
    with pytest.raises(FileNotFoundError):
        load_analysis("nope", cache_dir, settings)


# =================================================================================
# analyse_songs
# =================================================================================


def song_table(*rows):
    """A song table like step 01's, with extra columns analyse_songs must ignore."""
    return pd.DataFrame(
        [{"song_id": sid, "kind": "song", "name": "x.wav", "audio_proxy_path": str(p)} for sid, p in rows],
        columns=["song_id", "kind", "name", "audio_proxy_path"],
    )


def test_analyse_songs_caches_every_row(song, settings, cache_dir, tmp_path):
    path, model = song
    copy = tmp_path / "songs" / "copy.wav"
    copy.write_bytes(path.read_bytes())

    analyse_songs(song_table(("a", path), ("b", copy)), cache_dir, settings, model)

    assert model.calls == 2
    for song_id in ("a", "b"):
        assert load_analysis(song_id, cache_dir, settings).beats["t"].size == model.beats.size


def test_analyse_songs_skips_cached_songs(song, settings, cache_dir):
    path, model = song
    analyse_songs(song_table(("a", path)), cache_dir, settings, model)

    analyse_songs(song_table(("a", path)), cache_dir, settings, exploding_model)


@pytest.mark.parametrize("missing", ["song_id", "audio_proxy_path"])
def test_analyse_songs_checks_the_columns_first(song, settings, cache_dir, missing):
    path, _ = song
    songs = song_table(("a", path)).drop(columns=missing)

    with pytest.raises(ValueError, match=missing):
        analyse_songs(songs, cache_dir, settings, exploding_model)


def test_an_empty_table_does_nothing(settings, cache_dir):
    analyse_songs(song_table(), cache_dir, settings, exploding_model)

    assert not cache_dir.exists() or not any(cache_dir.iterdir())


def test_a_failing_song_stops_the_run_and_earlier_songs_stay_cached(song, settings, cache_dir, tmp_path):
    path, model = song
    silent = tmp_path / "songs" / "silent.wav"
    sf.write(silent, np.zeros(SR, np.float32), SR)

    with pytest.raises(ValueError):
        analyse_songs(song_table(("a", path), ("b", silent), ("c", path)), cache_dir, settings, model)

    assert all(p.is_file() for p in cache_paths("a", cache_dir, settings))
    assert not any(cache_dir.glob("c_*"))


# =================================================================================
# Config
# =================================================================================


def test_config_has_the_music_cache_dir():
    import amv.config as config

    assert config.PATHS.music_cache_dir == config.DATA_DIR / "cache" / "music"


# =================================================================================
# Real song (integration)
# =================================================================================


@pytest.mark.integration
def test_bluelock_is_analysed_end_to_end(beat_model, bluelock_wav, settings, tmp_path):
    """The real model on the real song: the doc's schema, a steady ~176 BPM, real sections."""
    result = analyse_song("bluelock", bluelock_wav, tmp_path, settings, beat_model)

    df = result.beats
    assert list(df.columns) == BEAT_COLUMNS
    assert not df.isna().any().any()
    assert 60 / df["ibi"].median() == pytest.approx(176, abs=3)
    assert df["section"].nunique() >= 3
    assert result.dense.values.shape[1] == 12
    pd.testing.assert_frame_equal(load_analysis("bluelock", tmp_path, settings).beats, df)
