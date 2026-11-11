import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from beat_this.inference import Audio2Beats
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from amv.config import MusicSettings
from amv.music.audio import load_audio
from amv.music.beats import BeatGrid, build_beat_grid, clean_beats, detect_beats
from amv.music.dense import DenseFeatures, compute_dense_features, compute_timbre_features
from amv.music.dynamics import beat_dynamics, pool_to_beats, smooth_dense
from amv.music.sections import find_section_boundaries, label_sections, section_novelty

logger = logging.getLogger(__name__)

_SONG_COLUMNS = ("song_id", "audio_proxy_path")  # what analyse_songs reads from the song table


@dataclass(frozen=True, slots=True)
class SongAnalysis:
    """Step 02's output for one song: what later steps read from the music cache.

    Attributes:
        beats (pd.DataFrame): The beat table, one row per beat (see build_beat_table).
        dense (DenseFeatures): The per-frame curves: the 6 normalised curves of
            compute_dense_features, then their smoothed "_s" versions, 12 columns."""

    beats: pd.DataFrame  # one row per beat, see build_beat_table
    dense: DenseFeatures  # 6 raw curves, then their 6 _s versions


def build_beat_table(grid: BeatGrid, dynamics: dict[str, np.ndarray], sections: np.ndarray) -> pd.DataFrame:
    """Put the beat grid, the beat curves and the section labels into one table.

    ibi is the gap to the next beat; the last beat repeats the gap before it, the same rule
    pool_to_beats uses for the last beat's interval.

    Args:
        grid (BeatGrid): The beat grid. Not modified.
        dynamics (dict[str, np.ndarray]): The beat curves from beat_dynamics, each of shape
            [B]; they become columns in this order. Not modified.
        sections (np.ndarray): The section label of every beat, shape [B]. Not modified.

    Returns:
        pd.DataFrame: One row per beat with the columns t (float64), is_downbeat (bool),
            beat_in_bar, bar_idx (int64), ibi (float64), then the dynamics, then section (str).

    Raises:
        ValueError: If there are fewer than 2 beats, or sections or a curve doesn't have one
            value per beat."""
    num_beats = grid.times.shape[0]
    if num_beats < 2:
        raise ValueError(f"Song needs to have atleast 2 beats, got {num_beats}")

    if len(sections) != num_beats or any(len(d) != num_beats for d in dynamics.values()):
        raise ValueError(f"All inputs need to have the same number of beats: {num_beats}")

    ibi = np.diff(grid.times)
    ibi = np.append(ibi, ibi[-1])
    data = {
        "t": grid.times.astype(np.float64),
        "is_downbeat": grid.is_downbeat.astype(bool),
        "beat_in_bar": grid.beat_in_bar.astype(np.int64),
        "bar_idx": grid.bar_idx.astype(np.int64),
        "ibi": ibi.astype(np.float64),
        **dynamics,
        "section": sections.astype(str),
    }
    return pd.DataFrame(data)


def cache_paths(song_id: str, cache_dir: str | os.PathLike, settings: MusicSettings) -> tuple[Path, Path]:
    """Return where a song's analysis is cached with these settings, without creating anything.

    The settings_id is part of both names, so changing a setting gives new files instead of
    reusing stale ones. Old files stay until deleted by hand.

    Args:
        song_id (str): The song's id from the song table.
        cache_dir (str | os.PathLike): The music cache folder. It does not have to exist.
        settings (MusicSettings): The settings the analysis is made with.

    Returns:
        tuple[Path, Path]: (cache_dir / "<song_id>_<settings_id>_beats.parquet",
            cache_dir / "<song_id>_<settings_id>_dense.npz")."""
    cache_dir = Path(cache_dir)
    return (
        cache_dir / f"{song_id}_{settings.settings_id}_beats.parquet",
        cache_dir / f"{song_id}_{settings.settings_id}_dense.npz",
    )


def _save(analysis: SongAnalysis, paths: tuple[Path, Path], settings: MusicSettings) -> None:
    """Write an analysis to its two cache files, creating the folder if missing.

    First "settings_<settings_id>.json" with the settings, so a human can tell what an id
    stands for. Each data file is written to a temp name and then renamed, so a crash never
    leaves a half-written file at the final path. The npz is written without pickle.

    Args:
        analysis (SongAnalysis): The analysis to write.
        paths (tuple[Path, Path]): The beat table and dense paths from cache_paths.
        settings (MusicSettings): The settings the analysis was made with."""
    beats_path, dense_path = paths
    cache_dir = beats_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = cache_dir / f"settings_{settings.settings_id}.json"
    with open(sidecar_path, "w", encoding="utf-8") as sidecar_json:
        json.dump(dataclasses.asdict(settings), sidecar_json, indent=4)

    tmp_beats_path = beats_path.with_suffix(".tmp.parquet")
    analysis.beats.to_parquet(tmp_beats_path)
    os.replace(tmp_beats_path, beats_path)

    tmp_dense_path = dense_path.with_suffix(".tmp.npz")
    np.savez(
        tmp_dense_path,
        times=analysis.dense.times,
        values=analysis.dense.values,
        names=np.array(analysis.dense.names),
    )
    os.replace(tmp_dense_path, dense_path)


def load_analysis(song_id: str, cache_dir: str | os.PathLike, settings: MusicSettings) -> SongAnalysis:
    """Read a song's analysis back from the cache, as later steps do.

    Args:
        song_id (str): The song's id from the song table.
        cache_dir (str | os.PathLike): The music cache folder.
        settings (MusicSettings): The settings the analysis was made with; they pick the files.

    Returns:
        SongAnalysis: Exactly what analyse_song computed. The dense names are a tuple of str.

    Raises:
        FileNotFoundError: If either cache file is missing."""
    beats_path, dense_path = cache_paths(song_id, cache_dir, settings)
    if not beats_path.exists():
        raise FileNotFoundError(f"Unable to load beat features: {beats_path}")
    if not dense_path.exists():
        raise FileNotFoundError(f"Unable to load dense music features: {dense_path}")

    with np.load(dense_path, allow_pickle=False) as z:
        dense = DenseFeatures(times=z["times"], values=z["values"], names=tuple(str(n) for n in z["names"]))
    beats = pd.read_parquet(beats_path)
    return SongAnalysis(
        beats=beats,
        dense=dense,
    )


def _analyse(y: np.ndarray, sample_rate: int, settings: MusicSettings, model: Audio2Beats) -> SongAnalysis:
    """Run tasks 3.2-3.5 on a loaded signal; no file is read or written.

    Beats -> beat grid; dense curves -> smoothed curves (window from the median IBI) and
    beat curves (the raw curves pooled per beat, then smoothed and differentiated); timbre
    pooled per bar -> novelty -> section boundaries -> one label per beat.

    Args:
        y (np.ndarray): The mono signal from load_audio. Not modified.
        sample_rate (int): The sample rate of y in Hz.
        settings (MusicSettings): The step 02 settings, passed on to every step.
        model (Audio2Beats): The beat model from load_beat_model.

    Returns:
        SongAnalysis: The beat table and the 12 dense curves.

    Raises:
        ValueError: From the steps, e.g. for a song with too few beats or bars."""
    beats, downbeats = detect_beats(y=y, sample_rate=sample_rate, model=model)
    grid = build_beat_grid(
        beats=clean_beats(beats, settings.min_ibi),
        downbeats=downbeats,
        beats_per_bar=settings.beats_per_bar,
        tol=settings.downbeat_tol,
        window_bars=settings.downbeat_window_bars,
    )
    dense = compute_dense_features(
        y=y,
        sample_rate=sample_rate,
        hop_length=settings.hop_length,
        n_fft=settings.n_fft,
        n_mels=settings.n_mels,
        low_band_hz=settings.low_band_hz,
        high_band_hz=settings.high_band_hz,
    )
    dense_s = smooth_dense(
        dense=dense,
        beat_period=float(np.median(np.diff(grid.times))),
        window_beats=settings.dense_sg_window_beats,
        polyorder=settings.sg_polyorder,
    )
    dyn = beat_dynamics(
        pooled=pool_to_beats(dense=dense, beat_times=grid.times),
        names=dense.names,
        window=settings.beat_sg_window,
        polyorder=settings.sg_polyorder,
    )
    timbre = compute_timbre_features(
        y=y,
        sample_rate=sample_rate,
        hop_length=settings.hop_length,
        n_fft=settings.n_fft,
        n_mels=settings.n_mels,
        n_mfcc=settings.n_mfcc,
    )
    bars = pool_to_beats(dense=timbre, beat_times=grid.times[grid.is_downbeat])
    bounds = find_section_boundaries(
        novelty=section_novelty(bar_features=bars, kernel_bars=settings.section_kernel_bars),
        min_section_bars=settings.min_section_bars,
        prominence=settings.novelty_prominence,
    )
    labels = label_sections(
        grid=grid,
        rms_s=dyn["rms_s"],
        rms_d1=dyn["rms_d1"],
        boundaries=bounds,
        drop_level=settings.drop_level,
        build_rise=settings.build_rise,
    )
    return SongAnalysis(
        beats=build_beat_table(grid=grid, dynamics=dyn, sections=labels),
        dense=DenseFeatures(
            times=dense.times,
            values=np.hstack([dense.values, dense_s.values]),
            names=dense.names + dense_s.names,
        ),
    )


def _is_cached(paths: tuple[Path, Path], audio_path: Path) -> bool:
    """Whether both cache files exist and are newer than the wav (the skip rule of step 01)."""
    beats_path, dense_path = paths
    if not (beats_path.exists() and dense_path.exists()):
        return False
    audio_mtime = audio_path.stat().st_mtime_ns
    return audio_mtime < beats_path.stat().st_mtime_ns and audio_mtime < dense_path.stat().st_mtime_ns


def _compute_and_save(
    song_id: str,
    audio_path: Path,
    paths: tuple[Path, Path],
    settings: MusicSettings,
    model: Audio2Beats,
) -> SongAnalysis:
    """Load the wav, analyse it and write the cache files; logs the time taken at debug level.

    Everything is computed before anything is written, so a failing song leaves no files."""
    start = time.perf_counter()
    audio, sample_rate = load_audio(audio_path, settings.sample_rate)
    analysis = _analyse(y=audio, sample_rate=sample_rate, settings=settings, model=model)
    _save(analysis=analysis, paths=paths, settings=settings)
    logger.debug("Analysed song %s in %.1f s.", song_id, time.perf_counter() - start)
    return analysis


def analyse_song(
    song_id: str,
    audio_path: str | os.PathLike,
    cache_dir: str | os.PathLike,
    settings: MusicSettings,
    model: Audio2Beats,
) -> SongAnalysis:
    """Get one song's analysis: from the cache if it is up to date, otherwise computed and cached.

    The cache is up to date if both files exist and are newer than the wav; then the model is
    not called.

    Args:
        song_id (str): The song's id from the song table; it names the cache files.
        audio_path (str | os.PathLike): The song's wav, as made by step 01.
        cache_dir (str | os.PathLike): The music cache folder; created if missing.
        settings (MusicSettings): The step 02 settings; they also pick the cache files.
        model (Audio2Beats): The beat model from load_beat_model, loaded once by the caller.

    Returns:
        SongAnalysis: The song's beat table and dense curves.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ValueError: From the steps, e.g. a wav with the wrong sample rate or too few beats.
            No cache file is written then."""
    audio_path = Path(audio_path)
    paths = cache_paths(song_id=song_id, cache_dir=cache_dir, settings=settings)
    if _is_cached(paths=paths, audio_path=audio_path):
        logger.debug("Loaded song %s from the cache.", song_id)
        return load_analysis(song_id=song_id, cache_dir=cache_dir, settings=settings)
    return _compute_and_save(
        song_id=song_id, audio_path=audio_path, paths=paths, settings=settings, model=model
    )


def analyse_songs(
    songs: pd.DataFrame,
    cache_dir: str | os.PathLike,
    settings: MusicSettings,
    model: Audio2Beats,
) -> None:
    """Make sure every song of the song table has an up-to-date analysis in the cache (task 3.6).

    Songs whose cache is up to date are skipped without loading them; the others are analysed
    one at a time with a progress bar. Later steps read the results with load_analysis.

    Args:
        songs (pd.DataFrame): The song table from step 01. Only song_id and
            audio_proxy_path are used; other columns are ignored.
        cache_dir (str | os.PathLike): The music cache folder; created if missing.
        settings (MusicSettings): The step 02 settings.
        model (Audio2Beats): The beat model from load_beat_model.

    Raises:
        ValueError: If songs lacks one of the two columns (before any song is analysed), or
            from the steps. An error stops the run; songs finished before it stay cached.
        FileNotFoundError: If a song's wav does not exist."""
    missing = [c for c in _SONG_COLUMNS if c not in songs.columns]
    if missing:
        raise ValueError(f"song table is missing columns: {missing}")

    with logging_redirect_tqdm():
        bar = tqdm(
            zip(songs["song_id"], songs["audio_proxy_path"], strict=True),
            total=len(songs),
            desc="Analysing songs",
            unit="song",
            dynamic_ncols=True,
            disable=None,
        )

        n_analysed = 0
        n_cached = 0
        start = time.perf_counter()
        for raw_song_id, audio_proxy_path in bar:
            song_id = str(raw_song_id)
            audio_path = Path(audio_proxy_path)
            bar.set_postfix_str(song_id[:8])
            paths = cache_paths(song_id=song_id, cache_dir=cache_dir, settings=settings)
            if _is_cached(paths=paths, audio_path=audio_path):
                n_cached += 1
                logger.debug("Skipped song %s: cache is up to date.", song_id)
                continue

            _compute_and_save(
                song_id=song_id, audio_path=audio_path, paths=paths, settings=settings, model=model
            )
            n_analysed += 1

        logger.info(
            "Music analysis done in %.1f s: %d songs analysed, %d already cached, in %s.",
            time.perf_counter() - start,
            n_analysed,
            n_cached,
            cache_dir,
        )
