"""Plot every song's music analysis and write a click track, the step 02 check (task 3.7).

Per song: one png with five panels on a shared time axis (waveform, onset, rms, rms_d1/_d2,
section novelty) and a stereo wav with the music on the left and clicks on the beats on the
right (higher and louder on downbeats). It also measures what is hard to hear: the median
offset between the beats and the onsets in the audio, and how hard the kick hits on each bar
position (position 0, the "1", should be strongest). Reads the step 02 cache; no beat model.

Run from the repository root, after scripts/analyse_music.py:
    python -m checks.plot_music
    python -m checks.plot_music --song "blue lock" --start 30 --end 40 --no-click

Output in data/out/music/: <name>.png (<name>_<start>-<end>s.png with a window) and
<name>_click.wav, always of the whole song.
"""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from amv.config import MUSIC, PATHS
from amv.music.analysis import load_analysis
from amv.music.audio import load_audio
from checks.music import (
    beat_offsets,
    kick_by_position,
    make_click_track,
    plot_song,
    recompute_sections,
    stale_boundaries,
    stereo_click_mix,
    waveform_envelope,
)
from checks.settings import PLOT_MUSIC
from scripts.common import configure_stdout, setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Args:
    """The parsed command line."""

    song: str | None
    start: float | None
    end: float | None
    no_click: bool
    song_table: Path
    cache_dir: Path
    out_dir: Path
    verbose: bool


def make_window(start: float | None, end: float | None) -> tuple[float, float] | None:
    """Turn --start/--end into a plot window: None without either, else (start or 0, end or inf)."""
    if start is None and end is None:
        return None
    return (0.0 if start is None else start, math.inf if end is None else end)


def select_songs(songs: pd.DataFrame, pattern: str | None) -> pd.DataFrame:
    """Return the rows whose name contains pattern (case-insensitive, literal); None returns all.

    Raises:
        ValueError: If no name matches."""
    if pattern is None:
        return songs
    matches = songs[songs["name"].str.contains(pattern, case=False, regex=False)]
    if len(matches) == 0:
        raise ValueError(f"No matching songs found for pattern: {pattern}")
    return matches


def output_stem(name: str, window: tuple[float, float] | None) -> str:
    """The png's file name without extension: the song's name, plus the window if there is one.

    "Song.mp4" -> "Song"; with (30, 40) -> "Song_30-40s"; with (30, inf) -> "Song_30s-end".
    A windowed plot so never overwrites the plot of the whole song."""
    stem = Path(name).stem
    if window is None:
        return stem
    start, end = window
    return f"{stem}_{start:g}s-end" if math.isinf(end) else f"{stem}_{start:g}-{end:g}s"


def plot_one(song_row: pd.Series, args: Args, window: tuple[float, float] | None) -> None:
    """Plot one song, measure its beat offset and kick pattern, and write its click track."""
    analysis = load_analysis(song_row["song_id"], args.cache_dir, MUSIC)
    beats = analysis.beats
    y, sr = load_audio(song_row["audio_proxy_path"], MUSIC.sample_rate)
    beat_times = beats["t"].to_numpy()

    envelope = waveform_envelope(y=y, sample_rate=sr, hop=MUSIC.hop_length)
    check = recompute_sections(y=y, sample_rate=sr, beats=beats, settings=MUSIC)
    boundaries = stale_boundaries(beats, check.boundaries)
    if len(boundaries) != 0:
        logger.warning("Stale bars: %s", boundaries)

    stem = Path(song_row["name"]).stem
    bpm = 60 / np.median(beats["ibi"])
    title = f"{stem}  ·  {bpm:.0f} BPM  ·  {MUSIC.settings_id[:8]}"
    fig = plot_song(
        title=title,
        analysis=analysis,
        envelope=envelope,
        check=check,
        prominence=MUSIC.novelty_prominence,
        edge_bars=MUSIC.edge_bars,
        window=window,
        fig_width=PLOT_MUSIC.fig_width,
        panel_height=PLOT_MUSIC.panel_height,
    )
    out_path = args.out_dir / f"{output_stem(song_row['name'], window)}.png"
    fig.savefig(out_path, dpi=PLOT_MUSIC.dpi)

    offsets = beat_offsets(
        y=y,
        sample_rate=sr,
        beat_times=beat_times,
        hop_length=PLOT_MUSIC.onset_hop_length,
        n_fft=PLOT_MUSIC.onset_n_fft,
        max_offset=PLOT_MUSIC.max_onset_offset,
        peak_percentile=PLOT_MUSIC.onset_peak_percentile,
    )
    kick = kick_by_position(
        y=y,
        sample_rate=sr,
        beat_times=beat_times,
        beat_in_bar=beats["beat_in_bar"].to_numpy(),
        beats_per_bar=MUSIC.beats_per_bar,
        hop_length=MUSIC.hop_length,
        n_fft=MUSIC.n_fft,
        max_hz=PLOT_MUSIC.kick_max_hz,
        window=PLOT_MUSIC.kick_window,
    )
    found = ~np.isnan(offsets)
    offset_ms = np.median(offsets[found]) * 1000 if found.any() else math.nan
    logger.info("%s (%.0f BPM)", stem, bpm)
    logger.info(
        "    beat offset %+.1f ms (%.0f%% of beats near an onset), kick by bar position [%s]",
        offset_ms,
        100 * found.mean(),
        " ".join(f"{k:.2f}" for k in kick),
    )

    if not args.no_click:
        clicks = make_click_track(
            beat_times=beat_times,
            is_downbeat=beats["is_downbeat"].to_numpy(),
            n_samples=len(y),
            sample_rate=sr,
            click_freq=PLOT_MUSIC.click_freq_hz,
            downbeat_click_freq=PLOT_MUSIC.downbeat_click_freq_hz,
            click_duration=PLOT_MUSIC.click_duration,
            beat_gain=PLOT_MUSIC.beat_click_gain,
            downbeat_gain=PLOT_MUSIC.downbeat_click_gain,
        )
        sf.write(args.out_dir / f"{stem}_click.wav", stereo_click_mix(y, clicks), sr)

    logger.debug("Wrote %s", out_path)


def parse_args(argv: list[str] | None = None) -> Args:
    """Parse the command line; exits with a usage message on a missing song table or a bad window.

    Args:
        argv (list[str] | None): The arguments without the program name, or None to read
            sys.argv.

    Returns:
        Args: The parsed arguments. Defaults come from amv/config.py and checks/settings.py."""
    parser = argparse.ArgumentParser(
        prog="python -m checks.plot_music",
        description=(
            "Plot the music analysis of every song (or one, with --song), measure the beat offset "
            "and kick pattern, and write a click track: music left, clicks right."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--song", type=str, default=None, help="part of the song's name, case-insensitive")
    parser.add_argument("--start", type=float, default=None, help="plot window start, in seconds")
    parser.add_argument("--end", type=float, default=None, help="plot window end, in seconds")
    parser.add_argument("--no-click", action="store_true", help="don't write the click track")
    parser.add_argument("--song-table", type=Path, default=PATHS.song_table_path, help="song table")
    parser.add_argument("--cache-dir", type=Path, default=PATHS.music_cache_dir, help="music cache")
    parser.add_argument("--out-dir", type=Path, default=PLOT_MUSIC.out_dir, help="where the png and wav go")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug messages of the amv package")
    args = Args(**vars(parser.parse_args(argv)))

    if not args.song_table.exists():
        parser.error(
            f"Song table does not exist: {args.song_table}. Run `python -m scripts.build_song_table` to create."
        )
    if args.start is not None and args.start < 0:
        parser.error(f"Start needs to be greater or equal than 0, was {args.start}.")
    if args.start is not None and args.end is not None and args.end <= args.start:
        parser.error(f"End needs to be greater than start. Start ({args.start}) >= End ({args.end}).")
    return args


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments, set up logging and plot every selected song into the output folder.

    Args:
        argv (list[str] | None): The arguments without the program name, or None to read
            sys.argv."""
    args = parse_args(argv)
    setup_logging(args.verbose)

    songs = pd.read_parquet(args.song_table)
    selected_songs = select_songs(songs, args.song)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    window = make_window(args.start, args.end)
    for _, row in selected_songs.iterrows():
        plot_one(row, args, window)
    logger.info("Plotted %d songs into %s", len(selected_songs), args.out_dir)


if __name__ == "__main__":
    configure_stdout()
    main()
