"""Analyse every song of the song table for step 02 (docs/02-music-features, task 3.6).

Runs beat tracking, the dense and beat curves and the section labels on the wav of every
AMV and song, and caches a beat table and the dense curves per song. Songs whose cache is
newer than their wav are skipped, so re-running after adding songs only analyses the new
ones. Changing a setting in MUSIC gives new cache names, so everything is analysed again.

Run from the repository root, after scripts/build_song_table.py:
    python -m scripts.analyse_music
    python -m scripts.analyse_music -v

Output: data/cache/music/<song_id>_<settings_id>_beats.parquet and _dense.npz per song,
plus settings_<settings_id>.json. Later steps read them with amv.music.analysis.load_analysis.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amv.config import MUSIC, PATHS
from amv.music.analysis import analyse_songs
from amv.music.beats import load_beat_model
from scripts.common import configure_stdout, setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Args:
    """The parsed command line."""

    song_table: Path
    cache_dir: Path
    verbose: bool


def parse_args(argv: list[str] | None = None) -> Args:
    """Parse the command line; exits with a usage message if the song table is missing.

    Args:
        argv (list[str] | None): The arguments without the program name, or None to read
            sys.argv.

    Returns:
        Args: The parsed arguments. Defaults come from amv/config.py."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.analyse_music",
        description=(
            "Analyse the wav of every AMV and song in the song table (beats, energy curves, "
            "sections) and cache the results. Songs with an up-to-date cache are skipped."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--song-table",
        type=Path,
        default=PATHS.song_table_path,
        help="song table from scripts.build_song_table",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PATHS.music_cache_dir, help="music cache, created if missing"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show debug messages of the amv package, including the time per song",
    )
    args = Args(**vars(parser.parse_args(argv)))

    if not args.song_table.exists():
        parser.error(
            f"Song table does not exist: {args.song_table}. Run `python -m scripts.build_song_table` first."
        )
    return args


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments, set up logging, load the beat model once and analyse every song.

    The model is loaded even if every song is cached (a few seconds), so analyse_songs gets
    one model for the whole run.

    Args:
        argv (list[str] | None): The arguments without the program name, or None to read
            sys.argv."""
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    songs = pd.read_parquet(args.song_table)
    logger.info("Loading beat model %s ...", MUSIC.beat_checkpoint)
    model = load_beat_model(MUSIC.beat_checkpoint)

    analyse_songs(
        songs=songs,
        cache_dir=args.cache_dir,
        settings=MUSIC,
        model=model,
    )


if __name__ == "__main__":
    configure_stdout()
    main()
