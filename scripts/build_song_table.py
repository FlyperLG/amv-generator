"""Build the song table for step 01 (docs/01-decoding-and-proxies, task 3.7).

Makes a 22050 Hz mono wav of every AMV and every song, then writes one row per file to
the song table that the music analysis reads. Finished wavs are skipped, so re-running
after adding songs only processes the new ones. Independent of build_video_table: the
two can run in either order.

Run from the repository root:
    python -m scripts.build_song_table
    python -m scripts.build_song_table -v --song-dir D:/music

Output: data/cache/songs.parquet, wavs in data/proxies/.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from amv.config import DECODING, PATHS
from amv.io.songs import build_song_table
from scripts.common import configure_stdout, setup_logging


@dataclass(frozen=True, slots=True)
class Args:
    """The parsed command line."""

    amv_dir: Path
    song_dir: Path
    proxy_dir: Path
    out_path: Path
    verbose: bool


def parse_args(argv: list[str] | None = None) -> Args:
    """Parse the command line; exits with a usage message if an input folder is missing.

    Args:
        argv (list[str] | None): The arguments without the program name, or None to read
            sys.argv.

    Returns:
        Args: The parsed arguments. Defaults come from amv/config.py."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.build_song_table",
        description=(
            "Make a 22050 Hz mono wav of every AMV and song and write the song table "
            "that the music analysis reads. Finished wavs are skipped."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--amv-dir", type=Path, default=PATHS.amv_dir, help="folder with the reference AMVs")
    parser.add_argument(
        "--song-dir",
        type=Path,
        default=PATHS.song_dir,
        help="folder with the input songs (.mp3, .wav, .flac)",
    )
    parser.add_argument(
        "--proxy-dir", type=Path, default=PATHS.proxy_dir, help="proxy cache, created if missing"
    )
    parser.add_argument("--out-path", type=Path, default=PATHS.song_table_path, help="parquet file to write")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show debug messages of the amv package, including skipped wavs",
    )
    args = Args(**vars(parser.parse_args(argv)))

    if not args.amv_dir.exists():
        parser.error(f"Input amv directory does not exist: {args.amv_dir}")
    if not args.song_dir.exists():
        parser.error(f"Input song directory does not exist: {args.song_dir}")
    return args


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments, set up logging and build the song table.

    Args:
        argv (list[str] | None): The arguments without the program name, or None to read
            sys.argv."""
    args = parse_args(argv)
    setup_logging(verbose=args.verbose, quiet=("amv.io.ffmpeg",))

    build_song_table(
        amv_dir=args.amv_dir,
        song_dir=args.song_dir,
        proxy_dir=args.proxy_dir,
        out_path=args.out_path,
        settings=DECODING,
    )


if __name__ == "__main__":
    configure_stdout()
    main()
