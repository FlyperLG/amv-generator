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
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from amv.config import AMV_DIR, PROXY_DIR, SONG_DIR, SONG_TABLE_PATH
from amv.io.songs import build_song_table


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
    parser.add_argument("--amv-dir", type=Path, default=AMV_DIR, help="folder with the reference AMVs")
    parser.add_argument(
        "--song-dir",
        type=Path,
        default=SONG_DIR,
        help="folder with the input songs (.mp3, .wav, .flac)",
    )
    parser.add_argument("--proxy-dir", type=Path, default=PROXY_DIR, help="proxy cache, created if missing")
    parser.add_argument("--out-path", type=Path, default=SONG_TABLE_PATH, help="parquet file to write")
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


def setup_logging(verbose: bool) -> None:
    """Log INFO and above to stderr; verbose turns on DEBUG for the amv package only.

    Third-party libraries stay at INFO either way (numba and matplotlib flood DEBUG).
    Without verbose, amv.io.ffmpeg is limited to warnings, which hides its "skipping"
    messages.

    Args:
        verbose (bool): Whether to show the amv package's DEBUG messages."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if verbose:
        logging.getLogger("amv").setLevel(logging.DEBUG)
    else:
        logging.getLogger("amv.io.ffmpeg").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments, set up logging and build the song table.

    Args:
        argv (list[str] | None): The arguments without the program name, or None to read
            sys.argv."""
    args = parse_args(argv)
    setup_logging(args.verbose)

    build_song_table(
        amv_dir=args.amv_dir,
        song_dir=args.song_dir,
        proxy_dir=args.proxy_dir,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    main()
