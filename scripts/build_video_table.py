"""Build the video table for step 01 (docs/01-decoding-and-proxies, task 3.6).

Makes a proxy of every video in the AMV and library folders, then writes one row per
video to the video table that all later steps read. Finished proxies are skipped,
so re-running after adding videos only processes the new ones.

Run from the repository root:
    python -m scripts.build_video_table
    python -m scripts.build_video_table -v --library-dir D:/anime

Output: data/cache/videos.parquet, proxies in data/proxies/. The AMVs' wavs come from
scripts/build_song_table.py.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from amv.config import AMV_DIR, LIBRARY_DIR, PROXY_DIR, VIDEO_TABLE_PATH
from amv.io.videos import build_video_table


@dataclass(frozen=True, slots=True)
class Args:
    """The parsed command line."""

    amv_dir: Path
    library_dir: Path
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
        prog="python -m scripts.build_video_table",
        description=(
            "Make low-res proxies of every AMV and library video and "
            "write the video table that all later steps read. Finished proxies are skipped."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--amv-dir", type=Path, default=AMV_DIR, help="folder with the reference AMVs")
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=LIBRARY_DIR,
        help="folder with the source footage, one sub-folder per series",
    )
    parser.add_argument("--proxy-dir", type=Path, default=PROXY_DIR, help="proxy cache, created if missing")
    parser.add_argument("--out-path", type=Path, default=VIDEO_TABLE_PATH, help="parquet file to write")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show debug messages of the amv package, including skipped proxies",
    )
    args = Args(**vars(parser.parse_args(argv)))

    if not args.amv_dir.exists():
        parser.error(f"Input amv directory does not exist: {args.amv_dir}")
    if not args.library_dir.exists():
        parser.error(f"Input library directory does not exist: {args.library_dir}")
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
    """Parse the arguments, set up logging and build the video table.

    Args:
        argv (list[str] | None): The arguments without the program name, or None to read
            sys.argv."""
    args = parse_args(argv)
    setup_logging(args.verbose)

    build_video_table(
        amv_dir=args.amv_dir,
        library_dir=args.library_dir,
        proxy_dir=args.proxy_dir,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    main()
