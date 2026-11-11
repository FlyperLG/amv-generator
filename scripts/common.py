"""Shared command-line plumbing for the entry points in scripts/."""

import logging
import sys


def setup_logging(verbose: bool, quiet: tuple[str, ...] = ()) -> None:
    """Log INFO and above to stderr; verbose turns on DEBUG for the amv package only.

    Third-party libraries stay at INFO either way (numba and matplotlib flood DEBUG).
    Without verbose, the loggers in quiet are limited to warnings, e.g. "amv.io.ffmpeg"
    to hide its "skipping" messages.

    Args:
        verbose (bool): Whether to show the amv package's DEBUG messages.
        quiet (tuple[str, ...]): Logger names limited to warnings when not verbose."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if verbose:
        logging.getLogger("amv").setLevel(logging.DEBUG)
    else:
        for name in quiet:
            logging.getLogger(name).setLevel(logging.WARNING)


def configure_stdout() -> None:
    """Write stdout as UTF-8, so non-ASCII file names print on Windows (cp1252 by default)."""
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
