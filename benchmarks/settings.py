"""Settings shared by every benchmark (bench_* scripts).

Same pattern as amv/config.py, but kept here because amv/config.py is the config of the
pipeline only: a frozen dataclass whose defaults are the configured values, plus one
instance, BENCH. The bench_* scripts use the values as argparse defaults, so the command
line overrides them. A setting of one benchmark only stays an argparse default in its script.
"""

from dataclasses import dataclass
from pathlib import Path

from amv.config import DATA_DIR


@dataclass(frozen=True, slots=True)
class BenchSettings:
    """What every benchmark shares: where runs are written and how often each is timed."""

    out_dir: Path = DATA_DIR / "out" / "bench"  # benchmark runs (json + png)
    warmup: int = 1  # unmeasured runs before each benchmark
    repeat: int = 5  # measured runs per benchmark


BENCH = BenchSettings()
