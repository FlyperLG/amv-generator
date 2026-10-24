"""Benchmark results: the BenchResult type, its JSON file format, and the console table.

A run file holds each MediaInfo once under "sources"; every result refers to its media by
key. See tests/benchmarks/test_results.py for the full specification.
"""

import dataclasses
import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from tabulate import tabulate

from amv.io.media import MediaInfo
from benchmarks.machine import get_machine_information


@dataclass(frozen=True, slots=True)
class BenchResult:
    """The timings of one benchmark, plus the values derived from them.

    All derived values (median, x_realtime, ...) are properties computed from seconds,
    so they can never disagree with the measurements.

    Attributes:
        name (str): The benchmark's name.
        input (MediaInfo): The file that was processed.
        seconds (list[float]): Wall-clock duration of each measured repeat, in order.
        n_frames (int): Frames processed per repeat.
        media_seconds (float): Duration of the processed media, the reference for
            "x real time"."""

    name: str
    input: MediaInfo
    seconds: list[float]
    n_frames: int
    media_seconds: float

    @property
    def median(self) -> float:
        return statistics.median(self.seconds)

    @property
    def min(self) -> float:
        return min(self.seconds)

    @property
    def max(self) -> float:
        return max(self.seconds)

    @property
    def x_realtime(self) -> float:
        """How many times faster than real time the median repeat ran (media seconds / median)."""
        return self.media_seconds / self.median

    @property
    def fps(self) -> float:
        """Frames processed per second of the median repeat."""
        return self.n_frames / self.median

    @property
    def x_realtime_range(self) -> tuple[float, float]:
        """(slowest, fastest) x real time over all repeats: from the max and the min seconds."""
        return self.media_seconds / self.max, self.media_seconds / self.min

    def summarize(self) -> tuple[str, float, float, int, float, float]:
        """One table row: (name, median s, min s, frames, fps, x real time), as print_table expects."""
        return (self.name, self.median, self.min, self.n_frames, self.fps, self.x_realtime)

    @classmethod
    def from_dict(cls, data: dict[str, Any], sources: dict[str, MediaInfo]) -> Self:
        """Rebuild a result from one entry of a run file's "results" (the inverse of to_dict).

        Derived values stored in data (median_s, fps, ...) are ignored; the properties
        recompute them from "seconds".

        Args:
            data (dict[str, Any]): One result as written by to_dict, after a JSON round trip.
            sources (dict[str, MediaInfo]): The run's sources; data["source"] is a key into it.
                Results with the same key get the same MediaInfo object.

        Returns:
            Self: The rebuilt result.

        Raises:
            KeyError: If a field is missing or data["source"] is not in sources."""
        return cls(
            name=data["name"],
            input=sources[data["source"]],
            seconds=data["seconds"],
            n_frames=data["n_frames"],
            media_seconds=data["media_seconds"],
        )

    def to_dict(self, sources: dict[str, MediaInfo]) -> dict[str, Any]:
        """Convert to a JSON-ready dict that refers to its media by key instead of embedding it.

        The key is looked up by identity (is), so two sources with equal metadata stay
        distinguishable. The derived values are added for people reading the file.

        Args:
            sources (dict[str, MediaInfo]): The run's sources; must contain this result's
                input object.

        Returns:
            dict[str, Any]: The fields except input, plus "source" (the key) and median_s,
                min_s, max_s, fps, x_realtime, x_realtime_range.

        Raises:
            ValueError: If this result's MediaInfo is not one of the objects in sources."""
        source = next((key for key, media in sources.items() if media is self.input), None)
        if source is None:
            raise ValueError(f"{self.name}: its MediaInfo is not in sources")
        result = {
            **dataclasses.asdict(self),
            "median_s": self.median,
            "min_s": self.min,
            "max_s": self.max,
            "fps": self.fps,
            "x_realtime": self.x_realtime,
            "x_realtime_range": self.x_realtime_range,
        }
        result.pop("input")
        result["source"] = source
        return result


def load_run(path: Path) -> tuple[list[BenchResult], dict[str, Any]]:
    """Read a run file written by save_run.

    Each source is rebuilt once, so results that shared a MediaInfo share one object
    again and the run can be saved again.

    Args:
        path (Path): The run's JSON file.

    Returns:
        tuple[list[BenchResult], dict[str, Any]]: (results, summary) where
            results are in file order;
            summary is the run's summary dict, unchanged (its keys belong to the bench script).

    Raises:
        FileNotFoundError: If path does not exist.
        json.JSONDecodeError: If the file is not valid JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sources: dict[str, MediaInfo] = {key: MediaInfo.from_dict(d) for key, d in data["sources"].items()}
    results = [BenchResult.from_dict(data=result, sources=sources) for result in data["results"]]
    return results, data["summary"]


def save_run(
    path: Path,
    created: datetime,
    args: dict[str, Any],
    sources: dict[str, MediaInfo],
    results: list[BenchResult],
    summary: dict[str, Any],
) -> None:
    """Write one benchmark run to a JSON file, creating missing parent directories.

    Top level: created, machine, args, sources, results, summary. Each MediaInfo is stored
    once under "sources"; results refer to it by key (see BenchResult.to_dict). Values JSON
    cannot hold (Path, Fraction) are written as strings.

    Args:
        path (Path): The file to write; overwritten if it exists.
        created (datetime): When the run started; stored as ISO 8601 to the second.
        args (dict[str, Any]): The settings the run used, e.g. dataclasses.asdict(args).
        sources (dict[str, MediaInfo]): Every media file a result refers to, by key.
        results (list[BenchResult]): The measured results.
        summary (dict[str, Any]): Script-specific conclusions, e.g. a speedup.

    Raises:
        ValueError: If a result's MediaInfo is not in sources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    json_result = {
        "created": created.isoformat(timespec="seconds"),
        "machine": get_machine_information(),
        "args": args,
        "sources": {name: dataclasses.asdict(media) for name, media in sources.items()},
        "results": [result.to_dict(sources) for result in results],
        "summary": summary,
    }

    with open(path, "w", encoding="utf-8") as out_json:
        json.dump(json_result, out_json, default=str, indent=4)


def print_table(results: list[BenchResult]) -> None:
    """Print one row per result (BenchResult.summarize) as a table, floats to 2 decimals."""
    results_summaries = [result.summarize() for result in results]
    print(
        tabulate(
            results_summaries,
            headers=["name", "median s", "min s", "frames", "fps", "x real time"],
            tablefmt="orgtbl",
            floatfmt=".2f",
        )
    )
