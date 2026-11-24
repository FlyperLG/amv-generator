"""Benchmark for step 02 (docs/02-music-features, task 3.8): what does analysing a song cost?

Measures, on one song's wav (no file I/O inside any timing):
  - model_load:   loading the beat_this model (cache cleared first), a one-time cost
  - analyse_cold: the whole analysis (_analyse) on its first call in this process, incl.
                  numba JIT and torch warm-up; with model_load it is the first song's cost
  - beats, dense, timbre, dynamics, sections: each stage of _analyse on its own
  - analyse:      the whole analysis again, warm
Model load and the cold run are measured once; the rest with --warmup / --repeat. Before
timing, the stage outputs are checked to give the same beat table as _analyse, so the
stages measure what the pipeline runs.

Prints the table, the time per song against the target (< 15 s for a 3-min song, i.e.
>= 12x real time), the first-song cost and each stage's share.

Run from the repository root:
    python -m benchmarks.bench_02_music --plot
    python -m benchmarks.bench_02_music --song "heroes" --threads 2
    python -m benchmarks.bench_02_music --from-json data/out/bench/<run>.json

Output: data/out/bench/bench_02_music_<timestamp>.json (+ .png with --plot).
"""

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from beat_this.inference import Audio2Beats
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from amv.config import DECODING, MUSIC, PATHS, MusicSettings
from amv.io.ffmpeg import probe
from amv.music.analysis import SongAnalysis, _analyse, build_beat_table
from amv.music.audio import load_audio
from amv.music.beats import BeatGrid, build_beat_grid, clean_beats, detect_beats, load_beat_model
from amv.music.dense import DenseFeatures, compute_dense_features, compute_timbre_features
from amv.music.dynamics import beat_dynamics, pool_to_beats, smooth_dense
from amv.music.sections import find_section_boundaries, label_sections, section_novelty
from benchmarks.plots import plot_throughput
from benchmarks.results import BenchResult, load_run, print_table, save_run
from benchmarks.settings import BENCH
from benchmarks.timing import benchmark
from scripts.common import configure_stdout

ORIGINAL_SONG = "blue lock"
TARGET_SECONDS, TARGET_SONG_SECONDS = 15.0, 180.0
TARGET_X_REALTIME = TARGET_SONG_SECONDS / TARGET_SECONDS


class Bench(StrEnum):
    MODEL_LOAD = "model_load"
    ANALYSE_COLD = "analyse_cold"
    BEATS = "beats"
    DENSE = "dense"
    TIMBRE = "timbre"
    DYNAMICS = "dynamics"
    SECTIONS = "sections"
    ANALYSE = "analyse"


PROCESS_STAGES = [Bench.BEATS, Bench.DENSE, Bench.TIMBRE, Bench.DYNAMICS, Bench.SECTIONS]


COLORS: dict[str, str] = {
    Bench.MODEL_LOAD: "tab:orange",
    Bench.ANALYSE_COLD: "tab:blue",
    Bench.BEATS: "tab:red",
    Bench.DENSE: "tab:green",
    Bench.TIMBRE: "tab:purple",
    Bench.DYNAMICS: "tab:brown",
    Bench.SECTIONS: "tab:pink",
    Bench.ANALYSE: "tab:cyan",
}


def _load_model(checkpoint: str) -> int:
    """Load the beat model with load_beat_model's cache cleared, so the load is really timed.

    Returns 1: there is nothing to count, but time_it needs an int."""
    load_beat_model.cache_clear()
    load_beat_model(checkpoint)
    return 1


def _run_beats(y: np.ndarray, sample_rate: int, model: Audio2Beats, settings: MusicSettings) -> BeatGrid:
    """Stage "beats": detect_beats, clean_beats and build_beat_grid, as in _analyse."""
    beats, downbeats = detect_beats(y=y, sample_rate=sample_rate, model=model)
    return build_beat_grid(
        beats=clean_beats(beats, settings.min_ibi),
        downbeats=downbeats,
        beats_per_bar=settings.beats_per_bar,
        tol=settings.downbeat_tol,
        window_bars=settings.downbeat_window_bars,
    )


def _run_dense(y: np.ndarray, sample_rate: int, settings: MusicSettings) -> DenseFeatures:
    """Stage "dense": the 6 normalised energy curves (compute_dense_features), shape [T, 6]."""
    return compute_dense_features(
        y=y,
        sample_rate=sample_rate,
        hop_length=settings.hop_length,
        n_fft=settings.n_fft,
        n_mels=settings.n_mels,
        low_band_hz=settings.low_band_hz,
        high_band_hz=settings.high_band_hz,
    )


def _run_timbre(y: np.ndarray, sample_rate: int, settings: MusicSettings) -> DenseFeatures:
    """Stage "timbre": MFCC and chroma per frame (compute_timbre_features), for the sections."""
    return compute_timbre_features(
        y=y,
        sample_rate=sample_rate,
        hop_length=settings.hop_length,
        n_fft=settings.n_fft,
        n_mels=settings.n_mels,
        n_mfcc=settings.n_mfcc,
    )


def _run_dynamics(dense: DenseFeatures, grid: BeatGrid, settings: MusicSettings) -> dict[str, np.ndarray]:
    """Stage "dynamics": the dense _s curves, then pooling per beat and the beat curves.

    The smoothed dense curves are discarded here but timed, because _analyse computes them
    for the npz. Returns the beat curves of beat_dynamics, each of shape [B]."""
    smooth_dense(
        dense=dense,
        beat_period=float(np.median(np.diff(grid.times))),
        window_beats=settings.dense_sg_window_beats,
        polyorder=settings.sg_polyorder,
    )
    return beat_dynamics(
        pooled=pool_to_beats(dense=dense, beat_times=grid.times),
        names=dense.names,
        window=settings.beat_sg_window,
        polyorder=settings.sg_polyorder,
    )


def _run_sections(
    timbre: DenseFeatures, grid: BeatGrid, rms_s: np.ndarray, rms_d1: np.ndarray, settings: MusicSettings
) -> np.ndarray:
    """Stage "sections": timbre pooled per bar, novelty, boundaries and one label per beat.

    The bar pooling is inside the stage, as in _analyse, so the stages add up to the whole."""
    bars = pool_to_beats(dense=timbre, beat_times=grid.times[grid.is_downbeat])
    novelty = section_novelty(bar_features=bars, kernel_bars=settings.section_kernel_bars)
    bounds = find_section_boundaries(
        novelty=novelty,
        min_section_bars=settings.min_section_bars,
        edge_bars=settings.edge_bars,
        prominence=settings.novelty_prominence,
    )
    return label_sections(
        grid=grid,
        rms_s=rms_s,
        rms_d1=rms_d1,
        boundaries=bounds,
        level_percentile=settings.level_percentile,
        drop_level=settings.drop_level,
        build_rise=settings.build_rise,
    )


def _run_analyse(
    y: np.ndarray, sample_rate: int, settings: MusicSettings, model: Audio2Beats
) -> SongAnalysis:
    """The whole analysis of one song, exactly what the pipeline runs (no file I/O)."""
    return _analyse(y=y, sample_rate=sample_rate, settings=settings, model=model)


def _check_stages_match(
    grid: BeatGrid, dyn: dict[str, np.ndarray], labels: np.ndarray, expected: SongAnalysis
) -> None:
    """Check that the stage outputs give the same beat table as _analyse.

    Raises:
        AssertionError: If they differ, i.e. the _run_* functions no longer wire the steps
            like _analyse does (e.g. a new setting was added there only)."""
    pd.testing.assert_frame_equal(build_beat_table(grid=grid, dynamics=dyn, sections=labels), expected.beats)


def _select_song(songs: pd.DataFrame, pattern: str) -> pd.Series:
    """Return the one song table row whose name contains pattern (case-insensitive, literal).

    Raises:
        ValueError: If no name or more than one name matches."""
    matches = songs[songs["name"].str.contains(pattern, case=False, regex=False)]
    if len(matches) == 0:
        raise ValueError(f"No song name contains {pattern!r} (the song table has {len(songs)} songs)")
    if len(matches) > 1:
        names = "\n".join(f"  - {name}" for name in matches["name"])
        raise ValueError(
            f"{len(matches)} song names contain {pattern!r}; use a longer --song to pick one:\n{names}"
        )
    return matches.iloc[0]


def _summarise(results: list[BenchResult], song_name: str) -> dict[str, Any]:
    """The run's conclusions, stored as the run file's "summary" and printed by _print_summary.

    Args:
        results (list[BenchResult]): One result per Bench member.
        song_name (str): The song's name from the song table.

    Returns:
        dict[str, Any]: song, song_seconds, torch_threads, torch_version, model_load_s,
            first_song_s (model load + cold analysis), analyse_s and analyse_x_realtime
            (warm median), projected_3min_s, target_x_realtime, passed, stage_sum_s,
            overhead_s (warm analysis minus the stage sum) and stage_share (per stage)."""
    by_name = {Bench(r.name): r for r in results}
    song_seconds = by_name[Bench.ANALYSE].media_seconds
    model_load_s = by_name[Bench.MODEL_LOAD].median
    analyse_s = by_name[Bench.ANALYSE].median
    analyse_x_realtime = by_name[Bench.ANALYSE].x_realtime
    stage_sum_s = sum(by_name[n].median for n in PROCESS_STAGES)
    return {
        "song": song_name,
        "song_seconds": song_seconds,
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "model_load_s": model_load_s,
        "first_song_s": model_load_s + by_name[Bench.ANALYSE_COLD].median,
        "analyse_s": analyse_s,
        "analyse_x_realtime": analyse_x_realtime,
        "projected_3min_s": analyse_s * TARGET_SONG_SECONDS / song_seconds,
        "target_x_realtime": TARGET_X_REALTIME,
        "passed": analyse_x_realtime >= TARGET_X_REALTIME,
        "stage_sum_s": stage_sum_s,
        "overhead_s": analyse_s - stage_sum_s,
        "stage_share": {n: by_name[n].median / stage_sum_s for n in PROCESS_STAGES},
    }


def _print_summary(summary: dict[str, Any]) -> None:
    """Print the conclusions of a run: target check, first-song cost and stage shares."""
    verdict = "pass" if summary["passed"] else "FAIL"
    shares = sorted(summary["stage_share"].items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nSong: {summary['song']} ({summary['song_seconds']:.1f} s)")
    print(f"torch: {summary['torch_version']}, {summary['torch_threads']} threads")
    print(
        f"Analysis: {summary['analyse_s']:.2f} s ({summary['analyse_x_realtime']:.0f}x real time; "
        f"{summary['projected_3min_s']:.1f} s for a 3-min song)  "
        f"target >= {summary['target_x_realtime']:.0f}x: {verdict}"
    )
    print(
        f"First song in a fresh process: {summary['model_load_s']:.2f} s model load + "
        f"{summary['first_song_s'] - summary['model_load_s']:.2f} s cold analysis "
        f"= {summary['first_song_s']:.2f} s"
    )
    stage_sum_ms = summary["stage_sum_s"] * 1000
    print(" | ".join(f"{name} {share * stage_sum_ms:.4g} ms ({share:.2%})" for name, share in shares))
    print(
        f"Stages add up to {summary['stage_sum_s']:.2f} s; "
        f"the warm analysis is {summary['overhead_s'] * 1000:+.0f} ms off that"
    )


def _plot(results: list[BenchResult], summary: dict[str, Any], out_path: Path) -> None:
    """Save the run's figure: the seconds per stage and cold/warm (left), x real time (right).

    The dashed line on the right is the target. Everything comes from results and summary,
    so a saved run can be re-plotted (--from-json)."""
    by_name = {Bench(r.name): r for r in results}
    throughputs = [
        Bench.ANALYSE_COLD,
        Bench.BEATS,
        Bench.DENSE,
        Bench.TIMBRE,
        Bench.DYNAMICS,
        Bench.SECTIONS,
        Bench.ANALYSE,
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    _plot_breakdown(ax1, by_name, summary)
    plot_throughput(ax2, results=[by_name[n] for n in throughputs], colors=COLORS)
    ax2.axvline(summary["target_x_realtime"], ls="--", color="gray")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_breakdown(ax: Axes, by_name: dict[Bench, BenchResult], summary: dict[str, Any]) -> None:
    """Stacked horizontal bars in seconds: the 5 stages, the warm analysis, and the first
    song (model load + cold analysis). Stage legend entries carry their share of the stage sum."""
    shares = summary["stage_share"]
    row = ["first song (cold)", "analysis (warm)", "stages"]
    width = {
        Bench.MODEL_LOAD: [by_name[Bench.MODEL_LOAD].median, 0, 0],
        Bench.ANALYSE_COLD: [by_name[Bench.ANALYSE_COLD].median, 0, 0],
        Bench.ANALYSE: [0, by_name[Bench.ANALYSE].median, 0],
        Bench.BEATS: [0, 0, by_name[Bench.BEATS].median],
        Bench.DENSE: [0, 0, by_name[Bench.DENSE].median],
        Bench.TIMBRE: [0, 0, by_name[Bench.TIMBRE].median],
        Bench.DYNAMICS: [0, 0, by_name[Bench.DYNAMICS].median],
        Bench.SECTIONS: [0, 0, by_name[Bench.SECTIONS].median],
    }

    running_sum = np.zeros(3)
    for label, weights in width.items():
        display_label = f"{label} ({shares[label]:.0%})" if label in shares else label
        ax.barh(row, weights, left=running_sum, color=COLORS[label], label=display_label)
        running_sum += weights

    ax.set_xlim(0, 1.05 * running_sum.max())
    ax.set_title("Stage breakdown")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=4, frameon=False)
    ax.set_xlabel("seconds")


@dataclass(frozen=True, slots=True)
class Args:
    """The parsed command line (see parse_args for the meaning of each option)."""

    song: str
    song_table: Path
    warmup: int
    repeat: int
    out: Path
    plot: bool
    from_json: Path | None
    threads: int | None
    ckpt: str


def parse_args(argv: list[str] | None = None) -> Args:
    """Parse and validate the command line.

    Args:
        argv (list[str] | None): The arguments without the program name; None reads sys.argv.

    Returns:
        Args: The typed settings. With --from-json, warmup, repeat and threads are not
            checked, since nothing is measured.

    Raises:
        SystemExit: On --help or invalid arguments (argparse prints the usage)."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.bench_02_music",
        description=(
            "Time the step 02 music analysis of one song: the beat model load, the cold and warm "
            "whole analysis, and each stage (beats, dense, timbre, dynamics, sections)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--song",
        type=str,
        default=ORIGINAL_SONG,
        help="part of the song's name in the song table, case-insensitive; must match exactly one song",
    )
    parser.add_argument(
        "--song-table",
        type=Path,
        default=PATHS.song_table_path,
        help="song table from scripts.build_song_table; gives the song's wav",
    )
    parser.add_argument(
        "--warmup", type=int, default=BENCH.warmup, help="unmeasured runs before each benchmark"
    )
    parser.add_argument("--repeat", type=int, default=BENCH.repeat, help="measured runs per benchmark")
    parser.add_argument("--out", type=Path, default=BENCH.out_dir, help="folder for the run JSON")
    parser.add_argument("--plot", action="store_true", help="also save a .png next to the JSON")
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        metavar="RUN_JSON",
        help="don't measure: re-plot a saved run, writing the .png next to it",
    )
    parser.add_argument(
        "--ckpt", type=str, default=MUSIC.beat_checkpoint, help="beat_this checkpoint name or path"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch threads (torch.set_num_threads), e.g. 2 to estimate a small CPU-only machine; "
        "None keeps torch's own choice. Affects the beats stage only",
    )
    args = Args(**vars(parser.parse_args(argv)))
    if args.from_json is not None:
        if not args.from_json.is_file():
            parser.error(f"The given run file doesn't exist: {args.from_json}")
        return args

    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be >= 1 and --warmup >= 0")

    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be >= 1")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark (or re-plot a saved run) as described in the module docstring.

    Args:
        argv (list[str] | None): Command-line arguments; None reads sys.argv."""
    configure_stdout()  # song names contain characters cp1252 can't print
    args = parse_args(argv)
    timestamp = datetime.now()
    json_path: Path = args.out / f"bench_02_music_{timestamp.strftime('%Y%m%d-%H%M%S')}.json"
    if args.from_json is not None:
        results, summary = load_run(args.from_json)
        _plot(results, summary, args.from_json.with_suffix(".png"))
        return

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    row = _select_song(pd.read_parquet(args.song_table), args.song)
    wav = Path(row["audio_proxy_path"])
    media_info = probe(wav, DECODING.probe_timeout)
    y, sample_rate = load_audio(wav, MUSIC.sample_rate)

    model_load = benchmark(partial(_load_model, args.ckpt), Bench.MODEL_LOAD, media_info, "audio", 0, 1)
    model = load_beat_model(args.ckpt)
    analyse_fn = partial(_run_analyse, y, sample_rate, MUSIC, model)
    analyse_cold = benchmark(lambda: len(analyse_fn().beats), Bench.ANALYSE_COLD, media_info, "audio", 0, 1)

    beats_fn = partial(_run_beats, y, sample_rate, model, MUSIC)
    grid = beats_fn()
    dense_fn = partial(_run_dense, y, sample_rate, MUSIC)
    dense = dense_fn()
    timbre_fn = partial(_run_timbre, y, sample_rate, MUSIC)
    timbre = timbre_fn()
    dyn_fn = partial(_run_dynamics, dense, grid, MUSIC)
    dyn = dyn_fn()
    sec_fn = partial(_run_sections, timbre, grid, dyn["rms_s"], dyn["rms_d1"], MUSIC)
    labels = sec_fn()
    _check_stages_match(grid, dyn, labels, analyse_fn())
    beats_r = benchmark(
        lambda: len(beats_fn().times), Bench.BEATS, media_info, "audio", args.warmup, args.repeat
    )
    dense_r = benchmark(
        lambda: dense_fn().values.shape[0], Bench.DENSE, media_info, "audio", args.warmup, args.repeat
    )
    timbre_r = benchmark(
        lambda: timbre_fn().values.shape[0], Bench.TIMBRE, media_info, "audio", args.warmup, args.repeat
    )
    dyn_r = benchmark(
        lambda: len(dyn_fn()["rms_s"]), Bench.DYNAMICS, media_info, "audio", args.warmup, args.repeat
    )
    sec_r = benchmark(lambda: len(sec_fn()), Bench.SECTIONS, media_info, "audio", args.warmup, args.repeat)

    analyse_r = benchmark(
        lambda: len(analyse_fn().beats), Bench.ANALYSE, media_info, "audio", args.warmup, args.repeat
    )

    results = [model_load, analyse_cold, beats_r, dense_r, timbre_r, dyn_r, sec_r, analyse_r]
    print_table(results)
    summary = _summarise(results, row["name"])
    _print_summary(summary)
    save_run(json_path, timestamp, dataclasses.asdict(args), {"song": media_info}, results, summary)
    if args.plot:
        plot_path = json_path.with_suffix(".png")
        _plot(results, summary, plot_path)


if __name__ == "__main__":
    main()
