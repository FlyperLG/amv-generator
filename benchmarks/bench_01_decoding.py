"""Benchmark for step 01 (docs/01-decoding-and-proxies, task 3.5): is the proxy worth it?

Measures, on one video:
  - proxy_transcode:      make_video_proxy, original -> 640 px proxy
  - iter_frames_original: a full iter_frames read of the original
  - iter_frames_proxy:    the same read of the proxy
and prints the proxy read speedup (target >= 5x) and after how many full reads the
transcode has paid for itself.

Run from the repository root:
    python -m benchmarks.bench_01_decoding --plot
    python -m benchmarks.bench_01_decoding --from-json data/out/bench/<run>.json

Output: data/out/bench/bench_01_decoding_<timestamp>.json (+ .png with --plot).
"""

import argparse
import dataclasses
import functools
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from amv.io.ffmpeg import iter_frames, make_video_proxy, probe
from benchmarks.plots import plot_repeats, plot_throughput
from benchmarks.results import BenchResult, load_run, print_table, save_run
from benchmarks.timing import benchmark

ORIGINAL_FILE: Path = Path(
    "data/amvs/Agony Hardstyle (Blue Lock - Turn it into Light! ft. Yoichi Isagi Remix_AMV).mp4"
)


class Bench(StrEnum):
    TRANSCODE = "proxy_transcode"
    READ_ORIGINAL = "iter_frames_original"
    READ_PROXY = "iter_frames_proxy"


COLORS: dict[str, str] = {
    Bench.TRANSCODE: "tab:orange",
    Bench.READ_ORIGINAL: "tab:blue",
    Bench.READ_PROXY: "tab:red",
}


def _measure_proxy_transcode(in_path: Path, n_frames: int) -> int:
    with tempfile.TemporaryDirectory() as out_dir:
        make_video_proxy(
            in_path,
            out_dir,
        )
    return n_frames


def _measure_iter_frames(in_path: Path, fps: float, width: int, height: int) -> int:
    generator = iter_frames(
        path=in_path,
        fps=fps,
        width=width,
        height=height,
    )

    return sum(1 for _ in generator)


def _plot(results: list[BenchResult], break_even: float | None, out_path: Path) -> None:
    results_by_name = {Bench(r.name): r for r in results}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    plot_throughput(ax1, results, COLORS)
    _plot_break_even(ax2, results_by_name, break_even)
    plot_repeats(ax3, results, COLORS)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_break_even(ax: Axes, by_name: dict[Bench, BenchResult], break_even: float | None) -> None:
    n = np.arange(0, 11)
    original_values = n * by_name[Bench.READ_ORIGINAL].median
    ax.plot(n, original_values, color=COLORS[Bench.READ_ORIGINAL], label="original")

    proxy_values = by_name[Bench.TRANSCODE].median + n * by_name[Bench.READ_PROXY].median
    ax.plot(n, proxy_values, color=COLORS[Bench.READ_PROXY], label="proxy")

    if break_even is not None:
        y = break_even * by_name[Bench.READ_ORIGINAL].median
        ax.plot(break_even, y, "o", color="black")
        ax.annotate(
            f"break-even ≈ {break_even:.1f} reads",
            xy=(break_even, y),
            xytext=(10, -20),
            textcoords="offset points",
        )

    ax.fill_between(
        n, proxy_values, original_values, where=original_values > proxy_values, alpha=0.15, interpolate=True
    )
    ax.set_xlabel("full reads")
    ax.set_ylabel("cumulative seconds")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Break-even")
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom=0)
    ax.legend()


@dataclass(frozen=True, slots=True)
class Args:
    video: Path
    warmup: int
    repeat: int
    fps: float
    width: int
    height: int
    proxy_dir: Path
    out: Path
    plot: bool
    from_json: Path | None


def parse_args(argv: list[str] | None = None) -> Args:
    """Parse and validate the command line.

    Args:
        argv (list[str] | None): The arguments without the program name; None reads sys.argv.

    Returns:
        Args: The typed settings. With --from-json, video, warmup and repeat are not checked,
            since nothing is measured.

    Raises:
        SystemExit: On --help or invalid arguments (argparse prints the usage)."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.bench_01_decoding",
        description="Time the proxy transcode and full frame reads of the original vs the proxy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, default=ORIGINAL_FILE, nargs="?", help="video to benchmark")
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured runs before each benchmark")
    parser.add_argument("--repeat", type=int, default=5, help="measured runs per benchmark")
    parser.add_argument("--fps", type=float, default=8, help="frame rate of the frame reads")
    parser.add_argument("--width", type=int, default=320, help="frame width of the frame reads, in px")
    parser.add_argument("--height", type=int, default=180, help="frame height of the frame reads, in px")
    parser.add_argument(
        "--proxy-dir", type=Path, default=Path("data/proxies/"), help="proxy cache used for the proxy read"
    )
    parser.add_argument("--out", type=Path, default=Path("data/out/bench/"), help="folder for the run JSON")
    parser.add_argument("--plot", action="store_true", help="also save a .png next to the JSON")
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        metavar="RUN_JSON",
        help="don't measure: re-plot a saved run, writing the .png next to it",
    )
    args = Args(**vars(parser.parse_args(argv)))
    if args.from_json is not None:
        if not args.from_json.is_file():
            parser.error(f"The given run file doesn't exist: {args.from_json}")
        return args

    if not args.video.exists():
        parser.error(f"The given video file doesn't exist: {args.video}")
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be >= 1 and --warmup >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark (or re-plot a saved run) as described in the module docstring.

    Args:
        argv (list[str] | None): Command-line arguments; None reads sys.argv."""
    args = parse_args(argv)
    timestamp = datetime.now()
    json_path: Path = args.out / f"bench_01_decoding_{timestamp.strftime('%Y%m%d-%H%M%S')}.json"
    if args.from_json is not None:
        results, summary = load_run(args.from_json)
        _plot(results, summary["break_even_reads"], args.from_json.with_suffix(".png"))
        return

    media_info = probe(args.video)
    if media_info.video is None:
        raise ValueError("The given file is not of type video.")

    n_frames = media_info.video.n_frames
    if n_frames is None:
        n_frames = round(media_info.video.duration * media_info.video.fps)
        print("VideoInfo doesn't contain n_frames, inferred from duration and fps instead.")

    proxy_file_path = make_video_proxy(
        args.video,
        args.proxy_dir,
    )
    proxy_media_info = probe(proxy_file_path)

    transcode_result = benchmark(
        fn=functools.partial(_measure_proxy_transcode, args.video, n_frames),
        name=Bench.TRANSCODE,
        media_info=media_info,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    iter_result = benchmark(
        fn=functools.partial(_measure_iter_frames, args.video, args.fps, args.width, args.height),
        name=Bench.READ_ORIGINAL,
        media_info=media_info,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    iter_proxy_result = benchmark(
        fn=functools.partial(_measure_iter_frames, proxy_file_path, args.fps, args.width, args.height),
        name=Bench.READ_PROXY,
        media_info=proxy_media_info,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results = [transcode_result, iter_result, iter_proxy_result]

    print_table(results)
    speedup: float = iter_proxy_result.fps / iter_result.fps
    print(f"\nProxy read speedup: {speedup:.1f}x  (target >= 5x: {'pass' if speedup >= 5 else 'FAIL'})")
    break_even: float | None = (
        transcode_result.median / (iter_result.median - iter_proxy_result.median)
        if (iter_result.median - iter_proxy_result.median) > 0
        else None
    )
    if break_even is not None:
        print(f"Proxy pays for itself after {break_even:.1f} full reads")
    else:
        print("Proxy will never pay for itself")

    save_run(
        path=json_path,
        created=timestamp,
        args=dataclasses.asdict(args),
        sources={"original": media_info, "proxy": proxy_media_info},
        results=results,
        summary={
            "proxy_read_speedup": speedup,
            "break_even_reads": break_even,
        },
    )

    if args.plot:
        plot_path = json_path.with_suffix(".png")
        _plot(results, break_even, plot_path)


if __name__ == "__main__":
    main()
