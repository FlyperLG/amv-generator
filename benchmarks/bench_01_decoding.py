import argparse
import dataclasses
import functools
import json
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
from tqdm import tqdm

from amv.io.ffmpeg import iter_frames, make_video_proxy, probe
from amv.io.media import MediaInfo
from benchmarks.utils import get_machine_information

ORIGINAL_FILE: Path = Path(
    "data/amvs/Agony Hardstyle (Blue Lock - Turn it into Light! ft. Yoichi Isagi Remix_AMV).mp4"
)


@dataclass(frozen=True, slots=True)
class BenchResult:
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
        return self.media_seconds / self.median

    @property
    def fps(self) -> float:
        return self.n_frames / self.median

    def summarize(self) -> tuple[str, float, float, int, float, float]:
        return (self.name, self.median, self.min, self.n_frames, self.fps, self.x_realtime)

    def to_dict(self, source_name: str) -> dict[str, Any]:
        result = {
            **dataclasses.asdict(self),
            "median_s": self.median,
            "min_s": self.min,
            "max_s": self.max,
            "fps": self.fps,
            "x_realtime": self.x_realtime,
        }
        result.pop("input")
        result["source"] = source_name
        return result


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


def _time_it(
    fn: Callable[[], int],
    warmup: int = 1,
    repeat: int = 5,
    name: str | None = None,
) -> tuple[list[float], int]:
    timings: list[float] = []
    processed_frames: list[int] = []
    for _ in tqdm(range(warmup + repeat), desc=name, dynamic_ncols=True):
        start = time.perf_counter()
        frame_counter = fn()
        end = time.perf_counter()
        processed_frames.append(frame_counter)
        timings.append(end - start)

    if len(set(processed_frames[warmup:])) != 1:
        raise RuntimeError(f"frame count differs between repeats: {processed_frames}")
    return (timings[warmup:], processed_frames[-1])


def _plot(data: dict, out_path: Path) -> None:
    ORIGINAL_COLOR: str = "tab:blue"
    PROXY_COLOR: str = "tab:red"
    TRANSCODE_COLOR: str = "tab:orange"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

    medians = []
    labels = []
    colors = []
    whiskers = [[], []]

    for result in data["results"]:
        medians.append(result["x_realtime"])
        labels.append(result["name"])
        colors.append(ORIGINAL_COLOR if result["source"] == "original" else PROXY_COLOR)
        whiskers[0].append(result["x_realtime"] - (result["media_seconds"] / result["max_s"]))
        whiskers[1].append((result["media_seconds"] / result["min_s"]) - result["x_realtime"])

    throughput_bars = ax1.barh(
        labels,
        medians,
        color=colors,
        xerr=whiskers,
    )
    ax1.set_xscale("log")
    ax1.bar_label(throughput_bars, fmt="%.2fx", padding=3)
    right = max(r["media_seconds"] / r["min_s"] for r in data["results"])  # end of the longest whisker
    ax1.set_xlim(right=right * 2)
    ax1.yaxis.set_inverted(True)
    ax1.set_title("Throughput (x real time)")
    ax1.set_xlabel("x real time")

    results_by_name = {r["name"]: r for r in data["results"]}
    n = np.arange(0, 11)
    step = results_by_name["iter_frames_original"]["median_s"]
    values = np.arange(0, step * 11, step)
    ax2.plot(n, values, color=ORIGINAL_COLOR, label="original")

    step = results_by_name["iter_frames_proxy"]["median_s"]
    values2 = np.arange(
        results_by_name["proxy_transcode"]["median_s"],
        results_by_name["proxy_transcode"]["median_s"] + step * 11,
        step,
    )
    ax2.plot(n, values2, color=PROXY_COLOR, label="proxy")

    break_even = data["summary"]["break_even_reads"]
    if break_even is not None:
        y = break_even * results_by_name["iter_frames_original"]["median_s"]
        ax2.plot(break_even, y, "o", color="black")
        ax2.annotate(
            f"break-even ≈ {break_even:.1f} reads",
            xy=(break_even, y),
            xytext=(10, -20),
            textcoords="offset points",
        )

    ax2.fill_between(n, values2, values, where=values > values2, alpha=0.15)
    ax2.set_xlabel("full reads (8 fps, 320x180)")
    ax2.set_ylabel("cumulative seconds")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title("Break-even")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(bottom=0)
    ax2.legend()

    for result in data["results"]:
        timings = result["seconds"]
        median = result["median_s"]
        color = (
            ORIGINAL_COLOR
            if result["name"] == "iter_frames_original"
            else PROXY_COLOR
            if result["name"] == "iter_frames_proxy"
            else TRANSCODE_COLOR
        )
        ax3.plot(
            range(1, len(timings) + 1),
            [x / median for x in timings],
            marker="o",
            label=f"{result['name']} (spread {(result['max_s'] - result['min_s']) / median:.0%})",
            color=color,
        )
    ax3.axhline(1.0, ls="--", color="gray", lw=1)
    ax3.legend()
    ax3.set_title("Timing per repeat")
    ax3.set_xlabel("repeat")
    ax3.set_ylabel("time / median")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def benchmark(
    fn: Callable[[], int],
    name: str,
    media_info: MediaInfo,
    warmup: int = 1,
    repeat: int = 5,
) -> BenchResult:
    if media_info.video is None:
        raise ValueError(f"Provided media was not a valid video: {media_info.video}")

    seconds, n_frames = _time_it(fn, warmup, repeat, name=name)
    return BenchResult(
        name=name,
        input=media_info,
        seconds=seconds,
        n_frames=n_frames,
        media_seconds=media_info.video.duration,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=Path, default=ORIGINAL_FILE, nargs="?")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--proxy-dir", type=Path, default=Path("data/proxies/"))
    parser.add_argument("--out", type=Path, default=Path("data/out/bench/"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--from-json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.from_json is not None:
        return args

    if not args.video.exists():
        parser.error(f"The given video file doesn't exist: {args.video}")
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be >= 1 and --warmup >= 0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    timestamp = datetime.now()
    json_path: Path = args.out / f"bench_01_decoding_{timestamp.strftime('%Y%m%d-%H%M%S')}.json"
    if args.from_json is not None:
        with open(args.from_json, encoding="utf-8") as f:
            data = json.load(f)
        _plot(data, args.from_json.with_suffix(".png"))
        return

    media_info = probe(args.video)
    if media_info.video is None or media_info.video.n_frames is None:
        raise ValueError("The given file is not of type video or has no n_frames.")

    proxy_file_path = make_video_proxy(
        args.video,
        args.proxy_dir,
    )
    proxy_media_info = probe(proxy_file_path)

    transcode_result = benchmark(
        fn=functools.partial(
            _measure_proxy_transcode,
            args.video,
            media_info.video.n_frames,
        ),
        name="proxy_transcode",
        media_info=media_info,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    iter_result = benchmark(
        fn=functools.partial(_measure_iter_frames, args.video, args.fps, args.width, args.height),
        name="iter_frames_original",
        media_info=media_info,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    iter_proxy_result = benchmark(
        fn=functools.partial(_measure_iter_frames, proxy_file_path, args.fps, args.width, args.height),
        name="iter_frames_proxy",
        media_info=proxy_media_info,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    results_summaries = [
        transcode_result.summarize(),
        iter_result.summarize(),
        iter_proxy_result.summarize(),
    ]

    print(
        tabulate(
            results_summaries,
            headers=["name", "median s", "min s", "frames", "fps", "x real time"],
            tablefmt="orgtbl",
            floatfmt=".2f",
        )
    )
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

    args.out.mkdir(parents=True, exist_ok=True)

    json_result = {
        "created": timestamp.isoformat(timespec="seconds"),
        "machine": get_machine_information(),
        "args": vars(args),
        "sources": {
            "original": dataclasses.asdict(media_info),
            "proxy": dataclasses.asdict(proxy_media_info),
        },
        "results": [
            transcode_result.to_dict("original"),
            iter_result.to_dict("original"),
            iter_proxy_result.to_dict("proxy"),
        ],
        "summary": {
            "proxy_read_speedup": speedup,
            "break_even_reads": break_even,
        },
    }

    with open(json_path, "w", encoding="utf-8") as out_json:
        json.dump(json_result, out_json, default=str, indent=4)

    if args.plot:
        plot_path = json_path.with_suffix(".png")
        _plot(json_result, plot_path)


if __name__ == "__main__":
    main()
