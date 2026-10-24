"""Plot panels that work for any benchmark run. Each draws into an Axes the caller provides."""

from collections.abc import Mapping

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

from benchmarks.results import BenchResult


def plot_throughput(ax: Axes, results: list[BenchResult], colors: Mapping[str, str]) -> None:
    """Horizontal bars of the median x real time, on a log axis starting at 1x (real time).

    Whiskers span the slowest to the fastest repeat; each bar is labelled with its value.

    Args:
        ax (Axes): The panel to draw into.
        results (list[BenchResult]): One bar per result, top to bottom in list order.
        colors (Mapping[str, str]): Matplotlib colour per result name; must contain every name."""
    medians = [r.x_realtime for r in results]
    labels = [r.name for r in results]
    plot_colors = [colors[r.name] for r in results]
    whiskers = [
        [r.x_realtime - r.x_realtime_range[0] for r in results],
        [r.x_realtime_range[1] - r.x_realtime for r in results],
    ]

    throughput_bars = ax.barh(
        labels,
        medians,
        color=plot_colors,
        xerr=whiskers,
    )
    ax.set_xscale("log")
    ax.bar_label(throughput_bars, fmt="%.2fx", padding=3)
    right = max(r.x_realtime_range[1] for r in results)  # end of the longest whisker
    ax.set_xlim(left=1, right=right * 3)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}x"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_inverted(True)
    ax.set_title("Throughput (x real time)")
    ax.set_xlabel("x real time")


def plot_repeats(ax: Axes, results: list[BenchResult], colors: Mapping[str, str]) -> None:
    """One line per result: each repeat's time divided by that result's median.

    Normalising puts fast and slow benchmarks on one scale, so drift (e.g. thermal
    throttling) shows as a percentage. The legend gives each result's spread,
    (max - min) / median; above ~10 % the run is noisy.

    Args:
        ax (Axes): The panel to draw into.
        results (list[BenchResult]): The results to draw.
        colors (Mapping[str, str]): Matplotlib colour per result name; must contain every name."""
    for result in results:
        timings = result.seconds
        median = result.median
        ax.plot(
            range(1, len(timings) + 1),
            [x / median for x in timings],
            marker="o",
            label=f"{result.name} (spread {(result.max - result.min) / median:.0%})",
            color=colors[result.name],
        )
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.legend()
    ax.set_title("Timing per repeat")
    ax.set_xlabel("repeat")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("time / median")
