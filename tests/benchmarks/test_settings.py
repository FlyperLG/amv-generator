"""Tests for benchmarks.settings: where the benchmarks' own settings live.

SPECIFICATION (decisions these tests encode)
--------------------------------------------
  - amv/config.py is the config of the pipeline and the model only. Side tools (benchmarks,
    manual checks) keep their settings in their own folder, with the same pattern: a frozen
    dataclass whose defaults are the configured values, plus one instance.
  - benchmarks/settings.py holds BenchSettings and BENCH, shared by every bench_* script.
  - BENCH.out_dir is data/out/bench/, built from the anchor amv.config.DATA_DIR, so it
    moves along with the data folder.
  - BENCH also holds the defaults every benchmark shares: warmup = 1 unmeasured run and
    repeat = 5 measured runs. Settings of one benchmark only (bench_01's --fps, --width,
    --height) stay argparse defaults in that script: the command line overrides them
    anyway, so a dataclass per benchmark would add nothing.
  - A bench_* script's --out, --warmup and --repeat default to the BENCH values.
"""

import dataclasses
import importlib

import pytest

from amv.config import DATA_DIR
from benchmarks.settings import BENCH, BenchSettings


def test_bench_is_the_default_instance():
    assert type(BENCH) is BenchSettings
    assert BENCH == BenchSettings()


def test_bench_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        BENCH.out_dir = BENCH.out_dir  # type: ignore[misc]


def test_out_dir_is_below_the_data_dir():
    assert BENCH.out_dir == DATA_DIR / "out" / "bench"


def test_shared_run_counts_have_the_planned_values():
    assert BENCH.warmup == 1
    assert BENCH.repeat == 5


@pytest.mark.parametrize("module_name", ["bench_01_decoding", "bench_02_music"])
def test_bench_defaults_come_from_bench(tmp_path, module_name):
    module = importlib.import_module(f"benchmarks.{module_name}")
    run_json = tmp_path / "run.json"
    run_json.write_text("{}", encoding="utf-8")
    args = module.parse_args(["--from-json", str(run_json)])  # no media needed
    assert args.out == BENCH.out_dir
    assert args.warmup == BENCH.warmup
    assert args.repeat == BENCH.repeat
