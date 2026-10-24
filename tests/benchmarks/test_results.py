"""Unit tests for benchmarks.results: BenchResult, save_run, load_run.

No ffmpeg, no media files. get_machine_information is replaced by a stub, because it
starts ffmpeg and its output depends on the machine.

SPECIFICATION (decisions these tests encode)
--------------------------------------------
BenchResult (frozen dataclass: name, input, seconds, n_frames, media_seconds)
  - median / min / max are over the measured seconds.
  - x_realtime = media_seconds / median;  fps = n_frames / median.
  - x_realtime_range = (slowest, fastest) = (media_seconds / max, media_seconds / min).

to_dict(sources) / from_dict(data, sources)
  - The JSON stores each MediaInfo once under "sources"; a result only stores its key
    ("source"). to_dict finds that key itself, by IDENTITY (is), not by equality: two
    sources with equal metadata stay distinguishable.
  - A result whose MediaInfo is not in sources -> ValueError (never write a broken reference).
  - to_dict also writes the derived values (median_s, min_s, max_s, fps, x_realtime, ...)
    for humans; from_dict ignores them and recomputes from "seconds".

save_run(path, created, args, sources, results, summary) / load_run(path)
  - save_run creates missing parent directories of path itself: whoever writes a file
    owns that precondition, so callers never have to mkdir first.
  - JSON top level: created (ISO 8601, seconds), machine, args, sources, results, summary.
  - load_run(path) -> (results, summary). The summary is returned as a whole dict: which
    keys it holds (e.g. "break_even_reads") is the business of each bench_* script, not
    of this shared module.
  - Round trip: load_run(save_run(...)) gives results equal to the saved ones, and results
    that shared one MediaInfo before share one object again (so a reloaded run can be
    saved again, since to_dict relies on identity).
"""

import json
from datetime import datetime

import pytest

import benchmarks.results
from benchmarks.results import BenchResult, load_run, save_run
from tests.benchmarks.helpers import make_media_info

FAKE_MACHINE = {"platform": "test", "processor": "test", "cpu_count": 1, "python": "3", "ffmpeg": "x"}
CREATED = datetime(2026, 10, 24, 14, 2, 11, 123456)


@pytest.fixture(autouse=True)
def fake_machine(monkeypatch):
    monkeypatch.setattr(benchmarks.results, "get_machine_information", lambda: dict(FAKE_MACHINE))


@pytest.fixture
def original():
    return make_media_info("data/amvs/clip.mp4", duration=30.0, width=1920)


@pytest.fixture
def proxy():
    return make_media_info("data/proxies/clip.mp4", duration=30.0, width=640)


@pytest.fixture
def run(original, proxy):
    """Two results on the original (sharing one MediaInfo) and one on the proxy."""
    sources = {"original": original, "proxy": proxy}
    results = [
        BenchResult("transcode", original, seconds=[1.5, 1.0, 2.0], n_frames=900, media_seconds=30.0),
        BenchResult("read_original", original, seconds=[3.0, 2.0, 4.0], n_frames=240, media_seconds=30.0),
        BenchResult("read_proxy", proxy, seconds=[0.5, 0.4, 0.6], n_frames=240, media_seconds=30.0),
    ]
    return sources, results


# =============================================================================
# BenchResult: derived values
# =============================================================================


class TestBenchResult:
    @pytest.fixture
    def result(self, original):
        return BenchResult("read", original, seconds=[3.0, 2.0, 4.0], n_frames=300, media_seconds=30.0)

    def test_median_min_max(self, result):
        assert result.median == 3.0
        assert result.min == 2.0
        assert result.max == 4.0

    def test_x_realtime(self, result):
        assert result.x_realtime == pytest.approx(10.0)

    def test_fps(self, result):
        assert result.fps == pytest.approx(100.0)

    def test_x_realtime_range_is_slowest_then_fastest(self, result):
        assert result.x_realtime_range == pytest.approx((7.5, 15.0))


# =============================================================================
# to_dict / from_dict
# =============================================================================


class TestDictConversion:
    def test_to_dict_writes_source_key_instead_of_media_info(self, run):
        sources, results = run

        d = results[2].to_dict(sources)

        assert d["source"] == "proxy"
        assert "input" not in d

    def test_to_dict_writes_derived_values(self, run):
        sources, results = run

        d = results[1].to_dict(sources)

        assert d["median_s"] == 3.0
        assert d["min_s"] == 2.0
        assert d["max_s"] == 4.0
        assert d["x_realtime"] == pytest.approx(10.0)
        assert d["fps"] == pytest.approx(80.0)

    def test_to_dict_matches_by_identity_not_equality(self, original):
        # Two sources with identical metadata: the key must be the one of THIS object.
        twin = make_media_info("data/amvs/clip.mp4", duration=30.0, width=1920)
        assert twin == original and twin is not original
        sources = {"first": original, "second": twin}
        result = BenchResult("read", twin, seconds=[1.0], n_frames=1, media_seconds=30.0)

        assert result.to_dict(sources)["source"] == "second"

    def test_to_dict_unknown_media_info_raises(self, original, proxy):
        result = BenchResult("read", proxy, seconds=[1.0], n_frames=1, media_seconds=30.0)

        with pytest.raises(ValueError, match="read"):
            result.to_dict({"original": original})

    def test_from_dict_round_trip(self, run):
        sources, results = run

        restored = [BenchResult.from_dict(r.to_dict(sources), sources) for r in results]

        assert restored == results

    def test_from_dict_ignores_stale_derived_values(self, run):
        sources, results = run
        d = results[1].to_dict(sources)
        d["median_s"] = 999.0  # edited by hand / out of date

        assert BenchResult.from_dict(d, sources).median == 3.0


# =============================================================================
# save_run / load_run
# =============================================================================


def save(path, run, summary=None):
    sources, results = run
    save_run(
        path=path,
        created=CREATED,
        args={"repeat": 3, "out": path.parent},
        sources=sources,
        results=results,
        summary=summary if summary is not None else {"speedup": 7.5, "break_even_reads": 1.6},
    )


class TestSaveRun:
    def test_writes_top_level_keys(self, tmp_path, run):
        path = tmp_path / "run.json"

        save(path, run)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"created", "machine", "args", "sources", "results", "summary"}

    def test_created_is_iso_without_microseconds(self, tmp_path, run):
        path = tmp_path / "run.json"

        save(path, run)

        assert json.loads(path.read_text(encoding="utf-8"))["created"] == "2026-10-24T14:02:11"

    def test_each_source_is_stored_once(self, tmp_path, run):
        path = tmp_path / "run.json"

        save(path, run)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data["sources"]) == {"original", "proxy"}
        assert [r["source"] for r in data["results"]] == ["original", "original", "proxy"]

    def test_sources_are_structured_not_repr_strings(self, tmp_path, run):
        # Regression: json.dump(default=str) on a MediaInfo writes "MediaInfo(path=...)".
        path = tmp_path / "run.json"

        save(path, run)

        original = json.loads(path.read_text(encoding="utf-8"))["sources"]["original"]
        assert isinstance(original, dict)
        assert original["video"]["fps"] == "30000/1001"

    def test_creates_missing_parent_directories(self, tmp_path, run):
        path = tmp_path / "out" / "bench" / "run.json"

        save(path, run)

        assert path.is_file()


class TestLoadRun:
    def test_round_trip_results(self, tmp_path, run):
        path = tmp_path / "run.json"
        save(path, run)

        results, _ = load_run(path)

        assert results == run[1]

    def test_returns_whole_summary(self, tmp_path, run):
        path = tmp_path / "run.json"
        summary = {"speedup": 7.5, "break_even_reads": None, "anything": "else"}
        save(path, run, summary=summary)

        _, loaded_summary = load_run(path)

        assert loaded_summary == summary

    def test_results_of_one_source_share_one_media_info(self, tmp_path, run):
        path = tmp_path / "run.json"
        save(path, run)

        results, _ = load_run(path)

        assert results[0].input is results[1].input
        assert results[0].input is not results[2].input

    def test_loaded_run_can_be_saved_again(self, tmp_path, run):
        # --from-json re-plots; a later tool may also re-save. to_dict relies on identity,
        # so the reloaded sources must be the very objects the results point to.
        first = tmp_path / "first.json"
        save(first, run)
        results, summary = load_run(first)
        sources = {"original": results[0].input, "proxy": results[2].input}

        second = tmp_path / "second.json"
        save(second, (sources, results), summary=summary)

        assert load_run(second) == load_run(first)
