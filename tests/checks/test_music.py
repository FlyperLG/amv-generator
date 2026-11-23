"""Tests for checks.music: the computations and panels behind the step 02 check (task 3.7).

No beat model and no real media, except the one integration test at the end. The plots are
checked by what they draw (patches, lines, limits), not by comparing images.

SPECIFICATION (decisions these tests encode)
--------------------------------------------
waveform_envelope(y, sample_rate, hop) -> (times, lo, hi)
  - The min and max of each whole hop of samples; the leftover < hop samples at the end are
    dropped. times is the centre of each bucket: (k + 0.5) * hop / sample_rate.

stale_boundaries(beats, boundaries)
  - The bar indices where the cached section label changes but no recomputed boundary is.
  - Only that direction: a boundary without a label change is normal (drop -> drop).
  - The first row never counts as a change, so a pickup (bar_idx -1) is never reported.

recompute_sections(y, sample_rate, beats, settings)
  - Pools the timbre on the beat table's downbeats, as _analyse does, so its boundaries are
    exactly find_section_boundaries of its own novelty.
  - peaks are every local maximum of the novelty (no filter), prominences one per peak, so
    every boundary except bar 0 is among the peaks.

make_click_track(...) / stereo_click_mix(y, clicks)
  - The click track is exactly n_samples long. Each click starts at its beat's sample.
  - Downbeats click only at downbeat pitch and gain (no second click on top), so the
    loudest sample is at most the larger gain: nothing clips when written to 16-bit.
  - No beats gives silence.
  - The mix is [n, 2] float32: music left, clicks right. Different lengths -> ValueError.

beat_offsets(...)
  - Per beat, the nearest onset minus the beat time, in seconds; NaN when none is within
    max_offset. Calibrated: exact beats on the kick loop read +4 to +5 ms (the onset strength
    peaks just after the attack), kicks 30 ms late read 30 ms more.
  - No beats -> empty; silence -> all NaN.

kick_by_position(...)
  - The mean low-band kick per bar position, divided by the largest: the louder kick on "1"
    gives position 0 = 1.0. Numbering the bar from the third beat moves the peak to
    position 2 (the It Has Begun pattern). A position with no beats scores 0, silence all 0.

window_mask / _beat_mask / _beat_steps
  - window_mask: None keeps everything; otherwise start <= t <= end, edges included.
  - _beat_mask keeps every beat whose interval [t, t + ibi) overlaps the window, so a step
    plot reaches the left edge of the window.
  - _beat_steps adds one point at the end of the last beat (t + ibi) with the last value,
    so ax.step(where="post") draws the last beat too. No beats -> two empty arrays.

Panels and plot_song
  - SECTION_COLORS has exactly the SECTION_LABELS.
  - draw_sections: one band per run of equal labels, from the run's first beat to the next
    run's start; the last ends at the last t + ibi.
  - draw_beat_ticks: two line collections, beats and downbeats, spanning the panel height.
  - plot_novelty labels every candidate peak in the window with its prominence, and hatches
    the first and last edge_bars bars (MUSIC.edge_bars, task 3.9 item 2: the zones where
    find_section_boundaries drops every peak; min_section_bars only spaces the peaks).
  - plot_song: a Figure (not pyplot) with 5 panels sharing x; x from 0 to the song end, or
    the window, with an open end clipped to the song end; the title; a section legend with
    only the labels the song has. It renders without error.
"""

import io

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from amv.config import MUSIC, PATHS
from amv.music.sections import SECTION_LABELS, find_section_boundaries
from checks.music import (
    SECTION_COLORS,
    SectionCheck,
    _beat_mask,
    _beat_steps,
    beat_offsets,
    draw_beat_ticks,
    draw_sections,
    kick_by_position,
    make_click_track,
    plot_novelty,
    plot_song,
    recompute_sections,
    stale_boundaries,
    stereo_click_mix,
    waveform_envelope,
    window_mask,
)
from checks.settings import PLOT_MUSIC
from tests.checks.helpers import SR, make_analysis, make_beats

# --- waveform_envelope ----------------------------------------------------------


def test_envelope_drops_the_leftover_samples_and_centres_the_times():
    y = np.zeros(1000, np.float32)
    times, lo, hi = waveform_envelope(y, sample_rate=100, hop=300)
    assert len(times) == len(lo) == len(hi) == 3
    np.testing.assert_allclose(times, [1.5, 4.5, 7.5])


def test_envelope_brackets_a_sine():
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 100 * t).astype(np.float32)  # 100 Hz: one period per 220.5 samples
    _, lo, hi = waveform_envelope(y, SR, hop=512)
    assert np.all(lo <= hi)
    np.testing.assert_allclose(lo, -1, atol=1e-3)
    np.testing.assert_allclose(hi, 1, atol=1e-3)


# --- stale_boundaries -----------------------------------------------------------


def _beats_with_labels(labels_per_bar: list[str], pickup: bool = False) -> pd.DataFrame:
    beats = make_beats(0.5 + np.arange(4 * len(labels_per_bar)) * 0.5, labels_per_bar)
    if pickup:
        beats.loc[0, "bar_idx"] = -1
    return beats


def test_labels_changing_on_boundaries_are_not_stale():
    beats = _beats_with_labels(["intro", "intro", "drop", "drop", "break"])
    assert stale_boundaries(beats, np.array([0, 2, 4])).size == 0


def test_a_boundary_without_a_label_change_is_not_stale():
    beats = _beats_with_labels(["drop", "drop", "drop", "drop"])
    assert stale_boundaries(beats, np.array([0, 2])).size == 0


def test_a_label_change_off_the_boundaries_is_stale():
    beats = _beats_with_labels(["intro", "intro", "drop", "drop", "break"])
    np.testing.assert_array_equal(stale_boundaries(beats, np.array([0, 2])), [4])


def test_a_pickup_first_row_is_never_reported():
    beats = _beats_with_labels(["drop", "drop"], pickup=True)
    assert stale_boundaries(beats, np.array([0])).size == 0


# --- recompute_sections -----------------------------------------------------------


def test_recompute_sections_matches_find_section_boundaries(kick_loop):
    y, beat_times = kick_loop(bpm=120, bars=16)
    check = recompute_sections(y, SR, make_beats(beat_times), MUSIC)

    assert isinstance(check, SectionCheck)
    np.testing.assert_array_equal(check.bar_times, beat_times[::4])
    assert check.novelty.shape == (16,)
    expected = find_section_boundaries(
        check.novelty, MUSIC.min_section_bars, MUSIC.edge_bars, MUSIC.novelty_prominence
    )
    np.testing.assert_array_equal(check.boundaries, expected)
    assert check.prominences.shape == check.peaks.shape
    assert set(check.boundaries[1:]) <= set(check.peaks)


# --- click track ----------------------------------------------------------------


def _clicks(beat_times, is_downbeat, n_samples, beat_gain=0.5, downbeat_gain=1.0):
    return make_click_track(
        beat_times=beat_times,
        is_downbeat=is_downbeat,
        n_samples=n_samples,
        sample_rate=SR,
        click_freq=1000.0,
        downbeat_click_freq=1500.0,
        click_duration=0.03,
        beat_gain=beat_gain,
        downbeat_gain=downbeat_gain,
    )


def test_click_track_has_the_song_length_and_float32():
    clicks = _clicks(np.array([0.5, 1.0]), np.array([True, False]), n_samples=3 * SR)
    assert clicks.shape == (3 * SR,)
    assert clicks.dtype == np.float32


def test_each_click_starts_on_its_beat():
    beat_times = np.array([0.5, 1.0, 1.5, 2.0])
    clicks = _clicks(beat_times, np.array([True, False, False, False]), n_samples=3 * SR)
    for t in beat_times:
        s = round(t * SR)
        assert np.all(clicks[s - 50 : s - 1] == 0)
        assert np.abs(clicks[s : s + 50]).max() > 0.1


def test_downbeats_click_louder_and_never_twice():
    beat_times = 0.5 + np.arange(8) * 0.5
    is_downbeat = np.arange(8) % 4 == 0
    clicks = _clicks(beat_times, is_downbeat, n_samples=5 * SR, beat_gain=0.5, downbeat_gain=1.0)
    width = int(0.03 * SR)
    peaks = np.array([np.abs(clicks[round(t * SR) : round(t * SR) + width]).max() for t in beat_times])
    np.testing.assert_allclose(peaks[is_downbeat] / peaks[~is_downbeat].mean(), 2.0, rtol=0.05)
    assert np.abs(clicks).max() <= 1.0


def test_no_beats_give_silence():
    clicks = _clicks(np.array([]), np.array([], dtype=bool), n_samples=SR)
    assert clicks.shape == (SR,)
    assert not clicks.any()


def test_stereo_mix_puts_music_left_and_clicks_right():
    y = np.linspace(-1, 1, 100, dtype=np.float32)
    clicks = np.linspace(0, 0.5, 100, dtype=np.float32)
    mix = stereo_click_mix(y, clicks)
    assert mix.shape == (100, 2)
    assert mix.dtype == np.float32
    np.testing.assert_array_equal(mix[:, 0], y)
    np.testing.assert_array_equal(mix[:, 1], clicks)


def test_stereo_mix_rejects_different_lengths():
    with pytest.raises(ValueError):
        stereo_click_mix(np.zeros(100, np.float32), np.zeros(99, np.float32))


# --- beat_offsets ---------------------------------------------------------------


def _offsets(y, beat_times):
    return beat_offsets(
        y=y,
        sample_rate=SR,
        beat_times=beat_times,
        hop_length=PLOT_MUSIC.onset_hop_length,
        n_fft=PLOT_MUSIC.onset_n_fft,
        max_offset=PLOT_MUSIC.max_onset_offset,
        peak_percentile=PLOT_MUSIC.onset_peak_percentile,
    )


@pytest.mark.parametrize("bpm", [90, 130, 176])
def test_exact_beats_read_the_calibrated_offset(kick_loop, bpm):
    y, beat_times = kick_loop(bpm=bpm, bars=8)
    offsets = _offsets(y, beat_times)
    assert offsets.shape == beat_times.shape
    assert not np.isnan(offsets).any()
    assert 0.002 <= np.median(offsets) <= 0.007


def test_a_late_grid_reads_late_by_that_much(kick_loop):
    y, beat_times = kick_loop(bpm=130, bars=8)
    y_late, _ = kick_loop(bpm=130, bars=8, shift=0.030)
    shift = np.median(_offsets(y_late, beat_times)) - np.median(_offsets(y, beat_times))
    assert shift == pytest.approx(0.030, abs=0.003)


def test_a_beat_far_from_any_onset_gets_nan(kick_loop):
    y, beat_times = kick_loop(bpm=120, bars=4)
    lonely = beat_times[-1] + 0.6  # in the silent tail, ~0.45 s after the last kick ends
    offsets = _offsets(y, np.append(beat_times, lonely))
    assert np.isnan(offsets[-1])
    assert not np.isnan(offsets[:-1]).any()


def test_no_beats_or_no_onsets():
    assert _offsets(np.zeros(SR, np.float32), np.array([])).shape == (0,)
    assert np.isnan(_offsets(np.zeros(SR, np.float32), np.array([0.5]))).all()


# --- kick_by_position -----------------------------------------------------------


def _kick(y, beat_times, beat_in_bar, beats_per_bar=4):
    return kick_by_position(
        y=y,
        sample_rate=SR,
        beat_times=beat_times,
        beat_in_bar=beat_in_bar,
        beats_per_bar=beats_per_bar,
        hop_length=MUSIC.hop_length,
        n_fft=MUSIC.n_fft,
        max_hz=PLOT_MUSIC.kick_max_hz,
        window=PLOT_MUSIC.kick_window,
    )


def test_the_louder_kick_on_one_scores_position_zero(kick_loop):
    y, beat_times = kick_loop(bpm=120, bars=8)
    kick = _kick(y, beat_times, np.arange(len(beat_times)) % 4)
    assert kick.shape == (4,)
    assert kick[0] == 1.0
    assert np.all(kick[1:] < 0.85)


def test_a_bar_two_beats_off_scores_position_two(kick_loop):
    y, beat_times = kick_loop(bpm=120, bars=8)
    kick = _kick(y, beat_times, (np.arange(len(beat_times)) + 2) % 4)
    assert np.argmax(kick) == 2


def test_a_position_without_beats_scores_zero(kick_loop):
    y, beat_times = kick_loop(bpm=120, bars=8)
    kick = _kick(y, beat_times, np.arange(len(beat_times)) % 4, beats_per_bar=5)
    assert kick[4] == 0.0


def test_silence_scores_zero_everywhere():
    assert not _kick(np.zeros(2 * SR, np.float32), np.array([0.5, 1.0]), np.array([0, 1])).any()


# --- window helpers ---------------------------------------------------------------


def test_window_mask():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert window_mask(t, None).all()
    np.testing.assert_array_equal(window_mask(t, (1.0, 2.0)), [False, True, True, False])
    np.testing.assert_array_equal(window_mask(t, (2.0, np.inf)), [False, False, True, True])


def test_beat_mask_keeps_the_beat_that_reaches_into_the_window():
    beats = make_beats(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(_beat_mask(beats, (1.5, 2.5)), [False, True, True, False, False])
    assert _beat_mask(beats, None).all()


def test_beat_steps_extend_the_last_beat():
    beats = make_beats(np.array([0.0, 1.0, 2.0, 3.0]))
    x, y = _beat_steps(beats, "rms_s", np.array([False, True, True, False]))
    np.testing.assert_allclose(x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y, [beats["rms_s"][1], beats["rms_s"][2], beats["rms_s"][2]])


def test_beat_steps_of_no_beats_are_empty():
    beats = make_beats(np.array([0.0, 1.0]))
    x, y = _beat_steps(beats, "rms_s", np.array([False, False]))
    assert x.size == 0 and y.size == 0


# --- panels -------------------------------------------------------------------------


def _ax():
    return Figure().subplots()


def test_section_colors_cover_exactly_the_labels():
    assert set(SECTION_COLORS) == set(SECTION_LABELS)


def test_draw_sections_draws_one_band_per_run():
    beats = _beats_with_labels(["intro", "drop", "drop", "break"])  # beats every 0.5 s from 0.5
    ax = _ax()
    draw_sections(ax, beats)
    spans = [(p.get_x(), p.get_x() + p.get_width()) for p in ax.patches]
    np.testing.assert_allclose(spans, [(0.5, 2.5), (2.5, 6.5), (6.5, 8.5)])


def test_draw_beat_ticks_draws_beats_and_downbeats_apart():
    beats = make_beats(0.5 + np.arange(8) * 0.5)
    ax = _ax()
    draw_beat_ticks(ax, beats)
    collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert [len(c.get_segments()) for c in collections] == [6, 2]


def test_plot_novelty_labels_every_peak_in_the_window():
    check = SectionCheck(
        bar_times=np.arange(10.0),
        novelty=np.array([0, 0.3, 0, 0.1, 0, 0, 0.5, 0, 0.2, 0]),
        boundaries=np.array([0, 6]),
        peaks=np.array([1, 3, 6, 8]),
        prominences=np.array([0.3, 0.1, 0.5, 0.2]),
    )
    ax = _ax()
    plot_novelty(ax, check, prominence=0.1, edge_bars=2, window=(2.0, 7.0))
    assert sorted(t.get_text() for t in ax.texts) == ["0.10", "0.50"]


def test_plot_novelty_hatches_the_first_and_last_edge_bars():
    """20 bars at 0, 1, ..., 19 s; edge_bars = 3: hatched from bar 0 to bar 3 and from bar 17 to bar 19."""
    check = SectionCheck(
        bar_times=np.arange(20.0),
        novelty=np.zeros(20),
        boundaries=np.array([0]),
        peaks=np.array([], dtype=int),
        prominences=np.array([]),
    )
    ax = _ax()
    plot_novelty(ax, check, prominence=0.1, edge_bars=3, window=None)

    hatched = sorted(
        (round(p.get_x(), 6), round(p.get_x() + p.get_width(), 6)) for p in ax.patches if p.get_hatch() == "//"
    )
    assert hatched == [(0.0, 3.0), (17.0, 19.0)]


# --- plot_song --------------------------------------------------------------------


@pytest.fixture
def song(kick_loop):
    y, beat_times = kick_loop(bpm=120, bars=16)
    labels = ["intro"] * 4 + ["drop"] * 8 + ["outro"] * 4
    analysis = make_analysis(make_beats(beat_times, labels), duration=len(y) / SR)
    check = recompute_sections(y, SR, analysis.beats, MUSIC)
    return analysis, waveform_envelope(y, SR, MUSIC.hop_length), check


def _plot(song, window):
    analysis, envelope, check = song
    return plot_song(
        title="Test song",
        analysis=analysis,
        envelope=envelope,
        check=check,
        prominence=MUSIC.novelty_prominence,
        edge_bars=MUSIC.edge_bars,
        window=window,
        fig_width=8,
        panel_height=1.5,
    )


def test_plot_song_lays_out_five_panels_on_one_time_axis(song):
    fig = _plot(song, window=None)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 5
    song_end = song[0].dense.times[-1]
    for ax in fig.axes:
        assert ax.get_xlim() == pytest.approx((0, song_end))
    assert fig.get_suptitle() == "Test song"
    fig.savefig(io.BytesIO(), format="png")


def test_plot_song_shows_only_the_window(song):
    assert _plot(song, window=(10.0, 20.0)).axes[0].get_xlim() == pytest.approx((10.0, 20.0))


def test_plot_song_clips_an_open_window_to_the_song_end(song):
    fig = _plot(song, window=(10.0, np.inf))
    assert fig.axes[0].get_xlim() == pytest.approx((10.0, song[0].dense.times[-1]))


def test_plot_song_legend_lists_only_the_song_s_labels(song):
    legend = _plot(song, window=None).axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["intro", "drop", "outro"]


# --- the real cache -------------------------------------------------------------------


@pytest.mark.integration
def test_the_cached_blue_lock_labels_are_not_stale():
    from amv.music.analysis import cache_paths, load_analysis
    from amv.music.audio import load_audio

    if not PATHS.song_table_path.exists():
        pytest.skip("song table missing; run scripts.build_song_table")
    songs = pd.read_parquet(PATHS.song_table_path)
    rows = songs[songs["name"].str.contains("Blue Lock", regex=False)]
    if rows.empty or not cache_paths(rows["song_id"].iloc[0], PATHS.music_cache_dir, MUSIC)[0].exists():
        pytest.skip("Blue Lock not analysed with the current MUSIC; run scripts.analyse_music")
    row = rows.iloc[0]
    analysis = load_analysis(row["song_id"], PATHS.music_cache_dir, MUSIC)
    y, sr = load_audio(row["audio_proxy_path"], MUSIC.sample_rate)
    check = recompute_sections(y, sr, analysis.beats, MUSIC)
    assert stale_boundaries(analysis.beats, check.boundaries).size == 0
