"""Tests for amv.io.end_card (docs/01-decoding-and-proxies, task 3.8).

Many downloaded AMVs end with a title card or the editor's logo: seconds of nearly frozen
video with no relation to the edit. They would show up in every later step as one long
"shot" with no cuts and unrelated music. Task 3.8 finds them so the proxies can stop before.

The pure functions get hand-made motion curves; detect_end_card gets synthetic clips from
make_amv_with_end_card (noise footage, then a flat colour card).

SPECIFICATION
  EndCardSettings (amv.config, frozen dataclass; DECODING.end_card, None turns detection off)
      scan       30    seconds at the end of a video that are searched (longest card seen: 11.5 s)
      fps        4     frames per second decoded for the motion curve
      width      64    px of the decoded frames
      height     36    px
      threshold  2.0   mean luma change (0..255) below which the smoothed curve is static
      smooth     1.0   seconds of the running median
      min_tail   3.0   seconds; a shorter static tail is kept (a held frame, a slow ending)
      lookback   6.0   seconds before the video cut searched for the song's end; 0 = off
      audio_rate 8000  Hz the lookback audio is decoded at
      audio_window 0.05 seconds per loudness value
      silence_db -60.0 dBFS; a quieter window is silent
      min_gap    0.25  seconds; a shorter silence is a pause inside the music
      min_dip    18.0  dB a valley must lie below the song AND the outro (no-silence fallback)

  motion_curve(frames) -> np.ndarray
  - frames (N, h, w) -> float64 (N - 1,), element i = mean(|frames[i+1] - frames[i]|).
  - Computed without uint8 wrap-around (0 -> 255 is a change of 255, not 1).
  - Fewer than 2 frames -> empty. Not 3-D -> ValueError. Input not modified.

  find_end_card_start(motion, fps, threshold, smooth, min_tail) -> float | None
  - Running median over round(smooth * fps) samples (odd: +1 if even; <= 1 = none).
  - The static tail is the longest run at the END of the smoothed curve below threshold.
    Starting at index j, the card's first frame is frame j, at j / fps seconds; the tail
    lasts (len(motion) + 1 - j) / fps seconds.
  - Returns j / fps (a float), or None if: the curve is empty, its last value is not
    static, or the tail is shorter than min_tail. The whole curve static -> 0.0.
  - (Decision: a static stretch that does NOT reach the end, e.g. a held frame mid-AMV, is
    never cut. Only the tail can be an end card.)
  - motion not 1-D, fps <= 0, threshold <= 0, smooth < 0 or min_tail < 0 -> ValueError.
  - Input not modified.

  rms_db(samples, sample_rate, window) -> np.ndarray
  - One float64 per COMPLETE window of round(window * sample_rate) samples (a partial last
    window is dropped): 20 * log10(rms), rms floored at 1e-12, so silence is exactly -240.0
    (never -inf/NaN) and a full-scale sine is about -3 dB.
  - Empty input -> empty. Not 1-D, sample_rate <= 0 or window <= 0 -> ValueError. Input not
    modified.

  find_song_end(db, window, silence_db, min_gap) -> float | None
  - Silent: value < silence_db. A gap: consecutive silent windows lasting >= min_gap s
    (count * window). Returns index * window of the first window of the LAST gap (a float;
    0.0 if it starts the curve), or None if there is no gap. Shorter runs are ignored, also
    when they come after the last real gap.
  - Not 1-D, window <= 0 or min_gap < 0 -> ValueError.

  find_valley(db, window, min_dip) -> float | None
  - The valley is the quietest window (the FIRST of equal minima). It counts only if
    max(db[:i+1]) - db[i] >= min_dip (the song before it) AND max(db[i:]) - db[i] >= min_dip
    (the outro after it). Returns i * window (a float), else None; empty -> None.
  - (Decision, 2026-11-17: Awaken and Super Villain fade or dip into their outro without
    digital silence. On 50 ms windows their valleys lie 40/49 and 23.6/22.6 dB below song/
    outro; the AMVs that were right as they were stay below 18 on at least one side (HEROES
    16.3/12.7, Fire & Ice 14.9/6.6), and a plain fade-out has ~0 dB of rise after it.)
  - Not 1-D, window <= 0 or min_dip <= 0 -> ValueError.

  detect_end_card(path, duration, settings, has_audio) -> float | None
  - Pass 1, video: decodes [max(0, duration - scan), end) gray at settings.fps / width /
    height and runs motion_curve + find_end_card_start. No tail -> None (audio not read).
  - Pass 2, audio (only if has_audio and lookback > 0): read_audio over
    [max(0, cut - lookback), cut) at audio_rate, then rms_db + find_song_end. A gap moves
    the cut back to the gap's start. No gap: find_valley on the same curve; a valley moves
    the cut there. Neither keeps the video cut. (Decision: the video finds
    THAT there is a card, the audio finds WHEN the song ended. Measured on 21 cut AMVs: the
    video cut was 0.9-6 s late in 17 of them, always after a digital silence.) A gap that
    starts before the lookback window is clipped to the window's start.
  - Returns seconds of the VIDEO (not of a scanned window), or None.
  - Deterministic: two calls give exactly the same float (both tables cut at it).
  - Missing file -> FileNotFoundError.
"""

import dataclasses

import numpy as np
import pytest

from amv.config import DECODING, EndCardSettings
from amv.io.end_card import (
    detect_end_card,
    find_end_card_start,
    find_song_end,
    find_valley,
    motion_curve,
    rms_db,
)
from amv.io.ffmpeg import probe

FPS = 4.0
PARAMS = {"fps": FPS, "threshold": 2.0, "smooth": 1.0, "min_tail": 3.0}


def footage(n: int, seed: int = 0) -> np.ndarray:
    """n motion samples of busy AMV footage: 10..20."""
    return np.random.default_rng(seed).uniform(10, 20, n)


def curve(*parts) -> np.ndarray:
    return np.concatenate([np.atleast_1d(np.asarray(p, dtype=np.float64)) for p in parts])


# =================================================================================
# motion_curve
# =================================================================================


def test_motion_has_one_value_per_pair_of_frames():
    frames = np.zeros((5, 4, 6), dtype=np.uint8)

    motion = motion_curve(frames)

    assert motion.shape == (4,)
    assert motion.dtype == np.float64


def test_identical_frames_have_no_motion():
    frames = np.full((3, 4, 4), 77, dtype=np.uint8)

    np.testing.assert_array_equal(motion_curve(frames), [0.0, 0.0])


@pytest.mark.parametrize(("a", "b"), [(0, 255), (255, 0)])
def test_motion_does_not_wrap_around_in_uint8(a, b):
    frames = np.stack([np.full((2, 2), a, np.uint8), np.full((2, 2), b, np.uint8)])

    assert motion_curve(frames)[0] == pytest.approx(255.0)


def test_motion_is_the_mean_over_all_pixels():
    first = np.zeros((2, 4), dtype=np.uint8)
    second = first.copy()
    second[0] = 100  # half the pixels change by 100

    assert motion_curve(np.stack([first, second]))[0] == pytest.approx(50.0)


def test_brighter_and_darker_pixels_do_not_cancel_out():
    """Real footage changes both ways at once; the change is |diff| per pixel, then the mean."""
    first = np.full((2, 4), 100, dtype=np.uint8)
    second = first.copy()
    second[0] = 200  # half the pixels +100
    second[1] = 0  # the other half -100

    assert motion_curve(np.stack([first, second]))[0] == pytest.approx(100.0)


@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_two_frames_give_an_empty_curve(n):
    assert motion_curve(np.zeros((n, 4, 4), dtype=np.uint8)).shape == (0,)


def test_rgb_frames_are_rejected():
    with pytest.raises(ValueError):
        motion_curve(np.zeros((3, 4, 4, 3), dtype=np.uint8))


def test_motion_does_not_modify_the_frames():
    frames = np.random.default_rng(1).integers(0, 256, (4, 3, 3), dtype=np.uint8)
    before = frames.copy()

    motion_curve(frames)

    np.testing.assert_array_equal(frames, before)


# =================================================================================
# find_end_card_start
# =================================================================================


def test_finds_the_first_frame_of_a_static_tail():
    # 40 footage samples, the cut into the card, 20 static samples: the card is frame 41
    motion = curve(footage(40), 50, np.zeros(20))

    start = find_end_card_start(motion, **PARAMS)

    assert start == pytest.approx(41 / FPS, abs=1 / FPS)


def test_returns_a_python_float():
    start = find_end_card_start(curve(footage(40), 50, np.zeros(20)), **PARAMS)

    assert type(start) is float


def test_footage_up_to_the_end_has_no_card():
    assert find_end_card_start(footage(60), **PARAMS) is None


def test_a_short_static_tail_is_kept():
    # 8 static samples = 9 frames = 2.25 s < min_tail
    motion = curve(footage(40), 50, np.zeros(8))

    assert find_end_card_start(motion, **PARAMS) is None


def test_min_tail_decides_what_is_long_enough():
    motion = curve(footage(40), 50, np.zeros(8))  # 2.25 s

    assert find_end_card_start(motion, **{**PARAMS, "min_tail": 2.0}) == pytest.approx(41 / FPS, abs=1 / FPS)


def test_a_held_frame_in_the_middle_is_not_a_card():
    motion = curve(footage(20), np.zeros(20), footage(20, seed=1))

    assert find_end_card_start(motion, **PARAMS) is None


def test_a_single_blip_inside_the_card_is_smoothed_away():
    motion = curve(footage(40), 50, np.zeros(10), 30, np.zeros(10))

    assert find_end_card_start(motion, **PARAMS) == pytest.approx(41 / FPS, abs=1 / FPS)


def test_without_smoothing_a_blip_splits_the_card():
    # the blip is sample 51, so only frames 52.. are static: 15 frames = 3.75 s
    motion = curve(footage(40), 50, np.zeros(10), 30, np.zeros(14))

    assert find_end_card_start(motion, **{**PARAMS, "smooth": 0.0}) == pytest.approx(52 / FPS)


def test_a_title_card_and_a_logo_count_as_one_card():
    # footage, cut, 10 s title card, one cut, 20 s logo: the card starts at the title
    motion = curve(footage(40), 50, np.zeros(40), 60, np.zeros(80))

    assert find_end_card_start(motion, **PARAMS) == pytest.approx(41 / FPS, abs=1 / FPS)


def test_a_slowly_animated_logo_is_still_static():
    logo = np.random.default_rng(2).uniform(0.5, 1.5, 20)  # shimmer, well below the threshold

    assert find_end_card_start(curve(footage(40), 50, logo), **PARAMS) == pytest.approx(41 / FPS, abs=1 / FPS)


def test_threshold_decides_what_is_static():
    motion = curve(footage(40), 50, np.full(20, 3.0))

    assert find_end_card_start(motion, **PARAMS) is None
    assert find_end_card_start(motion, **{**PARAMS, "threshold": 4.0}) == pytest.approx(41 / FPS, abs=1 / FPS)


def test_times_are_in_seconds_at_the_given_fps():
    motion = curve(footage(80), 50, np.zeros(40))

    at_4 = find_end_card_start(motion, **PARAMS)
    at_8 = find_end_card_start(motion, **{**PARAMS, "fps": 8.0})

    assert at_4 == pytest.approx(81 / 4, abs=1 / 4)
    assert at_8 == pytest.approx(81 / 8, abs=1 / 8)


def test_a_curve_that_is_static_throughout_starts_at_zero():
    assert find_end_card_start(np.zeros(30), **PARAMS) == 0.0


def test_an_empty_curve_has_no_card():
    assert find_end_card_start(np.zeros(0), **PARAMS) is None


@pytest.mark.parametrize(
    "change",
    [{"fps": 0.0}, {"fps": -1.0}, {"threshold": 0.0}, {"smooth": -0.5}, {"min_tail": -1.0}],
)
def test_bad_parameters_raise(change):
    with pytest.raises(ValueError):
        find_end_card_start(np.zeros(30), **{**PARAMS, **change})


def test_a_2d_curve_raises():
    with pytest.raises(ValueError):
        find_end_card_start(np.zeros((30, 2)), **PARAMS)


def test_does_not_modify_the_curve():
    motion = curve(footage(40), 50, np.zeros(10), 30, np.zeros(10))
    before = motion.copy()

    find_end_card_start(motion, **PARAMS)

    np.testing.assert_array_equal(motion, before)


# =================================================================================
# rms_db
# =================================================================================

RATE = 1000


def sine(seconds: float, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * 50 * t)).astype(np.float32)


def test_one_value_per_complete_window():
    assert rms_db(np.zeros(1000, np.float32), RATE, 0.1).shape == (10,)
    assert rms_db(np.zeros(1050, np.float32), RATE, 0.1).shape == (10,)  # partial window dropped


def test_a_full_scale_sine_is_about_minus_3_db():
    db = rms_db(sine(1.0), RATE, 0.1)

    assert db.dtype == np.float64
    np.testing.assert_allclose(db, -3.01, atol=0.1)


def test_half_the_amplitude_is_6_db_quieter():
    loud = rms_db(sine(1.0), RATE, 0.1)
    quiet = rms_db(sine(1.0, amplitude=0.5), RATE, 0.1)

    np.testing.assert_allclose(loud - quiet, 6.02, atol=0.1)


def test_silence_is_minus_240_db_not_minus_infinity():
    db = rms_db(np.zeros(500, np.float32), RATE, 0.1)

    np.testing.assert_array_equal(db, np.full(5, -240.0))


def test_empty_samples_give_an_empty_curve():
    assert rms_db(np.zeros(0, np.float32), RATE, 0.1).shape == (0,)


@pytest.mark.parametrize(
    ("samples", "rate", "window"),
    [
        (np.zeros((10, 2), np.float32), RATE, 0.1),
        (np.zeros(10, np.float32), 0, 0.1),
        (np.zeros(10, np.float32), RATE, 0.0),
    ],
)
def test_rms_db_rejects_bad_input(samples, rate, window):
    with pytest.raises(ValueError):
        rms_db(samples, rate, window)


def test_rms_db_does_not_modify_the_samples():
    samples = sine(0.5)
    before = samples.copy()

    rms_db(samples, RATE, 0.1)

    np.testing.assert_array_equal(samples, before)


# =================================================================================
# find_song_end
# =================================================================================

WIN = 0.05
GAP = {"window": WIN, "silence_db": -60.0, "min_gap": 0.25}
LOUD, SILENT = -20.0, -240.0


def loudness(*parts) -> np.ndarray:
    """parts: (value, count) pairs, e.g. (LOUD, 20), (SILENT, 10)."""
    return np.concatenate([np.full(n, v, dtype=np.float64) for v, n in parts])


def test_finds_the_start_of_the_gap():
    db = loudness((LOUD, 20), (SILENT, 10), (LOUD, 10))

    start = find_song_end(db, **GAP)

    assert start == pytest.approx(20 * WIN)
    assert type(start) is float


def test_a_short_pause_in_the_music_is_not_a_gap():
    db = loudness((LOUD, 20), (SILENT, 3), (LOUD, 10))  # 0.15 s

    assert find_song_end(db, **GAP) is None


def test_a_gap_of_exactly_min_gap_counts():
    db = loudness((LOUD, 20), (SILENT, 5), (LOUD, 10))  # 5 * 0.05 = 0.25 s

    assert find_song_end(db, **GAP) == pytest.approx(20 * WIN)


def test_the_last_gap_wins():
    db = loudness((LOUD, 10), (SILENT, 8), (LOUD, 10), (SILENT, 6), (LOUD, 5))

    assert find_song_end(db, **GAP) == pytest.approx(28 * WIN)


def test_a_short_pause_after_the_last_gap_is_ignored():
    db = loudness((LOUD, 10), (SILENT, 8), (LOUD, 10), (SILENT, 2), (LOUD, 5))

    assert find_song_end(db, **GAP) == pytest.approx(10 * WIN)


def test_a_gap_up_to_the_end_counts():
    db = loudness((LOUD, 20), (SILENT, 10))

    assert find_song_end(db, **GAP) == pytest.approx(20 * WIN)


def test_a_gap_at_the_start_is_zero():
    db = loudness((SILENT, 10), (LOUD, 20))

    assert find_song_end(db, **GAP) == 0.0


def test_silence_db_decides_what_is_silent():
    db = loudness((LOUD, 20), (-50.0, 10), (LOUD, 10))

    assert find_song_end(db, **GAP) is None
    assert find_song_end(db, **{**GAP, "silence_db": -40.0}) == pytest.approx(20 * WIN)


@pytest.mark.parametrize("db", [np.full(30, LOUD), np.zeros(0)])
def test_no_gap_gives_none(db):
    assert find_song_end(db, **GAP) is None


@pytest.mark.parametrize("change", [{"window": 0.0}, {"min_gap": -0.1}])
def test_find_song_end_rejects_bad_parameters(change):
    with pytest.raises(ValueError):
        find_song_end(np.full(10, LOUD), **{**GAP, **change})


def test_find_song_end_rejects_a_2d_curve():
    with pytest.raises(ValueError):
        find_song_end(np.full((10, 2), LOUD), **GAP)


# =================================================================================
# find_valley
# =================================================================================

DIP = {"window": WIN, "min_dip": 18.0}


def test_finds_a_valley_between_song_and_outro():
    db = loudness((-6.0, 20), (-30.0, 1), (-40.0, 1), (-30.0, 1), (-8.0, 10))  # Awaken-like

    start = find_valley(db, **DIP)

    assert start == pytest.approx(21 * WIN)
    assert type(start) is float


def test_a_fade_out_without_an_outro_is_not_a_valley():
    db = loudness((-8.0, 20), (-20.0, 10), (-40.0, 5))  # loud, then only quieter

    assert find_valley(db, **DIP) is None


def test_a_quiet_start_that_gets_loud_is_not_a_valley():
    db = loudness((-40.0, 5), (-8.0, 20))

    assert find_valley(db, **DIP) is None


def test_a_shallow_dip_is_not_a_valley():
    db = loudness((-14.0, 20), (-26.0, 5), (-16.0, 10))  # HEROES-like: 12 dB / 10 dB

    assert find_valley(db, **DIP) is None


def test_both_sides_must_be_deep_enough():
    deep_before = loudness((-5.0, 20), (-30.0, 5), (-20.0, 10))  # 25 dB before, 10 after
    deep_after = loudness((-20.0, 20), (-30.0, 5), (-5.0, 10))  # 10 before, 25 after

    assert find_valley(deep_before, **DIP) is None
    assert find_valley(deep_after, **DIP) is None


def test_a_dip_of_exactly_min_dip_counts():
    db = loudness((-6.0, 20), (-24.0, 5), (-6.0, 10))

    assert find_valley(db, **DIP) == pytest.approx(20 * WIN)


def test_the_first_of_equal_minima_wins():
    db = loudness((-6.0, 10), (-40.0, 1), (-20.0, 3), (-40.0, 1), (-6.0, 10))

    assert find_valley(db, **DIP) == pytest.approx(10 * WIN)


def test_min_dip_decides_what_is_deep_enough():
    db = loudness((-6.0, 20), (-26.0, 5), (-6.0, 10))  # 20 dB both sides

    assert find_valley(db, **DIP) == pytest.approx(20 * WIN)
    assert find_valley(db, **{**DIP, "min_dip": 25.0}) is None


@pytest.mark.parametrize("db", [np.full(30, LOUD), np.zeros(0)])
def test_a_flat_or_empty_curve_has_no_valley(db):
    assert find_valley(db, **DIP) is None


@pytest.mark.parametrize("change", [{"window": 0.0}, {"min_dip": 0.0}, {"min_dip": -5.0}])
def test_find_valley_rejects_bad_parameters(change):
    with pytest.raises(ValueError):
        find_valley(np.full(10, LOUD), **{**DIP, **change})


def test_find_valley_rejects_a_2d_curve():
    with pytest.raises(ValueError):
        find_valley(np.full((10, 2), LOUD), **DIP)


def test_find_valley_does_not_modify_the_curve():
    db = loudness((-6.0, 20), (-40.0, 3), (-6.0, 10))
    before = db.copy()

    find_valley(db, **DIP)

    np.testing.assert_array_equal(db, before)


# =================================================================================
# detect_end_card (runs ffmpeg)
# =================================================================================


def video_duration(path) -> float:
    return probe(path, DECODING.probe_timeout).video.duration


@pytest.mark.integration
def test_without_silence_the_video_cut_is_kept(make_amv_with_end_card):
    """The default sine runs on under the card: no gap, so the cut is the card's first frame."""
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=3.0)

    start = detect_end_card(src, video_duration(src), EndCardSettings(), has_audio=True)

    assert start == pytest.approx(4.0, abs=1 / EndCardSettings().fps)


@pytest.mark.integration
def test_the_time_is_in_the_video_not_in_the_scanned_window(make_amv_with_end_card):
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=3.0)
    settings = EndCardSettings(scan=5.0)  # scans 2..7 s

    start = detect_end_card(src, video_duration(src), settings, has_audio=True)

    assert start == pytest.approx(4.0, abs=1 / settings.fps)


@pytest.mark.integration
def test_a_clip_without_a_card_gives_none(make_media):
    src = make_media("amv.mp4", video=True, audio=True, duration=3.0)

    assert detect_end_card(src, video_duration(src), EndCardSettings(), has_audio=True) is None


@pytest.mark.integration
def test_a_card_shorter_than_min_tail_gives_none(make_amv_with_end_card):
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=2.0)

    assert detect_end_card(src, video_duration(src), EndCardSettings(min_tail=3.0), has_audio=True) is None


@pytest.mark.integration
def test_detection_is_deterministic(make_amv_with_end_card):
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=3.0, song_end=2.5)
    duration = video_duration(src)

    def run():
        return detect_end_card(src, duration, EndCardSettings(), has_audio=True)

    assert run() == run()


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        detect_end_card(tmp_path / "missing.mp4", 10.0, EndCardSettings(), has_audio=True)


@pytest.mark.integration
def test_the_cut_moves_back_to_where_the_song_ends(make_amv_with_end_card):
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=3.0, song_end=2.5)

    start = detect_end_card(src, video_duration(src), EndCardSettings(), has_audio=True)

    assert start == pytest.approx(2.5, abs=0.1)


@pytest.mark.integration
def test_without_audio_the_video_cut_is_kept(make_amv_with_end_card):
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=3.0, audio=False)

    start = detect_end_card(src, video_duration(src), EndCardSettings(), has_audio=False)

    assert start == pytest.approx(4.0, abs=1 / EndCardSettings().fps)


@pytest.mark.integration
def test_lookback_zero_turns_the_audio_pass_off(make_amv_with_end_card):
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=3.0, song_end=2.5)

    start = detect_end_card(src, video_duration(src), EndCardSettings(lookback=0.0), has_audio=True)

    assert start == pytest.approx(4.0, abs=1 / EndCardSettings().fps)


@pytest.mark.integration
def test_a_gap_longer_than_the_lookback_is_clipped_to_the_window(make_amv_with_end_card):
    # silence 2.5..4.0 s, video cut ~4.0 s, lookback 1 s: the whole window is silent
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=3.0, song_end=2.5)
    settings = EndCardSettings(lookback=1.0)
    video_only = dataclasses.replace(settings, lookback=0.0)
    video_cut = detect_end_card(src, video_duration(src), video_only, has_audio=True)

    start = detect_end_card(src, video_duration(src), settings, has_audio=True)

    assert start == pytest.approx(video_cut - 1.0, abs=1e-6)


@pytest.mark.integration
def test_without_silence_the_cut_moves_into_the_valley(make_amv_with_end_card):
    # song (-21 dB) until 2.0 s, the song 25 dB quieter until 3.0 s, the outro from 3.0 s,
    # while the footage still moves until 4.0 s: no digital silence, but a deep valley
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=3.0, song_end=2.0, outro_start=3.0, gap_db=-25)

    start = detect_end_card(src, video_duration(src), EndCardSettings(), has_audio=True)

    assert 2.0 - 0.1 <= start <= 3.0  # inside the valley, before the outro


@pytest.mark.integration
def test_a_shallow_valley_keeps_the_video_cut(make_amv_with_end_card):
    src = make_amv_with_end_card("amv.mp4", footage=4.0, card=3.0, song_end=2.0, outro_start=3.0, gap_db=-10)

    start = detect_end_card(src, video_duration(src), EndCardSettings(), has_audio=True)

    assert start == pytest.approx(4.0, abs=1 / EndCardSettings().fps)
