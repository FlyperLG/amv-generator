"""Tests for amv.io.ffmpeg.iter_frames / iter_frame_batches (docs/01-decoding-and-proxies, task 3.3).

Black-box tests: they only consume what the functions yield. Most use a 3 s test
clip with solid colours (red 0-1 s, green 1-2 s, blue 2-3 s), so the colour of a
frame tells exactly which time it came from.

SPECIFICATION
  iter_frames(path, fps, width, height, start=None, end=None, gray=False)
  - A generator yielding (timestamp, frame) for the time range [start, end).
    start=None means 0, end=None means the end of the video.
  - timestamp: float seconds in the *source* video, = start + i / fps.
  - frame: np.ndarray, dtype uint8, C-contiguous, RGB channel order (not BGR).
      gray=False -> shape (height, width, 3)
      gray=True  -> shape (height, width)        <- 2-D, no channel axis
    (Decision: optical flow / OpenCV want 2-D grey images; a trailing axis of 1
    only gets in the way.)
  - Each frame is its own writable array: `frame /= 255`-style in-place edits work,
    and later frames never overwrite earlier ones.
  - Frame count: (end - start) * fps for a full window (±1 at the very end of a file).
  - Seeking is frame-accurate: the frame at timestamp t shows the content at t.
  - start beyond the end of the video -> yields nothing (no error).

  iter_frame_batches(path, fps, width, height, batch_size, start=None, end=None, gray=False)
  - Same frames as iter_frames, grouped: yields (timestamps, frames) with
      timestamps: np.ndarray float64, shape (B,)
      frames:     np.ndarray uint8,  shape (B, height, width, 3) or (B, height, width)
    B == batch_size except for the last batch, which may be smaller (never empty).
    (Decision: a separate function, so iter_frames always yields one shape.)

  Errors (raised at the latest on the first next(), since these are generators)
  - Missing file -> FileNotFoundError.
  - fps <= 0, width <= 0, height <= 0, start < 0, end <= start, batch_size <= 0
    -> ValueError.
  - ffmpeg fails (not a video, audio-only file, ...) -> RuntimeError mentioning
    "ffmpeg". Never silently yield zero frames for a broken file.

  Resources
  - When the consumer stops early (break / close()), ffmpeg is stopped and the
    file is released. Nothing keeps running in the background.
  - No timeout: reading a 24-min episode must work.
"""

import shutil
import subprocess

import numpy as np
import pytest

from amv.io.ffmpeg import iter_frames, probe

pytestmark = pytest.mark.integration  # every test here runs the real ffmpeg

W, H = 32, 18  # tiny output size keeps the tests fast
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
GRAY_OF = {RED: 76, GREEN: 150, BLUE: 29}  # BT.601 luma, measured with ffmpeg


@pytest.fixture(scope="session")
def rgb_clip(tmp_path_factory):
    """3 s, 24 fps, 320x180 H.264 mp4: red 0-1 s, green 1-2 s, blue 2-3 s."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    out = tmp_path_factory.mktemp("frames") / "rgb.mp4"
    inputs = []
    for colour in ("red", "lime", "blue"):  # "lime" is pure (0, 255, 0) in ffmpeg
        inputs += ["-f", "lavfi", "-i", f"color={colour}:s=320x180:r=24:d=1"]
    subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *inputs,
         "-filter_complex", "[0][1][2]concat=n=3:v=1:a=0",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )  # fmt: skip
    return out


def colour_of(frame: np.ndarray) -> tuple[int, int, int]:
    """Nearest of RED/GREEN/BLUE to the frame's mean colour."""
    mean = frame.reshape(-1, 3).mean(axis=0)
    return min((RED, GREEN, BLUE), key=lambda c: np.abs(mean - c).sum())


def read_all(*args, **kwargs) -> list[tuple[float, np.ndarray]]:
    return list(iter_frames(*args, **kwargs))


# --- What a frame looks like ----------------------------------------------------


def test_yields_timestamp_and_rgb_frame(rgb_clip):
    t, frame = next(iter(iter_frames(rgb_clip, 8, W, H)))

    assert isinstance(t, float)
    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.shape == (H, W, 3)
    assert frame.flags.c_contiguous


def test_channel_order_is_rgb(rgb_clip):
    frames = [f for _, f in read_all(rgb_clip, 8, W, H)]

    # first frame is red: channel 0 high, the others ~0 (BGR would put red in channel 2)
    assert frames[0][..., 0].mean() > 240
    assert frames[0][..., 1:].mean() < 15
    assert colour_of(frames[12]) == GREEN
    assert colour_of(frames[-1]) == BLUE


@pytest.mark.parametrize(("width", "height"), [(160, 90), (224, 224), (31, 17)])
def test_output_size(rgb_clip, width, height):
    _, frame = next(iter(iter_frames(rgb_clip, 8, width, height)))

    assert frame.shape == (height, width, 3)


def test_gray_frames_are_2d_luma(rgb_clip):
    frames = [f for _, f in read_all(rgb_clip, 8, W, H, gray=True)]

    assert frames[0].shape == (H, W)
    assert frames[0].dtype == np.uint8
    assert frames[0].mean() == pytest.approx(GRAY_OF[RED], abs=5)
    assert frames[12].mean() == pytest.approx(GRAY_OF[GREEN], abs=5)
    assert frames[-1].mean() == pytest.approx(GRAY_OF[BLUE], abs=5)


def test_frames_are_writable_and_independent(rgb_clip):
    frames = [f for _, f in read_all(rgb_clip, 8, W, H)]

    frames[0][:] = 7  # must not raise "assignment destination is read-only"
    assert colour_of(frames[1]) == RED  # ...and must not change any other frame
    assert colour_of(frames[-1]) == BLUE


# --- How many frames, and when --------------------------------------------------


@pytest.mark.parametrize("fps", [8, 24, 2.5])
def test_frame_count_and_timestamps_full_video(rgb_clip, fps):
    timestamps = [t for t, _ in read_all(rgb_clip, fps, W, H)]

    assert len(timestamps) == pytest.approx(3.0 * fps, abs=1)
    assert timestamps[0] == pytest.approx(0.0)
    assert timestamps == pytest.approx([i / fps for i in range(len(timestamps))])


def test_start_seeks_accurately(rgb_clip):
    frames = read_all(rgb_clip, 8, W, H, start=1.5)

    assert len(frames) == 12  # 1.5 s left at 8 fps
    assert frames[0][0] == pytest.approx(1.5)
    assert frames[1][0] == pytest.approx(1.625)
    assert colour_of(frames[0][1]) == GREEN  # content at 1.5 s, not at 0 s


def test_end_stops_reading(rgb_clip):
    frames = read_all(rgb_clip, 8, W, H, end=1.0)

    assert len(frames) == 8
    assert all(colour_of(f) == RED for _, f in frames)


def test_start_and_end_window(rgb_clip):
    # -to must be the absolute end time, not a duration counted from start
    frames = read_all(rgb_clip, 8, W, H, start=1.25, end=2.75)

    assert len(frames) == 12
    assert frames[0][0] == pytest.approx(1.25)
    assert frames[-1][0] == pytest.approx(2.625)
    assert colour_of(frames[0][1]) == GREEN
    assert colour_of(frames[-1][1]) == BLUE


def test_start_past_end_of_video_yields_nothing(rgb_clip):
    assert read_all(rgb_clip, 8, W, H, start=10.0) == []


def test_accepts_str_path(rgb_clip):
    assert len(read_all(str(rgb_clip), 8, W, H, end=1.0)) == 8


# --- Batches --------------------------------------------------------------------


def test_batches_match_single_frames(rgb_clip):
    from amv.io.ffmpeg import iter_frame_batches

    singles = read_all(rgb_clip, 8, W, H)
    batches = list(iter_frame_batches(rgb_clip, 8, W, H, batch_size=5))

    assert [len(ts) for ts, _ in batches] == [5, 5, 5, 5, 4]  # 24 frames
    for ts, frames in batches:
        assert ts.dtype == np.float64
        assert frames.dtype == np.uint8
        assert frames.shape == (len(ts), H, W, 3)
    all_ts = np.concatenate([ts for ts, _ in batches])
    all_frames = np.concatenate([fr for _, fr in batches])
    assert all_ts == pytest.approx([t for t, _ in singles])
    assert np.array_equal(all_frames, np.stack([f for _, f in singles]))


def test_gray_batches(rgb_clip):
    from amv.io.ffmpeg import iter_frame_batches

    ts, frames = next(iter(iter_frame_batches(rgb_clip, 8, W, H, batch_size=4, start=1.0, gray=True)))

    assert frames.shape == (4, H, W)
    assert ts == pytest.approx([1.0, 1.125, 1.25, 1.375])


def test_batch_size_larger_than_video(rgb_clip):
    from amv.io.ffmpeg import iter_frame_batches

    batches = list(iter_frame_batches(rgb_clip, 8, W, H, batch_size=1000))

    assert len(batches) == 1
    assert batches[0][1].shape[0] == 24


# --- Stopping early -------------------------------------------------------------


def test_stopping_early_releases_the_file(rgb_clip, tmp_path):
    # On Windows a file that ffmpeg still has open cannot be deleted, so this fails
    # if the ffmpeg process is left running after the consumer stops.
    copy = tmp_path / "copy.mp4"
    shutil.copy(rgb_clip, copy)

    gen = iter_frames(copy, 24, W, H)
    for i, _ in enumerate(gen):
        if i == 2:
            break
    gen.close()

    copy.unlink()  # PermissionError here = ffmpeg still running
    assert not copy.exists()


def test_can_read_the_same_file_many_times(rgb_clip):
    # Leaking a process (or a pipe) per call shows up quickly when looping like step 04 will.
    for _ in range(20):
        gen = iter_frames(rgb_clip, 24, W, H)
        next(gen)
        gen.close()

    assert len(read_all(rgb_clip, 8, W, H)) == 24


# --- Errors ---------------------------------------------------------------------


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        next(iter(iter_frames(tmp_path / "missing.mp4", 8, W, H)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"fps": -1},
        {"width": 0},
        {"height": -18},
        {"start": -1.0},
        {"start": 2.0, "end": 1.0},
        {"start": 1.0, "end": 1.0},
    ],
)
def test_invalid_arguments_raise(rgb_clip, kwargs):
    args = {"fps": 8, "width": W, "height": H} | kwargs

    with pytest.raises(ValueError):
        next(iter(iter_frames(rgb_clip, **args)))


def test_invalid_batch_size_raises(rgb_clip):
    from amv.io.ffmpeg import iter_frame_batches

    with pytest.raises(ValueError):
        next(iter(iter_frame_batches(rgb_clip, 8, W, H, batch_size=0)))


def test_non_video_file_raises(tmp_path):
    fake = tmp_path / "not_a_video.mp4"
    fake.write_text("this is not a video")

    with pytest.raises(RuntimeError, match="ffmpeg"):
        read_all(fake, 8, W, H)


def test_audio_only_file_raises(make_media):
    song = make_media("song.mp3", video=False, audio=True)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        read_all(song, 8, W, H)


# --- Real AMV (doc §4 verification) ---------------------------------------------


def test_bluelock_frame_count_at_8_fps(bluelock_amv):
    duration = probe(bluelock_amv).video.duration

    n = sum(1 for _ in iter_frames(bluelock_amv, 8, 160, 90))

    assert n == pytest.approx(duration * 8, abs=2)


def test_bluelock_proxy_matches_original(bluelock_amv, tmp_path):
    # Same content at the same timestamp in the original and the proxy (no time offset).
    from amv.io.ffmpeg import make_video_proxy

    proxy = make_video_proxy(bluelock_amv, tmp_path)

    for t in (0.0, 30.0, 60.0):
        _, orig = next(iter(iter_frames(bluelock_amv, 8, 160, 90, start=t, end=t + 1)))
        _, prox = next(iter(iter_frames(proxy, 8, 160, 90, start=t, end=t + 1)))
        diff = np.abs(orig.astype(np.int16) - prox.astype(np.int16)).mean()
        assert diff < 10, f"frames at t={t}s differ by {diff:.1f} on average"
