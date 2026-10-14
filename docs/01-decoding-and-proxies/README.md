# Step 01: Decoding & Proxies

**Goal:** never decode 1080p more than once. Create low-res, low-fps **proxy** files for all analysis, and a fast frame reader that streams frames into numpy.

← [00 Setup](../00-setup/README.md) · [Project plan](../README.md) · Next → [02 Music features](../02-music-features/README.md)

---

## 1. Why this step matters (1080p on CPU)

Your AMV (and most library footage) is **1080p H.264**. Decoding 1080p is the single most expensive operation in preprocessing on CPU:

| Operation | Rough CPU speed (8 physical cores) |
|---|---|
| Decode 1080p H.264 | ~150–400 fps |
| Decode 360p H.264 | ~1500+ fps |
| Scaling after decode | cheap compared with decoding |

Important: `ffmpeg -vf scale=...` on the 1080p file **still decodes every full-resolution frame**. Scaling saves memory and downstream work, not decode time. That's why we decode once into proxies, which is the same proxy workflow professional editors use.

Every later step (shot detection, CLIP, optical flow, dataset building) reads the proxies. Only the **final render** (step 06) uses the 1080p originals.

## 2. Inputs / outputs

| | Path | Format |
|---|---|---|
| In | `data/amvs/*.mp4`, `data/library/**/*.mkv|mp4` | original videos |
| Out | `data/proxies/<video_id>_<settings_id>.mp4` | `PROXY_WIDTH` (640) px wide, even height (640×360 for 16:9), constant frame rate at the source's nominal fps, H.264 `-preset ultrafast -crf 28`, **no audio** (audio comes from the wav below) |
| Out | `data/proxies/<video_id>_<settings_id>.wav` | AMVs only: `AUDIO_SAMPLE_RATE` (22.05 kHz) mono wav for step 02/07 |
| Out | `data/cache/videos.parquet` | one row per video: id, original path, proxy path, duration, fps, width, height, n_frames, codec |

**`video_id`:** never use the raw file name (spaces, brackets, unicode). **Implemented:** `uuid5(NAMESPACE_URL, <resolved absolute path>)`, e.g. `29c61d55-e3db-54b6-8c10-619321a22807`. It is stable for the same file, and two `Episode 01.mkv` files in different folders get different ids. Moving the `data/` folder changes all ids, so proxies would be rebuilt. Keep the original name in `videos.parquet`.

**`settings_id`:** `uuid5` of the ffmpeg arguments that produced the file (shared helper `_ffmpeg_to_file`). Changing any setting in `amv/config.py` (`PROXY_*`, `AUDIO_*`) gives a new file name, so a stale cache is never reused. Old versions stay in `data/proxies/` until deleted by hand. Proxy and wav of the same video share the `video_id` prefix but have different `settings_id`s; later steps get the paths from `videos.parquet`, never by building them.

## 3. Tasks

| Task | Status | Code | Tests |
|---|---|---|---|
| 3.1 Probe | ✅ done | `amv/io/ffmpeg.py::probe`, `amv/io/media.py` | `tests/io/test_ffmpeg.py`, `tests/io/test_media.py` |
| 3.2 Proxy transcode | ✅ done | `amv/io/ffmpeg.py::make_video_proxy` | `tests/io/test_video_proxy.py` |
| 3.3 Frame reader | ✅ done | `amv/io/ffmpeg.py::iter_frames`, `iter_frame_batches` | `tests/io/test_frames.py` |
| 3.4 Audio proxy | ✅ done | `amv/io/ffmpeg.py::make_audio_proxy` | `tests/io/test_audio_proxy.py` |
| 3.5 Timing printout | ⏭ next (transcode already measured, see §8) | | |
| 3.6 Video table | open | | |

Order: 3.5 → 3.6. The benchmark (3.5) needs the frame reader. The video table (3.6) collects the outputs of all the other tasks, so it comes last.

### 3.1 Probe ✅
*Implemented as `probe(path) -> MediaInfo` (typed dataclasses instead of a dict).*
- Write `probe(path) -> dict` that calls `ffprobe -v error -print_format json -show_streams -show_format` and parses the JSON.
- Pull out: duration, `avg_frame_rate` (a fraction like `24000/1001`; parse it as a fraction, **not** a float string), width, height, codec, audio sample rate.
- Watch for **variable frame rate** (VFR): if `r_frame_rate` ≠ `avg_frame_rate`, record a flag. YouTube downloads are sometimes VFR.

### 3.2 Proxy transcode ✅
*Implemented as `make_video_proxy(in_path, out_dir) -> Path`. Skip rule: the proxy exists and is newer than the source (`st_mtime_ns`; not `st_birthtime`, which survives overwrites on NTFS). Transcode timeout: `PROXY_TRANSCODE_TIMEOUT` in `amv/config.py` (300 s, about 2.5 h of footage at the measured speed).*
- One ffmpeg call per video: scale to 640 px wide (keep aspect, even height), force constant frame rate (`-fps_mode cfr` or `-vsync cfr`) to remove VFR headaches, and use `-preset ultrafast`.
- Use all threads (`-threads 0`) but run **one video at a time**, since ffmpeg already parallelises internally.
- Write to a temp name and rename when finished, so a crash never leaves a half file that looks complete.
- Skip videos whose proxy already exists **and** whose source mtime/size hasn't changed.

### 3.3 Frame reader (the important API) ✅
*Implemented as `iter_frames(path, fps, width, height, start=0, end=None, gray=False)` plus `iter_frame_batches(...)`. `start` defaults to 0, not None. `-ss` and `-to` are both **input** options (before `-i`); an input-side `-to` is the absolute end time. Verified frame-accurate on the test clip. There is no ffprobe call: errors are detected from ffmpeg's exit code, with stderr written to a temp file so the pipe cannot deadlock. Measured: one ffprobe call costs ~100 ms on Windows, about as much as decoding a 1 s window, so leaving it out matters for step 04's many short reads.*

*Decisions (see the SPECIFICATION docstring in `tests/io/test_frames.py`): `gray=True` yields 2-D `(h, w)` frames; batches come from a separate `iter_frame_batches(..., batch_size, ...)` that yields `(timestamps[B], frames[B, ...])`; frames are writable; a broken or audio-only file raises `RuntimeError` instead of yielding nothing; there's no timeout.*

`iter_frames(path, fps, width, height, start=None, end=None, gray=False)` should:
- Launch ffmpeg with `-ss`/`-to` (if given), `-vf fps=<fps>,scale=<w>:<h>`, `-f rawvideo -pix_fmt rgb24` (or `gray`), and output to `pipe:1`.
- Read exactly `w*h*3` bytes per frame from stdout, and convert each with `np.frombuffer(...).reshape(h, w, 3)`.
- Yield `(timestamp_seconds, frame)`. Timestamp = `start + i / fps`.
- Optionally yield **batches** (`[B, h, w, 3]`), because CLIP and flow consume batches.
- Clean up the subprocess properly (close pipes, `wait()`), including when the consumer stops early.

Typical calls later:
- Shot detection: proxy, native fps, 320×180, RGB
- CLIP: proxy, specific timestamps (1–2 per clip), 224×224 crop
- Optical flow: proxy, 8 fps, 160×90, gray

### 3.4 Audio proxy (AMVs and songs) ✅
*Implemented as `make_audio_proxy(in_path, out_dir) -> Path`, sharing `_ffmpeg_to_file` with `make_video_proxy` (renamed from `run_proxy_transcode`). Both kinds also write a `<name>.json` sidecar (`source`, `task`, `ffmpeg_args`; written before the media file is renamed into place, so an existing proxy always has one), and `audio_proxy_path` / `video_proxy_path` compute the cached path without running ffmpeg (for 3.6 and cleanup).*
- `ffmpeg -i amv.mp4 -vn -ac 1 -ar 22050 out.wav`. Step 02 uses this for AMVs; songs in `data/songs/` get the same treatment.

### 3.5 Timing printout (needs 3.3)
Add a small benchmark script that prints, for the Blue Lock AMV:
- proxy transcode time and × real time
- frame reader throughput at 8 fps / 320 px from the **proxy** vs from the **original**

### 3.6 Video table
- For every video in `data/amvs/` and `data/library/`: `probe()` the original, `make_video_proxy()` it, and (AMVs only) extract the wav.
- Write `data/cache/videos.parquet` with one row per video: `video_id`, **`kind`** (`"amv"` or `"library"`, from the folder it came from), **`series`** (library only: the first sub-folder of `data/library/`, e.g. `data/library/Blue Lock/S01E01.mkv` → `"Blue Lock"`; step 03's clip table needs it), original path and name, proxy path, audio path (or null), duration, fps, `is_vfr`, width, height, n_frames, codec.
- `data/proxies/` is just a cache keyed by `video_id`; nothing ever lists that folder. Every later step selects videos through this table (`kind == "library"` / `"amv"`), which is why one shared proxy folder is fine.
- Process one video at a time (see 3.2), log progress, and re-running it should be cheap: everything already done is skipped.

## 4. Verification

- [ ] `videos.parquet` has correct duration/fps for the AMV (compare with VLC → Media Information)
- [x] The proxy's duration matches the original within 1 frame (Blue Lock: 135.869 s vs 135.840 s, 4072 vs 4070 frames; Δ < 1 frame at 29.97 fps)
- [x] Frame count from `iter_frames(proxy, fps=8)` ≈ `duration × 8` (`test_bluelock_frame_count_at_8_fps`)
- [x] Frames at t = 0, 30, 60 s from the original and the proxy show the same content, no offset (`test_bluelock_proxy_matches_original`: mean pixel difference < 10)
- [ ] Reading the proxy is ≥5× faster than reading the original at the same output size

## 5. Pitfalls

- **Seek accuracy:** `-ss` *before* `-i` is fast but snaps to keyframes in some cases. In modern ffmpeg it's frame-accurate when transcoding, but verify with the side-by-side test.
- **Pipe deadlock:** if you don't read stderr and ffmpeg writes a lot of logs, the pipe buffer can fill and block. Use `-loglevel error` and/or send stderr to `DEVNULL`.
- **Odd dimensions:** H.264 needs even width/height. Use `scale=640:-2`. *(Handled in 3.2, covered by a test.)*
- **Colour range:** anime is usually limited-range BT.709. It doesn't matter for analysis; remember it for rendering (step 06).
- **Interlaced / telecined** old anime: check for combing artifacts in frames. If present, add `yadif` to the proxy filter chain.

## 6. CPU budget

- 1080p → 360p proxy: estimated ~3–8× real time; **measured ~30×** (see §8), so a 3-min AMV takes ~6 s and a 24-min episode ~50 s.
- Do the whole library overnight once; never again.

## 7. Done when

All videos have proxies plus a `videos.parquet` row, `iter_frames` streams batches correctly, and the benchmark numbers are written down in §8 Measured.

## 8. Measured

Machine: Intel 12th-gen (Alder Lake) CPU, 16 logical cores, Windows 11, ffmpeg 9.0.2 (gyan.dev build).

| What | Input | Result |
|---|---|---|
| Proxy transcode (3.2) | Blue Lock AMV, 1920×1080 H.264, 135.8 s, VFR | **4.6 s ≈ 30× real time**; proxy 640×360 CFR 30000/1001, 20 MB (source 60 MB) |
| Frame reader, proxy vs original (3.5) | | *todo after 3.3* |
