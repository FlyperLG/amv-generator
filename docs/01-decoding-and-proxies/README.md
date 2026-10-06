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
| Out | `data/proxies/<video_id>_360p.mp4` | 640×360, original fps, H.264 `-preset ultrafast -crf 28`, audio dropped (or kept as low-bitrate AAC for AMVs) |
| Out | `data/proxies/<video_id>_audio.wav` | AMVs only: 22.05 kHz mono wav for step 02/07 |
| Out | `data/cache/videos.parquet` | one row per video: id, original path, proxy path, duration, fps, width, height, n_frames, codec |

**`video_id`:** never use the raw file name (spaces, brackets, unicode). Use a short stable hash (e.g. first 12 hex chars of SHA-1 of the *path* or of the first MB of the file), and keep the original name in `videos.parquet`.

## 3. Tasks

### 3.1 Probe
- Write `probe(path) -> dict` that calls `ffprobe -v error -print_format json -show_streams -show_format` and parses the JSON.
- Pull out: duration, `avg_frame_rate` (a fraction like `24000/1001`; parse it as a fraction, **not** a float string), width, height, codec, audio sample rate.
- Watch for **variable frame rate** (VFR): if `r_frame_rate` ≠ `avg_frame_rate`, record a flag. YouTube downloads are sometimes VFR.

### 3.2 Proxy transcode
- One ffmpeg call per video: scale to 640 px wide (keep aspect, even height), force constant frame rate (`-fps_mode cfr` or `-vsync cfr`) to remove VFR headaches, and use `-preset ultrafast`.
- Use all threads (`-threads 0`) but run **one video at a time**, since ffmpeg already parallelises internally.
- Write to a temp name and rename when finished, so a crash never leaves a half file that looks complete.
- Skip videos whose proxy already exists **and** whose source mtime/size hasn't changed.

### 3.3 Frame reader (the important API)
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

### 3.4 Audio extraction (AMVs)
- `ffmpeg -i amv.mp4 -vn -ac 1 -ar 22050 out.wav`. Step 02 uses this for AMVs; songs in `data/songs/` get the same treatment.

### 3.5 Timing printout
Add a small benchmark script that prints, for the Blue Lock AMV:
- proxy transcode time and × real time
- frame reader throughput at 8 fps / 320 px from the **proxy** vs from the **original**

## 4. Verification

- [ ] `videos.parquet` has correct duration/fps for the AMV (compare with VLC → Media Information)
- [ ] The proxy's duration matches the original within 1 frame
- [ ] Frame count from `iter_frames(proxy, fps=8)` ≈ `duration × 8`
- [ ] Save frames at t = 0, 30, 60 s from the original and the proxy side by side. They should show the same content (no offset)
- [ ] Reading the proxy is ≥5× faster than reading the original at the same output size

## 5. Pitfalls

- **Seek accuracy:** `-ss` *before* `-i` is fast but snaps to keyframes in some cases. In modern ffmpeg it's frame-accurate when transcoding, but verify with the side-by-side test.
- **Pipe deadlock:** if you don't read stderr and ffmpeg writes a lot of logs, the pipe buffer can fill and block. Use `-loglevel error` and/or send stderr to `DEVNULL`.
- **Odd dimensions:** H.264 needs even width/height. Use `scale=640:-2`.
- **Colour range:** anime is usually limited-range BT.709. It doesn't matter for analysis; remember it for rendering (step 06).
- **Interlaced / telecined** old anime: check for combing artifacts in frames. If present, add `yadif` to the proxy filter chain.

## 6. CPU budget

- 1080p → 360p proxy: ~3–8× real time, so a 3-min AMV takes ~30–60 s and a 24-min episode ~3–8 min.
- Do the whole library overnight once; never again.

## 7. Done when

All videos have proxies plus a `videos.parquet` row, `iter_frames` streams batches correctly, and the benchmark numbers are written down in this doc (add a "Measured" section).
