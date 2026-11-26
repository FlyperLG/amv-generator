# Step 03: Shot Detection

**Goal:** split every video (AMVs and library episodes) into **shots** (continuous footage between cuts). For the library, also split long shots into retrievable **clips**.

← [02 Music features](../02-music-features/README.md) · [Project plan](../README.md) · Next → [04 Clip features](../04-clip-features/README.md)

---

## 1. Concepts

- **Hard cut:** an instant change between frames. Easy to detect.
- **Soft transitions:** fades, dissolves, flash-whites, which are very common in AMVs. Harder to detect, and they matter: in AMVs, a white flash on a beat *is* a transition we later want to model.
- **Shot vs clip:**
  - AMV → shots only (they're the training targets; each shot = one step k).
  - Library → shots, then clips. A 40 s dialogue shot is useless as one unit, so split it into ~1.5 s windows (with hop ~0.75 s) so retrieval can choose sub-segments.

## 2. Inputs / outputs

| | Path | Content |
|---|---|---|
| In | proxies from step 01 | 640×360 |
| Out | `data/cache/shots/<video_id>_shots_v1.parquet` | one row per shot |
| Out | `data/cache/library/clips_v1.parquet` | one row per library clip (features added in step 04) |

### Shot schema

| Column | Meaning |
|---|---|
| `video_id`, `shot_idx` | |
| `t_in`, `t_out` | seconds (frame-accurate: use frame index / fps) |
| `frame_in`, `frame_out` | frame indices at native fps |
| `cut_type_in` | `hard` / `fade` / `flash` / `dissolve` / `start` |
| `score_in` | detector confidence |

### Clip schema (library)
`clip_id`, `video_id`, `shot_idx`, `t_in`, `t_out`, `is_full_shot`, `episode`, `series`.

## 3. Tasks

| Task | Status | Code | Tests |
|---|---|---|---|
| 3.4 Hand labels + metric | 🔄 AMV minute labelled (76 cuts); loader + P/R/F1 metric implemented and tested; episode minute open | `amv/shots/labels.py`, `amv/shots/metrics.py` | `tests/shots/test_labels.py`, `tests/shots/test_metrics.py` |
| 3.1 Baseline detector | open | | |
| 3.2 Flash / fade handling | open | | |
| 3.3 TransNetV2 (optional) | open | | |
| 3.5 Library clipping | open | | |

3.4 comes first: the labels and the metric are what 3.1–3.3 are tuned against.

### 3.1 Baseline detector
- PySceneDetect `ContentDetector` (HSV-difference based) on the proxy, downscaled further (e.g. `downscale` factor so the frames are ~320 px wide).
- Tune `threshold` (default 27) on hand-labelled data (3.4). AMVs often need a **lower** threshold (fast, stylised cuts); anime episodes are fine near the default.
- Set `min_scene_len` to ~3 frames for AMVs (very short shots exist) and ~8 for episodes.

### 3.2 Flash / fade handling
- Add `ThresholdDetector` (or your own brightness check) to catch fade-to-black/white.
- Per-frame mean luminance: a spike towards white lasting 1–4 frames = **flash**. Mark `cut_type_in = flash`, and don't let the flash frames become their own tiny shot. Merge them into the transition.
- Record flash events separately too (`time`, `duration`, `peak`). Step 07 uses them as effect labels.

### 3.3 Optional upgrade: TransNetV2
- A learned shot-boundary detector, more robust on dissolves. Runs on CPU at ~100–300 fps on 48×27 frames.
- Only switch if your F1 from 3.4 is < 0.9 with PySceneDetect.

### 3.4 Hand-label a validation set (important)
- Take 60 s of the Blue Lock AMV and 60 s of an anime episode. Step through them in a player (frame-by-frame), and write the cut frame numbers into a small CSV (`data/labels/cuts_<video_id>.csv`).
- Metric: precision/recall/F1 with a tolerance of ±2 frames. Write a small function for this; step 11 reuses it.
- *Implemented as:* labels are made in mpv (`.`/`,` step frames) with a small Lua script that appends `frame,type` rows on a key press (`c` hard, `C` soft, `w`/`W` window start/end). Frame numbers are 0-based frames of the **proxy** (constant fps, checked with ffprobe), so they match what the detector reads. The file is self-contained: `frame,type` header, one `window_start` and one `window_end` row: the first and last reviewed frame. A cut at `f` is the boundary between `f-1` and `f`, so it is only visible if both were reviewed: labels and detections count when `start < f <= end` (*decided:* `(start, end]`, not `[start, end)`).
- *Labelling conventions:*
  - `hard`: shot A is directly followed by shot B. The label is the first frame of B (= `frame_in`), also when B starts blurred, dark or filtered.
  - `soft`: there are frames where neither shot is clearly shown (dissolve, fade through black/white, flash, glitch noise). The label is the middle of those frames.
  - Effects inside one shot are **not** cuts: colour cycles, inverted/negative frames, black strobe frames, blur-in, zoom. On the Blue Lock AMV these give ~15 frame-difference peaks as large as real cuts (e.g. 1101–1105, 1140–1146, 1642–1657), a built-in false-positive test for `ContentDetector`.
- *Metric:* one-to-one maximum matching within ±`SHOTS.eval_tol` frames; detections outside the window are ignored; F1 is reported for all cuts and per type (hard / soft), with the detection − label offsets to spot off-by-one bias. Open: a wider tolerance for soft labels, decided once offsets on real detections are known.
  - *Implemented as:* `match_cuts` sorts both lists and walks them once; each label takes the earliest unused detection within reach. This is a maximum matching because all reaches are the same width. `score_cuts(..., cut_type=)` matches against **all** labels and then keeps the pairs whose label has that type, so a found soft cut is neither TP nor FP in the hard score, and `fp` (detections matched to no label) is the same for every type.
- *Labelled so far:* Blue Lock AMV, frames 899–2697 (~30–90 s): 76 cuts, 51 hard / 25 soft. Checked against a frame-difference signal and by eye; 6 fixes applied (one missed cut, two double labels, three soft labels moved to the transition middle).

### 3.5 Library clipping
- For each library shot: if the duration ≤ 2.5 s, make one clip = the whole shot. Otherwise make sliding windows of 1.5 s with a 0.75 s hop, plus one clip for the whole shot.
- Drop clips that are mostly black or white, credits/OP/ED (optional: detect repeated OP/ED by matching CLIP embeddings across episodes in step 04 and filter duplicates).

## 4. Verification

- [ ] F1 ≥ 0.9 at ±2 frames on both hand-labelled minutes
- [ ] Shot-length histogram for the AMV: hardstyle AMVs should peak around 0.2–0.8 s; episodes around 2–5 s
- [ ] Contact sheet: render the first frame of 50 random shots into a grid image. Each should be a distinct scene start, with no half-faded frames
- [ ] Count: AMV shots ≈ (song length × typical cuts per beat). For your AMV, expect a few hundred shots

## 5. Pitfalls

- **Fast camera motion / impact frames** in anime trigger false cuts. Your hand labels will show this. If it's a big problem, raise the threshold slightly and rely on step 04 features to merge nearly identical neighbouring shots.
- **Static held frames** (anime limited animation) produce *no* changes. That's fine; it isn't a cut.
- **Off-by-one:** decide once whether `t_out` is inclusive or exclusive (recommend exclusive: `[t_in, t_out)`) and test it.
- Your AMV has text overlays and effects baked in. This adds false positives; accept some noise in AMV labels.

## 6. CPU budget

PySceneDetect on 320 px proxies: ~300–1000 fps, so a 3-min AMV takes seconds and a 24-min episode < 1 min.

## 7. Done when

Shots exist for the AMV and at least one library episode, F1 is measured and written here, and `clips_v1.parquet` exists for the library.
