# Step 06: Physics Baseline (A0) & Renderer

**Goal:** the **first complete AMV**, with no machine learning. Heuristic cut timing, a hand-tuned damped-spring dynamics model that makes video intensity follow the music, retrieval from step 05, and an ffmpeg renderer working from the 1080p originals. This is milestone **M2** and ablation rung **A0**.

← [05 Retrieval](../05-retrieval/README.md) · [Project plan](../README.md) · Next → [07 Dataset builder](../07-dataset-builder/README.md)

---

## Part A: Cut timing heuristic

1. Candidate cut times = all beats from step 02.
2. Cut rule per beat:
   - always on a **downbeat**
   - on every beat if `rms_s` > 60th percentile of the song
   - on half-beats (between beats) if in a `drop` section and `onset` > 85th percentile
3. Enforce a minimum shot length (e.g. 0.25 s) and a maximum (e.g. 4 bars in calm parts; split long slots on downbeats).
4. Output: `slots = [(t_start, t_end, beat_idx, section), ...]`.

## Part B: Dynamics (A0)

State per slot k: `s_k = [z_sem, z_dyn]`, edit velocity `v_k`, acceleration `a_k` (only on `z_dyn`).

### B.1 Target intensity from music
Map music features at slot k to a **target `z_dyn`**:
- `target_motion = α₁ · rms_s + α₂ · onset_s` (in normalised units)
- `target_motion_peak = β · max(onset in slot)`
- `target_brightness` follows `high` band energy slightly
- `target_zoom` gets a positive pulse at downbeats in builds

### B.2 Damped spring toward the target (the "physics")
```text
a_k = κ · (target_k − z_dyn_k) − c · v_k + λ · Δ²m_k     # spring + damping + music acceleration kick
v_{k+1} = γ · v_k + a_k
ẑ_dyn_{k+1} = z_dyn_k + v_{k+1}
```
- κ (stiffness) controls how quickly the video follows the music, c/γ the damping, and λ how strongly build-ups push intensity ahead of the drop.
- Starting values: κ=0.3, γ=0.8, c=0.2, λ=0.5. Tune them by watching.
- Read this as "video intensity is a mass on a spring pulled by the music". It's the hand-written version of what steps 09–10 learn.

### B.3 Semantic continuity (z_sem)
- `ẑ_sem_{k+1} = z_sem_k + ε` with small random noise, and a bigger jump allowed at section boundaries (new section = new scene/arc).
- Retrieval's continuity bonus and adjacency penalty do the rest.

### B.4 Re-anchor
After retrieval picks clip c: `z_k+1 := real state of c`. Keep `v` and `a` from the integrator.

## Part C: Timeline format (shared by all later steps)

`data/out/<run_id>/timeline.json`:
```text
{
  "song": "...", "fps": 24, "resolution": [1920,1080],
  "segments": [
    {"dst_in": 12.000, "dst_out": 12.400,
     "clip_id": "...", "src_video": "<original path>", "src_in": 431.2,
     "speed": [[0.0, 1.0], [0.6, 0.5], [1.0, 1.0]],   # (normalised time in segment, playback speed)
     "effects": [{"type": "flash", "at": 0.0, "dur": 0.08, "strength": 0.8},
                 {"type": "zoom",  "at": 0.0, "dur": 0.25, "scale": 1.08}]}
  ]
}
```
Speed curves and effects stay simple for A0 (constant speed, flash on downbeats). They're placeholders for what step 10 generates.

## Part D: Renderer (ffmpeg)

Strategy for CPU and robustness: **render each segment to an intermediate file, then concatenate**.

1. Per segment: cut `src_in` → `src_in + needed_source_duration` from the **1080p original** (seek with `-ss` before `-i`, re-encode), apply speed (`setpts`; for speed curves, split into a few constant-speed pieces at first), scale/crop to the output resolution, force constant fps (`fps=24`), apply effects (`eq` for brightness flash, `zoompan` or `scale+crop` for zoom), and encode with libx264 `-preset veryfast -crf 18`, all at the same pixel format (`yuv420p`).
2. Concatenate with the concat demuxer (`-f concat -safe 0 -i list.txt -c copy`). This works because all segments share codec/fps/resolution.
3. Mux the song audio: `-map 0:v -map 1:a -shortest`, AAC 192k.
4. Frame-accurate durations: compute each segment's frame count as `round(dst_out·fps) − round(dst_in·fps)` so rounding errors don't accumulate and drift off the beat.

Parallelism: segment renders are independent, so run 2–4 ffmpeg processes in parallel (each with `-threads` limited) for a near-linear speedup on 16 threads.

## Part E: `main.py` CLI

`python main.py --song data/songs/x.mp3 --library data/library/ --out data/out/run1 [--seed 0] [--beam 4]`:
load cached features (or compute them if missing) → slots → A0 dynamics + beam search → timeline.json → render.

## Verification (M2)

- [ ] Rendered AMV duration = song duration ± 1 frame
- [ ] **Sync test:** at 3 random downbeats, step frame by frame. The cut lands on the beat frame (± 1 frame)
- [ ] Intensity follows the music: plot `z_dyn.motion_mean` of the chosen clips against `rms_s`. Correlation should be clearly positive (> 0.4)
- [ ] No repeated clip within 30 shots; no visible glitches between segments
- [ ] Model time (slots + dynamics + retrieval) < 2 s; total render time written down here

## Pitfalls

- **Audio/video drift** from accumulated rounding: use integer frame math (Part D.4).
- **Seeking into B-frames** gives a grey or garbled first frame. Always re-encode segments (never `-c copy` on cuts).
- **Speed < 1 needs a longer source**: required source duration = Σ(segment piece duration × speed). Check that it doesn't run past the end of the clip or shot.
- **Colour shifts** between segments: keep `-pix_fmt yuv420p` and the same colour metadata everywhere.
- **Windows paths in concat lists:** use forward slashes or escape properly. Quote paths with spaces.

## CPU budget

Segment render at 1080p `veryfast`: ~1–3× real time per process. With 3 parallel processes, a 3-min AMV takes ~1–2 min. Everything before rendering takes < 2 s.

## Done when

`main.py` produces a watchable, beat-synced AMV from a new song and the library, and the checks above pass. **Celebrate M2**, then start step 07.
