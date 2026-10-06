# Step 04: Clip Features (`z_sem`, `z_dyn`)

**Goal:** describe every shot/clip with a compact state `s = [z_sem (64), z_dyn (~10)]`. `z_sem` captures *what* is on screen; `z_dyn` captures *how* it moves and looks. This is the state the dynamics models operate on.

← [03 Shot detection](../03-shot-detection/README.md) · [Project plan](../README.md) · Next → [05 Retrieval](../05-retrieval/README.md)

---

## 1. Concepts

- **Why two parts?** CLIP space encodes semantics, not intensity. "Calm talk → fight" is not a straight line in CLIP space. Intensity lives in `z_dyn`, which music drives through velocity/acceleration physics. `z_sem` gets a continuity prior (stay coherent: same character or arc).
- **CLIP vectors live on the unit sphere.** Adding velocities moves them off it. Do the dynamics in **PCA space** (64 dims, roughly Euclidean), or renormalise after every step.
- **Per-clip, not per-frame:** you need 1–2 CLIP frames per clip, not all frames. This is the biggest CPU saving.

## 2. Inputs / outputs

| | Path | Content |
|---|---|---|
| In | proxies, `clips_v1.parquet`, AMV `shots_v1.parquet` | |
| Out | `data/cache/library/clip_feats_v1.npz` | `clip_ids[N]`, `clip_raw[N,512]` (float16 is fine), `z_sem[N,64]`, `z_dyn[N,D]`, `z_dyn_names` |
| Out | `data/cache/pca_v1.npz` | PCA mean, components, explained variance (fit **once**, reuse for AMVs and library) |
| Out | `data/cache/zdyn_stats_v1.json` | per-feature median / IQR for normalisation |
| Out | `data/cache/amv/<amv_id>_states_v1.npz` | same features for each AMV shot (step 07 input) |

## 3. Tasks: `z_sem`

### 3.1 Frame sampling
- For each clip: frames at 33 % and 66 % of its duration (1 frame if shorter than 0.5 s). Read them from the proxy with the frame reader (step 01).
- Preprocess with the CLIP transform (resize the short side to 224, centre-crop 224). Anime is 16:9, so centre-crop cuts off the sides. Optional: average the embeddings of the left/centre/right crops if characters are often off-centre (3× cost).

### 3.2 CLIP encoding on CPU
- `open_clip` model **ViT-B/32** (pretrained `laion2b_s34b_b79k` or `openai`). B/32 is ~4× cheaper than B/16 and good enough for retrieval.
- `model.eval()`, `torch.inference_mode()`, batch size 32–64, `torch.set_num_threads(<physical cores>)`.
- L2-normalise the image embeddings, average the 1–2 frames, then renormalise. That gives `clip_raw[512]`.

### 3.3 PCA → 64
- Fit PCA on library clip embeddings (≥5k samples), with whitening **off** at first.
- Check explained variance: 64 components should keep ~70–85 %. Save the transform; apply the **same** one to AMV shots.
- `z_sem = PCA(clip_raw)`. Keep `clip_raw` too, for text queries in step 13 (CLIP text ↔ image needs the raw space).

## 4. Tasks: `z_dyn`

Compute per clip from **gray 160×90 frames at 8 fps**:

| Feature | How | Meaning |
|---|---|---|
| `motion_mean` | Farneback optical flow magnitude, mean over pixels and frames | overall movement |
| `motion_peak` | 95th percentile over frames of per-frame mean magnitude | bursts / impacts |
| `motion_var` | std over frames | jerky vs steady |
| `zoom` | mean flow **divergence** (∂u/∂x + ∂v/∂y), sign kept | camera push-in (+) / pull-out (−) |
| `pan` | mean flow vector magnitude after removing the divergence part | camera pans |
| `brightness` | mean luma | |
| `contrast` | std luma | |
| `saturation` | mean HSV S (from the RGB proxy frames already read for CLIP) | colourful / desaturated |
| `edge_density` | Canny edge fraction | detail / action lines |
| `duration` | `t_out − t_in` (log-scaled) | |

Optional later: `closeup` (face/character area ratio using an anime face detector), but that adds a dependency, so skip it in v1.

### 4.1 Optical flow details
- `cv2.calcOpticalFlowFarneback(prev, next, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)` is a sensible start.
- Anime has **held frames** (same frame repeated 2–3 times on "twos/threes"). Flow is then 0 between duplicates, which pulls averages down. Option: skip pairs whose frame difference is ~0 before computing the mean.
- Cheap first version: replace flow with mean absolute frame difference. Get the pipeline working, then upgrade.

### 4.2 Normalisation
- Robust normalisation per feature over the **library**: `(x − median) / IQR`, clipped to [−5, 5]. Save the stats and reuse them for AMVs and at inference.

## 5. Verification (this is milestone M1)

- [ ] **Semantic NN test:** pick 10 query clips (fight, dialogue, landscape, close-up face, ...). Show the top 8 neighbours by cosine in `z_sem` as a contact sheet. They should be clearly the same kind of scene
- [ ] **Dynamic sanity:** sort the library by `motion_mean` and look at the top 20 vs bottom 20. Top = action, bottom = still dialogue / scenery
- [ ] **Zoom sign:** find a known push-in shot and check `zoom > 0`
- [ ] **Correlation matrix** of `z_dyn` features: drop or merge features with |r| > 0.9 (redundant)
- [ ] PCA explained-variance curve saved as an image

## 6. Pitfalls

- **Mixing PCA fits** between library and AMVs makes the spaces incompatible. One fit, versioned.
- **Text/subtitles** in frames: CLIP reads text. Hard-subbed episodes cluster by subtitle content. Prefer raw sources, or crop the bottom 15 %.
- **Letterboxing** (black bars) distorts brightness/contrast and CLIP. Detect it with ffmpeg's `cropdetect` in step 01 and crop.
- **AMV shots have effects** (colour grading, shakes, overlays), so their features differ from the clean library footage. That's a **domain gap**. Note it; step 07 handles it partly by normalising per AMV.

## 7. CPU budget (12-episode season ≈ 264 min)

| Part | Estimate |
|---|---|
| CLIP B/32, ~15k clips × 2 frames = 30k images @ ~15–30 img/s | **~20–35 min** |
| Farneback 160×90 @ 8 fps, ~127k frames @ ~300–800 fps | **~3–7 min** |
| Proxy decoding for these reads | a few minutes |

## 8. Done when

`clip_feats_v1.npz` exists for at least one season, the AMV's shot states exist, and the NN contact sheets look right (M1 reached).
