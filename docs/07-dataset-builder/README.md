# Step 07: Dataset Builder (AMVs → Training Samples)

**Goal:** turn a collection of real AMVs into aligned training data: music features, the shot state sequence, cut times, and **control windows** around each cut. All learned models (steps 08–10) train on this.

← [06 Physics baseline & renderer](../06-physics-baseline-and-renderer/README.md) · [Project plan](../README.md) · Next → [08 Cut timing model](../08-cut-timing-model/README.md)

---

## 1. Data collection (start now, in parallel with steps 01–06)

- **Target:** 200–500 AMVs. You have 1 (1080p). Prefer 720p+, music-driven edits (hardstyle, phonk, EDM, "flow edits"), clean song audio (no voice-overs), and few heavy overlays.
- Keep a manifest CSV `data/amvs/manifest.csv` with these columns: file, source URL, editor/channel, song title/artist, genre, anime series, quality notes, split (train/val/test).
- **Splits by song and by editor** (never random by shot): the same editor's style or the same song in both train and test inflates your scores.
- **Anime coverage:** training AMVs do **not** need to match the library, because cut timing and `z_dyn` dynamics are largely anime-independent. Diversity matters more: many series, editors, genres and songs. Two exceptions:
  - **val/test AMVs must be of anime you have in `data/library/`.** The Recall@k metric (step 11 §2.2) ranks the true next shot among *that anime's* library clips. Otherwise the true shot competes against footage from a different anime, which is trivially easy to tell apart.
  - **Some overlap in training helps** with the `z_sem` domain gap. A PCA fitted only on one series' library represents other art styles poorly. Aim for 20–30 % of AMVs from series in your library, and fit the step 04 PCA on library clips **plus** AMV shots.
  - Practical rule: pick 2–4 "core" series, collect their episodes for the library and their AMVs for val/test, and collect anything good for training.
- Use only for personal research; don't redistribute the footage or the trained outputs derived from specific editors' work.

## 2. Per-AMV pipeline

Uses steps 01–04 unchanged:

1. Proxy + audio extraction (01)
2. Music features on the AMV's audio (02)
3. Shot detection, including flash events (03)
4. Shot states `z_sem`, `z_dyn` per AMV shot (04), using the **library's** PCA and normalisation stats
5. Alignment + targets (below)

### 2.1 Alignment

- For each shot k: `t_in, t_out`, the beat index containing `t_in`, the offset of the cut relative to the nearest beat (`cut_offset = t_in − nearest_beat`, in beats), the section label, and `music_k` (pooled beat features over the shot interval plus their Δ, Δ²).
- **Per-AMV normalisation of `z_dyn`:** also store `z_dyn` normalised within the AMV (subtract the AMV median). This reduces the domain gap from colour grading and effects (see step 04 pitfalls). Train on the relative version; use library stats at inference.

### 2.2 Targets

| Target | Definition | Used by |
|---|---|---|
| `cut_beat[B]` | 1 if a cut lies within ±½ beat-interval of beat b (and its sub-beat position) | 08 |
| `s[K, 64+D]` | shot state sequence | 09, 10 |
| `v[K]`, `a[K]` | `s_k − s_{k−1}`, `v_k − v_{k−1}` (z_dyn part) | 09 analysis |
| `ctrl[K, 16, C]` | **control window** around each cut (see 2.3) | 10 |
| `effects` | flash / zoom / shake events with time + strength | 08, 10 |

### 2.3 Control windows (input for the flow-matching expert)

Around each cut at time `t_c`: 16 steps at 10 Hz covering `[t_c − 0.8 s, t_c + 0.8 s)`. Per step, C channels:

| Channel | Estimate from AMV |
|---|---|
| `motion` | per-frame flow magnitude (8 fps → resample to 10 Hz), AMV-normalised |
| `zoom` | per-frame flow divergence |
| `shake` | high-frequency part of the global flow (frame-to-frame flow minus its moving average) |
| `brightness` | mean luma (flashes show up as spikes) |
| `speed_proxy` | ratio of flow magnitude to the shot's median flow (weak proxy for slow-mo/speed ramp) |
| `cut_here` | 1 at the step containing the cut |

These are **weak, noisy labels**, and that's acceptable. The model learns the typical *shape* of transitions (dip-then-spike in brightness = flash; motion ramp into a cut = speed ramp).

Also store the conditioning for each window: the music dense features resampled to the same 16 steps, plus the state of the previous 4 shots.

## 3. Storage format

- One file per AMV: `data/cache/dataset/<amv_id>_v1.npz` with all arrays + a small JSON of metadata.
- A global `index_v1.parquet`: amv_id, split, n_shots, n_beats, genre, editor.
- The PyTorch `Dataset` class loads these files; for CPU training, load **everything into RAM** once (300 AMVs × ~300 shots × small arrays ≈ tens of MB).

## 4. Quality filters

Drop or flag AMVs with:
- < 50 shots or average shot length > 4 s (not really beat-synced)
- low cut–beat alignment: the fraction of cuts within ±70 ms of a beat < 40 % (either beat tracking failed or it isn't music-driven)
- heavy overlays: very high `edge_density` everywhere, or text detected in most frames

## 5. Verification (M3)

- [ ] Plot for the Blue Lock AMV: beat grid, cuts, `rms_s`, chosen shots' `motion_mean`, all on one timeline. Cuts should cluster on beats, and motion should rise with energy
- [ ] Cut–beat offset histogram: sharp peak at 0, smaller peaks at ±0.5 beat
- [ ] Average control window around all cuts (per channel): you should see a clear shape (e.g. brightness spike at the cut for flash-heavy editors)
- [ ] Loader returns tensors with the expected shapes; no NaNs; train/val/test don't share editors or songs

## 6. Pitfalls

- **Audio offset:** some AMVs have a short silence or intro, and some downloads have A/V offset. Check cut–beat alignment per AMV; if the histogram's peak is shifted, estimate a constant offset and correct it.
- **Song duplicates:** many AMVs use the same popular songs. Group by song in the split.
- **Shot detection noise** in AMVs with heavy effects creates fake short shots. The quality filters plus a minimum shot length of 2 frames help.

## 7. CPU budget

Per 3-min AMV: proxy ~40 s, music ~15 s, shots ~5 s, CLIP + flow ~60 s, so **~2 min per AMV**, and 300 AMVs ≈ **10 h** (one overnight batch). The script must be **resumable** (skip AMVs whose output exists).

## 8. Done when

At least 50 AMVs are processed (enough to start step 08/09 experiments), the plots look right, and the loader works. Keep adding AMVs over time.
