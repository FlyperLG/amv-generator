# AMV Generator: Project Plan

An AI system that turns **a song + a library of anime footage** into a beat-synced AMV.
It does **not** generate pixels. It generates a low-dimensional *edit trajectory* (which kind of clip goes where, how intense, how transitions behave). Physics-like latent dynamics and retrieval turn that trajectory into real clips, and ffmpeg renders the result.

Design constraints:
- **Runs entirely on CPU** (preprocessing, training, inference) so the approach can be checked without a GPU.
- **Performance first**: every stage has a CPU budget (see [CPU budget](#4-cpu-budget)).
- **Built by hand**: these docs describe *what* and *why*. You write the implementation.

---

## 1. Architecture at a glance

```text
            SONG                                      ANIME LIBRARY (1080p)
              │                                              │
     [02] music features                         [01] 360p proxies (one-time)
   beats · energy · Δ · Δ²                       [03] shot detection → clips
              │                                  [04] z_sem (CLIP→PCA64) + z_dyn (motion, ...)
              │                                              │
              ▼                                              ▼
     [08] cut timing model ──► cut slots          [05] clip index (numpy / FAISS flat)
              │                                              ▲
              ▼                                              │
   [09] latent dynamics  (damped v / a on z_dyn, continuity on z_sem)
   [10] flow-matching "edit expert" samples controls a_dyn, speed ramps, effects
              │                                              │
              └────────────► predicted state s_k ────────────┘
                                   │  scored retrieval + beam search
                                   ▼
                              timeline.json
                                   │
                    [06] renderer (ffmpeg, original 1080p sources)
                                   ▼
                                AMV.mp4
```

Inspiration: NVIDIA **Alpamayo-R1**. A heavy perception backbone feeds a *small flow-matching action expert*, which outputs **controls** (acceleration/curvature) that are integrated through a kinematic model. Here:

| Alpamayo-R1 | AMV generator |
|---|---|
| Vision-language backbone | Precomputed music + clip features (no large model at runtime) |
| Chain-of-causation reasoning | Discrete per-bar plan: intro / build / drop / outro |
| Flow-matching action expert | Flow-matching edit expert (~0.5–2M params) |
| Controls → kinematic integration | `a_dyn`, Δspeed → damped dynamics `v ← γv + a`, `s ← s + v` |
| Trajectory → vehicle | State → retrieval → real clip |

### Core state representation (per **shot** k, not per frame)

| Part | Size | Meaning | Changes |
|---|---|---|---|
| `z_sem` | 64 | *What* is on screen (CLIP ViT-B/32 → PCA) | slowly; continuity prior |
| `z_dyn` | ~10 | *How* it looks: motion mean/peak, zoom (flow divergence), brightness, saturation, close-up ratio, duration | follows the music; velocity/acceleration physics applies here |

Music state per shot window: `m_k` (smoothed features) plus `Δm_k` and `Δ²m_k`.

---

## 2. Steps

Each step has its own folder with a detailed description: goal, inputs/outputs, concepts, tasks, verification, pitfalls, and a definition of done.

| # | Step | Output | Phase | Status |
|---|---|---|---|---|
| 00 | [Setup](00-setup/README.md) | working env, ffmpeg, repo layout | Foundation | ✅ |
| 01 | [Decoding & proxies](01-decoding-and-proxies/README.md) | 360p proxy files, frame reader | Foundation | 🚧 3.1–3.4 done, next: 3.5 timing benchmark |
| 02 | [Music features](02-music-features/README.md) | per-beat feature table | Features |  |
| 03 | [Shot detection](03-shot-detection/README.md) | shot list per video | Features |  |
| 04 | [Clip features](04-clip-features/README.md) | clip table with `z_sem`, `z_dyn` | Features |  |
| 05 | [Retrieval](05-retrieval/README.md) | clip index, scoring, beam search | Baseline |  |
| 06 | [Physics baseline & renderer](06-physics-baseline-and-renderer/README.md) | **first full AMV (A0)** | Baseline |  |
| 07 | [Dataset builder](07-dataset-builder/README.md) | training samples from AMVs | Learning |  |
| 08 | [Cut timing model](08-cut-timing-model/README.md) | cut probability per beat | Learning |  |
| 09 | [Dynamics ladder A1–A5, B](09-dynamics-ladder/README.md) | deterministic models + baselines | Learning |  |
| 10 | [Flow-matching expert (A6)](10-flow-matching-expert/README.md) | generative edit expert | Learning |  |
| 11 | [Evaluation](11-evaluation/README.md) | metrics, ablation tables, A/B | Research |  |
| 12 | [Performance](12-performance/README.md) | profiled, tuned CPU pipeline | Polish |  |
| 13 | [Controls & UI](13-controls-and-ui/README.md) | aggression knob, style, Gradio app | Polish |  |

### Milestones

| Milestone | After step | What you can show |
|---|---|---|
| M1 Features | 04 | Nearest-neighbour search: a fight clip returns other fight clips |
| **M2 First AMV** | 06 | `python main.py --song x.mp3 --library data/library/` → watchable AMV, no learning |
| M3 Dataset | 07 | Loader returns aligned `(m, s, cuts, controls)` per AMV |
| M4 Learned edits | 09 | A3–A5 beat A0 on recall@k |
| **M5 Generative** | 10 | Flow-matching expert: sharper and more varied edits than A5 |
| M6 Results | 11 | Ablation + data-efficiency curves, blind A/B |
| M7 App | 13 | GUI with controls, sub-10 s model time per song on CPU |

Rough timeline (solo, part-time): M2 ≈ week 4, M5 ≈ month 3, M7 ≈ month 4.

---

## 3. Ablation ladder (the research core)

All variants share the same cut times, retrieval and renderer, so only the dynamics differ.

| ID | Dynamics | Params |
|---|---|---|
| A0 | Hand-tuned spring (step 06) | 0 |
| A1 | Constant velocity `s + v` | 0 |
| A2 | Damped velocity + acceleration, fitted γ | ~10 |
| A3 | A2 + MLP correction | ~0.3M |
| A4 | A2 + GRU correction | ~1M |
| A5 | A2 + small dynamics transformer | ~2–3M |
| A6 | Flow-matching expert generating controls, integrated by A2 physics | ~0.5–2M |
| B | Full transformer (music + shot tokens), CPU-sized | ~5M |

Headline result: **quality as a function of training-data size** (10 / 25 / 50 / 100 % of AMVs), not FLOPs.

---

## 4. CPU budget

Reference machine: 16 logical cores, PyTorch eager, ~30–80 GFLOP/s effective.

| Stage | Frequency | Target |
|---|---|---|
| 1080p → 360p proxy transcode | once per video | ~3–8× real time (decode is the bottleneck); **measured ~30×** |
| Shot detection on proxies | once | >10× real time |
| CLIP ViT-B/32, 1–2 frames per clip | once per library | 30–60 min per 12-episode season |
| Optical flow 160×90 @ 8 fps | once per library | 5–15 min per season |
| Training A6 (~1M params, 300 AMVs) | per experiment | 2–5 min per epoch |
| Inference (400 transitions, 8 FM steps, batched) | per song | 1–3 s |
| Retrieval (400 queries × 15k clips) | per song | <0.1 s |
| Render 3 min @ 1080p | per song | 1–3 min (dominant) |

Rule #1: **batch everything** (no Python loops over transitions or clips in hot paths).
Rule #2: **cache everything** (features are computed once and stored as parquet/npz).
Rule #3: **decode 1080p once** (all experiments read 360p proxies; only the final render touches the originals).

---

## 5. Data

| Folder | Content | Notes |
|---|---|---|
| `data/amvs/` | Reference AMVs (training data) | 1 so far (1080p). Target 200–500. Personal research use only. |
| `data/library/` | Source anime episodes | Used for retrieval and final render (keep 1080p) |
| `data/songs/` | Input songs | wav/mp3/flac |
| `data/proxies/` | 360p / low-fps proxies | generated, git-ignored. AMV and library proxies share this folder; `data/cache/videos.parquet` (`kind` column) says which is which |
| `data/cache/` | Features, indices, datasets | generated, git-ignored |
| `data/out/` | Rendered AMVs, timelines | generated, git-ignored |

**Biggest project risk: training data.** Start collecting AMVs now, while you build steps 00–06. AMVs don't need to match the library's anime, except for val/test; see [step 07 §1](07-dataset-builder/README.md#1-data-collection-start-now-in-parallel-with-steps-0106).

---

## 6. Glossary

| Term | Meaning |
|---|---|
| Shot | Continuous footage between two cuts |
| Clip | A retrievable library unit (a shot, or a ~1.5 s window of a long shot) |
| `z_sem`, `z_dyn` | Semantic / dynamic parts of a clip's state |
| `v_edit`, `a_dyn` | Edit velocity / acceleration over **song time** (shot index k) |
| `u_theta` | Flow-matching vector field over **noise time τ**. Not the same as `v_edit` |
| Control window | ±8 steps at 10 Hz around a cut: speed ramp, zoom, shake, flash, cut offset |
| Re-anchoring | Replacing the predicted state with the retrieved clip's real state after each step |
| Proxy | Low-res, low-fps copy of a video used for analysis |
