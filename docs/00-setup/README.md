# Step 00: Setup

**Goal:** a reproducible CPU-only Python environment, ffmpeg on PATH, and a repo layout that keeps code, data and generated artifacts apart.

← [Project plan](../README.md) · Next → [01 Decoding & proxies](../01-decoding-and-proxies/README.md)

---

## 1. Current state

- Python **3.12.9** installed. Fine for everything below.
- **16 logical cores** (likely 8 physical). Matters for thread settings in step 12.
- **ffmpeg / ffprobe are not installed yet.**
- Repo: `main.py`, `video_encoder.py` (empty), `data/amvs/` with one 1080p AMV.

## 2. Tasks

### 2.1 Install ffmpeg (Windows)
- Install a full build: `winget install Gyan.FFmpeg`, or download a "full" build from gyan.dev and add its `bin/` to PATH.
- Check in a **new** terminal: `ffmpeg -version` and `ffprobe -version`.
- Make sure the build includes `libx264` (for rendering) and `libdav1d` (in case you get AV1 sources).

### 2.2 Virtual environment
- `python -m venv .venv`, then activate it (`.venv\Scripts\Activate.ps1` in PowerShell).
- Pin dependencies in `requirements.txt` with exact versions once things work.

### 2.3 Dependencies (CPU builds)

| Package | Used in | Notes |
|---|---|---|
| `torch` (CPU wheel) | 04, 08–10 | Install from the PyTorch CPU index URL. This avoids pulling multi-GB CUDA wheels |
| `numpy`, `scipy`, `pandas`, `pyarrow` | everywhere | parquet for tables, npz for arrays |
| `librosa`, `soundfile` | 02 | audio loading + features |
| `beat_this` | 02 | modern beat/downbeat tracker, works on CPU. (`madmom` is a fallback but is awkward on 3.12) |
| `scenedetect[opencv]` | 03 | shot detection |
| `opencv-python` | 03, 04 | optical flow, image ops |
| `open_clip_torch` | 04 | CLIP ViT-B/32 |
| `scikit-learn` | 04, 11 | PCA, metrics |
| `faiss-cpu` (optional) | 05 | numpy matrix multiplication is enough up to ~100k clips |
| `matplotlib` | all | sanity plots |
| `tqdm` | all | progress bars |
| `pytest` | all | tests for data contracts |
| `gradio` (later) | 13 | UI |

### 2.4 Repo layout

```text
amv-generator/
  amv/
    __init__.py
    config.py              # paths, fps, resolutions, seeds (one place)
    io/ffmpeg.py           # 01: probe, proxy transcode, frame reader
    audio/analysis.py      # 02
    video/shots.py         # 03
    video/embed.py         # 04: z_sem  (your current video_encoder.py moves here)
    video/motion.py        # 04: z_dyn
    retrieval/index.py     # 05
    retrieval/search.py    # 05: scoring + beam search
    dynamics/physics.py    # 06/09: damped integrator
    dynamics/models.py     # 09: A3–A5, B
    dynamics/flow.py       # 10: flow-matching expert
    timing/cuts.py         # 08
    dataset/build.py       # 07
    render/timeline.py     # 06: timeline schema
    render/ffmpeg_render.py# 06
    eval/metrics.py        # 11
  scripts/                 # CLI entry points per step (preprocess_library.py, ...)
  tests/
  docs/
  data/                    # see project plan §5
  main.py
```

### 2.5 `.gitignore`
Ignore `.venv/`, `data/proxies/`, `data/cache/`, `data/out/`, `*.mp4`, `*.mkv`, `*.wav`, `*.mp3`, `__pycache__/`. Video and audio never belong in git.

### 2.6 Config & reproducibility
- One `config.py` (or a YAML file) with: proxy resolution (640×360), analysis fps (8), flow resolution (160×90), music frame rate, random seeds, paths.
- Every cached artifact gets a **version string** in its filename or metadata (e.g. `clips_v1.parquet`). When you change a feature definition, bump the version so stale cache can't silently mix in.

## 3. Verification

- [ ] `ffmpeg -version` works in a fresh terminal
- [ ] `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints `False` for CUDA. This is intended
- [ ] `python -c "import librosa, cv2, open_clip, scenedetect"` succeeds
- [ ] `pytest` runs (even with zero tests)

## 4. Pitfalls

- PowerShell execution policy can block venv activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- File names with spaces/brackets (like your AMV) break naive shell commands. Pass paths as **argument lists** to `subprocess`, never as one concatenated string.
- Installing `torch` without the CPU index URL downloads the CUDA build (~2.5 GB).

## 5. Done when

Environment installs from `requirements.txt` on a clean machine, and all imports and ffmpeg work.
