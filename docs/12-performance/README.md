# Step 12: Performance (CPU)

**Goal:** make the pipeline fast on CPU, guided by measurements. Targets: **model time < 5 s per song**, preprocessing at ~1 overnight run per season, and training runs that fit overnight.

← [11 Evaluation](../11-evaluation/README.md) · [Project plan](../README.md) · Next → [13 Controls & UI](../13-controls-and-ui/README.md)

---

Don't wait for this step to think about performance. The rules in the [project plan](../README.md#4-cpu-budget) apply from day one. This step is the dedicated profiling and tuning pass.

## 1. Measure first

- Wrap each pipeline stage with a timer (a small context manager that logs stage name + seconds). Print a table at the end of every `main.py` run.
- Profile hot spots: `python -m cProfile -o prof.out main.py ...` and view the output with `snakeviz`, or use `py-spy record` (sampling, works on running processes, good for finding Python-loop bottlenecks).
- For PyTorch: `torch.profiler` with `record_shapes=True` to see which ops dominate.
- Keep a `docs/12-performance/measurements.md` log: date, change, before/after.

## 2. Preprocessing levers (biggest wins)

| Lever | Expected gain |
|---|---|
| Decode 1080p once → 360p proxies (step 01) | 5–10× on all later reads |
| 8 fps analysis instead of native 24 fps | 3× |
| CLIP ViT-B/32 instead of B/16, 1–2 frames per clip | 4× × 2–4× |
| Flow at 160×90 instead of 640×360 | ~16× |
| Skip duplicate (held) anime frames before flow | 1.5–2× |
| Run *different* videos in parallel processes (`concurrent.futures.ProcessPoolExecutor`), each with limited threads | near-linear up to the physical core count |
| Overlap decoding and compute (reader thread fills a queue, main thread runs CLIP) | 1.2–1.5× |

## 3. PyTorch on CPU

- `torch.set_num_threads(n_physical_cores)` (likely 8 on your 16-thread CPU). Hyper-threads rarely help with matrix multiplication and can slow it down. Measure 4/8/16.
- `torch.inference_mode()` for all inference.
- Keep tensors `float32`. `bfloat16` on CPU helps only on CPUs with AVX-512-BF16/AMX; test before you rely on it.
- **Batch sizes:** inference batches of 32–256 tokens or samples. Tiny batches spend most of their time on Python/dispatch overhead.
- `torch.compile` (Inductor CPU backend) can give 1.3–2× on small transformers. Try it once models are stable (it requires a working C++ compiler on Windows, i.e. the MSVC build tools).
- **ONNX Runtime** export for inference (A5/A6): often 1.5–3× faster than eager PyTorch on CPU for small models, and it can quantise to int8 (dynamic quantisation) for another ~1.5–2× with minor accuracy loss. Check quality with the step 11 metrics.
- DataLoader on Windows: `num_workers=0` is often fastest when the whole dataset is in RAM (worker processes on Windows are spawned, not forked, which is slow to start).

## 4. Algorithmic levers

| Lever | Where |
|---|---|
| **Fewer FM steps:** pick N from the quality-vs-steps curve (step 11) | 10 |
| **Distillation** (reflow / consistency) to N = 1–2 steps | 10 |
| Batch beams × samples into one forward pass per cut | 05, 10 |
| Cache the music conditioning encodings once per song (they don't depend on the beam) | 09, 10 |
| KV caching for the shot-history tokens (A5/B), if the profile shows attention matters | 09 |
| Pre-filter candidates by `z_dyn` intensity bins before the full scoring | 05 |
| Reduce `z_sem` to 32 dims if recall barely changes | 04 |

## 5. Rendering levers

- Render segments in parallel (2–4 ffmpeg processes).
- `-preset veryfast` for previews, `-preset slow` for the final export.
- **Preview mode:** render from the 360p proxies at `ultrafast` for near-instant feedback while tuning; render from the 1080p originals only on export.

## 6. Verification

- [ ] Stage timing table for a 3-min song recorded before and after the tuning pass
- [ ] Model time < 5 s; preview render < 30 s; final 1080p render < 3 min
- [ ] Quality metrics (step 11) unchanged within noise after each optimisation (especially quantisation / fewer FM steps)

## 7. Done when

The measurements log shows the targets met, and every optimisation is recorded with its measured effect and quality check.
