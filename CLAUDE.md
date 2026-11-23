# Working rules for Claude in this repo

## Roles: the user implements, Claude guides and writes tests
- The user writes the implementation logic to learn by doing. Claude guides: concepts, equations, library choices, pitfalls, code review.
- Don't edit implementation files unless explicitly asked.
- Claude writes the pytest tests; they are the specification. For each new feature, write tests in `tests/` first, put open design decisions (edge-case behaviour) in a SPECIFICATION docstring at the top of the test file, run them, and report which fail and what each failure asks the user to implement.
- Record real outputs (ffprobe etc.) as fixtures where possible. Mark tests needing real binaries or media with `@pytest.mark.integration`.
- Exception: docstrings everywhere (`amv/`, `scripts/`, `benchmarks/`, `checks/`, …) are Claude's job. Write and update them without asking whenever the code they describe changes.
- The user always commits. Never run `git commit`; when a unit of work is reviewed, hand over a ready commit message (stage only if asked).

## Code comments
- Inline comments in `amv/` are one-line reminders of the non-obvious bit (e.g. `# zero row: makes index T valid for reduceat`), never more lines than the code they explain.
- The long "why" goes in the chat reply, docstrings or `docs/`. Docstrings keep their normal detail.

## Config and parameters
- Every tunable pipeline/model value lives in `amv/config.py`, with no duplicate literal in a function signature.
- Each config section is one frozen dataclass with field defaults plus a module-level instance: `PATHS`, `DECODING` (step 01, nested `video_proxy`/`audio_proxy`), `MUSIC` (step 02). Only `ROOT` and `DATA_DIR` stay loose (guarded by `tests/test_config.py`). New steps follow the same pattern.
- Algorithm functions take plain required arguments; entry points (scripts, benchmarks, plots) read config and pass them in. Cached producers/orchestrators (`make_video_proxy`, `build_*_table`, `analyse_song`) take the whole section (typed, e.g. `from amv.config import MusicSettings`); step functions get single fields (`MUSIC.min_ibi`). Variants via `dataclasses.replace`. No `from_config()` factories.
- `config.py` is a leaf module (imports nothing from `amv`). ffmpeg/ffprobe lookup lives in `amv/io/ffmpeg.py`.
- Values fixed by design (e.g. `load_audio` accepts mono only) stay hard requirements, not settings. Don't suggest knobs for them.
- Side tools keep their settings in their own top-level folder, never in `config.py` or `scripts/` (pipeline entry points only): `benchmarks/settings.py` (`BENCH`), `checks/settings.py` (`PLOT_MUSIC`, run as `python -m checks.<name>`). This also keeps them out of the `MUSIC.settings_id` cache key.
- Point out inconsistencies between functions in reviews.

## CPU must work, GPU optional; performance first
- The whole pipeline (preprocessing, training, inference) must keep running on CPU alone, since it may later run on a small CPU-only machine. Performance is the top concern.
- Training, dataloaders and models also support CUDA via a configurable `torch.device`. The device is a config setting (in the step's settings dataclass, e.g. `"cpu"`/`"cuda"`, optionally `"auto"` = cuda if available) passed in from the edge; functions take `device` as an argument and never hard-code `.cuda()`.
- Device-dependent dataloader options (`pin_memory`, `num_workers`) and mixed precision follow the device; CPU stays the default in tests, and GPU-only tests are marked and skipped when CUDA is unavailable.
- Architecture direction: shot-level latent dynamics (z_sem + z_dyn, damped velocity/acceleration) plus retrieval, with a flow-matching "action expert" that generates low-dim edit/transition trajectories, never pixels.
- Recommend small models (≤ a few M params), ViT-B/32-class encoders, low-res/low-fps decoding via ffmpeg, few-step flow-matching sampling, batched inference. Always give CPU cost estimates (GPU numbers in addition, not instead).

## Keep `docs/` up to date (Claude's job, don't ask the user)
After each reviewed task:
- update the step README's task status table (§3);
- record deviations from the plan as italic "*Implemented as …*" notes;
- tick verification boxes that were actually checked;
- put benchmark results in the step's "Measured" section;
- update the Status column in `docs/README.md`, plus the CPU budget table when measured.

If a different task order makes sense, change it in the doc and tell the user.
