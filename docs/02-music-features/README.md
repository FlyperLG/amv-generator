# Step 02: Music Features

**Goal:** turn a song into a **per-beat feature table** (plus a dense per-frame curve) that describes rhythm, energy and *how energy changes* (Δ, Δ²). These are the "music dynamics" that drive the edit.

← [01 Decoding & proxies](../01-decoding-and-proxies/README.md) · [Project plan](../README.md) · Next → [03 Shot detection](../03-shot-detection/README.md)

---

## 1. Concepts

- **Beat grid:** AMV editing is anchored to beats and especially **downbeats** (first beat of a bar). Cuts, flashes and zooms land on them.
- **Two time resolutions:**
  - *Dense*: fixed frame rate (e.g. 43 Hz, from hop 512 @ 22.05 kHz), used for smoothing, plots, control windows (resampled to 10 Hz in step 07).
  - *Beat*: one row per beat, the main unit for timing models and conditioning.
- **Dynamics:** the architecture uses `m`, `Δm`, `Δ²m`. Differentiation **amplifies noise**, so smooth first, then differentiate.

## 2. Inputs / outputs

| | Path | Content |
|---|---|---|
| In | `data/cache/songs.parquet` ([step 01, task 3.7](../01-decoding-and-proxies/README.md#37-song-table-)): one row per AMV in `data/amvs/` and per file in `data/songs/` | `song_id`, `audio_proxy_path` (22 050 Hz mono wav), the original's audio metadata |
| Out | `data/cache/music/<song_id>_<settings_id>_dense.npz` | `times[T]`, feature matrix `[T, 12]` (the 6 normalised curves, then their `_s` versions), feature names |
| Out | `data/cache/music/<song_id>_<settings_id>_beats.parquet` | one row per beat (schema below) |

### Beat table schema

| Column | Type | Meaning |
|---|---|---|
| `t` | float | beat time (s) |
| `is_downbeat` | bool | |
| `beat_in_bar` | int | 0..3 (for 4/4) |
| `bar_idx` | int | |
| `ibi` | float | inter-beat interval (s), local tempo = 60/ibi |
| `rms`, `onset`, `flux`, `low`, `mid`, `high` | float | features pooled over the beat interval [t_i, t_{i+1}) |
| `*_s` | float | smoothed version of each |
| `*_d1`, `*_d2` | float | first/second difference of the smoothed version |
| `section` | str | intro / build / drop / break / outro (heuristic, see 3.5) |

## 3. Tasks

| Task | Status | Code | Tests |
|---|---|---|---|
| 3.1 Load | ✅ done | `amv/music/audio.py::load_audio` | `tests/music/test_audio.py` |
| 3.2 Beats & downbeats | ✅ done | `amv/music/beats.py` | `tests/music/test_beats.py` |
| 3.3 Dense features | ✅ done | `amv/music/dense.py::compute_dense_features` | `tests/music/test_dense.py` |
| 3.4 Smoothing + derivatives | ✅ done | `amv/music/dynamics.py` | `tests/music/test_dynamics.py` |
| 3.5 Section labels | ✅ done | `amv/music/dense.py::compute_timbre_features`, `amv/music/sections.py` (`section_novelty` and `find_section_boundaries` are separate, so the 3.7 plot can draw the novelty) | `tests/music/test_dense.py`, `tests/music/test_sections.py` |
| 3.6 Beat table & script | ✅ done | `amv/music/analysis.py`, `scripts/analyse_music.py` (shared CLI helpers in `scripts/common.py`) | `tests/music/test_analysis.py` |
| 3.7 Plot & click track | to do | | |
| 3.8 Timing benchmark | to do | | |

Order: 3.6 → 3.7. 3.1–3.5 are only building blocks: 3.6 runs them on every song and writes the outputs in section 2. The plot (3.7) reads those files, so it checks exactly what later steps read. The benchmark (3.8) can be done any time after 3.6.

### 3.1 Load ✅
- Load mono at 22 050 Hz (librosa `load(sr=22050, mono=True)`). Normalise peak or loudness so features are comparable across songs.
- *Implemented as:* the inputs are the rows of `data/cache/songs.parquet`, built in step 01 (task 3.7, `python -m scripts.build_song_table`), which also makes every wav. `amv.music.audio.load_audio` only reads mono wavs at the sample rate passed in (`MUSIC.sample_rate`, 22 050 Hz) with `soundfile` (anything else raises) and peak-normalises. Resampling stays in step 01. Spec: `tests/music/test_audio.py`.

### 3.2 Beats & downbeats ✅
- Use `beat_this` (CPU mode) to get beat and downbeat times. Fallback: `librosa.beat.beat_track` (no downbeats; infer them as every 4th beat starting at the strongest-onset phase).
- Sanity rules: drop beats with IBI < 0.25 s or > 1.5 s, then interpolate gaps.
- *Implemented as (measured on Blue Lock):* `beat_this` tracks the 60–90 s breakdown at half tempo (0.68 s instead of 0.34 s), and its downbeats give bars of 1–7 beats. So `clean_beats` drops beats < 0.25 s after the last kept beat, then splits every interval of ≈ n × the median IBI (n ≥ 2) into n parts, capped so no part is shorter than `min_ibi`. This replaces the fixed 1.5 s rule. Known limitation: the median is song-wide, so a real tempo change can be misread as half tempo (0.353 s then 0.6 s turned the slow section into a false 200 BPM), and a song tracked mostly at half tempo stays unrepaired in its faster sections. Switch to a local median if a real song hits this. To keep this from happening silently, a warning is logged if over 5% of the intervals split into parts more than 10% off the median. Measured on 7 proxies: at most 2.1% (Blue Lock), against 40% on a synthetic 1.7x tempo change; no proxy is known to contain a real tempo change, and an exact 2x change is undetectable. Settings are required arguments, passed in from `amv.config` by the caller. `build_beat_grid` treats downbeats as votes: per beat, the bar phase is the majority residue (beat index mod 4) over ±8 bars. Result: a steady 176.5 BPM, and 89/90 bars with 4 beats. Measured on CPU: `detect_beats` ~3 s on the 2:13 song; `clean_beats` + `build_beat_grid` under 15 ms on a synthetic 387-beat grid of the same length. Spec: `tests/music/test_beats.py`.

### 3.3 Dense features (hop = 512 → ~43 Hz) ✅
- `rms` (loudness), `onset_strength` (percussive attack), spectral flux, and band energies from a mel spectrogram: low (<200 Hz: kick/bass), mid, high (>4 kHz: hats/snares).
- Optional: HPSS (harmonic/percussive separation). The percussive part sharpens onsets. It costs ~2× time but is still fast.
- Convert power to dB, then robust-normalise per song (subtract median, divide by IQR) so a quiet song and a loud song are comparable.
- *Implemented as (measured on 7 proxies of 133–200 s):* one STFT feeds every feature. HPSS is left out for now, to be added only if the onset curve looks muddy in the 3.7 plot. Only `rms` and the bands go through dB, because `onset` and `flux` are differences with many zeros. A column with an IQR of 0 becomes all zeros. `flux` uses the linear STFT magnitude and gets the same frame shift as `onset_strength` (`1 + n_fft // (2 * hop_length)`); without it, `flux` runs 2 frames (46 ms) ahead of `onset`. It stays, because its correlation with `onset` once aligned is only 0.29–0.67 (0.61 on the 133 s proxy, taken to be Blue Lock from its length). `compute_dense_features` takes 0.12–0.21 s per song on CPU. Normalised `onset` peaks reach 12–49 and `flux` peaks 4–10, against at most 4.1 for `rms` and the bands, so the two attack curves are heavy-tailed. Spec: `tests/music/test_dense.py`.

### 3.4 Smoothing + derivatives ✅
- **Savitzky–Golay** filter (`scipy.signal.savgol_filter`) on the dense curves. Window ≈ 1 beat for `_s`, polynomial order 2–3. SG can also return derivatives directly (`deriv=1, 2`), which is cleaner than `np.diff` on smoothed data.
- Pool dense → beat by averaging over each beat interval (and max for `onset`).
- Compute `_d1`, `_d2` at **beat level** too (on the smoothed beat series). These are what the dynamics model sees.
- *Implemented as:* the **raw** dense curves are pooled, not the smoothed ones, because averaging over a beat already smooths over one beat. `smooth_dense` makes the dense `_s` curves only, for the npz, the 3.7 plot and step 07; it uses polyorder 2 and the odd window nearest to `MUSIC.dense_sg_window_beats` (1.0) × the median IBI in frames, 15 frames at Blue Lock's 176 BPM. `pool_to_beats` gives beat *i* the frames in [t_i, t_{i+1}). The last beat gets one more interval as long as the previous one, not the time up to the song end, so an outro after the last beat isn't averaged in. A beat with no frame takes its nearest frame. The pooling uses `np.add.reduceat` / `np.maximum.reduceat`, with no Python loop. `beat_dynamics` runs SG on the pooled series with `MUSIC.beat_sg_window` = 5 beats and polyorder 2, using `mode="interp"` and `delta=1`. So `_d1`/`_d2` are **per beat, not per second**, and don't change with tempo. Measured on CPU for a 3-min song (7750 frames, 530 beats, synthetic): `pool_to_beats` 0.11 ms (the loop version took 2.1 ms), the whole of 3.4 1.5 ms. Not checked yet: whether `rms_d2` looks smooth rather than white noise (the pitfall in section 5). That needs the 3.7 plot. Spec: `tests/music/test_dynamics.py`.

### 3.5 Section labels (heuristic, the "plan" tokens) ✅
- Bar-level energy (mean `rms_s` per bar), then a novelty curve (self-similarity of MFCC/chroma per bar, `librosa.segment` utilities), then boundaries at novelty peaks snapped to downbeats, typically every 8 or 16 bars.
- Label segments by energy rank and trend: high energy = drop, rising `rms_d1` = build, low = intro/break/outro by position.
- Imperfect is fine. Step 07 can learn better labels later, and step 13 lets the user override them.
- *Implemented as (measured on the 7 AMV songs, 47–96 bars):* `compute_timbre_features` (12 MFCCs without mfcc_0, plus 12 chroma) is pooled per bar with `pool_to_beats` on the downbeat times. `section_novelty` z-scores each column, takes the cosine self-similarity of the bars, and slides an 8-bar Gaussian checkerboard kernel (σ = 2 bars, |K| summing to 1) along the diagonal. The novelty is **not** scaled to max 1, so a perfect change scores 1 and `MUSIC.novelty_prominence` = 0.1 is an absolute threshold (median novelty 0.08–0.13, 90th percentile 0.22–0.42). `find_section_boundaries` then picks the peaks with `scipy.signal.find_peaks`, with `distance` = `MUSIC.min_section_bars` (4), with peaks within 4 bars of either end dropped. No `librosa.segment` and no 8/16-bar snapping: boundaries are bar indices, so they are on downbeats by construction, and they mostly land 4–12 bars apart anyway. `label_sections` ranks the segments **within the song**: level = (energy − min) / (max − min), energy = mean `rms_s`. "0.5 IQR above the song median" almost never fired, because in a loud song the drops are the median. Rules, first match wins: level ≥ `MUSIC.drop_level` (0.75) → drop; rise (sum of `rms_d1`) ≥ `MUSIC.build_rise` (0.5) **and the next segment is a drop** → build; otherwise intro / outro / break by position. Requiring a drop after a build keeps fade-in intros from becoming builds. Result: 4–13 sections per song; Blue Lock's half-time breakdown (53.7–76.6 s) is a break, with drops at 32–54 s and 105 s+. Known limitation: z-scoring stretches a song's variation to unit size, so a song without real sections still gets boundaries. Not checked yet: whether the labels sound right (needs the 3.7 plot), and whether `MUSIC.build_rise` = 0.5 is a good threshold. Measured on CPU: timbre features ~0.22 s per song (0.8–1.7 s on the first call of a session, a one-time warm-up); boundaries + labels 0.4 ms on a 130-bar, 520-beat synthetic song. Spec: `tests/music/test_sections.py`.

### 3.6 Beat table & script ✅
- Run 3.1–3.5 on every row of `data/cache/songs.parquet` and write the two outputs in section 2 for each song: the beat table (schema above) and the dense npz (the 6 normalised curves plus their `_s` versions). Load the `beat_this` model once per run, not once per song.
- Skip a song whose outputs already exist and are newer than its wav, like the proxies in step 01, so a re-run only analyses new songs. The file name has to change when a setting that affects the outputs changes (like `settings_id` in step 01). Otherwise stale results are reused without any warning.
- Write each file to a temp name first, then rename it, so a crash never leaves half a file.
- Entry point `scripts/analyse_music.py`, following the pattern of `scripts/build_song_table.py`. It reads `amv.config` and passes the settings in.
- Only what a later step reads goes into the cache. Values used only for checking (waveform envelope, novelty curve) are recomputed by 3.7.
- *Implemented as (spec: `tests/music/test_analysis.py`) `amv/music/analysis.py`, the step 02 counterpart of `amv/io/songs.py`:* The step 02 settings move from loose constants into one typed section of `amv/config.py`: `MusicSettings`, a frozen dataclass of the 20 settings that change the outputs, with the configured values as defaults, and `MUSIC = MusicSettings()`. Entry points pass `MUSIC` whole to `analyse_song(s)` or single fields (`MUSIC.min_ibi`) to a step function; variants come from `dataclasses.replace`. Its `settings_id` (stable across processes, safe in file names) replaces the planned `_v1` suffix, and a `settings_<id>.json` sidecar records what each id stands for. `build_beat_table` builds the schema above; the last beat's `ibi` repeats the one before it. `analyse_song(song_id, audio_path, cache_dir, settings, model)` does one song, computing everything before it writes anything. `load_analysis` reads it back for later steps, and `analyse_songs(songs, cache_dir, settings, model)` loops over the song table. That table is passed in as a DataFrame read by the caller, and its `song_id` / `audio_proxy_path` columns are checked before any work. The model is an argument, so the tests use a fake one on a synthetic drum loop. Errors stop the run, as in step 01. Known limitation, as in step 01: a code change without a settings change doesn't invalidate the cache. `analyse_song` loads a cached song ("get it, computing it if needed", for the plot, the benchmark and later steps), but `analyse_songs` skips cached songs without loading them, since it only makes sure the cache is complete. A private `_analyse` runs the chain without any I/O, and `_compute_and_save` adds loading the wav, writing the files and a debug line with the time per song. Run with `python -m scripts.analyse_music [-v]`; the beat model is loaded once per run, even if every song is cached. *Measured 2026-11-11 on the 7 AMV songs (133–200 s, laptop CPU):* 3.9–7.5 s per song, 38.6 s in total, so 25–35× real time; the 200 s song took 6.0 s. The last three songs ran at ~25× against ~34× for the first four, probably the CPU dropping from its boost limit as in step 01's benchmark; 3.8 measures this properly. A second run analysed nothing (0.0 s).

### 3.7 Plot & click track (do this; it's your main check)
- One figure per song, panels on a shared time axis: waveform envelope, `rms_s` (dense and per beat), `rms_d1` / `rms_d2`, beat ticks, downbeat ticks (bold), and section colours as background bands. Plus the section novelty curve, with the `MUSIC.novelty_prominence` threshold and the chosen boundaries. That panel shows why a boundary was or wasn't picked, which is what's needed to tune 3.5.
- A time window (start/end), because a whole song at full width can't show whether a beat lands on the kick.
- Click track: short clicks at the beat times (louder on downbeats) mixed into the wav, for the listening checks in section 4.
- Reads the 3.6 outputs and recomputes the rest; writes into `data/out/`, not the cache.

### 3.8 Timing benchmark
- `benchmarks/bench_02_music.py`, reusing `benchmarks/timing.py`, `results.py` and `plots.py`. Measures the whole analysis of a song and each stage separately (beats, dense, timbre, dynamics, sections), against the < 15 s target in section 4.

## 4. Verification

- [ ] Beat ticks line up with the kick when you listen. Use a click-track check: mix short clicks at beat times into the audio, export, listen
- [ ] Downbeats land on "1" of the bar
- [ ] In the Blue Lock AMV song (hardstyle): the build-up shows rising `rms_d1`, and the drop shows a clear `rms_d2` spike just before/at the drop
- [x] Tempo of the Blue Lock song ≈ 176 BPM (measured with `beat_this` and librosa; not 150 as first assumed), steady through the half-time breakdown after `clean_beats` (`test_bluelock_tempo_is_steady_after_cleaning`)
- [x] Processing a 3-min song takes < 15 s on CPU (3.9–7.5 s per song for 133–200 s songs, measured with `scripts/analyse_music.py`, see 3.6; 3.8 adds a proper benchmark)

## 5. Pitfalls

- **Tempo octave errors** (75 vs 150 BPM): pick the octave whose beat rate best matches the onset density, or let the user specify the BPM range per genre.
- **Ad-hoc normalisation** that uses statistics from other songs leaks information across the train/test split. Normalise per song.
- **Edge effects** of SG filtering at song start/end: use `mode='interp'`.
- **Δ on raw features** looks like white noise. If your `_d2` plot looks like that, the smoothing window is too small.

## 6. CPU budget

librosa features + SG: ~2–5 s per 3-min song. beat_this on CPU: ~5–15 s. Cache the results; a song is analysed once.

## 7. Later upgrade (optional)

Add a learned music embedding pooled per beat (e.g. a small CLAP/MERT-type model). Only do this if steps 09/10 show the hand-crafted features are the bottleneck; it adds minutes per song on CPU.

## 8. Done when

The beat table and dense curves exist for the AMV's audio and 2–3 test songs, and the plots and click-track check pass.
