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
| Out | `data/cache/music/<song_id>_dense_v1.npz` | `times[T]`, feature matrix `[T, F_dense]`, feature names |
| Out | `data/cache/music/<song_id>_beats_v1.parquet` | one row per beat (schema below) |

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
| 3.5 Section labels | in progress: timbre features done | `amv/music/dense.py::compute_timbre_features` | `tests/music/test_dense.py` |
| 3.6 Plot | to do | | |

Nothing calls these functions in the pipeline yet: there is no script, and the outputs in section 2 aren't written yet.

### 3.1 Load ✅
- Load mono at 22 050 Hz (librosa `load(sr=22050, mono=True)`). Normalise peak or loudness so features are comparable across songs.
- *Implemented as:* the inputs are the rows of `data/cache/songs.parquet`, built in step 01 (task 3.7, `python -m scripts.build_song_table`), which also makes every wav. `amv.music.audio.load_audio` only reads mono wavs at the sample rate passed in (`AUDIO_SAMPLE_RATE`, 22 050 Hz) with `soundfile` (anything else raises) and peak-normalises. Resampling stays in step 01. Spec: `tests/music/test_audio.py`.

### 3.2 Beats & downbeats ✅
- Use `beat_this` (CPU mode) to get beat and downbeat times. Fallback: `librosa.beat.beat_track` (no downbeats; infer them as every 4th beat starting at the strongest-onset phase).
- Sanity rules: drop beats with IBI < 0.25 s or > 1.5 s, then interpolate gaps.
- *Implemented as (measured on Blue Lock):* `beat_this` tracks the 60–90 s breakdown at half tempo (0.68 s instead of 0.34 s), and its downbeats give bars of 1–7 beats. So `clean_beats` drops beats < 0.25 s after the last kept beat, then splits every interval of ≈ n × the median IBI (n ≥ 2) into n parts, capped so no part is shorter than `min_ibi`. This replaces the fixed 1.5 s rule. Known limitation: the median is song-wide, so a real tempo change can be misread as half tempo (0.353 s then 0.6 s turned the slow section into a false 200 BPM), and a song tracked mostly at half tempo stays unrepaired in its faster sections. Switch to a local median if a real song hits this. To keep this from happening silently, a warning is logged if over 5% of the intervals split into parts more than 10% off the median. Measured on 7 proxies: at most 2.1% (Blue Lock), against 40% on a synthetic 1.7x tempo change; no proxy is known to contain a real tempo change, and an exact 2x change is undetectable. Settings are required arguments, passed in from `amv.config` by the caller. `build_beat_grid` treats downbeats as votes: per beat, the bar phase is the majority residue (beat index mod 4) over ±8 bars. Result: a steady 176.5 BPM, and 89/90 bars with 4 beats. Measured on CPU: `detect_beats` ~3 s on the 2:13 song; `clean_beats` + `build_beat_grid` under 15 ms on a synthetic 387-beat grid of the same length. Spec: `tests/music/test_beats.py`.

### 3.3 Dense features (hop = 512 → ~43 Hz) ✅
- `rms` (loudness), `onset_strength` (percussive attack), spectral flux, and band energies from a mel spectrogram: low (<200 Hz: kick/bass), mid, high (>4 kHz: hats/snares).
- Optional: HPSS (harmonic/percussive separation). The percussive part sharpens onsets. It costs ~2× time but is still fast.
- Convert power to dB, then robust-normalise per song (subtract median, divide by IQR) so a quiet song and a loud song are comparable.
- *Implemented as (measured on 7 proxies of 133–200 s):* one STFT feeds every feature. HPSS is left out for now, to be added only if the onset curve looks muddy in the 3.6 plot. Only `rms` and the bands go through dB, because `onset` and `flux` are differences with many zeros. A column with an IQR of 0 becomes all zeros. `flux` uses the linear STFT magnitude and gets the same frame shift as `onset_strength` (`1 + n_fft // (2 * hop_length)`); without it, `flux` runs 2 frames (46 ms) ahead of `onset`. It stays, because its correlation with `onset` once aligned is only 0.29–0.67 (0.61 on the 133 s proxy, taken to be Blue Lock from its length). `compute_dense_features` takes 0.12–0.21 s per song on CPU. Normalised `onset` peaks reach 12–49 and `flux` peaks 4–10, against at most 4.1 for `rms` and the bands, so the two attack curves are heavy-tailed. Spec: `tests/music/test_dense.py`.

### 3.4 Smoothing + derivatives ✅
- **Savitzky–Golay** filter (`scipy.signal.savgol_filter`) on the dense curves. Window ≈ 1 beat for `_s`, polynomial order 2–3. SG can also return derivatives directly (`deriv=1, 2`), which is cleaner than `np.diff` on smoothed data.
- Pool dense → beat by averaging over each beat interval (and max for `onset`).
- Compute `_d1`, `_d2` at **beat level** too (on the smoothed beat series). These are what the dynamics model sees.
- *Implemented as:* the **raw** dense curves are pooled, not the smoothed ones, because averaging over a beat already smooths over one beat. `smooth_dense` makes the dense `_s` curves only, for the npz, the 3.6 plot and step 07; it uses polyorder 2 and the odd window nearest to `DENSE_SG_WINDOW_BEATS` (1.0) × the median IBI in frames, 15 frames at Blue Lock's 176 BPM. `pool_to_beats` gives beat *i* the frames in [t_i, t_{i+1}). The last beat gets one more interval as long as the previous one, not the time up to the song end, so an outro after the last beat isn't averaged in. A beat with no frame takes its nearest frame. The pooling uses `np.add.reduceat` / `np.maximum.reduceat`, with no Python loop. `beat_dynamics` runs SG on the pooled series with `BEAT_SG_WINDOW` = 5 beats and polyorder 2, using `mode="interp"` and `delta=1`. So `_d1`/`_d2` are **per beat, not per second**, and don't change with tempo. Measured on CPU for a 3-min song (7750 frames, 530 beats, synthetic): `pool_to_beats` 0.11 ms (the loop version took 2.1 ms), the whole of 3.4 1.5 ms. Not checked yet: whether `rms_d2` looks smooth rather than white noise (the pitfall in section 5). That needs the 3.6 plot. Spec: `tests/music/test_dynamics.py`.

### 3.5 Section labels (heuristic, the "plan" tokens)
- Bar-level energy (mean `rms_s` per bar), then a novelty curve (self-similarity of MFCC/chroma per bar, `librosa.segment` utilities), then boundaries at novelty peaks snapped to downbeats, typically every 8 or 16 bars.
- Label segments by energy rank and trend: high energy = drop, rising `rms_d1` = build, low = intro/break/outro by position.
- Imperfect is fine. Step 07 can learn better labels later, and step 13 lets the user override them.

### 3.6 Plot (do this; it's your main check)
One figure per song: waveform envelope, `rms_s`, `rms_d2`, beat ticks, downbeat ticks (bold), section colours as background bands.

## 4. Verification

- [ ] Beat ticks line up with the kick when you listen. Use a click-track check: mix short clicks at beat times into the audio, export, listen
- [ ] Downbeats land on "1" of the bar
- [ ] In the Blue Lock AMV song (hardstyle): the build-up shows rising `rms_d1`, and the drop shows a clear `rms_d2` spike just before/at the drop
- [x] Tempo of the Blue Lock song ≈ 176 BPM (measured with `beat_this` and librosa; not 150 as first assumed), steady through the half-time breakdown after `clean_beats` (`test_bluelock_tempo_is_steady_after_cleaning`)
- [ ] Processing a 3-min song takes < 15 s on CPU

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
