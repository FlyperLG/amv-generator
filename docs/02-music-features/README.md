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

### 3.1 Load
- Load mono at 22 050 Hz (librosa `load(sr=22050, mono=True)`). Normalise peak or loudness so features are comparable across songs.

### 3.2 Beats & downbeats
- Use `beat_this` (CPU mode) to get beat and downbeat times. Fallback: `librosa.beat.beat_track` (no downbeats; infer them as every 4th beat starting at the strongest-onset phase).
- Sanity rules: drop beats with IBI < 0.25 s or > 1.5 s, then interpolate gaps.

### 3.3 Dense features (hop = 512 → ~43 Hz)
- `rms` (loudness), `onset_strength` (percussive attack), spectral flux, and band energies from a mel spectrogram: low (<200 Hz: kick/bass), mid, high (>4 kHz: hats/snares).
- Optional: HPSS (harmonic/percussive separation). The percussive part sharpens onsets. It costs ~2× time but is still fast.
- Convert power to dB, then robust-normalise per song (subtract median, divide by IQR) so a quiet song and a loud song are comparable.

### 3.4 Smoothing + derivatives
- **Savitzky–Golay** filter (`scipy.signal.savgol_filter`) on the dense curves. Window ≈ 1 beat for `_s`, polynomial order 2–3. SG can also return derivatives directly (`deriv=1, 2`), which is cleaner than `np.diff` on smoothed data.
- Pool dense → beat by averaging over each beat interval (and max for `onset`).
- Compute `_d1`, `_d2` at **beat level** too (on the smoothed beat series). These are what the dynamics model sees.

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
- [ ] Tempo for hardstyle ≈ 150 BPM (or 75 if the tracker halves it; detect and fix the octave)
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
