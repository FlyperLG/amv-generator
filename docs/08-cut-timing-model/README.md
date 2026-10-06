# Step 08: Cut Timing Model

**Goal:** learn **when** to cut (and where effects go) from real AMVs, replacing the step 06 heuristic. It's small, fast, and has the most reliable labels in the whole project, so it's your first learned model and a good test of the training loop.

← [07 Dataset builder](../07-dataset-builder/README.md) · [Project plan](../README.md) · Next → [09 Dynamics ladder](../09-dynamics-ladder/README.md)

---

## 1. Problem definition

- Unit: **sub-beat positions**. Each beat is split into 2 (or 4) positions, so a 3-min song at 150 BPM gives 900 positions.
- Input per position: beat features (step 02: `rms_s`, `onset`, bands, `_d1`, `_d2`), beat phase (`beat_in_bar`, sub-position, encoded as sin/cos), section one-hot.
- Output per position:
  - `p_cut` ∈ [0,1]
  - `p_flash`, `p_zoom`, `p_shake` (multi-label effect heads)
- Inference: peak picking on `p_cut` with a minimum gap (e.g. 0.25 s), plus a threshold chosen on the validation set (or chosen to hit a target cut density, which becomes a user control in step 13).

## 2. Model (CPU-sized)

Start with the smallest model that could work, then grow:

| Version | Model | Params |
|---|---|---|
| T0 | Logistic regression on hand features | ~100 |
| T1 | 1D CNN: 4 layers, kernel 5, dilation 1/2/4/8, 64 channels | ~60k |
| T2 | 2-layer transformer encoder, d=64, local attention window ±16 positions | ~100k |

The dilated CNN (T1) sees ±30 positions (~4 bars) and is usually enough. Only use T2 if T1 plateaus.

## 3. Training

- **Loss:** binary cross-entropy with **positive weighting** (cuts are maybe 25–50 % of positions in hardstyle, far fewer in slow edits). Use `pos_weight` = negatives/positives per batch, or focal loss.
- **Label smoothing in time:** a cut at position p also gives 0.3 to p±1. Editors aren't sample-accurate, and this stabilises training.
- **Batches:** random crops of 256 positions from random AMVs; batch size 32.
- **Optimiser:** AdamW, lr 1e-3, weight decay 1e-2, cosine schedule, 30–50 epochs.
- **Conditioning on style (optional):** a genre embedding (manifest genre), so "sad edit" vs "hardstyle" densities differ.

## 4. Evaluation

- **Cut F1 at ±70 ms** (and at ±1 sub-beat) on the test split, vs the heuristic from step 06.
- **Density error:** |predicted cuts per minute − true cuts per minute|.
- **Per-section F1:** drops vs calm parts. The heuristic is usually fine in drops and bad in calm parts.
- Listen/watch: render 2–3 test songs with A0 dynamics + learned cuts vs heuristic cuts.

## 5. Verification

- [ ] T0 already beats random; T1 beats the heuristic F1 on the test split
- [ ] Training curve: val loss decreases and doesn't diverge from train loss early (if it does, you're overfitting; with few AMVs, that means a smaller model plus dropout)
- [ ] `p_cut` plotted over a test song: peaks on downbeats, denser in drops
- [ ] Inference time for a 3-min song: < 50 ms on CPU

## 6. Pitfalls

- **Leakage:** the same song in train and test makes cut timing look trivially learnable. Group by song.
- **Beat tracking errors** produce misaligned labels. The AMV quality filter in step 07 (cut–beat alignment) matters here.
- **Over-reliance on the beat phase feature:** the model may learn "always cut on downbeats". That's a strong baseline, but check the per-section F1 for real gains.

## 7. CPU budget

T1 on 300 AMVs × ~900 positions ≈ 270k positions per epoch, about **5–20 s per epoch**. All experiments here take minutes.

## 8. Done when

The learned cut model beats the heuristic on the test split and is plugged into `main.py` behind a flag (`--cuts learned|heuristic`).
