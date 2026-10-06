# Step 10: Flow-Matching Edit Expert (A6)

**Goal:** an Alpamayo-R1-style **generative action expert**. Instead of predicting one averaged next state (which tends to be bland), it **samples** a sharp, plausible edit: the controls for the next shot transition (acceleration of `z_dyn`, speed ramp, zoom, shake, flash, cut offset). These controls are integrated through the shared physics and then retrieved. It never touches pixels.

← [09 Dynamics ladder](../09-dynamics-ladder/README.md) · [Project plan](../README.md) · Next → [11 Evaluation](../11-evaluation/README.md)

---

## 1. Why flow matching here

- The same music admits **many good edits** (a multimodal distribution). Regression (A3–A5) averages the modes, the average state is generic, and retrieval returns generic clips.
- Flow matching learns the whole distribution and samples one mode, while staying cheap: a small network, a few sampling steps.
- Alpamayo-R1 uses exactly this pattern for driving: a flow-matching expert outputs **controls** (acceleration, curvature) that are integrated through a kinematic model, which guarantees physically valid trajectories. Here, the damped integrator plays the kinematic model's role.

**Naming discipline:** flow matching has its own vector field over *noise time* τ. Call it `u_theta`. The edit velocity over *song time* is `v_edit`. They are unrelated; keep the names separate in code.

## 2. What is generated: `x1`

Per cut k, a **control tensor** `x1 ∈ R^{L × C}`:

| Part | Shape | Content |
|---|---|---|
| Shot controls | 1 × (64 + D) | `â_k` for the next state (as in A2–A5) |
| Transition window | 16 × C_t | at 10 Hz around the cut: speed, zoom, shake, brightness/flash, cut-offset logits (from step 07 §2.3) |

Simplest layout: treat it as a sequence of L = 17 tokens (1 shot token + 16 window tokens), each projected to d = 128.

Normalise every channel to zero mean and unit variance over the training set; flow matching assumes data on a scale similar to N(0, I).

## 3. Conditioning `c`

- Last H = 8 shot states and their `v`, `a` (from the physics)
- Music window: the dense features resampled to the 16 window steps + pooled `m_k`, `Δm_k`, `Δ²m_k`
- Section token (the discrete "plan", the analogue of Alpamayo's reasoning stage)
- Optional style/genre embedding
- Encode `c` into a few prefix tokens (e.g. 1 for shot history, 1 for music, 1 for section/style), so it's all one sequence with self-attention and no cross-attention module.

## 4. Training (rectified flow / conditional flow matching)

For each training sample (x1, c):
1. `x0 ~ N(0, I)` with the same shape as x1
2. `τ ~ U(0, 1)`. Try also logit-normal sampling, which emphasises the middle τ and often helps
3. `x_τ = (1 − τ) · x0 + τ · x1`
4. Target field: `x1 − x0`
5. `L_fm = ‖ u_theta(x_τ, τ, c) − (x1 − x0) ‖²` (mean over tokens and channels)

Network `u_theta`:
- Input tokens: [prefix tokens of c] + [L tokens of x_τ]
- τ embedding: sinusoidal (64 dims) → MLP → added to every token, or used as **FiLM/adaLN** (scale + shift on each LayerNorm), which works better at small sizes
- 4 layers, d = 128, 4 heads, FFN 512, so **~0.8M params**
- Output: a linear head on the x tokens → `u` with the same shape as x1

Classifier-free guidance (optional, for controllability): drop `c` (replace it with a learned null token) 10 % of the time during training. At sampling: `u = u_uncond + w · (u_cond − u_uncond)` with w ∈ [1, 3]. This doubles the inference cost.

Optimiser: AdamW, lr 3e-4, wd 0.01, batch 128, 100–200 epochs, EMA of the weights (decay 0.999) for sampling. EMA noticeably improves sample quality.

## 5. Sampling (inference)

```text
x ← N(0, I)
for i in 0..N−1:   τ_i = i / N
    x ← x + (1/N) · u_theta(x, τ_i, c)        # Euler
return x                                       # ≈ x1
```
- N = 4–10 Euler steps (rectified flow paths are nearly straight). Try Heun (2nd order) at N=4 as an alternative.
- **Batch all candidates:** for beam search, sample S = 4–8 control tensors per beam in one batch, giving diverse proposals, and let retrieval scoring pick the best. Generate, then select.

Integrate:
```text
v_edit ← γ ⊙ v_edit + â_k            # â_k = shot-control part of x
ŝ_{k+1} = s_k + v_edit  → retrieval (step 05) → re-anchor
transition window part → speed curve + effects in timeline.json (step 06 format)
```

## 6. Ablations within A6

| Variant | Question |
|---|---|
| A6-states | Generate `ŝ` directly instead of controls. Does the physics integration still help? |
| A6-noWindow | Shot controls only, transitions from heuristics. How much do the generated transitions add? |
| N = 1, 2, 4, 8, 16 | **Quality vs speed curve.** Your main performance knob |
| CFG w = 0, 1.5, 3 | Adherence vs diversity |
| Distilled (later) | Reflow / consistency distillation to N=1–2 if sampling time matters |

## 7. Evaluation (see step 11)

- Recall@10 of the *best of S samples* vs A5 (sampling should match or beat it)
- **Diversity:** mean pairwise distance between S samples for the same context (should be > 0; A5 has 0 by construction)
- **Transition realism:** the mean generated window per channel compared with the mean real window (step 07 §5 plot). Shapes should match
- Blind A/B: A6 vs A5 vs A0 on 10+ test songs

## 8. Verification

- [ ] **Toy test first:** train u_theta on a 2-D mixture of Gaussians (no AMV data). Samples must reproduce the separate modes. This catches 90 % of bugs in minutes
- [ ] Train loss decreases smoothly (FM loss doesn't go to 0; it plateaus, which is normal)
- [ ] Sampled windows look like real ones (plot 20 samples vs 20 real windows)
- [ ] Rollouts over a full song stay bounded
- [ ] Inference: 400 cuts × S=4 × N=8 batched, in **< 3 s** on CPU

## 9. Pitfalls

- **Unnormalised channels** (e.g. a cut-offset in seconds next to normalised features) → the model ignores small-scale channels.
- **Mixing τ conventions**: some papers run noise→data with τ: 0→1, others the reverse. Pick one (here: τ=0 noise, τ=1 data) and stick with it in both training and sampling.
- **Sequential sampling in Python loops over cuts:** it is much faster to batch across beams × samples. Over cuts it's inherently sequential (re-anchoring), but each step is one batched call.
- **Exposure bias:** train some epochs with conditioning built from the model's own rollouts (as in step 09 §5) once the base model works.

## 10. CPU budget

| | Estimate |
|---|---|
| Params | ~0.8M, ≈ 1.6 MFLOP per token per forward |
| Training: 90k samples × ~20 tokens ≈ 1.8M tokens per epoch, × 6 × params | ~9 TFLOP, **~2–5 min per epoch** |
| 150 epochs | ~5–12 h, overnight |
| Inference: 400 × S 4 × N 8 × 20 tokens × 1.6 MFLOP | ~0.4 TFLOP; in practice **~1–3 s** batched |

## 11. Done when

A6 trains, passes the toy test, beats or matches A5 on recall@10 (best of S), shows clearly higher diversity, wins the blind A/B against A0, and is selectable in `main.py` (`--dynamics a6 --fm-steps 8 --samples 4`).
