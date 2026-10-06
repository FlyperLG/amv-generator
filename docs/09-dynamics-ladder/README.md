# Step 09: Dynamics Ladder (A1–A5, Baseline B)

**Goal:** learn **what kind of clip comes next**, given the edit so far and the music. Build it as a ladder of increasingly capable models that all share the same physics, retrieval and renderer, so you can measure exactly what each piece of structure buys.

← [08 Cut timing model](../08-cut-timing-model/README.md) · [Project plan](../README.md) · Next → [10 Flow-matching expert](../10-flow-matching-expert/README.md)

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| k | shot index (one step = one shot) |
| `s_k = [z_sem_k, z_dyn_k]` | state of shot k (64 + D) |
| `v_k = s_k − s_{k−1}` | edit velocity (per shot) |
| `a_k = v_k − v_{k−1}` | edit acceleration |
| `m_k` | music features for slot k: pooled beat features, `Δm`, `Δ²m`, section, slot duration |
| H | context length in shots (8–16) |

Optional: divide by slot duration to get velocity per second instead of per shot. Try both; per-shot is simpler, while per-second handles varying shot lengths better.

## 2. Shared physics core (the damped integrator)

```text
v_{k+1} = γ ⊙ v_k + â_k
ŝ_{k+1} = s_k + v_{k+1}
```
- γ is a vector (one damping per state dim) in (0,1). Learn it as `sigmoid(param)`, or fix it.
- **Predict acceleration, never jerk.** `a_{k+1} = a_k + f(...)` integrates errors three times and blows up over long horizons.
- `z_sem` dims: use stronger damping (γ ≈ 0.5) since semantics shouldn't build momentum the way intensity does.

## 3. The ladder

| ID | `â_k` comes from | Params | What it tests |
|---|---|---|---|
| A1 | `â = 0`, γ = 1 (constant velocity) | 0 | Is naive extrapolation useful at all? |
| A2 | `â = W · [Δm_k, Δ²m_k]` (linear in music dynamics), learned γ | ~100 | Does music acceleration explain video acceleration? |
| A3 | `â = A2 + MLP(s_k, v_k, a_k, m_k)` | ~0.3M | Non-linear correction, no temporal context |
| A4 | `â = A2 + GRU(last H steps of [s, v, a, m])` | ~1M | Temporal context, recurrent |
| A5 | `â = A2 + Transformer(last H steps + music window ±8 beats)` | ~2–3M | Temporal context, attention |
| B | Full transformer: all music beat tokens + previous shot tokens → `ŝ_{k+1}` directly, **no physics** | ~5M (CPU-sized) | Does the structure help vs. learning everything? |

The A3–A5 corrections are **residuals** on top of A2. Initialise their last layer to zero so training starts from A2's behaviour; this makes training much more stable.

### A5 architecture details
- Tokens: H previous shots, each `Linear([s, v, a, m]) → d=128`, plus 16 music tokens (beats around the next slot), plus 1 query token for the next slot.
- 3–4 layers, 4 heads, d=128, FFN 512, pre-norm, dropout 0.1, learned positional embeddings (or relative position in beats).
- Output: the query token → `Linear → â` (64 + D dims).
- Note: this is a single sequence with self-attention over everything, so no separate cross-attention block is needed. With ~33 tokens, the cost is negligible.

### B architecture details
- Sequence: all beat tokens of a ±32-beat window + previous H shot tokens + query; 4 layers, d=192.
- Output `ŝ_{k+1}` directly. Same loss and retrieval as the others.

## 4. Losses

Retrieval is ranking, so train for ranking:

1. **InfoNCE (main):**
   `L_nce = −log( exp(sim(ŝ, s⁺)/τ) / Σ_j exp(sim(ŝ, s_j)/τ) )`
   - `s⁺` = the true next shot's state.
   - Negatives `s_j`: 255 random **library** clips + other shots from the same AMV (hard negatives).
   - `sim` = `cos` on `z_sem` + `−‖·‖²` on normalised `z_dyn` (the same terms as the retrieval score), τ ≈ 0.07–0.1.
2. **MSE on `z_dyn`** (weight 0.5): keeps the intensity prediction calibrated.
3. **Smoothness** (weight 0.01–0.1): `‖â_k‖²` and `‖â_k − â_{k−1}‖²` (penalises jerk).

Note on negatives: the *true* next shot comes from the AMV (with effects), while the library clips are clean. That domain gap makes it easy to tell them apart for the wrong reason. Mitigate it by using per-AMV normalised `z_dyn` (step 07) and by mixing in negatives from **other AMVs**.

## 5. Training curriculum

1. **Teacher forcing:** context = true previous shots; predict 1 step. Train until val NCE plateaus.
2. **Multi-step rollout (horizon 4 → 8):** unroll the physics with the model's own predictions for h steps. At each step, re-anchor to the true state with probability p (scheduled sampling, p: 1.0 → 0.5), and sum the losses over steps. This teaches the model to recover from its own errors.
3. Optional: **retrieval in the loop** at validation time (snap to the nearest library clip each step), which is exactly what inference does.

Optimiser: AdamW, lr 3e-4, wd 0.01, batch 64 sequences, gradient clipping 1.0, 50–100 epochs, early stopping on val recall@10.

## 6. Data-efficiency experiment (the headline result)

Train A2, A3, A5 and B on **10 / 25 / 50 / 100 %** of the training AMVs (same val/test). Plot val recall@10 against the data fraction. Hypothesis: A5 ≥ B at low data, B catches up at high data. Whatever the outcome, the plot is the result.

## 7. Verification

- [ ] A1 and A2 run with zero training code bugs; A2's learned W has interpretable signs (rising energy → rising motion)
- [ ] Teacher-forced recall@10 (true next shot among same-anime clips) improves A2 → A3 → A5
- [ ] Rollouts of 32 steps stay bounded (plot `‖v‖` over k; it should not explode)
- [ ] Rendered test songs: A5 vs A0 side by side (blind if possible)
- [ ] Inference: 400 slots × beam 4 in < 3 s on CPU (batch the beam dimension through the model)

## 8. Pitfalls

- **Predicting `s` instead of `a` in A3–A5** quietly discards the physics prior. Keep the residual-on-A2 form.
- **Training on library states while testing on AMV states** (or the reverse) without normalisation = domain gap.
- **Collapse to the mean:** if all predictions converge to the average state, the model is averaging several valid futures. That's the multimodality problem, and it's exactly why step 10 exists. Measure the prediction variance across contexts to detect it.

## 9. CPU budget

| Model | Per-epoch (90k transitions) |
|---|---|
| A2 | < 1 s |
| A3 | ~5–15 s |
| A4 | ~30–90 s |
| A5 | ~1–4 min |
| B (5M) | ~5–15 min |

With `torch.set_num_threads(physical cores)` and batch 64–128. Rollout training costs ~h× more per epoch.

## 10. Done when

The full ladder is trained, the recall@k table and the data-efficiency plot exist (step 11), and the best deterministic model (probably A5) is wired into `main.py` (`--dynamics a0|a2|a5|b`).
