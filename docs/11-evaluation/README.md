# Step 11: Evaluation

**Goal:** a fixed, reproducible evaluation suite that every model (A0–A6, B) runs through, producing comparable tables, plots and blind human ratings. Build it **early**: the parts for step 06 can exist as soon as M2 is reached.

← [10 Flow-matching expert](../10-flow-matching-expert/README.md) · [Project plan](../README.md) · Next → [12 Performance](../12-performance/README.md)

---

## 1. Principles

- **Frozen test set:** test AMVs/songs chosen once, never used for tuning. Use the val split for all decisions.
- **Same components:** when comparing dynamics models, the cut times, retrieval weights, library and renderer are identical.
- **Seeds:** run stochastic models (A6, beam sampling) with 3 seeds and report mean ± std.
- **One command:** `python -m amv.eval.run --model a5 --split test` writes `results/<model>_<date>.json`.

## 2. Automatic metrics

### 2.1 Timing (step 08)
| Metric | Definition |
|---|---|
| Cut F1 @ ±70 ms | precision/recall of predicted vs real cuts on test AMVs |
| Cuts per minute error | absolute difference |
| Beat alignment | % of predicted cuts within ±40 ms of a beat |

### 2.2 Next-clip prediction (steps 09, 10)
| Metric | Definition |
|---|---|
| Recall@1/10/50 | teacher-forced: the true next shot's state is added to that anime's library clips; rank by the retrieval score |
| Mean rank | mean rank of the true next shot |
| Rollout drift | ‖ŝ − s‖ after 8/16/32 free-running steps (no re-anchoring to the truth) |

Recall needs the true AMV shot to be in the candidate pool. Insert it into the pool for this metric, which is standard practice.

### 2.3 Edit quality (on generated AMVs for test songs)
| Metric | Definition | Desired |
|---|---|---|
| Music–motion correlation | Pearson(`motion_mean` of chosen clips, `rms_s` per slot) | high (> 0.5) |
| Drop response | mean `motion_peak` in the first 2 bars after a drop − mean in the 2 bars before | clearly > 0 |
| Repetition rate | % slots whose source shot was used in the previous 30 slots | low |
| Semantic coherence | mean cos(`z_sem`) of consecutive clips | medium: too high = monotone, too low = random |
| Diversity (A6) | mean pairwise distance between S samples' chosen clip sequences | > 0 |
| Smoothness | mean ‖Δ²`z_dyn`‖ over the chosen sequence | moderate |
| Real-AMV statistics match | distance between the generated and real distributions of shot length and transition window shape | low |

### 2.4 Cost
Params, train time per epoch, inference time per song (model only, and end-to-end incl. render), peak RAM.

## 3. Human evaluation (the metric that matters most)

- 10–20 test songs × 3 systems (e.g. A0, A5, A6), 30–45 s excerpts around a drop.
- Blind pairwise comparison: randomise order and labels. Raters answer "Which edit fits the music better?" plus a 1–5 rating for sync, intensity, coherence and "would watch".
- Raters: you plus 3–5 friends. Even small samples show clear winners.
- Analyse with win rates and a simple Bradley–Terry model or a sign test.

A tiny Gradio page (step 13) can serve the A/B pairs and log answers to a CSV.

## 4. Reports

- `results/ablation_table.md`: rows = models, columns = the metrics above.
- `results/data_efficiency.png`: recall@10 vs training-data fraction (10/25/50/100 %) for A2, A3, A5, A6, B. **Main research figure.**
- `results/fm_steps.png`: quality vs N sampling steps vs time for A6.
- Per-song diagnostic plots: music energy, chosen clips' motion, cuts.

## 5. Verification

- [ ] Running the eval twice with the same seed gives identical numbers
- [ ] A0 metrics computed and stored as the reference row
- [ ] A "random" baseline row (random clips on heuristic cuts) exists. Every model must beat it
- [ ] An "oracle" row (true AMV shots) exists where applicable. It shows the upper bound

## 6. Pitfalls

- Optimising automatic metrics that don't match human judgement. Check correlation with the human ratings once you have both.
- Comparing models trained on different data splits or feature versions. Store the dataset version in each result file.

## 7. Done when

The ablation table, data-efficiency plot and FM-steps plot exist, and one round of blind A/B is complete with results written into `results/`.
