# Step 05: Retrieval (Index, Scoring, Beam Search)

**Goal:** given a predicted state `ŝ_k` (and context), pick the **best real clip** from the library, avoiding repetition and keeping the edit coherent. Every dynamics model (A0–A6, B) shares this component.

← [04 Clip features](../04-clip-features/README.md) · [Project plan](../README.md) · Next → [06 Physics baseline & renderer](../06-physics-baseline-and-renderer/README.md)

---

## 1. Concepts

- **Exact search is enough.** 15k clips × 64 dims ≈ 1M multiply-adds per query, microseconds with numpy. ANN (HNSW) only pays off at ~1M+ clips. Start with a single numpy matrix multiplication; FAISS `IndexFlatIP` is optional.
- **Nearest neighbour alone makes bad edits:** it repeats the same clip, picks near-duplicates of the previous shot, or ignores the slot length. So retrieval = **candidate generation** (cheap, top-K) + **scoring** (rich, on K candidates) + **beam search** over the sequence.
- **Re-anchoring:** after choosing a clip, the next step starts from that clip's **real** state, not the prediction. This keeps the trajectory on real footage and prevents drift.

## 2. Interfaces

```text
ClipIndex
  .z_sem [N,64], .z_dyn [N,D], .meta (video_id, shot_idx, t_in, t_out, duration, episode)
  .candidates(ŝ, k=200, mask=None) -> ids[k], base_scores[k]

score(candidate, ŝ_k, history, slot) -> float
beam_search(predict_fn, slots, beam=4..8, k=200) -> list[clip_id per slot]
```

`slot` = (start, end) time and required duration from the cut timing (step 06/08).
`predict_fn(history_states, slot, music_k) -> ŝ_k` is supplied by the dynamics model.

## 3. Scoring function

```text
score = + w_sem · cos(ẑ_sem, z_sem_c)
        − w_dyn · ‖ẑ_dyn − z_dyn_c‖₂ (on normalised features)
        − w_rep · repeat_penalty(c, history)       # same clip_id / same shot recently
        − w_adj · adjacency_penalty(c, prev)       # cosine too high to the previous clip = "no visible cut"
        + w_cont · continuity_bonus(c, history)    # same episode/arc as recent picks
        − w_len · length_penalty(c, slot)          # clip shorter than slot (needs slow-mo / loop)
```

Starting weights: `w_sem=1, w_dyn=1, w_rep=5, w_adj=2, w_cont=0.2, w_len=1`. Tune them by watching outputs; step 11 adds metrics.

Details:
- **repeat_penalty:** decays with distance in shots, e.g. `exp(−Δk / 20)` for the same shot (any overlapping window of the same source shot counts), plus a hard block for the exact same clip within 30 shots.
- **adjacency_penalty:** `max(0, cos(z_sem_c, z_sem_prev) − 0.9)`. A cut between nearly identical frames looks like a glitch.
- **length_penalty:** if `clip.duration < slot.duration / max_slowdown`, the clip can't fill the slot even with slow motion, so penalise heavily or mask it out.

## 4. Beam search

- State of a beam = (list of chosen clip ids, cumulative score, current dynamics state `v`, `a`).
- For each slot: for every beam, call `predict_fn`, get top-K candidates, score them, and expand into the top `beam` children. Keep the global top `beam`.
- `beam=1` = greedy. Compare greedy vs beam 4/8 in step 11: the gain usually shows up as fewer repetitions and smoother intensity.
- Vectorise the scoring over K candidates. Never loop in Python over K × beam.

## 5. Optional extras (later)

- **Masks/filters:** only certain series, characters (via CLIP text query in step 13), or exclude OP/ED clips.
- **Diversity:** small temperature-based sampling among the top-3 instead of argmax, for multiple variants of one song.

## 6. Verification

- [ ] Unit test: querying with the exact state of clip X returns X first (when no penalties apply)
- [ ] Repetition test: a 400-slot greedy run with a constant query state never repeats a clip within 30 slots
- [ ] Timing: 400 slots × beam 8 × K 200 completes in < 1 s on CPU
- [ ] Visual: render 20 consecutive picks as a contact sheet. The intensity trend should follow the query trend

## 7. Pitfalls

- Mixing raw CLIP (512) and PCA (64) spaces between candidates and queries. Use one space per comparison.
- Unnormalised `z_dyn` makes one feature (e.g. duration) dominate the distance.
- Scores at very different scales: print the mean/std of each term on a real run, then balance the weights.

## 8. Done when

`ClipIndex` + `score` + `beam_search` work with a dummy `predict_fn` (e.g. "follow a hand-drawn intensity curve") and pass the tests above.
