# Step 13: Controls & UI

**Goal:** make the generator *steerable* and usable. Interpretable control knobs come from the explicit dynamics (the advantage over an opaque end-to-end model), plus a simple Gradio app for generating, previewing and exporting.

← [12 Performance](../12-performance/README.md) · [Project plan](../README.md)

---

## 1. Controls (how each one hooks into the system)

| Control | Mechanism | Hook |
|---|---|---|
| **Aggression** | scale music dynamics before conditioning: `Δ²m ← α · Δ²m`, `Δm ← α · Δm`; also scale the spring stiffness κ | 06, 09, 10 |
| **Cut density** | shift the `p_cut` threshold or use a target cuts per minute (pick the threshold that hits it) | 08 |
| **Smoothness** | increase damping γ; lower the FM sampling temperature (scale the initial noise `x0` by σ < 1) | 09, 10 |
| **Variety** | higher repetition penalty, larger beam with sampling, CFG weight down | 05, 10 |
| **Style** | genre/style embedding (hardstyle / sad / flow) in the cut model and the FM conditioning | 08, 10 |
| **Section plan override** | user edits section labels (intro/build/drop) on a timeline | 02 → 06–10 |
| **Content filter** | CLIP text prompt ("Isagi", "rain", "goal celebration") → cosine on `clip_raw` → boost or mask candidates | 05 |
| **Pinned clips** | user forces a specific clip at a time (e.g. on the drop); beam search treats it as a fixed slot | 05 |
| **Seed** | reproducible variants | all |

Each control needs a check that it actually does what it claims: e.g. aggression α=2 should raise the music–motion correlation slope and the average `motion_peak` (step 11 metrics).

## 2. Gradio app

Layout:
1. **Inputs:** song upload, library folder selector (only preprocessed libraries listed), dynamics model dropdown (A0/A5/A6), controls (sliders/dropdowns from §1), seed.
2. **Analyse:** shows the music plot (energy, Δ², beats, sections) with editable section labels.
3. **Generate:** runs the pipeline → shows a timeline strip (thumbnails per segment, coloured by intensity) + a **preview video** (360p proxy render).
4. **Tweak:** re-roll a single segment (resample that slot only), pin a clip, change the controls, regenerate from slot k onward.
5. **Export:** 1080p render + `timeline.json` download.

Also a separate **A/B page** for the human evaluation in step 11 (two videos, buttons, results appended to a CSV).

## 3. Optional exports

- Export `timeline.json` to an editor format (e.g. FCPXML or an EDL) so a human can finish the edit in DaVinci Resolve or Premiere. Very useful in practice: the AI does the rough cut, you polish it.

## 4. Verification

- [ ] Every control changes the output in the documented direction (checked with step 11 metrics, not only by eye)
- [ ] Re-rolling one segment doesn't change the others
- [ ] End-to-end on CPU: song upload → preview in < 1 min (preprocessed library)

## 5. Done when

A non-technical friend can generate, tweak and export an AMV from the app without help. **M7 reached.**
