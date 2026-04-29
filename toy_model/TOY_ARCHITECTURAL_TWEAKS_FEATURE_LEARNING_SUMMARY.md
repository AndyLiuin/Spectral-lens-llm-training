# Architectural Tweaks: Spectrum + Feature-Learning Interpretation

## Scope

This note focuses on **token-based** and **spectrum / feature-learning** evidence.

I am **not** using wall-clock training time as primary evidence here, because GPU sharing can distort those measurements.

The evidence sources are:

- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation/variant_concat_ablation_summary.csv`
- per-run `metrics_over_time.csv` and saved activation spectra under the same selected-run directory
- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/feature_learning_analysis/*`

## Main conclusion

The tweaks separate into three qualitatively different classes:

1. `rope` gives a **small structural feature-learning assist**.
2. `muon` gives a **large optimization / token-efficiency gain**, but only a modest and inconsistent feature-learning shift.
3. `untie_embed` is the first tweak that produces a **large qualitative change in feature formation itself**.

Later tweaks after `untie_embed` are mostly **incremental or neutral** in token/spectrum terms, and there is not yet matching feature-probe evidence for them.

## A. Consecutive-transition spectrum/token evidence

Below, `loss_proximal_tok_gain_mean` is the mean token-efficiency gain between consecutive stages at matched loss levels. Positive means the newer architecture needs fewer tokens at the same loss. `loss_proximal_js_div_mean` is the matched-loss activation-spectrum JS divergence between the two consecutive stages.

| transition | loss-proximal token gain | matched-loss JS | interpretation |
|---|---:|---:|---|
| `baseline -> rope` | `0.578` | `0.0063` | modest improvement, small spectral change |
| `rope -> muon` | `2.476` | `0.0176` | very large efficiency gain, but only moderate spectral change |
| `muon -> untie_embed` | `1.898` | `0.1617` | large efficiency gain and by far the largest spectral shift |
| `untie_embed -> value_mix` | `0.209` | `0.0035` | small incremental change |
| `value_mix -> unet` | `-0.064` | `0.0102` | slightly worse token efficiency on average |
| `unet -> fixed_window` | `-0.207` | `0.0065` | slightly worse token efficiency on average |
| `fixed_window -> attn_scale` | `0.102` | `0.0015` | near-null change |

Interpretation:

- `rope` changes the model a little and helps a little.
- `muon` changes the model behavior a lot in optimization terms, but not in a way that strongly rewrites the learned feature spectrum.
- `untie_embed` is different: it combines strong token gains with a **much larger matched-loss spectral displacement** than any other transition.
- After `untie_embed`, the later tweaks do not show another major spectral regime change.

## B. Core4 feature-learning evidence

For feature learning, the clearest reliable evidence comes from the corrected `core4_bybatch` probe artifacts, normalized by batch trajectory.

### Stage-level feature concentration

Using matched-band probes:

- `baseline` `H_peak` rises slowly: `0.0219` at checkpoint `500`, `0.0357` at `1000`, `0.0531` at `1500`
- `rope` is consistently above baseline but only modestly: `0.0299` at `500`, `0.0449` at `1000`, `0.0714` at `1500`
- `muon` is also above baseline and usually above rope early: `0.0399` at `500`, `0.0649` at `1000`, `0.0662` at `1500`
- `untie_embed` is in a different regime entirely: `0.1324` at `500`, `0.1880` at `1000`, `0.1487` at `1500`

The same pattern appears in PC1 peak mass:

- `baseline`: `0.0505` at `500`, `0.0574` at `1000`, `0.1035` at `1500`
- `rope`: `0.0664` at `500`, `0.0838` at `1000`, `0.1467` at `1500`
- `muon`: `0.0819` at `500`, `0.1026` at `1000`, `0.1255` at `1500`
- `untie_embed`: `0.2290` at `500`, `0.3136` at `1000`, `0.3398` at `1500`

Interpretation:

- `rope` slightly sharpens feature concentration.
- `muon` also sharpens concentration, but not explosively.
- `untie_embed` causes a **step-change** in concentration/localization of the learned representation.

### Transition-level feature deltas

Average transition deltas from the probe map:

- `baseline -> rope`:
  - `delta_H_peak_mean = 0.0080` at checkpoint `500`
  - `delta_H_peak_mean = 0.0093` at `1000`
  - `delta_H_peak_mean = 0.0183` at `1500`
- `rope -> muon`:
  - `delta_H_peak_mean = 0.0100` at checkpoint `500`
  - `delta_H_peak_mean = 0.0200` at `1000`
  - `delta_H_peak_mean = 0.0386` at `1500`
- `muon -> untie_embed`:
  - `delta_H_peak_mean = 0.1065` at checkpoint `500`
  - `delta_H_peak_mean = 0.1655` at `1000`
  - `delta_H_peak_mean = 0.1222` at `1500`

The PCA-side jump is similarly concentrated in `muon -> untie_embed`:

- `delta_pca_mean = 0.1751` at `500`
- `delta_pca_mean = 0.2520` at `1000`
- `delta_pca_mean = 0.2833` at `1500`

Interpretation:

- `rope` and `muon` each move feature concentration a bit.
- `untie_embed` is the only transition that produces a **large, immediate, and persistent jump** in the feature-spectrum concentration metrics.

## C. What each tweak is doing

### `rope`

Best reading:

- weak-to-moderate feature-learning improvement
- small matched-loss spectral displacement
- modest token-efficiency improvement

So `rope` looks like a **helpful architectural prior**, not a regime change.

### `muon`

Best reading:

- very large matched-loss token-efficiency gain
- only moderate matched-loss spectral displacement
- feature concentration improves, but not at the same scale as the optimization gain

So `muon` looks primarily like an **optimization accelerator**, not a feature-learning regime shift.

### `untie_embed`

Best reading:

- large matched-loss token-efficiency gain
- the largest matched-loss spectral displacement by far
- huge jump in `H_peak` and PC1 concentration very early

So `untie_embed` is the clearest **feature-learning architecture change** in the toy study.

### `value_mix`, `unet`, `fixed_window`, `attn_scale`

What the current evidence supports:

- they do **not** produce another large spectral transition after `untie_embed`
- their token gains are small, mixed, or near-null on average
- we do **not** yet have matching feature-probe artifacts for these later stages, so they should not be described as feature-learning improvements without further evidence

## D. Supervisor-safe bottom line

If the question is:

“Which tweak actually changes feature learning, rather than just making optimization easier?”

the strongest answer from the current toy evidence is:

- `rope`: small feature-learning assist
- `muon`: mostly optimization / efficiency
- `untie_embed`: real feature-learning regime change

And for the later tweaks:

- current evidence says they are **fine-tuning tweaks**, not another major change in the feature-learning mechanism

## Caveats

- The feature-probe evidence is strongest only for the corrected core4 chain.
- The study is still effectively one underlying random seed.
- Causal asymmetry metrics (`drop_advantage`) are noisier and less monotone than `H_peak` / PCA concentration, so the cleanest architectural interpretation should lean on concentration/localization plus matched-loss token/spectrum comparisons.
