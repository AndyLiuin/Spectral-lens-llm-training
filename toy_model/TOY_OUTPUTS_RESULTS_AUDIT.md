# Toy Outputs Results Audit

This note audits the spectrum-analysis and feature-learning claims in
`TOY_OUTPUTS_RESULTS_ANALYSIS.md` against the current code and output artifacts.

## Executive status

Use the corrected four-stage analysis rooted at
`runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/`
for supervisor-facing discussion.

Do not present Section 8 of `TOY_OUTPUTS_RESULTS_ANALYSIS.md` as rigorous evidence.
Its transition-map construction mixed batch trajectories because the original
feature-learning tables did not carry `B` through the map key.

## What is safe to present

- The corrected four-stage chain `baseline -> rope -> muon -> untie_embed`
  supports the constant-loss performance story.
- The corrected feature-learning probe shows real early signal at checkpoint `500`,
  with `|rho(delta_H_peak, tok_gain)| = 0.8333` in
  `feature_early_prediction.csv`.
- The corrected map has no self-transitions and only contains the intended edges:
  `baseline->rope`, `rope->muon`, and `muon->untie_embed`.
- The strongest early feature shift is on `muon -> untie_embed`, especially in
  `delta_H_peak`.

## Main audit findings

### 1. Old Section 8 transition-map analysis is structurally invalid

The uncorrected feature-learning map at
`runs/modarith_a_seed0_steps5000_asha_20260331/concat_batch_regime_selected_runs/variant_concat_ablation/feature_learning_analysis/feature_variant_transition_map.csv`
contains many self-transitions such as `baseline->baseline` and `rope->rope`.

This happened because the original map builder grouped replay rows by
`(track, seed)` while the selected-run summary still contained three batch
trajectories under the same seed.

Consequence:
- Section 8 should be treated as exploratory only.
- Sections 9-10 are the presentation-safe replacements.

### 2. Fixed-sample spectrum comparisons still had a run-specific sampling floor

In the training runner, the spectral measurement subset was keyed by run name.
That means two runs on the same cached dataset used different measurement
examples even when `fixed_samples=True`.

Evidence:
- Step-0 JS divergence between the three baseline runs is already nonzero:
  about `2.6e-4`, `3.6e-4`, and `4.7e-4`.
- Since the models are identical at step `0`, this is a measurement-subset
  floor, not a learned geometry difference.

Consequence:
- Matched-loss JS divergences are still informative when they are much larger
  than this floor.
- Smaller JS effects should be discussed relative to that floor.

### 3. “Target checkpoint” in the feature probe means nearest-to-target, not guaranteed hit

The probe checkpoint selector chooses the checkpoint whose monitored loss is
closest to `target_loss`; it does not require `loss <= target_loss`.

Evidence:
- In the corrected probe manifest, baseline runs are tagged with
  `checkpoint_selection_sources=final,manual,target` at step `5000` even though
  baseline never reaches validation loss `<= 2.0`.

Consequence:
- In presentation, say “closest-to-target checkpoint” rather than “target
  checkpoint”.
- Do not interpret a `target` source tag as proof that the run hit the target.

### 4. Early-vs-endpoint predictor comparisons were only partly apples-to-apples

The reported endpoint baseline in `feature_early_prediction.csv` is a single
global correlation from all transitions, not the endpoint correlation on the
same subset of transitions available at each checkpoint.

Consequence:
- The strongest corrected claim is the checkpoint-`500` result.
- The statements that the early signal stays above the endpoint baseline at
  checkpoints `1000` and `1500` should be weakened.

## Supervisor-safe wording

Use wording like this:

"In the corrected four-stage toy sweep, the trusted chain
`baseline -> rope -> muon -> untie_embed` shows monotone improvement in
constant-loss performance. The accompanying feature-learning probe shows that
early spectral concentration metrics, especially `delta_H_peak` at checkpoint
`500`, carry predictive signal about later transition quality. We treat the old
multi-stage Section 8 analysis as exploratory because its original transition
map mixed batch trajectories, and we treat matched-loss JS values relative to a
small but nonzero step-0 sampling floor."

## Code fixes applied

To prevent these issues in future reruns:

- `runner.py` now keys fixed-sample spectral measurements by cached dataset
  identity instead of full run name.
- `run_feature_learning_probe.py` now writes `B` into the probe outputs.
- `analyze_feature_learning_variant_map.py` now uses `B` as part of the
  trajectory key when available, so batch trajectories do not silently mix.
