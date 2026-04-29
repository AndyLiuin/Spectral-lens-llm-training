# Toy Outputs Results Analysis

This report is based on artifacts in `toy_model/toy_outputs` as of this run.

## Data Sources Used

- `toy_outputs/variant_ablation/variant_ablation_summary.csv`
- `toy_outputs/variant_ablation/variant_ablation_matched_pairs.csv`
- `toy_outputs/variant_ablation/variant_ablation_matched_aggregate.csv`
- `toy_outputs/noise_scale_ablation/noise_scale_matched_loss.csv`
- `toy_outputs/noise_scale_ablation/noise_scale_comparison.csv`
- `toy_outputs/phase_ablation/phase_summary.csv`
- `toy_outputs/phase_ablation/phase_trajectories.csv`
- `toy_outputs/scaling_link_ablation/scaling_fit.csv`
- `toy_outputs/scaling_link_ablation/scaling_correlation.json`
- `toy_outputs/distribution_ablation/distribution_ablation_summary.csv`
- `toy_outputs/distribution_ablation/distribution_ablation_aggregate.csv`
- `toy_outputs/estimator_ablation/estimator_ablation_summary.csv`

## Important Reliability Caveat

Most ablations here appear to be single-seed snapshots in this output dump:
- `variant_ablation_summary.csv`: seed = 2 only, track `a` only.
- `distribution_ablation_summary.csv`: seed = 0 only.
- `scaling_runs.csv`: seed = 2 only.

So interpretation below is best treated as **directional evidence**, not final robust effect-size estimates.

---

## 1) Variant Ablation (Track A)

## Endpoint test loss ranking
From `variant_ablation_summary.csv` (lower is better):

1. `attn_scale`: 0.000463
2. `baseline`: 0.000469
3. `rope`: 0.000489
4. `unet`: 0.000509
5. `fixed_window`: 0.000509
6. `muon`: 0.000552

Interpretation:
- In this run, `attn_scale` is the only variant that slightly outperforms baseline on final test loss.
- `muon` underperforms baseline on this setup.

## Matched-loss geometry distance to baseline
From `variant_ablation_matched_aggregate.csv` (`js_div_mean`, lower means closer geometry to baseline):

1. `attn_scale`: 0.000396
2. `muon`: 0.000482
3. `fixed_window`: 0.000714
4. `rope`: 0.001322
5. `unet`: 0.002213

Interpretation:
- `unet` and `rope` are the largest geometric departures from baseline at matched loss.
- `attn_scale` is both best endpoint loss and smallest matched-loss geometry drift.
- `muon` has moderate matched-loss geometry drift but worse final loss, suggesting optimization dynamics differences dominate over pure geometry drift in this setting.

---

## 2) Noise-Scale Ablation

From `noise_scale_matched_loss.csv` and `noise_scale_comparison.csv`:

- Track A mean JS divergence:
  - `unconstrained`: 0.000271
  - `fixed_noise`: 0.051081
  - delta (`fixed - unconstrained`): +0.050811

- Track B mean JS divergence:
  - `unconstrained`: 0.000029
  - `fixed_noise`: 0.000025
  - delta: -0.000004

Interpretation:
- For Track A, your current “fixed noise” recipe (`lr ∝ B`) produced **much larger** spectral divergence at matched loss than unconstrained.
- For Track B, both regimes are essentially invariant.
- This means the specific noise-preserving assumption used here is not stabilizing Track A geometry in this setting; it may over-correct or alter optimization trajectory.

Diagnostic note:
- Matched-loss errors are small (`mean_match_error` around 1e-6 to 1e-5), so this is unlikely to be a bad matching artifact.

---

## 3) Phase Ablation

From `phase_summary.csv`:

- Track A, `phase_like` vs `muted`:
  - `rankme_delta`: -0.7906 vs +2.2333
  - `rankme_sign_changes`: 8 vs 2
  - `alpha_tail_delta`: -0.4800 vs -0.1023

- Track B:
  - `rankme_delta`: 0 in both regimes
  - `rankme_sign_changes`: 0 in both regimes
  - `alpha_tail_delta`: 0 in both regimes

Interpretation:
- Track A shows clear phase-like non-monotonicity and stronger geometry shift when the transition is introduced.
- Track B is effectively phase-inert under the same intervention.
- This supports the intended narrative: phase-like transitions are architecture/dynamics dependent, not a generic property of the data generator.

---

## 4) Scaling-Link Ablation

From `scaling_fit.csv` and `scaling_correlation.json`:

- Overall correlation between scaling exponent `s` and `alpha_proxy`: `pearson_r = -0.1848`
- Bootstrap CI: `[-0.3977, 0.0512]` (crosses zero)

Track-conditional view (computed from `scaling_fit.csv`):
- Track A: corr(`s`, `alpha_proxy`) = +0.4181 (n=36)
- Track B: corr(`s`, `alpha_proxy`) = -0.6064 (n=36)

Interpretation:
- Pooled correlation is weak/negative because Track A and Track B show opposite signs.
- The relationship is **not universal across tracks** in this snapshot.
- You should analyze scaling-link per track (or with interaction terms), not only pooled.

---

## 5) Distribution Ablation

From `distribution_ablation_aggregate.csv`:

## Track-level means
- Track A:
  - `alpha_tail_mean`: 3.4734 ± 0.0207
  - `rankme_mean`: 48.19 ± 1.65
  - `loss_mean`: 0.000490 ± 0.000017

- Track B:
  - `alpha_tail_mean`: 0.8103 ± 0.1005
  - `rankme_mean`: 270.83 ± 50.04
  - `loss_mean`: 0.000443 ± 0.000011

Interpretation:
- Track A has much steeper tail exponents and much lower effective rank than Track B.
- Track B representations are broader (higher RankMe), consistent with linear pooled RFF behavior.

## Best/worst conditions by loss
- Track A:
  - best: `gaussian + powerlaw(gamma=1.2)` -> 0.000471
  - worst: `student_t + powerlaw(gamma=1.2)` -> 0.000522

- Track B:
  - best: `gaussian + powerlaw(gamma=1.2)` -> 0.000428
  - worst: `uniform + isotropic` -> 0.000459

Interpretation:
- Heavy-tail + strong anisotropy hurts Track A more in this run.
- Track B is comparatively stable, with smaller absolute degradation range.

---

## 6) Estimator Ablation

From `estimator_ablation_summary.csv`:

- Strong dependence of `alpha_tail` on `d_over_n`:
  - Track A corr(`d_over_n`, `alpha_tail`) ≈ 0.922 (fixed and nonfixed)
  - Track B corr ≈ 0.477

- Fixed vs non-fixed sampling gap is small:
  - Track A mean `|alpha_fixed - alpha_nonfixed|` ≈ 0.027
  - Track B mean `|alpha_fixed - alpha_nonfixed|` ≈ 0.008

Interpretation:
- Finite-sample ratio `d/n` is a major driver of measured `alpha_tail`, especially for Track A.
- The estimator appears more sensitive to sample size than to fixed/non-fixed sampling mode in this setup.

---

## Cross-Ablation Synthesis

1. Track A geometry is highly intervention-sensitive (phase shifts, variant effects, noise-regime changes).
2. Track B acts as a stable control in multiple ablations (especially phase and noise-scale).
3. Matched-loss comparisons are crucial; endpoint-only comparisons can hide large geometry differences.
4. Estimator/sample-ratio effects are nontrivial and must be controlled when interpreting `alpha` metrics.
5. Scaling-link evidence is track-dependent and currently not robust as a pooled law.

---

## Recommended Next Runs (to firm up conclusions)

1. Re-run variant/distribution/scaling with 3+ seeds and report mean±std.
2. For noise-scale, test alternative scaling rules (e.g., smaller LR exponent than linear in `B`).
3. For scaling-link, fit separate per-track models and include confidence intervals per track.
4. For estimator ablation, pin a fixed `d/n` target grid when comparing across tracks.

---

## 7) Mod-Arith Constant-Loss Sweep (`runs/modarith_a_seed0_steps5000_asha_20260331`)

Files used for this section:
- `concat_batch_regime_ablation/concat_batch_regime_selected_lrs.csv`
- `concat_batch_regime_ablation/constant_loss_stage_summary.csv`
- `concat_batch_regime_ablation/concat_batch_regime_early_prediction.csv`
- `concat_batch_regime_ablation/concat_batch_regime_matched_pairs.csv`
- `concat_batch_regime_selected_runs/variant_concat_ablation/variant_concat_ablation_summary.csv`

This run is a **single-seed** ASHA batch/LR sweep on `task=mod_arith_lm` with cumulative variant stages
`baseline -> rope -> muon -> untie_embed -> value_mix -> unet -> fixed_window -> attn_scale`,
followed by replay of the selected runs with checkpoints and spectra.

## High-level verdict

This toy run supports some of the paper's main qualitative claims, but not all, and not yet at paper-ready strength.

- Strongly/directionally supported:
  - batch regime changes spectral state at matched loss
  - the efficient batch regime is architecture-dependent
  - early spectral metrics have predictive signal for within-variant token efficiency

- Not established by this run:
  - clean learning-side vs systems-side decomposition
  - the paper's random Fourier feature causal-sanity-check claim

## Claim-by-claim assessment

### Claim 1: Spectral instrumentation is a useful training diagnostic

Verdict: **supported**

Reasoning:
- The run produces nontrivial matched-loss spectral differences in `concat_batch_regime_matched_pairs.csv`, even when losses are aligned.
- It also produces nontrivial early-prediction signal in `concat_batch_regime_early_prediction.csv`, which is already enough to show that the spectral summaries carry information beyond endpoint loss alone.
- In the replay summary, `measurement_time_s` is often large relative to `train_time_s`, which reinforces that the instrumentation is materially doing work rather than being a cosmetic log.

Interpretation:
- As a toy diagnostic panel, this setup is already useful.
- As a strict paper result, it still needs more seeds and a cleaner target-hit bookkeeping path.

### Claim 2: Batch regime is a representation-geometry control variable

Verdict: **supported**

Reasoning:
- In every variant stage, the mean matched-loss activation-spectrum JS divergence is largest for the `(32, 512)` pair.
- Examples from `concat_batch_regime_matched_pairs.csv`:
  - `baseline`: mean JS `(32,512)=0.0162` vs `(32,128)=0.0055` and `(128,512)=0.0043`
  - `rope`: mean JS `(32,512)=0.0261` vs `(32,128)=0.0171` and `(128,512)=0.0087`
  - `untie_embed`: mean JS `(32,512)=0.0286` vs `(32,128)=0.0132` and `(128,512)=0.0105`
- This is exactly the pattern you would expect if increasing batch size is moving the model into systematically different spectral states rather than only changing speed.

Architecture-dependent efficient regimes also appear in the replay summary:
- `muon`: best token-efficient strict-hit batch is `B=128`
- `untie_embed`: best token-efficient strict-hit batch is `B=128`
- `value_mix`: best token-efficient strict-hit batch is `B=32`
- `unet`: best token-efficient strict-hit batch is `B=128`
- `fixed_window`: best token-efficient strict-hit batch is `B=32`
- `attn_scale`: best token-efficient strict-hit batch is `B=32`

Interpretation:
- The toy run clearly supports the narrower regime claim:
  batch size changes geometry at matched loss, and the best regime depends on the variant.

Caveat:
- `baseline` and `rope` do not cleanly reach the common `2.0` target across all three batches within `5000` steps, so those two stages are weaker evidence for the constant-loss part of the story.

### Claim 3: Spectral fingerprints separate learning-side vs systems-side gains

Verdict: **not really tested here**

Reasoning:
- This toy run is dominated by learning-side/optimization-side changes inside the model and optimizer.
- It does not include a clearly systems-only intervention analogous to the paper's throughput-side examples.
- Replay timing also includes substantial `measurement_time_s`, so any end-to-end wall-clock comparison in these instrumented runs would be confounded by logging overhead.

Interpretation:
- This toy run can support the learning-efficiency side of the paper.
- It does **not** independently justify the paper's systems-vs-learning decomposition claim.

### Claim 4: Early spectra predict relative token efficiency within a variant family

Verdict: **partially supported**

Scope:
- Finite early-prediction rows appear only for the later stages:
  `untie_embed`, `value_mix`, `unet`, `fixed_window`, `attn_scale`
- `baseline`, `rope`, and `muon` do not all provide clean finite within-variant token-ratio labels under this run configuration.

From `concat_batch_regime_early_prediction.csv`, mean absolute Spearman over the finite stages is:
- `act_grad_tail_gap`: `0.8065`
- `rankme`: `0.8065`
- `alpha_head`: `0.7732` with negative sign
- `alpha_tail`: `0.7065`
- `grad_alpha_tail`: `0.6821`
- `grad_alpha_head`: `0.6732`
- `grad_rankme`: `0.6488`
- `act_grad_head_gap`: `0.6244`

Checkpoint-wise:
- At `500`, the strongest average predictors are `alpha_head`, `rankme`, and `grad_rankme`, with `act_grad_tail_gap` and `alpha_tail` close behind.
- At `2500`, `act_grad_tail_gap` becomes the strongest average predictor.

Interpretation:
- The toy run supports the **narrow** claim that early spectra predict relative token efficiency within a fixed variant family.
- It does **not** yet isolate `alpha_tail` or activation-gradient mismatch as uniquely strongest, because `rankme` is just as strong in this single-seed run.
- It also does not test the paper's fraction-based checkpoints (`10%`, `25%`, `50%`) since this run only used absolute checkpoints (`500`, `1250`, `2500`).

### Claim 5: The toy model provides a causal sanity check under minimal assumptions

Verdict: **not justified by this specific run**

Reasoning:
- The paper text explicitly frames the causal sanity check as a **random Fourier feature** toy model.
- This run used `task=mod_arith_lm`, not `rff_regression`.
- So while the current mod-arithmetic sweep is still useful as a controlled toy benchmark, it is not the direct evidence for the RFF claim in the paper.

Interpretation:
- Do **not** cite this run alone as support for the paper's "random Fourier feature toy model" language.
- You need a corresponding `rff_regression` sweep for that claim.

## Important anomalies in the finished outputs

There is a real bookkeeping issue that should be stated explicitly.

There is also a more serious post-hoc code issue discovered after this run finished:

- from `muon` onward, the cumulative variants still use the Muon optimizer path
- before the fix in `runner.py`, `muon_notebook_lr_defaults=True` caused the notebook split LRs to ignore the swept base `lr`
- so the LR sweep for `muon`, `untie_embed`, `value_mix`, `unet`, `fixed_window`, and `attn_scale` was not a faithful LR sweep

Practical implication:
- `baseline` and `rope` rows remain interpretable
- comparisons from `muon` onward should be treated as provisional until rerun after the Muon LR fix
- any claim about monotone improvement or later-stage ranking from this exact run should therefore be downgraded accordingly

- In `concat_batch_regime_selected_lrs.csv`, two rows are flagged as hits even though the recorded validation loss is still above `2.0`:
  - `rope`, `B=512`, `final_val_loss=2.2138`
  - `unet`, `B=128`, `final_val_loss=2.3475`

- In the replay summary, one anomaly remains:
  - `rope`, `B=512`, `loss=2.2138`, `stop_reason=target_loss`

- The `unet`, `B=128` replay run is cleaner and ends at `loss=1.9367`, so that one appears to be a search-side inconsistency rather than a replay-side one.

Practical rule:
- For strict analysis, trust the numeric condition `loss <= target_loss` rather than `stop_reason` or `hit_target` alone.

## Additional useful readout from the selected LRs

From `muon` onward, the selected LRs are pinned to the bottom of the tested grid:
- `B=32 -> 1.5e-4`
- `B=128 -> 3.0e-4`
- `B=512 -> 6.0e-4`

Interpretation:
- The later variants are probably being selected on the **lower edge** of the current LR sweep.
- If you care about precise token-efficiency comparisons in those stages, the next run should extend the LR grid downward rather than upward.

## Bottom line for the paper

What this mod-arithmetic toy run **does** justify:
- matched-loss spectral states depend on batch regime
- the efficient batch regime depends on architecture/variant
- early spectral metrics have predictive utility for within-variant token efficiency

What it **only directionally** justifies:
- which early spectral metric is best
- how strong the predictive result is after controlling for checkpoint definition

What it **does not justify**:
- the systems-vs-learning decomposition claim
- the random Fourier feature toy-model claim

## Recommended next steps before using this as paper evidence

1. Re-run the same `mod_arith_lm` sweep with `3+` seeds.
2. Add fraction-based early checkpoints (`10%`, `25%`, `50%`) to match the paper protocol.
3. Extend the LR sweep downward for stages from `muon` onward.
4. Fix the target-hit bookkeeping so `stop_reason` and numeric loss agree.
5. Run the parallel `task=rff_regression` toy sweep before claiming the paper's RFF causal-sanity-check result.

## 8) Variant-by-variant monotonicity check and feature-learning readout

This section uses the completed replay summary in
`runs/modarith_a_seed0_steps5000_asha_20260331/concat_batch_regime_selected_runs/variant_concat_ablation/variant_concat_ablation_summary.csv`
and the probe outputs in
`runs/modarith_a_seed0_steps5000_asha_20260331/concat_batch_regime_selected_runs/variant_concat_ablation/feature_learning_analysis/`.

### Is each cumulative variant addition strictly better?

Verdict: **no**

The strongest statement supported by this run is:
- performance improves sharply through `muon` and `untie_embed`
- later additions produce mixed tradeoffs rather than strict monotone gains

Using the best replayed validation loss available at each stage:
- `baseline -> rope`: slightly **worse** on validation (`2.1866 -> 2.2138`), though test improves (`2.1967 -> 1.9484`)
- `rope -> muon`: clearly better (`2.2138 -> 1.9576`)
- `muon -> untie_embed`: clearly better (`1.9576 -> 1.8272`)
- `untie_embed -> value_mix`: **worse** (`1.8272 -> 1.9043`)
- `value_mix -> unet`: **worse** (`1.9043 -> 1.9367`)
- `unet -> fixed_window`: slightly **worse** on validation (`1.9367 -> 1.9416`)
- `fixed_window -> attn_scale`: better (`1.9416 -> 1.8767`)

So the cumulative story is not "every added variant helps." It is:
- early architecture/optimizer upgrades help a lot
- later additions shift the best batch regime and trade token-efficiency, throughput, and final loss against one another

This is also visible at fixed batch size:
- at `B=512`, `untie_embed -> value_mix` regresses on both validation and test loss
- at `B=512`, `value_mix -> unet` regresses again
- at `B=128`, `unet -> fixed_window` is slightly worse on validation loss

### What the feature-learning analysis shows

The feature-learning pipeline completed successfully after re-running the map analysis with checkpoint-aligned early checkpoints:
- `450, 850, 1550`

The key outputs are:
- `feature_early_prediction.csv`
- `feature_transition_correlations.csv`
- `feature_variant_onset.csv`
- `feature_learning_main_2x2.png`

#### A. Early feature-learning signal exists, but only directionally in this one-seed run

From `feature_early_prediction.csv`:
- at checkpoint `450`, `delta_H_peak` has absolute Spearman `0.8660` with `tok_gain`
- the corresponding final-activation endpoint baseline is `0.3286`
- so the early spectral signal beats the late endpoint baseline in this run

However:
- this strongest result is based on `n=3`
- later checkpoints are too underpowered here (`n=1`) to support a stable ranking claim

Interpretation:
- this is supportive evidence for the paper's "early spectrum predicts later efficiency" claim
- it is not yet strong enough to treat as a definitive replication

#### B. Transition-level spectral shifts align with later gains

From `feature_transition_correlations.csv`, the strongest finite transition-level correlations are around `|rho| = 0.7071`:
- `delta_H_peak` vs `tok_gain`
- `delta_drop_advantage` vs `tok_gain`
- similar magnitudes for `thr_gain`

These peaks appear at early checkpoints `450` and `550`, in both probe bands.

Interpretation:
- early changes in spectral concentration and drop/keep asymmetry are carrying real signal about later transition gains
- this is consistent with the paper's feature-learning interpretation

#### C. Stronger variants tend to show earlier onset

From `feature_variant_onset.csv` in the matched `179:50` band:
- `baseline`: `H_peak` onset at `5000`
- `rope`: onset at `3500`
- `muon`: onset at `1200`
- `untie_embed`: onset at `300`
- `value_mix`: onset at `300`
- `unet`: onset at `300`
- `fixed_window`: onset at `450`
- `attn_scale`: onset at `450`

Interpretation:
- the better-performing post-`muon` variants tend to enter concentrated spectral structure much earlier than `baseline` or `rope`
- this is qualitatively aligned with the paper's feature-learning story

### Bottom line from the toy outputs

What is supported:
- early feature-learning metrics contain predictive signal
- stronger variants tend to exhibit earlier spectral onset
- feature-learning changes track later token/throughput gains directionally

What is not supported:
- the claim that every cumulative variant addition is strictly better
- a stable ranking of early predictors from one seed alone

Practical reading:
- use this toy run as evidence for **non-monotone but interpretable architectural progress**
- do **not** describe the cumulative variant chain as uniformly improving at every step

### Constant-loss comparison: pure train time to validation loss `<= 2.0`

For this sweep, the cleaner comparison is not final loss, but numeric target attainment using `train_time_s`, which excludes validation and checkpoint overhead.

Using `variant_concat_ablation_summary.csv` and the numeric rule `loss <= 2.0`:
- `baseline`: no batch reaches `2.0` within the 5k-step budget
- `rope`: no batch reaches `2.0` within the 5k-step budget
- `muon`: reaches target; fastest hit is `B=512`, `10.83s`
- `untie_embed`: reaches target; fastest hit is `B=512`, `2.77s`
- `value_mix`: reaches target; fastest hit is `B=512`, `3.28s`
- `unet`: reaches target; fastest hit is `B=512`, `3.32s`
- `fixed_window`: reaches target; fastest hit is `B=512`, `4.98s`
- `attn_scale`: reaches target; fastest hit is `B=512`, `4.98s`

So even on the paper-aligned constant-loss criterion, the chain is still **not** strictly monotone:
- `muon -> untie_embed`: much faster
- `untie_embed -> value_mix`: slower
- `value_mix -> unet`: slightly slower
- `unet -> fixed_window`: slower
- `fixed_window -> attn_scale`: essentially flat on time (`4.98s -> 4.98s`)

At fixed batch size, the same non-monotonicity appears:
- `B=128`: `muon 13.90s -> untie_embed 4.03s -> value_mix 4.73s -> unet 5.66s -> fixed_window 8.24s -> attn_scale 7.41s`
- `B=512`: `muon 10.83s -> untie_embed 2.77s -> value_mix 3.28s -> unet 3.32s -> fixed_window 4.98s -> attn_scale 4.98s`

Interpretation:
- the right constant-loss claim is that some additions dramatically improve efficiency, especially `muon` and `untie_embed`
- the data do **not** support saying that each added variant strictly reduces time-to-target

## 9) Corrected four-stage feature-learning analysis (`runs/modarith_a_seed0_steps5000_asha_fix_20260331`)

This section supersedes Section 8 for `muon` onward.
It uses the corrected Muon-LR sweep, restricts attention to the first four cumulative stages
`baseline -> rope -> muon -> untie_embed`,
and builds the transition map on the three batch trajectories separately by encoding batch as the trajectory id (`seed = 32, 128, 512`).

Inputs:
- corrected replay summary:
  `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/variant_concat_ablation_summary.csv`
- corrected four-stage probe outputs:
  `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/feature_learning_analysis/`

### Constant-loss picture for the trusted four-stage chain

Using the numeric rule `loss <= 2.0` and pure `train_time_s`:
- `baseline`: no batch reaches target within the 5k-step budget
- `rope`: reaches target first at `B=512`, `27.29s`, `121.57M` tokens
- `muon`: fastest hit is `B=512`, `9.88s`; token-optimal hit is `B=128`, `15.56M` tokens
- `untie_embed`: fastest hit is `B=512`, `2.29s`; token-optimal hit is `B=32`, `3.75M` tokens

Interpretation:
- within this restricted four-stage chain, the wall-clock constant-loss story is now monotone in the intended direction:
  `baseline` fails to hit target, `rope` hits it, `muon` is much faster, and `untie_embed` is much faster again
- the batch-regime claim is still architecture-dependent in token terms:
  `rope` prefers `B=512`, `muon` prefers `B=128`, and `untie_embed` prefers `B=32`

### Feature-learning probe setup

For the corrected four-stage subset, the probe was re-run with shared explicit checkpoints:
- `0, 500, 1000, 1500, 2000, 2500, 5000`
- plus final/target checkpoints when those differ

The main derived outputs are:
- `feature_variant_transition_map.csv`
- `feature_early_prediction.csv`
- `feature_transition_correlations.csv`
- `feature_variant_onset.csv`
- `feature_transition_summary.csv`

### What is supported

#### A. Early feature signal is present in the corrected run

From `feature_early_prediction.csv`:
- at checkpoint `500`, `delta_H_peak` has absolute Spearman `0.8333` with transition `tok_gain`
- the corresponding final-activation endpoint baseline is only `0.1333`
- this early signal remains above the endpoint baseline at checkpoints `1000` and `1500`
  (`0.2143` and `0.2571` versus `0.1333`)

Interpretation:
- the paper's "early spectrum has predictive power" claim is supported directionally in the corrected toy run
- the strongest evidence is at the first shared early checkpoint, `500`

#### B. The largest performance jump also has the clearest early feature shift

At checkpoint `500`, averaged over the two matched probe bands:
- `baseline -> rope` has small positive `delta_H_peak`
  (`0.0047` to `0.0113`)
- `rope -> muon` is still modest and band-dependent
  (`0.0038` to `0.0162`)
- `muon -> untie_embed` is much larger
  (`0.0930` to `0.1200`)

Interpretation:
- the strongest step in the corrected four-stage chain, `muon -> untie_embed`, also shows the clearest early increase in feature concentration
- this is consistent with the feature-learning story in the paper

#### C. Early slope is more informative than onset threshold in this toy setting

The onset thresholds are coarse because the checkpoint grid is coarse and many trajectories only share a few checkpoints.
The median `H_peak_onset_checkpoint` therefore ties often and is not the best discriminator here.

The median early `H_peak` slope is more useful.
For matched band `179:50`:
- `baseline`: `2.3e-5`
- `rope`: `1.7e-5`
- `muon`: `7.7e-5`
- `untie_embed`: `3.25e-4`

For matched band `97:200`:
- `baseline`: `1.1e-5`
- `rope`: `7.0e-6`
- `muon`: `3.5e-5`
- `untie_embed`: `2.23e-4`

Interpretation:
- the corrected toy run supports a stronger claim about **early growth rate** of feature concentration than about a single hard onset time

#### D. Drop/keep asymmetry is also predictive

In `feature_transition_correlations.csv`, `delta_drop_advantage` is competitive with, and sometimes stronger than, `delta_H_peak`.
In matched band `97:200`:
- checkpoint `500`: `rho = -0.9286` with `tok_gain`
- checkpoint `1000`: `rho = -0.9643` with `tok_gain`

Interpretation:
- this corrected toy run does **not** isolate a single universally dominant early predictor
- both concentration (`delta_H_peak`) and causal asymmetry (`delta_drop_advantage`) carry useful signal

### What is not cleanly supported yet

- the transition taxonomy is too brittle to push hard:
  at taxonomy checkpoint `500`, only the `B=32` `muon -> untie_embed` transition is labeled feature-dominant, while most other transitions are near-null
- the transition map's `tok_gain` and `time_gain` are endpoint-based, not constant-loss time-to-target metrics
- PCA-based transition features were not informative in this run and should not be emphasized

### Paper-facing read

For the corrected four-stage toy chain, the safe claim is:
- after the Muon-LR fix, the toy agrees with the main performance narrative through `untie_embed`:
  `rope -> muon -> untie_embed` improves constant-loss performance sharply
- early feature-learning diagnostics, especially `delta_H_peak` at checkpoint `500`, have predictive power for later transition quality
- in this small one-seed toy, early `H_peak` **slope** is a cleaner discriminator than onset threshold, and `delta_drop_advantage` is as important as `delta_H_peak`

What I would not claim from this run:
- a stable feature-vs-optimization taxonomy
- that one early predictor is uniquely best
- anything about `value_mix` or later variants

## 10) Detailed summary of corrected four-stage outputs and analyses

This section is the full audit trail for the corrected four-stage toy result.
It is the version I would use if we need one place that says:
- what analysis was actually run
- what files it produced
- what the corrected run says about performance
- what the corrected run says about feature learning

### Scope and trusted inputs

Trusted run root:
- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/`

Trusted stages:
- `baseline`
- `rope`
- `muon`
- `untie_embed`

Trusted replay summary:
- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/variant_concat_ablation_summary.csv`

Important construction detail:
- the four-stage analysis is done on the `core4_bybatch` subset
- here, the three batch trajectories are encoded as `seed = 32, 128, 512`
- this lets the transition map compare
  `baseline -> rope -> muon -> untie_embed`
  within each batch trajectory instead of mixing the three batch sizes together

### What analysis was done

#### 1. Corrected constant-loss replay analysis

From the trusted replay summary, I examined:
- selected LR per trusted run
- final validation loss
- final test loss
- `tokens_seen`
- pure `train_time_s`
- `stop_reason`
- target attainment using the numeric rule `loss <= 2.0`

This is the clean performance analysis because `train_time_s` excludes validation, checkpointing, and measurement overhead.

#### 2. Corrected feature probe re-run on shared checkpoints

I re-ran the feature probe on the trusted four-stage subset with:
- probe bands: `97:200`, `179:50`
- probe regimes: `clean_band` and `matched_band`
- shared manual checkpoints:
  `500, 1000, 1500, 2000, 2500, 5000`
- plus `initial`, `final`, and `target` checkpoints when present

This produced:
- `feature_probe_checkpoint_manifest.csv` with `68` selected checkpoint rows
- `feature_learning_summary.csv` with `272` rows
- `feature_causal_effects.csv` with `272` rows
- `feature_pca_summary.csv` with `2176` rows

The probe manifest confirms that the trusted four-stage analysis now has real shared early checkpoints instead of only `step 0`.

#### 3. Cross-variant transition-map analysis

I then ran `analyze_feature_learning_variant_map.py` on the corrected four-stage probe outputs with:
- checkpoints:
  `0, 500, 1000, 1500, 2000, 2500, 5000`
- early checkpoints:
  `500, 1000, 1500`
- taxonomy checkpoint:
  `500`

This produced:
- `feature_variant_transition_map.csv`
- `feature_variant_onset.csv`
- `feature_transition_summary.csv`
- `feature_variant_taxonomy.csv`
- `feature_transition_correlations.csv`
- `feature_early_prediction.csv`
- `feature_transition_stats.csv`
- `feature_alignment_checks.csv`
- `feature_variant_map_protocol.csv`

Key metadata from those outputs:
- `feature_variant_transition_map.csv` contains `82` matched-band transition rows
- the analysis protocol selected `matched_band` as the main probe regime
- checkpoint alignment is marked `ok`
- loss-proximal alignment is unavailable because `variant_concat_ablation_matched_pairs.csv` is not present for this subset

### Detailed performance table for the trusted four-stage chain

| stage | B | lr | val loss | test loss | tokens seen | train time (s) | stop reason | final step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| baseline | 32 | 0.000600 | 3.7632 | 3.7977 | 10,240,000 | 10.72 | `max_steps` | 5000 |
| baseline | 128 | 0.001697 | 3.1412 | 3.1362 | 40,960,000 | 11.50 | `max_steps` | 5000 |
| baseline | 512 | 0.003394 | 2.1866 | 2.1967 | 163,840,000 | 29.70 | `max_steps` | 5000 |
| rope | 32 | 0.000600 | 3.5154 | 3.5449 | 10,240,000 | 16.19 | `max_steps` | 5000 |
| rope | 128 | 0.002400 | 2.6315 | 2.6044 | 40,960,000 | 16.88 | `max_steps` | 5000 |
| rope | 512 | 0.006787 | 1.9938 | 1.9829 | 121,569,280 | 27.29 | `target_loss` | 3710 |
| muon | 32 | 0.000424 | 2.0907 | 2.1252 | 10,240,000 | 35.21 | `max_steps` | 5000 |
| muon | 128 | 0.001697 | 1.9929 | 2.0016 | 15,564,800 | 13.59 | `target_loss` | 1900 |
| muon | 512 | 0.002400 | 1.9839 | 1.9785 | 36,044,800 | 9.88 | `target_loss` | 1100 |
| untie_embed | 32 | 0.000600 | 1.9938 | 2.0082 | 3,747,840 | 13.31 | `target_loss` | 1830 |
| untie_embed | 128 | 0.001200 | 1.9741 | 1.9861 | 4,423,680 | 3.98 | `target_loss` | 540 |
| untie_embed | 512 | 0.001697 | 1.9905 | 1.9962 | 8,192,000 | 2.29 | `target_loss` | 250 |

### What these performance numbers mean

#### A. The corrected four-stage chain now supports the intended performance story

Using the strict numeric target rule `loss <= 2.0`:
- `baseline` never hits target
- `rope` does hit target, but only at `B=512`
- `muon` hits target at `B=128` and `B=512`, and is much faster than `rope`
- `untie_embed` hits target at all three trusted batch sizes, and is much faster than `muon`

So the corrected wall-clock constant-loss story is:
- `baseline`: fail
- `rope`: first success
- `muon`: major speedup
- `untie_embed`: another major speedup

#### B. Wall-clock-optimal and token-optimal batch size diverge by stage

Wall-clock-optimal batch:
- `rope`: `B=512`
- `muon`: `B=512`
- `untie_embed`: `B=512`

Token-optimal batch:
- `rope`: `B=512`
- `muon`: `B=128`
- `untie_embed`: `B=32`

Interpretation:
- the toy supports the paper's "efficient batch regime is architecture-dependent" claim
- the wall-clock story and the token-efficiency story are not the same object

### What the corrected feature-learning analysis says

#### A. Early feature metrics have predictive signal

From `feature_early_prediction.csv`:

| checkpoint | `|rho(delta_H_peak, tok_gain)|` | final endpoint baseline | n |
| ---: | ---: | ---: | ---: |
| 500 | 0.8333 | 0.1333 | 8 |
| 1000 | 0.2143 | 0.1333 | 7 |
| 1500 | 0.2571 | 0.1333 | 6 |

Interpretation:
- the corrected toy supports the claim that early feature-learning signal has predictive power
- the strongest evidence is early, at `checkpoint = 500`
- the signal weakens later because the sample size shrinks as fewer transitions still share the same checkpoint

#### B. `muon -> untie_embed` looks most feature-learning-heavy

At checkpoint `500`, from `feature_transition_summary.csv`:
- `baseline -> rope`
  - `tok_gain_mean = 0.1278`
  - `time_gain_mean = -0.1894`
  - `delta_H_peak_mean = 0.0113` in band `179:50`
  - `delta_H_peak_mean = 0.0047` in band `97:200`
- `rope -> muon`
  - `tok_gain_mean = 0.3238`
  - `time_gain_mean = 0.4880`
  - `delta_H_peak_mean = 0.0162` in band `179:50`
  - `delta_H_peak_mean = 0.0038` in band `97:200`
- `muon -> untie_embed`
  - `tok_gain_mean = 0.0330`
  - `time_gain_mean = 2.0313`
  - `delta_H_peak_mean = 0.1200` in band `179:50`
  - `delta_H_peak_mean = 0.0930` in band `97:200`

Interpretation:
- `rope -> muon` gives the largest endpoint-loss improvement
- `muon -> untie_embed` gives by far the largest early increase in feature concentration and by far the largest time gain
- so the cleanest feature-learning inference is:
  `muon -> untie_embed` looks like the transition where the model starts organizing useful structure much earlier, not just ending with a better final loss

#### C. `H_peak` slope is more useful than onset threshold

The onset threshold itself is not very discriminative here because:
- the checkpoint grid is coarse
- the shared-checkpoint set differs across trajectories
- several stages tie on onset

The early slope is more informative.
Median early `H_peak` slope by stage:

Matched band `179:50`:
- `baseline`: `2.3e-5`
- `rope`: `1.7e-5`
- `muon`: `7.7e-5`
- `untie_embed`: `3.25e-4`

Matched band `97:200`:
- `baseline`: `1.1e-5`
- `rope`: `7.0e-6`
- `muon`: `3.5e-5`
- `untie_embed`: `2.23e-4`

Interpretation:
- the corrected toy supports a strong claim about **how fast** feature concentration ramps up
- it supports a weaker claim about a single onset checkpoint

#### D. Causal asymmetry is also part of the story

`delta_drop_advantage` is often as strong as `delta_H_peak`.
From `feature_transition_correlations.csv` in matched band `97:200`:
- at `500`, `rho(delta_drop_advantage, tok_gain) = -0.9286`
- at `1000`, `rho(delta_drop_advantage, tok_gain) = -0.9643`

Interpretation:
- the better transitions are not only more spectrally concentrated
- they also become more causally selective, in the sense that dropping the key feature matters more relative to control structure

#### E. Auxiliary direct check: larger early `delta_H_peak` also aligns with faster runs

As an additional direct calculation on `feature_variant_transition_map.csv`:
- at checkpoint `500`, pooled across both matched bands, `delta_H_peak` has Spearman `0.75` with `time_gain`
- at checkpoint `500`, pooled across both matched bands, `delta_H_peak` has Spearman `0.83` with `thr_gain`

Interpretation:
- in this corrected toy run, bigger early gains in feature concentration are associated not only with better endpoint transitions, but also with faster later execution metrics
- this is auxiliary evidence only, because these are endpoint-based transition metrics, not matched constant-loss time-to-target metrics

### What the corrected feature-learning outputs let us claim

Safe claim:
- the trusted four-stage toy result now supports the narrative that
  `rope -> muon -> untie_embed`
  corresponds to progressively stronger early feature organization
- early feature-learning diagnostics have predictive power
- `untie_embed` appears to accelerate the emergence of concentrated, task-relevant structure especially strongly

Still too weak to claim:
- a stable feature-vs-optimization taxonomy
- that one early predictor is universally best
- that onset threshold is the main statistic to report

### Recommended paper usage from this corrected summary

If this is used in the paper, the safest wording is:
- in the corrected toy model, the trusted early chain
  `baseline -> rope -> muon -> untie_embed`
  shows monotone improvement in constant-loss performance
- those gains are accompanied by stronger early feature-learning signal
- the best single early toy readout is not "onset time" but rather
  early `delta_H_peak` and early `H_peak` growth rate
- causal drop/keep asymmetry should be reported as a secondary but real contributor
