# Toy Outputs Methods, Analysis, and Results Synthesis

This document consolidates:

- `TOY_OUTPUTS_RESULTS_ANALYSIS.md`
- `TOY_OUTPUTS_RESULTS_AUDIT.md`
- `TOY_ARCHITECTURAL_TWEAKS_FEATURE_LEARNING_SUMMARY.md`

into one supervisor-safe writeup that covers:

- what data and code were used
- what was actually measured
- which conclusions are reliable
- what the toy outputs say about spectra, efficiency, and feature learning
- which claims should remain provisional

This synthesis is intentionally conservative. Where the original analysis and the audit disagree, this document follows the audit and the corrected artifact path.

## 1. Executive Summary

The toy results support three broad conclusions.

First, the earlier toy ablations already showed that spectral summaries are not decorative diagnostics. Across multiple settings, matched-loss geometry, phase sensitivity, noise-regime sensitivity, and estimator/sample-ratio sensitivity all change in systematic ways that differ across Track A and Track B.

Second, the mod-arithmetic constant-loss sweep supports the claim that architecture and optimizer choices change the efficient batch regime, and that matched-loss spectrum comparisons reveal nontrivial representation differences across those regimes.

Third, the strongest current toy evidence for **feature learning** comes from the corrected four-stage chain
`baseline -> rope -> muon -> untie_embed`.
Within that chain:

- `rope` looks like a small feature-learning assist.
- `muon` looks mainly like an optimization / token-efficiency improvement.
- `untie_embed` is the first clear feature-learning regime change.

That last point is the most important supervisor-facing interpretation from the current toy suite.

## 2. Reliability Status and Audit Outcome

### 2.1 High-level reliability status

The toy outputs are useful, but they are not all equally trustworthy.

- Many of the earlier ablations are effectively single-seed snapshots.
- Some results are presentation-safe only as directional evidence.
- The old multi-stage feature-learning transition analysis had a structural bug and should not be used as rigorous evidence.

### 2.2 What is safe to present

The safest current results are:

- the early ablation summaries in Sections 1-6 below, with single-seed caveats
- the corrected four-stage constant-loss story
- the corrected four-stage feature-learning analysis
- the architectural interpretation that separates `rope`, `muon`, and `untie_embed`

### 2.3 What should not be presented as rigorous evidence

Do **not** present Section 8 of the original `TOY_OUTPUTS_RESULTS_ANALYSIS.md` as rigorous.

Reason:

- the original transition map grouped by `(track, seed)` while multiple batch trajectories lived under the same seed
- that produced invalid self-transitions such as `baseline->baseline` and `rope->rope`

So the trusted feature-learning evidence should come from the corrected four-stage path, not the old all-stage feature map.

### 2.4 Important audit findings

The audit established four key caveats.

1. Old Section 8 is structurally invalid because batch trajectories were mixed.
2. Fixed-sample spectrum comparisons had a nonzero step-0 floor because the sampled measurement subset was keyed by run name rather than cached dataset identity.
3. “Target checkpoint” in the probe means nearest-to-target checkpoint, not guaranteed target hit.
4. The early-vs-endpoint baseline in the old `feature_early_prediction.csv` comparison was not fully apples-to-apples.

These issues have been fixed in code for future reruns, but the existing supervisor-safe interpretation must still respect them.

## 3. Inputs, Artifacts, and Code Paths

### 3.1 Original ablation outputs used

The earlier toy analysis used:

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

### 3.2 Mod-arithmetic constant-loss sweep artifacts

For the mod-arithmetic sweep, the main artifact roots are:

- uncorrected sweep:
  `runs/modarith_a_seed0_steps5000_asha_20260331/`
- corrected sweep:
  `runs/modarith_a_seed0_steps5000_asha_fix_20260331/`

Main files used:

- `concat_batch_regime_ablation/concat_batch_regime_selected_lrs.csv`
- `concat_batch_regime_ablation/constant_loss_stage_summary.csv`
- `concat_batch_regime_ablation/concat_batch_regime_early_prediction.csv`
- `concat_batch_regime_ablation/concat_batch_regime_matched_pairs.csv`
- `concat_batch_regime_selected_runs/variant_concat_ablation/variant_concat_ablation_summary.csv`

### 3.3 Corrected four-stage feature-learning artifacts

Trusted feature-learning inputs are:

- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/variant_concat_ablation_summary.csv`
- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/feature_learning_analysis/feature_learning_summary.csv`
- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/feature_learning_analysis/feature_causal_effects.csv`
- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/feature_learning_analysis/feature_variant_transition_map.csv`
- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/feature_learning_analysis/feature_transition_correlations.csv`
- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/feature_learning_analysis/feature_early_prediction.csv`
- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/concat_batch_regime_selected_runs/variant_concat_ablation_core4_bybatch/feature_learning_analysis/feature_transition_summary.csv`

### 3.4 Relevant code paths

The main code paths underlying the audited analysis are:

- [runner.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/runner.py)
- [run_concat_batch_regime_ablation.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/run_concat_batch_regime_ablation.py)
- [run_feature_learning_probe.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/run_feature_learning_probe.py)
- [analyze_feature_learning_variant_map.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/analyze_feature_learning_variant_map.py)
- [data_modarith.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/data_modarith.py)
- [models_modarith.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/models_modarith.py)
- [variant_utils.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/variant_utils.py)
- [modarith_variants/baseline.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/modarith_variants/baseline.py)
- [modarith_variants/rope.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/modarith_variants/rope.py)
- [modarith_variants/untie_embed.py](/nfs/roberts/project/pi_jks79/zl664/Scaling_law_final/FF_new/toy_model/modarith_variants/untie_embed.py)

## 4. Methods

### 4.1 Scope of the trusted toy setup

For the setup details in this synthesis, the trusted object is deliberately narrow:

- task: `mod_arith_lm`
- track: `a`
- seed: `0`
- variants: `baseline`, `baseline+rope`, `baseline+rope+muon`, `baseline+rope+muon+untie_embed`
- batch sizes: `32`, `128`, `512`

This is the four-stage chain that supports the current supervisor-facing feature-learning claims.
Later stages are omitted from the setup discussion because they are not carrying the main conclusion.

### 4.2 Toy task and dataset construction

The toy task is autoregressive next-token prediction on synthetic modular-arithmetic sequences.

Each example is built as follows.

1. Sample one latent modular component because the trusted runs use `mix_components_min = mix_components_max = 1`.
2. Sample a modulus / band width `c` from `5..1024` with Zipf weight `p(c) ∝ c^-1.3`.
3. Sample an offset `o` from `0..(V-c)` with Zipf-style weight `p(o) ∝ (o+1)^-1.2`.
4. Sample a step size `d` from `[max(1, floor(0.125 c)), c-1]`.
5. With probability `0.3`, force `d` to share a nontrivial factor with `c`, so the sequence is sometimes non-coprime with the modulus.
6. Sample a start phase `a` uniformly from `0..c-1`.
7. Generate the token path `t_k = o + ((a + d k) mod c)` for `k = 0, ..., 64`.
8. Form the language-model pair by setting input tokens `x = t_0, ..., t_63` and targets `y = t_1, ..., t_64`.

The trusted sweep uses:

- vocabulary size `V = 1024`
- sequence length `64`
- training set size `32,000`
- validation set size `2,048`
- test set size `4,096`
- no extra token noise: `token_noise_std = 0.0`
- no heavy-tailed mixture weighting: `component_weight_pareto_alpha = 0.0`

The dataset bundle is cached under a deterministic dataset key, so all compared variants for a given seed read the same train/validation/test arrays. The generator also saves latent arrays such as `c`, `o`, `d`, `a`, and component weights, which is useful for downstream probing or audit work.

### 4.3 Common model backbone

All four trusted stages use the Track A token-model family with the same backbone width/depth:

- `d_model = 128`
- `n_heads = 4`
- `num_layers = 2`
- `ff_mult = 2`

At a block level, the backbone is a small pre-norm causal transformer:

- residual self-attention branch
- residual `SquaredMLP` branch with `ReLU(x)^2`
- RMS-style normalization via `apply_norm`

Two implementation details matter for interpretation.

1. Attention and MLP output projections are zero-initialized, so residual updates start small.
2. The spectral measurements for `mod_arith_lm` use `modarith_measurement_pooling = last`, so activation spectra are built from the last-token hidden state rather than a token-average.

### 4.4 Variant definitions in the trusted chain

`baseline`

- learned token embedding plus learned positional embedding
- tied output head: `lm_head.weight = tok_embed.weight`
- standard causal self-attention with one joint `qkv` projection

`rope`

- removes learned positional embeddings
- uses rotary position embedding on normalized `q` and `k`
- keeps tied token embedding / output head
- otherwise keeps the same depth, width, and MLP structure

`muon`

- keeps the `rope` architecture
- changes the optimizer from plain AdamW to a split Muon setup
- embedding parameters, output-head parameters, and scalar parameters stay on AdamW
- matrix-shaped parameters move to Muon with momentum `0.95`, backend `newtonschulz5`, backend steps `5`, and Nesterov enabled

Because the trusted Muon runs use notebook-style defaults with no explicit per-group LR overrides:

- embedding/head/scalar groups use the swept base LR
- matrix groups use `50x` that base LR
- non-Muon AdamW groups use weight decay `0.01`

`untie_embed`

- keeps the `rope` block structure and the Muon optimizer
- unties the output head from the token embedding
- initializes the untied `lm_head` to zero
- adds an RMS normalization immediately after token embedding and before the first block

This is the first trusted stage that changes the readout geometry directly, not just positional encoding or optimization.

### 4.5 Training protocol and selected hyperparameters

The trusted four-stage sweep uses:

- `max_steps = 5000`
- target rule: validation loss `<= 2.0`
- `target_loss_metric = val`
- `eval_every = 10`
- `measurement_every = 250`
- `checkpoint_every = 500`
- checkpoints saved for later feature-probe reanalysis

The spectral measurement config in the trusted selected runs is:

- `n_samples = 1024`
- `fixed_samples = True`
- `trace_normalize = True`
- `alpha_head_range = 1:10`
- `alpha_tail_range = 50:200`

The selected learning rates come from the earlier ASHA search and differ by stage and batch:

| stage | optimizer | `B=32` | `B=128` | `B=512` |
| --- | --- | ---: | ---: | ---: |
| `baseline` | AdamW | `0.000600` | `0.001697` | `0.003394` |
| `rope` | AdamW | `0.000600` | `0.002400` | `0.006787` |
| `muon` | Muon split | `0.000424` | `0.001697` | `0.002400` |
| `untie_embed` | Muon split | `0.000600` | `0.001200` | `0.001697` |

This matters for interpretation: across the trusted chain, the data distribution, model width/depth, and sequence length are held fixed, while positional encoding, optimizer, and output-head tying are the controlled differences.

### 4.6 General measurement philosophy

Across these toy studies, two comparison principles matter most.

1. Endpoint metrics alone are insufficient.
The same final loss can hide different representation geometry and different training trajectories.

2. Matched-loss comparisons are more informative than raw endpoint comparisons.
Whenever possible, runs are compared at aligned loss levels, not just at the end of training.

### 4.7 Main metric families

The analyses use several recurring metric families.

#### Spectral geometry metrics

- `RankMe`
- `alpha_head`
- `alpha_tail`
- top-k spectral mass summaries
- matched-loss JS divergence between activation covariance spectra

Interpretation:

- larger JS at matched loss means the model occupies a more different representation geometry even when performance is aligned
- `RankMe` and tail-exponent summaries characterize effective rank / spectral steepness

#### Constant-loss efficiency metrics

Two types appear.

1. Numeric target attainment:
   validation loss `<= 2.0` in the mod-arithmetic sweep
2. Matched-loss efficiency:
   token or time comparisons at aligned loss levels

For architecture interpretation, token-based comparisons are more robust than wall-clock when the GPU may be shared.

#### Feature-learning probe metrics

The feature-learning probe tracks:

- `H_peak`: Fourier concentration of hidden activations
- `E_peak`, `Emb_peak`: related concentration summaries
- causal keep/drop perturbation losses
- hidden-state and embedding-space causal deltas
- PCA concentration summaries such as `pca_peak_mass_pc1`

Derived transition-level metrics include:

- `delta_H_peak`
- `delta_drop_advantage`
- `delta_keep_advantage`
- `delta_pca_peak_mass_pc1`

Interpretation:

- larger `delta_H_peak` means stronger concentration / localization of task-relevant feature structure
- `drop_advantage` captures how much more damaging it is to remove key structure than control structure

### 4.8 Constant-loss sweep design

The mod-arithmetic sweep is a cumulative-stage chain:

`baseline -> rope -> muon -> untie_embed -> value_mix -> unet -> fixed_window -> attn_scale`

with batch sizes:

- `B = 32`
- `B = 128`
- `B = 512`

and selected learning rates chosen from an ASHA sweep.

The most trustworthy performance comparisons use:

- `tokens_seen`
- `metrics_over_time.csv`
- numeric target attainment
- matched-loss pairings

rather than loosely interpreted stop reasons.

### 4.9 Corrected four-stage feature-learning protocol

The trusted feature-learning analysis restricts to:

- `baseline`
- `rope`
- `muon`
- `untie_embed`

and keeps the three batch trajectories separate.

The corrected probe uses shared early checkpoints:

- `0`
- `500`
- `1000`
- `1500`
- `2000`
- `2500`
- `5000`

plus `final` and nearest-to-target checkpoints where relevant.

The main probe regime used in the trusted map is `matched_band`.

### 4.10 Audit-driven interpretation rules

All later conclusions in this document follow these rules.

1. Treat old Section 8 as exploratory only.
2. Treat small matched-loss JS effects relative to the step-0 floor:
   roughly `2.6e-4` to `4.7e-4`.
3. Say “closest-to-target checkpoint,” not “target checkpoint.”
4. Treat the checkpoint-`500` early-prediction result as the cleanest one.
5. For architecture interpretation, prioritize:
   matched-loss token gain, matched-loss JS, `H_peak`, and PCA concentration.

## 5. Earlier Toy Ablations (Sections 1-6 of the Original Analysis)

These ablations are still useful, but mostly as **directional** evidence because many appear to be single-seed snapshots.

### 5.1 Variant ablation

From `variant_ablation_summary.csv`, endpoint test-loss ranking was:

1. `attn_scale`
2. `baseline`
3. `rope`
4. `unet`
5. `fixed_window`
6. `muon`

From `variant_ablation_matched_aggregate.csv`, matched-loss geometry distance to baseline ranked:

1. `attn_scale`
2. `muon`
3. `fixed_window`
4. `rope`
5. `unet`

Interpretation:

- `attn_scale` looked best on endpoint loss in that snapshot while staying geometrically close to baseline
- `rope` and `unet` showed larger geometry departures from baseline
- `muon` looked more like an optimization-dynamics intervention than a pure geometry-shaping intervention in that specific setup

### 5.2 Noise-scale ablation

For Track A, the fixed-noise regime produced much larger matched-loss JS divergence than the unconstrained regime.
For Track B, both regimes were nearly invariant.

Interpretation:

- the particular “fixed noise” recipe used there was not stabilizing Track A geometry
- Track B behaved like a stable control

### 5.3 Phase ablation

Track A showed clear phase-like non-monotonicity and stronger geometry shift under the transition intervention.
Track B was effectively phase-inert.

Interpretation:

- phase-like spectral transitions are architecture/dynamics dependent
- they are not just a property of the data generator

### 5.4 Scaling-link ablation

The pooled scaling-link correlation was weak and unstable, with opposite signs by track:

- Track A: positive
- Track B: negative

Interpretation:

- the scaling-link relation is not universal in the current snapshot
- it should be modeled per track, not as one pooled law

### 5.5 Distribution ablation

Track A had:

- steeper tails
- lower effective rank
- somewhat higher sensitivity to heavy-tail + anisotropy combinations

Track B had:

- broader representations
- higher RankMe
- more stable loss under the tested distribution changes

Interpretation:

- Track A is more geometry-sensitive
- Track B behaves as the broader, more stable control

### 5.6 Estimator ablation

The estimator study showed strong dependence of `alpha_tail` on `d/n`, especially for Track A.

Interpretation:

- finite-sample ratio is a major driver of measured tail-exponent estimates
- sample-ratio control matters more here than fixed/non-fixed sampling mode

### 5.7 Cross-ablation synthesis from the earlier toy suite

The earlier ablations collectively support:

1. Track A geometry is intervention-sensitive.
2. Track B often acts as a control.
3. Matched-loss comparisons are necessary.
4. Estimator/sample-ratio effects must be controlled before over-interpreting `alpha`-type metrics.
5. Pooled scaling-law claims are currently too weak.

## 6. Mod-Arith Constant-Loss Sweep: What It Establishes

The mod-arithmetic sweep is the bridge from generic toy ablations to architecture- and feature-learning-specific conclusions.

### 6.1 High-level conclusions from the sweep

What the sweep supports:

- batch regime changes representation geometry at matched loss
- efficient batch regime depends on architecture
- early spectral metrics have predictive signal for within-family efficiency

What it does **not** establish on its own:

- a clean systems-vs-learning decomposition
- the random Fourier feature causal-sanity-check claim

### 6.2 Important anomalies and fixes

The original uncorrected sweep had two serious issues.

1. From `muon` onward, the Muon optimizer path was affected by a learning-rate handling bug in the old run, so the old later-stage rankings should not be treated as definitive.
2. Search bookkeeping sometimes labeled target hits even when the numeric validation loss was still above `2.0`.

Therefore:

- the old multi-stage chain is useful for exploratory pattern-finding
- the corrected four-stage chain is the trusted performance/feature-learning path

## 7. Trusted Corrected Four-Stage Chain

This is the supervisor-safe core result.

Trusted chain:

- `baseline`
- `rope`
- `muon`
- `untie_embed`

Trusted run root:

- `runs/modarith_a_seed0_steps5000_asha_fix_20260331/`

### 7.1 Performance table

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

### 7.2 Constant-loss interpretation

Using the strict numeric target rule `loss <= 2.0`:

- `baseline` never hits target
- `rope` hits target only at `B=512`
- `muon` hits target at `B=128` and `B=512`
- `untie_embed` hits target at all three trusted batch sizes

This supports the intended qualitative performance story:

- `baseline`: fail
- `rope`: first success
- `muon`: major efficiency improvement
- `untie_embed`: another major efficiency improvement

### 7.3 Batch-regime interpretation

Wall-clock-optimal batch within the trusted chain:

- `rope`: `B=512`
- `muon`: `B=512`
- `untie_embed`: `B=512`

Token-optimal batch:

- `rope`: `B=512`
- `muon`: `B=128`
- `untie_embed`: `B=32`

Interpretation:

- efficient batch regime depends on architecture
- wall-clock-optimal and token-optimal regimes are different objects

## 8. Feature-Learning Analysis of the Trusted Chain

### 8.1 What was done

The corrected feature-learning analysis:

- re-ran probes on shared checkpoints
- used matched-band probes as the main regime
- built transitions within each batch trajectory
- produced transition summaries, early-prediction tables, and onset/slope summaries

### 8.2 Cleanest supported result: early feature signal exists

From the corrected `feature_early_prediction.csv`:

| checkpoint | `|rho(delta_H_peak, tok_gain)|` | endpoint baseline | n |
| ---: | ---: | ---: | ---: |
| 500 | 0.8333 | 0.0476 on the same available subset | 8 |
| 1000 | 0.2143 | 0.2143 on the same available subset | 7 |
| 1500 | 0.2571 | 0.2571 on the same available subset | 6 |

Interpretation:

- checkpoint `500` is the clean result
- the current corrected toy supports the claim that early feature-learning signal has predictive power
- later checkpoints are weaker and should not be oversold

### 8.3 Early feature concentration across stages

Using matched-band probes:

`H_peak` means:

- `baseline`: `0.0219` at `500`, `0.0357` at `1000`, `0.0531` at `1500`
- `rope`: `0.0299` at `500`, `0.0449` at `1000`, `0.0714` at `1500`
- `muon`: `0.0399` at `500`, `0.0649` at `1000`, `0.0662` at `1500`
- `untie_embed`: `0.1324` at `500`, `0.1880` at `1000`, `0.1487` at `1500`

PC1 peak mass means:

- `baseline`: `0.0505` at `500`, `0.0574` at `1000`, `0.1035` at `1500`
- `rope`: `0.0664` at `500`, `0.0838` at `1000`, `0.1467` at `1500`
- `muon`: `0.0819` at `500`, `0.1026` at `1000`, `0.1255` at `1500`
- `untie_embed`: `0.2290` at `500`, `0.3136` at `1000`, `0.3398` at `1500`

Interpretation:

- `rope` increases early concentration a little
- `muon` increases it somewhat more
- `untie_embed` changes the regime qualitatively

### 8.4 Transition-level feature deltas

Average transition deltas:

`baseline -> rope`

- `delta_H_peak_mean = 0.0080` at checkpoint `500`
- `delta_H_peak_mean = 0.0093` at `1000`
- `delta_H_peak_mean = 0.0183` at `1500`

`rope -> muon`

- `delta_H_peak_mean = 0.0100` at `500`
- `delta_H_peak_mean = 0.0200` at `1000`
- `delta_H_peak_mean = 0.0386` at `1500`

`muon -> untie_embed`

- `delta_H_peak_mean = 0.1065` at `500`
- `delta_H_peak_mean = 0.1655` at `1000`
- `delta_H_peak_mean = 0.1222` at `1500`

PCA-side jump for `muon -> untie_embed`:

- `delta_pca_mean = 0.1751` at `500`
- `delta_pca_mean = 0.2520` at `1000`
- `delta_pca_mean = 0.2833` at `1500`

Interpretation:

- `rope` and `muon` each nudge feature concentration upward
- `untie_embed` is the only transition producing a large, immediate, and persistent jump

### 8.5 Early slope vs onset threshold

The corrected toy supports a stronger claim about **early growth rate** than about a hard onset checkpoint.

Median early `H_peak` slope, matched band `179:50`:

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

- the best “when does structure appear?” statistic here is slope, not thresholded onset

### 8.6 Causal asymmetry as a secondary signal

`delta_drop_advantage` is also informative and sometimes as strong as `delta_H_peak`.

In the corrected transition-correlation outputs:

- at checkpoint `500` in matched band `97:200`, `rho(delta_drop_advantage, tok_gain) = -0.9286`
- at checkpoint `1000`, `rho(delta_drop_advantage, tok_gain) = -0.9643`

Interpretation:

- good transitions become not only more concentrated
- they also become more causally selective

Still, for the cleanest overall interpretation, `H_peak` and PCA concentration are easier to read and more monotone.

## 9. Architecture-Specific Interpretation

This is the most direct answer to:

“How do these tweaks affect feature learning?”

### 9.1 Matched-loss token gain and matched-loss JS by consecutive transition

Using the trusted selected-run summary plus matched-loss reconstruction from `metrics_over_time.csv`:

| transition | loss-proximal token gain | matched-loss JS | reading |
| --- | ---: | ---: | --- |
| `baseline -> rope` | `0.578` | `0.0063` | modest efficiency gain, small spectral displacement |
| `rope -> muon` | `2.476` | `0.0176` | very large efficiency gain, moderate spectral displacement |
| `muon -> untie_embed` | `1.898` | `0.1617` | large efficiency gain and very large spectral displacement |

This table is central because it separates:

- “how much more efficient is the next architecture at the same loss?”
from
- “how much does the learned representation geometry actually move?”

This synthesis intentionally stops at `untie_embed`, because those three transitions are the ones supporting the current feature-learning claim.

### 9.2 `rope`

Best reading:

- small matched-loss token gain
- small matched-loss spectral displacement
- small but consistent early `H_peak` increase over baseline

Interpretation:

`rope` looks like a **mild feature-learning assist** or architectural prior, not a regime change.

### 9.3 `muon`

Best reading:

- very large matched-loss token gain
- only moderate matched-loss spectral displacement
- modest and somewhat inconsistent feature-concentration increase relative to rope

Interpretation:

`muon` looks primarily like an **optimization accelerator**.
It helps the model get to good states much more efficiently, but it does not by itself produce the strongest qualitative change in the feature-learning trajectory.

### 9.4 `untie_embed`

Best reading:

- strong matched-loss token gain
- by far the largest matched-loss spectral displacement
- huge early jump in `H_peak`
- huge early jump in PC1 concentration

Interpretation:

`untie_embed` is the clearest **feature-learning architecture change** in the current toy study.

This is the transition where the model appears to start organizing task-relevant structure much earlier and much more sharply, not merely optimizing faster.

## 10. What the Toy Results Let Us Claim

### 10.1 Safe claims

The current toy suite safely supports the following.

1. Spectral instrumentation is informative.
2. Batch regime changes representation geometry at matched loss.
3. Efficient batch regime depends on architecture.
4. Early feature-learning diagnostics have predictive signal.
5. Within the trusted chain, `untie_embed` is the clearest feature-learning regime change.
6. `muon` is more naturally interpreted as optimization/efficiency than as the primary source of new feature structure.

### 10.2 Claims that are only directional

These are suggestive, but not yet robust.

1. The precise ranking of early feature predictors.
2. Any claim about a universal scaling-link relation.
3. Quantitative effect sizes from single-seed earlier ablations.

### 10.3 Claims not established here

These should not be claimed from the current toy outputs.

1. A clean systems-vs-learning decomposition.
2. A random-Fourier-feature causal sanity check from the current mod-arithmetic run alone.
3. A stable universal feature-vs-optimization taxonomy beyond the trusted four-stage chain.

## 11. Main Limitations

1. Much of the earlier toy suite is effectively single-seed.
2. The old multi-stage feature map was invalid and had to be superseded.
3. Some original matched-loss JS values had a small step-0 measurement floor.
4. The strongest feature-learning evidence currently covers only the corrected four-stage chain.
5. This synthesis does not attempt to interpret later variants after `untie_embed`.
6. Any use of wall-clock time should be interpreted cautiously when jobs may share a GPU.

## 12. Recommended Next Steps

To make the toy results stronger for paper use:

1. Replicate the corrected four-stage chain with multiple seeds.
2. Extend beyond the four-stage chain only if later variants become important to the paper story.
3. Keep using matched-loss token gain and matched-loss JS as the primary architecture-comparison axes.
4. Report checkpoint-`500` early feature results as the main predictive finding.
5. Phrase the architecture story as:
   `rope` helps a bit,
   `muon` optimizes faster,
   `untie_embed` changes feature learning.

## 13. Bottom Line

The toy suite now tells a coherent story.

The earlier ablations establish that spectral diagnostics are meaningful and that Track A geometry is intervention-sensitive.
The corrected mod-arithmetic chain then shows that architecture changes alter both the efficient batch regime and the emergence of feature structure.
Within that chain, the key distinction is:

- `rope`: small feature-learning assist
- `muon`: mostly optimization / token-efficiency
- `untie_embed`: qualitative feature-learning shift

That is the strongest current supervisor-facing conclusion from the toy outputs.
