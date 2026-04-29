# Toy RFF Two-Track Pipeline

This package implements the two-track toy setup:
- Track A: causal Transformer on frozen random Fourier feature (RFF) inputs.
- Track B: linear RFF control on the same generated teacher/data.

It now supports two task regimes:
- `rff_regression` (default): continuous latent -> fixed RFF teacher-student regression.
- `mod_arith_lm`: notebook-aligned banded modular arithmetic next-token language modeling.

It provides:
- deterministic latent/RFF teacher data generation with exported train/val/test splits,
- optional modular-arithmetic latent-component exports (`component_count`, `c/o/d/a`, `weights`) for mechanistic probes,
- configurable latent distributions (`gaussian`, `uniform`, `student_t`) and anisotropy (`isotropic`, `powerlaw`),
- controlled Transformer variant toggles (`baseline`, `rope`, `muon`, `untie_embed`, `value_mix`, `unet`, `fixed_window`, `attn_scale`),
- cumulative variant stacks such as `baseline+rope+muon+...`,
- unified spectral measurement (activation + gradient spectra, RankMe, alpha head/tail, top10),
- large-model-style Muon split: AdamW on embeddings/head/scalars, Muon on hidden-layer matrices,
- matched-step logs (`metrics_over_time.csv`) and global summary (`toy_summary.csv`),
- batch/LR sweep tooling for cumulative stages, and feature-learning probe/analysis scripts.

## Outputs

Each run writes:
- `toy_outputs/<ablation>/<run_name>/metrics_over_time.csv`
- `toy_outputs/<ablation>/<run_name>/spectra/*.npy`
- `toy_outputs/<ablation>/<run_name>/run_config.json`
- `toy_outputs/<ablation>/<run_name>/dataset_ref.txt`

Shared dataset exports:
- `toy_outputs/datasets/<dataset_key>/train_split.npz`
- `toy_outputs/datasets/<dataset_key>/val_split.npz`
- `toy_outputs/datasets/<dataset_key>/test_split.npz`
- `toy_outputs/datasets/<dataset_key>/teacher_params.npz`
- `toy_outputs/datasets/<dataset_key>/teacher_metadata.json`

Global summary:
- `toy_outputs/toy_summary.csv`

## Commands

Run these from the repository root shown in this checkout. The in-place module entrypoints are `python -m <script_name>`, not `python -m toy_model.<script_name>`.

Estimator ablation:
```bash
python -m run_estimator_ablation \
  --tracks a,b --d 8 --beta 1.5 --P 512 --D 32000 --B 128 --lr 3e-4 --max-steps 1000
```

Noise-scale ablation:
```bash
python -m run_noise_scale_ablation \
  --tracks a,b --d 8 --beta 1.5 --P 512 --D 32000 --B-list 32,128,512 --base-lr 3e-4 --max-steps 1000
```

Phase ablation:
```bash
python -m run_phase_ablation \
  --tracks a,b --d 8 --beta 1.5 --P 512 --D 32000 --max-steps 1200
```

Scaling-link ablation:
```bash
python -m run_scaling_link_ablation \
  --tracks a,b --d-values 4,8,16 --beta-values 0.75,1.0,1.5,2.0 --P-values 256,512,1024 \
  --D-values 2000,8000,32000,128000 --seeds 0,1,2 --B 128 --lr 3e-4 --max-steps 1000
```

Variant ablation (representatives from large-model suite):
```bash
python -m run_variant_ablation \
  --tracks a --variants baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale \
  --d 8 --beta 1.5 --P 512 --D 32000 --B 128 --lr 3e-4 --seeds 0,1,2
```

Variant ablation on notebook-aligned modular arithmetic:
```bash
python -m run_variant_ablation \
  --task mod_arith_lm \
  --tracks a --variants baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale \
  --D 32000 --B 128 --lr 3e-4 --seeds 0,1,2 \
  --seq-len 64 --vocab-size 1024 --zipf-c 1.3 --zipf-o 1.2 \
  --c-min 5 --min-step-frac 0.125 --allow-noncoprime --noncoprime-prob 0.3 \
  --modarith-measurement-pooling last
```

Cumulative variant ablation on modular arithmetic:
```bash
python -m old_scripts.run_variant_concat_ablation \
  --task mod_arith_lm \
  --tracks a \
  --variant-order baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale \
  --D 32000 --B 128 --lr 3e-4 --seeds 0,1,2 \
  --max-steps 1000 \
  --seq-len 64 --vocab-size 1024 --zipf-c 1.3 --zipf-o 1.2 \
  --c-min 5 --min-step-frac 0.125 --allow-noncoprime --noncoprime-prob 0.3 \
  --modarith-measurement-pooling last
```

Cumulative-stage batch/LR sweep on modular arithmetic:
```bash
python -m run_concat_batch_regime_ablation \
  --task mod_arith_lm \
  --tracks a \
  --variant-order baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale \
  --D 32000 \
  --seq-len 64 \
  --B-list 32,128,512 \
  --base-lr 3e-4 \
  --lr-scaling both \
  --lr-multipliers 0.5,0.707,1.0,1.414,2.0 \
  --seeds 0,1,2 \
  --max-steps 1000 \
  --target-loss 2.0 \
  --target-loss-metric val \
  --vocab-size 1024 --zipf-c 1.3 --zipf-o 1.2 \
  --c-min 5 --min-step-frac 0.125 --allow-noncoprime --noncoprime-prob 0.3 \
  --modarith-measurement-pooling last
```

To prune LR candidates progressively instead of training the full grid to completion, add
`--asha --asha-rungs 0.125,0.5,1.0 --asha-eta 2`. This runs successive rungs within each
`(variant stage, batch size, seed)` search cell, then keeps the existing selected-run replay path unchanged.

Cumulative variant ablation with a fixed LR/B schedule (exploratory only; not the fair constant-loss benchmark):
```bash
python -m old_scripts.run_variant_concat_ablation \
  --task mod_arith_lm \
  --tracks a \
  --variant-order baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale \
  --D 32000 --B 128 --lr 3e-4 --seeds 0,1,2 \
  --max-steps 2000 \
  --save-checkpoints --checkpoint-every 200 \
  --seq-len 64 --vocab-size 1024 --zipf-c 1.3 --zipf-o 1.2 \
  --c-min 5 --min-step-frac 0.125 --allow-noncoprime --noncoprime-prob 0.3 \
  --modarith-measurement-pooling last
```

Primary fairness benchmark: per-stage batch/LR sweep with selected-run replay, validation target loss, checkpoints, and matrix spectra:
```bash
python -m run_concat_batch_regime_ablation \
  --task mod_arith_lm \
  --tracks a \
  --variant-order baseline,rope,muon,untie_embed,value_mix,unet,fixed_window,attn_scale \
  --D 32000 --B-list 32,128,512 --base-lr 3e-4 --lr-scaling both \
  --lr-multipliers 0.5,0.707,1.0,1.414,2.0 \
  --seeds 0,1,2 \
  --max-steps 4000 \
  --target-loss 3.0 --target-loss-metric val --target-loss-patience 1 --target-loss-min-steps 100 \
  --eval-every 10 --measurement-every 50 \
  --materialize-selected-runs \
  --save-checkpoints --checkpoint-every 200 \
  --save-param-spectra --grad-svd-samples 64 \
  --selected-out-root toy_model/concat_batch_regime_selected_runs \
  --seq-len 64 --vocab-size 1024 --zipf-c 1.3 --zipf-o 1.2 \
  --c-min 5 --min-step-frac 0.125 --allow-noncoprime --noncoprime-prob 0.3 \
  --out-root toy_model \
  --modarith-measurement-pooling last
```

This writes the sweep diagnostics under `toy_model/concat_batch_regime_ablation/` and the probe-ready selected replays under `toy_model/concat_batch_regime_selected_runs/variant_concat_ablation/`.

Run standardized FFT + causal keep/drop + PCA probes over the selected replayed runs. By default the probe script includes the initial checkpoint, the final checkpoint, and the checkpoint nearest the run's validation target:
```bash
python -m run_feature_learning_probe \
  --ablation-dir toy_model/concat_batch_regime_selected_runs/variant_concat_ablation \
  --bands 97:200,179:50 \
  --probe-regime both
```

Cross-variant transition synthesis and taxonomy. This is the primary per-trick feature-learning analysis:
```bash
python -m analyze_feature_learning_variant_map \
  --ablation-dir toy_model/concat_batch_regime_selected_runs/variant_concat_ablation \
  --feature-dir toy_model/concat_batch_regime_selected_runs/variant_concat_ablation/feature_learning_analysis \
  --bands 97:200,179:50
```

Optional global bridge analysis. Use this for a pooled sanity check, not as the main per-trick diagnostic:
```bash
python -m old_scripts.analyze_feature_learning_bridge \
  --ablation-dir toy_model/concat_batch_regime_selected_runs/variant_concat_ablation \
  --feature-analysis-dir toy_model/concat_batch_regime_selected_runs/variant_concat_ablation/feature_learning_analysis
```

Paper-facing 2x2 feature-learning panel + appendix plots:
```bash
python -m plot_feature_learning_panels \
  --feature-dir toy_model/concat_batch_regime_selected_runs/variant_concat_ablation/feature_learning_analysis
```

One-command pipeline (fairness-first wrapper):
```bash
python -m run_feature_learning_upgrade_pipeline \
  --ablation-dir toy_model/concat_batch_regime_selected_runs/variant_concat_ablation \
  --run-training
```

Analyze cumulative modular arithmetic runs and produce notebook-style plots:
```bash
python -m old_scripts.plot_modarith_variant_concat \
  --out-root toy_outputs \
  --track a \
  --seed 0 \
  --max-matrices 4 \
  --alpha1 0,5 \
  --alpha2 9,50
```

Heavy-tail modular arithmetic (multi-component mixtures + heavy-tailed weights/noise):
```bash
python -m run_variant_ablation \
  --task mod_arith_lm \
  --tracks a --variants baseline \
  --D 32000 --B 128 --lr 3e-4 --seeds 0,1,2 \
  --seq-len 64 --vocab-size 1024 \
  --mix-components-min 2 --mix-components-max 6 \
  --component-weight-pareto-alpha 1.5 \
  --token-noise-std 0.5 --token-noise-t-df 3.0 \
  --modarith-measurement-pooling last
```

Distribution (data-property vs architecture-property) ablation:
```bash
python -m run_distribution_ablation \
  --tracks a,b --d 8 --beta 1.5 --P 512 --D 32000 --B 128 --lr 3e-4 --seeds 0,1,2 \
  --latent-dists gaussian,uniform,student_t --anisotropy-modes isotropic,powerlaw --anisotropy-gammas 0.8,1.2
```

Heavy-tail latent setup example:
```bash
python -m run_distribution_ablation \
  --tracks a,b --latent-dists student_t --latent-df 3.0 --anisotropy-modes powerlaw --anisotropy-gammas 1.2
```

Plot toy 2x2:
```bash
python -m plot_toy_main_2x2 --out-root toy_outputs --out-file toy_outputs/toy_main_2x2.png
```

Plot distribution ablation:
```bash
python -m plot_distribution_ablation --out-root toy_outputs
```

Run full suite + plot:
```bash
python -m run_all_ablation_suite --out-root toy_outputs --device cpu
```

## Notes

- `run_scaling_link_ablation.py` supports `--max-runs` for capped smoke tests.
- `run_estimator_ablation.py` supports `--n-list` to set measurement sample sizes.
- `run_variant_ablation.py` is the controlled one-change-at-a-time runner for variant probing.
- When `muon` is active and `--muon-notebook-lr-defaults` is left on, the notebook split ratios are now scaled by the run's base `--lr`; Muon-family LR sweeps therefore still sweep.
- Legacy compatibility entrypoints live under `toy_model.old_scripts/`.
- `old_scripts/run_variant_concat_ablation.py` runs cumulative stacks such as
  `baseline -> baseline+rope -> baseline+rope+muon -> ...`, matching the large-model
  “add one trick on top of the previous one” workflow.
- `old_scripts/run_variant_concat_ablation.py` can save checkpoints via
  `--save-checkpoints --checkpoint-every <steps>` for downstream feature-learning probes.
- `old_scripts/plot_modarith_variant_concat.py` reads saved `metrics_over_time.csv` and spectrum `.npy`
  files to produce per-stage dynamics CSVs, endpoint summaries, activation/gradient-proxy
  spectrum evolution plots, rankme-vs-tokens plots, optional GIFs, and (when available)
  per-matrix gradient-SVD and weight-spectrum plots.
- `train_toy_run` now supports fair-stop training by target loss via:
  `target_loss`, `target_loss_metric`, `target_loss_patience`, `target_loss_min_steps`.
  `old_scripts/run_variant_concat_ablation.py` and the primary fairness runner expose these through matching CLI flags.
- `run_concat_batch_regime_ablation.py` performs the large-model-style cumulative-stage
  batch/LR sweep. It writes:
  - `concat_batch_regime_trials.csv`
  - `concat_batch_regime_selected_lrs.csv`
  - `concat_batch_regime_matched_loss_rows.csv`
  - `concat_batch_regime_matched_pairs.csv`
  - `concat_batch_regime_early_prediction.csv`
- Matrix spectra are optional and controlled via:
  `--save-param-spectra --grad-svd-samples <n> [--param-spectrum-paths <csv>]`.
  Outputs are written per run to:
  - `spectra/gradsvd_spectrum_step_<step>__<matrix>.npy`
  - `spectra/weight_spectrum_step_<step>__<matrix>.npy`
  - `param_spectra_over_time.csv`
- `run_feature_learning_probe.py` writes:
  `feature_learning_summary.csv`, `feature_causal_effects.csv`, `feature_pca_summary.csv`,
  `feature_probe_checkpoint_manifest.csv`, and `feature_probe_missing_runs.csv` under the chosen
  feature-analysis directory.
- Feature probes now support two regimes:
  - `clean_band`: fixed single-band mechanistic microscope.
  - `matched_band`: anchored target band plus nuisance components/noise sampled from the
    training distribution.
  Use `--probe-regime auto|clean|matched|both` to control which tables are written.
- `old_scripts/analyze_feature_learning_bridge.py` generates a conservative pooled claim matrix combining
  current multi-variant spectra, matched-distribution probe evidence, and prior notebook evidence.
- `analyze_feature_learning_variant_map.py` creates
  `feature_variant_transition_map.csv`, `feature_transition_summary.csv`, and transition
  taxonomy/correlation/statistics tables.
  When hidden-state causal projections are available, they are used as the primary causal
  signal; embedding-band edits remain supporting evidence.
- `plot_feature_learning_panels.py` produces the toy feature-learning 2x2 main panel and
  appendix seed/band robustness plots.
- Most ablation scripts now expose `--task` so you can switch between `rff_regression` and `mod_arith_lm`.
- For `mod_arith_lm`, heavy-tail controls include:
  - `--mix-components-min/--mix-components-max` (superposed modular components),
  - `--component-weight-pareto-alpha` (heavy-tailed component weights),
  - `--token-noise-std` and `--token-noise-t-df` (optional heavy-tailed token noise).
- For `mod_arith_lm`, representation/gradient measurement pooling is configurable via
  `--modarith-measurement-pooling {token,last,mean}`:
  - `last` (default): one sample per sequence, aligned with RFF measurement semantics.
  - `mean`: one sample per sequence using mean pooled hidden states/residuals.
  - `token`: one sample per token (legacy behavior; effective sample count scales by `seq_len`).
- Cross-task gradient metrics are still not strictly apples-to-apples: the `mod_arith_lm`
  gradient proxy uses the true-class CE residual slice `(p_true - 1) * h`, while
  `rff_regression` uses the exact scalar-readout MSE gradient.
- The toy `muon` stage now follows the large-model split convention:
  embeddings/head/scalars use AdamW, and hidden-layer matrices use Muon.
- All runners expose latent controls: `--latent-dist`, `--latent-anisotropy`, and `--latent-anisotropy-gamma`.
- Track A depth can be switched between 1/2 layers with `--num-layers` (all runners expose this).
- All scripts support `--device cpu` or `--device cuda`.
