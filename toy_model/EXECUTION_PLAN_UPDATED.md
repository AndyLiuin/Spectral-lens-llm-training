# Updated Toy Execution Plan (Mechanism-Focused)

## Main claims to prioritize

1. Batch size/noise scale changes spectral geometry at matched loss.
2. Spectrum can distinguish representational effects from hardware-throughput effects.
3. Observed spectral differences are not only estimator artifacts.

## Why linear-RFF is retained

- It is a control to separate data/optimization mechanism from Transformer-specific architecture effects.
- Agreement across Track A/Track B strengthens mechanism claims.
- Disagreement is informative and should be reported as architecture dependence.

## Updated ablation set

1. Estimator artifact ablation
- Hold trained model fixed.
- Vary measurement sample count `n` and mode (fixed vs non-fixed samples).
- Report RankMe, `alpha_tail` vs `d/n`.

2. Noise-scale / batch-size ablation (core)
- Sweep batch sizes with unconstrained LR and with approximately fixed noise scale.
- Compare matched-loss spectrum divergence.
- Primary readout: divergence reduction under noise-scale control.

3. Distribution ablation (new, data vs architecture)
- Latent distributions: `gaussian`, `uniform`, `student_t`.
- Anisotropy modes: `isotropic`, `powerlaw` (with gamma sweep).
- Run both tracks for each condition.
- Primary readout: whether spectral shape trends persist across tracks under the same data distribution.

4. Heavy-tail latent stress-test (new)
- Use `student_t` latent + power-law anisotropy.
- Compare against Gaussian and Uniform baselines at matched setup.
- Primary readout: stronger tail/heavy-tail signatures in measured spectra and downstream metrics.

5. Scaling-link ablation
- Fit `L(D)=aD^{-s}+c` and relate `s` to spectral proxies (`alpha_tail`).
- Report correlation + bootstrap CI.

6. Variant ablation (new)
- Controlled one-change-at-a-time variants on the toy Transformer:
  `baseline`, `rope`, `muon`, `unet`, `fixed_window`, `attn_scale`.
- Keep data generation fixed while toggling exactly one mechanism axis.
- Compare final and matched-loss spectra against baseline.

## Decision criteria

- If Track A and Track B agree in direction under distribution sweeps, claim is likely data/optimization-driven.
- If only Track A changes strongly, flag architecture-specific mechanism.
- If Uniform-isotropic still shows strong heavy-tail exponents, architecture contributes substantially.
- If Student-t + powerlaw anisotropy strengthens tails in both tracks, data-distribution role is supported.

## Practical run order

1. `run_estimator_ablation.py`
2. `run_noise_scale_ablation.py`
3. `run_distribution_ablation.py`
4. `run_phase_ablation.py` (supporting narrative)
5. `run_scaling_link_ablation.py`
6. `run_variant_ablation.py`
7. `plot_toy_main_2x2.py` (+ add distribution/variant appendix plot/table)
