# Toy Model Script-by-Script Interpretation Guide

This guide explains what each script in `toy_model` does, what files it writes, and how to interpret the outputs.

## 1) Big Picture

The pipeline is:
1. Build a synthetic teacher + dataset in latent space (`data.py`).
2. Train either Track A (Transformer on RFF sequence) or Track B (linear RFF control) (`models.py`, `runner.py`).
3. Measure representation and gradient spectra over training (`metrics.py`, `runner.py`).
4. Run ablations that stress specific hypotheses (`run_*_ablation.py`).
5. Aggregate and plot (`plot_*.py`).

The **core interpretation theme** is how spectral geometry (RankMe / power-law slopes / top eigenvalue concentration) changes with architecture, optimizer, data distribution, batch-noise regime, and scaling.

---

## 2) Core Infrastructure Scripts

## `config.py`
Defines all experiment configs and run naming.

Key snippet (`config.py:27-63`):
```python
@dataclass(frozen=True)
class RunConfig:
    track: str
    d: int
    beta: float
    p: int
    D: int
    B: int
    lr: float
    ...
    variant: str = "baseline"
    optimizer_name: str = "adamw"
```

How to interpret:
- `d, beta, p` control teacher/data complexity.
- `D, B, lr` are the optimization/data regime knobs.
- `variant` controls architecture/optimizer perturbations used in ablations.

Run naming (`config.py:64-90`) encodes all important knobs into folder names, so you can decode a run from its directory string.

---

## `data.py`
Generates the synthetic teacher and train/val/test sets.

### Teacher spectrum setup
Snippet (`data.py:25-31`):
```python
ranks = np.arange(1, p + 1, dtype=np.float64)
magnitudes = ranks ** (-beta)
...
coeff /= np.linalg.norm(coeff) + 1e-12
```
Interpretation:
- Teacher coefficients are power-law decayed with exponent `beta`.
- Larger `beta` => steeper teacher spectrum => fewer dominant modes.

### Latent distribution + anisotropy controls
Snippet (`data.py:55-67`, `34-44`):
```python
if latent_dist == "gaussian": ...
elif latent_dist == "uniform": ...
elif latent_dist == "student_t": ...
...
elif mode == "powerlaw":
    eig = ranks ** (-gamma)
```
Interpretation:
- `latent_dist` controls tail heaviness.
- `latent_anisotropy=powerlaw` with `gamma` controls directional variance imbalance.

### Label generation mechanism
Snippet (`data.py:115-120`):
```python
phi = rff_features_numpy(z, omega=omega, phase=phase)
pooled = phi.mean(axis=1)
y = pooled @ teacher_a
y = y + noise
```
Interpretation:
- Ground truth is linear in pooled RFF features.
- Track B (linear RFF readout) is an aligned control; Track A adds sequence modeling capacity.

Outputs:
- `train_split.npz`, `val_split.npz`, `test_split.npz`
- `teacher_params.npz`, `teacher_metadata.json`

---

## `models.py`
Defines model classes and Track A/B switch.

### Shared feature map
Snippet (`models.py:10-17`):
```python
feats = torch.cos(flat @ omega.t() + phase.unsqueeze(0))
feats = feats * (2.0 / p) ** 0.5
```
Interpretation:
- Both tracks see identical frozen RFF features.
- Differences are due to architecture/optimization, not changing input featurization.

### Track A: TransformerRFFRegressor
Snippet (`models.py:253-262`):
```python
phi = rff_features_torch(z, self.omega, self.phase)
x = self.input_proj(phi)
x = self._forward_blocks(x)
h = x[:, -1, :]
pred = self.readout(h).squeeze(-1)
```
Interpretation:
- Uses causal sequence processing, then predicts from last token representation.
- The measured activation spectrum is on `h` (the final representation).

### Variant toggles
Snippet (`models.py:191-196`):
```python
use_rope = variant in {"rope"}
qk_rmsnorm = variant in {"rope"}
use_unet = variant in {"unet"}
local_window = window_size if variant == "fixed_window" else 0
attn_scale = attention_scale if variant == "attn_scale" else None
```
Interpretation:
- Variant ablation isolates one mechanism at a time.

### Track B: LinearRFFRegressor
Snippet (`models.py:276-280`):
```python
phi = rff_features_torch(z, self.omega, self.phase)
h = phi.mean(dim=1)
pred = self.readout(h).squeeze(-1)
```
Interpretation:
- Serves as low-capacity/control baseline with same teacher/data.

---

## `metrics.py`
Computes spectral summaries.

### RankMe
Snippet (`metrics.py:34-42`):
```python
p = s / total
return float(np.exp(-np.sum(p * np.log(p))))
```
Interpretation:
- Effective rank via entropy.
- Higher RankMe => representation spread across more eigen-directions.

### Power-law slope (`alpha`)
Snippet (`metrics.py:44-59`):
```python
lx = np.log(x[good])
ly = np.log(y[good])
slope, _ = np.polyfit(lx, ly, deg=1)
return float(-slope)
```
Interpretation:
- Fits `log eigenvalue` vs `log rank` slope.
- Larger `alpha` => steeper decay => stronger spectral concentration.
- `alpha_head` and `alpha_tail` capture different spectrum regions.

### Covariance eigenspectrum
Snippet (`metrics.py:71-83`):
```python
svals = np.linalg.svd(x, full_matrices=False, compute_uv=False)
ev = (svals ** 2) / max(n - 1, 1)
```
Interpretation:
- Eigenvalues are from covariance of centered activations/gradients.

### Fixed vs non-fixed sampling
Snippet (`metrics.py:111-115`):
```python
if fixed_samples:
    rng = np.random.default_rng(base_seed)
else:
    rng = np.random.default_rng(base_seed + 9973 * max(step, 1))
```
Interpretation:
- Fixed sampling reduces measurement noise.
- Non-fixed sampling reveals estimator variance sensitivity.

---

## `optimizers.py`
Contains Muon optimizer used in variant runs.

Snippet (`optimizers.py:47-51`):
```python
if gg.ndim >= 2:
    mat = gg.reshape(shape[0], -1)
    mat = _orthogonalize(mat)
    gg = mat.reshape(shape)
```
Interpretation:
- Muon-style update orthogonalizes matrix-shaped gradients.
- Useful for testing whether geometry changes come from optimizer-induced update structure.

---

## `runner.py`
Training + measurement backbone used by all ablation scripts.

### Dataset caching
Snippet (`runner.py:78-81`):
```python
dataset_dir = config.output_root / "datasets" / _dataset_key(config)
if dataset_dir.exists():
    return _load_dataset_bundle(dataset_dir), dataset_dir
```
Interpretation:
- Ensures runs with same data knobs share identical dataset.

### Measurement logic
Snippet (`runner.py:156-170`):
```python
pred, h = model(zb, return_repr=True)
residual = (pred - yb)[:, None]
g_np = 2.0 * residual * h_np
act_spectrum = covariance_eigenspectrum(h_all, center=True)
grad_spectrum = covariance_eigenspectrum(g_all, center=True)
```
Interpretation:
- Activation spectrum from representation covariance.
- Gradient proxy spectrum from per-sample readout-gradient structure (`2 * residual * h`).

### Logged row schema
Snippet (`runner.py:317-336`):
```python
row = {
    "loss": latest_val_loss,
    "train_loss": train_loss,
    **measured["metrics"],
}
```
Interpretation:
- `metrics_over_time.csv` is your primary time-series artifact.
- `loss` is validation MSE at logging points; `train_loss` is instantaneous minibatch loss.

### Phase transition hooks
Snippet (`runner.py:343-350`):
```python
if step == config.transition_step:
    ... set transition_lr ...
    ... set transition_batch ...
```
Interpretation:
- Enables controlled regime shifts mid-training (used by phase ablation).

### Matched-loss utility
Snippet (`runner.py:419-424`):
```python
idx = (work["loss"] - float(target)).abs().idxmin()
row["matched_loss_error"] = abs(row["loss"] - target)
```
Interpretation:
- Lets you compare geometry at equal loss, reducing confounding from optimization progress mismatch.

---

## 3) Ablation Scripts

## `run_estimator_ablation.py`
Purpose: isolate **measurement estimator effects** (`n_samples`, fixed vs non-fixed sampling).

Key mechanism (`run_estimator_ablation.py:97-106`):
```python
for fixed_mode in (True, False):
    for n in n_list:
        meas = replace(base_cfg.measurement, n_samples=n, fixed_samples=fixed_mode)
        measured = measure_model(...)
```

Interpretation of outputs:
- File: `estimator_ablation_summary.csv`
- Look at `alpha_tail` or `rankme` vs `d_over_n` (`repr_dim / n_samples`).
- If curves drift strongly as `d_over_n` increases, your estimator is finite-sample biased/noisy.
- Gap between `fixed` and `nonfixed` quantifies sampling-induced variance.

---

## `run_noise_scale_ablation.py`
Purpose: test if spectral differences are from optimization noise scale or true geometry differences.

Two regimes (`run_noise_scale_ablation.py:107-115`):
```python
if regime == "unconstrained":
    lr = base_lr
else:
    lr = base_lr * (bsz / base_b)
```

Interpretation:
- `unconstrained`: changing `B` changes effective noise scale.
- `fixed_noise`: scales `lr` with batch to partially keep noise scale aligned.

Matched-loss JS analysis (`run_noise_scale_ablation.py:205-209`):
```python
s1 = load_cov_spectrum(...)
s2 = load_cov_spectrum(...)
divergences.append(js_divergence(s1, s2))
```

Interpretation of outputs:
- `noise_scale_matched_loss.csv`: lower `mean_js_div` means spectra are more invariant across batch settings at matched loss.
- `noise_scale_comparison.csv`: key field is `delta_js_fixed_minus_unconstrained`.
  - Negative: fixing noise scale reduced spectral divergence (supports noise-scale explanation).
  - Near zero: geometry differences are not mainly noise-scale artifacts.

---

## `run_phase_ablation.py`
Purpose: probe **phase-like dynamics** by introducing a mid-training batch/lr transition.

Regime construction (`run_phase_ablation.py:82-108` and `109-132`):
- `phase_like`: with transition step + post-transition `(B, lr)`.
- `muted`: identical setup but no transition.

Summary metrics (`run_phase_ablation.py:27-42`):
- `rankme_delta`, `rankme_sign_changes`, `alpha_tail_delta`, `grad_top10_delta`.

Interpretation of outputs:
- `phase_trajectories.csv`: inspect `rankme(step)` and `alpha_tail(step)` around transition.
- `phase_summary.csv`: compare `phase_like` vs `muted`.
- More sign changes / abrupt deltas in `phase_like` indicate regime-induced geometric phase shifts.

---

## `run_scaling_link_ablation.py`
Purpose: connect dataset scaling exponent `s` to spectral proxy (`alpha_tail`).

Scaling fit (`run_scaling_link_ablation.py:24-25`, `28-50`):
```python
def scaling_curve(d, a, s, c):
    return a * d**(-s) + c
```

Then per group fit and proxy extraction (`run_scaling_link_ablation.py:194-197`):
```python
a, s, c = fit_scaling_exponent(dvals, losses)
alpha_proxy = runs_df.loc[idx_max_d, "alpha_tail"]
```

Interpretation of outputs:
- `scaling_fit.csv`: each row gives fitted `(a, s, c)` and `alpha_proxy`.
- `scaling_correlation.json`: `pearson_r` between `s` and `alpha_proxy` (+ bootstrap CI).
- Positive stable `r` suggests steeper geometric spectra associate with stronger data-scaling behavior.

---

## `run_variant_ablation.py`
Purpose: compare Track A variants against baseline at both endpoint and matched-loss geometry.

Variant settings (`run_variant_ablation.py:43-50`) control optimizer, window, attention scaling, depth for U-Net mode.

Matched-loss baseline-vs-variant JS (`run_variant_ablation.py:210-219`):
```python
sb = load_cov_spectrum(...baseline step...)
sv = load_cov_spectrum(...variant step...)
js_divergence(sb, sv)
```

Interpretation of outputs:
- `variant_ablation_summary.csv`: endpoint metrics (loss + spectral summaries).
- `variant_ablation_matched_pairs.csv`: per seed/loss-target geometry distance vs baseline.
- `variant_ablation_matched_aggregate.csv`: averaged effect size.
- If endpoint changes are large but matched-loss JS is small, variant mainly changes optimization speed, not geometry.

---

## `run_distribution_ablation.py`
Purpose: test robustness to latent distribution and anisotropy.

Grid loops (`run_distribution_ablation.py:62-65`):
- Dist: `gaussian`, `uniform`, `student_t`
- Anisotropy: `isotropic` or `powerlaw(gamma)`

Aggregation (`run_distribution_ablation.py:135-146`):
- Means of `loss`, `rankme`, `alpha_tail`, `grad_alpha_tail` over seeds.

Interpretation of outputs:
- `distribution_ablation_summary.csv`: per-run values.
- `distribution_ablation_aggregate.csv`: condition means.
- Compare Track A vs B sensitivity under heavy-tail/anisotropic conditions to see whether geometry effects are architecture-driven or data-driven.

---

## `run_all_ablation_suite.py`
Purpose: orchestrate all ablations + plotting in one command.

Snippet (`run_all_ablation_suite.py:27-167`, `168-243`):
- `--fast` uses small smoke settings.
- default mode runs full suite.

Interpretation:
- Use this when you want reproducible end-to-end regeneration of all core tables and figures.

---

## 4) Plot Scripts

## `plot_toy_main_2x2.py`
Builds a four-panel summary figure from saved CSV/JSON files:
1. Estimator effect (`alpha_tail` vs `d/n`) (`plot_toy_main_2x2.py:37-47`)
2. Noise-scale matched-loss JS (`:51-61`)
3. Phase rank trajectories (`:65-75`)
4. Scaling-link scatter and trend (`:79-100`)

Interpretation:
- This is a compact diagnostic dashboard, not a substitute for per-ablation raw tables.

## `plot_distribution_ablation.py`
Creates per-track bar plots across data conditions.

Snippet (`plot_distribution_ablation.py:28-34`):
```python
df["condition"] = latent_dist + "|" + latent_anisotropy + "|g=" + gamma
```

Interpretation:
- Reads `distribution_ablation_aggregate.csv` and visualizes `alpha_tail_mean` + `rankme_mean` condition-by-condition.

---

## 5) Package Helper Script

## `__init__.py`
Exports convenient top-level imports (`RunConfig`, `train_toy_run`) for interactive/programmatic use.

---

## 6) How To Read Key Metrics Correctly

- `loss` (`metrics_over_time.csv`): validation MSE at logging points.
- `test_loss` (summary rows): final held-out performance.
- `rankme`: effective spectral dimensionality.
- `alpha_head`, `alpha_tail`: steepness of spectral decay in head/tail windows.
- `top10`: concentration of variance in top 10 eigen-directions.
- `grad_*` metrics: same geometry summary on gradient proxy covariance.

Practical interpretation rules:
1. Prefer **matched-loss** comparisons for geometry claims (`run_variant_ablation.py`, `run_noise_scale_ablation.py`).
2. Treat endpoint-only differences as potentially confounded by optimization speed.
3. Use multiple seeds and aggregated files when available.
4. Check `matched_loss_error` filters (especially noise-scale ablation) before trusting JS numbers.

---

## 7) What To Trust Most For Each Hypothesis

- Estimator bias/variance hypothesis: `estimator_ablation_summary.csv`.
- Noise-scale confounding hypothesis: `noise_scale_matched_loss.csv` + `noise_scale_comparison.csv`.
- Phase transition hypothesis: `phase_trajectories.csv` + `phase_summary.csv`.
- Scaling-law linkage hypothesis: `scaling_fit.csv` + `scaling_correlation.json`.
- Architecture/optimizer mechanism hypothesis: `variant_ablation_matched_aggregate.csv`.
- Data-distribution robustness hypothesis: `distribution_ablation_aggregate.csv`.

