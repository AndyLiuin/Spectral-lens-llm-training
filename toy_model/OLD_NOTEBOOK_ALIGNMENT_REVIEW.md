# Old Notebook Alignment Review

## Scope Checked

Compared `old_notebooks/*.ipynb` against the current packaged pipeline (`data.py`, `models.py`, `runner.py`).

## Verdict

The current implementation is **not faithful** to the old notebooks' dataset/task.

It is faithful to the newer "toy RFF regression" plan, but it is a different experimental regime from the old notebook setup.

## Key Mismatches

## 1) Dataset family is different

Old notebooks:
- Use `BandedModArithDataset` in multiple notebooks (`rope_test.ipynb`, `muon_test.ipynb`, `fixed_window_test.ipynb`, `squared_norm_test.ipynb`, `fp8lmhead_test.ipynb`).
- Data generation samples `(c, o, d)` and builds modular arithmetic token sequences:
  - `s = (a + d * k) % c`
  - `t = o + s`
  - returns `t[:-1], t[1:]` (next-token prediction).

Current package:
- Uses continuous latent sampling + frozen RFF teacher regression (`data.py`).
- Labels are scalar regression targets from pooled RFF features:
  - `phi = rff_features_numpy(...)`
  - `pooled = phi.mean(axis=1)`
  - `y = pooled @ teacher_a`.

## 2) Objective/loss is different

Old notebooks:
- Language-model style token prediction with CE loss (`F.cross_entropy(...)`) and `lm_head` (weight tying appears in notebook model code).

Current package:
- Scalar regression with MSE in both eval and train (`runner.py`, `F.mse_loss(...)`).

## 3) Output semantics differ

Old notebooks:
- Spectra derived from transformer hidden activations in token modeling runs.
- Tracks dual alpha ranges on normalized spectra over training time/tokens.

Current package:
- Spectra from representation covariance and a gradient proxy in RFF regression.
- Metrics and conclusions are specific to the RFF-teacher setup.

## Why this happened

The packaged code is a clean, reproducible ablation framework for the RFF two-track setup, while the notebooks are from a modular-arithmetic LM phase. Both are valid, but they answer different questions.

## Recommendation

If your target is to stay close to old experiments, we should **borrow directly from notebooks** and add a compatibility path.

## Practical plan (recommended)

1. Add a `task` switch to `RunConfig`: `task in {"rff_regression", "mod_arith_lm"}`.
2. Implement `data_modarith.py` with `BandedModArithDataset` (ported from notebooks).
3. Implement `models_lm.py` with LM head + CE objective.
4. Keep shared logging/metrics runner plumbing, but branch loss/model/data by task.
5. Keep variant toggles (`rope`, `muon`, `fixed_window`, etc.) in both tasks when meaningful.
6. Label outputs by task so analyses are not mixed.

## Bottom line

Your concern is correct: the current dataset construction does not match the old notebook dataset. We can and should borrow from the notebooks if alignment to original behavior is the priority.

