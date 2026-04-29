# Language-Model Experiment Scripts

This directory contains the scripts needed to reproduce the language-model
training and spectral-extraction pipeline.

## Directory map

- `d12_training/`: final constant-loss training scripts for the 12-layer model
  families in the paper.
- `d12_lr_sweep/`: learning-rate sweep scripts for the 12-layer families.
- `d12_spectrum_analysis/`: activation/gradient/weight spectrum extraction for
  12-layer checkpoints.
- `scale_training/`: scaled d36/d48 training scripts.
- `scale_lr_sweep/`: learning-rate sweeps for scaled runs.
- `scale_spectrum_analysis/`: spectrum extraction for scaled checkpoints.
- `docs/`: historical experiment notes summarizing the intervention chain and
  scale-run procedure.

## Reproduction order

1. Sweep learning rates for each model family and batch tier.
2. Train the selected configuration to the target validation loss.
3. Run the corresponding spectrum-analysis script on saved checkpoints.
4. Copy metrics and spectra into the `paper_figures/data/` layout described in
   `../data/README.md`.

The d12 scripts cover the baseline/RoPE/Muon/Untied prefix and the long-context
intervention chain through LSWA/attention-scale variants. The scaled scripts
cover the d36/d48 robustness runs used in the current draft.

