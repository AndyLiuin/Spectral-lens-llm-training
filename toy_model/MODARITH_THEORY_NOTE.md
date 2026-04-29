# Modular-Arithmetic Theory Note

## Single-Band Regime

Consider one clean modular component with modulus `c`, offset `o`, start state `a`, and step `d`:

\[
s_k = (a + dk) \bmod c, \qquad t_k = o + s_k.
\]

The next-token rule is a shift on the cyclic group `Z_c`:

\[
(S_d f)(m) = f((m + d) \bmod c).
\]

The Fourier characters

\[
\chi_r(m) = e^{2\pi i r m / c}, \qquad r \in \{0,\dots,c-1\}
\]

are exact eigenfunctions of this operator:

\[
S_d \chi_r = e^{2\pi i r d / c} \chi_r.
\]

So in the clean single-band regime, the mathematically natural task basis is harmonic. A model that solves the task efficiently should therefore organize the latent state variable `m` into Fourier-aligned directions.

## Why The Probe Looks For Fourier Concentration

For a hidden-state conditional mean table

\[
H[m] = \mathbb{E}[h(x) \mid s = m] \in \mathbb{R}^{d_{\text{model}}},
\]

we compute the Fourier energy over the token-state axis:

\[
\widehat H_r = \frac{1}{c} \sum_{m=0}^{c-1} H[m] e^{-2\pi i r m/c},
\qquad
E_r = \lVert \widehat H_r \rVert_2^2.
\]

If feature learning is happening in the intended way, mass should move from a diffuse distribution over `r` toward a smaller set of harmonics. That is why the toy pipeline tracks quantities such as:

- `H_peak`: largest positive-frequency Fourier mass
- `H_gini`: concentration of Fourier mass
- PCA dominant frequency of `H[m]`
- causal keep/drop effects for key-frequency subspaces

## Mixed-Component Regime

With multiple components per sample,

\[
t_k^{\text{mix}} = \frac{\sum_{j=1}^m w_j (o_j + s_{j,k})}{\sum_{j=1}^m w_j},
\qquad
s_{j,k} = (a_j + d_j k) \bmod c_j,
\]

there is no single cyclic group that exactly describes the observed token sequence. The clean Fourier story becomes a local mechanistic microscope rather than a full description of the training distribution.

That is why the upgraded probe pipeline distinguishes two regimes:

- `clean_band`: isolates one cyclic component and asks whether the model has learned the expected harmonic structure.
- `matched_band`: anchors one target component but keeps nuisance components/noise sampled from the training distribution, so the probe remains in-distribution.

## Stronger Causal Claim

The old embedding-band intervention only supports the claim that the token band embedding carries useful Fourier structure.

The upgraded hidden-state probe is stronger:

1. Build `H[m]` using explicit latent state labels.
2. Compute an SVD `H = U S V^T`.
3. Identify feature-space directions whose left singular vectors are concentrated on the learned key Fourier mode.
4. Intervene on the final-layer hidden state by keeping or dropping that subspace.

If dropping the key subspace hurts more than dropping a matched-dimension control subspace, that is direct evidence that the learned representation itself, not only the embedding table, contains task-relevant harmonic features.
