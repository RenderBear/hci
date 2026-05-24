# HCI — equations (code-aligned)

This file records the **notation and equations implemented in this repository**: `hci/L0.py`, `hci/L1.py`, `hci/renderer.py`, `hci/seed.py`, `train.py`, `infer.py`, `test.py`. Hyperparameters and defaults live in `params.py` (`L0`, `L1`, `SEED`, `RENDER`, `TRAIN`, …).

---

## 1. End-to-end pipeline

1. **L0** — RGB → directional differences $d_k^{\mathrm{lum}}, d_k^{\mathrm{chr}}$ (optionally **cached** across training steps when geometry and `L0.L0_DIST_CACHE_VERSION` match). Naka–Rushton with **fixed** scalars $\eta_0^{\mathrm{lum}}, \eta_0^{\mathrm{chr}}$ (`L0.ETA_LUM`, `L0.ETA_CHR`) and gain $\gamma$ (`L0.GAMMA`). Outputs include pixel $h_{2m}$, split $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$, and orientation $\theta_h = \tfrac12 \arg z_2$.
2. **L1** — $\cos^2$ hypercolumns on $h_{2m}$; **divisive NR** on raw bin energies with learned $\eta_z$ (**pre-GABA**); **logit-space GABA** on $K$ bins with learned $\beta_{\mathrm{seed}}, \beta_{\mathrm{coll}}, \beta_{\mathrm{cross}}$; **dominant-bin** scalars and diagnostics for the cell grid.
3. **Seed module** — `RhoSeedModule` passes dominant $\rho$ from `cells_flat["lam"][...,0]` through to the renderer (no extra recurrent dynamics on the seed tensor).
4. **Renderer** — θ combing on cells, bilinear interp of $\bar\rho, \bar\theta, \bar\kappa_{\mathrm{col}}$ to pixels, tangential / normal **stencils** on $h_{2m}^{\mathrm{lum}}+h_{2m}^{\mathrm{chr}}$, **14→8→1** thinning MLP gate:
   $$\hat B(p) = \bigl(h_{2m}^{\mathrm{lum}}(p)+h_{2m}^{\mathrm{chr}}(p)\bigr)\,\bar\rho(p)\,\mathrm{gate}(p).$$

There is **no** second L0 pass and **no** regional $\eta$-MLP inside the training or inference graphs documented here; $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$ at render time are those from the initial L0 forward.

---

## 2. L0 — split luminance / chrominance harmonics

Eight offsets $\delta_k \in \mathbb{Z}^2$ (`L0.OFFSETS`). RGB is mapped to orthonormal luminance $L$ and chrominance $C$.

**Directional differences** (magnitude on $L$, $\ell_2$ on $C$):

$$
d_k^{\mathrm{lum}}(p) = \bigl|L(p) - L(p+\delta_k)\bigr|, \qquad
d_k^{\mathrm{chr}}(p) = \bigl\|C(p) - C(p+\delta_k)\bigr\|_2 .
$$

**Per-direction min subtraction** (index $j$ runs over directions):

$$
\tilde d_k^{\mathrm{lum}} = d_k^{\mathrm{lum}} - \min_j d_j^{\mathrm{lum}}, \qquad
\tilde d_k^{\mathrm{chr}} = d_k^{\mathrm{chr}} - \min_j d_j^{\mathrm{chr}} .
$$

**Naka–Rushton** with fixed $\eta_0^{\mathrm{lum}}, \eta_0^{\mathrm{chr}}$ and denominator floor in code so flat regions stay numerically stable:

$$
h_k^{\mathrm{lum}} = \gamma\,\frac{(\tilde d_k^{\mathrm{lum}})^2}{(\eta_0^{\mathrm{lum}})^2 + (\tilde d_k^{\mathrm{lum}})^2}, \qquad
h_k^{\mathrm{chr}} = \gamma\,\frac{(\tilde d_k^{\mathrm{chr}})^2}{(\eta_0^{\mathrm{chr}})^2 + (\tilde d_k^{\mathrm{chr}})^2}.
$$

**Second harmonic** (complex $z_2$, magnitude $h_{2m}$, orientation $\theta_h$):

$$
z_2(p) = \sum_k \bigl(h_k^{\mathrm{lum}} + h_k^{\mathrm{chr}}\bigr)\, e^{2i\varphi_k}, \qquad
h_{2m}(p) = |z_2(p)|, \qquad
\theta_h(p) = \tfrac12 \arg z_2(p).
$$

Pixel fields $h_{2m}^{\mathrm{lum}}(p)$, $h_{2m}^{\mathrm{chr}}(p)$ are the magnitudes of the lum-only and chr-only harmonic sums (`compute_l0_rgb`).

---

## 3. L1 — hypercolumn binning and pre-GABA NR

Patch size $P$, stride $S = P - \texttt{patch\_overlap}$ → cell grid $(n_H, n_W)$, $K$ bins (`L1.COL_K_BINS`), bin centres $\bar\theta_k = k\pi/K$.

**Oriented energy** into bin $k$ at cell $c$ (sum over pixels in the patch):

$$
\rho_k^{\mathrm{raw}}(c) = \sum_{p \in \mathrm{patch}(c)} h_{2m}(p)\,\cos^2\!\bigl(\theta_h(p) - \bar\theta_k\bigr).
$$

Border cells zero $\rho_k^{\mathrm{raw}}$. **Cell total oriented energy** (used downstream / diagnostics):

$$
z_0(c) = \sum_{k=0}^{K-1} \rho_k^{\mathrm{raw}}(c).
$$

**Divisive NR vs.\ learned $\eta_z$** on raw bin energies (single scalar per module, $\eta_z = \mathrm{softplus}(\tilde\eta_z)$; seed eps $\varepsilon$):

$$
\rho_k^{(0)}(c) = \frac{\bigl(\rho_k^{\mathrm{raw}}(c)\bigr)^2}{\bigl(\rho_k^{\mathrm{raw}}(c)\bigr)^2 + \eta_z^2 + \varepsilon}.
$$

This $\rho_k^{(0)}$ is the **input** to GABA. A detached copy is stored as pre-GABA / “initial” $\rho$ for visualization.

---

## 4. GABA — logit-space collinear recurrence

Learned tangent / normal **kernel scales** (HCI-style scaling by collinear radius $R =$ `L1.COL_RADIUS`):

$$
\sigma_d = \mathrm{softplus}(\tilde\alpha_d)\,R, \qquad
\sigma_t = \mathrm{softplus}(\tilde\alpha_t)\,R.
$$

For each bin $k$, a nonnegative kernel $G_k$ is built on the $(2R+1)^2$ patch: Gaussian falloff in pixel distance, modulated by tangential selectivity at orientation $\bar\theta_k$, center omitted, support clipped to the disk of radius $R$. Stacking $K$ such kernels gives depthwise weights.

**Normalized depthwise convolution** so each output is a **weighted average** of neighbor $\rho$ in $[0,1]$ (implementation divides by kernel sums with a floor):

$$
\tilde S_k^{(t)}(c) = \frac{(G_k * \rho_k^{(t)})(c)}{\sum_{u,v} G_k(u,v) + \varepsilon_{\mathrm{ker}}}.
$$

Clamped to $(10^{-4},\,1-10^{-4})$ before logits. **Cross-orientation mean** (same for all $k$ at a cell):

$$
\bar S_k^{(t)}(c) = \frac{1}{K}\sum_{j=0}^{K-1} \tilde S_j^{(t)}(c).
$$

**Logit-space β weights** (all positive via softplus of raw parameters):

$$
\beta_{\mathrm{seed}} = \mathrm{softplus}(\tilde\beta_s),\quad
\beta_{\mathrm{coll}} = \mathrm{softplus}(\tilde\beta_c),\quad
\beta_{\mathrm{cross}} = \mathrm{softplus}(\tilde\beta_x).
$$

**Seed logits** are fixed for the whole recurrence from the NR seed grid $\rho^{(0)}$ (after border masking), clamped and mapped with $\mathrm{logit}$.

**One recurrence step** $t = 0,\ldots,T-1$ (`L1.COL_PASSES`):

$$
\ell_k^{(t)} =
\beta_{\mathrm{seed}}\,\mathrm{logit}\bigl(\rho_k^{\mathrm{seed}}\bigr)
+ \beta_{\mathrm{coll}}\,\mathrm{logit}\bigl(\tilde S_k^{(t)}\bigr)
- \beta_{\mathrm{cross}}\,\mathrm{logit}\bigl(\bar S_k^{(t)}\bigr),
$$

$$
\rho_k^{(t+1)}(c) = \sigma\bigl(\ell_k^{(t)}(c)\bigr),
$$

again zeroing border cells. The tensor fed into the first pass is $\rho_k^{(0)}$ from §3; after $T$ passes, write $\rho_k^{(T)}$ for the final bin masses.

**Pass-0 diagnostic** (per bin, stored for viz): $\tilde S_k^{(0)}$ — collinear pool output after the first pass’s conv, **before** the logit update (gathered at the **final** dominant bin downstream for `kappa_pass0_cell`).

**Post-hoc κ (per bin, after all passes)** — peakedness of the final profile (not a recurrence gate):

$$
\kappa_k(c) = \mathrm{clip}_{[0,1]}\left(
\frac{\max_j \rho_j^{(T)}(c)}{\frac{1}{K}\sum_j \rho_j^{(T)}(c) + \varepsilon}
\right)
$$

(extended to all $k$ as the same scalar row in code, then gathered at the dominant bin for the scalar $\kappa_{\mathrm{col}}$ passed to the renderer).

An optional callback `eta_update_fn` remains in `gaba_recurrence` for API compatibility; **training and bundled infer paths pass `None`.**

---

## 5. Post-recurrence dominant channel

Dominant bin:

$$
b^*(c) = \arg\max_{k} \rho_k^{(T)}(c).
$$

**Cached / renderer-facing scalars** per cell (modulo border and tile-interior masking in code):

$$
\rho_{\mathrm{dom}}(c) = \rho_{b^*}^{(T)}(c), \qquad
\theta(c) = \bigl(\bar\theta_{b^*},\, \bar\theta_{b^*}\bigr), \qquad
\kappa_{\mathrm{col}}(c) = \kappa_{b^*}(c), \qquad
\rho_{\max}(c) = \max_k \rho_k^{(T)}(c),
$$

plus collinear energy $e_{\mathrm{col}}}$ at the dominant bin, initial $\rho$ at $b^*$, etc., as produced by `run_l1_hypercolumn` and flattened in `build_cells_flat`.

---

## 6. Renderer — interp, stencils, readout MLP

**θ combing** on the cell grid (double-angle, ρ-weighted smoothing, `RENDER.THETA_SMOOTH_PASSES`), then **bilinear** sampling of stacked cell fields to pixels → $\bar\rho(p)$, $\bar\theta(p)$, $\bar\kappa_{\mathrm{col}}(p)$. Interpolated orientation used for stencil geometry may be **stopped** for autograd where noted in `hci/renderer.py`.

**Stencils** on $h_{2m}^{\Sigma} = h_{2m}^{\mathrm{lum}} + h_{2m}^{\mathrm{chr}}$ with unit tangent $\hat t = (\cos\bar\theta,\sin\bar\theta)$ and normal $\hat n = (-\sin\bar\theta,\cos\bar\theta)$, learned spacings $s_t, s_n = \mathrm{softplus}(\cdot)$:

$$
\mathrm{tang}_j(p) = h_{2m}^{\Sigma}\bigl(p + j\, s_t\, \hat t(p)\bigr), \qquad
\mathrm{norm}_j(p) = h_{2m}^{\Sigma}\bigl(p + j\, s_n\, \hat n(p)\bigr), \qquad j \in \{-2,-1,0,1,2\}.
$$

**Feature vector** $F_p \in \mathbb{R}^{14}$ (no separate $\eta$ modulation channel):

$$
F_p = \bigl[
h_{2m}^{\mathrm{lum}},\, h_{2m}^{\mathrm{chr}},\, \bar\rho,\, \bar\kappa_{\mathrm{col}},\,
\mathrm{tang}_{-2},\ldots,\mathrm{tang}_{2},\,
\mathrm{norm}_{-2},\ldots,\mathrm{norm}_{2}
\bigr].
$$

**Boundary map**:

$$
\mathrm{gate}(p) = \sigma\bigl(\mathrm{MLP}_{\mathrm{read}}(F_p)\bigr), \qquad
\hat B(p) = h_{2m}^{\Sigma}(p)\,\bar\rho(p)\,\mathrm{gate}(p).
$$

$\mathrm{MLP}_{\mathrm{read}}$: **14 → 8 → 1** (ReLU between linear layers). Checkpoints that stored a **15**-input first layer can be loaded via `upgrade_renderer_state_dict`, which drops the last input column of `thinning.fc1.weight` to match the current head.

---

## 7. Training (`train.py`)

- **Disk cache** (`precompute_image`): L0 and geometry only; stores `h2m`, `theta_h`, `border_mask`, `l0_pix`, optional `d_lum` / `d_chr`, GT, projection metadata — **not** `cells_flat`. Invalidation: `TRAIN.CACHE_VERSION`. Directional tensors may be reused when `L0.L0_DIST_CACHE_VERSION` and the stored signature match.
- **Each step** (`prepare_batch` → `run_l1_live_cells` → `HarmonicContourE2E.forward_batch`): L1 runs **live** with `cells_format="torch"` so gradients reach `HypercolumnSeed` ($\tilde\eta_z$, $\tilde\beta_{\cdot}$, $\tilde\alpha_{d,t}$). `RhoSeedModule` then reads dominant $\rho$; the renderer consumes the same cached L0 pixel tensors (no interleaved L0 recompute inside GABA).

**Loss** (defaults in `params.TRAIN`): weighted sum of soft-Dice and/or BCE on the valid edge band vs.\ ground-truth boundaries.

---

## 8. Learned parameter count (current architecture)

| Block | Count (approx.) | Notes |
|------:|----------------:|------|
| $\tilde\alpha_d, \tilde\alpha_t$ | 2 | Collinear kernel scale ratios → $\sigma_d,\sigma_t$ |
| $\tilde\eta_z$ | 1 | Pre-GABA NR |
| $\tilde\beta_{\mathrm{seed}}, \tilde\beta_{\mathrm{coll}}, \tilde\beta_{\mathrm{cross}}$ | 3 | Logit GABA |
| $s_t, s_n$ | 2 | Stencil spacing |
| $\mathrm{MLP}_{\mathrm{read}}$ (14→8→1) | 129 | $14\cdot 8 + 8 + 8 + 1$ |
| **Total (E2E)** | **137** | `HarmonicContourE2E` |

Legacy checkpoints may still contain `eta_mlp.*` or `seed.hc_seed._gaba_alpha_raw`; `remap_checkpoint_state_dict` strips those keys so `load_state_dict(..., strict=False)` stays clean.

---

## 9. Module map

| Stage | Primary code |
|-------|----------------|
| $d_k$, NR, harmonics, interior mask | `hci/L0.py` |
| Hypercolumns, GABA recurrence, dominant extract, `HypercolumnSeed` | `hci/L1.py` |
| `RhoSeedModule` wrapper | `hci/seed.py` |
| Interp, stencils, thinning | `hci/renderer.py` |
| Cache, batching, loss, checkpoint remap | `train.py` |
| Single-image pipeline, viz hooks | `infer.py`, `hci/diagnostics_viz.py` |
| Test-set sweep | `test.py` |

---

## 10. Revision note

This document was rewritten to match the **logit β GABA** recurrence, **normalized** collinear kernels, **peakedness** κ, **14-dimensional** readout, and removal of the **regional η-MLP** from the training and default inference paths. If `HCI_SYSTEM_EQUATIONS.md` exists elsewhere in the project tree, treat it as design narrative unless it explicitly matches this file’s section numbering.
