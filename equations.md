# HCI — equations (code-aligned)

This file records the **notation and equations implemented in this repository**: `hci/L0.py`, `hci/L1.py`, `hci/renderer.py`, `hci/seed.py`, `train.py`, `infer.py`, `test.py`. Hyperparameters and defaults live in `params.py` (`L0`, `L1`, `SEED`, `RENDER`, `TRAIN`, …).

---

## 1. End-to-end pipeline

1. **L0** — RGB → directional differences $d_k^{\mathrm{lum}}, d_k^{\mathrm{chr}}$ (optionally **cached** across training steps when geometry and `L0.L0_DIST_CACHE_VERSION` match). Naka–Rushton with **fixed** scalars $\eta_0^{\mathrm{lum}}, \eta_0^{\mathrm{chr}}$ (`L0.ETA_LUM`, `L0.ETA_CHR`) and gain $\gamma$ (`L0.GAMMA`). Outputs include pixel $h_{2m}$, split $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$, and orientation $\theta_h = \tfrac12 \arg z_2$.
2. **L1** — $\cos^2$ hypercolumns on $h_{2m}$; **seed NR** uses a learned **scalar** $\eta_z$ on raw $\boldsymbol{\mu}$; **collinear passes** use raw-space $\beta_{\cdot}$, nonnegative $u$, then divisive NR with **spatial** $\eta^{(t)}(c)=\eta_0\cdot\sigma(\mathrm{MLP}(\bar\kappa,\bar z))$ where $\bar\kappa$ pools cosine $(\boldsymbol{\rho}\cdot\mathbf{S})/(\|\boldsymbol{\rho}\|\|\mathbf{S}\|)$ with per-bin pools $\tilde S_k = G_k * \rho_k$ (**unnormalized** kernel sums), and $\bar z$ pools normalized $\sum_k u_k$. Diagnostic $\kappa$ is **not** multiplied into $\rho$; it feeds the renderer MLP as $\bar\kappa_{\mathrm{col}}$.
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

## 3. L1 — hypercolumn binning (raw oriented mass)

Patch size $P$, stride $S = P - \texttt{patch\_overlap}$ → cell grid $(n_H, n_W)$, $K$ bins (`L1.COL_K_BINS`), bin centres $\bar\theta_k = k\pi/K$.

**Oriented energy** into bin $k$ at cell $c$ (sum over pixels in the patch):

$$
\rho_k^{\mathrm{raw}}(c) = \sum_{p \in \mathrm{patch}(c)} h_{2m}(p)\,\cos^2\!\bigl(\theta_h(p) - \bar\theta_k\bigr).
$$

Border cells zero $\rho_k^{\mathrm{raw}}$. **Cell total oriented energy** (used downstream / diagnostics):

$$
z_0(c) = \sum_{k=0}^{K-1} \rho_k^{\mathrm{raw}}(c).
$$

Take $\mu_k(c) = \rho_k^{\mathrm{raw}}(c)$ as the nonnegative **drive** for the seed NR in §4 (no sigmoid / logit layer on this grid).

---

## 4. GABA — raw-space collinear recurrence + scalar seed $\eta_z$ + spatial $\eta$ (passes)

Learned **scalar** $\eta_z = \mathrm{softplus}(\tilde\eta_z)$ (`HypercolumnSeed._eta_z_raw`) controls **only** the initial divisive NR on raw $\boldsymbol{\mu}$ — **no** MLP and **no** $\kappa$/$z$ inputs to $\eta$ at that step. **At initialization**, $\eta_z$ is set equal to $\eta_0$ (same positive value from `SEED.ETA0_INIT` unless `eta_z_init` is passed); $\tilde\eta_z$ and $\tilde\eta_0$ are still **separate** parameters and may diverge during training.

From the **first collinear pass onward**, a learned $\eta_0 = \mathrm{softplus}(\tilde\eta_0)$ scales a **per-cell** modulation from a **2→8→1** MLP (`EtaGabaMLP`; sigmoid output clamped to $(10^{-3},1]$ in code).

Let $\boldsymbol{\rho}^{(t)}(c)\in\mathbb{R}_+^K$ be the bin vector at the **start** of recurrence pass $t$ (the post–seed-NR grid for $t{=}0,1,\ldots$ inside the loop). Let $\mathbf{S}^{(t)}(c)\in\mathbb{R}_+^K$ stack the collinear pool outputs $\tilde S_k^{(t)}(c)$ **before** the $\beta$-mixture that forms $u$ (depthwise conv with **unnormalized** nonnegative kernels — a weighted **sum** over neighbors, not divided by $\sum G_k$). **Surround inhibition** is **per bin** $k$: at each cell, form the leave-one-out mean of the *other* orientations $\bar Z_k^{(t)}(c)=\frac{1}{\max(K-1,1)}\sum_{j\neq k}\rho_j^{(t)}(c)$, then apply the **same radial** Gaussian × disk as $G_k$ (center omitted), **without** tangential selectivity — i.e.\ $\mathcal{I}_k^{(t)}(c)=(H*\bar Z_k^{(t)})(c)$ with isotropic neighbor pooling on the competing-orientation field (bin $k$ does not feed its own surround). Define

$$
\kappa^{(t)}(c) = \frac{\boldsymbol{\rho}^{(t)}(c)\cdot \mathbf{S}^{(t)}(c)}{\bigl\|\boldsymbol{\rho}^{(t)}(c)\bigr\|\,\bigl\|\mathbf{S}^{(t)}(c)\bigr\| + \varepsilon}.
$$

After $u_k^{(t)}$ is formed (see below), the second MLP input channel uses

$$
z^{(t)}(c) = \sum_{k=0}^{K-1} u_k^{(t)}(c).
$$

Mean-pool over a $(2r_\eta+1)\times(2r_\eta+1)$ **cell** neighborhood (`L1.GABA_ETA_POOL_RADIUS` = $r_\eta$; default $r_\eta=10$ → $21\times21$), with border cells zeroed before pooling (`regional_mean_pool_cells` in `hci/L0.py`):

$$
\bar\kappa_c = \mathrm{meanpool}_{(2r_\eta+1)^2}\bigl(\kappa^{(t)}\bigr), \qquad
\bar z_c = \frac{\mathrm{meanpool}_{(2r_\eta+1)^2}\bigl(z^{(t)}\bigr)}{\max_{c'}\mathrm{meanpool}_{(2r_\eta+1)^2}\bigl(z^{(t)}\bigr) + \varepsilon}.
$$

**Per-cell NR scale (passes only)**:

$$
\eta^{(t)}(c) = \eta_0 \cdot \sigma\bigl(\mathrm{MLP}([\bar\kappa_c,\, \bar z_c])\bigr).
$$

Learned tangent / normal **kernel scales** (HCI-style scaling by collinear radius $R =$ `L1.COL_RADIUS`):

$$
\sigma_d = \mathrm{softplus}(\tilde\alpha_d)\,R, \qquad
\sigma_t = \mathrm{softplus}(\tilde\alpha_t)\,R.
$$

For each bin $k$, a nonnegative kernel $G_k$ is built on the $(2R+1)^2$ patch (Gaussian × tangential selectivity, center omitted, disk clip). Let $H$ be the **radial** factor alone (Gaussian × disk, center omitted — the isotropic weight shared with all $G_k$). **Unnormalized depthwise convolution** for collinear pools (same as the ``e_col`` diagnostic conv):

$$
\tilde S_k^{(t)}(c) = (G_k * \rho_k^{(t)})(c), \qquad
\bar Z_k^{(t)}(c) = \frac{1}{\max(K-1,1)}\sum_{j\neq k} \rho_j^{(t)}(c), \qquad
\mathcal{I}_k^{(t)}(c) = (H * \bar Z_k^{(t)})(c).
$$

The inhibitory drive $\mathcal{I}_k^{(t)}$ is **per orientation channel** (depthwise isotropic conv on the LOO mean map $\bar Z_k^{(t)}$).

**Raw-space β** (softplus of raw parameters):

$$
\beta_{\mathrm{seed}} = \mathrm{softplus}(\tilde\beta_s),\quad
\beta_{\mathrm{coll}} = \mathrm{softplus}(\tilde\beta_c),\quad
\beta_{\mathrm{cross}} = \mathrm{softplus}(\tilde\beta_x).
$$

**Seed NR.** Let $\mu_k(c)=\rho_k^{\mathrm{raw}}(c)$ after border masking. With learned scalar $\eta_z>0$,

$$
\rho_k^{\mathrm{seed}}(c) = \frac{\mu_k(c)^2}{\mu_k(c)^2 + \eta_z^2 + \varepsilon}.
$$

Detach $\rho^{\mathrm{seed}}$ for the fixed $\beta_{\mathrm{seed}}$ branch; keep nondetached copy for the conv chain. `rho_k_initial` snapshots this post–seed-NR state.

**Recurrence** $t = 0,\ldots,T-1$ (`L1.COL_PASSES`). Border cells zero $u$ and $\rho$ after each map.

$$
u_k^{(t)}(c) = \max\Bigl(0,\;
\beta_{\mathrm{seed}}\,\rho_k^{\mathrm{seed}}(c)
+ \beta_{\mathrm{coll}}\,\tilde S_k^{(t)}(c)
- \beta_{\mathrm{cross}}\,\mathcal{I}_k^{(t)}(c)
\Bigr),
$$

with $\kappa^{(t)}$ from $(\boldsymbol{\rho}^{(t)},\mathbf{S}^{(t)})$ and $z^{(t)}(c)=\sum_k u_k^{(t)}(c)$, pool → MLP → $\eta^{(t)}(c)$,

$$
\rho_k^{(t+1)}(c) = \frac{\bigl(u_k^{(t)}(c)\bigr)^2}{\bigl(u_k^{(t)}(c)\bigr)^2 + \bigl(\eta^{(t)}(c)\bigr)^2 + \varepsilon}.
$$

**Diagnostics.** `kappa_pass0_cell` / `kappa_col_cell` store $\kappa^{(t)}$ (cosine sim) after the first / last pass’s $(\boldsymbol{\rho},\mathbf{S})$ pair — same quantity fed to the η-MLP (before pooling). Not multiplied into $\rho$ in the recurrence.

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

plus collinear energy $e_{\mathrm{col}}$ at the dominant bin, initial $\rho$ at $b^*$ (NR seed snapshot at the dominant bin), etc., as produced by `run_l1_hypercolumn` and flattened in `build_cells_flat`.

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
- **Each step** (`prepare_batch` → `run_l1_live_cells` → `HarmonicContourE2E.forward_batch`): L1 runs **live** with `cells_format="torch"` so gradients reach `HypercolumnSeed` ($\tilde\eta_0$, GABA η-MLP, $\tilde\beta_{\cdot}$, $\tilde\alpha_{d,t}$). `RhoSeedModule` then reads dominant $\rho$; the renderer consumes the same cached L0 pixel tensors (no interleaved L0 recompute inside GABA).

**Loss** (defaults in `params.TRAIN`): weighted sum of soft-Dice and/or BCE on the valid edge band vs.\ ground-truth boundaries.

---

## 8. Learned parameter count (current architecture)

| Block | Count (approx.) | Notes |
|------:|----------------:|------|
| $\tilde\alpha_d, \tilde\alpha_t$ | 2 | Collinear kernel scale ratios → $\sigma_d,\sigma_t$ |
| $\tilde\eta_z$ | 1 | Scalar seed NR (``softplus``) |
| $\tilde\eta_0$ | 1 | Base scale in $\eta=\eta_0\cdot\sigma(\mathrm{MLP})$ on passes |
| $\mathrm{MLP}_{\eta}$ (2→8→1) | 33 | $2\cdot 8 + 8 + 8 + 1$ |
| $\tilde\beta_{\mathrm{seed}}, \tilde\beta_{\mathrm{coll}}, \tilde\beta_{\mathrm{cross}}$ | 3 | Raw-space GABA (clamp-$u$, NR squash) |
| $s_t, s_n$ | 2 | Stencil spacing |
| $\mathrm{MLP}_{\mathrm{read}}$ (14→8→1) | 129 | $14\cdot 8 + 8 + 8 + 1$ |
| **Total (E2E)** | **171** | `HarmonicContourE2E` |

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

This document matches **raw-space β** GABA with **scalar** seed $\eta_z$, then **spatial** $\eta=\eta_0\cdot\sigma(\mathrm{MLP}(\bar\kappa,\bar z))$ on passes where $\bar\kappa$ pools cosine alignment and $\bar z$ pools $\sum_k u_k$, **unnormalized** collinear kernel sums $\tilde S_k=G_k*\rho_k$, **14-dimensional** readout, and removal of the **regional η-MLP** from the training and default inference paths. If `HCI_SYSTEM_EQUATIONS.md` exists elsewhere in the project tree, treat it as design narrative unless it explicitly matches this file’s section numbering.
