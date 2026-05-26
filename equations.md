# HCI — equations (code-aligned)

This file records the **notation and equations implemented in this repository**: `hci/L0.py`, `hci/L1.py`, `hci/renderer.py`, `hci/seed.py`, `train.py`, `infer.py`, `test.py`. Hyperparameters and defaults live in `params.py` (`L0`, `L1`, `SEED`, `RENDER`, `TRAIN`, …).

---

## 1. End-to-end pipeline

1. **L0** — RGB → directional differences $d_k^{\mathrm{lum}}, d_k^{\mathrm{chr}}$ (optionally **cached** across training steps when geometry and `L0.L0_DIST_CACHE_VERSION` match). Naka–Rushton with **fixed** scalars $\eta_0^{\mathrm{lum}}, \eta_0^{\mathrm{chr}}$ (`L0.ETA_LUM`, `L0.ETA_CHR`) and gain $\gamma$ (`L0.GAMMA`). Outputs include pixel $h_{2m}$, split $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$, and orientation $\theta_h = \tfrac12 \arg z_2$.
2. **L1** — $\cos^2$ hypercolumns on $h_{2m}$; **seed NR** with learned $\eta_z$ maps raw $\boldsymbol{\mu}$ to $\boldsymbol{\rho}^{\mathrm{seed}}\in[0,1]^K$; then $T$ **pass NR** steps with kernel-normalized collinear / flank / cross pools, seed-gated excitatory drive $\rho^{\mathrm{seed}}(\beta_{\mathrm{seed}}+\beta_c\mathbf{s}_{\mathrm{coll}})$, and divisive update $\boldsymbol{\rho}\leftarrow \mathrm{drive}^2/(\mathrm{drive}^2+\eta_p^2+\beta_f\mathbf{s}_{\mathrm{flank}}^2+\beta_x\mathbf{s}_{\mathrm{cross}}^2+\varepsilon)$ with learned scalars $\eta_z,\eta_p,\beta_{\cdot}$ and kernel scales $\sigma_d,\sigma_t,\sigma_{\mathrm{iso}}$. Diagnostic $\kappa$ (cosine alignment of $\boldsymbol{\rho}$ vs raw collinear conv) is **not** multiplied into $\rho$; the renderer reads $\bar\kappa_{\mathrm{col}}$ from the dominant bin.
3. **Seed module** — `RhoSeedModule` passes dominant $\rho$ from `cells_flat["lam"][...,0]` through to the renderer (no extra recurrent dynamics on the seed tensor).
4. **Renderer** — θ combing on cells, bilinear interp of $\bar\rho, \bar\theta, \bar\kappa_{\mathrm{col}}$ to pixels, tangential / normal **stencils** on $h_{2m}^{\mathrm{lum}}+h_{2m}^{\mathrm{chr}}$, **14→8→1** thinning MLP gate ($\bar\rho$ is feature index 2, not multiplied out):
   $$\hat B(p) = \bigl(h_{2m}^{\mathrm{lum}}(p)+h_{2m}^{\mathrm{chr}}(p)\bigr)\,\mathrm{gate}(p).$$

There is **no** second L0 pass, **no** spatial $\eta$-MLP, and **no** regional $\eta$ modulation inside the training or inference graphs; $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$ at render time are those from the initial L0 forward.

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

Take $\mu_k(c) = \rho_k^{\mathrm{raw}}(c)$. **Seed NR** (once, before passes) compresses raw drive to $[0,1]$ so it matches the normalized pools in §4:

$$
\rho_k^{\mathrm{seed}}(c) = \frac{\mu_k(c)^2}{\mu_k(c)^2 + \eta_z^2 + \varepsilon}, \qquad
\eta_z = \mathrm{softplus}(\tilde\eta_z).
$$

Recurrence starts from $\boldsymbol{\rho}^{(0)}=\boldsymbol{\rho}^{\mathrm{seed}}$. `rho_k_initial` in `cells` stores **post–seed-NR** $\boldsymbol{\rho}^{\mathrm{seed}}$.

---

## 4. GABA — seed NR + three-pool pass recurrence

Let $\boldsymbol{\rho}^{(t)}(c)\in\mathbb{R}_+^K$ be the bin vector at the **start** of pass $t$ ($\boldsymbol{\rho}^{(0)}=\boldsymbol{\rho}^{\mathrm{seed}}$ from §3). $\boldsymbol{\rho}^{\mathrm{seed}}$ is **fixed** for all passes. Border cells zero $\boldsymbol{\rho}$ after each map.

### Learned scalars (`HypercolumnSeed`, softplus of raw parameters)

$$
\eta_z = \mathrm{softplus}(\tilde\eta_z), \quad
\eta_p = \mathrm{softplus}(\tilde\eta_p), \quad
\beta_{\mathrm{seed}} = \mathrm{softplus}(\tilde\beta_{\mathrm{seed}}), \quad
\beta_c = \mathrm{softplus}(\tilde\beta_c), \quad
\beta_f = \mathrm{softplus}(\tilde\beta_f), \quad
\beta_x = \mathrm{softplus}(\tilde\beta_x).
$$

Defaults in `params.SEED`: `ETA_Z`, `ETA_P`, `BETA_SEED`, `BETA_C`, `BETA_F`, `BETA_X`.

### Kernel scales ($R =$ `L1.COL_RADIUS`)

$$
\sigma_d = \mathrm{softplus}(\tilde\alpha_d)\,R, \qquad
\sigma_t = \mathrm{softplus}(\tilde\alpha_t)\,R, \qquad
\sigma_{\mathrm{iso}} = \mathrm{softplus}(\tilde\alpha_{\mathrm{iso}})\,R.
$$

On the $(2R+1)^2$ patch (disk clip, center omitted), radial Gaussian $w_d(r)=\exp(-r^2/2\sigma_d^2)$ and oriented envelope $\mathrm{gauss}(r,\sigma_d,\sigma_t)$ use offset angle $\phi=\mathrm{atan2}(d_i,d_j)$ ($d_j$ horizontal, $d_i$ vertical). The tangential Gaussian suppresses neighbors **perpendicular** to bin axis $\bar\theta_k=k\pi/K$ via $d_\perp = d_i\cos\bar\theta_k + d_j\sin\bar\theta_k$ (at $\bar\theta_k=0$, $d_\perp=d_i$ kills above/below, preserving the horizontal strip). Then:

$$
G^{\mathrm{coll}}_k = \mathrm{gauss}(r,\sigma_d,\sigma_t)\,\cos^2(\phi-\bar\theta_k), \qquad
G^{\mathrm{flank}}_k = \mathrm{gauss}(r,\sigma_d,\sigma_t)\,\sin^2(\phi-\bar\theta_k).
$$

**Cross-orientation** pool uses a single isotropic kernel (no angular weighting):

$$
G^{\mathrm{cross}} = \mathrm{gauss}(r,\sigma_{\mathrm{iso}}).
$$

### Kernel-normalized depthwise convolution

For nonnegative kernel $G$ (per bin $k$ where applicable):

$$
\hat s_k^{(t)}(c) = \frac{(G_k * \rho_k^{(t)})(c)}{(G_k * \mathbf{1})(c) + \varepsilon}.
$$

Implementation: `_norm_conv` in `hci/L1.py`. Pools $\hat s_{\mathrm{coll},k}$, $\hat s_{\mathrm{flank},k}$ convolve $\rho_k^{(t)}$ with $G^{\mathrm{coll}}_k$, $G^{\mathrm{flank}}_k$. **Cross pool** mixes other bins with orientation distance weights before spatial pooling:

$$
w_{k,j} = \frac{\sin^2\!\bigl(\pi(k-j)/K\bigr)}{\sum_{j'\neq k}\sin^2\!\bigl(\pi(k-j')/K\bigr)}, \quad j\neq k; \qquad w_{k,k}=0,
$$

$$
\tilde\rho_k^{(t)}(c) = \sum_{j=0}^{K-1} w_{k,j}\,\rho_j^{(t)}(c),
\qquad
\hat s_{\mathrm{cross},k}^{(t)}(c) = \mathrm{norm\_conv}(\tilde\rho_k^{(t)}, G^{\mathrm{cross}}).
$$

Neighboring bins ($\approx 7.5°$ at $K=24$) get weight $\sin^2(\pi/K)\approx 0.017$; orthogonal bins ($90°$) get weight $1.0$.

### Pass update ($t = 0,\ldots,T-1$, `L1.COL_PASSES`)

**Drive** (seed-gated excitation — collinear facilitation requires local $\rho^{\mathrm{seed}}$):

$$
\mathrm{drive}_k^{(t)}(c) =
\rho_k^{\mathrm{seed}}(c)\,\Bigl(
\beta_{\mathrm{seed}}
+ \beta_c\,\hat s_{\mathrm{coll},k}^{(t)}(c)
\Bigr).
$$

Where $\rho_k^{\mathrm{seed}}$ is low (no local oriented energy), drive vanishes regardless of neighbor collinear context; where $\rho_k^{\mathrm{seed}}$ is high, collinear pooling adds facilitation as before.

**Divisive NR** (scalar floor $\eta_p$; flank and cross as independent suppressive channels):

$$
\rho_k^{(t+1)}(c) = \frac{\bigl(\mathrm{drive}_k^{(t)}(c)\bigr)^2}
{\bigl(\mathrm{drive}_k^{(t)}(c)\bigr)^2 + \eta_p^2 + \beta_f\,\bigl(\hat s_{\mathrm{flank},k}^{(t)}(c)\bigr)^2 + \beta_x\,\bigl(\hat s_{\mathrm{cross},k}^{(t)}(c)\bigr)^2 + \varepsilon}.
$$

Each suppressive term in the denominator is an independent channel; they accumulate without interaction. Clean contours ($\hat s_{\mathrm{coll}}$ high, $\hat s_{\mathrm{flank}}$ low) yield large drive and small denominator → high $\rho$. Parallel texture or ramp edges ($\hat s_{\mathrm{flank}}$ high) grow the denominator and suppress $\rho$. $\varepsilon$ is `SEED.EPS` (`nr_eps` in code).

### Diagnostic $\kappa$

Raw collinear conv $S_k^{\mathrm{raw}}=(G^{\mathrm{coll}}_k * \rho_k^{(t)})(c)$ (unnormalized). Per-cell cosine over bins:

$$
\kappa^{(t)}(c) = \frac{\boldsymbol{\rho}^{(t)}(c)\cdot \mathbf{S}^{\mathrm{raw}(t)}(c)}
{\bigl\|\boldsymbol{\rho}^{(t)}(c)\bigr\|\,\bigl\|\mathbf{S}^{\mathrm{raw}(t)}(c)\bigr\| + \varepsilon}.
$$

`kappa_pass0_cell` / `kappa_col_cell` store $\kappa^{(t)}$ after the first / last pass. $\kappa$ is **not** multiplied into $\rho$ in the recurrence; it is exported to the renderer feature stack as $\bar\kappa_{\mathrm{col}}$.

**Geometry viz** (`infer.py` → `{stem}_geometry.png`): pass-0 max over bins of $\hat s_{\mathrm{coll}}$, $\hat s_{\mathrm{flank}}$, $\hat s_{\mathrm{cross}}$ on the cell grid.

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

plus collinear energy $e_{\mathrm{col}}$ at the dominant bin (unnormalized collinear conv on final $\boldsymbol{\rho}$), initial $\rho$ at $b^*$ from the seed-NR snapshot, etc., as produced by `run_l1_hypercolumn` and flattened in `build_cells_flat`.

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

**Boundary map** ($\bar\rho$ enters only via $F_p$, so loss gradients on $\hat B$ reach L1 $\rho$ through the MLP, not a compensating multiplicative shortcut):

$$
\mathrm{gate}(p) = \sigma\bigl(\mathrm{MLP}_{\mathrm{read}}(F_p)\bigr), \qquad
\hat B(p) = h_{2m}^{\Sigma}(p)\,\mathrm{gate}(p).
$$

$\mathrm{MLP}_{\mathrm{read}}$: **14 → 8 → 1** (ReLU between linear layers). Checkpoints that stored a **15**-input first layer can be loaded via `upgrade_renderer_state_dict`, which drops the last input column of `thinning.fc1.weight` to match the current head.

---

## 7. Training (`train.py`)

- **Disk cache** (`precompute_image`): L0 and geometry only; stores `h2m`, `theta_h`, `border_mask`, `l0_pix`, optional `d_lum` / `d_chr`, GT, projection metadata — **not** `cells_flat`. Invalidation: `TRAIN.CACHE_VERSION`. Directional tensors may be reused when `L0.L0_DIST_CACHE_VERSION` and the stored signature match.
- **Each step** (`prepare_batch` → `run_l1_live_cells` → `HarmonicContourE2E.forward_batch`): L1 runs **live** with `cells_format="torch"` so gradients reach `HypercolumnSeed` ($\tilde\eta_z$, $\tilde\eta_p$, $\tilde\beta_{\cdot}$, $\tilde\alpha_{d,t,\mathrm{iso}}$). `RhoSeedModule` then reads dominant $\rho$; the renderer consumes the same cached L0 pixel tensors (no interleaved L0 recompute inside GABA).

**Loss** (defaults in `params.TRAIN`): weighted sum of soft-Dice and/or BCE on the valid edge band vs.\ ground-truth boundaries.

---

## 8. Learned parameter count (current architecture)

| Block | Count | Notes |
|------:|------:|------|
| $\tilde\eta_z, \tilde\eta_p$ | 2 | Seed NR + pass NR floor |
| $\tilde\beta_{\mathrm{seed}}, \tilde\beta_c, \tilde\beta_f, \tilde\beta_x$ | 4 | Seed-gated drive ($\tilde\beta_{\mathrm{seed}}, \tilde\beta_c$) + NR denom ($\tilde\beta_f, \tilde\beta_x$) |
| $\tilde\alpha_d, \tilde\alpha_t, \tilde\alpha_{\mathrm{iso}}$ | 3 | Kernel scales → $\sigma_d,\sigma_t,\sigma_{\mathrm{iso}}$ |
| $s_t, s_n$ | 2 | Stencil spacing |
| $\mathrm{MLP}_{\mathrm{read}}$ (14→8→1) | 129 | $14\cdot 8 + 8 + 8 + 1$ |
| **Total (E2E)** | **140** | `HarmonicContourE2E` (9 seed + 131 renderer) |

Legacy checkpoints may contain `eta_mlp.*`, `seed.hc_seed._alpha_raw`, or `seed.hc_seed._gaba_alpha_raw`; `remap_checkpoint_state_dict` drops those keys and warm-starts missing $\tilde\beta_{\cdot}$, $\tilde\alpha_t$, $\tilde\alpha_{\mathrm{iso}}$ from `params.SEED` / `params.L1` defaults so `load_state_dict(..., strict=False)` stays clean.

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

This document matches the **three-pool GABA recurrence** in `hci/L1.py`: learned **seed NR** ($\eta_z$), kernel-normalized **collinear / flank / cross** pools, **seed-gated excitatory drive** ($\rho^{\mathrm{seed}}(\beta_{\mathrm{seed}}+\beta_c s_{\mathrm{coll}})$), **divisive flank and cross** in the NR denominator, scalar pass floor $\eta_p$, learned $\sigma_{\mathrm{iso}}$ for cross-orientation inhibition, **14-dimensional** renderer readout, and **no** spatial η-MLP.
