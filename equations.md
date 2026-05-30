# STRIATE — equations (code-aligned)

This file records the **notation and equations implemented in this repository**: `hci/L0.py`, `hci/L1.py`, `hci/seed.py`, `hci/renderer.py`, `train.py`, `infer.py`, `test.py`. Hyperparameters and defaults live in `params.py` (`L0`, `L1`, `SEED`, `RENDER`, `TRAIN`, …).

---

## 1. End-to-end pipeline

1. **L0** — RGB → split luminance / chrominance directional differences; per-direction min subtraction; independent Naka–Rushton per channel with **fixed** scalars $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ (`L0.ETA_LUM`, `L0.ETA_CHR`) and gain $\gamma$ (`L0.GAMMA`). Produces harmonic stack $s$, magnitudes $h_{1m}, h_{2m}$, and split $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$. Complex fields $z_1, z_2$ are read from $s$ (`z_from_l0_harmonics`). L0 is **precomputed and cached** for training; it is not updated during training.
2. **L1** — From L0 pixel field $z_2$, **sum-pool** over $P\times P$ patches → per-cell complex moment $Z_2$, scalar $\rho_{\mathrm{total}}$, coherence $R$, orientation $\theta = \tfrac12 \arg Z_2$, and $h_{2m}$-weighted splat anchors. **Runs live** each training step from cached L0 (`run_moments_cells_flat` in `prepare_batch`); **not** written to the L0 disk cache.
3. **Seed** (`AndGateSeed` / `CellSeed`) — surround-normalized AND gate on $(R, E_{\mathrm{rel}})$ with **learned** $R_0, a, b$ (init `SEED.R0_INIT`, `SEED.A_INIT`, `SEED.B_INIT`). Scalar export $\rho(c)=g_R(c)\,g_E(c)\,\mathrm{ok}(c)$; $\theta(c)$ passes through from L1 unchanged.
4. **Renderer** (`ModulationRenderer`) — Cell-grid $\theta$ combing and $\rho$-gated anchor smoothing; **Gaussian-line splat** of $\rho$ to pixels; coherence map; tangential / normal **9-tap stencils on $\bar\rho$**; **20→12→1** thinning MLP gate:
   $$\hat B(p) = \bar\rho(p)\,\mathrm{gate}(p).$$

There is **no** spatial $\eta$-MLP and **no** second L0 pass inside the training graph. The renderer's splat-footprint coherence map $\mathrm{coh}(p)$ is distinct from per-cell $R(c)$.

At inference, optional **ridge NMS** (`ridge_nms`) thins $\hat B$ using splat-dominant orientation $\theta^\star(p)$.

---

## 2. L0 — split luminance / chrominance harmonics

Eight offsets $\delta_k \in \mathbb{Z}^2$ (`L0.OFFSETS`). RGB maps to luminance $L=(R+G+B)/3$ and chrominance $C=(R,G,B)-L\mathbf{1}$ (3-vector per pixel).

**Directional differences**:

$$
d_k^{\mathrm{lum}}(p) = \bigl|L(p) - L(p+\delta_k)\bigr|, \qquad
d_k^{\mathrm{chr}}(p) = \bigl\|C(p) - C(p+\delta_k)\bigr\|_2 .
$$

**Per-direction min subtraction** (index $j$ runs over directions):

$$
\tilde d_k^{\mathrm{lum}} = d_k^{\mathrm{lum}} - \min_j d_j^{\mathrm{lum}}, \qquad
\tilde d_k^{\mathrm{chr}} = d_k^{\mathrm{chr}} - \min_j d_j^{\mathrm{chr}} .
$$

**Naka–Rushton** (independent per channel, per direction; $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ fixed from `params.L0`):

$$
h_k^{\mathrm{lum}} = \gamma\,\frac{(\tilde d_k^{\mathrm{lum}})^2}{\eta_{\mathrm{lum}}^2 + (\tilde d_k^{\mathrm{lum}})^2}, \qquad
h_k^{\mathrm{chr}} = \gamma\,\frac{(\tilde d_k^{\mathrm{chr}})^2}{\eta_{\mathrm{chr}}^2 + (\tilde d_k^{\mathrm{chr}})^2}.
$$

Combined directional response $h_k = h_k^{\mathrm{lum}} + h_k^{\mathrm{chr}}$.

**Harmonics** (unit bearings $\hat u_k$ from offsets; $F$ stacks $\cos\varphi_k, \sin\varphi_k, \cos 2\varphi_k, \sin 2\varphi_k$):

$$
s(p) = \sum_k h_k(p)\, F_k \in \mathbb{R}^4, \qquad
z_1(p) = s_0 + i s_1, \quad z_2(p) = s_2 + i s_3,
$$
$$
h_{1m}(p) = |z_1(p)|, \qquad h_{2m}(p) = |z_2(p)|.
$$

Split second-harmonic magnitudes $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$ use the same projection on $h_k^{\mathrm{lum}}, h_k^{\mathrm{chr}}$ alone (`compute_l0_rgb`). Border pixels are zeroed before L1.

*(Legacy path: grayscale / non-RGB uses divisive normalization $h_k = \gamma\, d_k^2 / (\eta_0^2 + \sum_j d_j^2)$ in `compute_contrast_field`.)*

---

## 3. L1 — per-cell z₂ moments

Patch size $P$ (`L1.PATCH_SIZE`), stride $S = P - \texttt{patch\_overlap}$ (`L1.PATCH_OVERLAP`) → cell grid $(n_H, n_W)$. Border mask $\mathrm{ok}(c) = \neg\,\texttt{is\_border}(c)$. A cell is **border** when the mean border mask over its patch exceeds `L1.BORDER_PATCH_MAX_FRAC`.

From the L0 pixel field $z_2(p) = s_2(p) + i\,s_3(p)$, compute **one complex moment and one scalar** per cell by sum-pool over $\mathrm{patch}(c)$:

$$
Z_2(c) = \sum_{p \in \mathrm{patch}(c)} z_2(p), \qquad
\rho_{\mathrm{total}}(c) = \sum_{p \in \mathrm{patch}(c)} |z_2(p)|.
$$

Coherence and orientation read off the same moment in closed form:

$$
R(c) = \frac{|Z_2(c)|}{\rho_{\mathrm{total}}(c) + \varepsilon} \in [0,1],
\qquad
\theta(c) = \tfrac{1}{2}\arg Z_2(c).
$$

**Splat anchors** ($h_{2m}$-weighted centroid within the patch; stored as `cx_z2`, `cy_z2`):

$$
c_x(c) = \frac{\sum_{p \in \mathrm{patch}(c)} h_{2m}(p)\, x(p)}{\sum_{p \in \mathrm{patch}(c)} h_{2m}(p) + \varepsilon}, \qquad
c_y(c) \text{ analogously}.
$$

**Seed / renderer fields** (`cells_flat`):

| Key | Value |
|-----|--------|
| `coherence_R` | $R(c)$ |
| `rho_total`, `z0` | $\rho_{\mathrm{total}}(c)$ |
| `theta` | $\theta(c) = \tfrac12 \arg Z_2(c)$ |
| `cx_z2`, `cy_z2` | $h_{2m}$-weighted splat anchors |

Border cells → $0$ on $\rho_{\mathrm{total}}, R, \theta$ as before.

---

## 4. Seed — surround-normalized AND gate (`AndGateSeed`)

**Surround energy** via Gaussian kernel $G_\sigma$ on the cell grid (radius 5, $\sigma \approx 2$ cells, center-excluded, reflect-padded at borders):

$$
\langle\rho_{\mathrm{total}}\rangle_{\mathcal{N}}(c) = (G_\sigma * \rho_{\mathrm{total}})(c),
\qquad
E_{\mathrm{rel}}(c) = \frac{\rho_{\mathrm{total}}(c)}{\varepsilon + \langle\rho_{\mathrm{total}}\rangle_{\mathcal{N}}(c)}.
$$

**Two gates** with learned scalars $\tilde R_0, \tilde a, \tilde b$ (softplus on $a, b$ for positivity; $R_0$ unconstrained but init near 0.45):

$$
g_R(c) = \sigma\!\bigl(a\,(R(c) - R_0)\bigr),
\qquad
g_E(c) = \sigma\!\bigl(b\,\log E_{\mathrm{rel}}(c)\bigr).
$$

**Cell amplitude** (scalar export to renderer):

$$
\rho(c) = g_R(c)\,\cdot\,g_E(c)\,\cdot\,\mathrm{ok}(c).
$$

Border cells → $0$ on $\rho, \theta$. **Orientation** $\theta(c)$ is read from L1 (`cells_flat['theta']`); seed does not refine $\theta$.

**Initialization**

| Param | Init | Role |
|------:|-----:|------|
| $R_0$ | $0.45$ | Coherence floor |
| $a$ | $\mathrm{softplus}^{-1}(12)$ | $g_R$ steepness |
| $b$ | $\mathrm{softplus}^{-1}(5)$ | $g_E$ steepness |
| $\sigma$ (surround) | $2.0$ cells | Fixed, not learned |

---

## 5. Renderer — splat, coherence, stencils, thinning head

**Step 1 — cell grid.** $\rho$-weighted double-angle $\theta$ smoothing (`RENDER.THETA_SMOOTH_PASSES`); $\rho$- and orientation-gated smoothing of anchors $(c_x^{z_2}, c_y^{z_2})$. Coordinates and $\theta$ are **detached** before splat (no coordinate gradients into seed).

**Step 2 — Anisotropic Gaussian splat.** For each active cell $c$ with $\rho_c > 0$, deposit a finite oriented kernel centered at anchor $a_c = (c_x, c_y)$ with learned widths $\sigma_\perp = \mathrm{softplus}(\tilde\sigma_\perp)$, $\sigma_\parallel = \mathrm{softplus}(\tilde\sigma_\parallel)$ (init $\approx$ cell stride $S$):

$$
\hat t_c = (\cos\theta_c,\,\sin\theta_c), \quad
\hat n_c = (-\sin\theta_c,\,\cos\theta_c),
$$
$$
d_\parallel(p,c) = (p - a_c)\cdot \hat t_c, \quad
d_\perp(p,c) = (p - a_c)\cdot \hat n_c,
$$
$$
\phi_c(p) = \exp\!\left(
-\frac{d_\perp(p,c)^2}{2\sigma_\perp^2}
-\frac{d_\parallel(p,c)^2}{2\sigma_\parallel^2}
\right).
$$

**Amplitude** (unnormalized accumulation — dead cells contribute zero):

$$
\bar\rho(p) = \sum_c \rho_c\,\phi_c(p).
$$

At the anchor, $\phi_c(a_c)=1$ so $\bar\rho(a_c)=\rho_c$ when no overlap. Adjacent active cells sum constructively along edges.

Dominant orientation per pixel (scatter-max by $\rho_c\,\phi_c$):

$$
\theta^\star(p) = \theta_{c^\star(p)}, \qquad
c^\star(p) = \arg\max_c \rho_c\,\phi_c(p).
$$

**Step 3 — coherence** (within splat footprint):

$$
\mathrm{coh}(p) = \frac{\sum_c \rho_c\,\phi_c(p)\,\cos^2\!\bigl(\theta_c - \theta^\star(p)\bigr)}
{\sum_c \rho_c\,\phi_c(p) + \varepsilon}.
$$

**Step 4 — stencils on $\bar\rho$** with unit tangent $\hat t = (\cos\theta^\star, \sin\theta^\star)$, normal $\hat n = (-\sin\theta^\star, \cos\theta^\star)$, learned spacings $s_t, s_n$ (softplus not applied to $s_t, s_n$ in code—they are raw `nn.Parameter` initialized at 1):

$$
\mathrm{tang}_j(p) = \bar\rho\!\left(p + j\, s_t\, \hat t(p)\right), \qquad
\mathrm{norm}_j(p) = \bar\rho\!\left(p + j\, s_n\, \hat n(p)\right), \qquad j \in \{-4,\ldots,4\}.
$$

(Bilinear sampling via `grid_sample`.)

**Feature vector** $F_p \in \mathbb{R}^{20}$:

$$
F_p = \bigl[
\bar\rho,\, \mathrm{coh},\,
\mathrm{tang}_{-4},\ldots,\mathrm{tang}_{4},\,
\mathrm{norm}_{-4},\ldots,\mathrm{norm}_{4}
\bigr].
$$

**Thinning head** (structural priors at init: Mexican-hat on `norm9`, flat $1/9$ smooth on `tang9`, $\sigma(\mathrm{MLP}(F_p)) \approx 0.88$ at $t{=}0$):

$$
\mathrm{gate}(p) = \sigma\!\bigl(W_2\,\mathrm{ReLU}(W_1 F_p + b_1) + b_2\bigr), \qquad
\hat B(p) = \bar\rho(p)\,\mathrm{gate}(p).
$$

$\mathrm{MLP}$: **20 → 12 → 1**. Output is cropped to original content size $(H_0, W_0)$.

`upgrade_renderer_state_dict` strips legacy renderer keys (e.g. `_sigma_par_raw`, `perp_conv.*`) before `load_state_dict(..., strict=False)`.

---

## 6. Training (`train.py`)

- **Disk cache** (`precompute_image`): L0 only — padded RGB, `l0_pix` ($h_{2m}$, $z_1$, $z_2$ channels), `border_mask`, GT, `proj_info` grid dims. **No** L1 cells. Invalidation: `TRAIN.L0_CACHE_VERSION` (bump when L0/pad/`l0_pix` change only).
- **Each step** (`prepare_batch`): **live moment pooling** from cached L0 → `cells_flat_dev`, then AND-gate seed + renderer.

**Loss** (defaults `TRAIN.LAM_DICE=0`, `TRAIN.LAM_BCE=1`): weighted sum of soft-Dice and/or BCE on the **η± edge band** — valid pixels where $\mathrm{GT} \ge \eta_{\mathrm{pos}}$ or $\mathrm{GT} < \eta_{\mathrm{neg}}$ (default $\eta_{\mathrm{pos}} = \eta_{\mathrm{neg}} = 0.5$). Positive label on $\mathrm{GT} \ge \eta_{\mathrm{pos}}$.

Checkpoints store `{"model_state": state_dict}` (`intermediate.pt`, `final.pt`).

---

## 7. Learned parameter count (current architecture)

| Block | Count | Notes |
|------:|------:|------|
| $R_0, \tilde a, \tilde b$ | 3 | AND-gate scalars |
| $\tilde\sigma_\perp, \tilde\sigma_\parallel, s_t, s_n$ | 4 | Splat widths + stencil spacings |
| $\mathrm{MLP}_{\mathrm{thin}}$ (20→12→1) | 265 | $20\cdot 12 + 12 + 12 + 1$ |
| **Renderer subtotal** | **269** | |
| **Total (`StriateE2E`)** | **272** | 3 AND-gate + 269 renderer |

L0 $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ are **fixed** (`params.L0`). Legacy checkpoints may contain `_eta_z_raw`, `_l1_von_mises_kappa_raw`; `upgrade_model_state_dict` strips these and inits AND-gate params if missing. Renderer weights load unchanged via `upgrade_renderer_state_dict` and `load_state_dict(..., strict=False)` with `report_checkpoint_compatibility`.

---

## 8. Module map

| Stage | Primary code |
|-------|----------------|
| $d_k$, NR, harmonics, $z_1, z_2$, interior mask | `hci/L0.py` |
| z₂ moment pooling, $R$, $\theta$, $h_{2m}$ anchors | `hci/L1.py` |
| AND gate $g_R \cdot g_E$, scalar $\rho$ export | `hci/seed.py` |
| Splat, coherence, stencils, thinning head, NMS | `hci/renderer.py` |
| Cache, batching, loss, checkpoints | `train.py` |
| Single-image pipeline, diagnostics | `infer.py`, `hci/diagnostics_viz.py` |
| Test-set sweep (ODS / OIS / AP) | `test.py` |

---

## 9. Revision note

This document matches the **STRIATE** stack: **z₂ moment pooling** (§3), **surround-normalized AND gate** (§4), **anisotropic Gaussian splat** renderer with 9-tap stencils (§5), **`StriateE2E`** (272 learned parameters). L0 only is disk-cached for training (§6).
