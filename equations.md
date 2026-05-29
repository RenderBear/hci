# STRIATE — equations (code-aligned)

This file records the **notation and equations implemented in this repository**: `hci/L0.py`, `hci/L1.py`, `hci/L2.py`, `hci/renderer.py`, `train.py`, `infer.py`, `test.py`. Hyperparameters and defaults live in `params.py` (`L0`, `L1`, `L2`, `RENDER`, `TRAIN`, …).

---

## 1. End-to-end pipeline

1. **L0** — RGB → split luminance / chrominance directional differences; per-direction min subtraction; independent Naka–Rushton per channel with **fixed** scalars $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ (`L0.ETA_LUM`, `L0.ETA_CHR`) and gain $\gamma$ (`L0.GAMMA`). Produces harmonic stack $s$, magnitudes $h_{1m}, h_{2m}$, and split $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$. Complex fields $z_1, z_2$ are read from $s$ (`z_from_l0_harmonics`). L0 is **precomputed and cached** for training; it is not updated during training.
2. **L1** — Pixel $K$-bin von Mises (or $\cos^p$) projection of $h_{2m}$ (`L1.K`; von Mises $\kappa$ is **learned**, softplus, init `L1.COL_VON_MISES_KAPPA`$=8$); sum-pool over $P\times P$ patches → per-cell $\rho_{\mathrm{bins}}^{(k)}$, dominant $\theta$, polarity $q$, anisotropy $\delta$, optional orientation $\kappa$, and $h_{2m}$-weighted splat anchors. **Precomputed and cached** with L0.
3. **L2** (`TileDynamics`) — $K$-channel state $\rho^{(k)}(c)$: **seed** $\rho^{(0)}$ from $\rho_{\mathrm{bins}}$ (total-normalize → per-bin NR); **drive** $b_{\mathrm{seed}}\rho_{\mathrm{seed}}^{(k)} + b_{\mathrm{coll}}\tilde\rho_{\mathrm{coll}}^{(k)}$; **inhibition** iso + spatial cross; **NR pass** + **mixing** $\rho \leftarrow (1-\alpha)\rho + \alpha\rho_{\mathrm{nr}}$. Seven learned L2 scalars; runs **live each training step**.
4. **Renderer** (`ModulationRenderer`) — Cell-grid $\theta$ combing and $\rho$-gated anchor smoothing; **Gaussian-line splat** of refined $\rho$ to pixels; coherence map; tangential / normal **stencils on $\bar\rho$**; **12→8→1** thinning MLP gate:
   $$\hat B(p) = \bar\rho(p)\,\mathrm{gate}(p).$$

There is **no** spatial $\eta$-MLP and **no** second L0 pass inside the training graph. L1 $\kappa$ is computed but **not** fed to the renderer. `L1.K` must match `L2.K` for hard orientation bins.

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

## 3. L1 — pixel K-bin projection

Patch size $P$ (`L1.PATCH_SIZE`), stride $S = P - \texttt{patch\_overlap}$ (`L1.PATCH_OVERLAP`) → cell grid $(n_H, n_W)$, $K$ bins (`L1.K`, default 24, same as `L2.K`). Bin centres $\bar\theta_k = k\pi/K$. A cell is **border** when the mean border mask over its patch exceeds `L1.BORDER_PATCH_MAX_FRAC`.

Pixel orientation from L0 ($\S$2):

$$
\theta_{2m}(p) = \tfrac12 \arg z_2(p).
$$

**Pixel-level bin channels** (default: von Mises, `L1.COL_BIN_TUNING`; $\kappa = \mathrm{softplus}(\tilde\kappa)$, init `L1.COL_VON_MISES_KAPPA` $= 8$):

$$
e_k(p) = h_{2m}(p)\,\exp\!\Bigl(\kappa\,\cos\bigl(2(\theta_{2m}(p) - \bar\theta_k)\bigr)\Bigr).
$$

Alternate (`COL_BIN_TUNING` = `cos_pow`, `COL_COS_POWER` $= p$):

$$
e_k(p) = h_{2m}(p)\,\cos^{p}\!\bigl(\theta_{2m}(p) - \bar\theta_k\bigr).
$$

At $\kappa{=}0$ the von Mises factor is flat across bins; at $\kappa{=}4$ it is much sharper than $\cos^2$ tuning. Mass is **not** normalized across $k$ at the pixel (contrast comes from pooling and L2).

**Per-cell bin mass** (sum over patch; implemented as sum-pool / `avg_pool2d` × $P^2$):

$$
\rho_{\mathrm{bins}}^{(k)}(c) = \sum_{p \in \mathrm{patch}(c)} e_k(p).
$$

**Dominant orientation and anisotropy**:

$$
k^\*(c) = \arg\max_k \rho_{\mathrm{bins}}^{(k)}(c), \qquad
\theta(c) = \bar\theta_{k^\*(c)} \;\text{(parabolic sub-bin refinement in L1/L2 export)},
$$
$$
\rho_{\mathrm{peak}}(c) = \rho_{\mathrm{bins}}^{(k^\*)}(c), \qquad
\rho_{\mathrm{total}}(c) = \sum_k \rho_{\mathrm{bins}}^{(k)}(c),
$$
$$
\delta(c) = \frac{\rho_{\mathrm{peak}}(c)}{\rho_{\mathrm{total}}(c) + \varepsilon}.
$$

**Polarity** (winning-bin $\theta$, patch sum of $z_1$):

$$
q(c) = \Re\!\Bigl(\overline{Z_1(c)}\, e^{i\theta(c)}\Bigr), \qquad
Z_1(c) = \sum_{p \in \mathrm{patch}(c)} z_1(p).
$$

**Orientation confidence** $\kappa \in [0,1]$ (`_patch_orientation_kappa`): $z_1$ polarity agreement along the edge normal implied by $(\theta, q)$.

**Splat anchors** ($h_{2m}$-weighted centroid within the patch; stored as `cx_z2`, `cy_z2`):

$$
c_x(c) = \frac{\sum_{p \in \mathrm{patch}(c)} h_{2m}(p)\, x(p)}{\sum_{p \in \mathrm{patch}(c)} h_{2m}(p) + \varepsilon}, \qquad
c_y(c) \text{ analogously}.
$$

**L2-facing fields** (`cells_flat`):

| Key | Value |
|-----|--------|
| `rho_bins` | $(N, K)$ per-cell $\rho_{\mathrm{bins}}^{(k)}$ (L2 seed input) |
| `rho_peak` | $\rho_{\mathrm{peak}}$ (max bin mass) |
| `z0` | $\rho_{\mathrm{total}}$ (sum over bins) |
| `theta`, `k_star`, … | orientation / parabolic $\theta$ export |

Border cells zero $\theta, q, \delta, \kappa, \rho_{\mathrm{bins}}, \rho_{\mathrm{peak}}, \rho_{\mathrm{total}}$.

---

## 4. L2 — seed and tile dynamics (`TileDynamics`)

Interior mask $\mathrm{ok}(c) = \neg\,\texttt{is\_border}(c)$. State is **$K$ channels per cell** ($K =$ `L2.K`, default 24), not a single hard bin. Initial $\rho^{(0)}(c,k) = \rho_{\mathrm{seed}}^{(k)}(c)$; after refine, scalar export uses $\max_k \rho^{(k)}$ and parabolic $\theta$ from final bins (`collapse_rho_bins`).

### Seed (IC only; fixed in drive)

From L1 `rho_bins` (same $\rho_{\mathrm{bins}}^{(k)}$ as §3). Same pattern as L0 (§2): per-cell min subtraction on normalized bins, then Naka–Rushton squash. Code names the normalized fraction `rho_raw` and the min-subtracted contrast `rho_tilde`:

$$
\rho_{\mathrm{total}}(c) = \sum_k \rho_{\mathrm{bins}}^{(k)}(c), \qquad
\rho_{\mathrm{raw}}^{(k)}(c) = \frac{\rho_{\mathrm{bins}}^{(k)}(c)}{\rho_{\mathrm{total}}(c) + \varepsilon},
$$
$$
\tilde\rho^{(k)}(c) = \rho_{\mathrm{raw}}^{(k)}(c) - \min_j \rho_{\mathrm{raw}}^{(j)}(c),
$$
$$
\rho_{\mathrm{seed}}^{(k)}(c) = \frac{\bigl(\tilde\rho^{(k)}(c)\bigr)^2}{\bigl(\tilde\rho^{(k)}(c)\bigr)^2 + \eta_z^2}, \qquad
\eta_z = \texttt{L2.ETA\_Z} = 0.05\ \text{(fixed, same scale as L0 $\eta$)}.
$$

Min subtraction kills uniform texture (all bins $\approx 1/K$ → $\tilde\rho \approx 0$); anisotropic edges preserve peak-vs-min contrast regardless of absolute magnitude. Border cells: $\rho_{\mathrm{seed}}^{(k)} = 0$. With $T_{\mathrm{refine}}=0$ (`infer --no-dynamics`), $\rho^{(k)}$ stays at $\rho_{\mathrm{seed}}^{(k)}$.

### Learned scalars (softplus of raw parameters)

$$
b_{\mathrm{seed}},\; b_{\mathrm{coll}},\; b_{\mathrm{iso}},\; b_{\mathrm{cross}},\; \eta_p,\; \alpha .
$$

Defaults in `params.L2`: `B_SEED_INIT`, `B_COLL_INIT`, `B_ISO_INIT`, `B_CROSS_INIT`, `ETA_P_INIT`, `ALPHA_INIT`. $\rho_{\mathrm{seed}}$ sets $\rho^{(0)}$ and is **fixed** in the drive (minimum evidence floor for isolated edges).

### Fixed conv kernels (buffers, no gradients)

Radii $R_{\mathrm{fac}} =$ `L2.R_FAC_POOL`, $R_{\mathrm{sup}} =$ `L2.R_SUP_POOL`. Bin axes $\bar\theta_k = k\pi/K$. Offset $(d_i, d_j)$ from cell centre, $\hat t_k = (\cos\bar\theta_k, \sin\bar\theta_k)$, $\hat n_k = (-\sin\bar\theta_k, \cos\bar\theta_k)$, $\hat d = (d_i, d_j)/\|d\|$.

**Collinear** (facilitation radius $R_{\mathrm{fac}}$, centre omitted):

$$
W^{\mathrm{coll}}_k(d) = \bigl|\hat t_k \cdot \hat d\bigr|.
$$

**Iso-orientation** (suppression radius $R_{\mathrm{sup}}$):

$$
W^{\mathrm{iso}}_k(d) = (\hat n_k \cdot \hat d)^2.
$$

**Cross pool** (isotropic, suppression radius $R_{\mathrm{sup}}$, centre omitted):

$$
W^{\mathrm{disk}}(d) = 1 \quad (d \neq 0).
$$

### Pools on the cell grid

State $\rho^{(t)}(c,k)$ has $K$ channels per cell. **Coll/iso** pool the evolving $\rho^{(t)}$ via grouped `conv2d` (count-normalized per $k$). $\rho_{\mathrm{seed}}^{(k)}$ is **fixed** in the drive.

$$
\rho_{\mathrm{coll}}^{(k)} = \frac{W^{\mathrm{coll}}_k * \rho^{(t)}_k}{W^{\mathrm{coll}}_k * \mathbf{1} + \varepsilon}, \qquad
c_{\mathrm{iso}}^{(k)} = \frac{W^{\mathrm{iso}}_k * (\rho^{(t)}_k)^2}{W^{\mathrm{iso}}_k * \mathbf{1} + \varepsilon}.
$$

**Cross** — disk-pool each bin, then mean **other-bin** pooled mass (spatial neighborhood via $W^{\mathrm{disk}}$; enters inhibition as $b_{\mathrm{cross}}\,c_{\mathrm{cross}}^{(k,t)}$):

$$
\bar\rho^{(k,t)}(c) = \frac{W^{\mathrm{disk}} * \rho^{(k,t)}}{W^{\mathrm{disk}} * \mathbf{1} + \varepsilon}, \qquad
c_{\mathrm{cross}}^{(k,t)}(c) = \mathrm{ok}(c)\cdot\frac{\sum_{k' \neq k} \bar\rho^{(k',t)}(c)}{K - 1}.
$$

Clean edge (neighbors weak in off-peak bins): low $c_{\mathrm{cross}}$ at the peak bin. Texture / mixed orientations in the disk: high cross. Diagnostics (`geometry` snapshots): $\max_k \tilde\rho_{\mathrm{coll}}$, $\max_k c_{\mathrm{iso}}$, $\max_k c_{\mathrm{cross}}$, $\max_k \rho^{(k)}$ at $t{=}0$ and $t{=}$last; `l2_rho_seed_post` also plots $\max_k \rho_{\mathrm{seed}}$, $\max_k \rho$, and $\Delta\rho = \rho - \rho_{\mathrm{seed}}$ on those scalars.

### Pass update ($t = 0,\ldots,T_{\mathrm{refine}}-1$, `L2.T_REFINE`)

$$
\mathrm{drive}^{(t)}(c,k) = b_{\mathrm{seed}}\,\rho_{\mathrm{seed}}^{(k)}(c) + b_{\mathrm{coll}}\,\tilde\rho_{\mathrm{coll}}^{(k,t)}(c),
$$
$$
\rho_{\mathrm{nr}}^{(k,t+1)}(c) = \mathrm{ok}(c)\cdot
\frac{\bigl(\mathrm{drive}^{(t)}(c,k)\bigr)^2}
{\bigl(\mathrm{drive}^{(t)}(c,k)\bigr)^2
+ b_{\mathrm{iso}}\,c_{\mathrm{iso}}^{(k,t)}(c)
+ b_{\mathrm{cross}}\,c_{\mathrm{cross}}^{(k,t)}(c)
+ \eta_p^2 + \varepsilon},
$$
$$
\rho^{(k,t+1)}(c) = (1 - \alpha)\,\rho^{(k,t)}(c) + \alpha\,\rho_{\mathrm{nr}}^{(k,t+1)}(c),
\qquad \alpha = \mathrm{clamp}_{[0,1]}\!\bigl(\mathrm{softplus}(\tilde\alpha)\bigr),
$$
with defaults `B_SEED_INIT = B_COLL_INIT = 0.5`, `ALPHA_INIT = 0.5`. **Mixing** dampens coll/iso/cross oscillation across steps. **Seed pin** ($b_{\mathrm{seed}}$) gives a minimum drive from the initial hypothesis so short edges without long collinear context do not collapse; $b_{\mathrm{seed}}$ vs $b_{\mathrm{coll}}$ trades local evidence vs neighborhood consensus.

**TBPTT**: `grad_window = \texttt{L2\_SNAPSHOT\_MAX}` ($=5$) when $T_{\mathrm{refine}} > 5$ (detach every 5 steps → at most 5 refine steps of gradient per segment). If $T_{\mathrm{refine}} \le 5$, no in-loop detach — full backprop through all refine steps so seed→coll compounding is visible to $b_{\mathrm{seed}}$, etc.

**Diagnostics** (`return_surface_diags`): raw pool snapshots at $t{=}0$ and $t{=}$last; optional bimodality $\sum_c \rho(1-\rho)$ per snapshot step.

---

## 5. Renderer — splat, coherence, stencils, thinning head

**Step 1 — cell grid.** $\rho$-weighted double-angle $\theta$ smoothing (`RENDER.THETA_SMOOTH_PASSES`); $\rho$- and orientation-gated smoothing of anchors $(c_x^{z_2}, c_y^{z_2})$. Coordinates and $\theta$ are **detached** before splat (no coordinate gradients into L2).

**Step 2 — Gaussian-line splat.** For each active cell $c$ with $\rho_c > 0$, deposit along normal to $\theta_c$ with learned width $\sigma_\perp = \mathrm{softplus}(\tilde\sigma_\perp)$:

$$
\phi_c(p) = \exp\!\left(-\frac{d_\perp(p,c)^2}{2\sigma_\perp^2}\right),
\qquad
\bar\rho(p) = \frac{\sum_c \rho_c\,\phi_c(p)}{\sum_c \phi_c(p) + \varepsilon}.
$$

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
\mathrm{norm}_j(p) = \bar\rho\!\left(p + j\, s_n\, \hat n(p)\right), \qquad j \in \{-2,-1,0,1,2\}.
$$

(Bilinear sampling via `grid_sample`.)

**Feature vector** $F_p \in \mathbb{R}^{12}$:

$$
F_p = \bigl[
\bar\rho,\, \mathrm{coh},\,
\mathrm{tang}_{-2},\ldots,\mathrm{tang}_{2},\,
\mathrm{norm}_{-2},\ldots,\mathrm{norm}_{2}
\bigr].
$$

**Thinning head** (structural priors at init: Mexican-hat on `norm5`, flat smooth on `tang5`, $\sigma(\mathrm{MLP}(F_p)) \approx 0.88$ at $t{=}0$):

$$
\mathrm{gate}(p) = \sigma\!\bigl(W_2\,\mathrm{ReLU}(W_1 F_p + b_1) + b_2\bigr), \qquad
\hat B(p) = \bar\rho(p)\,\mathrm{gate}(p).
$$

$\mathrm{MLP}$: **12 → 8 → 1**. Output is cropped to original content size $(H_0, W_0)$.

`upgrade_renderer_state_dict` strips legacy renderer keys (e.g. `_sigma_par_raw`, `perp_conv.*`) before `load_state_dict(..., strict=False)`.

---

## 6. Training (`train.py`)

- **Disk cache** (`precompute_image`): L0 only — padded RGB, `l0_pix` ($h_{2m}$, $z_1$, $z_2$ channels), `border_mask`, GT, `proj_info` grid dims. **No** L1 $\rho$/cells. Invalidation: `TRAIN.L0_CACHE_VERSION` (bump when L0/pad/`l0_pix` change, **not** von Mises / L1 tuning).
- **Each step** (`prepare_batch`): **live L1** from cached L0 → `cells_flat_dev`, then L2 + renderer. Changing `L1.COL_BIN_TUNING`, `COL_VON_MISES_KAPPA`, etc. does not require rebuilding the cache.

**Loss** (defaults `TRAIN.LAM_DICE=0`, `TRAIN.LAM_BCE=1`): weighted sum of soft-Dice and/or BCE on the **η± edge band** — valid pixels where $\mathrm{GT} \ge \eta_{\mathrm{pos}}$ or $\mathrm{GT} < \eta_{\mathrm{neg}}$ (default $\eta_{\mathrm{pos}} = \eta_{\mathrm{neg}} = 0.5$). Positive label on $\mathrm{GT} \ge \eta_{\mathrm{pos}}$.

Checkpoints store `{"model_state": state_dict}` (`intermediate.pt`, `final.pt`).

---

## 7. Learned parameter count (current architecture)

| Block | Count | Notes |
|------:|------:|------|
| $\tilde b_{\mathrm{seed}}, \tilde b_{\mathrm{coll}}, \tilde b_{\mathrm{iso}}, \tilde b_{\mathrm{cross}}, \tilde\eta_p, \tilde\alpha$ | 6 | L2: drive / inhibition + mixing |
| $\tilde\kappa$ (L1 von Mises) | 1 | Bin sharpness in $e_k(p)$ |
| $\tilde\sigma_\perp, s_t, s_n$ | 3 | Splat width + stencil spacings |
| $\mathrm{MLP}_{\mathrm{thin}}$ (12→8→1) | 113 | $12\cdot 8 + 8 + 8 + 1$ |
| **Total (`StriateE2E`)** | **123** | 6 L2 + 1 L1 $\kappa$ + 116 renderer |

L0 $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ and L2 $\eta_z$ are **fixed** (`params.L0`, `params.L2`). Legacy checkpoints from older K-bin / 14-feature renderer architectures may load with missing/unexpected keys; use `upgrade_renderer_state_dict` and `load_state_dict(..., strict=False)` with `report_checkpoint_compatibility`.

---

## 8. Module map

| Stage | Primary code |
|-------|----------------|
| $d_k$, NR, harmonics, $z_1, z_2$, interior mask | `hci/L0.py` |
| Pixel K-bin projection, $\kappa$, $h_{2m}$ anchors | `hci/L1.py` |
| $\rho_{\mathrm{seed}}$, conv pools, recurrent $\rho$ refine | `hci/L2.py` |
| Splat, coherence, stencils, thinning head, NMS | `hci/renderer.py` |
| Cache, batching, loss, checkpoints | `train.py` |
| Single-image pipeline, diagnostics | `infer.py`, `hci/diagnostics_viz.py` |
| Test-set sweep (ODS / OIS / AP) | `test.py` |

---

## 9. Revision note

This document matches the **STRIATE** stack: L1 **K-bin projection** (§3), L2 **`TileDynamics`** — $\rho_{\mathrm{raw}}=\rho_{\mathrm{bins}}/\rho_{\mathrm{total}}$, min-subtract + NR seed, seed pin + coll drive, spatial cross, mixing (§4), **Gaussian-line splat** renderer (§5), **`StriateE2E`** (123 learned scalars).
