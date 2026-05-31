# STRIATE — equations (code-aligned)

This file records the **notation and equations implemented in this repository**: `hci/L0.py`, `hci/L1.py`, `hci/seed.py`, `hci/renderer.py`, `train.py`, `infer.py`, `test.py`. Hyperparameters and defaults live in `params.py` (`L0`, `L1`, `SEED`, `RENDER`, `TRAIN`, …).

---

## 1. End-to-end pipeline

1. **L0** — RGB directional differences → per-direction min subtraction → independent Naka–Rushton per channel with **fixed** $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ (`L0.ETA_LUM`, `L0.ETA_CHR`) and gain $\gamma$ (`L0.GAMMA`). Produces harmonic stack $s$, magnitudes $h_{1m}, h_{2m}$, split $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$, and complex fields $z_1, z_2$ (`z_from_l0_harmonics`). With **`L0.LEARNED_METRIC`** (default), a **learned** $3\times3$ matrix $W$ (`L0LearnedMetric`) replaces the fixed lum/chr split for distance; L0 is recomputed **live each training step** from cached RGB. Without it, L0 uses the fixed orthonormal lum/chr split and may be precomputed into the disk cache.
2. **L1** — From L0 pixel field $z_2$, pool over $P\times P$ patches → per-cell $\rho_{\mathrm{total}}$, $\rho_{\mathrm{peak}}$, orientation $\theta$, and $h_{2m}$-weighted splat anchors. **Runs live** each training step (`run_moments_cells_flat` in `prepare_batch`).
3. **Seed** (`AndGateSeed` / `ContourSeed`) — NR-normalise $\rho_{\mathrm{peak}}$ → $\rho_{\mathrm{seed}}$; collinear readback $\rho_{\mathrm{coll}}$; excitation $e = \beta_{\mathrm{seed}}\rho_{\mathrm{seed}} + \beta_{\mathrm{coll}}\rho_{\mathrm{coll}}$; divisive export $\rho = e^2/(e^2 + \eta^2 + \lambda S^2)$ with **learned** $\beta_{\mathrm{seed}}, \beta_{\mathrm{coll}}, \kappa_\theta, \eta_{\mathrm{seed}}, \eta, \lambda, \sigma_f$.
4. **Renderer** (`ModulationRenderer`) — Cell-grid $\theta$ combing and $\rho$-gated anchor smoothing; **Gaussian-line splat** of $\rho$ to pixels; **splat-footprint coherence** map $\mathrm{coh}(p)$; tangential / normal **9-tap stencils on $\bar\rho$**; **20→12→1** thinning MLP gate:
   $$\hat B(p) = \bar\rho(p)\,\mathrm{gate}(p).$$

The renderer's $\mathrm{coh}(p)$ (§5) is **not** an L1 quantity — it measures orientation agreement within the splat footprint at each pixel.

At inference, optional **ridge NMS** (`ridge_nms`) thins $\hat B$ using splat-dominant orientation $\theta^\star(p)$.

---

## 2. L0 — split luminance / chrominance harmonics

Eight offsets $\delta_k \in \mathbb{Z}^2$ (`L0.OFFSETS`).

### Fixed metric (default split, or init of learned $W$)

RGB maps to luminance $L=(R{+}G{+}B)/3$ and chrominance $C=(R,G,B)-L\mathbf{1}$. Directional differences:

$$
d_k^{\mathrm{lum}}(p) = \bigl|L(p) - L(p+\delta_k)\bigr|, \qquad
d_k^{\mathrm{chr}}(p) = \bigl\|C(p) - C(p+\delta_k)\bigr\|_2 .
$$

### Learned metric (`L0LearnedMetric`, `M = W^\top W$)

For RGB difference $\Delta\mathbf{c}_k(p) = \mathbf{I}(p)-\mathbf{I}(p+\delta_k)$:

$$
d_k^{\mathrm{lum}}(p) = \bigl|(W\Delta\mathbf{c}_k)_0\bigr|, \qquad
d_k^{\mathrm{chr}}(p) = \bigl\|(W\Delta\mathbf{c}_k)_{1:3}\bigr\|_2 .
$$

$W$ is initialized to the orthonormal lum/chr basis (row 0 = luminance, rows 1–2 = chrominance). Norms use $\varepsilon$ (`L0.EPS`) inside the square root for stable backprop.

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
h_{1m}(p) = \sqrt{|z_1(p)|^2 + \varepsilon}, \qquad h_{2m}(p) = \sqrt{|z_2(p)|^2 + \varepsilon}.
$$

Split second-harmonic magnitudes $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$ use the same projection on $h_k^{\mathrm{lum}}, h_k^{\mathrm{chr}}$ alone (`compute_l0_rgb`). Border pixels are zeroed before L1.

*(Legacy path: grayscale / non-RGB uses divisive normalization $h_k = \gamma\, d_k^2 / (\eta_0^2 + \sum_j d_j^2)$ in `compute_contrast_field`.)*

---

## 3. L1 — per-cell z₂ moments

Patch size $P$ (`L1.PATCH_SIZE`), stride $S = P - \texttt{patch\_overlap}$ (`L1.PATCH_OVERLAP`) → cell grid $(n_H, n_W)$. A cell is **border** when the mean border mask over its patch exceeds `L1.BORDER_PATCH_MAX_FRAC`.

From the L0 pixel field $z_2(p) = s_2(p) + i\,s_3(p)$:

$$
\rho_{\mathrm{total}}(c) = \sum_{p \in \mathrm{patch}(c)} |z_2(p)|.
$$

**Coherent peak** (min-subtracted top-weighted complex pool on $z_2$; fixed $q = 2$):

$$
\tilde m(p) = \max\!\bigl(0,\;|z_2(p)| - \min_{q \in \mathrm{patch}} |z_2(q)|\bigr), \qquad
w(p) = \frac{\tilde m(p)^q}{\sum_{r \in \mathrm{patch}} \tilde m(r)^q + \varepsilon},
$$
$$
\rho_{\mathrm{peak}}(c) = \Bigl|\sum_{p \in \mathrm{patch}(c)} w(p)\, z_2(p)\Bigr|.
$$

With $u(p) = w(p)|z_2(p)|$, $U(c) = \sum_p u(p)$: $\rho_{\mathrm{peak}} = U \cdot R$ where $R(c) = \rho_{\mathrm{peak}}/U \in [0,1]$ is stored as **`rho_coherence`**.

**Orientation** uses the $h_{2m}$-weighted complex moment:

$$
Z_2^{w}(c) = \sum_{p \in \mathrm{patch}(c)} h_{2m}(p)\, z_2(p), \qquad
\theta(c) = \tfrac{1}{2}\operatorname{atan2}\!\bigl(\Im Z_2^{w}, \Re Z_2^{w} + \varepsilon\bigr).
$$

**Splat anchors** ($h_{2m}$-weighted centroid within the patch; stored as `cx_z2`, `cy_z2`):

$$
c_x(c) = \frac{\sum_{p \in \mathrm{patch}(c)} h_{2m}(p)\, x(p)}{\sum_{p \in \mathrm{patch}(c)} h_{2m}(p) + \varepsilon}, \qquad
c_y(c) \text{ analogously}.
$$

**Seed / renderer fields** (`cells_flat`):

| Key | Value |
|-----|--------|
| `rho_peak` | $\rho_{\mathrm{peak}}(c)$ — **seed drive** (coherent $|Z_2^w|$ on $z_2$) |
| `rho_coherence` | $R(c) \in [0,1]$ — orientation agreement among top-$|z_2|$ pixels |
| `rho_total`, `z0` | $\rho_{\mathrm{total}}(c)$ — surround / $E_{\mathrm{rel}}$ diagnostics |
| `theta` | $\theta(c)$ |
| `cx_z2`, `cy_z2` | $h_{2m}$-weighted splat anchors |

Border cells → $0$ on $\rho_{\mathrm{total}}, \rho_{\mathrm{peak}}, \theta$.

---

## 4. Seed — NR peak + collinear readback (`ContourSeed` / `AndGateSeed`)

Let $R(c) = \rho_{\mathrm{peak}}(c)\,\mathrm{ok}(c)$ denote the masked peak drive.

**NR-normalised seed field**:

$$
\rho_{\mathrm{seed}}(c) = \frac{R(c)^2}{R(c)^2 + \eta_{\mathrm{seed}}^2 + \varepsilon}\,\mathrm{ok}(c).
$$

**Collinear readback** on $\rho_{\mathrm{seed}}$ (default `SEED.FACIL_MODE = "collinear"`):

$$
\rho_{\mathrm{coll}}(c) = \mathrm{relu}\!\left(
\frac{\sum_{\delta \in \mathcal{N}} w_\delta(c)\, \rho_{\mathrm{seed}}(c+\delta)\, a_\kappa(\Delta\theta_\delta)}
{\sum_{\delta \in \mathcal{N}} w_\delta(c) + \varepsilon}
\right),
$$
$$
w_\delta = G(|\delta|)\,\mathrm{pos}(\delta; \hat t_c), \quad
a_\kappa(\Delta) = \exp\!\bigl(\kappa_\theta(\cos\Delta - 1)\bigr),
$$
with $\hat t_c = (\cos\theta_c, \sin\theta_c)$ and $\Delta\theta$ the wrapped double-angle difference.

**Excitation**:

$$
e(c) = \beta_{\mathrm{seed}}\,\rho_{\mathrm{seed}}(c) + \beta_{\mathrm{coll}}\,\rho_{\mathrm{coll}}(c).
$$

**Surround** — center-excluded pool of $\rho_{\mathrm{seed}}$ (`broadside` mode weights toward the normal of $\theta$):

$$
S(c) = \langle\rho_{\mathrm{seed}}\rangle_{\mathcal{N}}(c).
$$

**Divisive cell export** ($\lambda$ is **not** squared):

$$
\rho(c) = \frac{e(c)^2}{e(c)^2 + \eta^2 + \lambda\, S(c)^2 + \varepsilon}\,\mathrm{ok}(c).
$$

**Learned** (softplus-positive): $\beta_{\mathrm{seed}}$, $\beta_{\mathrm{coll}}$, $\kappa_\theta$, $\eta_{\mathrm{seed}}$, $\eta$, $\lambda$, $\sigma_f$. Inits from `params.SEED`.

**Relative energy** (diagnostic only, from $\rho_{\mathrm{total}}$):

$$
E_{\mathrm{rel}}(c) = \frac{\rho_{\mathrm{total}}(c)}{\varepsilon + \langle\rho_{\mathrm{total}}\rangle_{\mathcal{N}}(c)}.
$$

Border cells → $0$ on $\rho, \theta$. **Orientation** $\theta(c)$ is read from L1; seed does not refine $\theta$.

---

## 5. Renderer — splat, footprint coherence, stencils, thinning head

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

**Amplitude** (unnormalized accumulation):

$$
\bar\rho(p) = \sum_c \rho_c\,\phi_c(p).
$$

Dominant orientation per pixel (scatter-max by $\rho_c\,\phi_c$):

$$
\theta^\star(p) = \theta_{c^\star(p)}, \qquad
c^\star(p) = \arg\max_c \rho_c\,\phi_c(p).
$$

**Step 3 — splat-footprint coherence** (renderer feature, not L1):

$$
\mathrm{coh}(p) = \frac{\sum_c \rho_c\,\phi_c(p)\,\cos^2\!\bigl(\theta_c - \theta^\star(p)\bigr)}
{\sum_c \rho_c\,\phi_c(p) + \varepsilon}.
$$

**Step 4 — stencils on $\bar\rho$** with unit tangent $\hat t = (\cos\theta^\star, \sin\theta^\star)$, normal $\hat n = (-\sin\theta^\star, \cos\theta^\star)$, learned spacings $s_t, s_n$:

$$
\mathrm{tang}_j(p) = \bar\rho\!\left(p + j\, s_t\, \hat t(p)\right), \qquad
\mathrm{norm}_j(p) = \bar\rho\!\left(p + j\, s_n\, \hat n(p)\right), \qquad j \in \{-4,\ldots,4\}.
$$

**Feature vector** $F_p \in \mathbb{R}^{20}$:

$$
F_p = \bigl[
\bar\rho,\, \mathrm{coh},\,
\mathrm{tang}_{-4},\ldots,\mathrm{tang}_{4},\,
\mathrm{norm}_{-4},\ldots,\mathrm{norm}_{4}
\bigr].
$$

**Thinning head**:

$$
\mathrm{gate}(p) = \sigma\!\bigl(W_2\,\mathrm{ReLU}(W_1 F_p + b_1) + b_2\bigr), \qquad
\hat B(p) = \bar\rho(p)\,\mathrm{gate}(p).
$$

$\mathrm{MLP}$: **20 → 12 → 1**. Output is cropped to original content size $(H_0, W_0)$.

---

## 6. Training (`train.py`)

- **Disk cache** (`precompute_image`): padded RGB `img`, `l0_pix` (fallback when `--no-l0-metric`), `border_mask`, GT, `proj_info`. Invalidation: `TRAIN.L0_CACHE_VERSION` (currently **2**; bump when L0/pad/cache schema changes).
- **Each step** (`prepare_batch`): if `L0.LEARNED_METRIC`, **live L0** from cached RGB + learned $W$ → live L1 moment pooling → seed + renderer. Gradients flow through $W$ when enabled.

**Loss** (defaults `TRAIN.LAM_DICE=1`, `TRAIN.LAM_BCE=0`): weighted sum of soft-Dice and/or BCE on the **η± edge band** — valid pixels where $\mathrm{GT} \ge \eta_{\mathrm{pos}}$ or $\mathrm{GT} < \eta_{\mathrm{neg}}$ (default $\eta_{\mathrm{pos}} = \eta_{\mathrm{neg}} = 0.5$).

Checkpoints store `{"model_state": state_dict}` (`intermediate.pt`, `final.pt`).

---

## 7. Learned parameter count (current architecture)

| Block | Count | Notes |
|------:|------:|------|
| $W$ (`L0LearnedMetric`) | 9 | $3\times3$ RGB metric; omitted with `--no-l0-metric` |
| $\beta_{\mathrm{seed}}, \beta_{\mathrm{coll}}, \kappa_\theta, \eta_{\mathrm{seed}}, \eta, \lambda, \sigma_f$ | 7 | Seed NR peak + readback + divisive readout |
| $\tilde\sigma_\perp, \tilde\sigma_\parallel, s_t, s_n$ | 4 | Splat widths + stencil spacings |
| $\mathrm{MLP}_{\mathrm{thin}}$ (20→12→1) | 265 | $20\cdot 12 + 12 + 12 + 1$ |
| **Renderer subtotal** | **269** | |
| **Total (`StriateE2E`, default)** | **285** | 9 L0 + 7 seed + 269 renderer |

L0 $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ are **fixed** (`params.L0`). Legacy checkpoints may contain dropped keys; `upgrade_model_state_dict` / `upgrade_renderer_state_dict` + `load_state_dict(..., strict=False)` handle migration.

---

## 8. Module map

| Stage | Primary code |
|-------|----------------|
| $d_k$, NR, harmonics, $z_1, z_2$, learned $W$, interior mask | `hci/L0.py` |
| z₂ moment pooling, $\rho_{\mathrm{peak}}$, $\rho_{\mathrm{total}}$, $\theta$, $h_{2m}$ anchors | `hci/L1.py` |
| Association-field seed, $\rho_{\mathrm{seed}}$, $\rho_{\mathrm{coll}}$, cell $\rho$ export | `hci/seed.py` |
| Splat, footprint $\mathrm{coh}$, stencils, thinning head, NMS | `hci/renderer.py` |
| Cache, batching, loss, checkpoints | `train.py` |
| Single-image pipeline, diagnostics | `infer.py`, `hci/diagnostics_viz.py` |
| Test-set sweep (ODS / OIS / AP) | `test.py` |

---

## 9. Revision note

This document matches the **STRIATE** stack: **learned L0 metric** (optional, §2), **z₂ moment pooling** with $\rho_{\mathrm{peak}}$ seed drive (§3), **association-field seed** (§4), **anisotropic Gaussian splat** renderer (§5), **`StriateE2E`** (284 learned parameters with default L0 metric). L1 orientation coherence $R$ has been removed from compute and diagnostics; renderer splat-footprint $\mathrm{coh}(p)$ remains as a pixel feature for thinning.
