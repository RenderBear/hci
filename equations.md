# STRIATE — equations (code-aligned)

This file records the **notation and equations implemented in this repository**: `hci/L0.py`, `hci/L1.py`, `hci/seed.py`, `hci/renderer.py`, `train.py`, `infer.py`, `test.py`. Hyperparameters and defaults live in `params.py` (`L0`, `L1`, `SEED`, `RENDER`, `TRAIN`, …).

---

## 1. End-to-end pipeline

1. **L0** — RGB directional differences → per-direction min subtraction → independent Naka–Rushton per channel with **fixed** $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ (`L0.ETA_LUM`, `L0.ETA_CHR`) and gain $\gamma$ (`L0.GAMMA`). Produces harmonic stack $s$, magnitudes $h_{1m}, h_{2m}$, split $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$, and complex fields $z_1, z_2$ (`z_from_l0_harmonics`). With **`L0.LEARNED_METRIC`** (default), a **learned** $3\times3$ matrix $W$ (`L0LearnedMetric`) replaces the fixed lum/chr split for distance; L0 is recomputed **live each training step** from cached RGB. Without it, L0 uses the fixed orthonormal lum/chr split and may be precomputed into the disk cache.
2. **L1** — From L0 pixel field $z_2$, pool over $P\times P$ patches → **$K$ orientation bins** (von Mises on $\theta_p$) giving $\rho^{(k)}, a_x^{(k)}, a_y^{(k)}, \mathrm{coh}^{(k)}$, plus legacy scalar moments $\rho_{\mathrm{total}}, \rho_{\mathrm{peak}}=|Z_2|, \theta, h_{2m}$ anchors. **Runs live** each training step (`run_moments_cells_flat` in `prepare_batch`) with **`seed.kappa_vm`** passed in so $\kappa_{\mathrm{vm}}$ is trainable end-to-end.
3. **Seed** (`AndGateSeed` / `ContourSeed`) — **Per orientation bin** $k$: $\rho_{\mathrm{NR}}^{(k)} = (\rho^{(k)})^2/((\rho^{(k)})^2+\eta_z^2)$; same-bin collinear readback $\rho_{\mathrm{coll}}^{(k)}$; orthogonal-bin surround $S^{(k)}$ via fixed matrix $B$; excitation $e^{(k)} = \beta_{\mathrm{seed}}\rho_{\mathrm{NR}}^{(k)} + \beta_{\mathrm{coll}}\rho_{\mathrm{coll}}^{(k)}$; divisive readout $\rho_{\mathrm{out}}^{(k)} = (e^{(k)})^2/((e^{(k)})^2 + \eta_{\mathrm{readout}}^2 + \lambda (S^{(k)})^2)$. **Export** scalar cell $\rho$ via STE hard-max over bins; $\theta$ via double-angle soft blend; splat anchors from argmax bin. Also exports the **per-bin tensor** $\rho_{\mathrm{out}}^{(k)}$ (gradient-bearing) which is the renderer's primary input. (**Learned:** $\kappa_{\mathrm{vm}}, \eta_z, \beta_{\mathrm{seed}}, \beta_{\mathrm{coll}}, \eta_{\mathrm{readout}}, \lambda, \sigma_f, \sigma_S$.)
4. **Renderer** (`ModulationRenderer`) — **FBP-style filtered back-projection** of the per-bin tensor. Per active $(c, k)$ pair: 5 contextual features $F^{(k)}_c$ → MLP $5\to 12\to 4$ → bounded corrections $(\kappa, e, \delta_n, \log\alpha)$; sparsity gate $g^{(k)}_c$; stroke synthesis via **learned 1D reconstruction kernels** $h_\perp, h_\parallel$ sampled at continuous query positions; **noisy-OR** aggregation across ALL $(c, k)$:
   $$\hat B(p) = 1 - \prod_{c,k} \bigl(1 - \alpha^{(k)}_c\, g^{(k)}_c\, \rho_{\mathrm{out}}^{(k)}(c)\, f^{(k)}_c(p)\bigr).$$
   The renderer applies no thinning or divisive suppression; precision lives at the seed.

At inference, optional **ridge NMS** (`ridge_nms`) thins $\hat B$ using per-pixel dominant orientation $\theta^\star(p)$.

---

## 2. L0 — split luminance / chrominance harmonics

Eight offsets $\delta_k \in \mathbb{Z}^2$ (`L0.OFFSETS`).

### Fixed metric (default split, or init of learned $W$)

RGB maps to luminance $L=(R{+}G{+}B)/3$ and chrominance $C=(R,G,B)-L\mathbf{1}$. Directional differences:

$$
d_k^{\mathrm{lum}}(p) = \bigl|L(p) - L(p+\delta_k)\bigr|, \qquad
d_k^{\mathrm{chr}}(p) = \bigl\|C(p) - C(p+\delta_k)\bigr\|_2 .
$$

### Learned metric (`L0LearnedMetric`, $M = W^\top W$)

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

## 3. L1 — per-cell z₂ moments and orientation bins

Patch size $P$ (`L1.PATCH_SIZE`, default 5), stride $S = P - \texttt{patch\_overlap}$ (`L1.PATCH_OVERLAP`, default 3) → cell grid $(n_H, n_W)$. A cell is **border** when the mean border mask over its patch exceeds `L1.BORDER_PATCH_MAX_FRAC`.

Number of orientation bins $K$ (`L1.NUM_ORIENT_BINS`, default 8). Bin centers:

$$
\bar\theta_k = \frac{k\pi}{K}, \qquad k = 0,\ldots,K-1.
$$

$\bar\theta_k$ is a **gradient angle** (the argument of $z_2$, which is the doubled-angle complex field). The tangent direction in $(\mathrm{dy}, \mathrm{dx}) = (\mathrm{row}, \mathrm{col})$ order is $\hat t_k = (\cos\bar\theta_k, \sin\bar\theta_k)$; the normal is $\hat n_k = (-\sin\bar\theta_k, \cos\bar\theta_k)$. The renderer's frame projections use this convention.

### 3.1 Legacy scalar moments (diagnostics + $E_{\mathrm{rel}}$)

From the L0 pixel field $z_2(p) = s_2(p) + i\,s_3(p)$, **unweighted** patch sums:

$$
Z_2(c) = \sum_{p \in \mathrm{patch}(c)} z_2(p), \qquad
\rho_{\mathrm{peak}}(c) = |Z_2(c)|, \qquad
\rho_{\mathrm{total}}(c) = \sum_{p \in \mathrm{patch}(c)} |z_2(p)|.
$$

**Orientation** (same moment as $|Z_2|$):

$$
\theta(c) = \tfrac{1}{2}\operatorname{atan2}\!\bigl(\Im Z_2, \Re Z_2 + \varepsilon\bigr).
$$

**Diagnostics:** `rho_coherence` stores $\rho_{\mathrm{peak}}/\max(\rho_{\mathrm{total}},\varepsilon)$ clamped to $[0,1]$.

**$h_{2m}$ splat anchors** (legacy fallback; stored as `cx_z2`, `cy_z2` before seed overwrites with per-bin gather):

$$
c_x(c) = \frac{\sum_{p \in \mathrm{patch}(c)} h_{2m}(p)\, x(p)}{\sum_{p \in \mathrm{patch}(c)} h_{2m}(p) + \varepsilon}, \qquad
c_y(c) \text{ analogously}.
$$

On border cells, legacy anchors fall back to the cell center $(c_x^{\mathrm{cell}}, c_y^{\mathrm{cell}})$.

### 3.2 Orientation bins (von Mises, primary seed input)

Per pixel in a patch, $\theta_p = \tfrac{1}{2}\operatorname{atan2}(\Im z_2, \Re z_2 + \varepsilon)$. Von Mises kernel in the **doubled angle** (peak-normalized to 1 at $\theta_p = \bar\theta_k$):

$$
g_k(p) = \exp\!\Bigl(\kappa_{\mathrm{vm}}\bigl(\cos\bigl(2(\theta_p - \bar\theta_k)\bigr) - 1\bigr)\Bigr).
$$

With $\mathrm{ok}_p = 1$ on interior patch pixels and $|z_2(p)|$ the pixel magnitude:

$$
\rho^{(k)}(c) = \sum_{p \in \mathrm{patch}(c)} |z_2(p)|\, g_k(p)\, \mathrm{ok}_p,
$$
$$
\mathrm{coh}^{(k)}(c) = \frac{\rho^{(k)}(c)}{\max\!\bigl(\sum_p |z_2(p)|\,\mathrm{ok}_p,\, \varepsilon\bigr)}.
$$

**Per-bin sub-pixel anchors** (same weights $|z_2| g_k \mathrm{ok}$):

$$
a_x^{(k)}(c) = \frac{\sum_p |z_2(p)|\, g_k(p)\, \mathrm{ok}_p\, x(p)}{\rho^{(k)}(c) + \varepsilon}, \qquad
a_y^{(k)}(c) \text{ analogously}.
$$

**Border handling:** $\rho^{(k)}, \mathrm{coh}^{(k)} \to 0$; $a_x^{(k)}, a_y^{(k)} \to (c_x^{\mathrm{cell}}, c_y^{\mathrm{cell}})$ (matching legacy anchor fallback). $\kappa_{\mathrm{vm}} \ge 0$ (clamped in L1; softplus on seed).

**$\kappa_{\mathrm{vm}}$ wiring:** stored as `seed._kappa_vm_raw`; L1 receives `seed.kappa_vm` on every `prepare_batch` / inference call — not referenced inside `seed.forward`.

### 3.3 `cells_flat` keys

| Key | Value |
|-----|--------|
| `rho_bin` | $(N, K)$ — $\rho^{(k)}$ per cell (input to seed) |
| `ax_bin`, `ay_bin` | $(N, K)$ — per-bin sub-pixel anchors |
| `rho_bin_coh` | $(N, K)$ — $\mathrm{coh}^{(k)}$ |
| `theta_bins` | $(K,)$ — $\bar\theta_k$ |
| `rho_peak` | $|Z_2(c)|$ — legacy diagnostic |
| `rho_total`, `z0` | $\rho_{\mathrm{total}}(c)$ — $E_{\mathrm{rel}}$ only |
| `theta` | $\theta(c)$ from $Z_2$ — legacy diagnostic |
| `cx_z2`, `cy_z2` | $h_{2m}$ anchors until seed overwrites from argmax bin |
| `rho_out_bins` | $(n_H, n_W, K)$ — per-bin seed readout; **renderer's primary input** (added by seed) |
| `is_border` | $(N,)$ bool |

---

## 4. Seed — per-bin NR, collinear, B-surround (`ContourSeed`)

Default path requires `rho_bin` in `cells_flat`. Let $\mathrm{ok}(c) = 1$ on interior cells, $R^{(k)}(c) = \rho^{(k)}(c)\,\mathrm{ok}(c)$.

### 4.1 Naka–Rushton per bin

Learned $\eta_z > 0$ (softplus; init `SEED.ETA_Z_INIT`):

$$
\rho_{\mathrm{NR}}^{(k)}(c) = \frac{\bigl(R^{(k)}(c)\bigr)^2}{\bigl(R^{(k)}(c)\bigr)^2 + \eta_z^2 + \varepsilon}\,\mathrm{ok}(c).
$$

### 4.2 Same-bin collinear readback

For each bin $k$, tangent $\hat t_k = (\cos\bar\theta_k, \sin\bar\theta_k)$ in $(\mathrm{dy}, \mathrm{dx})$ order. Neighbor offset $\delta = (\Delta y, \Delta x)$:

$$
G_f(\delta) = \exp\!\Bigl(-\frac{|\delta|^2}{2\sigma_f^2}\Bigr), \qquad
\mathrm{pos}_t(\delta) = \frac{(\delta\cdot\hat t_k)^2}{|\delta|^2 + \varepsilon},
$$
$$
w_\delta = G_f(\delta)\,\mathrm{pos}_t(\delta), \qquad
\rho_{\mathrm{coll}}^{(k)}(c) = \mathrm{ReLU}\!\left(
\frac{\sum_{\delta\neq 0} w_\delta\,\rho_{\mathrm{NR}}^{(k)}(c+\delta)}
{\sum_{\delta\neq 0} w_\delta + \varepsilon}
\right).
$$

No $\kappa_\theta$ affinity term — orientation is already separated by bin index. $\sigma_f$ learned (softplus, `clamp_min(0.3)`).

### 4.3 B-weighted orthogonal surround

Fixed orthogonal affinity matrix:

$$
B_{jk} = \sin^2(\bar\theta_j - \bar\theta_k).
$$

Gaussian neighbor kernel $G_S(\delta) = \exp(-|\delta|^2 / 2\sigma_S^2)$, $\sigma_S$ learned (softplus, `clamp_min(0.3)`):

$$
\tilde S^{(k)}(c) = \sum_{\delta\neq 0} G_S(\delta)\,\sum_j B_{jk}\,\rho_{\mathrm{NR}}^{(j)}(c+\delta),
$$
$$
S^{(k)}(c) = \frac{\tilde S^{(k)}(c)}
{\bigl(\sum_{\delta\neq 0} G_S(\delta)\bigr)\,\bigl(\sum_j B_{jk}\bigr) + \varepsilon}.
$$

### 4.4 Excitation, divisive readout, cell export

$$
e^{(k)} = \beta_{\mathrm{seed}}\,\rho_{\mathrm{NR}}^{(k)} + \beta_{\mathrm{coll}}\,\rho_{\mathrm{coll}}^{(k)},
$$
$$
\rho_{\mathrm{out}}^{(k)} = \frac{\bigl(e^{(k)}\bigr)^2}{\bigl(e^{(k)}\bigr)^2 + \eta_{\mathrm{readout}}^2 + \lambda\,\bigl(S^{(k)}\bigr)^2 + \varepsilon}\,\mathrm{ok}(c).
$$

**Per-bin tensor export:** $\rho_{\mathrm{out}}^{(k)}$ is written to `cells_flat['rho_out_bins']` with shape $(n_H, n_W, K)$; this is the renderer's primary gradient-bearing input.

**Scalar export to renderer** (straight-through estimator, temperature $\tau =$ `SEED.RHO_STE_TAU`):

$$
\rho(c) = \max_k \rho_{\mathrm{out}}^{(k)} \;-\; \Bigl(\sum_k w_k \rho_{\mathrm{out}}^{(k)}\Bigr)_{\!\!\mathrm{stopgrad}} + \sum_k w_k \rho_{\mathrm{out}}^{(k)},
\quad w_k = \mathrm{softmax}\bigl(\rho_{\mathrm{out}}^{(k)}/\tau\bigr).
$$

**Orientation export** (double-angle soft blend over $\rho_{\mathrm{out}}^{(k)}$):

$$
\theta(c) = \tfrac{1}{2}\operatorname{atan2}\!\Bigl(
\sum_k \rho_{\mathrm{out}}^{(k)} \sin 2\bar\theta_k,\;
\sum_k \rho_{\mathrm{out}}^{(k)} \cos 2\bar\theta_k + \varepsilon
\Bigr).
$$

**Anchors:** $k^\star(c) = \arg\max_k \rho_{\mathrm{out}}^{(k)}$; $(c_x, c_y) = (a_x^{(k^\star)}, a_y^{(k^\star)})$, written to `cx_z2`, `cy_z2`.

**Learned** (softplus-positive): $\kappa_{\mathrm{vm}}, \eta_z, \beta_{\mathrm{seed}}, \beta_{\mathrm{coll}}, \eta_{\mathrm{readout}}, \lambda, \sigma_f, \sigma_S$. Inits from `params.SEED` / `params.L1`.

**Relative energy** (diagnostic only, from $\rho_{\mathrm{total}}$; center-excluded Gaussian surround):

$$
E_{\mathrm{rel}}(c) = \frac{\rho_{\mathrm{total}}(c)}{\varepsilon + \langle\rho_{\mathrm{total}}\rangle_{\mathcal{N}}(c)}.
$$

### 4.5 Legacy scalar path (`_forward_legacy`)

Used when `rho_bin` is absent. Drive $R(c) = \rho_{\mathrm{peak}}(c)\,\mathrm{ok}(c)$; scalar $\rho_{\mathrm{NR}}, \rho_{\mathrm{coll}}$ with $\kappa_\theta$ orientation affinity $a_\kappa(\Delta\theta) = \exp(\kappa_\theta(\cos\Delta\theta - 1))$; surround from `SEED.SURROUND_MODE` (`isotropic` Gaussian mean or `broadside` normal-weighted pool). Single-bin divisive readout; no STE export. Retained for checkpoint compatibility and ablations.

---

## 5. Renderer — FBP-style filtered back-projection

**Algebraic framing.** The full stack factors as **forward Radon** (L0 + L1: directional sampling and patch-pooled orientation bins) → **gain control** (seed: NR + collinear + surround divisive readout) → **inverse Radon** (renderer: filtered back-projection of the per-bin tensor). The renderer's job is to invert L1's per-cell orientation pooling — placing the cell's per-bin energy back onto the pixel grid along the direction $\bar\theta_k$ — through a *learned* radial filter, exactly as FBP applies a learned ramp filter before back-projecting Radon measurements.

**Inputs from seed / L1:**

$$
\rho_{\mathrm{out}}^{(k)}(c) \in \mathbb{R},\quad
a^{(k)}(c) = (a_x^{(k)}, a_y^{(k)}) \in \mathbb{R}^2,\quad
\bar\theta_k = k\pi/K,\quad
\mathrm{ok}(c) \in \{0, 1\}.
$$

$\rho_{\mathrm{out}}^{(k)}$ carries gradient back to the seed. Anchors and $\bar\theta_k$ are detached / fixed buffers.

**Frame.** In $(\mathrm{dy}, \mathrm{dx}) = (\mathrm{row}, \mathrm{col})$ order, the tangent and normal unit vectors are:

$$
\hat t_k = (\cos\bar\theta_k,\; \sin\bar\theta_k), \qquad
\hat n_k = (-\sin\bar\theta_k,\; \cos\bar\theta_k).
$$

In image $(\mathrm{col}, \mathrm{row})$ order: $\hat t_k^{\mathrm{img}} = (\sin\bar\theta_k, \cos\bar\theta_k)$, $\hat n_k^{\mathrm{img}} = (\cos\bar\theta_k, -\sin\bar\theta_k)$.

Deposit half-width: $H_w = \mathrm{clamp}(\lceil \texttt{DEPOSIT\_HALF\_WIDTH\_STRIDES} \cdot S\rceil,\; [\texttt{MIN}, \texttt{MAX}])$. Filters have $2 H_w + 1$ taps each. $H_w$ is **fixed at module construction** so the learned 1D filters have fixed length.

### 5.1 Per-(cell, bin) features

Five scalars per $(c, k)$, all computed from `rho_out_bins` (detached) and bin geometry:

| Index | Feature | Role |
|------:|---------|------|
| $F_0$ | $\rho_{\mathrm{out}}^{(k)}(c)$ | self-bin energy |
| $F_1$ | $\bigl\langle\rho_{\mathrm{out}}^{(k)} \cdot \mathrm{pos}_t^{(k)}\bigr\rangle_{\mathcal{N}}\big/\bigl(\langle \mathrm{pos}_t^{(k)}\rangle_{\mathcal{N}} + \varepsilon_f\bigr)$ | collinear support |
| $F_2$ | $\bigl\langle \rho_{\mathrm{out}}^{(k)} \cdot (\delta\cdot\hat t_k)\bigr\rangle_{\mathcal{N}}\big/\bigl(\bigl\langle\rho_{\mathrm{out}}^{(k)}\cdot \lvert\delta\cdot\hat t_k\rvert\bigr\rangle_{\mathcal{N}} + \varepsilon_f\bigr) \in [-1, 1]$ | signed tangent asymmetry → drives $e$ |
| $F_3$ | $\sum_{j \neq k} \rho_{\mathrm{out}}^{(j)}(c)$ | competing-bin energy |
| $F_4$ | $\bigl\langle \rho_{\mathrm{out}}^{(k)} \cdot (\delta\cdot\hat n_k)\bigr\rangle_{\mathcal{N}}\big/\bigl(\bigl\langle\rho_{\mathrm{out}}^{(k)}\cdot \lvert\delta\cdot\hat n_k\rvert\bigr\rangle_{\mathcal{N}} + \varepsilon_f\bigr) \in [-1, 1]$ | signed normal asymmetry → drives $\delta_n$ |

where $\delta = (\Delta y, \Delta x)$ ranges over the 3×3 neighborhood excluding self, $\mathrm{pos}_t^{(k)}(\delta) = (\delta\cdot\hat t_k)^2/|\delta|^2$, and $\varepsilon_f = 0.05$ is the softfloor.

Border cells: $F_i \to 0$.

### 5.2 Correction MLP (5 → 12 → 4, shared across (c, k))

$$
\mathbf{u}^{(k)}(c) = W_2\,\mathrm{ReLU}(W_1 F^{(k)}(c) + b_1) + b_2 \in \mathbb{R}^4.
$$

Bounded outputs via tanh on learned softplus-positive bounds:

$$
\kappa^{(k)} = \kappa_{\max}\tanh(u_0^{(k)}),\quad
e^{(k)} = e_{\max}\tanh(u_1^{(k)}),\quad
\delta_n^{(k)} = \delta_{n,\max}\tanh(u_2^{(k)}),\quad
\log\alpha^{(k)} = \alpha_{\mathrm{range}}\tanh(u_3^{(k)}),
$$
$$
\alpha^{(k)}(c) = \exp\bigl(\log\alpha^{(k)}(c)\bigr) \in [e^{-\alpha_{\mathrm{range}}}, e^{+\alpha_{\mathrm{range}}}].
$$

$\kappa$: signed curvature (1/pixel). $e$: signed tangent shift of stroke vertex (pixels). $\delta_n$: signed normal-direction anchor correction (pixels) — the new 2D anchor correction. $\alpha$: per-(cell, bin) amplitude modulation around 1.

### 5.3 Sparsity gate

$$
g^{(k)}(c) = \sigma\!\Bigl(\alpha_g\bigl(\rho_{\mathrm{out}}^{(k)}(c) - \tau\cdot\max_j \rho_{\mathrm{out}}^{(j)}(c)\bigr)\Bigr)\,\mathrm{ok}(c).
$$

$\tau \in [0, 1]$ via sigmoid on a raw param; $\alpha_g > 0$ via softplus.

**Active set:** $\mathcal{A} = \{(c, k) : g^{(k)}(c) > g_{\min}\text{ and }\rho_{\mathrm{out}}^{(k)}(c) > \rho_{\min}\}$. $g_{\min} = 10^{-3}$, $\rho_{\min} = 10^{-4}$ are fixed (not learned). Only active $(c, k)$ pairs are evaluated downstream.

### 5.4 Learned 1D reconstruction kernels

Both filters defined on integer offsets $m \in \{-H_w, \ldots, +H_w\}$, evaluated at continuous query positions via linear interpolation. Both shared across all $(c, k)$.

**Radial filter $h_\perp[m]$** (perpendicular to tangent — FBP-analogue ramp filter):

Raw vector $\phi_\perp \in \mathbb{R}^{H_w + 1}$ indexed by $|m|$, with:
- $\phi_\perp[0] = \mathrm{softplus}(\phi_\perp^{\mathrm{raw}}[0])$  — positive peak amplitude
- $\phi_\perp[m] = \phi_\perp^{\mathrm{raw}}[m]$ for $|m| \ge 1$  — free-sign side-lobes

Even symmetry: $h_\perp[m] = \phi_\perp[|m|]$ for $m \in [-H_w, +H_w]$.

Free-sign side-lobes give $h_\perp$ the algebraic capacity of an FBP ramp filter — negative regions surrounding a positive central peak produce sharpening via suppression of off-edge claims.

Init: $\phi_\perp[m] \approx \exp(-m^2/2\sigma_\perp^2)$ with $\sigma_\perp = $ `SIGMA_PERP_INIT` (Gaussian at start; learns freely).

**Longitudinal profile $h_\parallel[m]$** (along tangent — even, monotone-decay from peak):

Raw $\psi_\parallel \in \mathbb{R}^{H_w}$ and peak scalar $h_\parallel^{\mathrm{peak,\,raw}}$.

$$
\mathrm{peak} = \mathrm{softplus}(h_\parallel^{\mathrm{peak,\,raw}}),\qquad
r[j] = \sigma(\psi_\parallel[j]) \in (0, 1)\ \text{for}\ j = 0, \ldots, H_w - 1,
$$
$$
\tilde h_\parallel[0] = 1,\qquad
\tilde h_\parallel[m] = \prod_{i = 0}^{|m| - 1} r[i]\ \text{for}\ |m| \ge 1,\qquad
h_\parallel[m] = \mathrm{peak}\cdot\tilde h_\parallel[|m|].
$$

Cumulative product of sigmoid-bounded ratios in $(0, 1)$ guarantees even symmetry, peak at center, and monotone decay outward.

Init: $\tilde h_\parallel[m] \approx \exp(-m^2/2\sigma_\parallel^2)$ with $\sigma_\parallel = $ `SIGMA_PAR_INIT`.

**Kernel sampling at continuous queries.** For $u \in \mathbb{R}$:

$$
h_\bullet(u) = \begin{cases}
(1 - \{u\})\,h_\bullet[\lfloor u\rfloor] + \{u\}\,h_\bullet[\lceil u\rceil] & u \in [-H_w, +H_w] \\
0 & \text{otherwise}
\end{cases}
$$

where $\{u\} = u - \lfloor u\rfloor$. This is the discrete-to-continuous sampling analogous to FBP's interpolation when back-projecting at arbitrary angles. Differentiable w.r.t. both kernel taps and query positions.

### 5.5 Per-(c, k) stroke synthesis

For active $(c, k) \in \mathcal{A}$ and pixel $p = (p_y, p_x)$ in the deposit window centered on $\lfloor\tilde a^{(k)}(c)\rfloor$:

**Anchor with normal correction** (in image $(\mathrm{col}, \mathrm{row})$ coords):

$$
\tilde a_x^{(k)}(c) = a_x^{(k)}(c) + \delta_n^{(k)}(c)\cos\bar\theta_k,\qquad
\tilde a_y^{(k)}(c) = a_y^{(k)}(c) - \delta_n^{(k)}(c)\sin\bar\theta_k.
$$

**Frame projection:**

$$
\Delta y = p_y - \tilde a_y^{(k)},\quad \Delta x = p_x - \tilde a_x^{(k)},
$$
$$
s =  \Delta y\cos\bar\theta_k + \Delta x\sin\bar\theta_k,\quad
n = -\Delta y\sin\bar\theta_k + \Delta x\cos\bar\theta_k.
$$

**Tangent shift and curvature bend:**

$$
\tilde s = s - e^{(k)}(c),\qquad
n_c = n - \tfrac{1}{2}\kappa^{(k)}(c)\,\tilde s^2.
$$

**Stroke profile** (separable, via interpolated 1D kernel sampling):

$$
f^{(k)}_c(p) = \max\!\bigl(0,\; h_\perp(n_c)\cdot h_\parallel(\tilde s)\bigr).
$$

The ReLU keeps the claim non-negative for noisy-OR probabilistic semantics; $h_\perp$'s negative side-lobes drive sharpening by suppressing $f$ to 0 on off-edge pixels rather than by subtraction.

### 5.6 Per-(c, k) claim

$$
c^{(k)}_c(p) = \alpha^{(k)}(c)\cdot g^{(k)}(c)\cdot \rho_{\mathrm{out}}^{(k)}(c)\cdot f^{(k)}_c(p),
$$

clamped to $[0,\,1 - 10^{-5}]$ for log stability.

### 5.7 Noisy-OR aggregation

Across all active $(c, k)$:

$$
\hat B(p) = 1 - \prod_{(c, k) \in \mathcal{A}} \bigl(1 - c^{(k)}_c(p)\bigr)
= 1 - \exp\!\Bigl(\sum_{(c, k) \in \mathcal{A}} \log\bigl(1 - c^{(k)}_c(p)\bigr)\Bigr).
$$

Pixels outside every active deposit window contribute $c = 0$, so $\hat B(p) = 0$ on untouched pixels.

### 5.8 Per-pixel dominant orientation (for inference NMS)

$$
(c^\star(p), k^\star(p)) = \arg\max_{(c, k) \in \mathcal{A}} c^{(k)}_c(p),\qquad
\theta^\star(p) = \bar\theta_{k^\star(p)}.
$$

Tracked during the scatter; no gradient.

### 5.9 Why FBP-like and what this buys vs. the prior renderer

**Algebraic.** Per-bin reading of $\rho_{\mathrm{out}}^{(k)}$ at the bin's anchor with a *filter* (rather than a fixed mollifier) closes the algebraic loop: L1 forward-projects pixel evidence onto bins via von Mises pooling; the renderer back-projects bin energy onto pixels via the analogous interpolation, optionally sharpened by negative side-lobes in $h_\perp$. Multi-curve at junctions falls out of the noisy-OR across $(c, k)$ — a single cell with two active bins synthesizes two strokes at two angles.

**Analytic.** The 1D kernels learn the *shape* of the inverse — width, side-lobe structure, longitudinal extent — instead of being fixed at construction time. The MLP-driven corrections $(\kappa, e, \delta_n)$ are per-(cell, bin) refinements on top of the rigid back-projection geometry; the amplitude modulation $\alpha$ lets the synthesis depart from strict linearity in $\rho^{(k)}$ when context supports it.

The renderer applies **no thinning, splat coherence, or divisive suppression** — precision lives upstream at the seed; the renderer's job is to render cleanly.

---

## 6. Training (`train.py`)

- **Disk cache** (`precompute_image`): padded RGB `img`, `l0_pix` (fallback when `--no-l0-metric`), `border_mask`, GT, `proj_info`. Invalidation: `TRAIN.L0_CACHE_VERSION` (currently **2**; bump when L0/pad/cache schema changes — not L1 binning or seed).
- **Each step** (`prepare_batch`): if `L0.LEARNED_METRIC`, **live L0** from cached RGB + learned $W$ → live L1 (`kappa_vm=seed.kappa_vm`) → seed (which writes `rho_out_bins` into `cells_flat`) → renderer. Gradients flow through $W$, $\kappa_{\mathrm{vm}}$, all seed scalars, and all renderer parameters.

**Loss** (defaults `TRAIN.LAM_DICE=0`, `TRAIN.LAM_BCE=1`): weighted sum of soft-Dice and/or BCE on the **η± edge band** — valid pixels where $\mathrm{GT} \ge \eta_{\mathrm{pos}}$ or $\mathrm{GT} < \eta_{\mathrm{neg}}$ (default $\eta_{\mathrm{pos}} = \eta_{\mathrm{neg}} = 0.5$).

Checkpoints store `{"model_state": state_dict}` (`intermediate.pt`, `final.pt`). `upgrade_model_state_dict` / `upgrade_renderer_state_dict` migrate legacy keys: incompatible-shape correction-MLP weights and obsolete keys from prior renderer architectures are stripped on load.

---

## 7. Learned parameter count (current architecture)

| Block | Count | Notes |
|------:|------:|------|
| $W$ (`L0LearnedMetric`) | 9 | $3\times3$ RGB metric; omitted with `--no-l0-metric` |
| $\kappa_{\mathrm{vm}}, \eta_z, \beta_{\mathrm{seed}}, \beta_{\mathrm{coll}}, \kappa_\theta, \eta_{\mathrm{readout}}, \lambda, \sigma_f, \sigma_S$ | 9 | Seed ($\kappa_\theta$ used in legacy path only) |
| Correction MLP $5\to 12\to 4$ | 124 | $5\cdot 12 + 12 + 12\cdot 4 + 4$ |
| $\kappa_{\max}, e_{\max}, \delta_{n,\max}, \alpha_{\mathrm{range}}$ | 4 | correction bounds (softplus) |
| $\tau, \alpha_g$ | 2 | sparsity gate (sigmoid, softplus) |
| $\phi_\perp \in \mathbb{R}^{H_w + 1}$ | 5 | radial filter taps (peak softplus, sides free-sign), at $H_w = 4$ |
| $\psi_\parallel \in \mathbb{R}^{H_w}, h_\parallel^{\mathrm{peak}}$ | 5 | longitudinal monotone-decay logits + peak softplus, at $H_w = 4$ |
| **Renderer subtotal** | **140** | |
| **Total (`StriateE2E`, default)** | **158** | 9 L0 + 9 seed + 140 renderer |

L0 $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ are **fixed** (`params.L0`). Fixed buffers on seed: `theta_bins`, `B_orth` ($K\times K$). Filter half-width $H_w$ is fixed at renderer construction from canonical L1 stride; changing `L1.PATCH_SIZE` or `PATCH_OVERLAP` requires fresh-init of the filters.

---

## 8. Module map

| Stage | Primary code |
|-------|----------------|
| $d_k$, NR, harmonics, $z_1, z_2$, learned $W$, interior mask | `hci/L0.py` |
| z₂ moments, von Mises bins $\rho^{(k)}$, per-bin anchors, legacy $|Z_2|$ | `hci/L1.py` |
| Per-bin NR, collinear, $B$-surround, per-bin tensor + STE export, legacy scalar path | `hci/seed.py` |
| Per-(c, k) features + MLP, learned 1D kernels, FBP synthesis, noisy-OR, ridge NMS | `hci/renderer.py` |
| Cache, batching, loss, checkpoints | `train.py` |
| Single-image pipeline, diagnostics | `infer.py`, `hci/diagnostics_viz.py` |
| Test-set sweep (ODS / OIS / AP) | `test.py` |

---

## 9. Revision note

This document matches the **STRIATE** stack as of the **K-tensor L1 + per-bin seed + FBP-style renderer** architecture:

- **L1:** $K = 8$ von Mises orientation bins with learned $\kappa_{\mathrm{vm}}$ (on seed, consumed by L1 each step). $\bar\theta_k$ is a gradient angle; tangent is $(\cos\bar\theta, \sin\bar\theta)$ in $(\mathrm{row}, \mathrm{col})$ order.
- **Seed:** per-bin NR → same-bin collinear → $B$-orthogonal surround → divisive readout; exports both per-bin tensor `rho_out_bins` (renderer input) and STE scalar $\rho$ + soft double-angle $\theta$ + argmax-bin anchors (legacy / diagnostics).
- **Renderer:** 5-feature per-(c, k) context → MLP $5\to 12\to 4$ → bounded $(\kappa, e, \delta_n, \log\alpha)$ corrections; sparsity gate; **learned 1D reconstruction kernels** $h_\perp$ (free-sign side-lobes) and $h_\parallel$ (monotone-decay) sampled by linear interpolation at continuous query; per-(c, k) Gaussian-stroke synthesis; noisy-OR across all active $(c, k)$ — **not** the earlier soft-indicator basis splat, anisotropic Gaussian splat + 20→12→1 thinning head, or the intermediate 82-param Gaussian-stroke renderer.

Legacy scalar seed path (`_forward_legacy`) and legacy L1-only inputs remain for compatibility but are not the default training/inference path when `rho_bin` is present.
