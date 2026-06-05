# STRIATE — equations (code-aligned)

This file records the **notation and equations implemented in this repository**: `hci/L0.py`, `hci/L1.py`, `hci/seed.py`, `hci/renderer.py`, `train.py`, `infer.py`, `test.py`. Hyperparameters and defaults live in `params.py` (`L0`, `L1`, `SEED`, `RENDER`, `TRAIN`, …).

---

## 1. End-to-end pipeline

1. **L0** — RGB directional differences → per-direction min subtraction → independent Naka–Rushton per channel with **fixed** $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ (`L0.ETA_LUM`, `L0.ETA_CHR`) and gain $\gamma$ (`L0.GAMMA`). Produces harmonic stack $s$, magnitudes $h_{1m}, h_{2m}$, split $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$, and complex fields $z_1, z_2$ (`z_from_l0_harmonics`). With **`L0.LEARNED_METRIC`** (default), a **learned** $3\times3$ matrix $W$ (`L0LearnedMetric`) replaces the fixed lum/chr split for distance; L0 is recomputed **live each training step** from cached RGB. Without it, L0 uses the fixed orthonormal lum/chr split and may be precomputed into the disk cache.
2. **L1** — From L0 pixel field $z_2$, pool over $P\times P$ patches → **$K$ orientation bins** (von Mises on $\theta_p$) giving $\rho^{(k)}, a_x^{(k)}, a_y^{(k)}, \mathrm{coh}^{(k)}$, plus legacy scalar moments $\rho_{\mathrm{total}}, \rho_{\mathrm{peak}}=|Z_2|, \theta, h_{2m}$ anchors. **Runs live** each training step (`run_moments_cells_flat` in `prepare_batch`) with **`seed.kappa_vm`** passed in so $\kappa_{\mathrm{vm}}$ is trainable end-to-end.
3. **Seed** (`AndGateSeed` / `ContourSeed`) — **Per orientation bin** $k$: $\rho_{\mathrm{NR}}^{(k)} = (\rho^{(k)})^2/((\rho^{(k)})^2+\eta_z^2)$; same-bin collinear readback $\rho_{\mathrm{coll}}^{(k)}$; orthogonal-bin surround $S^{(k)}$ via fixed matrix $B$; excitation $e^{(k)} = \beta_{\mathrm{seed}}\rho_{\mathrm{NR}}^{(k)} + \beta_{\mathrm{coll}}\rho_{\mathrm{coll}}^{(k)}$; divisive readout $\rho_{\mathrm{out}}^{(k)} = (e^{(k)})^2/((e^{(k)})^2 + \eta_{\mathrm{readout}}^2 + \lambda (S^{(k)})^2)$. **Export** scalar cell $\rho$ via STE hard-max over bins; $\theta$ via double-angle soft blend; splat anchors from argmax bin. (**Learned:** $\kappa_{\mathrm{vm}}, \eta_z, \beta_{\mathrm{seed}}, \beta_{\mathrm{coll}}, \eta_{\mathrm{readout}}, \lambda, \sigma_f, \sigma_S$.)
4. **Renderer** (`ModulationRenderer`) — Cell-grid $\theta$ combing and $\rho$-gated anchor smoothing; **6→16→8 DepositMLP** → basis weights; per-cell **soft-indicator** footprint $m_c(p)$; bilinear sub-pixel scatter; **noisy-OR** aggregation:
   $$\hat B(p) = 1 - \prod_c \bigl(1 - \rho_c\, m_c(p)\bigr).$$
   The renderer places mass only — no thinning or divisive suppression.

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
| `rho_bin` | $(N, K)$ — $\rho^{(k)}$ per cell |
| `ax_bin`, `ay_bin` | $(N, K)$ — per-bin sub-pixel anchors |
| `rho_bin_coh` | $(N, K)$ — $\mathrm{coh}^{(k)}$ |
| `theta_bins` | $(K,)$ — $\bar\theta_k$ |
| `rho_peak` | $|Z_2(c)|$ — legacy diagnostic |
| `rho_total`, `z0` | $\rho_{\mathrm{total}}(c)$ — $E_{\mathrm{rel}}$ only |
| `theta` | $\theta(c)$ from $Z_2$ — legacy / renderer smoothing input |
| `cx_z2`, `cy_z2` | $h_{2m}$ anchors until seed overwrites from argmax bin |

---

## 4. Seed — per-bin NR, collinear, B-surround (`ContourSeed`)

Default path requires `rho_bin` in `cells_flat`. Let $\mathrm{ok}(c) = 1$ on interior cells, $R^{(k)}(c) = \rho^{(k)}(c)\,\mathrm{ok}(c)$.

### 4.1 Naka–Rushton per bin

Learned $\eta_z > 0$ (softplus; init `SEED.ETA_Z_INIT`):

$$
\rho_{\mathrm{NR}}^{(k)}(c) = \frac{\bigl(R^{(k)}(c)\bigr)^2}{\bigl(R^{(k)}(c)\bigr)^2 + \eta_z^2 + \varepsilon}\,\mathrm{ok}(c).
$$

### 4.2 Same-bin collinear readback

For each bin $k$, tangent $\hat t_k = (\cos\bar\theta_k, \sin\bar\theta_k)$. Neighbor offset $\delta = (\Delta y, \Delta x)$:

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

## 5. Renderer — soft-indicator deposit + noisy-OR

**Step 1 — cell grid.** $\rho$-weighted double-angle $\theta$ smoothing (`RENDER.THETA_SMOOTH_PASSES`); $\rho$- and orientation-gated smoothing of anchors $(c_x, c_y)$. Coordinates and $\theta$ are **detached** before deposit (no coordinate gradients into seed/L0). $\rho$ and $\theta$ are also detached for the feature MLP path.

**Step 2 — per-cell features** $F_c \in \mathbb{R}^6$ from a 3×3 neighborhood (excluding self):

| Index | Feature |
|------:|---------|
| 0 | $\rho_c$ |
| 1 | $\langle\rho\rangle_{\mathcal{N}}$ (8-neighbor mean) |
| 2 | $\langle\rho\cos(2(\theta' - \theta_c))\rangle_{\mathcal{N}} / (\sum_{nbr}\rho + \varepsilon_{\mathrm{soft}})$ — collinearity |
| 3 | $\bigl|\langle\rho\sin(2(\theta' - \theta_c))\rangle_{\mathcal{N}}\bigr| / (\sum_{nbr}\rho + \varepsilon_{\mathrm{soft}})$ — disagreement |
| 4 | tangent $\rho$ asymmetry (projected neighbor offset · $\hat t_c$) |
| 5 | normal $\rho$ asymmetry (projected neighbor offset · $\hat n_c$) |

$\varepsilon_{\mathrm{soft}} = 0.05$ caps feature magnitude when neighborhoods are empty.

**Step 3 — DepositMLP** (6 → 16 → 8, ReLU hidden):

$$
\mathbf{w}_c = W_2\,\mathrm{ReLU}(W_1 F_c + b_1) + b_2 \in \mathbb{R}^8.
$$

Init bias favors basis $b_1$ (line along tangent). Learned **extension scale** `ext_scale` maps asymmetry features to per-cell offset $(\mathrm{ext}_s, \mathrm{ext}_n)$ for basis $b_7$.

**Step 4 — Gestalt basis** in cell-local coords $(s, n)$ (tangent / normal, normalized to deposit half-width):

$$
\begin{aligned}
b_0 &= 0 & b_1 &= -n^2 & b_2 &= -s^2 & b_3 &= -(s^2+n^2) \\
b_4 &= s & b_5 &= -(n-\kappa_0 s^2)^2 & b_6 &= -(n+\kappa_0 s^2)^2 & b_7 &= -((s,n) - (\mathrm{ext}_s,\mathrm{ext}_n))^2
\end{aligned}
$$

Logits over patch pixels $p$ in cell $c$:

$$
\ell_c(p) = \sum_{j=0}^{7} w_{c,j}\, b_j(s(p), n(p)), \qquad
m_c^{\mathrm{raw}}(p) = \exp\bigl(\ell_c(p) - \max_{p'} \ell_c(p')\bigr) \in [0,1].
$$

Optional **deposit envelope** (default on): $E(s,n)=\exp\bigl(-(s^2+n^2)/(2\sigma_E^2)\bigr)$ in half-width–normalized $(s,n)$, with hyperparameter `DEPOSIT_ENVELOPE_SIGMA` $=\sigma_E$ (set to $0$ to disable). Then

$$
m_c(p) = \frac{m_c^{\mathrm{raw}}(p)\, E(s(p),n(p))}{\max_{p'} \bigl(m_c^{\mathrm{raw}}(p')\, E(s(p'),n(p'))\bigr)} \in [0,1],
$$

so the peak is still $1$ but mass cannot stay large far from the anchor in cell frame — reducing flat-surface “tongues” from unbounded $b_1$ / non-decaying $b_4$ within the splat footprint.

**Step 5 — claim and bilinear scatter.** Per-cell claim $c_c(p) = \rho_c\, m_c(p)$, scattered to four pixel targets via bilinear weights of the sub-pixel anchor $(a_x \bmod 1, a_y \bmod 1)$.

**Step 6 — noisy-OR aggregation:**

$$
\hat B(p) = 1 - \prod_c \bigl(1 - c_c(p)\bigr)
= 1 - \exp\Bigl(\sum_c \log\bigl(1 - c_c(p)\bigr)\Bigr).
$$

Dominant orientation per pixel (scatter-max by bilinear claim magnitude):

$$
\theta^\star(p) = \theta_{c^\star(p)}, \qquad
c^\star(p) = \arg\max_c c_c(p).
$$

Deposit half-width: $\lceil \texttt{DEPOSIT\_HALF\_WIDTH\_STRIDES} \cdot S \rceil$, clamped to `[DEPOSIT_HALF_WIDTH_MIN, DEPOSIT_HALF_WIDTH_MAX]`. Envelope $\sigma_E$ defaults to `DEPOSIT_ENVELOPE_SIGMA` in `params.RENDER`. Output cropped to content size $(H_0, W_0)$.

The renderer applies **no thinning, splat coherence, or divisive suppression** — precision is upstream in the seed.

---

## 6. Training (`train.py`)

- **Disk cache** (`precompute_image`): padded RGB `img`, `l0_pix` (fallback when `--no-l0-metric`), `border_mask`, GT, `proj_info`. Invalidation: `TRAIN.L0_CACHE_VERSION` (currently **2**; bump when L0/pad/cache schema changes — not L1 binning or seed).
- **Each step** (`prepare_batch`): if `L0.LEARNED_METRIC`, **live L0** from cached RGB + learned $W$ → live L1 (`kappa_vm=seed.kappa_vm`) → seed + renderer. Gradients flow through $W$ and $\kappa_{\mathrm{vm}}$ when enabled.

**Loss** (defaults `TRAIN.LAM_DICE=0`, `TRAIN.LAM_BCE=1`): weighted sum of soft-Dice and/or BCE on the **η± edge band** — valid pixels where $\mathrm{GT} \ge \eta_{\mathrm{pos}}$ or $\mathrm{GT} < \eta_{\mathrm{neg}}$ (default $\eta_{\mathrm{pos}} = \eta_{\mathrm{neg}} = 0.5$).

Checkpoints store `{"model_state": state_dict}` (`intermediate.pt`, `final.pt`). `upgrade_model_state_dict` / `upgrade_renderer_state_dict` migrate legacy keys.

---

## 7. Learned parameter count (current architecture)

| Block | Count | Notes |
|------:|------:|------|
| $W$ (`L0LearnedMetric`) | 9 | $3\times3$ RGB metric; omitted with `--no-l0-metric` |
| $\kappa_{\mathrm{vm}}, \eta_z, \beta_{\mathrm{seed}}, \beta_{\mathrm{coll}}, \kappa_\theta, \eta_{\mathrm{readout}}, \lambda, \sigma_f, \sigma_S$ | 9 | Seed ($\kappa_\theta$ used in legacy path only) |
| DepositMLP (6→16→8) + `ext_scale` | 249 | $6\cdot16 + 16 + 16\cdot8 + 8 + 1$ |
| **Total (`StriateE2E`, default)** | **267** | 9 L0 + 9 seed + 249 renderer |

L0 $\eta_{\mathrm{lum}}, \eta_{\mathrm{chr}}$ are **fixed** (`params.L0`). Fixed buffers on seed: `theta_bins`, `B_orth` ($K\times K$).

---

## 8. Module map

| Stage | Primary code |
|-------|----------------|
| $d_k$, NR, harmonics, $z_1, z_2$, learned $W$, interior mask | `hci/L0.py` |
| z₂ moments, von Mises bins $\rho^{(k)}$, per-bin anchors, legacy $|Z_2|$ | `hci/L1.py` |
| Per-bin NR, collinear, $B$-surround, STE export, legacy scalar path | `hci/seed.py` |
| Soft-indicator deposit, noisy-OR, ridge NMS | `hci/renderer.py` |
| Cache, batching, loss, checkpoints | `train.py` |
| Single-image pipeline, diagnostics | `infer.py`, `hci/diagnostics_viz.py` |
| Test-set sweep (ODS / OIS / AP) | `test.py` |

---

## 9. Revision note

This document matches the **STRIATE** stack as of the **K-tensor L1 + per-bin seed + deposit renderer** architecture:

- **L1:** $K=8$ von Mises orientation bins with learned $\kappa_{\mathrm{vm}}$ (on seed, consumed by L1 each step).
- **Seed:** per-bin NR → same-bin collinear → $B$-orthogonal surround → divisive readout; STE hard-max $\rho$ export; soft double-angle $\theta$; argmax-bin anchors.
- **Renderer:** 6→16→8 basis MLP, soft-indicator footprint, bilinear scatter, noisy-OR — **not** the earlier anisotropic Gaussian splat + 20→12→1 thinning head.

Legacy scalar seed path (`_forward_legacy`) and legacy L1-only inputs remain for compatibility but are not the default training/inference path when `rho_bin` is present.
