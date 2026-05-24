# HCI — system equations

> **Repository note.** This file is the HCI system reference. The implementation in this repo follows the same staged story (cached directional \(d_k\), two-pass L0, L1 hypercolumn, GABA recurrence, renderer readout) and matches the **learned** pieces described here: collinear \(\sigma_d,\sigma_t\) from `HypercolumnSeed.collinear_sigmas` (softplus\((\alpha)\cdot R)\)); regional \(\eta\) from `EtaRegionalMLP` + `compute_eta_modulation_mlp` on pooled \((\bar\kappa,\bar z_0,\bar\rho_{\max})\); renderer `ThinningHead` **14→8→1** on \([h2m_{lum},h2m_{chr},\bar\rho,\bar\kappa,\text{tang}_5,\text{norm}_5]\) with learned \(s_t,s_n\). See `params.py` and the cited modules for exact shapes and hyperparameters.

## Architecture

Two-pass pipeline with regional gain feedback:

$$
\text{Pass 1:}\quad \text{L0}(\eta_0) \;\to\; \text{L1 hypercolumn} \;\to\; \text{GABA recurrence} \;\to\; \text{statistics}
$$

$$
\text{Gain feedback:}\quad \text{pool}(\text{statistics}) \;\to\; \text{MLP}_\eta \;\to\; \eta(p)
$$

$$
\text{Pass 2:}\quad \text{L0}\bigl(\eta(p)\bigr) \;\to\; \text{L1 hypercolumn} \;\to\; \text{GABA recurrence} \;\to\; \hat{B}
$$

---

## L0 — retinal / LGN preprocessing

Per-pixel, 8-connected.  Directional differences $d_k$ computed once, cached.

$$
d_k^{\text{lum}} = |L(p) - L(p+\delta_k)|, \qquad
d_k^{\text{chr}} = \|C(p) - C(p+\delta_k)\|_2
$$

$$
\tilde{d}_k = d_k - \min_j d_j
$$

$$
h_k^{\text{lum}} = \gamma\,\frac{\tilde{d}_k^2}{\eta_{\text{lum}}(p)^2 + \tilde{d}_k^2}, \qquad
h_k^{\text{chr}} = \gamma\,\frac{\tilde{d}_k^2}{\eta_{\text{chr}}(p)^2 + \tilde{d}_k^2}
$$

$$
z_2(p) = \sum_k (h_k^{\text{lum}} + h_k^{\text{chr}})\, e^{2i\varphi_k}
$$

$$
h_{2m}(p) = |z_2(p)|, \qquad \theta_h(p) = \tfrac{1}{2}\arg z_2(p)
$$

Pass 1: $\eta(p) = \eta_0$.  Pass 2: $\eta(p)$ from gain feedback.

---

## L1 — hypercolumn construction

Patches of size $P \times P$ at stride $S$ → cell grid $(n_H, n_W)$.
Each cell is a hypercolumn: $K$ orientation-tuned units (default $K = 24$).

### Oriented energy binning

$$
\rho_k^{\text{raw}}(c) = \sum_{p \in \text{patch}(c)} h_{2m}(p) \;\cos^2\!\bigl(\theta_h(p) - \bar{\theta}_k\bigr)
$$

### Min subtraction + NR squash

Per-cell, across the $K$ bins:

$$
\tilde{\rho}_k(c) = \rho_k^{\text{raw}}(c) - \min_j \rho_j^{\text{raw}}(c)
$$

$$
\rho_k^{(0)}(c) = \frac{\tilde{\rho}_k^2}{\tilde{\rho}_k^2 + \eta_z^2}
$$

$\eta_z = \text{softplus}(\tilde{\eta}_z)$ is learned (1 parameter).  The NR maps
each bin into $[0, 1]$ while preserving the cos² profile shape across ~6 active bins.

---

## GABA recurrence — cosine-similarity gating

Recurrent lateral dynamics on the $(K, n_H, n_W)$ representation.  Each pass
computes a cosine similarity between each cell's orientation profile and the
pooled neighborhood profile, then multiplicatively gates all bins.

### Kernels (learned shape)

$K$ depthwise kernels, parameterized by learned $\alpha_d$ and $\alpha_t$:

$$
\sigma_d = \alpha_d \cdot R, \qquad \sigma_t = \alpha_t \cdot R
$$

$$
W_k(\Delta i, \Delta j) = \exp\!\Bigl(-\frac{\|\Delta\|^2}{2\sigma_d^2}\Bigr)
\cdot \exp\!\Bigl(-\frac{d_\perp^2}{2\sigma_t^2}\Bigr)
$$

$$
d_\perp = \Delta j\cos\bar{\theta}_k - \Delta i\sin\bar{\theta}_k
$$

Centre excluded, zero beyond $R$.  Kernels are rebuilt each forward pass
from the learned $\alpha_d, \alpha_t$ (via softplus on raw parameters).

### Per-pass update

**For** $t = 0, \dots, T-1$:

**Stage A — depthwise convolution:**

$$
S_k^{(t)}(c) = \bigl(W_k * \rho_k^{(t)}\bigr)(c) \qquad \text{(one conv2d call, groups=}K\text{)}
$$

**Stage B — cosine similarity:**

The cell's orientation profile $\boldsymbol{\rho}^{(t)}(c) = [\rho_0, \dots, \rho_{K-1}]$
is compared against the neighborhood's pooled profile
$\mathbf{S}^{(t)}(c) = [S_0, \dots, S_{K-1}]$:

$$
\kappa^{(t)}(c) = \frac{\sum_{k=0}^{K-1} \rho_k^{(t)}(c) \cdot S_k^{(t)}(c)}
{\sqrt{\sum_k \bigl(\rho_k^{(t)}(c)\bigr)^2} \;\cdot\;
\sqrt{\sum_k \bigl(S_k^{(t)}(c)\bigr)^2} \;+\; \epsilon}
$$

Clamped to $[0, 1]$.  This is a single scalar per cell — "how well does my
orientation profile match my neighborhood's?"

**Stage C — modulate (all bins equally):**

$$
\rho_k^{(t+1)}(c) = \rho_k^{(t)}(c) \cdot \kappa^{(t)}(c)
$$

### Compact per-pass update

$$
\boxed{
\rho_k^{(t+1)}(c) = \rho_k^{(t)}(c) \;\cdot\;
\frac{\boldsymbol{\rho}^{(t)}(c) \cdot \mathbf{S}^{(t)}(c)}
{\|\boldsymbol{\rho}^{(t)}(c)\| \;\|\mathbf{S}^{(t)}(c)\| + \epsilon}
}
$$

### Why cosine similarity works with NR-squashed inputs

The cosine similarity compares **profile shape**, not magnitude.  After
NR squash, each cell has a K-vector in $[0, 1]^K$ with ~6 active bins from
the cos² tuning.  The similarity measures whether the cell's orientation
distribution matches the neighborhood's — independent of absolute scale.

- **Straight edge:** profile peaks at bin $k$, tangent neighbors also peak
  at $k$ → $\mathbf{S}$ peaks at $k$ → cosine ≈ 1 → preserved.
- **Texture:** profile peaks at $k$, neighbors peak at random bins →
  $\mathbf{S}$ is spread flat → cosine ≈ 0.3 → suppressed.
- **Orientation boundary:** profile peaks at $k$, neighbors peak at $k'$ →
  $\mathbf{S}$ peaks at $k'$ → cosine ≈ cos² overlap → partially suppressed.

### Properties

- **Shape-based:** NR squash doesn't kill dynamics because κ depends on
  profile alignment, not absolute energy.
- **Scalar κ, uniform modulation:** all K bins at a cell scale by the same
  κ.  The orientation profile is preserved; only magnitude changes.
- **Collinear facilitation:** tangent-selective kernels preferentially pool
  co-oriented neighbors along the tangent → higher S in the matching bin →
  higher cosine similarity on edges.
- **Texture suppression:** incoherent neighbors produce a flat $\mathbf{S}$
  → low cosine similarity → κ < 1 → suppressed over T passes.
- **Junction handling:** junction cell has multi-peaked profile.  Tangent
  neighbors have single-peaked profiles.  Cosine similarity is moderate →
  partial suppression, not full kill.

---

## Post-recurrence extraction

$$
b^*(c) = \arg\max_k \rho_k^{(T)}(c)
$$

$$
\rho(c) = \rho_{b^*}^{(T)}(c), \qquad
\theta(c) = \bar{\theta}_{b^*}, \qquad
\kappa(c) = \kappa^{(T-1)}(c)
$$

$$
\rho_{\max}(c) = \max_k \rho_k^{(T)}(c), \qquad
\bar{z}_0(c) = \frac{1}{K}\sum_k \rho_k^{(0)}(c)
$$

---

## Regional gain feedback

### Pooled statistics

Smooth over large cell-grid radius $R_\eta$ (default $2R$–$3R$):

$$
\bar{\kappa}_c = \text{pool}_{R_\eta}(\kappa), \qquad
\bar{z}_{0,c} = \text{pool}_{R_\eta}(z_0), \qquad
\bar{\rho}_c = \text{pool}_{R_\eta}(\rho_{\max})
$$

### Learned MLP

$$
\boxed{
\eta(c) = \eta_0 \cdot \sigma\!\bigl(\text{MLP}_\eta(\bar{\kappa}_c,\; \bar{z}_{0,c},\; \bar{\rho}_c)\bigr)
}
$$

$$
\text{MLP}_\eta: \; 3 \xrightarrow{W_1, b_1} 8 \xrightarrow{\text{ReLU}} \xrightarrow{W_2, b_2} 1
$$

**Init:** $W_1 = 0,\; b_1 = 0,\; W_2 = 0,\; b_2 = 2$ → $\sigma(2) \approx 0.88$ →
near-identity, no modulation.

$$
\eta(p) = \text{bilinear\_interp}(\eta_c)
$$

Large pooling radius → smooth regional η map.

### Pass 2

Reuse cached $d_k$.  Re-run NR with modulated $\eta(p)$, recompute harmonics,
re-run full L1 hypercolumn + GABA recurrence.  Cheap — no eigendecomposition.

---

## Renderer — contour tracing readout

The upstream pipeline (GABA + η feedback) has decided what's an edge.
The renderer traces contours at pixel resolution: find ridge peaks,
bridge small gaps, produce clean lines.

### Interpolation (fixed)

Cell-grid fields → pixel resolution via bilinear interpolation.
Orientation in double-angle representation:

$$
\bar{\theta}(p) = \tfrac{1}{2}\text{atan2}\!\bigl(\bar{v}/\bar{\rho},\; \bar{u}/\bar{\rho}\bigr)
$$

### Stencils on $h_{2m}$

$$
\text{tang}_j(p) = h_{2m}\bigl(p + j \cdot s_t \cdot \hat{t}(p)\bigr), \quad j \in \{-2,\dots,2\}
$$

$$
\text{norm}_j(p) = h_{2m}\bigl(p + j \cdot s_n \cdot \hat{n}(p)\bigr), \quad j \in \{-2,\dots,2\}
$$

$s_t, s_n$ are learned spacings (2 params).

### Feature vector

$$
F_p = \bigl[\;h_{2m}^{\text{lum}},\; h_{2m}^{\text{chr}},\;
\bar{\rho},\; \bar{\kappa},\;
\text{tang}_5,\; \text{norm}_5\;\bigr] \in \mathbb{R}^{14}
$$

### Tracing MLP

$$
\hat{B}(p) = \bigl(h_{2m}^{\text{lum}}(p) + h_{2m}^{\text{chr}}(p)\bigr) \cdot
\sigma\!\bigl(\text{MLP}_{\text{read}}(F_p)\bigr)
$$

$$
\text{MLP}_{\text{read}}: \; 14 \xrightarrow{W_1, b_1} 8 \xrightarrow{\text{ReLU}} \xrightarrow{W_2, b_2} 1
$$

**Structural priors:**

| Unit | Wired to | Purpose |
|------|----------|---------|
| 0 | tang5 (flat 0.2), norm5 (Mexican hat) | Ridge profile |
| 1 | $\bar{\rho}$ | Cell-grid survival |
| 2 | $h_{2m}^{\text{lum}} + h_{2m}^{\text{chr}}$ | Pixel contrast |
| 3 | $\bar{\kappa}$ | Collinear agreement |

$b_2 = 2$ → near-identity gate at init.

---

## Parameter budget

| Component | Params | Role |
|---|---:|---|
| $\alpha_d, \alpha_t$ | 2 | Kernel shape (fraction of R) |
| $\tilde{\eta}_z$ | 1 | NR gain floor |
| $\text{MLP}_\eta$ (3→8→1) | 41 | Regional gain feedback |
| $s_t, s_n$ | 2 | Stencil spacings |
| $\text{MLP}_{\text{read}}$ (14→8→1) | 129 | Contour tracing readout |
| **Total learned** | **175** | |

### By function

| Function | Params |
|---|---:|
| V1 circuit (dynamics) | 3 |
| V2→V1 feedback (gain) | 41 |
| Readout (tracing) | 131 |
| **Total** | **175** |

### Fixed components

| Component | Note |
|---|---|
| L0 NR | Cached $d_k$; pass 2 reuses |
| cos² binning | Patch-level projection |
| Min subtraction | Per-cell, across K bins |
| Cosine-similarity κ | Profile shape comparison |
| Multiplicative gating | κ applied to all bins |
| Kernel structure | Rebuilt from α_d, α_t each pass |
| Bilinear interpolation | Cell grid → pixels |
