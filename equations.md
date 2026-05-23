# HCI renderer — equations

## Overview

The renderer produces a boundary map $\hat{B}(p)$ at every pixel $p$ by gating
the pixel-native second-harmonic magnitude $h_{2m}(p)$ from L0 with a learned
per-pixel MLP.  Cell-grid features reach pixel resolution via bilinear
interpolation, not scatter-add splatting.

$$
\hat{B}(p) \;=\; h_{2m}(p)\;\cdot\;\sigma\!\bigl(\text{MLP}(F_p)\bigr)
$$

Two-pass pipeline: L0 (fixed η) → L1 → seed → collinear recurrence →
η modulation → L0 (spatially-varying η) → render with updated $h_{2m}$.

---

## L0 — split-channel harmonic projection

RGB split into luminance $L = (R+G+B)/3$ and chrominance $C = I - L\cdot\mathbf{1}$.

Per-pixel directional differences along 8-connected offsets $\delta_k$ with bearing $\varphi_k$:

$$
d_k^{\text{lum}} = |L(p) - L(p+\delta_k)|, \qquad
d_k^{\text{chr}} = \|C(p) - C(p+\delta_k)\|_2
$$

Per-direction Naka–Rushton with minimum subtraction:

$$
\tilde{d}_k = d_k - \min_j d_j
$$

$$
h_k^{\text{lum}} = \gamma\,\frac{\tilde{d}_k^2}{\eta_{\text{lum}}(p)^2 + \tilde{d}_k^2}, \qquad
h_k^{\text{chr}} = \gamma\,\frac{\tilde{d}_k^2}{\eta_{\text{chr}}(p)^2 + \tilde{d}_k^2}
$$

Pass 1: $\eta_{\text{lum}}(p) = \eta_0^{\text{lum}}$ (scalar, fixed).
Pass 2: $\eta(p)$ is spatially varying (see Step 7).

Combined harmonic projection:

$$
h_k = h_k^{\text{lum}} + h_k^{\text{chr}}, \qquad
z_n = \sum_k h_k\, e^{in\varphi_k}, \quad n \in \{1, 2\}
$$

Second-harmonic magnitudes (the pixel-native edge signal):

$$
h_{2m}(p) = |z_2(p)| = \Bigl|\sum_k h_k\, e^{2i\varphi_k}\Bigr|
$$

Split-channel versions:

$$
h_{2m}^{\text{lum}}(p) = \Bigl|\sum_k h_k^{\text{lum}}\, e^{2i\varphi_k}\Bigr|, \qquad
h_{2m}^{\text{chr}}(p) = \Bigl|\sum_k h_k^{\text{chr}}\, e^{2i\varphi_k}\Bigr|
$$

**Key optimization:** the directional differences $d_k^{\text{lum}}, d_k^{\text{chr}}$
depend only on the image, not on $\eta$.  They are computed once and cached.
Pass 2 reuses them and only reapplies the Naka–Rushton with the modulated $\eta(p)$.

---

## L1 — per-patch eigendecomposition (precomputed, fixed)

Moment matrix from patch-summed harmonics:

$$
M = \begin{pmatrix} Z_0 & \bar{Z}_2 & \bar{Z}_4 \\
Z_2 & Z_0 & \bar{Z}_2 \\
Z_4 & Z_2 & Z_0 \end{pmatrix}, \qquad
Z_n = \sum_{p \in \text{patch}} z_2(p)^{n/2}
$$

Eigendecomposition $M = V \Lambda V^*$ yields eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$
and orientations $\theta_0, \theta_1$ from the leading eigenvectors.

Photometric separability from partition means:

$$
s_{\text{lum}} = \frac{(\Delta L)^2}{(\Delta L)^2 + \eta_{\text{lum}}^2}, \qquad
s_{\text{chr}} = \frac{\|\Delta C\|^2}{\|\Delta C\|^2 + \eta_{\text{chr}}^2}
$$

---

## Seed — cell strength (learned $\eta_z$)

Raw eigenvalue ratio, normalized by total harmonic energy:

$$
r_c = \frac{\lambda_{1,c}}{\max(z_{0,c} + \eta_z,\;\epsilon)}
$$

with $\eta_z = \text{softplus}(\tilde{\eta}_z)$.

Seed output (no NR pool in current configuration):

$$
\rho^{(0)}_c = r_c \cdot \mathbf{1}[\text{tile interior}]
$$

---

## Step 1 — cell-grid $\theta$ combing

Iterative ρ-weighted double-angle smoothing over 3×3 cell neighborhoods:

$$
\theta_c \;\leftarrow\; \frac{1}{2}\,\text{atan2}\!\Bigl(
\frac{\sum_{c'} \rho_{c'}\sin 2\theta_{c'}}{\sum_{c'} \rho_{c'}},\;
\frac{\sum_{c'} \rho_{c'}\cos 2\theta_{c'}}{\sum_{c'} \rho_{c'}}
\Bigr)
$$

Repeated for $T$ passes (default 4).  Border cells are excluded.

---

## Step 2 — recurrent collinear facilitation (GABA-budget normalization)

V1-style horizontal connections on the cell grid.  Each pass facilitates
co-aligned collinear neighbors and suppresses all other orientations via
a firing-budget normalization modeled on GABAergic untuned inhibition.

### Kernels

$\theta$ is quantized into $K$ bins (default 24, spacing $\pi/K = 7.5°$).
For bin $k$ with reference angle $\bar{\theta}_k = k\pi/K$, precompute a $(2R+1)^2$ kernel:

$$
W_k(\Delta i, \Delta j) = \underbrace{\exp\!\Bigl(-\frac{\Delta i^2 + \Delta j^2}{2\sigma_d^2}\Bigr)}_{w_d\;\text{(distance)}}
\;\cdot\; \underbrace{\exp\!\Bigl(-\frac{d_\perp^2}{2\sigma_t^2}\Bigr)}_{w_t\;\text{(tangent selectivity)}}
$$

$$
d_\perp = \Delta j\,\cos\bar{\theta}_k - \Delta i\,\sin\bar{\theta}_k
$$

Centre pixel excluded ($W_k(0,0) = 0$), and $W_k = 0$ for $\|\Delta\| > R$.
Default: $\sigma_d = R/2$, $\sigma_t = 1.0$.

### Recurrent dynamics

Initialize $\rho^{(0)}$ from the seed (border-masked).  Each cell has
a fixed bin assignment $b(c) = \lfloor \theta_c K / \pi \rfloor$.

**For** $t = 0, \dots, T-1$:

**Stage A — convolutions.**
Convolve with all $K$ kernels using the current $\rho^{(t)}$:

$$
S_u^{(k)} = W_k * \bigl(\rho^{(t)} \cos 2\theta\bigr), \qquad
S_v^{(k)} = W_k * \bigl(\rho^{(t)} \sin 2\theta\bigr), \qquad
S_\rho^{(k)} = W_k * \rho^{(t)}
$$

**Stage B — collinear energy.**
Each cell reads from its own θ-bin and projects onto its own orientation:

$$
E_{\text{col}}(c) = \Bigl[S_u^{(b(c))}(c)\Bigr] \cos 2\theta_c
\;+\; \Bigl[S_v^{(b(c))}(c)\Bigr] \sin 2\theta_c
$$

$$
= \sum_{c' \in \mathcal{N}_{\text{col}}} W_{b(c)}(c'-c)\;\rho^{(t)}_{c'}\;\cos 2(\theta_{c'} - \theta_c)
$$

Clamped to $\ge 0$.

**Stage C — GABA budget.**
Total ρ pooled across all $K$ bins (untuned inhibitory pool):

$$
E_{\text{total}}(c) = \sum_{k=0}^{K-1} S_\rho^{(k)}(c)
$$

Fair-share normalization:

$$
\kappa^{(t)}(c) = \frac{E_{\text{col}}(c)}{E_{\text{total}}(c) / K + \epsilon}
$$

Clamped to $[0, 1]$.

**Stage D — modulate ρ:**

$$
\rho^{(t+1)}_c = \rho^{(t)}_c \cdot \kappa^{(t)}(c)
$$

### Full per-pass update (compact)

$$
\boxed{
\rho^{(t+1)}_c = \rho^{(t)}_c \;\cdot\; \min\!\Biggl(1,\;\;
\frac{E_{\text{col}}(c)}{E_{\text{total}}(c)/K + \epsilon}\Biggr)
}
$$

### Properties

- **Collinear facilitation:** where $E_{\text{col}}$ dominates the budget,
  $\kappa = 1$ and $\rho$ is preserved.  Facilitation propagates along
  contours with effective range $\approx T \times R$.

- **All-orientation competition:** unlike the old col-vs-cross formulation
  (two bins), the GABA budget sees all $K$ bins — edges at any angle
  compete, not just orthogonal ones.

- **Texture suppression:** in an incoherent field, energy is spread across
  many bins, so $E_{\text{col}} \ll E_{\text{total}}/K$ and $\kappa \ll 1$.

- **Self-stabilizing:** the denominator $E_{\text{total}}$ scales with $\rho$.
  Uniform collinear fields converge gently ($\rho \approx 0.95^T$), not collapse.

- **Junction handling:** at junctions, competing orientations inflate the
  budget proportionally to the number of arms.

- **Zero learned parameters.** Kernels are fixed geometry.  The GABA-budget
  dynamics are parameter-free.

**Detached from the gradient graph.** $\bar{\theta}(p)$ at pixel resolution
is also detached.

$E_{\text{col}}$ (raw, unnormalized) is returned for use in Step 7.

---

## Step 3 — bilinear interpolation to pixel coordinates

Cell-grid fields are interpolated to pixel resolution via `F.grid_sample`.
Cell $c$ at grid position $(i, j)$ has pixel coordinates $(jS + P/2,\; iS + P/2)$.

The interpolated $\rho$ is the **modulated** $\rho^{(T)}$ from Step 2.

Orientation is interpolated in double-angle representation to handle $\pi$-periodicity:

$$
u_c = \rho_c\cos 2\theta_c, \qquad v_c = \rho_c\sin 2\theta_c
$$

$$
\bar{u}(p) = \text{interp}(u), \quad \bar{v}(p) = \text{interp}(v), \quad \bar{\rho}(p) = \text{interp}(\rho)
$$

$$
\bar{\theta}(p) = \tfrac{1}{2}\,\text{atan2}\!\bigl(\bar{v}/\bar{\rho},\;\bar{u}/\bar{\rho}\bigr)
$$

Similarly interpolated: $\bar{\kappa}_{\text{col}}(p)$ and $\bar{\rho}^{(0)}(p)$
(pre-recurrence seed).

---

## Step 4 — stencils on $h_{2m}$ (pixel-native)

5-tap stencils along $\bar{\theta}(p)$ sample the pixel-native $h_{2m}$ field:

$$
\text{tang}_k(p) = h_{2m}\!\bigl(p + k\,s_t\,\hat{t}(p)\bigr), \quad k \in \{-2,\dots,2\}
$$

$$
\text{norm}_k(p) = h_{2m}\!\bigl(p + k\,s_n\,\hat{n}(p)\bigr), \quad k \in \{-2,\dots,2\}
$$

where $\hat{t} = (\cos\bar{\theta}, \sin\bar{\theta})$, $\hat{n} = (-\sin\bar{\theta}, \cos\bar{\theta})$,
and $s_t$, $s_n$ are learned stencil spacings.

---

## Step 5 — feature vector

$$
F_p = \bigl[\;h_{2m}^{\text{lum}},\; h_{2m}^{\text{chr}},\;
\bar{\rho}^{(T)},\; \cos 2\bar{\theta},\; \sin 2\bar{\theta},\;
\bar{\kappa}_{\text{col}},\;
\bar{\rho}^{(T)}/(\bar{\rho}^{(0)}+\epsilon),\;
\text{tang}_5,\; \text{norm}_5,\;
\eta_{\mathrm{lum}}(p)\;\bigr]
\;\in\;\mathbb{R}^{18}
$$

The last entry is the pass-2 luminance $\eta$ map (zeros when absent). Training detaches the NR/harmonics path into $h_{2m}^{*}$ but keeps $\eta_{\mathrm{lum}}$ in $F_p$ so gradients to the modulation scalars $(a,b,c)$ flow through the thinning MLP only.

All pixel-native or bilinearly interpolated — no scatter artifacts.

---

## Step 6 — thinning head

$$
\hat{B}(p) = \bigl(h_{2m}^{\text{lum}}(p) + h_{2m}^{\text{chr}}(p)\bigr)\;\cdot\;
\sigma\!\bigl(W_2\,\text{ReLU}(W_1\,F_p + b_1) + b_2\bigr)
$$

18→16→1 MLP on $F_p \in \mathbb{R}^{18}$ (last component: pass-2 luminance $\eta_{\mathrm{lum}}(p)$ when provided; otherwise zero).  The gate $\in (0,1)$ can only thin the harmonic edge signal.

Structural priors at initialization:

| Unit | Wired to | Purpose |
|------|----------|---------|
| 0 | tang5 (flat 0.2), norm5 (Mexican hat) | Ridge profile detection |
| 1 | $\bar{\rho}^{(T)}$ | Cell-grid edge strength after recurrence |
| 2 | $\bar{\rho}^{(T)}/(\bar{\rho}^{(0)}+\epsilon)$ | Collinear preservation ratio |
| 3 | $h_{2m}^{\text{lum}} + h_{2m}^{\text{chr}}$ | Harmonic evidence |
| 4 | $\bar{\kappa}_{\text{col}}$ | Collinear coherence boost |

$b_2 = 2$ so $\sigma(2) \approx 0.88$ at initialization (near-identity gate).

---

## Step 7 — η modulation (two-pass L0 feedback)

After pass 1 produces the collinear recurrence signals $\bar{\kappa}$ and
$E_{\text{col}}$, the L0 semi-saturation constants are spatially modulated
and L0 is re-run with the updated $\eta$.

### Modulation signal

Two signals from the collinear recurrence, interpolated to pixel resolution:

- $\bar{\kappa}(p)$: normalized collinear coherence — "does geometric structure
  exist here?"
- $\bar{E}_{\text{col}}(p)$: raw (unnormalized) collinear energy, normalized to
  $[0, 1]$ by $E_{\max}$ — "how strong is the absolute evidence?"

### Learned modulation

$$
\boxed{
\eta(p) = \eta_0 \;\cdot\; \sigma\!\bigl(a - b\cdot\bar{\kappa}(p) + c\cdot\bar{E}_{\text{col}}(p)\bigr)
}
$$

Three learned scalars $(a, b, c)$ control the modulation:

| Parameter | Role | Effect when large |
|-----------|------|-------------------|
| $a$ | Bias | $\sigma(a) \to 1$, η stays near η₀ by default |
| $b$ | κ coefficient | High κ (geometric structure) → η drops → L0 more sensitive |
| $c$ | $E_{\text{col}}$ coefficient | High absolute energy → η rises → L0 stays stable |

**Initialization:** $a = 2, b = 0, c = 0$ → $\sigma(2) \approx 0.88$ everywhere,
so $\eta \approx 0.88\,\eta_0$ — near-identity, no modulation.

### Behavior

| Condition | κ | $E_{\text{col}}$ | $\sigma(\cdot)$ | $\eta(p)$ | Effect |
|-----------|---|-----------------|-----------------|-----------|--------|
| Faint collinear edge | high | low | small | $\ll \eta_0$ | L0 more sensitive, recovers edge |
| Strong edge | high | high | $\approx 1$ | $\approx \eta_0$ | Already detected, no change |
| Incoherent texture | low | varies | $\approx \sigma(a)$ | $\approx \eta_0$ | No structure → don't boost |
| Flat region | low | low | $\approx \sigma(a)$ | $\approx \eta_0$ | Nothing to recover |

### Two-pass pipeline (compact)

$$
\text{Pass 1:}\quad d_k \;\xrightarrow{\;\eta_0\;}\; h_{2m}^{(1)} \;\to\; \text{L1} \;\to\; \rho \;\to\; \text{collinear} \;\to\; \kappa,\, E_{\text{col}}
$$

$$
\text{Pass 2:}\quad d_k \;\xrightarrow{\;\eta(p)\;}\; h_{2m}^{(2)} \;\to\; \hat{B} = h_{2m}^{(2)} \cdot \text{gate}(F_p)
$$

$d_k$ (directional differences) are cached from pass 1.  L1 eigendecomposition
is **not** re-run; the cell grid ($\rho, \theta, \lambda$) is from pass 1.

---

## Parameter budget

| Component | Params | Note |
|---|---:|---|
| $s_t$, $s_n$ | 2 | learned stencil spacings |
| $W_1$: 18×16 + $b_1$: 16 | 304 | thinning hidden layer |
| $W_2$: 16×1 + $b_2$: 1 | 17 | thinning output |
| $\tilde{\eta}_z$ | 1 | seed (softplus → $\eta_z$) |
| $a, b, c$ | 3 | η modulation (sigmoid) |
| **Total learned** | **327** | |
| Collinear kernels | $K \times (2R+1)^2$ | fixed, not learned |
| Pass-2 L0 recompute | $d_k$ cached | ~10× cheaper than full L0 |
