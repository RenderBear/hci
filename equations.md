# HCE renderer — equations

## Overview

The renderer produces a boundary map $\hat{B}(p)$ at every pixel $p$ by gating
the pixel-native second-harmonic magnitude $h_{2m}(p)$ from L0 with a learned
per-pixel MLP.  Cell-grid features reach pixel resolution via bilinear
interpolation, not scatter-add splatting.

$$
\hat{B}(p) \;=\; h_{2m}(p)\;\cdot\;\sigma\!\bigl(\text{MLP}(F_p)\bigr)
$$

---

## L0 — split-channel harmonic projection (precomputed, fixed)

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
h_k^{\text{lum}} = \gamma\,\frac{\tilde{d}_k^2}{\eta_{\text{lum}}^2 + \tilde{d}_k^2}, \qquad
h_k^{\text{chr}} = \gamma\,\frac{\tilde{d}_k^2}{\eta_{\text{chr}}^2 + \tilde{d}_k^2}
$$

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

Photometric separability from partition means (same $\eta$ as L0):

$$
s_{\text{lum}} = \frac{(\Delta L)^2}{(\Delta L)^2 + \eta_{\text{lum}}^2}, \qquad
s_{\text{chr}} = \frac{\|\Delta C\|^2}{\|\Delta C\|^2 + \eta_{\text{chr}}^2}
$$

---

## Seed — NR-normalized cell strength (learned $\eta_z$, $\eta_\rho$)

$$
r_c = \frac{\lambda_{1,c}}{z_{0,c} + \eta_z}, \qquad
\rho_c = \frac{r_c^2}{r_c^2 + \mu_R^2(r) + \eta_\rho^2}\;\cdot\;\mathbf{1}[\text{tile interior}]
$$

where $\mu_R^2(r)$ is the local mean of $r^2$ in a pool of radius $R$ on the cell grid,
and $\eta_z = \text{softplus}(\tilde{\eta}_z)$, $\eta_\rho = \text{softplus}(\tilde{\eta}_\rho)$
are the two learned parameters.

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

## Step 2 — recurrent collinear facilitation + cross-orientation suppression

V1-style horizontal connections on the cell grid.  Each pass facilitates
co-aligned collinear neighbors and suppresses cross-oriented cells via
divisive normalization.  Recurrence propagates facilitation along contours
(effective range ≈ $T \times R$) and cascades cross-orientation suppression.

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

Initialize $\rho^{(0)} = \rho$ (seed output, border-masked).  Each cell has
a fixed bin assignment $b(c) = \lfloor \theta_c K / \pi \rfloor$ and orthogonal
bin $b_\perp(c) = (b(c) + K/2) \bmod K$.

**For** $t = 0, \dots, T-1$:

Convolve with all $K$ kernels using the current $\rho^{(t)}$:

$$
S_u^{(k)} = W_k * \bigl(\rho^{(t)} \cos 2\theta\bigr), \qquad
S_v^{(k)} = W_k * \bigl(\rho^{(t)} \sin 2\theta\bigr)
$$

Each cell reads from its own bin (collinear) and the orthogonal bin (cross),
then projects onto its own orientation $(\cos 2\theta_c,\, \sin 2\theta_c)$:

**Collinear energy** — projection of collinear-kernel neighbors onto cell's orientation:

$$
E_{\text{col}}(c) = \Bigl[S_u^{(b(c))}(c)\Bigr] \cos 2\theta_c
\;+\; \Bigl[S_v^{(b(c))}(c)\Bigr] \sin 2\theta_c
$$

$$
= \sum_{c' \in \mathcal{N}_{\text{col}}} W_{b(c)}(c'-c)\;\rho^{(t)}_{c'}\;\cos 2(\theta_{c'} - \theta_c)
$$

Clamped to $\ge 0$ (negative means anti-aligned; treated as co-aligned since
orientations are $\pi$-periodic).

**Cross-oriented energy** — projection of orthogonal-kernel neighbors onto cell's
normal direction:

$$
E_{\text{cross}}(c) = \Bigl|\Bigl[S_v^{(b_\perp(c))}(c)\Bigr] \cos 2\theta_c
\;-\; \Bigl[S_u^{(b_\perp(c))}(c)\Bigr] \sin 2\theta_c\Bigr|
$$

$$
= \Bigl|\sum_{c' \in \mathcal{N}_{\text{cross}}} W_{b_\perp(c)}(c'-c)\;\rho^{(t)}_{c'}\;\sin 2(\theta_{c'} - \theta_c)\Bigr|
$$

Absolute value since cross-orientation can be $\pm 90°$.

**Divisive normalization** (Naka–Rushton):

$$
\kappa^{(t)}_{\text{col}}(c) = \frac{E_{\text{col}}(c)}{E_{\text{col}}(c) + E_{\text{cross}}(c) + \epsilon}
$$

**Modulate ρ:**

$$
\rho^{(t+1)}_c = \rho^{(t)}_c \cdot \kappa^{(t)}_{\text{col}}(c)
$$

### Properties

After $T$ passes (default 3):

- **Collinear facilitation:** where $E_{\text{col}} \gg E_{\text{cross}}$,
  $\kappa \approx 1$ and $\rho$ is preserved.  Co-aligned neighbors along the
  tangent mutually reinforce across passes, propagating facilitation along
  contours with effective range $\approx T \times R$.

- **Cross-orientation suppression:** where $E_{\text{cross}} \gg E_{\text{col}}$,
  $\kappa \approx 0$ and $\rho$ is killed.  A vertical cell in a horizontal field
  receives strong cross energy and gets fully suppressed.  This cascades across
  passes — once a cross-oriented cell is suppressed, it stops contributing cross
  energy to its neighbors in subsequent passes.

- **Iso-orientation surround suppression:** in a uniform random field, both
  $E_{\text{col}}$ and $E_{\text{cross}}$ are moderate, giving $\kappa \approx 0.3$–$0.5$.
  After 3 multiplicative passes, $\rho_{\text{random}} \to \rho \times 0.3^3 \approx 0.03$.
  Incoherent texture is strongly suppressed.

- **Junction handling:** at T- and L-junctions, cells along each arm have strong
  collinear support from their own arm and weak cross energy.  The junction cell
  itself has moderate support from both arms — it survives with reduced $\kappa$
  rather than being fully suppressed.

- **Modulatory, not driving:** the recurrence can only preserve or suppress $\rho$,
  never amplify it.  $\rho^{(t+1)} \le \rho^{(t)}$ at every cell.  This matches
  V1 horizontal connections which are modulatory (boost gain) rather than driving
  (create response).

- **Zero learned parameters.** The kernels are fixed geometry.  The recurrent
  dynamics are parameter-free Naka–Rushton.

**Detached from the gradient graph** — inputs to collinear coherence are detached
to prevent gradient flow through the iterative conv chain back into seed parameters.

---

## Step 3 — bilinear interpolation to pixel coordinates

Cell-grid fields are interpolated to pixel resolution via `F.grid_sample`.
Cell $c$ at grid position $(i, j)$ has pixel coordinates $(jS + P/2,\; iS + P/2)$.

The interpolated $\rho$ is the **modulated** $\rho^{(T)}$ from Step 2 — after
collinear facilitation and cross-orientation suppression have reshaped the
cell-grid signal.

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

Similarly interpolated: $\bar{\kappa}_{\text{col}}(p)$, and the **pre-recurrence**
seed $\bar{\rho}^{(0)}(p)$ (same bilinear map; border cells contribute $0$).

The thinning head uses the **collinear preservation ratio**
$r(p) = \bar{\rho}^{(T)}(p) / (\bar{\rho}^{(0)}(p) + \varepsilon)$
(clamped to a modest upper bound for numerical stability), which is high when
recurrence leaves $\rho$ intact (long coherent edges) and low when texture is
suppressed despite a strong initial seed.

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
\bar{\rho}^{(T)},\; \bar{u}/\bar{\rho}^{(T)},\; \bar{v}/\bar{\rho}^{(T)},\;
\bar{\kappa}_{\text{col}},\;
\frac{\bar{\rho}^{(T)}}{\bar{\rho}^{(0)} + \varepsilon},\;
\text{tang}_5,\; \text{norm}_5\;\bigr]
\;\in\;\mathbb{R}^{17}
$$

All pixel-native or bilinearly interpolated — no scatter artifacts.

---

## Step 6 — thinning head

$$
\hat{B}(p) = \bigl(h_{2m}^{\text{lum}}(p) + h_{2m}^{\text{chr}}(p)\bigr)\;\cdot\;
\sigma\!\bigl(W_2\,\text{ReLU}(W_1\,F_p + b_1) + b_2\bigr)
$$

17→16→1 MLP.  The gate $\in (0,1)$ can only thin the harmonic edge signal.

Structural priors at initialization:

| Unit | Wired to | Purpose |
|------|----------|---------|
| 0 | tang5 (flat 0.2), norm5 (Mexican hat) | Ridge profile detection |
| 1 | $\bar{\rho}^{(T)}$ | Cell-grid edge strength after recurrence |
| 2 | $\bar{\rho}^{(T)}/(\bar{\rho}^{(0)}+\varepsilon)$ (clamped) | Collinear preservation — texture crushed → low |
| 3 | $h_{2m}^{\text{lum}} + h_{2m}^{\text{chr}}$ | Harmonic evidence |
| 4 | $\bar{\kappa}_{\text{col}}$ | Collinear coherence boost |

$b_2 = 2$ so $\sigma(2) \approx 0.88$ at initialization (near-identity gate).

---

## Parameter budget

| Component | Params | Note |
|---|---:|---|
| $s_t$, $s_n$ | 2 | learned stencil spacings |
| $W_1$: 17×16 + $b_1$: 16 | 288 | thinning hidden layer |
| $W_2$: 16×1 + $b_2$: 1 | 17 | thinning output |
| $\tilde{\eta}_z$, $\tilde{\eta}_\rho$ | 2 | seed (in RhoSeedModule) |
| **Total learned** | **309** | |
| Collinear kernels | $K \times (2R+1)^2$ | fixed, not learned |