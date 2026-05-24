# HCI — hypercolumn architecture

## Motivation

Replace the per-patch eigendecomposition (L1) with a K-channel oriented
energy histogram at each cell position.  Each cell becomes a hypercolumn:
K orientation-tuned units pooling over the same receptive field.  The
collinear recurrence operates natively on the K-channel representation.
No eigensolver, no branch selection, no 2-orientation limitation.

---

## L0 — unchanged

Per-pixel harmonic projection produces, at every pixel $p$:

$$
z_2(p) = \sum_k h_k(p)\, e^{2i\varphi_k}
$$

$$
h_{2m}(p) = |z_2(p)|, \qquad
\theta_h(p) = \tfrac{1}{2}\,\text{atan2}\!\bigl(\text{Im}\,z_2,\;\text{Re}\,z_2\bigr)
$$

These are pixel-native: orientation $\theta_h$ and magnitude $h_{2m}$ at
full resolution.

---

## Hypercolumn construction (replaces L1 eigendecomposition)

### Oriented energy binning

Patches of size $P \times P$ at stride $S$ (with overlap) tile the image,
producing a cell grid of size $n_H \times n_W$.  At each cell position $c$,
project the patch's pixel-level oriented energy onto $K$ bins
(default $K = 24$, spacing $\pi/K = 7.5°$):

$$
\rho_k^{\text{raw}}(c) = \sum_{p \in \text{patch}(c)} h_{2m}(p) \;\cdot\;
\cos^2\!\bigl(\theta_h(p) - \bar{\theta}_k\bigr)
$$

where $\bar{\theta}_k = k\pi/K$ is the centre angle of bin $k$.

The $\cos^2$ weighting is the orientation tuning curve — a pixel whose
orientation matches the bin contributes fully; a pixel $90°$ away
contributes zero.  This is the standard model of V1 simple cell
orientation selectivity.

### Local energy and normalization

Total energy at cell $c$ (sum over all bins; used elsewhere, not in the seed NR):

$$
z_0(c) = \sum_{k=0}^{K-1} \rho_k^{\text{raw}}(c)
$$

**Baseline removal** per cell: subtract the weakest bin so all channels are
non-negative *excess* orientation energy relative to the patch minimum:

$$
\tilde{\rho}_k^{\text{raw}}(c) = \rho_k^{\text{raw}}(c) - \min_{j} \rho_j^{\text{raw}}(c)
$$

**Naka–Rushton vs.\ the learned floor only** (no $z_0$ in the denominator —
L0 NR and GABA κ gating already handle contrast and competition):

$$
\rho_k^{(0)}(c) = \frac{\bigl(\tilde{\rho}_k^{\text{raw}}(c)\bigr)^2}{\bigl(\tilde{\rho}_k^{\text{raw}}(c)\bigr)^2 + \eta_z^2}
$$

$\eta_z$ sets how much excess counts as a “real” oriented response; values lie in $[0, 1)$.

This is a K-vector at each cell position: the hypercolumn's orientation
response profile.  Edges produce a sharply peaked profile (one dominant
bin).  Junctions produce 2–3 peaks.  Texture produces a flat profile.

### What's eliminated

- **Eigendecomposition**: no 3×3 moment matrix, no eigensolver, no
  eigenvalue sorting.
- **Branch selection**: no $\theta_0, \theta_1$, no branch index, no
  `_theta_on_branch`.
- **2-orientation limit**: junctions with 3+ arms are naturally represented
  as 3+ active bins.

### What's preserved

- **Patch geometry**: same $P, S$, overlap as before.
- **Cell grid**: same $n_H \times n_W$ spatial layout.
- **Divisive normalization**: per-cell min subtraction + NR vs.\ $\eta_z$
  only (`params.SEED.ETA_Z_INIT`); GABA handles cross-bin competition.

---

## Collinear recurrence on K-channel representation

The collinear recurrence now operates on a $(K, n_H, n_W)$ tensor
instead of $(1, n_H, n_W)$ with separate double-angle convolutions.

### Kernels (same as before)

$K$ tangent-selective kernels $W_k$ of size $(2R+1)^2$, precomputed
from Gaussian distance × tangent selectivity at each bin's angle.

### Per-pass dynamics

Initialize $\rho_k^{(0)}$ from the hypercolumn construction above.

**For** $t = 0, \dots, T-1$:

**Stage A — per-bin convolution.**
Each bin $k$ convolves its own channel with its own tangent-selective kernel:

$$
S_k^{(t)}(c) = \bigl(W_k * \rho_k^{(t)}\bigr)(c)
$$

This is a single `F.conv2d` on a $(1, K, n_H, n_W)$ tensor with $K$
**depthwise** kernels — each channel convolved with its matched kernel.
Cost: one depthwise conv2d per pass.

**Stage B — GABA gate (κ).**

Default (`L1.COL_KAPPA_NORM="cosine"`): one **scalar** $\kappa^{(t)}(c)$ per cell
(cosine similarity between the cell's $K$-vector $\rho_k^{(t)}(c)$ and the
collinear neighborhood response $S_k^{(t)}(c) = (W_k * \rho_k^{(t)})(c)$):

$$
\kappa^{(t)}(c) = \frac{\sum_k \rho_k^{(t)}(c)\, S_k^{(t)}(c)}{
\sqrt{\sum_k \rho_k^{(t)}(c)^2}\;\sqrt{\sum_k S_k^{(t)}(c)^2} + \epsilon}
$$

Clamped to $[0,1]$. High when the cell's orientation profile matches the
neighborhood's pooled collinear support (coherent contour); lower when
neighbors disagree (texture / orientation boundaries). The same $\kappa$
multiplies **every** bin — recovering "do neighbors agree with *my* profile?"
rather than only "am I as strong as the local max bin?"

**Alternative modes** (`L1.COL_KAPPA_NORM`): **max** — per-bin
$\kappa_k^{(t)}(c) = S_k^{(t)}(c) / (\max_{j} S_j^{(t)}(c) + \epsilon)$;
**fair-share** —
$\kappa_k^{(t)}(c) = S_k^{(t)}(c) / (E_{\text{total}}(c)/K + \epsilon)$ with
$E_{\text{total}}=\sum_k S_k$. Fair-share is degenerate with cos² + large $K$
(active bins $\ll K$ ⇒ often $\kappa_k\approx 1$).

**Stage C — modulate.**

Cosine mode (scalar $\kappa$):

$$
\rho_k^{(t+1)}(c) = \rho_k^{(t)}(c) \cdot \kappa^{(t)}(c)
$$

Per-bin modes (max / fair-share): $\rho_k^{(t+1)}(c) = \rho_k^{(t)}(c) \cdot \kappa_k^{(t)}(c)$.

### Compact per-pass update (cosine default)

$$
\boxed{
\rho_k^{(t+1)}(c) = \rho_k^{(t)}(c) \;\cdot\;
\frac{\sum_j \rho_j^{(t)}(c)\,(W_j * \rho_j^{(t)})(c)}{
\sqrt{\sum_j \rho_j^{(t)}(c)^2}\;
\sqrt{\sum_j \bigl((W_j * \rho_j^{(t)})(c)\bigr)^2} + \epsilon}
}
$$

(Per-bin $\kappa_k$ variants use the same $S_k^{(t)}$ with different normalizations.)

### Key difference from current architecture

Currently: scalar $\rho$ per cell, collinear energy computed by projecting
double-angle fields onto the cell's own orientation.  The projection acts
as an implicit orientation filter.

Hypercolumn version: K-channel $\rho$ per cell, each channel explicitly
filtered by its own tangent-selective kernel.  With **cosine** κ, a single
scalar compares the full cell profile to the neighborhood's pooled $S_k$,
restoring projection-like "agreement with me" semantics across bins.

### What this enables

- **Depthwise conv2d**: one call per pass instead of $2K$ separate
  convolutions (cos2θ and sin2θ channels per bin).  Faster on GPU.
- **Per-bin collinear facilitation**: each orientation channel is
  facilitated independently along its own tangent.  A junction cell
  with bins at 0° and 90° both active gets facilitation along *both*
  tangent directions simultaneously.  The current architecture picks
  one branch and ignores the other.
- **Natural junction representation**: 3-way junctions have 3 active
  bins, each facilitated by its own collinear neighbors.  No branch
  selection heuristic.

---

## Readout (replaces renderer)

The output at each cell is a K-vector $\rho_k^{(T)}(c)$.  To produce
a scalar edge map at pixel resolution:

### Option A — max projection + interpolation

$$
\rho_{\max}(c) = \max_k \rho_k^{(T)}(c)
$$

Interpolate $\rho_{\max}$ to pixel resolution, multiply by $h_{2m}$:

$$
\hat{B}(p) = h_{2m}(p) \cdot \bar{\rho}_{\max}(p)
$$

Simple, no learned parameters.  The collinear recurrence and GABA budget
do all the work; the readout just picks the strongest surviving orientation.

### Option B — orientation-matched readout

At each pixel, the orientation $\theta_h(p)$ from L0 selects the
matching bin.  Interpolate that bin's $\rho$ to pixel resolution:

$$
\hat{B}(p) = h_{2m}(p) \cdot \bar{\rho}_{b(p)}^{(T)}(p)
$$

where $b(p) = \lfloor \theta_h(p) \cdot K / \pi \rfloor$.

This is tighter — a pixel's edge strength is gated only by the
survival of its orientation's channel, not the max across all
orientations.

### Option C — minimal MLP (current approach, adapted)

Keep the small MLP but feed it per-bin features:

$$
F_p = \bigl[\;h_{2m}^{\text{lum}},\; h_{2m}^{\text{chr}},\;
\bar{\rho}_{b(p)}^{(T)},\; \bar{\rho}_{\max}^{(T)},\;
\bar{\kappa}_{b(p)},\; r_{\text{pres}},\;
\text{tang}_5,\; \text{norm}_5\;\bigr]
$$

Same MLP structure, similar feature count, but the features come
from the K-channel representation instead of eigendecomposition.

---

## η modulation (unchanged in principle)

The collinear energy $E_{\text{col}}$ becomes the per-bin sum:

$$
E_{\text{col}}(c) = S_{b_{\max}(c)}^{(T)}(c)
$$

where $b_{\max}(c) = \arg\max_k \rho_k^{(T)}(c)$ is the dominant bin.

The κ is already computed per-bin from the GABA budget.  Use the
dominant bin's κ:

$$
\eta(p) = \eta_0 \cdot \sigma\!\bigl(a - b\cdot\bar{\kappa}_{b_{\max}} + c\cdot\bar{E}_{\text{col}}\bigr)
$$

Same 3 learned scalars, same sigmoid, same two-pass pipeline.

---

## Computational comparison

| Operation | Current (eigendecomp) | Hypercolumn (K-bin) |
|---|---|---|
| Binning / eigendecomp | 3×3 eigensolver per patch | $K$ dot products per patch |
| Collinear per pass | $2K$ conv2d on $(1,1,n_H,n_W)$ | 1 depthwise conv2d on $(1,K,n_H,n_W)$ |
| Representation per cell | $(\lambda_1, \theta_0, \lambda_2, \theta_1)$ | K-vector $\rho_k$ |
| Branch selection | pick 1 of 2 | not needed |
| Junction handling | limited to 2 orientations | natural, K orientations |

---

## Parameter budget

### Without MLP (Options A or B)

| Component | Params |
|---|---:|
| $\tilde{\eta}_z$ | 1 |
| $a, b, c$ (η mod) | 3 |
| **Total** | **4** |

### With minimal MLP (Option C)

| Component | Params |
|---|---:|
| $\tilde{\eta}_z$ | 1 |
| $a, b, c$ (η mod) | 3 |
| Thinning MLP (4→4→1) | 25 |
| **Total** | **~29** |

---

## Summary

The hypercolumn architecture replaces the eigendecomposition with direct
orientation binning.  Each cell position becomes a K-channel unit
(hypercolumn) whose channels compete via the GABA budget and facilitate
along their respective tangent directions.  The collinear recurrence
is simpler (depthwise conv instead of double-angle projection), more
expressive (K orientations instead of 2), and naturally handles junctions.

The eigensolver was an analytical shortcut for finding dominant orientations
in a patch.  The K-bin + GABA recurrence achieves the same result
empirically — dominant orientations survive the competition — while
preserving the full orientation spectrum for junction and texture analysis.
