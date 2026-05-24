# HCI — equations (code-aligned)

This file records the **notation and equations implemented in this repository** (`hci/L0.py`, `hci/L1.py`, `hci/renderer.py`, `hci/seed.py`, `train.py`). For a longer system narrative and design rationale, see `HCI_SYSTEM_EQUATIONS.md`. Geometry and defaults live in `params.py`.

---

## 1. End-to-end pipeline

1. **L0 pass 1** — fixed base semi-saturation $\eta_0^{\mathrm{lum}}, \eta_0^{\mathrm{chr}}$; directional differences $d_k^{\mathrm{lum}}, d_k^{\mathrm{chr}}$ may be **cached** (η-independent).
2. **L1** — $\cos^2$ hypercolumns on $h_{2m}$, per-cell divisive Naka–Rushton vs.\ learned $\eta_z$ on raw bin energies (no min-subtract), then **GABA** collinear recurrence on $K$ bins, then **dominant-bin** scalars for the cell grid.
3. **Renderer** — θ combing on cells, bilinear interp of $\rho,\theta,\kappa_{\mathrm{col}}$ to pixels, tangential / normal **stencils** on $h_{2m}^{\mathrm{lum}}+h_{2m}^{\mathrm{chr}}$, **15→8→1** gate, $\hat B = (h_{2m}^{\mathrm{lum}}+h_{2m}^{\mathrm{chr}})\,\bar\rho \cdot \mathrm{gate}$.
4. **L0 pass 2 (optional)** — regional maps $\eta_{\mathrm{lum}}(p), \eta_{\mathrm{chr}}(p)$ from **MLP$_\eta$**; fast re-NR + harmonics from cached $d_k$; **re-render** only (full L1 is not re-run inside `train.forward_batch` — see §6).

---

## 2. L0 — split luminance / chrominance harmonics

Eight offsets $\delta_k \in \mathbb{Z}^2$ (`L0.OFFSETS`). RGB → orthonormal $L$ and chroma $C$.

**Directional differences** (magnitude / $\ell_2$ on $C$):

$$
d_k^{\mathrm{lum}}(p) = \bigl|L(p) - L(p+\delta_k)\bigr|, \qquad
d_k^{\mathrm{chr}}(p) = \bigl\|C(p) - C(p+\delta_k)\bigr\|_2 .
$$

**Per-pixel min across directions** (subscript $k$ is direction index):

$$
\tilde d_k^{\mathrm{lum}} = d_k^{\mathrm{lum}} - \min_j d_j^{\mathrm{lum}}, \qquad
\tilde d_k^{\mathrm{chr}} = d_k^{\mathrm{chr}} - \min_j d_j^{\mathrm{chr}} .
$$

**Naka–Rushton** with scalar or spatial $\eta$ (pass 1: constants $\eta_0^{\mathrm{lum}}, \eta_0^{\mathrm{chr}}$; pass 2: maps). Implementation uses a small denominator floor so $\eta\to 0$ with flat $\tilde d$ does not produce $0/0$.

$$
h_k^{\mathrm{lum}} = \gamma\,\frac{(\tilde d_k^{\mathrm{lum}})^2}{\eta_{\mathrm{lum}}(p)^2 + (\tilde d_k^{\mathrm{lum}})^2}, \qquad
h_k^{\mathrm{chr}} = \gamma\,\frac{(\tilde d_k^{\mathrm{chr}})^2}{\eta_{\mathrm{chr}}(p)^2 + (\tilde d_k^{\mathrm{chr}})^2}.
$$

**Second harmonic** (complex $z_2$, magnitude $h_{2m}$, split lum/chr for the renderer):

$$
z_2(p) = \sum_k \bigl(h_k^{\mathrm{lum}} + h_k^{\mathrm{chr}}\bigr)\, e^{2i\varphi_k}, \qquad
h_{2m}(p) = |z_2(p)|, \qquad
\theta_h(p) = \tfrac12 \arg z_2(p).
$$

Pixel fields $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$ are the magnitudes of the lum-only and chr-only harmonic sums (see `compute_l0_rgb`).

---

## 3. L1 — hypercolumn binning and pre-GABA NR

Patch size $P$, stride $S$ → cell grid $(n_H, n_W)$, $K$ bins (`L1.COL_K_BINS`), bin centres $\bar\theta_k = k\pi/K$.

**Oriented energy** into bin $k$ at cell $c$:

$$
\rho_k^{\mathrm{raw}}(c) = \sum_{p \in \mathrm{patch}(c)} h_{2m}(p)\,\cos^2\!\bigl(\theta_h(p) - \bar\theta_k\bigr).
$$

**Divisive NR vs.\ learned $\eta_z$** on raw bin energies (single scalar per forward, $\eta_z = \mathrm{softplus}(\tilde\eta_z)$; small $\varepsilon$ in code):

$$
\rho_k^{(0)}(c) = \frac{\bigl(\rho_k^{\mathrm{raw}}(c)\bigr)^2}{\bigl(\rho_k^{\mathrm{raw}}(c)\bigr)^2 + \eta_z^2 + \varepsilon}.
$$

**Cell total oriented energy** (used for η feedback):

$$
z_0(c) = \sum_{k=0}^{K-1} \rho_k^{\mathrm{raw}}(c).
$$

---

## 4. GABA — additive collinear recurrence + per-pass NR

Learned tangent / normal scales ($\tilde\alpha_d, \tilde\alpha_t$) and lateral step $\tilde\alpha$:

$$
\sigma_d = \mathrm{softplus}(\tilde\alpha_d)\,R, \qquad
\sigma_t = \mathrm{softplus}(\tilde\alpha_t)\,R, \qquad
\alpha = \mathrm{softplus}(\tilde\alpha),
$$

with collinear radius $R =$ `L1.COL_RADIUS` (pixels). Kernels $W_k$ are Gaussians in tangential / normal coordinates, masked to a $(2R+1)^2$ support.

**One pass** $t = 0,\ldots,T-1$ (`L1.COL_PASSES`):

$$
S_k^{(t)}(c) = (W_k * \rho_k^{(t)})(c) \quad \text{(depthwise conv, groups } K\text{)}, \qquad
\bar S^{(t)}(c) = \frac{1}{K}\sum_{j=0}^{K-1} S_j^{(t)}(c).
$$

Let $u_k^{(t)} = \max\bigl(0,\; \rho_k^{(t)}(c) + \alpha\,(S_k^{(t)}(c) - \bar S^{(t)}(c))\bigr)$.  With the same learned $\eta_z$ as §3,

$$
\rho_k^{(t+1)}(c) = \frac{\bigl(u_k^{(t)}(c)\bigr)^2}{\bigl(u_k^{(t)}(c)\bigr)^2 + \eta_z^2 + \varepsilon}.
$$

**Diagnostic κ (renderer / MLP features, not a gate):** e.g.\ bin mass share $\rho_k^{(t+1)} / (\sum_j \rho_j^{(t+1)} + \varepsilon)$ clamped to $[0,1]$.

---

## 5. Post-recurrence dominant channel (cached to disk)

Dominant bin $b^*(c) = \arg\max_k \rho_k^{(T)}(c)$. Cached scalars per cell include e.g.

$$
\rho_{\mathrm{dom}}(c) = \rho_{b^*}^{(T)}(c), \quad
\theta(c) = \bar\theta_{b^*}, \quad
\kappa_{\mathrm{col}}(c), \quad
\rho_{\max}(c) = \max_k \rho_k^{(T)}(c),
$$

and intermediate diagnostics (`e_{\mathrm{col}}`, etc.) as produced by `run_l1_hypercolumn` / `build_cells_flat`.

---

## 6. Regional η — MLP$_\eta$ (training / inference pass 2)

On the **cell** grid, mean-pool with radius $R_\eta$ in cell indices (`L0.ETA_POOL_RADIUS_CELLS`), borders zeroed before pool:

$$
\bar\kappa_c = \mathrm{pool}_{R_\eta}(\kappa_{\mathrm{col}}), \quad
\bar z_{0,c} = \mathrm{pool}_{R_\eta}(z_0), \quad
\bar\rho_{\max,c} = \mathrm{pool}_{R_\eta}(\rho_{\max}).
$$

**Max-normalize** pooled $\bar z_0$ and $\bar\rho_{\max}$ on the grid (per forward, for stable MLP inputs); concatenate with $\bar\kappa_c$ → **MLP$_\eta$**: $3 \to 8 \to 1$, $\sigma$ on the logit, then **bilinear interp** to pixels:

$$
\eta_{\mathrm{lum}}(p) = \eta_0^{\mathrm{lum}}\cdot m(p), \qquad
\eta_{\mathrm{chr}}(p) = \eta_0^{\mathrm{chr}}\cdot m(p), \qquad
m = \sigma(\mathrm{MLP}_\eta(\cdot)) \in [m_{\min},1]
$$

(modulation $m$ is clamped after sigmoid / interp in code so NR stays stable).

**Training gradient path** (`train.HarmonicContourE2E.forward_batch`): pass-2 $h_{2m}^{\mathrm{lum}}, h_{2m}^{\mathrm{chr}}$ from `fast_l0_pass2` are **detached** (no grad through L0 NR/harmonics). The map $\eta_{\mathrm{lum}}(p)$ is copied into `l0_pix["eta_mod_map"]` **without** detaching from MLP$_\eta$. The renderer appends $\hat\eta_{\mathrm{mod}} = \eta_{\mathrm{lum}}$ as the **15th** input to the readout MLP so gradients reach MLP$_\eta$ through the gate only.

---

## 7. Renderer — interp, stencils, readout MLP

**θ combing** on the cell grid (double-angle, ρ-weighted smooth), then **bilinear** sampling of stacked fields to pixels → $\bar\rho(p), \bar\theta(p), \bar\kappa_{\mathrm{col}}(p)$.

**Stencils** on $h_{2m}^{\Sigma} = h_{2m}^{\mathrm{lum}} + h_{2m}^{\mathrm{chr}}$ with unit tangent $\hat t = (\cos\bar\theta,\sin\bar\theta)$ and normal $\hat n = (-\sin\bar\theta,\cos\bar\theta)$, learned spacings $s_t, s_n = \mathrm{softplus}(\cdot)$:

$$
\mathrm{tang}_j(p) = h_{2m}^{\Sigma}\bigl(p + j\, s_t\, \hat t(p)\bigr), \quad
\mathrm{norm}_j(p) = h_{2m}^{\Sigma}\bigl(p + j\, s_n\, \hat n(p)\bigr), \quad j \in \{-2,-1,0,1,2\}.
$$

**Feature vector** $F_p \in \mathbb{R}^{15}$:

$$
F_p = \bigl[
h_{2m}^{\mathrm{lum}},\, h_{2m}^{\mathrm{chr}},\, \bar\rho,\, \bar\kappa_{\mathrm{col}},\,
\mathrm{tang}_{-2},\ldots,\mathrm{tang}_{2},\,
\mathrm{norm}_{-2},\ldots,\mathrm{norm}_{2},\,
\hat\eta_{\mathrm{mod}}
\bigr].
$$

If `eta_mod_map` is absent (pass 1 only), $\hat\eta_{\mathrm{mod}} \equiv 0$.

**Boundary map**:

$$
\mathrm{gate}(p) = \sigma\bigl(\mathrm{MLP}_{\mathrm{read}}(F_p)\bigr), \qquad
\hat B(p) = h_{2m}^{\Sigma}(p)\,\bar\rho(p)\,\mathrm{gate}(p).
$$

$\mathrm{MLP}_{\mathrm{read}}$: **15 → 8 → 1** (ReLU between linear layers). Interp of $\bar\theta$ for geometry is **stopped** for autograd where noted in code.

---

## 8. Training wrap-up (`train.py`)

- **Disk cache** (`precompute_image`): L0 only; stores `h2m`, `theta_h`, `border_mask`, `l0_pix`, $d_k$, GT, geometry — **not** `cells_flat`. Versioned by `TRAIN.CACHE_VERSION`.
- **Each training step** (`prepare_batch` → `run_l1_live_cells`): L1 runs with the trainable `HypercolumnSeed` inside `model.seed.hc_seed` (`cells_format="torch"`), then `RhoSeedModule` reads dominant $\rho$ from the resulting `cells_flat["lam"][...,0]`. So **loss does update** $\tilde\eta_z$, $\tilde\alpha$ (GABA lateral step), and $\tilde\alpha_{d,t}$ through the live L1 graph (pass-2 $h_{2m}$ from `fast_l0_pass2` remains detached as before).
- **Does receive gradients**: `ModulationRenderer` (readout MLP + $s_t,s_n$), **`EtaRegionalMLP`** via the $\hat\eta_{\mathrm{mod}}$ feature path when pass-2 tensors are present, and **L1 seed** parameters via live L1.

**Spec vs.\ training reality (read this once):**

| Claim in prose | Training-time truth |
|----------------|---------------------|
| $\eta_z$, $\tilde\alpha$ (lateral), $\alpha_{d,t}$ are “learned” | **Updated** by the loss: L1 is re-run each batch with `model.seed.hc_seed`. |
| MLP$_\eta$ drives spatial $\eta(p)$ | **Yes** for the forward image, but pass-2 **$h_{2m}$** is **detached**, so the loss does **not** backprop through L0 NR/harmonics. Gradients to MLP$_\eta$ go only through the **15th readout feature** ($\hat\eta_{\mathrm{mod}}$). That path is weaker than modulating $h_{2m}$ directly. |
| MLP$_\eta$ init “near identity” | `EtaRegionalMLP` uses **small random** `fc1`/`fc2` weights (not all zeros) so ReLU is not dead at step 0; `fc2.bias` ≈ 2 keeps $\sigma(\text{logit})$ in a sensible band initially. |

Loss (default mix in `params.TRAIN`): soft-Dice and/or BCE on the valid η-band vs.\ ground-truth edges.

---

## 9. Learned parameter count (current spec)

| Block | Params | Notes |
|------:|-------:|------|
| $\tilde\alpha_d, \tilde\alpha_t$ | 2 | Collinear kernel scales |
| $\tilde\alpha$ (GABA lateral) | 1 | ``softplus`` → α in additive lateral term |
| $\tilde\eta_z$ | 1 | NR scale (pre-GABA and each GABA pass) |
| MLP$_\eta$ (3→8→1) | 41 | Regional η |
| $s_t, s_n$ | 2 | Stencil spacing |
| MLP$_{\mathrm{read}}$ (15→8→1) | 137 | Thinning gate |
| **Total** | **184** | |

---

## 10. Module map

| Symbol / stage | Primary code |
|----------------|----------------|
| $d_k$, NR, harmonics, `fast_l0_pass2` | `hci/L0.py` |
| Binning, GABA, dominant extract | `hci/L1.py` |
| `HypercolumnSeed`, $\eta_z$, $\alpha$, $\alpha_{d,t}$ | `hci/L1.py` (seed), `hci/seed.py` (wrapper) |
| Pool + MLP$_\eta$ | `hci/L0.py` (`EtaRegionalMLP`, `compute_eta_modulation_mlp`) |
| Renderer | `hci/renderer.py` |
| Cache, batch, loss | `train.py` |
| Single-image tooling | `infer.py`, `test.py` |

---

## Revision note

Earlier drafts of this file described a **hypothetical** K-channel readout replacing the renderer; the **running code** uses the pipeline above (dominant-bin cache + 15-D renderer + detached pass-2 h2m with $\eta_{\mathrm{lum}}$ in $F_p$). This document tracks the implementation.
