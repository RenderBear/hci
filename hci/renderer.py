r"""Renderer — soft-indicator deposit + noisy-OR aggregation (no suppression).

Pipeline:
  1. Cell grid: ρ-weighted θ combing → ρ-gated anchor smoothing.
  2. Per-cell features (6 scalars/cell): own ρ, neighborhood ρ, collinearity,
     curvature/disagreement, tangent-ρ asymmetry, normal-ρ asymmetry.
  3. MLP (6 → 16 → 8) emits basis weights w_k per cell.
  4. Each active cell produces a soft-indicator m_c(p) over a (2H+1)² patch:
        m_c(p) = exp(Σ_k w_k b_k(s,n) − max_p ...)  ∈ [0, 1],   max_p m_c = 1,
     optionally multiplied by a fixed isotropic envelope E(s,n)=exp(−(s²+n²)/(2σ_E²))
     and re-max-normalized so peaks stay at 1 while tails cannot extend far in (s,n)
     (structural cap on line / half-plane basis leakage into neighbors).
     (s, n) is the patch pixel in cell-tangent frame (half-width units).
  5. Per-cell claim c_c(p) = ρ_c · m_c(p) ∈ [0, 1].
  6. Output via noisy-OR of independent claims:
        B̂(p) = 1 − ∏_c (1 − c_c(p)).
     The renderer applies no suppression; precision lives at the seed.

Why soft-indicator not sum-normalized:
    Sum-normalize (Σ_p m_c = 1) is exactly conservative but spreads ρ_c=1 over
    the cell's full footprint → per-pixel values fall to ~1/|footprint| ≈ 0.01,
    below any NMS threshold.  Soft-indicator + noisy-OR makes ρ_c=1 mean "edge
    passes through my claimed pixel at confidence 1" and combines overlapping
    claims probabilistically.  Renderer remains a pure depositor.

Gestalt basis b_0..b_7 in cell-local coords (s = tangent, n = normal):
  b_0 = 0                              # uniform claim (flat after max-normalize)
  b_1 = -n²                             # line along edge
  b_2 = -s²                             # dot localized in tangent
  b_3 = -(s² + n²)                     # round blob
  b_4 = s                               # forward bias (MLP can output ±)
  b_5 = -(n - κ₀ s²)²                   # curve A
  b_6 = -(n + κ₀ s²)²                   # curve B
  b_7 = -((s, n) - L·v̂_c)²             # extremal deposit (self-gating)

The b_7 basis enables a cell to place its deposit *offset* from the anchor along
a per-cell extension vector v̂_c, derived from the cell's tangent / normal ρ
asymmetry features.  This addresses the "spike tip" failure mode: a spike-body
cell sees high own-ρ and high tangent-asymmetry (ρ trails off ahead along the
tangent) — b_7 lets the deposit shift forward toward the spike tip rather than
sit at the magnitude centroid behind the tip.  When asymmetry is small (a clean
line cell), |v̂_c| → 0 and b_7 collapses to b_3 (round blob), so the basis is
inactive on non-extremal cells.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import RENDER


# ═══════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════

# Deposit footprint: half-width in pixels = ceil(DEPOSIT_HALF_WIDTH_STRIDES * S)
_DEPOSIT_HALF_WIDTH_STRIDES = float(getattr(RENDER, "DEPOSIT_HALF_WIDTH_STRIDES", 2.0))
_DEPOSIT_HALF_WIDTH_MIN = int(getattr(RENDER, "DEPOSIT_HALF_WIDTH_MIN", 4))
_DEPOSIT_HALF_WIDTH_MAX = int(getattr(RENDER, "DEPOSIT_HALF_WIDTH_MAX", 24))

_FEATURE_DIM = 6
_BASIS_DIM = 8
_HIDDEN_DIM = int(getattr(RENDER, "DEPOSIT_HIDDEN", 16))

# Curvature shape parameter for curve basis (in normalized cell-frame coords)
_BASIS_KAPPA = float(getattr(RENDER, "BASIS_KAPPA", 0.5))

# Reference offset length for the extremal basis b_7 (in half-width units).
# b_7 places its peak at (L * v̂_s, L * v̂_n) — at half a footprint by default.
# Tunable via RENDER.EXTENSION_REF; range (0, 1].
_EXTENSION_REF = float(getattr(RENDER, "EXTENSION_REF", 0.5))

# Optional isotropic Gaussian on (s, n) in half-width units — applied to m_c after
# max-normalize, then re-max-normalized. σ_E≈0.4–0.6 matches stride-to-half-width scale.
_DEPOSIT_ENVELOPE_SIGMA = float(getattr(RENDER, "DEPOSIT_ENVELOPE_SIGMA", 0.52))

# Init priors for basis MLP output bias (favor "line" deposit at t=0)
_BASIS_INIT_BIAS = {
    0: 0.0,   # uniform
    1: 5.0,   # line  ← dominant at init
    2: 0.0,   # dot
    3: 0.0,   # blob
    4: 0.0,   # forward bias
    5: 0.0,   # curve A
    6: 0.0,   # curve B
    7: 0.0,   # extremal — inactive at init, opens when MLP learns to use it
}


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


# ═══════════════════════════════════════════════════════════════
# Cell-grid smoothing (unchanged from prior renderer)
# ═══════════════════════════════════════════════════════════════

def _smooth_theta_rho_double_angle(
    theta: torch.Tensor,
    rho: torch.Tensor,
    is_border: torch.Tensor,
    n_passes: int = RENDER.THETA_SMOOTH_PASSES,
    eps: float = 1e-6,
) -> torch.Tensor:
    nH, nW = theta.shape
    th, rh, ib = theta, rho, is_border
    pad = lambda t: F.pad(t[None, None], (1, 1, 1, 1)).squeeze(0).squeeze(0)
    for _ in range(int(n_passes)):
        rh_p = pad(rh)
        u_p = pad(rh * torch.cos(2.0 * th))
        v_p = pad(rh * torch.sin(2.0 * th))
        su = torch.zeros_like(th)
        sv = torch.zeros_like(th)
        sw = torch.zeros_like(th)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s = slice(1 + dy, 1 + dy + nH), slice(1 + dx, 1 + dx + nW)
                w = rh_p[s]
                su += w * u_p[s]
                sv += w * v_p[s]
                sw += w
        th_new = 0.5 * torch.atan2(sv / sw.clamp_min(eps), su / sw.clamp_min(eps))
        use = (~ib) & (sw > eps)
        th = torch.where(use, th_new, th)
    return th


def _smooth_anchors_rho_gated(
    cx: torch.Tensor, cy: torch.Tensor,
    theta: torch.Tensor, rho: torch.Tensor,
    is_border: torch.Tensor,
    nH: int, nW: int, eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    cx_g, cy_g = cx.reshape(nH, nW), cy.reshape(nH, nW)
    th_g, rh_g = theta.reshape(nH, nW), rho.reshape(nH, nW)
    ib = is_border.reshape(nH, nW)
    pad = lambda t: F.pad(t[None, None], (1, 1, 1, 1)).squeeze(0).squeeze(0)
    cx_p, cy_p, th_p, rh_p = pad(cx_g), pad(cy_g), pad(th_g), pad(rh_g)
    sum_wcx = torch.zeros_like(cx_g)
    sum_wcy = torch.zeros_like(cy_g)
    sum_w = torch.zeros_like(cx_g)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s = slice(1 + dy, 1 + dy + nH), slice(1 + dx, 1 + dx + nW)
            gate = torch.cos(2.0 * (th_p[s] - th_g)).pow(2)
            w = rh_p[s] * gate
            sum_wcx += w * cx_p[s]
            sum_wcy += w * cy_p[s]
            sum_w += w
    fallback = sum_w < 0.5 * rh_g.clamp_min(0.0)
    cx_new = torch.where(fallback | ib, cx_g, sum_wcx / sum_w.clamp_min(eps))
    cy_new = torch.where(fallback | ib, cy_g, sum_wcy / sum_w.clamp_min(eps))
    return cx_new.reshape(-1), cy_new.reshape(-1)


# ═══════════════════════════════════════════════════════════════
# Per-cell features for the deposit MLP
# ═══════════════════════════════════════════════════════════════

def _per_cell_features(
    rho_g: torch.Tensor,        # (nH, nW)
    theta_g: torch.Tensor,      # (nH, nW)  smoothed
    ib_g: torch.Tensor,         # (nH, nW)  bool
    eps: float = 1e-6,
) -> torch.Tensor:
    """6 scalar features per cell, computed via 3×3 neighborhood operations.

      f0: ρ_c
      f1: <ρ>_N                                  (3×3 neighborhood mean, excl. self)
      f2: <ρ cos(2(θ_nbr - θ_c))>_N / <ρ>_N      collinearity (+1 = aligned)
      f3: |<ρ sin(2(θ_nbr - θ_c))>_N| / <ρ>_N    disagreement magnitude
      f4: <ρ (Δ·t̂_c)>_N / <ρ>_N                  tangent asymmetry of ρ
      f5: <ρ (Δ·n̂_c)>_N / <ρ>_N                  normal asymmetry of ρ
    """
    nH, nW = rho_g.shape

    cos_t = torch.cos(theta_g)
    sin_t = torch.sin(theta_g)
    # Tangent t̂ = (cos θ, sin θ); normal n̂ = (-sin θ, cos θ)  in (x, y) = (col, row).

    cos2 = torch.cos(2.0 * theta_g)
    sin2 = torch.sin(2.0 * theta_g)

    def _pad0(t: torch.Tensor) -> torch.Tensor:
        return F.pad(t[None, None], (1, 1, 1, 1), value=0.0).squeeze(0).squeeze(0)

    rho_p  = _pad0(rho_g)
    rcos_p = _pad0(rho_g * cos2)
    rsin_p = _pad0(rho_g * sin2)

    sum_rho   = torch.zeros_like(rho_g)
    sum_rcos  = torch.zeros_like(rho_g)
    sum_rsin  = torch.zeros_like(rho_g)
    sum_rho_t = torch.zeros_like(rho_g)
    sum_rho_n = torch.zeros_like(rho_g)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sl = (slice(1 + dy, 1 + dy + nH), slice(1 + dx, 1 + dx + nW))
            rn = rho_p[sl]
            sum_rho  += rn
            sum_rcos += rcos_p[sl]
            sum_rsin += rsin_p[sl]
            # Neighbor offset direction in (x, y) = (col, row): (dx, dy).
            proj_t =  float(dx) * cos_t + float(dy) * sin_t
            proj_n = -float(dx) * sin_t + float(dy) * cos_t
            sum_rho_t += rn * proj_t
            sum_rho_n += rn * proj_n

    # Soft floor so empty-neighborhood cells produce 0-valued (not 1e6-scale)
    # features. With sum_rho ∈ [0, ~1] typically, soft=5e-2 caps inv_w at 20 and
    # smoothly suppresses ratios when fewer than ~0.05 worth of neighbor ρ is
    # present. Critical for forward numerical stability of the MLP / softmax.
    soft = 5e-2
    inv_w = 1.0 / (sum_rho + soft)

    # Rotate neighborhood phasor by -2θ_c:
    #   Σ ρ cos(2(θ_n - θ_c)) = sum_rcos · cos2_c + sum_rsin · sin2_c
    #   Σ ρ sin(2(θ_n - θ_c)) = sum_rsin · cos2_c - sum_rcos · sin2_c
    coh_num  = sum_rcos * cos2 + sum_rsin * sin2
    anti_num = sum_rsin * cos2 - sum_rcos * sin2

    mean_rho = sum_rho * (1.0 / 8.0)
    coh    = coh_num  * inv_w
    anti   = torch.abs(anti_num) * inv_w
    asym_t = sum_rho_t * inv_w
    asym_n = sum_rho_n * inv_w

    use = (~ib_g).to(dtype=rho_g.dtype)
    mean_rho = mean_rho * use
    coh      = coh      * use
    anti     = anti     * use
    asym_t   = asym_t   * use
    asym_n   = asym_n   * use

    feats = torch.stack([rho_g, mean_rho, coh, anti, asym_t, asym_n], dim=-1)
    return feats.reshape(-1, _FEATURE_DIM)


# ═══════════════════════════════════════════════════════════════
# Conservative deposit
# ═══════════════════════════════════════════════════════════════

def _deposit_half_width(S: int) -> int:
    h = int(math.ceil(_DEPOSIT_HALF_WIDTH_STRIDES * max(S, 1)))
    return max(_DEPOSIT_HALF_WIDTH_MIN, min(_DEPOSIT_HALF_WIDTH_MAX, h))


def _compute_basis(
    s: torch.Tensor,
    n: torch.Tensor,
    ext_s: torch.Tensor,
    ext_n: torch.Tensor,
) -> torch.Tensor:
    """Evaluate 8 basis functions at patch pixels.

    s, n: (A, P) cell-frame coordinates, normalized to half-width units
          (so |s|, |n| ≲ 1 within the deposit window).
    ext_s, ext_n: (A,) per-cell extension components in the tangent frame,
                  in half-width units, clamped to [-EXTENSION_REF, +EXTENSION_REF].
                  When |ext| → 0 the cell wants no displacement (b_7 collapses to
                  a centered blob); larger |ext| pushes the deposit toward the
                  spike-tip end of the cell's footprint.

    Returns: (A, P, 8).
    """
    s2 = s * s
    n2 = n * n
    kappa = _BASIS_KAPPA

    b0 = torch.zeros_like(s)                    # uniform
    b1 = -n2                                     # line along tangent
    b2 = -s2                                     # dot in tangent direction
    b3 = -(s2 + n2)                              # round blob (centered)
    b4 = s                                       # forward bias
    b5 = -(n - kappa * s2).pow(2)                # curve A
    b6 = -(n + kappa * s2).pow(2)                # curve B
    # b_7: extremal — round Gaussian peaked at (ext_s, ext_n).  Self-gating: when
    # |ext| ≈ 0 this is b_3 and adds nothing new (the MLP can already produce a
    # centered blob); when |ext| > 0 it places mass *offset* from the anchor by
    # an amount derived from the cell's own ρ-asymmetry features.
    ds = s - ext_s.unsqueeze(1)
    dn = n - ext_n.unsqueeze(1)
    b7 = -(ds * ds + dn * dn)
    return torch.stack([b0, b1, b2, b3, b4, b5, b6, b7], dim=-1)


def _conservative_deposit(
    rho_active: torch.Tensor,      # (A,)  ρ_c for active cells (grad-bearing from seed)
    cx_active: torch.Tensor,       # (A,)  anchor x (detached)
    cy_active: torch.Tensor,       # (A,)  anchor y (detached)
    theta_active: torch.Tensor,    # (A,)  tangent angle (detached)
    basis_w: torch.Tensor,         # (A, 8) basis weights from MLP
    ext_s: torch.Tensor,           # (A,) tangent-frame extension, in half-width units
    ext_n: torch.Tensor,           # (A,) normal-frame extension, in half-width units
    H: int, W: int, half_w: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cell soft-indicator deposit + noisy-OR aggregation.

    Per cell c, the patch-pixel mass is normalized to **peak at 1**, not to **sum to 1**:

        m_c(p) = exp(ℓ_c(p) − max_p ℓ_c(p))  ∈ [0, 1],   max_p m_c(p) = 1,
     optionally times E(s,n)=exp(−(s²+n²)/(2σ_E²)) with a second max-normalize
     (see RENDER.DEPOSIT_ENVELOPE_SIGMA; 0 disables).

    The cell's per-pixel **claim** is c_c(p) = ρ_c · m_c(p) ∈ [0, 1].  Each claim is
    scattered to the four nearest pixels via bilinear weights of the sub-pixel anchor
    (integer base + fractional part).  Pixel-wise aggregation is **noisy-OR**:

        B̂(p) = 1 − ∏_c (1 − c_c(p))   =   1 − exp(Σ_c log(1 − c_c(p))).

    Why this and not the sum/softmax (Σ_p m_c = 1) form:
      The sum-normalized form spreads ρ_c=1 over the cell's footprint, so per-pixel
      values fall to ~1/|footprint| ≈ 0.01 — below any NMS threshold.  Soft-indicator
      + noisy-OR makes ρ_c=1 mean "edge passes through my claimed pixel at confidence 1"
      and combines overlapping claims probabilistically (two cells at ρ=0.5 → 0.75; two
      collinear cells at ρ=1 → 1).  The renderer still applies no suppression —
      precision still comes from upstream (the seed); the renderer only places mass.

    Returns:
        bmap: (H, W) ∈ [0, 1]   — pixel-wise noisy-OR of all cell claims.
        theta_star: (H, W)       — tangent angle of the dominant claimant per pixel
                                  (used for inference-time NMS).
    """
    A = rho_active.shape[0]
    device, dtype = rho_active.device, rho_active.dtype
    n_pix = H * W

    if A == 0:
        z = torch.zeros(H, W, device=device, dtype=dtype)
        return z, z

    # Patch offsets (oy, ox) in pixel units; flatten to (P,).
    offsets = torch.arange(-half_w, half_w + 1, device=device, dtype=dtype)
    oy_g, ox_g = torch.meshgrid(offsets, offsets, indexing="ij")
    oy = oy_g.reshape(-1)
    ox = ox_g.reshape(-1)
    P = oy.shape[0]

    inv_hw = 1.0 / float(max(half_w, 1))

    # Process in batches to bound peak memory.
    max_batch = max(1, 4_000_000 // P)

    # Noisy-OR accumulator: Σ_c log(1 − claim_c(p)).  Starts at 0 → bmap(p)=0.
    log_neg_acc = torch.zeros(n_pix, device=device, dtype=dtype)

    # Track max-claim per pixel for θ★ (used for ridge NMS at inference).
    max_claim = torch.full((n_pix,), -1.0, device=device, dtype=torch.float32)
    theta_star = torch.zeros(n_pix, device=device, dtype=dtype)

    # Clamp ρ to [0, 1−ε_clip] for log-stability of log1p(−claim).
    # The seed's divisive readout ρ = e²/(e²+η_readout²+λS²+ε) is already in [0, 1] but a
    # rare cell can come in slightly above due to numerical noise.
    _CLIP = 1.0 - 1e-5
    rho_clamped = rho_active.clamp(min=0.0, max=_CLIP)

    # Sub-pixel anchor: integer base + fractional bilinear weights.
    ax_int = torch.floor(cx_active).long()
    ay_int = torch.floor(cy_active).long()
    ax_frac = cx_active - ax_int.to(dtype=dtype)
    ay_frac = cy_active - ay_int.to(dtype=dtype)

    for b0 in range(0, A, max_batch):
        b1 = min(b0 + max_batch, A)
        bs = b1 - b0

        ax_f = ax_frac[b0:b1].unsqueeze(1)   # (bs, 1)
        ay_f = ay_frac[b0:b1].unsqueeze(1)
        w_NN = (1.0 - ax_f) * (1.0 - ay_f)
        w_NE = ax_f * (1.0 - ay_f)
        w_SW = (1.0 - ax_f) * ay_f
        w_SE = ax_f * ay_f

        # Cell-frame coords relative to sub-pixel anchor (not integer base).
        cos_a = torch.cos(theta_active[b0:b1]).unsqueeze(1)   # (bs, 1)
        sin_a = torch.sin(theta_active[b0:b1]).unsqueeze(1)
        ox_eff = ox.unsqueeze(0) - ax_f
        oy_eff = oy.unsqueeze(0) - ay_f
        s_loc = (ox_eff * cos_a + oy_eff * sin_a) * inv_hw   # (bs, P)
        n_loc = (-ox_eff * sin_a + oy_eff * cos_a) * inv_hw

        basis = _compute_basis(
            s_loc, n_loc,
            ext_s[b0:b1], ext_n[b0:b1],
        )                                                       # (bs, P, 8)
        logits = (basis * basis_w[b0:b1].unsqueeze(1)).sum(dim=-1)   # (bs, P)

        # Bilinear scatter targets from integer anchor base + patch offsets.
        ax_b = ax_int[b0:b1].unsqueeze(1)
        ay_b = ay_int[b0:b1].unsqueeze(1)
        px_NN = ax_b + ox.unsqueeze(0)
        py_NN = ay_b + oy.unsqueeze(0)
        px_NE = px_NN + 1
        py_NE = py_NN
        px_SW = px_NN
        py_SW = py_NN + 1
        px_SE = px_NN + 1
        py_SE = py_NN + 1

        in_bounds_NN = (py_NN >= 0) & (py_NN < H) & (px_NN >= 0) & (px_NN < W)
        in_bounds_NE = (py_NE >= 0) & (py_NE < H) & (px_NE >= 0) & (px_NE < W)
        in_bounds_SW = (py_SW >= 0) & (py_SW < H) & (px_SW >= 0) & (px_SW < W)
        in_bounds_SE = (py_SE >= 0) & (py_SE < H) & (px_SE >= 0) & (px_SE < W)
        in_bounds_any = in_bounds_NN | in_bounds_NE | in_bounds_SW | in_bounds_SE

        # Soft-max-normalize: mass peaks at 1 (not Σ = 1).
        # Mask OOB to −∞ so they don't influence the max or contribute to the deposit.
        logits = torch.where(in_bounds_any, logits, torch.full_like(logits, -1e9))
        logits_max = logits.max(dim=1, keepdim=True).values
        mass = torch.exp(logits - logits_max) * in_bounds_any.to(dtype=dtype)

        # Compact footprint: basis b_1 / b_4 can assign near-peak mass far along s or n
        # within the patch; multiply by E(s,n) and re-max-normalize so max m_c = 1.
        if _DEPOSIT_ENVELOPE_SIGMA > 0.0:
            sig = float(_DEPOSIT_ENVELOPE_SIGMA)
            rsq = s_loc * s_loc + n_loc * n_loc
            env = torch.exp(-rsq / (2.0 * sig * sig))
            mass = mass * env
            mm = mass.max(dim=1, keepdim=True).values
            mass = torch.where(mm > eps, mass / mm, mass)

        # Per-cell claim, clipped strictly below 1 for log1p stability.
        claim = (rho_clamped[b0:b1].unsqueeze(1) * mass).clamp(min=0.0, max=_CLIP)

        claim_NN = (claim * w_NN * in_bounds_NN.to(dtype=dtype)).clamp(min=0.0, max=_CLIP)
        claim_NE = (claim * w_NE * in_bounds_NE.to(dtype=dtype)).clamp(min=0.0, max=_CLIP)
        claim_SW = (claim * w_SW * in_bounds_SW.to(dtype=dtype)).clamp(min=0.0, max=_CLIP)
        claim_SE = (claim * w_SE * in_bounds_SE.to(dtype=dtype)).clamp(min=0.0, max=_CLIP)

        log_neg_NN = torch.log1p(-claim_NN)
        log_neg_NE = torch.log1p(-claim_NE)
        log_neg_SW = torch.log1p(-claim_SW)
        log_neg_SE = torch.log1p(-claim_SE)

        flat_idx_NN = (py_NN.long().clamp(0, H - 1) * W + px_NN.long().clamp(0, W - 1))
        flat_idx_NE = (py_NE.long().clamp(0, H - 1) * W + px_NE.long().clamp(0, W - 1))
        flat_idx_SW = (py_SW.long().clamp(0, H - 1) * W + px_SW.long().clamp(0, W - 1))
        flat_idx_SE = (py_SE.long().clamp(0, H - 1) * W + px_SE.long().clamp(0, W - 1))

        log_neg_acc.scatter_add_(0, flat_idx_NN.reshape(-1), log_neg_NN.reshape(-1))
        log_neg_acc.scatter_add_(0, flat_idx_NE.reshape(-1), log_neg_NE.reshape(-1))
        log_neg_acc.scatter_add_(0, flat_idx_SW.reshape(-1), log_neg_SW.reshape(-1))
        log_neg_acc.scatter_add_(0, flat_idx_SE.reshape(-1), log_neg_SE.reshape(-1))

        # Track dominant cell per pixel by max bilinear-fraction claim (for θ★).
        th_expand = theta_active[b0:b1].unsqueeze(1).expand(bs, P).reshape(-1)
        for flat_idx_v, claim_t in (
            (flat_idx_NN.reshape(-1), claim_NN.detach().to(torch.float32).reshape(-1)),
            (flat_idx_NE.reshape(-1), claim_NE.detach().to(torch.float32).reshape(-1)),
            (flat_idx_SW.reshape(-1), claim_SW.detach().to(torch.float32).reshape(-1)),
            (flat_idx_SE.reshape(-1), claim_SE.detach().to(torch.float32).reshape(-1)),
        ):
            cur = max_claim[flat_idx_v]
            upd = claim_t > cur
            if upd.any():
                upd_idx = flat_idx_v[upd]
                max_claim[upd_idx] = claim_t[upd]
                theta_star[upd_idx] = th_expand[upd].to(dtype=dtype)

    # Noisy-OR readout: B̂(p) = 1 − exp(Σ_c log(1−claim)) = −expm1(accumulator).
    bmap = -torch.expm1(log_neg_acc)   # in [0, 1]
    return bmap.reshape(H, W), theta_star.reshape(H, W)


# ═══════════════════════════════════════════════════════════════
# Basis MLP
# ═══════════════════════════════════════════════════════════════

class DepositMLP(nn.Module):
    """6 → 16 → 7 MLP producing basis weights per cell.

    Init: fc1 small-Gaussian, fc2.weight small-Gaussian (~0.01 std), fc2.bias set
    so basis-1 ("line") dominates at t = 0 → renderer starts as a Gaussian-line-
    shaped deposit.  Basis-7 (extremal) has bias 0 — inactive at init, activates
    when the MLP learns to use it.
    """

    def __init__(
        self,
        in_dim: int = _FEATURE_DIM,
        hidden_dim: int = _HIDDEN_DIM,
        out_dim: int = _BASIS_DIM,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self._init_priors()

    def _init_priors(self) -> None:
        # fc1: standard ReLU init (Kaiming) so the hidden layer activates from t = 0.
        # fc2: SMALL Gaussian (not zero) so features influence the output immediately
        #      — bias still dominates (line prior), but fc1.weight receives gradient
        #      from the first backward pass instead of being dead until fc2.weight
        #      itself starts moving.
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
            self.fc1.bias.zero_()
            nn.init.normal_(self.fc2.weight, std=0.01)
            for k, b in _BASIS_INIT_BIAS.items():
                if 0 <= k < self.fc2.bias.numel():
                    self.fc2.bias[k] = float(b)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, 6). Returns: (N, 8) basis weights (unbounded)."""
        h = F.relu(self.fc1(features))
        return self.fc2(h)


# ═══════════════════════════════════════════════════════════════
# ModulationRenderer
# ═══════════════════════════════════════════════════════════════

class ModulationRenderer(nn.Module):
    """Soft-indicator deposit + noisy-OR renderer.

    Learned: DepositMLP 6→16→7 (231 params).  Total: 231.

    Aggregation:
        Per cell c, soft-indicator m_c(p) ∈ [0, 1] with max_p m_c(p) = 1.
        Per-cell claim c_c(p) = ρ_c · m_c(p).
        Output B̂(p) = 1 − ∏_c (1 − c_c(p))   ∈ [0, 1].

        The renderer never suppresses or amplifies — the seed is the only
        amplitude decision-maker.  The renderer chooses *where* each cell
        places its claim within its footprint and combines overlapping
        claims as independent evidence.

    Note: this is NOT mass-conservative (Σ_p B̂ ≠ Σ_c ρ_c in general).  An
    earlier version was conservative via per-cell softmax (Σ_p m_c = 1),
    but that diluted ρ_c=1 over the footprint into per-pixel values too
    small to survive NMS thresholding.  Soft-indicator + noisy-OR keeps
    the "renderer-only-places-mass" principle while letting ρ_c=1 produce
    pixel values near 1.
    """

    def __init__(self, **kwargs):
        super().__init__()
        _ = kwargs  # tolerate legacy kwargs
        self.deposit = DepositMLP()
        # Learnable scale mapping ρ-asymmetry features (in cell-grid units) to
        # the extension vector (in half-width units).  softplus so it stays > 0;
        # init at ~1.0 so initial extension magnitudes match the natural feature
        # scale.  When the system finds extension unhelpful it can shrink this
        # toward 0, collapsing b_7 back to a centered blob.
        self._ext_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0)))

    @property
    def ext_scale(self) -> torch.Tensor:
        return F.softplus(self._ext_scale_raw).view(())

    # ── Legacy compat properties / methods (no effect on output) ─────────
    @property
    def sigma_perp(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def sigma_par(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def sigma_pre(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def smooth_sigma(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def eta_h(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def s_t(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def s_n(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    def pixel_map(self, profile: torch.Tensor) -> torch.Tensor:
        return torch.ones(profile.shape[0], device=profile.device, dtype=profile.dtype)


def upgrade_renderer_state_dict(state_dict: dict, prefix: str = "") -> dict:
    """Strip legacy renderer keys before load_state_dict(strict=False)."""
    legacy_keys = {
        f"{prefix}_sigma_pre_raw",
        f"{prefix}_sigma_perp_raw",
        f"{prefix}_sigma_par_raw",
        f"{prefix}_eta_h_raw",
        f"{prefix}_smooth_sigma_raw",
        f"{prefix}s_t",
        f"{prefix}s_n",
        f"{prefix}perp_conv.conv.weight",
        f"{prefix}perp_conv.conv.bias",
        f"{prefix}perp_conv.fc.weight",
        f"{prefix}perp_conv.fc.bias",
        # legacy thinning head — fully removed in this renderer
        f"{prefix}thinning.fc1.weight",
        f"{prefix}thinning.fc1.bias",
        f"{prefix}thinning.fc2.weight",
        f"{prefix}thinning.fc2.bias",
    }
    return {k: v for k, v in state_dict.items() if k not in legacy_keys}


# ═══════════════════════════════════════════════════════════════
# Proj helpers
# ═══════════════════════════════════════════════════════════════

def proj_to_device(proj: dict, device: torch.device) -> dict:
    return {
        "H": proj["H"], "W": proj["W"],
        "n_cells": proj["n_cells"],
        "nH": proj["nH"], "nW": proj["nW"],
    }


def compute_render_features(
    z2_image: np.ndarray, img: np.ndarray,
    cells: dict, border_mask: np.ndarray,
    eps: float = 1e-9, **kwargs,
) -> dict:
    _ = (z2_image, img, cells, border_mask, eps, kwargs)
    H, W = z2_image.shape
    nH, nW = cells["nH"], cells["nW"]
    return {"H": H, "W": W, "n_cells": nH * nW, "nH": nH, "nW": nW}


def _theta_on_branch(theta, branch_pick, n_cells, device):
    if theta.dim() == 1:
        return theta.to(device=device)
    if branch_pick is not None:
        b = branch_pick.to(device=device, dtype=torch.long).view(-1)
        idx_n = torch.arange(n_cells, device=device, dtype=torch.long)
        return theta[idx_n, b]
    return theta[:, 0]


# ═══════════════════════════════════════════════════════════════
# render_boundary_map_torch
# ═══════════════════════════════════════════════════════════════

def render_boundary_map_torch(
    rho_cell: torch.Tensor,
    proj_dev: dict,
    renderer: ModulationRenderer,
    cells_flat: dict,
    Hp: int, Wp: int,
    l0_pix: Optional[dict[str, torch.Tensor]] = None,
    eps: float = 1e-6,
    training: bool = False,
    branch_pick: Optional[torch.Tensor] = None,
    content_h: Optional[int] = None,
    content_w: Optional[int] = None,
    return_dominant_theta: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Soft-indicator deposit + noisy-OR: B̂(p) = 1 − ∏_c (1 − ρ_c · m_c(p))."""
    _ = (training, l0_pix)
    H, W = proj_dev["H"], proj_dev["W"]
    device, dtype = rho_cell.device, rho_cell.dtype

    if "cx_z2" not in cells_flat or "cy_z2" not in cells_flat:
        raise ValueError("cells_flat must include cx_z2 and cy_z2")

    nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
    n_cells = int(proj_dev["n_cells"])
    theta_all = cells_flat["theta"].to(device=device, dtype=dtype)
    theta_c = _theta_on_branch(theta_all, branch_pick, n_cells, device)
    S = int(cells_flat.get("S", max(1, W // max(nW, 1))))

    # ── Step 1: Cell-grid θ smoothing ─────────────────────────
    rho_grid = rho_cell.reshape(nH, nW).to(dtype=dtype)
    ib_grid = cells_flat["is_border"].to(device=device).reshape(nH, nW).bool()
    theta_c = _smooth_theta_rho_double_angle(
        theta_c.reshape(nH, nW), rho_grid, ib_grid, eps=eps,
    ).reshape(-1)

    # ── Step 2: ρ-gated anchor smoothing ──────────────────────
    cx = cells_flat["cx_z2"].to(device=device, dtype=dtype)
    cy = cells_flat["cy_z2"].to(device=device, dtype=dtype)
    rho_flat = rho_cell.reshape(-1).to(dtype=dtype)
    is_border = cells_flat["is_border"].to(device=device).reshape(-1).bool()

    _cx, _cy = _smooth_anchors_rho_gated(
        cx, cy, theta_c, rho_flat, is_border, nH, nW, eps=eps,
    )

    active = (rho_flat > 0) & (~is_border)
    if not bool(active.any().item()):
        z = (rho_cell.sum() * 0.0).to(dtype=dtype, device=device)
        out = z.expand(H, W)[:Hp, :Wp].contiguous()
        if return_dominant_theta:
            return out, torch.zeros_like(out)
        return out

    # Detach anchors and θ before deposit: no coordinate gradients into seed.
    cx_det, cy_det = _cx.detach(), _cy.detach()
    theta_det = theta_c.detach()
    rho_safe = torch.where(is_border, torch.zeros_like(rho_flat), rho_flat)

    # ── Step 3: Per-cell features (cell grid, all cells) ──────
    # Features are pure descriptors — they decide deposit *shape*, not amplitude.
    # Detach both ρ and θ so gradient cannot loop back to L0/seed via the feature
    # path; the MLP weights still receive gradient (their gradient depends on the
    # *values* of features, not on features being part of the autograd graph).
    # Without this, normalization terms like 1/(sum_rho + eps) on empty-
    # neighborhood cells inflate Jacobians by ~1e6 and produce non-finite grads
    # in l0_metric.W and the seed's NR / β parameters.
    feats_all = _per_cell_features(
        rho_safe.reshape(nH, nW).detach(),
        theta_c.reshape(nH, nW).detach(),
        ib_grid,
        eps=eps,
    )  # (N, 6)

    # ── Step 4: MLP → basis weights for active cells ──────────
    active_idx = active.nonzero(as_tuple=True)[0]
    basis_w_active = renderer.deposit(feats_all[active_idx])  # (A, 8)

    # ── Step 4b: Extension vector (ext_s, ext_n) per active cell ─────────
    # Derived from the tangent / normal ρ-asymmetry features (cols 4, 5 of F_c),
    # mapped to half-width units via a learned scale (renderer.ext_scale, ≥ 0)
    # and clamped to ±EXTENSION_REF.  Conceptually: when ρ trails off ahead of the
    # cell along the tangent (positive asym_t = neighbors-have-more-ρ in +t̂),
    # the cell is *behind* the perceptual feature — the extension points in −t̂
    # (toward the magnitude-centroid side), so we negate.
    asym_t = feats_all[active_idx, 4]
    asym_n = feats_all[active_idx, 5]
    ext_scale = renderer.ext_scale  # softplus, learnable scalar; init ≈ 1.0
    ext_s = (-asym_t * ext_scale).clamp(min=-_EXTENSION_REF, max=_EXTENSION_REF)
    ext_n = (-asym_n * ext_scale).clamp(min=-_EXTENSION_REF, max=_EXTENSION_REF)

    # ── Step 5: Conservative deposit ──────────────────────────
    half_w = _deposit_half_width(S)
    bmap, theta_star = _conservative_deposit(
        rho_active=rho_safe[active_idx],
        cx_active=cx_det[active_idx],
        cy_active=cy_det[active_idx],
        theta_active=theta_det[active_idx],
        basis_w=basis_w_active,
        ext_s=ext_s,
        ext_n=ext_n,
        H=H, W=W, half_w=half_w, eps=eps,
    )

    # ── Crop to content region ────────────────────────────────
    ch = Hp if content_h is None else content_h
    cw = Wp if content_w is None else content_w
    ch, cw = min(ch, H), min(cw, W)
    if content_h is not None and content_w is not None:
        crop = torch.ones_like(bmap)
        if ch < H: crop[ch:, :] = 0.0
        if cw < W: crop[:, cw:] = 0.0
        bmap = bmap * crop
        theta_star = theta_star * crop

    out = bmap[:Hp, :Wp]
    if return_dominant_theta:
        return out, theta_star[:Hp, :Wp]
    return out


# ═══════════════════════════════════════════════════════════════
# NumPy wrapper (unchanged interface)
# ═══════════════════════════════════════════════════════════════

def render_boundary_map(
    rho_cell: np.ndarray, proj: dict,
    renderer: ModulationRenderer, cells_flat: dict,
    l0_pix: Optional[dict[str, np.ndarray]] = None,
    device: torch.device = torch.device("cpu"),
    eps: float = 1e-6,
    branch_pick: Optional[np.ndarray] = None,
    content_h: Optional[int] = None, content_w: Optional[int] = None,
) -> np.ndarray:
    proj_dev = proj_to_device(proj, device)
    rho_t = torch.from_numpy(np.asarray(rho_cell, dtype=np.float32)).to(device)
    cf_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in cells_flat.items()}
    bp = None
    if branch_pick is not None:
        bp = torch.from_numpy(np.asarray(branch_pick, dtype=np.int64).ravel()).to(device)
    with torch.no_grad():
        bmap_t = render_boundary_map_torch(
            rho_t, proj_dev, renderer, cf_dev, proj["H"], proj["W"], None,
            eps=eps, training=False, branch_pick=bp,
            content_h=content_h, content_w=content_w,
        )
    return bmap_t.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# NMS (unchanged)
# ═══════════════════════════════════════════════════════════════

def _nms_unit_normal_from_theta(theta, eps=1e-8):
    _ = eps
    t = np.asarray(theta, dtype=np.float64)
    return np.cos(t).astype(np.float32), (-np.sin(t)).astype(np.float32)


def _nms_unit_normal_from_gradient(mag, eps=1e-8):
    m = np.asarray(mag, dtype=np.float64)
    gx, gy = ndimage.sobel(m, axis=1), ndimage.sobel(m, axis=0)
    norm = np.sqrt(gx * gx + gy * gy) + eps
    return (gx / norm).astype(np.float32), (gy / norm).astype(np.float32)


def _nms_bilinear_sample(mag, row_off, col_off):
    coords = np.stack([row_off.astype(np.float64), col_off.astype(np.float64)])
    return ndimage.map_coordinates(
        mag.astype(np.float64), coords, order=1, mode="nearest"
    ).astype(np.float32)


def ridge_nms(mag, *, theta=None, grad_norm_floor=1e-7):
    m = np.asarray(mag, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"ridge_nms expects 2D, got {m.shape}")
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m_work = m.copy()
    if theta is not None:
        nx, ny = _nms_unit_normal_from_theta(np.asarray(theta, dtype=np.float32))
        weak = np.zeros((H, W), dtype=bool)
    else:
        gx = ndimage.sobel(m_work.astype(np.float64), axis=1)
        gy = ndimage.sobel(m_work.astype(np.float64), axis=0)
        gnorm = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        nx, ny = _nms_unit_normal_from_gradient(m_work)
        weak = gnorm < grad_norm_floor
    ahead = _nms_bilinear_sample(m_work, yy + ny, xx + nx)
    behind = _nms_bilinear_sample(m_work, yy - ny, xx - nx)
    keep = ((m_work >= ahead) & (m_work >= behind)) | weak
    return np.where(keep, m_work, 0.0).astype(np.float32)


def ridge_nms_binary(mag, threshold, *, theta=None, grad_norm_floor=1e-7):
    return (
        ridge_nms(mag, theta=theta, grad_norm_floor=grad_norm_floor) >= threshold
    ).astype(np.uint8) * 255


def cell_rho_to_2branch(rho_cell, branch):
    out = np.zeros((*rho_cell.shape, 2), dtype=rho_cell.dtype)
    ii, jj = np.indices(rho_cell.shape)
    out[ii, jj, branch.astype(np.int64)] = rho_cell
    return out


# Legacy aliases
HarmonicThinRenderer = ModulationRenderer
StampRenderer = ModulationRenderer
AnisoDiffusionRenderer = ModulationRenderer
