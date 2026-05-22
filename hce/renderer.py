r"""Renderer — Gaussian-line splat + collinear coherence + photometric thinning.

Pipeline:
  1. Cell grid: ρ-weighted θ combing → ρ-gated anchor smoothing.
  2. Collinear coherence κ_col on cell grid (binned-θ conv, radius R).  (NEW)
  3. Gaussian-line splat of ρ★ → ρ̄(p), θ★(p).
  4. Splat s_photo, κ_col to pixel resolution.
  5. Per-pixel feature vector F_p ∈ R¹⁶:
       [ρ̄, s̄_photo, h2m_lum, h2m_chr, κ̄_col, coh, tang5(5), norm5(5)]
  6. Thinning head: B̂(p) = ρ̄(p) · σ(W₂ ReLU(W₁ F_p + b₁) + b₂).
     16→12→1 MLP with structural priors.

Collinear coherence (Step 2):
  For each cell c, measure orientation agreement with cells along c's
  tangent direction within radius R on the cell grid.  θ is quantized
  into K bins; each bin has a precomputed (2R+1)² kernel encoding
  Gaussian distance weighting × Gaussian tangent-line selectivity.
  Two convolutions per bin (cos2θ, sin2θ channels) → per-cell κ_col.
  Reinforces straight co-oriented edges; suppresses isolated/noisy cells.

Learned: σ⊥ (1), s_t (1), s_n (1), ThinningHead 16→12→1 (217 params).
Fixed: collinear kernels (precomputed from R, σ_d, σ_t, K).
Total: 220 learned scalars.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import RENDER


# ═══════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════

_SIGMA_PERP_INIT = getattr(RENDER, "SIGMA_PERP_INIT", 1.5)
_SIGMA_PERP_MAX = getattr(RENDER, "SIGMA_PERP_MAX", 8.0)
_SPLAT_RADIUS_SIGMAS = getattr(RENDER, "SPLAT_RADIUS_SIGMAS", 3.0)
_SPLAT_HALF_W_PERP = getattr(RENDER, "SPLAT_HALF_W_PERP", 3)

# Collinear coherence defaults
_COL_RADIUS = getattr(RENDER, "COL_RADIUS", 5)
_COL_K_BINS = getattr(RENDER, "COL_K_BINS", 24)
_COL_SIGMA_D = getattr(RENDER, "COL_SIGMA_D", None)   # default: R/2
_COL_SIGMA_T = getattr(RENDER, "COL_SIGMA_T", 1.0)


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


# ═══════════════════════════════════════════════════════════════
# Cell-grid smoothing  (unchanged)
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
        th_p, rh_p = pad(th), pad(rh)
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
# Collinear coherence (binned-θ convolution on cell grid)
# ═══════════════════════════════════════════════════════════════

def _build_collinear_kernels(
    R: int,
    K: int,
    sigma_d: float | None,
    sigma_t: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute K kernels of shape (2R+1, 2R+1).

    For bin k with angle θ_k = k·π/K:
      W_k(di, dj) = w_d(di,dj) · w_t(di,dj; θ_k)
      w_d = exp(-(di²+dj²) / 2σ_d²)
      w_t = exp(-d_perp² / 2σ_t²)   where d_perp = dj·cos(θ_k) - di·sin(θ_k)

    Centre pixel (0,0) is excluded (self not counted).

    Returns: (K, 1, 2R+1, 2R+1) ready for F.conv2d.
    """
    if sigma_d is None:
        sigma_d = max(R / 2.0, 0.5)
    offsets = torch.arange(-R, R + 1, device=device, dtype=dtype)
    di, dj = torch.meshgrid(offsets, offsets, indexing="ij")  # (2R+1, 2R+1)
    dist_sq = di * di + dj * dj
    w_d = torch.exp(-dist_sq / (2.0 * sigma_d * sigma_d))
    # Zero the centre
    w_d[R, R] = 0.0
    # Zero outside radius
    w_d[dist_sq > R * R] = 0.0

    kernels = torch.zeros(K, 2 * R + 1, 2 * R + 1, device=device, dtype=dtype)
    for k in range(K):
        theta_k = k * math.pi / K
        d_perp = dj * math.cos(theta_k) - di * math.sin(theta_k)
        w_t = torch.exp(-d_perp * d_perp / (2.0 * sigma_t * sigma_t))
        kernels[k] = w_d * w_t

    return kernels.unsqueeze(1)  # (K, 1, 2R+1, 2R+1)


def compute_collinear_coherence(
    theta_grid: torch.Tensor,
    rho_grid: torch.Tensor,
    is_border_grid: torch.Tensor,
    R: int = _COL_RADIUS,
    K: int = _COL_K_BINS,
    sigma_d: float | None = _COL_SIGMA_D,
    sigma_t: float = _COL_SIGMA_T,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-cell collinear coherence κ_col on the cell grid.

    θ is quantized into K bins. For each bin, a precomputed (2R+1)²
    kernel encodes distance × tangent-selectivity weighting.  The
    double-angle fields ρ·cos(2θ) and ρ·sin(2θ) are convolved with
    each bin's kernel; each cell reads from its θ-bin's output.

    Args:
        theta_grid: (nH, nW) orientations in [0, π)
        rho_grid:   (nH, nW) cell strengths
        is_border_grid: (nH, nW) bool
        R, K, sigma_d, sigma_t: kernel parameters

    Returns:
        kappa_col: (nH, nW) collinear coherence in [0, 1]
    """
    device, dtype = theta_grid.device, theta_grid.dtype
    nH, nW = theta_grid.shape

    # Mask borders
    rho_m = torch.where(is_border_grid, torch.zeros_like(rho_grid), rho_grid)

    # Double-angle representation (handles π-periodicity)
    u = rho_m * torch.cos(2.0 * theta_grid)  # ρ·cos(2θ)
    v = rho_m * torch.sin(2.0 * theta_grid)  # ρ·sin(2θ)

    # Build kernels: (K, 1, 2R+1, 2R+1)
    kernels = _build_collinear_kernels(R, K, sigma_d, sigma_t, device, dtype)

    # Reshape for conv2d: (1, 1, nH, nW)
    u_4d = u.unsqueeze(0).unsqueeze(0)
    v_4d = v.unsqueeze(0).unsqueeze(0)
    rho_4d = rho_m.unsqueeze(0).unsqueeze(0)

    # Convolve: each bin k gives the weighted sum over the neighborhood
    # using that bin's tangent-selective kernel.
    # Output: (1, K, nH, nW)
    conv_u = F.conv2d(u_4d, kernels, padding=R)   # (1, K, nH, nW)
    conv_v = F.conv2d(v_4d, kernels, padding=R)   # (1, K, nH, nW)
    conv_rho = F.conv2d(rho_4d, kernels, padding=R)  # (1, K, nH, nW)

    # Assign each cell to its θ-bin
    # θ ∈ [0, π) → bin index ∈ [0, K)
    bin_idx = ((theta_grid % math.pi) * (K / math.pi)).long().clamp(0, K - 1)
    # (nH, nW)

    # Gather from the K channels using each cell's bin index
    # conv_u is (1, K, nH, nW), we need to pick channel bin_idx[i,j] at each (i,j)
    bin_idx_4d = bin_idx.unsqueeze(0).unsqueeze(0)  # (1, 1, nH, nW)
    sum_u = torch.gather(conv_u, 1, bin_idx_4d).squeeze(0).squeeze(0)  # (nH, nW)
    sum_v = torch.gather(conv_v, 1, bin_idx_4d).squeeze(0).squeeze(0)
    sum_rho = torch.gather(conv_rho, 1, bin_idx_4d).squeeze(0).squeeze(0)

    # κ_col = |weighted double-angle vector| / (weighted ρ sum)
    # This measures how well neighbors along the tangent agree on orientation.
    # |Σ w·ρ·exp(2iθ)| / Σ w·ρ  ∈ [0, 1]
    agreement_mag = (sum_u * sum_u + sum_v * sum_v).sqrt()
    kappa_col = agreement_mag / sum_rho.clamp_min(eps)
    kappa_col = kappa_col.clamp(0.0, 1.0)

    # Zero borders
    kappa_col = torch.where(is_border_grid, torch.zeros_like(kappa_col), kappa_col)

    return kappa_col


# ═══════════════════════════════════════════════════════════════
# Gaussian-line splat (vectorized scatter_add) — unchanged
# ═══════════════════════════════════════════════════════════════

def _gaussian_line_splat(
    values: torch.Tensor,
    cx: torch.Tensor, cy: torch.Tensor,
    theta: torch.Tensor, is_border: torch.Tensor,
    sigma_perp: torch.Tensor,
    H: int, W: int, S: int,
    radius_sigmas: float = _SPLAT_RADIUS_SIGMAS,
    half_w_perp: int = _SPLAT_HALF_W_PERP,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deposit values + compute dominant θ★ per pixel (scatter-max by ρ★φ).

    Returns: (rho_bar, theta_star) both (H, W).
    """
    device, dtype = values.device, values.dtype
    sig = sigma_perp.to(dtype=dtype, device=device).clamp(min=0.3, max=_SIGMA_PERP_MAX)
    foot_r = max(int(math.ceil(radius_sigmas * S)), half_w_perp + 1)

    active = (~is_border) & (values.abs() > eps)
    active_idx = active.nonzero(as_tuple=True)[0]
    A = active_idx.shape[0]
    if A == 0:
        return (torch.zeros(H, W, device=device, dtype=dtype),
                torch.zeros(H, W, device=device, dtype=dtype))

    val_a = values[active_idx]
    cx_a, cy_a = cx[active_idx], cy[active_idx]
    cos_a = torch.cos(theta[active_idx])
    sin_a = torch.sin(theta[active_idx])
    th_a = theta[active_idx]

    offsets = torch.arange(-foot_r, foot_r + 1, device=device, dtype=dtype)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    oy, ox = oy.reshape(-1), ox.reshape(-1)
    P = oy.shape[0]

    sum_wv = torch.zeros(H * W, device=device, dtype=dtype)
    sum_w = torch.zeros(H * W, device=device, dtype=dtype)
    max_rho_phi = torch.full((H * W,), -1.0, device=device, dtype=dtype)
    theta_star = torch.zeros(H * W, device=device, dtype=dtype)

    max_batch = max(1, 4_000_000 // P)
    for b0 in range(0, A, max_batch):
        b1 = min(b0 + max_batch, A)
        py_b = cy_a[b0:b1].unsqueeze(1) + oy.unsqueeze(0)
        px_b = cx_a[b0:b1].unsqueeze(1) + ox.unsqueeze(0)
        d_perp = (ox.unsqueeze(0) * cos_a[b0:b1].unsqueeze(1)
                  - oy.unsqueeze(0) * sin_a[b0:b1].unsqueeze(1))
        valid = ((py_b >= 0) & (py_b < H) &
                 (px_b >= 0) & (px_b < W) &
                 (d_perp.abs() <= (half_w_perp + 0.5)))
        phi = torch.exp(-d_perp * d_perp / (2.0 * sig * sig + eps)) * valid.to(dtype=dtype)
        flat_idx = (py_b.long() * W + px_b.long()).clamp(0, H * W - 1)
        wv = val_a[b0:b1].unsqueeze(1) * phi
        sum_wv.scatter_add_(0, flat_idx.reshape(-1), wv.reshape(-1))
        sum_w.scatter_add_(0, flat_idx.reshape(-1), phi.reshape(-1))

        rho_phi = val_a[b0:b1].unsqueeze(1) * phi
        th_expand = th_a[b0:b1].unsqueeze(1).expand_as(rho_phi)
        rp_flat = rho_phi.reshape(-1)
        th_flat = th_expand.reshape(-1)
        fi_flat = flat_idx.reshape(-1)
        update = rp_flat > max_rho_phi[fi_flat]
        update_idx = fi_flat[update]
        if update_idx.numel() > 0:
            max_rho_phi[update_idx] = rp_flat[update]
            theta_star[update_idx] = th_flat[update]

    rho_bar = (sum_wv / (sum_w + eps)).reshape(H, W)
    return rho_bar, theta_star.reshape(H, W)


# ═══════════════════════════════════════════════════════════════
# Splat a scalar cell field to pixel resolution
# ═══════════════════════════════════════════════════════════════

def _splat_cell_scalar(
    values: torch.Tensor,
    cx: torch.Tensor, cy: torch.Tensor,
    theta: torch.Tensor, is_border: torch.Tensor,
    sigma_perp: torch.Tensor,
    H: int, W: int, S: int,
    radius_sigmas: float = _SPLAT_RADIUS_SIGMAS,
    half_w_perp: int = _SPLAT_HALF_W_PERP,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Splat an arbitrary per-cell scalar to pixel grid (weighted average).

    Returns: (H, W) field.
    """
    device, dtype = values.device, values.dtype
    sig = sigma_perp.to(dtype=dtype, device=device).clamp(min=0.3, max=_SIGMA_PERP_MAX)
    foot_r = max(int(math.ceil(radius_sigmas * S)), half_w_perp + 1)

    active = (~is_border) & (values.abs() > eps)
    active_idx = active.nonzero(as_tuple=True)[0]
    A = active_idx.shape[0]
    if A == 0:
        return torch.zeros(H, W, device=device, dtype=dtype)

    val_a = values[active_idx]
    cx_a, cy_a = cx[active_idx], cy[active_idx]
    cos_a = torch.cos(theta[active_idx])
    sin_a = torch.sin(theta[active_idx])

    offsets = torch.arange(-foot_r, foot_r + 1, device=device, dtype=dtype)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    oy, ox = oy.reshape(-1), ox.reshape(-1)
    P = oy.shape[0]

    sum_wv = torch.zeros(H * W, device=device, dtype=dtype)
    sum_w = torch.zeros(H * W, device=device, dtype=dtype)

    max_batch = max(1, 4_000_000 // P)
    for b0 in range(0, A, max_batch):
        b1 = min(b0 + max_batch, A)
        py_b = cy_a[b0:b1].unsqueeze(1) + oy.unsqueeze(0)
        px_b = cx_a[b0:b1].unsqueeze(1) + ox.unsqueeze(0)
        d_perp = (ox.unsqueeze(0) * cos_a[b0:b1].unsqueeze(1)
                  - oy.unsqueeze(0) * sin_a[b0:b1].unsqueeze(1))
        valid = ((py_b >= 0) & (py_b < H) &
                 (px_b >= 0) & (px_b < W) &
                 (d_perp.abs() <= (half_w_perp + 0.5)))
        phi = torch.exp(-d_perp * d_perp / (2.0 * sig * sig + eps)) * valid.to(dtype=dtype)
        flat_idx = (py_b.long() * W + px_b.long()).clamp(0, H * W - 1)
        wv = val_a[b0:b1].unsqueeze(1) * phi
        sum_wv.scatter_add_(0, flat_idx.reshape(-1), wv.reshape(-1))
        sum_w.scatter_add_(0, flat_idx.reshape(-1), phi.reshape(-1))

    return (sum_wv / (sum_w + eps)).reshape(H, W)


# ═══════════════════════════════════════════════════════════════
# Bilinear sampling for stencil taps  (unchanged)
# ═══════════════════════════════════════════════════════════════

def _bilinear_sample_2d(
    field: torch.Tensor, row: torch.Tensor, col: torch.Tensor,
) -> torch.Tensor:
    H, W = field.shape
    r = row.reshape(1, 1, -1, 1)
    c = col.reshape(1, 1, -1, 1)
    gx = 2.0 * c / max(W - 1, 1) - 1.0
    gy = 2.0 * r / max(H - 1, 1) - 1.0
    grid = torch.cat([gx, gy], dim=-1)
    out = F.grid_sample(field.reshape(1, 1, H, W), grid,
                        mode="bilinear", padding_mode="border", align_corners=True)
    return out.reshape(row.shape)


def _sample_stencil_5(
    field: torch.Tensor,
    theta: torch.Tensor,
    spacing: torch.Tensor,
    direction: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """5-tap stencil along tangent or normal. Returns (H, W, 5)."""
    H, W = field.shape
    device, dtype = field.device, field.dtype
    rows = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
    cols = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)
    sp = spacing.to(dtype=dtype, device=device)
    if direction == "tangent":
        dr, dc = torch.cos(theta), torch.sin(theta)
    else:  # normal
        dr, dc = -torch.sin(theta), torch.cos(theta)
    taps = []
    for k in range(-2, 3):
        r_off = rows + k * sp * dr
        c_off = cols + k * sp * dc
        taps.append(_bilinear_sample_2d(field, r_off, c_off))
    return torch.stack(taps, dim=-1)


# ═══════════════════════════════════════════════════════════════
# Coherence diagnostic  (unchanged)
# ═══════════════════════════════════════════════════════════════

def _compute_coherence(
    rho_bar: torch.Tensor,
    theta_star: torch.Tensor,
    values: torch.Tensor,
    cx: torch.Tensor, cy: torch.Tensor,
    theta: torch.Tensor, is_border: torch.Tensor,
    sigma_perp: torch.Tensor,
    H: int, W: int, S: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """coh(p) = Σ_c ρ★_c cos²(θ_c − θ★_p) / (Σ_c ρ★_c + ε), within splat footprint."""
    device, dtype = values.device, values.dtype
    sig = sigma_perp.to(dtype=dtype, device=device).clamp(min=0.3, max=_SIGMA_PERP_MAX)
    foot_r = max(int(math.ceil(_SPLAT_RADIUS_SIGMAS * S)), _SPLAT_HALF_W_PERP + 1)

    active = (~is_border) & (values.abs() > eps)
    active_idx = active.nonzero(as_tuple=True)[0]
    A = active_idx.shape[0]
    if A == 0:
        return torch.zeros(H, W, device=device, dtype=dtype)

    val_a = values[active_idx]
    cx_a, cy_a = cx[active_idx], cy[active_idx]
    cos_a = torch.cos(theta[active_idx])
    sin_a = torch.sin(theta[active_idx])
    th_a = theta[active_idx]

    offsets = torch.arange(-foot_r, foot_r + 1, device=device, dtype=dtype)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    oy, ox = oy.reshape(-1), ox.reshape(-1)
    P = oy.shape[0]

    sum_rho_cos2 = torch.zeros(H * W, device=device, dtype=dtype)
    sum_rho = torch.zeros(H * W, device=device, dtype=dtype)

    max_batch = max(1, 4_000_000 // P)
    for b0 in range(0, A, max_batch):
        b1 = min(b0 + max_batch, A)
        py_b = cy_a[b0:b1].unsqueeze(1) + oy.unsqueeze(0)
        px_b = cx_a[b0:b1].unsqueeze(1) + ox.unsqueeze(0)
        d_perp = (ox.unsqueeze(0) * cos_a[b0:b1].unsqueeze(1)
                  - oy.unsqueeze(0) * sin_a[b0:b1].unsqueeze(1))
        valid = ((py_b >= 0) & (py_b < H) &
                 (px_b >= 0) & (px_b < W) &
                 (d_perp.abs() <= (_SPLAT_HALF_W_PERP + 0.5)))
        phi = torch.exp(-d_perp * d_perp / (2.0 * sig * sig + eps)) * valid.to(dtype=dtype)
        flat_idx = (py_b.long() * W + px_b.long()).clamp(0, H * W - 1)

        th_star_at_pix = theta_star.reshape(-1)[flat_idx]
        cos2_diff = torch.cos(th_a[b0:b1].unsqueeze(1) - th_star_at_pix).pow(2)

        rho_phi = val_a[b0:b1].unsqueeze(1) * phi
        sum_rho_cos2.scatter_add_(0, flat_idx.reshape(-1), (rho_phi * cos2_diff).reshape(-1))
        sum_rho.scatter_add_(0, flat_idx.reshape(-1), rho_phi.reshape(-1))

    return (sum_rho_cos2 / (sum_rho + eps)).reshape(H, W)


# ═══════════════════════════════════════════════════════════════
# Thinning head (16 → 12 → 1 MLP)
# ═══════════════════════════════════════════════════════════════

class ThinningHead(nn.Module):
    """16→12→1 MLP: σ(W₂ ReLU(W₁ F + b₁) + b₂).

    F = [ρ̄, s̄_photo, h2m_lum, h2m_chr, κ̄_col, coh, tang5(5), norm5(5)] ∈ R¹⁶.
    Initialised with structural priors.
    """

    def __init__(self, in_dim: int = 16, hidden: int = 12):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 1, bias=True)
        self._init_priors()

    def _init_priors(self) -> None:
        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.bias.zero_()
            self.fc2.weight.zero_()

            # Feature layout (in_dim=16):
            #   0: ρ̄
            #   1: s̄_photo
            #   2: h2m_lum
            #   3: h2m_chr
            #   4: κ̄_col      (collinear coherence)
            #   5: coh
            #   6-10: tang5
            #   11-15: norm5

            # Unit 0: Mexican-hat on norm5 (channels 11–15)
            mex_hat = torch.tensor([-0.25, -0.25, 1.0, -0.25, -0.25])
            self.fc1.weight[0, 11:16] = mex_hat

            # Unit 0: flat smoothing on tang5 (channels 6–10)
            self.fc1.weight[0, 6:11] = 0.2

            # Unit 1: s_photo × ρ̄ agreement
            self.fc1.weight[1, 0] = 0.5   # ρ̄
            self.fc1.weight[1, 1] = 0.5   # s̄_photo

            # Unit 2: h2m evidence
            self.fc1.weight[2, 2] = 0.5   # h2m_lum
            self.fc1.weight[2, 3] = 0.5   # h2m_chr

            # Unit 3: collinear coherence boosts confidence
            self.fc1.weight[3, 4] = 1.0   # κ̄_col

            # b₂ = 2 so σ(0 + 2) ≈ 0.88 — near-identity gate at t=0
            self.fc2.bias.fill_(2.0)
            self.fc2.weight[0, 1] = 0.3
            self.fc2.weight[0, 2] = 0.2
            self.fc2.weight[0, 3] = 0.3   # collinear unit gets positive weight

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, 16). Returns: (N,) gate in (0, 1)."""
        h = F.relu(self.fc1(features))
        return torch.sigmoid(self.fc2(h).squeeze(-1))


# ═══════════════════════════════════════════════════════════════
# ModulationRenderer
# ═══════════════════════════════════════════════════════════════

class ModulationRenderer(nn.Module):
    """Gaussian splat + collinear coherence + photometric thinning.

    Learned: σ⊥ (1), s_t (1), s_n (1), ThinningHead 16→12→1 (217).
    Fixed: collinear kernels (precomputed, not learned).
    Total: 220 learned parameters.
    """

    def __init__(
        self,
        hidden: int | None = None,
        col_radius: int = _COL_RADIUS,
        col_k_bins: int = _COL_K_BINS,
        col_sigma_d: float | None = _COL_SIGMA_D,
        col_sigma_t: float = _COL_SIGMA_T,
        **kwargs,
    ):
        super().__init__()
        _ = (hidden, kwargs)
        self._sigma_perp_raw = nn.Parameter(
            torch.tensor(_inv_softplus(_SIGMA_PERP_INIT), dtype=torch.float32)
        )
        self.s_t = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.s_n = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.thinning = ThinningHead(in_dim=16, hidden=12)

        # Collinear coherence config (kernels built on first use)
        self.col_radius = col_radius
        self.col_k_bins = col_k_bins
        self.col_sigma_d = col_sigma_d
        self.col_sigma_t = col_sigma_t

    @property
    def sigma_perp(self) -> torch.Tensor:
        return F.softplus(self._sigma_perp_raw)

    # Legacy compat
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
    def pixel_map(self, profile: torch.Tensor) -> torch.Tensor:
        return torch.ones(profile.shape[0], device=profile.device, dtype=profile.dtype)

    # Legacy compat for code that checks these
    @property
    def n_refine(self) -> int:
        return 0
    @property
    def refine_head(self):
        return None
    @property
    def alpha_refine(self):
        return None


def upgrade_renderer_state_dict(state_dict: dict, prefix: str = "") -> dict:
    """Upgrade old 12→8→1 state dicts to new 16→12→1."""
    remove = {
        f"{prefix}_sigma_par_raw",
        f"{prefix}_sigma_pre_raw",
        f"{prefix}_eta_h_raw",
        f"{prefix}_smooth_sigma_raw",
        f"{prefix}perp_conv.conv.weight",
        f"{prefix}perp_conv.conv.bias",
        f"{prefix}perp_conv.fc.weight",
        f"{prefix}perp_conv.fc.bias",
    }
    # Also strip old refine head keys if present
    remove_prefixes = (
        f"{prefix}refine_head.",
        f"{prefix}_alpha_refine_raw",
    )
    out = {}
    for k, v in state_dict.items():
        if k in remove:
            continue
        if any(k.startswith(rp) or k == rp for rp in remove_prefixes):
            continue
        # Handle old thinning head dimensions (12→8→1 → 16→12→1)
        if k == f"{prefix}thinning.fc1.weight" and v.shape == (8, 12):
            new_w = torch.zeros(12, 16, dtype=v.dtype)
            # old: [0=ρ̄, 1=coh, 2-6=tang5, 7-11=norm5]
            # new: [0=ρ̄, 1=s_photo, 2=h2m_lum, 3=h2m_chr, 4=κ_col, 5=coh, 6-10=tang5, 11-15=norm5]
            old_to_new = {0: 0, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 10: 14, 11: 15}
            for old_j, new_j in old_to_new.items():
                new_w[:8, new_j] = v[:, old_j]
            out[k] = new_w
        elif k == f"{prefix}thinning.fc1.bias" and v.shape == (8,):
            new_b = torch.zeros(12, dtype=v.dtype)
            new_b[:8] = v
            out[k] = new_b
        elif k == f"{prefix}thinning.fc2.weight" and v.shape == (1, 8):
            new_w = torch.zeros(1, 12, dtype=v.dtype)
            new_w[0, :8] = v[0]
            out[k] = new_w
        else:
            out[k] = v
    return out


# ═══════════════════════════════════════════════════════════════
# Proj / feature helpers
# ═══════════════════════════════════════════════════════════════

def compute_render_features(
    z2_image: np.ndarray, img: np.ndarray,
    cells: dict, border_mask: np.ndarray,
    eps: float = 1e-9, **kwargs,
) -> dict:
    _ = (z2_image, img, cells, border_mask, eps, kwargs)
    H, W = z2_image.shape
    nH, nW = cells["nH"], cells["nW"]
    return {"H": H, "W": W, "n_cells": nH * nW, "nH": nH, "nW": nW}


def proj_to_device(proj: dict, device: torch.device) -> dict:
    return {"H": proj["H"], "W": proj["W"], "n_cells": proj["n_cells"],
            "nH": proj["nH"], "nW": proj["nW"]}


def _theta_on_branch(theta, branch_pick, n_cells, device):
    if branch_pick is not None:
        b = branch_pick.to(device=device, dtype=torch.long).view(-1)
        idx_n = torch.arange(n_cells, device=device, dtype=torch.long)
        return theta[idx_n, b]
    return theta[:, 0]


def _s_photo_on_branch(s_photo, branch_pick, n_cells, device):
    """Select s_photo for chosen branch."""
    if branch_pick is not None:
        b = branch_pick.to(device=device, dtype=torch.long).view(-1)
        idx_n = torch.arange(n_cells, device=device, dtype=torch.long)
        return s_photo[idx_n, b]
    return s_photo[:, 0]


# ═══════════════════════════════════════════════════════════════
# render_boundary_map_torch
# ═══════════════════════════════════════════════════════════════

def render_boundary_map_torch(
    rho_cell: torch.Tensor,
    proj_dev: dict,
    renderer: ModulationRenderer,
    cells_flat: dict,
    Hp: int, Wp: int,
    l0_pix: dict[str, torch.Tensor] | None = None,
    eps: float = 1e-6,
    training: bool = False,
    branch_pick: torch.Tensor | None = None,
    content_h: int | None = None,
    content_w: int | None = None,
    return_dominant_theta: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    _ = (training, )
    H, W = proj_dev["H"], proj_dev["W"]
    device, dtype = rho_cell.device, rho_cell.dtype

    if "cx_z2" not in cells_flat or "cy_z2" not in cells_flat:
        raise ValueError("cells_flat must include cx_z2 and cy_z2")

    nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
    n_cells = int(proj_dev["n_cells"])
    theta_all = cells_flat["theta"].to(device=device, dtype=dtype)
    theta_c = _theta_on_branch(theta_all, branch_pick, n_cells, device)
    S = int(cells_flat.get("S", max(1, W // max(nW, 1))))

    # ── Step 1: Cell-grid operations ─────────────────────────
    rho_grid = rho_cell.reshape(nH, nW).to(dtype=dtype)
    ib_grid = cells_flat["is_border"].to(device=device).reshape(nH, nW).bool()
    theta_c = _smooth_theta_rho_double_angle(
        theta_c.reshape(nH, nW), rho_grid, ib_grid, eps=eps,
    ).reshape(-1)

    cx = cells_flat["cx_z2"].to(device=device, dtype=dtype)
    cy = cells_flat["cy_z2"].to(device=device, dtype=dtype)
    rho_flat = rho_cell.reshape(-1).to(dtype=dtype)
    is_border = cells_flat["is_border"].to(device=device).reshape(-1).bool()

    _cx, _cy = _smooth_anchors_rho_gated(
        cx, cy, theta_c, rho_flat, is_border, nH, nW, eps=eps,
    )

    active = (rho_flat > 0) & (~is_border)
    if not active.any().item():
        z = (rho_cell.sum() * 0.0).to(dtype=dtype, device=device)
        out = z.expand(H, W)[:Hp, :Wp]
        if return_dominant_theta:
            return out, torch.zeros_like(out)
        return out

    # ── Step 2: Collinear coherence on cell grid (NEW) ───────
    #   Fixed geometric computation — detach to avoid gradient
    #   flow through the conv kernels back into seed params.
    kappa_col_grid = compute_collinear_coherence(
        theta_c.reshape(nH, nW).detach(),
        rho_grid.detach(),
        ib_grid,
        R=renderer.col_radius,
        K=renderer.col_k_bins,
        sigma_d=renderer.col_sigma_d,
        sigma_t=renderer.col_sigma_t,
        eps=eps,
    )
    kappa_col_flat = kappa_col_grid.reshape(-1)

    cx_det, cy_det = _cx.detach(), _cy.detach()
    theta_det = theta_c.detach()
    rho_splat = torch.where(is_border, torch.zeros_like(rho_flat), rho_flat)

    # ── Step 3: Gaussian-line splat → ρ̄(p), θ★(p) ───────────
    rho_bar, theta_star = _gaussian_line_splat(
        rho_splat, cx_det, cy_det, theta_det, is_border,
        sigma_perp=renderer.sigma_perp,
        H=H, W=W, S=S, eps=eps,
    )

    # ── Step 4: Splat photometric + collinear fields to pixels ─
    has_s_photo = "s_photo" in cells_flat
    if has_s_photo:
        s_photo_all = cells_flat["s_photo"].to(device=device, dtype=dtype)
        s_photo_c = _s_photo_on_branch(s_photo_all, branch_pick, n_cells, device)
        s_photo_splat = torch.where(is_border, torch.zeros_like(s_photo_c), s_photo_c)
        s_photo_pix = _splat_cell_scalar(
            s_photo_splat, cx_det, cy_det, theta_det, is_border,
            sigma_perp=renderer.sigma_perp,
            H=H, W=W, S=S, eps=eps,
        )
    else:
        s_photo_pix = torch.zeros(H, W, device=device, dtype=dtype)

    # Splat κ_col to pixel resolution
    kappa_col_splat = torch.where(is_border, torch.zeros_like(kappa_col_flat), kappa_col_flat)
    kappa_col_pix = _splat_cell_scalar(
        kappa_col_splat, cx_det, cy_det, theta_det, is_border,
        sigma_perp=renderer.sigma_perp,
        H=H, W=W, S=S, eps=eps,
    )

    # Pixel-level h2m from L0 (already at pixel resolution)
    if l0_pix is not None and "h2m_lum" in l0_pix:
        h2m_lum_pix = l0_pix["h2m_lum"].to(device=device, dtype=dtype)
        if h2m_lum_pix.shape != (H, W):
            h2m_lum_pix = h2m_lum_pix[:H, :W]
    else:
        h2m_lum_pix = torch.zeros(H, W, device=device, dtype=dtype)

    if l0_pix is not None and "h2m_chr" in l0_pix:
        h2m_chr_pix = l0_pix["h2m_chr"].to(device=device, dtype=dtype)
        if h2m_chr_pix.shape != (H, W):
            h2m_chr_pix = h2m_chr_pix[:H, :W]
    else:
        h2m_chr_pix = torch.zeros(H, W, device=device, dtype=dtype)

    # ── Step 5: Coherence ────────────────────────────────────
    coh = _compute_coherence(
        rho_bar, theta_star,
        rho_splat, cx_det, cy_det, theta_det, is_border,
        renderer.sigma_perp, H, W, S, eps=eps,
    )

    # ── Step 6: Feature vector F_p ∈ R¹⁶ ────────────────────
    tang5 = _sample_stencil_5(rho_bar, theta_star, renderer.s_t, "tangent", eps)
    norm5 = _sample_stencil_5(rho_bar, theta_star, renderer.s_n, "normal", eps)

    # F_p = [ρ̄, s̄_photo, h2m_lum, h2m_chr, κ̄_col, coh, tang5(5), norm5(5)] ∈ R¹⁶
    features = torch.cat([
        rho_bar.unsqueeze(-1),          # 0
        s_photo_pix.unsqueeze(-1),      # 1
        h2m_lum_pix.unsqueeze(-1),      # 2
        h2m_chr_pix.unsqueeze(-1),      # 3
        kappa_col_pix.unsqueeze(-1),    # 4  collinear coherence
        coh.unsqueeze(-1),              # 5
        tang5,                          # 6-10
        norm5,                          # 11-15
    ], dim=-1)  # (H, W, 16)

    # ── Step 7: Thinning head → B̂(p) = ρ̄(p) · gate(p) ──────
    feat_flat = features.reshape(H * W, 16)
    gate = renderer.thinning(feat_flat).reshape(H, W)
    bmap = rho_bar * gate

    # ── Crop ─────────────────────────────────────────────────
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
# NumPy wrapper
# ═══════════════════════════════════════════════════════════════

def render_boundary_map(
    rho_cell: np.ndarray, proj: dict,
    renderer: ModulationRenderer, cells_flat: dict,
    l0_pix: dict[str, np.ndarray] | None = None,
    device: torch.device = torch.device("cpu"),
    eps: float = 1e-6,
    branch_pick: np.ndarray | None = None,
    content_h: int | None = None, content_w: int | None = None,
) -> np.ndarray:
    proj_dev = proj_to_device(proj, device)
    rho_t = torch.from_numpy(np.asarray(rho_cell, dtype=np.float32)).to(device)
    cf_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in cells_flat.items()}
    bp = None
    if branch_pick is not None:
        bp = torch.from_numpy(np.asarray(branch_pick, dtype=np.int64).ravel()).to(device)
    l0_dev = None
    if l0_pix is not None:
        l0_dev = {}
        for k, v in l0_pix.items():
            if isinstance(v, np.ndarray):
                l0_dev[k] = torch.from_numpy(v.astype(np.float32)).to(device)
            elif isinstance(v, torch.Tensor):
                l0_dev[k] = v.to(device)
            else:
                l0_dev[k] = v
    with torch.no_grad():
        bmap_t = render_boundary_map_torch(
            rho_t, proj_dev, renderer, cf_dev, proj["H"], proj["W"], l0_dev,
            eps=eps, training=False, branch_pick=bp,
            content_h=content_h, content_w=content_w,
        )
    return bmap_t.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# NMS  (unchanged)
# ═══════════════════════════════════════════════════════════════

def _nms_unit_normal_from_theta(theta, eps=1e-8):
    t = np.asarray(theta, dtype=np.float64)
    return np.cos(t).astype(np.float32), (-np.sin(t)).astype(np.float32)

def _nms_unit_normal_from_gradient(mag, eps=1e-8):
    m = np.asarray(mag, dtype=np.float64)
    gx, gy = ndimage.sobel(m, axis=1), ndimage.sobel(m, axis=0)
    norm = np.sqrt(gx*gx + gy*gy) + eps
    return (gx/norm).astype(np.float32), (gy/norm).astype(np.float32)

def _nms_bilinear_sample(mag, row_off, col_off):
    coords = np.stack([row_off.astype(np.float64), col_off.astype(np.float64)])
    return ndimage.map_coordinates(mag.astype(np.float64), coords, order=1, mode="nearest").astype(np.float32)

def ridge_nms(mag, *, theta=None, grad_norm_floor=1e-7):
    m = np.asarray(mag, dtype=np.float32)
    if m.ndim != 2: raise ValueError(f"ridge_nms expects 2D, got {m.shape}")
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m_work = m.copy()
    if theta is not None:
        nx, ny = _nms_unit_normal_from_theta(np.asarray(theta, dtype=np.float32))
        weak = np.zeros((H, W), dtype=bool)
    else:
        gx, gy = ndimage.sobel(m_work.astype(np.float64), axis=1), ndimage.sobel(m_work.astype(np.float64), axis=0)
        gnorm = np.sqrt(gx*gx + gy*gy).astype(np.float32)
        nx, ny = _nms_unit_normal_from_gradient(m_work)
        weak = gnorm < grad_norm_floor
    ahead = _nms_bilinear_sample(m_work, yy+ny, xx+nx)
    behind = _nms_bilinear_sample(m_work, yy-ny, xx-nx)
    keep = ((m_work >= ahead) & (m_work >= behind)) | weak
    return np.where(keep, m_work, 0.0).astype(np.float32)

def ridge_nms_binary(mag, threshold, *, theta=None, grad_norm_floor=1e-7):
    return (ridge_nms(mag, theta=theta, grad_norm_floor=grad_norm_floor) >= threshold).astype(np.uint8) * 255

def cell_rho_to_2branch(rho_cell, branch):
    out = np.zeros((*rho_cell.shape, 2), dtype=rho_cell.dtype)
    ii, jj = np.indices(rho_cell.shape)
    out[ii, jj, branch.astype(np.int64)] = rho_cell
    return out

HarmonicThinRenderer = ModulationRenderer
StampRenderer = ModulationRenderer
AnisoDiffusionRenderer = ModulationRenderer