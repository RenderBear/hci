r"""L1 — hypercolumn oriented-energy construction + GABA-budget recurrence.

Replaces per-patch eigendecomposition with direct K-bin oriented energy
projection at each cell position.  Each cell becomes a hypercolumn:
K orientation-tuned units pooling over the same receptive field (patch).

Pipeline:
  1. Extract patches from pixel-level h2m and θ_h fields (from L0).
  2. Project each patch's oriented energy onto K bins via cos² tuning → raw ``μ_k``.
  3. **Seed NR:** ``ρ^{\\mathrm{seed}}_k = μ_k^2/(μ_k^2+η_z^2+ε)`` with learned ``η_z=\\mathrm{softplus}(\\tilde\\eta_z)``.
     Detach ``ρ^{\\mathrm{seed}}`` as the multiplicative **envelope** for passes.
  4. ``T`` passes on current ``ρ^{(t)}``: collinear ``\\tilde{S}_k``, LOO surround ``I_k``,
     ``u_k = β_{\\mathrm{coll}} \\tilde{S}_k - β_{\\mathrm{cross}} I_k`` (signed),
     ``g_k = 1 + α \\tanh(u_k/τ)`` with learned ``α\\in(0,0.49]``, ``τ>0``,
     ``ρ^{(t+1)}_k = ρ^{\\mathrm{seed}}_k \\cdot g_k`` (no in-loop divisive NR).
  5. Extract dominant ``ρ``, ``θ``, and diagnostic ``κ`` for the renderer / readout.

``rho_k_initial`` stores **post–seed-NR** ``ρ^{\\mathrm{seed}}``.  ``kappa_pass0_cell`` /
``kappa_col_cell`` store cosine ``κ`` after the first / last pass (``ρ`` vs ``S``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import L1, SEED


# ═══════════════════════════════════════════════════════════════
# Patch utilities (reused from old L1)
# ═══════════════════════════════════════════════════════════════

def stride_from_patch_overlap(P: int, patch_overlap: int) -> int:
    o = int(patch_overlap)
    p = int(P)
    if o < 0 or o >= p:
        raise ValueError(f"overlap must be in [0, {p}), got {o}")
    return p - o


def pad_for_patch_grid(
    img: np.ndarray,
    patch_size: int,
    patch_overlap: int,
) -> tuple[np.ndarray, int, int]:
    H0, W0 = img.shape[:2]
    S = stride_from_patch_overlap(patch_size, patch_overlap)

    def pd(d0):
        if d0 <= patch_size:
            return int(patch_size)
        n = (d0 - patch_size + S - 1) // S + 1
        return (n - 1) * S + patch_size

    Hp, Wp = pd(H0), pd(W0)
    if Hp == H0 and Wp == W0:
        return img, H0, W0
    ndim = img.ndim
    if ndim == 3:
        out = np.pad(img, ((0, Hp - H0), (0, Wp - W0), (0, 0)), mode="reflect")
    else:
        out = np.pad(img, ((0, Hp - H0), (0, Wp - W0)), mode="reflect")
    return out, H0, W0


def _extract_patches_2d(
    field: torch.Tensor,
    nH: int, nW: int,
    P: int, S: int,
) -> torch.Tensor:
    """Extract (nH*nW, P*P) patches from a (H, W) field."""
    # Use unfold for efficient patch extraction
    patches = field.unfold(0, P, S).unfold(1, P, S)  # (nH, nW, P, P)
    return patches.contiguous().reshape(nH * nW, P * P)


# ═══════════════════════════════════════════════════════════════
# Legacy compatibility: z_from_l0_harmonics
# ═══════════════════════════════════════════════════════════════

def z_from_l0_harmonics(
    s: torch.Tensor,
    border_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z1 = torch.complex(s[..., 0], s[..., 1])
    z2 = torch.complex(s[..., 2], s[..., 3])
    z1[border_mask] = 0.0
    z2[border_mask] = 0.0
    return z1, z2


# ═══════════════════════════════════════════════════════════════
# Collinear kernels (same as renderer, factored out)
# ═══════════════════════════════════════════════════════════════

def _build_collinear_kernels(
    R: int, K: int,
    sigma_d: torch.Tensor,
    sigma_t: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute K depthwise kernels of shape (K, 1, 2R+1, 2R+1).

    ``sigma_d`` and ``sigma_t`` are 0-dim tensors (differentiable w.r.t. learned
    α in ``HypercolumnSeed``).
    """
    sigma_d = sigma_d.to(device=device, dtype=dtype).clamp_min(torch.tensor(1e-4, device=device, dtype=dtype))
    sigma_t = sigma_t.to(device=device, dtype=dtype).clamp_min(torch.tensor(1e-4, device=device, dtype=dtype))
    offsets = torch.arange(-R, R + 1, device=device, dtype=dtype)
    di, dj = torch.meshgrid(offsets, offsets, indexing="ij")
    dist_sq = di * di + dj * dj
    # Out-of-place masks only: in-place slices on ``w_d`` break ExpBackward when
    # ``sigma_d`` / ``sigma_t`` are learned (live L1 training).
    disc = (dist_sq <= float(R * R)).to(dtype=dtype)
    omit_center = (dist_sq > 0).to(dtype=dtype)
    w_d = torch.exp(-dist_sq / (2.0 * sigma_d * sigma_d)) * disc * omit_center

    kernels = torch.zeros(K, 2 * R + 1, 2 * R + 1, device=device, dtype=dtype)
    for k in range(K):
        theta_k = k * math.pi / K
        d_perp = dj * math.cos(theta_k) - di * math.sin(theta_k)
        w_t = torch.exp(-d_perp * d_perp / (2.0 * sigma_t * sigma_t))
        kernels[k] = w_d * w_t
    return kernels.unsqueeze(1)


def _build_isotropic_surround_kernel(
    R: int,
    sigma_d: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Single-channel kernel ``(1,1,2R+1,2R+1)``: radial Gaussian × disk, center omitted.

    Matches the **radial** factor ``w_d`` of ``_build_collinear_kernels`` (no tangential
    selectivity). Applied **per bin** to the leave-one-out mean map
    ``(1/\\max(K-1,1))\\sum_{j\\neq k} ρ_j`` so inhibition excludes bin ``k`` from its own surround.
    """
    sigma_d = sigma_d.to(device=device, dtype=dtype).clamp_min(torch.tensor(1e-4, device=device, dtype=dtype))
    offsets = torch.arange(-R, R + 1, device=device, dtype=dtype)
    di, dj = torch.meshgrid(offsets, offsets, indexing="ij")
    dist_sq = di * di + dj * dj
    disc = (dist_sq <= float(R * R)).to(dtype=dtype)
    omit_center = (dist_sq > 0).to(dtype=dtype)
    w = torch.exp(-dist_sq / (2.0 * sigma_d * sigma_d)) * disc * omit_center
    return w.unsqueeze(0).unsqueeze(0)


def _build_tile_grid(nH, nW, R, stride, dev):
    ti = torch.arange(R, nH - R, stride, device=dev)
    tj = torch.arange(R, nW - R, stride, device=dev)
    if ti.numel() == 0:
        ti = torch.tensor([min(R, nH - 1)], device=dev)
    if tj.numel() == 0:
        tj = torch.tensor([min(R, nW - 1)], device=dev)
    ti_g, tj_g = torch.meshgrid(ti, tj, indexing="ij")
    return ti_g.reshape(-1), tj_g.reshape(-1)


def _build_tile_membership(ti, tj, nW, R, dev):
    offsets = torch.arange(-R, R + 1, device=dev)
    di, dj = torch.meshgrid(offsets, offsets, indexing="ij")
    mi = ti.unsqueeze(1) + di.reshape(-1).unsqueeze(0)
    mj = tj.unsqueeze(1) + dj.reshape(-1).unsqueeze(0)
    return (mi * nW + mj).to(torch.int64)


def tile_interior_flat(
    nH: int,
    nW: int,
    is_border: torch.Tensor,
    r_pool: int,
    stride: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-cell 0/1 mask (tile coverage ∧ ¬border), shape (N,) flat."""
    N = nH * nW
    ib = is_border.reshape(-1).bool()
    ti, tj = _build_tile_grid(nH, nW, int(r_pool), int(stride), device)
    mi = _build_tile_membership(ti, tj, nW, int(r_pool), device)
    tile_cov = torch.zeros(N, dtype=torch.bool, device=device)
    tile_cov[mi.reshape(-1)] = True
    return (~ib & tile_cov).to(torch.float32)


def _e_col_dominant_bin(
    rho_k: torch.Tensor,
    dominant_bin: torch.Tensor,
    nH: int,
    nW: int,
    K: int,
    R: int,
    sigma_d: torch.Tensor,
    sigma_t: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """One depthwise conv on final ρ; return S at dominant bin per cell (N,)."""
    device, dtype = rho_k.device, rho_k.dtype
    kernels = _build_collinear_kernels(R, K, sigma_d, sigma_t, device, dtype)
    x = rho_k.reshape(nH, nW, K).permute(2, 0, 1).unsqueeze(0)
    s_k = F.conv2d(x, kernels, padding=R, groups=K)
    s_nk = s_k.squeeze(0).permute(1, 2, 0).reshape(-1, K)
    return s_nk.gather(1, dominant_bin.unsqueeze(-1)).squeeze(-1).clamp_min(eps)


# ═══════════════════════════════════════════════════════════════
# Hypercolumn construction
# ═══════════════════════════════════════════════════════════════

def build_hypercolumns(
    h2m: torch.Tensor,
    theta_h: torch.Tensor,
    border_mask: torch.Tensor,
    P: int,
    patch_overlap: int,
    border_patch_max_frac: float,
    K: int,
    eps: float = 1e-15,
) -> dict:
    """Build K-bin oriented energy hypercolumns from pixel-level L0 output.

    Args:
        h2m: (H, W) second-harmonic magnitude from L0
        theta_h: (H, W) pixel-level orientation = 0.5 * atan2(z2_im, z2_re)
        border_mask: (H, W) bool
        P: patch size
        patch_overlap: patch overlap
        border_patch_max_frac: fraction of border pixels to mark cell as border
        K: number of orientation bins

    Returns:
        dict with:
            nH, nW: cell grid dimensions
            P, S: patch size and stride
            rho_k: (nH*nW, K) per-bin oriented energy (unnormalized)
            z0: (nH*nW,) total energy per cell
            is_border: (nH*nW,) bool
            cx, cy: (nH*nW,) cell centre pixel coords
            bin_centers: (K,) angle of each bin centre
    """
    H, W = h2m.shape
    device = h2m.device
    dtype = h2m.dtype
    S = stride_from_patch_overlap(P, patch_overlap)
    nH = (H - P) // S + 1 if H >= P else 0
    nW = (W - P) // S + 1 if W >= P else 0
    N = nH * nW

    # Extract patches
    h2m_patches = _extract_patches_2d(h2m, nH, nW, P, S)            # (N, P²)
    theta_patches = _extract_patches_2d(theta_h, nH, nW, P, S)      # (N, P²)
    bm_patches = _extract_patches_2d(border_mask.float(), nH, nW, P, S)
    is_border = bm_patches.mean(dim=-1) > border_patch_max_frac      # (N,)

    # Bin centres
    bin_centers = torch.arange(K, device=device, dtype=dtype) * (math.pi / K)

    # Project onto K bins: cos²(θ_pixel - θ_bin) weighting
    # theta_patches: (N, P²), bin_centers: (K,)
    # diff: (N, P², K)
    diff = theta_patches.unsqueeze(-1) - bin_centers.unsqueeze(0).unsqueeze(0)
    cos2_weight = torch.cos(diff).pow(2)  # (N, P², K)

    # Weighted sum: (N, K)
    rho_k = (h2m_patches.unsqueeze(-1) * cos2_weight).sum(dim=1)

    # Zero border cells
    rho_k[is_border] = 0.0

    # Total energy per cell
    z0 = rho_k.sum(dim=-1)  # (N,)

    # Cell centre pixel coordinates
    ci = torch.arange(nH, device=device, dtype=dtype) * S + P / 2.0
    cj = torch.arange(nW, device=device, dtype=dtype) * S + P / 2.0
    cy = ci.unsqueeze(1).expand(nH, nW).reshape(-1)
    cx = cj.unsqueeze(0).expand(nH, nW).reshape(-1)

    return {
        "nH": nH,
        "nW": nW,
        "N": N,
        "P": P,
        "S": S,
        "rho_k": rho_k,
        "z0": z0,
        "is_border": is_border,
        "cx": cx,
        "cy": cy,
        "bin_centers": bin_centers,
        "K": K,
    }


# ═══════════════════════════════════════════════════════════════
# GABA-budget recurrence on K-channel representation
# ═══════════════════════════════════════════════════════════════


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


def nr_squash_k_bins(u: torch.Tensor, eta: torch.Tensor, nr_eps: float) -> torch.Tensor:
    """Divisive NR ``u²/(u²+η²+ε)``; ``η`` scalar or ``(nH,nW)`` broadcast to all K."""
    sq = u * u
    ep = torch.as_tensor(nr_eps, device=u.device, dtype=u.dtype)
    if eta.dim() == 0:
        eta_sq = eta * eta
    else:
        eta_sq = (eta * eta).view(1, 1, eta.shape[-2], eta.shape[-1])
    return sq / (sq + eta_sq + ep)


def _cosine_kappa_grid(
    rho_1knw: torch.Tensor,
    s_1knw: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Per-cell cosine ``κ = (ρ·S)/(‖ρ‖ ‖S‖ + ε)``; tensors ``(1,K,nH,nW)``."""
    rf = rho_1knw.squeeze(0)
    sf = s_1knw.squeeze(0)
    dot = (rf * sf).sum(dim=0)
    nr = rf.norm(dim=0)
    ns = sf.norm(dim=0)
    ep = torch.as_tensor(eps, device=dot.device, dtype=dot.dtype)
    return (dot / (nr * ns + ep)).clamp(0.0, 1.0)


def _kappa_k_from_grid(kappa_nw: torch.Tensor, K: int) -> torch.Tensor:
    """Broadcast per-cell κ to ``(N, K)`` for ``extract_dominant`` gather API."""
    nH, nW = kappa_nw.shape
    return kappa_nw.reshape(-1, 1).expand(-1, int(K)).contiguous()


def gaba_recurrence(
    rho_k_raw: torch.Tensor,
    nH: int, nW: int,
    is_border: torch.Tensor,
    K: int,
    R: int,
    sigma_d: torch.Tensor,
    sigma_t: torch.Tensor,
    n_passes: int,
    beta_coll: torch.Tensor,
    beta_cross: torch.Tensor,
    eta_z: torch.Tensor,
    tau: torch.Tensor,
    gain_alpha: torch.Tensor,
    nr_eps: float,
    eps: float = 1e-6,
    eta_update_fn: Callable | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Seed divisive NR, then multiplicative **tanh gate** passes (no in-loop NR).

    **Seed:** ``ρ^{\\mathrm{seed}} = \\mathrm{NR}(μ, η_z)`` on raw hypercolumns; detach for the
    envelope ``ρ^{\\mathrm{seed}}`` in ``ρ ← ρ^{\\mathrm{seed}} \\cdot (1 + α \\tanh(u/τ))``.

    **Passes:** ``\\tilde{S}_k = G_k * ρ_k``, LOO ``I_k``, ``u_k = β_c \\tilde{S}_k - β_x I_k``,
    ``g_k = 1 + α\\tanh(u_k/τ)`` (``α`` clamped ``\\le 0.49`` so ``g\\ge 1-α > 0``),
    ``ρ^{(t+1)} = ρ^{\\mathrm{seed}} \\odot g``.  ``κ`` from ``(ρ,S)`` each pass.

    Returns:
        rho_k_out, kappa_k, kappa_k_pass0, rho_k_seed_snap, s_k_first, s_bar_first.
        ``rho_k_seed_snap`` is post–seed-NR ``ρ^{\\mathrm{seed}}`` ``(N,K)`` (``rho_k_initial``).
    """
    device = rho_k_raw.device
    dtype = rho_k_raw.dtype
    kernels = _build_collinear_kernels(R, K, sigma_d, sigma_t, device, dtype)
    iso_kernel = _build_isotropic_surround_kernel(R, sigma_d, device, dtype)

    ib_grid = is_border.reshape(nH, nW)
    border_mask_4d = ib_grid.unsqueeze(0).unsqueeze(0)  # (1,1,nH,nW)

    rho_grid = rho_k_raw.reshape(nH, nW, K).permute(2, 0, 1).unsqueeze(0)
    rho_grid = torch.where(
        border_mask_4d.expand_as(rho_grid),
        torch.zeros_like(rho_grid), rho_grid,
    )

    eta_z_t = torch.as_tensor(eta_z, device=device, dtype=dtype)
    rho_grid = nr_squash_k_bins(rho_grid, eta_z_t, nr_eps)
    rho_grid = torch.where(
        border_mask_4d.expand_as(rho_grid),
        torch.zeros_like(rho_grid), rho_grid,
    )

    rho_seed = rho_grid.detach()
    rho_k_seed_snap = (
        rho_grid.squeeze(0).permute(1, 2, 0).reshape(-1, K).detach().clone()
    )

    kappa_k_pass0 = torch.zeros_like(rho_k_raw)
    kappa_last_nw = torch.zeros(nH, nW, device=device, dtype=dtype)
    s_k_first: torch.Tensor | None = None
    s_bar_first: torch.Tensor | None = None

    tau_t = torch.as_tensor(tau, device=device, dtype=dtype).view(()).clamp_min(1e-6)
    alpha_t = torch.as_tensor(gain_alpha, device=device, dtype=dtype).view(()).clamp(max=0.49)

    for t in range(n_passes):
        if eta_update_fn is not None:
            rho_flat = rho_grid.squeeze(0).permute(1, 2, 0).reshape(-1, K)
            kappa_flat = torch.zeros_like(rho_flat)
            rho_k_new = eta_update_fn(
                rho_flat, kappa_flat, is_border, nH, nW, K, t,
            )
            rho_grid = rho_k_new.reshape(nH, nW, K).permute(2, 0, 1).unsqueeze(0)
            rho_grid = torch.where(
                border_mask_4d.expand_as(rho_grid),
                torch.zeros_like(rho_grid), rho_grid,
            )

        s_k = F.conv2d(rho_grid, kernels, padding=R, groups=K)
        kap_t = _cosine_kappa_grid(rho_grid, s_k, eps)
        k_i = int(K)
        den_other = float(max(k_i - 1, 1))
        rho_total = rho_grid.sum(dim=1, keepdim=True)
        rho_other_mean = (rho_total - rho_grid) / den_other
        w_iso = iso_kernel.expand(k_i, 1, iso_kernel.shape[2], iso_kernel.shape[3])
        s_surround = F.conv2d(rho_other_mean, w_iso, padding=R, groups=k_i)
        s_inhib = s_surround
        if t == 0:
            s_k_first = s_k.detach().clone()
            s_bar_first = s_inhib.detach().clone()

        u = beta_coll * s_k - beta_cross * s_inhib
        u = torch.where(border_mask_4d.expand_as(u), torch.zeros_like(u), u)

        gain = 1.0 + alpha_t * torch.tanh(u / tau_t)
        rho_grid = rho_seed * gain
        rho_grid = torch.where(
            border_mask_4d.expand_as(rho_grid),
            torch.zeros_like(rho_grid), rho_grid,
        )

        kappa_last_nw = kap_t
        if t == 0:
            kappa_k_pass0 = _kappa_k_from_grid(kap_t, K)

    rho_k_out = rho_grid.squeeze(0).permute(1, 2, 0).reshape(-1, K)
    kappa_k_out = _kappa_k_from_grid(kappa_last_nw, K)

    if s_k_first is None:
        zshape = (1, int(K), nH, nW)
        s_k_first = torch.zeros(zshape, device=device, dtype=dtype)
        s_bar_first = torch.zeros_like(s_k_first)

    return rho_k_out, kappa_k_out, kappa_k_pass0, rho_k_seed_snap, s_k_first, s_bar_first


# ═══════════════════════════════════════════════════════════════
# Extract dominant orientation + scalar ρ for renderer interface
# ═══════════════════════════════════════════════════════════════

def extract_dominant(
    rho_k: torch.Tensor,
    kappa_k: torch.Tensor,
    rho_k_initial: torch.Tensor,
    bin_centers: torch.Tensor,
    is_border: torch.Tensor,
    K: int,
    nH: int,
    nW: int,
    R: int,
    sigma_d: torch.Tensor,
    sigma_t: torch.Tensor,
    eps: float = 1e-15,
) -> dict:
    """Extract per-cell scalar ρ, θ, κ from K-bin representation.

    ``kappa_k`` holds per-cell cosine ``κ`` broadcast to bins; ``kappa`` is the value at
    the dominant orientation bin (identical across ``k`` here).
    """
    N = rho_k.shape[0]
    device = rho_k.device
    dtype = rho_k.dtype

    dominant_bin = rho_k.argmax(dim=-1)
    idx = dominant_bin.unsqueeze(-1)
    rho = rho_k.gather(1, idx).squeeze(-1)
    kappa = kappa_k.gather(1, idx).squeeze(-1)
    rho_init = rho_k_initial.gather(1, idx).squeeze(-1)

    theta_dominant = bin_centers[dominant_bin]
    theta = torch.stack([theta_dominant, theta_dominant], dim=-1)

    e_col = _e_col_dominant_bin(
        rho_k, dominant_bin, nH, nW, K, R, sigma_d, sigma_t, eps,
    )

    rho = torch.where(is_border, torch.zeros_like(rho), rho)
    kappa = torch.where(is_border, torch.zeros_like(kappa), kappa)
    e_col = torch.where(is_border, torch.zeros_like(e_col), e_col)

    rho_max = rho_k.max(dim=-1).values
    rho_max = torch.where(is_border, torch.zeros_like(rho_max), rho_max)

    return {
        "rho": rho,
        "theta": theta,
        "kappa_col": kappa,
        "rho_max": rho_max,
        "rho_initial": rho_init,
        "e_col": e_col,
        "dominant_bin": dominant_bin,
    }


# ═══════════════════════════════════════════════════════════════
# Seed module: learned η_z + tanh gate τ, α + raw β_coll/β_cross (HypercolumnSeed)
# ═══════════════════════════════════════════════════════════════


class HypercolumnSeed(nn.Module):
    """Learned ``η_z`` (seed NR), ``τ``, ``α`` (tanh gain), ``β_coll``, ``β_cross``, σ_d/σ_t."""

    def __init__(
        self,
        r_pool: int = SEED.R_POOL,
        stride: int = SEED.STRIDE,
        eps: float = SEED.EPS,
        n_gaba_passes: int | None = None,
        eta_z_init: float | None = None,
        eta_pass_init: float | None = None,
        tau_init: float | None = None,
        gain_alpha_init: float | None = None,
        beta_coll_init: float = SEED.BETA_COLL_INIT,
        beta_cross_init: float = SEED.BETA_CROSS_INIT,
    ):
        super().__init__()
        self.R = int(r_pool)
        self.stride = int(stride)
        self.eps = float(eps)
        n_g = int(L1.COL_PASSES if n_gaba_passes is None else n_gaba_passes)
        self.n_gaba_passes = n_g
        if eta_pass_init is not None:
            ez = float(eta_pass_init)
        elif eta_z_init is not None:
            ez = float(eta_z_init)
        else:
            ez = float(SEED.ETA_Z)
        ez = max(float(ez), 1e-6)
        self._eta_z_raw = nn.Parameter(
            torch.tensor(_inv_softplus(ez), dtype=torch.float32)
        )
        t0 = float(tau_init if tau_init is not None else SEED.TAU_INIT)
        self._tau_raw = nn.Parameter(
            torch.tensor(_inv_softplus(max(t0, 1e-4)), dtype=torch.float32)
        )
        ga0 = float(gain_alpha_init if gain_alpha_init is not None else SEED.GAIN_ALPHA_INIT)
        self._gain_alpha_raw = nn.Parameter(
            torch.tensor(_inv_softplus(max(ga0, 1e-4)), dtype=torch.float32)
        )
        self._beta_coll_raw = nn.Parameter(
            torch.tensor(
                _inv_softplus(max(float(beta_coll_init), 1e-6)),
                dtype=torch.float32,
            )
        )
        self._beta_cross_raw = nn.Parameter(
            torch.tensor(
                _inv_softplus(max(float(beta_cross_init), 1e-6)),
                dtype=torch.float32,
            )
        )
        R_geom = float(L1.COL_RADIUS)
        sd0 = float(L1.COL_SIGMA_D) if L1.COL_SIGMA_D is not None else R_geom / 2.0
        st0 = float(L1.COL_SIGMA_T)
        ratio_d = max(sd0 / R_geom, 1e-4)
        ratio_t = max(st0 / R_geom, 1e-4)
        self._alpha_d_raw = nn.Parameter(
            torch.tensor(_inv_softplus(ratio_d), dtype=torch.float32)
        )
        self._alpha_t_raw = nn.Parameter(
            torch.tensor(_inv_softplus(ratio_t), dtype=torch.float32)
        )

    def collinear_sigmas(self, R_col: int | float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(σ_d, σ_t)`` as 0-dim tensors = softplus(α) · R_col."""
        Rt = torch.as_tensor(float(R_col), device=self._alpha_d_raw.device, dtype=self._alpha_d_raw.dtype)
        sigma_d = F.softplus(self._alpha_d_raw) * Rt
        sigma_t = F.softplus(self._alpha_t_raw) * Rt
        return sigma_d, sigma_t

    @property
    def eta_z(self) -> torch.Tensor:
        """Positive seed-NR scale ``η_z = softplus(raw)`` (clamped ``≥ 10^{-6}``)."""
        return F.softplus(self._eta_z_raw).clamp_min(1e-6).view(())

    @property
    def tau(self) -> torch.Tensor:
        """Positive gate temperature ``τ`` (``softplus`` of raw)."""
        return F.softplus(self._tau_raw)

    @property
    def gain_alpha(self) -> torch.Tensor:
        """Gain span ``α``; ``gain = 1 + α·tanh(u/τ)``, ``α`` clamped ``≤ 0.49``."""
        return F.softplus(self._gain_alpha_raw).clamp(max=0.49)

    @property
    def beta_coll(self) -> torch.Tensor:
        return F.softplus(self._beta_coll_raw)

    @property
    def beta_cross(self) -> torch.Tensor:
        return F.softplus(self._beta_cross_raw)

    def normalize_pre_gaba(self, rho_k_raw: torch.Tensor) -> torch.Tensor:
        """Scalar ``η_z`` NR on flat ``(N,K)`` (matches seed NR before GABA passes)."""
        u = rho_k_raw
        if u.dim() != 2:
            raise ValueError("normalize_pre_gaba expects (N, K) flat tensor")
        return nr_squash_k_bins(u, self.eta_z, float(self.eps))


# ═══════════════════════════════════════════════════════════════
# Full L1 pipeline: hypercolumn + seed + recurrence
# ═══════════════════════════════════════════════════════════════

def run_l1_hypercolumn(
    h2m: torch.Tensor,
    theta_h: torch.Tensor,
    border_mask: torch.Tensor,
    seed: HypercolumnSeed,
    P: int = L1.PATCH_SIZE,
    patch_overlap: int = L1.PATCH_OVERLAP,
    border_patch_max_frac: float = L1.BORDER_PATCH_MAX_FRAC,
    K: int = L1.COL_K_BINS,
    R: int = L1.COL_RADIUS,
    sigma_d: float | None = None,
    sigma_t: float | None = None,
    n_passes: int = L1.COL_PASSES,
    eps: float = 1e-6,
    verbose: bool = True,
    cells_format: str = "numpy",
) -> dict:
    """Run the full hypercolumn L1 pipeline.

    L0 output → K-bin hypercolumns → **seed NR** (learned ``η_z``) → ``T`` multiplicative
    tanh-gate passes (``ρ ← ρ^{\\mathrm{seed}} \\odot (1+α\\tanh(u/τ))``), dominant readout.

    Args:
        h2m: (H, W) from L0
        theta_h: (H, W) from L0
        border_mask: (H, W) bool
        seed: HypercolumnSeed (``η_z``, ``τ``, ``α``, ``β_coll``, ``β_cross``, σ_d/σ_t; all learned scalars)
        P, patch_overlap, border_patch_max_frac: patch geometry
        K, R, n_passes: recurrence params (``n_passes`` must match ``seed.n_gaba_passes``)
        sigma_d, sigma_t: optional **fixed** floats for tests; default uses
            ``seed.collinear_sigmas(R)`` (learned).
        verbose: print diagnostics
        cells_format: ``"numpy"`` (default, disk / infer) or ``"torch"`` — when
            ``"torch"``, return tensors on ``device`` so autograd can reach
            ``seed`` (training with live L1).

    Returns:
        cells dict compatible with the renderer interface
    """
    device = h2m.device
    R_int = int(R)
    k_int = int(K)
    if sigma_d is not None and sigma_t is not None:
        sigma_d_t = torch.as_tensor(float(sigma_d), device=device, dtype=torch.float32)
        sigma_t_t = torch.as_tensor(float(sigma_t), device=device, dtype=torch.float32)
    else:
        sigma_d_t, sigma_t_t = seed.collinear_sigmas(float(R_int))

    # Step 1: Build hypercolumns
    hc = build_hypercolumns(
        h2m, theta_h, border_mask,
        P=P, patch_overlap=patch_overlap,
        border_patch_max_frac=border_patch_max_frac,
        K=K, eps=eps,
    )
    nH, nW, N = hc["nH"], hc["nW"], hc["N"]
    S = hc["S"]

    if int(n_passes) != int(seed.n_gaba_passes):
        raise ValueError(
            f"n_passes={n_passes} must match seed.n_gaba_passes={seed.n_gaba_passes}"
        )

    if verbose:
        print(f"  hypercolumn grid {nH}×{nW} = {N} cells, K={K} bins")

    rho_k_raw = hc["rho_k"]  # (N, K) raw cos² bin mass
    z0 = hc["z0"]            # (N,)

    if verbose:
        interior = ~hc["is_border"]
        rho_max_raw = rho_k_raw[interior].max(dim=-1).values
        ez = float(seed.eta_z.detach().cpu().item())
        ta = float(seed.tau.detach().cpu().item())
        al = float(seed.gain_alpha.detach().cpu().item())
        print(
            f"  η_z={ez:.3f}  τ={ta:.4f}  α={al:.4f}  "
            f"β_coll={seed.beta_coll.item():.3f}  β_cross={seed.beta_cross.item():.3f}",
        )
        print(f"  raw ρ_k max (interior): mean={rho_max_raw.mean():.4f} "
              f"max={rho_max_raw.max():.4f}")

    rho_k_gaba, kappa_k, kappa_k_pass0, rho_k_initial, s_k_first, s_bar_first = (
        gaba_recurrence(
            rho_k_raw, nH, nW, hc["is_border"], K,
            R=R_int, sigma_d=sigma_d_t, sigma_t=sigma_t_t,
            n_passes=n_passes,
            beta_coll=seed.beta_coll,
            beta_cross=seed.beta_cross,
            eta_z=seed.eta_z,
            tau=seed.tau,
            gain_alpha=seed.gain_alpha,
            nr_eps=float(seed.eps),
            eps=eps,
            eta_update_fn=None,
        )
    )
    sk_max_hw = s_k_first.squeeze(0).max(dim=0).values
    rho0_hwk = rho_k_initial.reshape(nH, nW, k_int).permute(2, 0, 1)
    dom_hw = rho0_hwk.argmax(dim=0)
    sb_k_hw = s_bar_first.squeeze(0)
    sbar_hw = torch.gather(sb_k_hw, 0, dom_hw.unsqueeze(0)).squeeze(0)

    if verbose:
        interior = ~hc["is_border"]
        rho_max_seed = rho_k_initial[interior].max(dim=-1).values
        print(f"  after seed NR: ρ_max mean={rho_max_seed.mean():.4f} "
              f"max={rho_max_seed.max():.4f}")
        rho_max_gaba = rho_k_gaba[interior].max(dim=-1).values
        print(f"  after {n_passes} GABA passes: "
              f"ρ_max mean={rho_max_gaba.mean():.4f} "
              f"max={rho_max_gaba.max():.4f}")

    # Step 4: Extract dominant orientation for renderer interface
    dom = extract_dominant(
        rho_k_gaba, kappa_k, rho_k_initial,
        hc["bin_centers"], hc["is_border"], K,
        nH, nW, R_int, sigma_d_t, sigma_t_t, eps=eps,
    )

    interior_flat = tile_interior_flat(
        nH, nW, hc["is_border"], seed.R, seed.stride, device,
    )
    idx_dom = dom["dominant_bin"].unsqueeze(-1)
    kappa_pass0_dom = kappa_k_pass0.gather(1, idx_dom).squeeze(-1)
    kappa_pass0_dom = torch.where(
        hc["is_border"], torch.zeros_like(kappa_pass0_dom), kappa_pass0_dom,
    )

    dom["rho"] = dom["rho"] * interior_flat
    dom["rho_initial"] = dom["rho_initial"] * interior_flat
    dom["kappa_col"] = dom["kappa_col"] * interior_flat
    kappa_pass0_dom = kappa_pass0_dom * interior_flat
    dom["e_col"] = dom["e_col"] * interior_flat
    dom["rho_max"] = dom["rho_max"] * interior_flat
    rho_pair = torch.stack([dom["rho"], dom["rho"]], dim=-1)
    kappa_pair = torch.stack([dom["kappa_col"], dom["kappa_col"]], dim=-1)

    if cells_format == "torch":
        lam3_hw = torch.zeros(nH, nW, device=device, dtype=h2m.dtype)
        cells = {
            "nH": nH,
            "nW": nW,
            "P": P,
            "S": S,
            "theta": dom["theta"],
            "lam": rho_pair,
            "lam3": lam3_hw.reshape(-1),
            "z0": z0,
            "cx": hc["cx"],
            "cy": hc["cy"],
            "cx_z2": hc["cx"],
            "cy_z2": hc["cy"],
            "is_border": hc["is_border"].bool().contiguous(),
            "kappa": kappa_pair,
            "q": torch.zeros(N, 2, device=device, dtype=h2m.dtype),
            "z1_abs_sum": torch.zeros(N, device=device, dtype=h2m.dtype),
            "rho_k": rho_k_gaba,
            "kappa_k": kappa_k,
            "rho_k_initial": rho_k_initial,
            "dominant_bin": dom["dominant_bin"],
            "K": K,
            "kappa_col_cell": dom["kappa_col"].reshape(nH, nW),
            "kappa_pass0_cell": kappa_pass0_dom.reshape(nH, nW),
            "e_col_cell": dom["e_col"].reshape(nH, nW),
            "rho_initial_cell": dom["rho_initial"].reshape(nH, nW),
            "rho_max_cell": dom["rho_max"].reshape(nH, nW),
            "sk_max_cell": sk_max_hw,
            "sbar_cell": sbar_hw,
        }
        return cells

    is_border_out = hc["is_border"].cpu().numpy()
    kappa_cell = dom["kappa_col"].reshape(nH, nW).detach().cpu().numpy()
    kappa_pass0_cell = kappa_pass0_dom.reshape(nH, nW).detach().cpu().numpy()
    e_cell = dom["e_col"].reshape(nH, nW).detach().cpu().numpy()
    rho_initial_cell = dom["rho_initial"].reshape(nH, nW).detach().cpu().numpy()
    rho_max_cell = dom["rho_max"].reshape(nH, nW).detach().cpu().numpy()

    lam3_hw = torch.zeros(nH, nW, device=device, dtype=h2m.dtype).cpu().numpy()

    # Build renderer-compatible cells dict
    cells = {
        "nH": nH,
        "nW": nW,
        "P": P,
        "S": S,
        "theta": dom["theta"].cpu().numpy(),
        "lam": rho_pair.detach().cpu().numpy(),
        "lam3": lam3_hw,
        "z0": z0.detach().cpu().numpy(),
        "cx": hc["cx"].cpu().numpy(),
        "cy": hc["cy"].cpu().numpy(),
        "cx_z2": hc["cx"].cpu().numpy(),
        "cy_z2": hc["cy"].cpu().numpy(),
        "is_border": is_border_out,
        "kappa": kappa_pair.detach().cpu().numpy(),
        "q": torch.zeros(N, 2, device=device).cpu().numpy(),
        "z1_abs_sum": torch.zeros(N, device=device).cpu().numpy(),
        "rho_k": rho_k_gaba.detach().cpu().numpy(),
        "kappa_k": kappa_k.detach().cpu().numpy(),
        "rho_k_initial": rho_k_initial.detach().cpu().numpy(),
        "dominant_bin": dom["dominant_bin"].cpu().numpy(),
        "K": K,
        "kappa_col_cell": kappa_cell,
        "kappa_pass0_cell": kappa_pass0_cell,
        "e_col_cell": e_cell,
        "rho_initial_cell": rho_initial_cell,
        "rho_max_cell": rho_max_cell,
        "sk_max_cell": sk_max_hw.detach().cpu().numpy(),
        "sbar_cell": sbar_hw.detach().cpu().numpy(),
    }
    return cells


# ═══════════════════════════════════════════════════════════════
# Legacy compatibility aliases
# ═══════════════════════════════════════════════════════════════

# Old code imports run_l1 from L1 — provide a wrapper
def run_l1(
    z1: torch.Tensor,
    z2: torch.Tensor,
    P: int,
    border_mask: torch.Tensor,
    patch_overlap: int,
    border_patch_max_frac: float,
    eps: float,
    img: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    verbose: bool = True,
    # New: pass seed module and L0 fields for hypercolumn construction
    seed: HypercolumnSeed | None = None,
    h2m: torch.Tensor | None = None,
    theta_h: torch.Tensor | None = None,
) -> dict:
    """Legacy-compatible wrapper.

    If seed, h2m, theta_h are provided: runs the new hypercolumn pipeline.
    Otherwise: falls back to computing h2m/theta_h from z2 and running
    the hypercolumn pipeline with a default seed (``HypercolumnSeed``;
    ``eta_z_init`` / ``eta_pass_init`` / ``params.SEED`` defaults).

    ``img`` is accepted for API compatibility and ignored (RGB partition
    photometry was removed).
    """
    _ = img
    dev = torch.device(device) if device is not None else z2.device

    if h2m is None:
        # Reconstruct from z2
        h2m = z2.abs().float().to(dev)
    if theta_h is None:
        theta_h = (0.5 * torch.angle(z2)).float().to(dev)

    if seed is None:
        seed = HypercolumnSeed().to(dev)

    border_mask = border_mask.to(dev)

    cells = run_l1_hypercolumn(
        h2m.to(dev), theta_h.to(dev), border_mask,
        seed=seed,
        P=P, patch_overlap=patch_overlap,
        border_patch_max_frac=border_patch_max_frac,
        verbose=verbose,
        eps=eps,
    )

    return cells
