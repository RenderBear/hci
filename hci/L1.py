r"""L1 — hypercolumn oriented-energy construction + GABA-budget recurrence.

Replaces per-patch eigendecomposition with direct K-bin oriented energy
projection at each cell position.  Each cell becomes a hypercolumn:
K orientation-tuned units pooling over the same receptive field (patch).

Pipeline:
  1. Extract patches from pixel-level h2m and θ_h fields (from L0).
  2. Project each patch's oriented energy onto K bins via cos² tuning.
  3. Per cell: subtract min across bins, then divisive NR vs learned η_z
     (pre-GABA squash only).  Recurrence runs on this normalized tensor.
  4. Run recurrent GABA-budget collinear facilitation (T passes); no further
     min-subtraction or squashing inside or after recurrence.
  5. Extract dominant orientation, ρ, κ per cell for the renderer.

``rho_k_initial`` / ``rho_initial_cell`` store the **same** pre-GABA
normalized ``ρ_k`` (for Δρ diagnostics: value at the post-recurrence
dominant bin vs final ρ at that bin).  ``kappa_pass0_cell`` / ``kappa_col_cell``
store κ after the first / final GABA pass (with ``L1.COL_KAPPA_NORM="cosine"``,
κ is scalar per cell and equal in every bin, so "winner bin" is arbitrary).

The collinear recurrence uses depthwise conv2d — each bin convolved with
its own tangent-selective kernel.  Junctions with 3+ arms are naturally
represented as 3+ active bins.

Learned parameters: η_z (1 scalar), collinear Gaussian widths via
``α_d, α_t`` (softplus, scaled by ``R``).  Everything else is fixed geometry.
"""

from __future__ import annotations

import math

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
    w_d = torch.exp(-dist_sq / (2.0 * sigma_d * sigma_d))
    w_d[R, R] = 0.0
    w_d[dist_sq > R * R] = 0.0

    kernels = torch.zeros(K, 2 * R + 1, 2 * R + 1, device=device, dtype=dtype)
    for k in range(K):
        theta_k = k * math.pi / K
        d_perp = dj * math.cos(theta_k) - di * math.sin(theta_k)
        w_t = torch.exp(-d_perp * d_perp / (2.0 * sigma_t * sigma_t))
        kernels[k] = w_d * w_t
    return kernels.unsqueeze(1)


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

def gaba_recurrence(
    rho_k: torch.Tensor,
    nH: int, nW: int,
    is_border: torch.Tensor,
    K: int,
    R: int,
    sigma_d: torch.Tensor,
    sigma_t: torch.Tensor,
    n_passes: int,
    eps: float = 1e-6,
    kappa_norm: str = "cosine",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recurrent GABA-budget collinear facilitation on K-channel cell grid.

    Args:
        rho_k: (N, K) per-bin cell energies
        nH, nW: cell grid dimensions
        is_border: (N,) bool
        K: number of orientation bins
        R: kernel radius (integer support)
        sigma_d, sigma_t: 0-dim tensors, Gaussian scales for isotropic / tangent factors
        n_passes: number of recurrence iterations
        kappa_norm:
            ``"cosine"`` — scalar :math:`\\kappa(c)` = cosine sim between
            :math:`\\rho_k(c)` and :math:`S_k(c)` at each cell; same :math:`\\kappa`
            applied to every bin (neighborhood agrees with cell profile).
            ``"max"`` — per-bin :math:`\\kappa_k=S_k/(\\max_j S_j+\\epsilon)`.
            ``"fair_share"`` — per-bin :math:`\\kappa_k=S_k/(E_{\\text{total}}/K+\\epsilon)`.

    Returns:
        rho_k_out: (N, K) modulated per-bin energies after all passes
        kappa_k: (N, K) final-pass κ stored per bin (cosine mode repeats scalar K-wise)
        kappa_k_pass0: (N, K) κ after the first pass (zeros if ``n_passes == 0``)
    """
    device = rho_k.device
    dtype = rho_k.dtype

    # Build depthwise kernels: (K, 1, 2R+1, 2R+1)
    kernels = _build_collinear_kernels(R, K, sigma_d, sigma_t, device, dtype)

    # Reshape to (1, K, nH, nW) for depthwise conv
    ib_grid = is_border.reshape(nH, nW)
    rho_grid = rho_k.reshape(nH, nW, K).permute(2, 0, 1).unsqueeze(0)  # (1, K, nH, nW)

    # Zero border cells
    border_mask_4d = ib_grid.unsqueeze(0).unsqueeze(0).expand_as(rho_grid)
    rho_grid = torch.where(border_mask_4d, torch.zeros_like(rho_grid), rho_grid)

    kappa_grid = torch.zeros_like(rho_grid)
    kappa_k_pass0 = torch.zeros_like(rho_k)

    for t in range(n_passes):
        # Depthwise conv: each bin k convolved with its own kernel W_k
        # groups=K means each channel is convolved independently
        s_k = F.conv2d(rho_grid, kernels, padding=R, groups=K)  # (1, K, nH, nW)

        if kappa_norm == "cosine":
            # Scalar gate: agreement between cell profile ρ and neighborhood pool S.
            num = (rho_grid * s_k).sum(dim=1, keepdim=True)
            norm_rho = (rho_grid * rho_grid).sum(dim=1, keepdim=True).clamp_min(0.0).sqrt()
            norm_s = (s_k * s_k).sum(dim=1, keepdim=True).clamp_min(0.0).sqrt()
            kappa_scalar = num / (norm_rho * norm_s + eps)
            kappa_scalar = kappa_scalar.clamp(0.0, 1.0)
            kappa_grid = kappa_scalar.expand_as(rho_grid)
        elif kappa_norm == "max":
            # Winner-referenced gate: strongest bin in neighborhood → κ=1, rest < 1.
            s_max = s_k.amax(dim=1, keepdim=True)
            kappa_grid = s_k / (s_max + eps)
        elif kappa_norm == "fair_share":
            e_total = s_k.sum(dim=1, keepdim=True)
            per_bin_avg = e_total / K
            kappa_grid = s_k / (per_bin_avg + eps)
        else:
            raise ValueError(
                "kappa_norm must be 'cosine', 'max', or 'fair_share', "
                f"got {kappa_norm!r}",
            )
        kappa_grid = kappa_grid.clamp(0.0, 1.0)

        # Zero borders
        kappa_grid = torch.where(border_mask_4d, torch.zeros_like(kappa_grid), kappa_grid)

        if t == 0:
            kappa_k_pass0 = (
                kappa_grid.squeeze(0).permute(1, 2, 0).reshape(-1, K).detach().clone()
            )

        # Modulate
        rho_grid = rho_grid * kappa_grid
        rho_grid = torch.where(border_mask_4d, torch.zeros_like(rho_grid), rho_grid)

    # Reshape back to (N, K)
    rho_k_out = rho_grid.squeeze(0).permute(1, 2, 0).reshape(-1, K)
    kappa_k_out = kappa_grid.squeeze(0).permute(1, 2, 0).reshape(-1, K)

    return rho_k_out, kappa_k_out, kappa_k_pass0


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
    """Extract per-cell scalar ρ, θ, κ from K-bin representation."""
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
# Seed module (simplified: just η_z on K-bin energies)
# ═══════════════════════════════════════════════════════════════

def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


class HypercolumnSeed(nn.Module):
    """Learned η_z: per-bin NR **before** GABA only (min-subtract + squash).

    Also learns collinear kernel scales ``σ_d = softplus(α̃_d)·R``,
    ``σ_t = softplus(α̃_t)·R`` (HCI spec), used in GABA depthwise convs.

    ``normalize_pre_gaba`` applies min across bins then
    ``ρ̃²/(ρ̃²+η_z²+ε)``.  That tensor feeds ``gaba_recurrence`` and is
    cached as ``rho_k_initial`` for diagnostics.  No squashing after
    recurrence.
    """

    def __init__(
        self,
        r_pool: int = SEED.R_POOL,
        stride: int = SEED.STRIDE,
        eps: float = SEED.EPS,
        eta_z_init: float = SEED.ETA_Z_INIT,
    ):
        super().__init__()
        self.R = int(r_pool)
        self.stride = int(stride)
        self.eps = float(eps)
        self._eta_z_raw = nn.Parameter(
            torch.tensor(_inv_softplus(max(eta_z_init, 1e-6)), dtype=torch.float32)
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
        return F.softplus(self._eta_z_raw)

    def normalize_pre_gaba(self, rho_k_raw: torch.Tensor) -> torch.Tensor:
        """Min-subtract per cell, then NR vs η_z — **input** to GABA recurrence.

        ``rho_tilde_k = rho_k_raw - min_j rho_j_raw``; then
        ``rho_tilde_k² / (rho_tilde_k² + η_z² + ε)``.
        """
        rho_min = rho_k_raw.amin(dim=-1, keepdim=True)
        rho_t = rho_k_raw - rho_min
        rho_sq = rho_t * rho_t
        eta_sq = self.eta_z * self.eta_z
        return rho_sq / (rho_sq + eta_sq + self.eps)


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
    kappa_norm: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run the full hypercolumn L1 pipeline.

    L0 output → K-bin hypercolumns → min-subtract + η_z NR (pre-GABA only)
    → GABA recurrence → extract dominant orientation for renderer.

    Args:
        h2m: (H, W) from L0
        theta_h: (H, W) from L0
        border_mask: (H, W) bool
        seed: HypercolumnSeed module (provides learned η_z)
        P, patch_overlap, border_patch_max_frac: patch geometry
        K, R, n_passes: recurrence params
        sigma_d, sigma_t: optional **fixed** floats for tests; default uses
            ``seed.collinear_sigmas(R)`` (learned).
        kappa_norm: ``"cosine"``, ``"max"``, or ``"fair_share"``; default
            ``L1.COL_KAPPA_NORM``.
        verbose: print diagnostics

    Returns:
        cells dict compatible with the renderer interface
    """
    device = h2m.device
    kn = str(L1.COL_KAPPA_NORM) if kappa_norm is None else str(kappa_norm)
    R_int = int(R)
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

    if verbose:
        print(f"  hypercolumn grid {nH}×{nW} = {N} cells, K={K} bins")

    # Step 2: Min-subtract + η_z NR (pre-GABA); same tensor cached for ρ_initial
    rho_k_raw = hc["rho_k"]  # (N, K)
    z0 = hc["z0"]            # (N,)
    rho_k_pre = seed.normalize_pre_gaba(rho_k_raw)
    rho_k_initial = rho_k_pre.detach().clone()

    if verbose:
        interior = ~hc["is_border"]
        rho_max_raw = rho_k_raw[interior].max(dim=-1).values
        rho_max_pre = rho_k_pre[interior].max(dim=-1).values
        print(f"  η_z={seed.eta_z.item():.3f}  "
              f"raw ρ_k max (interior): mean={rho_max_raw.mean():.4f} "
              f"max={rho_max_raw.max():.4f}")
        print(f"  pre-GABA NR ρ_k max (interior, GABA in): mean={rho_max_pre.mean():.4f} "
              f"max={rho_max_pre.max():.4f}")

    # Step 3: GABA-budget recurrence (no squashing inside or after)
    rho_k_gaba, kappa_k, kappa_k_pass0 = gaba_recurrence(
        rho_k_pre, nH, nW, hc["is_border"], K,
        R=R_int, sigma_d=sigma_d_t, sigma_t=sigma_t_t,
        n_passes=n_passes, eps=eps, kappa_norm=kn,
    )

    if verbose:
        interior = ~hc["is_border"]
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
    the hypercolumn pipeline with a default seed (``HypercolumnSeed``, η_z from
    ``params.SEED.ETA_Z_INIT``).

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
