r"""L1 — per-cell z₂ moments → cell grid (GPU-native).

From L0 pixel field z₂ = s₂ + i s₃, sum-pool over P×P patches:
  ρ_total(c) = Σ |z₂(p)|  (unweighted; seed presence / surround),
  Z₂ʷ(c) = Σ h₂m(p) z₂(p),  ρ_totalʷ(c) = Σ h₂m(p) |z₂(p)|,
  ρ_peak(c) = max_{p∈patch} |z₂(p)|,
  R(c) = |Z₂ʷ|/(ρ_totalʷ + ε),  θ(c) = ½ arg Z₂ʷ(c).
Splat anchors: h₂m-weighted centroid within patch (cx_z2, cy_z2).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from params import L1


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
    out = np.pad(img, ((0, Hp - H0), (0, Wp - W0), (0, 0)), mode="reflect")
    return out, H0, W0


def z_from_l0_harmonics(
    s: torch.Tensor,
    border_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z1 = torch.complex(s[..., 0], s[..., 1])
    z2 = torch.complex(s[..., 2], s[..., 3])
    z1[border_mask] = 0.0
    z2[border_mask] = 0.0
    return z1, z2


def _extract_patches_torch(
    t: torch.Tensor,
    nH: int,
    nW: int,
    P: int,
    S: int,
) -> torch.Tensor:
    patches = t.unfold(0, P, S).unfold(1, P, S)
    return patches.contiguous().reshape(nH * nW, P * P)


def _sum_pool2d(field: torch.Tensor, P: int, S: int) -> torch.Tensor:
    """Sum over P×P windows; field is (H, W)."""
    x = field.unsqueeze(0).unsqueeze(0)
    return F.avg_pool2d(x, kernel_size=P, stride=S).squeeze(0).squeeze(0) * (P * P)


def compute_cell_moments(
    h2m: torch.Tensor,
    z2: torch.Tensor,
    P: int,
    border_mask: torch.Tensor,
    patch_overlap: int,
    border_patch_max_frac: float,
    eps: float,
    device: torch.device | str | None = None,
    verbose: bool = True,
    return_torch: bool = False,
) -> dict:
    """Sum-pool z₂ moments and h₂m anchors per cell."""
    dev = torch.device(device) if device is not None else h2m.device
    h2m = h2m.to(dev, dtype=torch.float32)
    z2 = z2.to(dev)
    border_mask = border_mask.to(dev)

    H, W = h2m.shape
    S = stride_from_patch_overlap(P, patch_overlap)
    nH = (H - P) // S + 1 if H >= P else 0
    nW = (W - P) // S + 1 if W >= P else 0
    n = nH * nW

    if verbose:
        print(f"  grid {nH}x{nW} = {n} cells  (z₂ moments)")

    bm_float = border_mask.float()
    bm_patches = _extract_patches_torch(bm_float, nH, nW, P, S)
    is_border_flat = bm_patches.mean(dim=-1) > border_patch_max_frac

    z2_patches = _extract_patches_torch(z2, nH, nW, P, S)
    h2m_patches = _extract_patches_torch(h2m, nH, nW, P, S)

    # Unweighted presence energy — seed surround / E_rel (unchanged).
    z2_abs_sum = z2_patches.abs().sum(dim=-1).to(torch.float32)
    rho_total_flat = z2_abs_sum.reshape(-1)

    # h₂m-weighted orientation support — R and θ only.
    weighted_abs = (h2m_patches * z2_patches.abs()).to(torch.float32)
    Z2_w = (h2m_patches * z2_patches).sum(dim=-1)
    rho_tot_w = weighted_abs.sum(dim=-1)
    rho_peak_flat = z2_patches.abs().max(dim=-1).values.to(torch.float32).reshape(-1)
    del z2_patches, h2m_patches

    coherence_R_flat = (
        Z2_w.abs().to(torch.float32) / (rho_tot_w + eps)
    ).clamp(0.0, 1.0).reshape(-1)
    theta_flat = (
        0.5 * torch.atan2(Z2_w.imag, Z2_w.real + eps)
    ).to(torch.float32).reshape(-1)

    theta_flat = torch.where(is_border_flat, torch.zeros_like(theta_flat), theta_flat)
    coherence_R_flat = torch.where(
        is_border_flat, torch.zeros_like(coherence_R_flat), coherence_R_flat,
    )
    rho_total_flat = torch.where(
        is_border_flat, torch.zeros_like(rho_total_flat), rho_total_flat,
    )
    rho_peak_flat = torch.where(
        is_border_flat, torch.zeros_like(rho_peak_flat), rho_peak_flat,
    )
    ib_grid = is_border_flat.reshape(nH, nW)

    pis = torch.arange(nH, device=dev).repeat_interleave(nW)
    pjs = torch.arange(nW, device=dev).repeat(nH)
    cx_flat = pjs.double() * S + P / 2.0
    cy_flat = pis.double() * S + P / 2.0
    cx_grid = cx_flat.reshape(nH, nW)
    cy_grid = cy_flat.reshape(nH, nW)

    rows = torch.arange(H, device=dev, dtype=torch.float32).unsqueeze(1).expand(H, W)
    cols = torch.arange(W, device=dev, dtype=torch.float32).unsqueeze(0).expand(H, W)
    mass_2d = _sum_pool2d(h2m, P, S)
    cx_num = _sum_pool2d(h2m * cols, P, S)
    cy_num = _sum_pool2d(h2m * rows, P, S)
    has_mass = mass_2d > eps
    cx_anchor = torch.where(has_mass, cx_num / mass_2d.clamp_min(eps), cx_grid)
    cy_anchor = torch.where(has_mass, cy_num / mass_2d.clamp_min(eps), cy_grid)
    cx_anchor = torch.where(ib_grid, cx_grid, cx_anchor)
    cy_anchor = torch.where(ib_grid, cy_grid, cy_anchor)

    if verbose:
        rt = rho_total_flat.detach().cpu().numpy()
        n_border = int(is_border_flat.sum().item())
        print(f"  active={n - n_border}  border={n_border}")
        print(f"  rho_total: mean={rt.mean():.2f} max={rt.max():.2f}")
        rp = rho_peak_flat.detach().cpu().numpy()
        print(f"  rho_peak: mean={rp.mean():.2f} max={rp.max():.2f}")
        print(f"  R (h₂m-weighted |Z₂|/Σh₂m|z₂|): mean={float(coherence_R_flat.mean()):.3f}")

    if return_torch:
        return {
            "nH": nH,
            "nW": nW,
            "S": S,
            "P": P,
            "theta": theta_flat.reshape(nH, nW),
            "coherence_R": coherence_R_flat.reshape(nH, nW),
            "rho_total": rho_total_flat.reshape(nH, nW),
            "rho_peak": rho_peak_flat.reshape(nH, nW),
            "z0": rho_total_flat.reshape(nH, nW),
            "cx": cx_grid,
            "cy": cy_grid,
            "cx_z2": cx_anchor,
            "cy_z2": cy_anchor,
            "is_border": is_border_flat.reshape(nH, nW),
        }

    def _to_np(t):
        return t.detach().cpu().numpy()

    rho_hw = rho_total_flat.reshape(nH, nW)
    return {
        "nH": nH,
        "nW": nW,
        "S": S,
        "P": P,
        "theta": _to_np(theta_flat.reshape(nH, nW)),
        "coherence_R": _to_np(coherence_R_flat.reshape(nH, nW)),
        "rho_total": _to_np(rho_hw),
        "rho_peak": _to_np(rho_peak_flat.reshape(nH, nW)),
        "z0": _to_np(rho_hw),
        "cx": _to_np(cx_grid),
        "cy": _to_np(cy_grid),
        "cx_z2": _to_np(cx_anchor),
        "cy_z2": _to_np(cy_anchor),
        "is_border": _to_np(is_border_flat.reshape(nH, nW)),
    }


def run_l1(
    h2m: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    P: int,
    border_mask: torch.Tensor,
    patch_overlap: int,
    border_patch_max_frac: float,
    eps: float,
    device: torch.device | str | None = None,
    verbose: bool = True,
    return_torch: bool = False,
    **kw,
) -> dict:
    """Legacy name for ``compute_cell_moments`` (``z1`` and bin kwargs ignored)."""
    _ = (z1, kw)
    return compute_cell_moments(
        h2m,
        z2,
        P,
        border_mask,
        patch_overlap,
        border_patch_max_frac,
        eps,
        device=device,
        verbose=verbose,
        return_torch=return_torch,
    )
