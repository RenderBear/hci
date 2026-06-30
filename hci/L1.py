r"""L1 — per-cell z₂: orientation bins (von Mises) + legacy Z₂ moments"""

from __future__ import annotations

import math

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
    kappa_vm: float | torch.Tensor = 2.0,
    num_orient_bins: int | None = None,
    **kw,
) -> dict:
    _ = kw
    K = int(num_orient_bins if num_orient_bins is not None else getattr(L1, "NUM_ORIENT_BINS", 8))
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
        print(f"  grid {nH}x{nW} = {n} cells  (z₂ moments, K={K})")

    bm_float = border_mask.float()
    bm_patches = _extract_patches_torch(bm_float, nH, nW, P, S)
    is_border_flat = bm_patches.mean(dim=-1) > border_patch_max_frac

    z2_patches = _extract_patches_torch(z2, nH, nW, P, S)
    PP = P * P

    z2_abs = z2_patches.abs().to(torch.float32)
    z2_abs_sum = z2_abs.sum(dim=-1)
    rho_total_flat = z2_abs_sum.reshape(-1)

    Z2 = z2_patches.sum(dim=-1)
    rho_peak_cell = Z2.abs().to(torch.float32)
    rho_coherence_cell = torch.where(
        z2_abs_sum > eps,
        (rho_peak_cell / z2_abs_sum.clamp_min(eps)).clamp(0.0, 1.0),
        torch.zeros_like(rho_peak_cell),
    )
    rho_peak_flat = rho_peak_cell.reshape(-1)
    rho_coherence_flat = rho_coherence_cell.reshape(-1)

    theta_flat = (
        0.5 * torch.atan2(Z2.imag, Z2.real + eps)
    ).to(torch.float32).reshape(-1)

    ok_pix = (1.0 - bm_patches).clamp(0.0, 1.0)
    theta_p = 0.5 * torch.atan2(z2_patches.imag, z2_patches.real + eps).to(torch.float32)
    bar_theta = (torch.arange(K, device=dev, dtype=torch.float32) * (math.pi / float(K))).view(1, 1, K)
    if isinstance(kappa_vm, torch.Tensor):
        kvm = kappa_vm.to(dev, dtype=torch.float32).reshape(())
    else:
        kvm = torch.tensor(float(kappa_vm), device=dev, dtype=torch.float32)
    kvm = kvm.clamp_min(0.0)
    diff = 2.0 * (theta_p.unsqueeze(-1) - bar_theta)
    g = torch.exp(kvm * (torch.cos(diff) - 1.0))
    w_mag = z2_abs.unsqueeze(-1) * g * ok_pix.unsqueeze(-1)
    rho_bin_flat = w_mag.sum(dim=1)
    den_anch = rho_bin_flat + float(eps)

    pis = torch.arange(nH, device=dev).repeat_interleave(nW)
    pjs = torch.arange(nW, device=dev).repeat(nH)
    offs = torch.arange(PP, device=dev)
    ii = (offs // P).to(torch.float32).view(1, PP)
    jj = (offs % P).to(torch.float32).view(1, PP)
    rows_pix = pis.float().unsqueeze(1) * float(S) + ii
    cols_pix = pjs.float().unsqueeze(1) * float(S) + jj
    ax_bin_flat = (w_mag * cols_pix.unsqueeze(-1)).sum(dim=1) / den_anch
    ay_bin_flat = (w_mag * rows_pix.unsqueeze(-1)).sum(dim=1) / den_anch

    z2_ok_sum = (z2_abs * ok_pix).sum(dim=-1).clamp_min(float(eps))
    rho_bin_coh_flat = rho_bin_flat / z2_ok_sum.unsqueeze(-1)

    del z2_patches

    theta_flat = torch.where(is_border_flat, torch.zeros_like(theta_flat), theta_flat)
    rho_total_flat = torch.where(
        is_border_flat, torch.zeros_like(rho_total_flat), rho_total_flat,
    )
    rho_peak_flat = torch.where(
        is_border_flat, torch.zeros_like(rho_peak_flat), rho_peak_flat,
    )
    rho_coherence_flat = torch.where(
        is_border_flat, torch.zeros_like(rho_coherence_flat), rho_coherence_flat,
    )
    z0 = torch.zeros(n, K, device=dev, dtype=torch.float32)
    ib_exp = is_border_flat.unsqueeze(-1).expand(-1, K)
    rho_bin_flat = torch.where(ib_exp, z0, rho_bin_flat)
    rho_bin_coh_flat = torch.where(ib_exp, z0, rho_bin_coh_flat)

    ib_grid = is_border_flat.reshape(nH, nW)

    cx_flat = pjs.double() * S + P / 2.0
    cy_flat = pis.double() * S + P / 2.0
    cx_grid = cx_flat.reshape(nH, nW)
    cy_grid = cy_flat.reshape(nH, nW)

    cx_cell = cx_grid.reshape(-1, 1).expand(-1, K).to(ax_bin_flat.dtype)
    cy_cell = cy_grid.reshape(-1, 1).expand(-1, K).to(ay_bin_flat.dtype)
    ax_bin_flat = torch.where(ib_exp, cx_cell, ax_bin_flat)
    ay_bin_flat = torch.where(ib_exp, cy_cell, ay_bin_flat)

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

    rho_bin_hwk = rho_bin_flat.reshape(nH, nW, K)
    ax_bin_hwk = ax_bin_flat.reshape(nH, nW, K)
    ay_bin_hwk = ay_bin_flat.reshape(nH, nW, K)
    rho_bin_coh_hwk = rho_bin_coh_flat.reshape(nH, nW, K)

    if verbose:
        rt = rho_total_flat.detach().cpu().numpy()
        n_border = int(is_border_flat.sum().item())
        print(f"  active={n - n_border}  border={n_border}")
        print(f"  rho_total: mean={rt.mean():.2f} max={rt.max():.2f}")
        rp = rho_peak_flat.detach().cpu().numpy()
        print(f"  rho_peak |Z|: mean={rp.mean():.2f} max={rp.max():.2f}")
        rc = rho_coherence_flat.detach().cpu().numpy()
        print(f"  rho_coherence |Z|/ρ_total: mean={rc.mean():.3f} max={rc.max():.3f}")
        rb = rho_bin_flat.detach().cpu().numpy()
        print(f"  rho_bin: mean={rb.mean():.3f} max={rb.max():.3f}")

    if return_torch:
        return {
            "nH": nH,
            "nW": nW,
            "S": S,
            "P": P,
            "K": K,
            "theta": theta_flat.reshape(nH, nW),
            "rho_total": rho_total_flat.reshape(nH, nW),
            "rho_peak": rho_peak_flat.reshape(nH, nW),
            "rho_coherence": rho_coherence_flat.reshape(nH, nW),
            "z0": rho_total_flat.reshape(nH, nW),
            "cx": cx_grid,
            "cy": cy_grid,
            "cx_z2": cx_anchor,
            "cy_z2": cy_anchor,
            "is_border": is_border_flat.reshape(nH, nW),
            "rho_bin": rho_bin_hwk,
            "ax_bin": ax_bin_hwk,
            "ay_bin": ay_bin_hwk,
            "rho_bin_coh": rho_bin_coh_hwk,
            "theta_bins": bar_theta.reshape(K),
        }

    def _to_np(t):
        return t.detach().cpu().numpy()

    rho_hw = rho_total_flat.reshape(nH, nW)
    return {
        "nH": nH,
        "nW": nW,
        "S": S,
        "P": P,
        "K": K,
        "theta": _to_np(theta_flat.reshape(nH, nW)),
        "rho_total": _to_np(rho_hw),
        "rho_peak": _to_np(rho_peak_flat.reshape(nH, nW)),
        "rho_coherence": _to_np(rho_coherence_flat.reshape(nH, nW)),
        "z0": _to_np(rho_hw),
        "cx": _to_np(cx_grid),
        "cy": _to_np(cy_grid),
        "cx_z2": _to_np(cx_anchor),
        "cy_z2": _to_np(cy_anchor),
        "is_border": _to_np(is_border_flat.reshape(nH, nW)),
        "rho_bin": _to_np(rho_bin_hwk),
        "ax_bin": _to_np(ax_bin_hwk),
        "ay_bin": _to_np(ay_bin_hwk),
        "rho_bin_coh": _to_np(rho_bin_coh_hwk),
        "theta_bins": _to_np(bar_theta.reshape(K)),
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
    kappa_vm: float | torch.Tensor = 2.0,
    num_orient_bins: int | None = None,
    **kw,
) -> dict:
    _ = z1
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
        kappa_vm=kappa_vm,
        num_orient_bins=num_orient_bins,
        **kw,
    )
