r"""L1 — pixel K-bin orientation projection → cell grid (GPU-native).

Per pixel (default von Mises): e_k = h_{2m} · exp(κ cos(2(θ_{2m} − kπ/K))).
Alternate: e_k = h_{2m} · cos^p(θ_{2m} − kπ/K).  Sum-pool over P×P patches →
ρ_raw^{(k)}(c).  k* + parabolic sub-bin θ for export/q/κ/renderer.
"""

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


def _patch_orientation_kappa(
    z1_patches: torch.Tensor,
    theta: torch.Tensor,
    q: torch.Tensor,
    P: int,
    eps: float,
) -> torch.Tensor:
    """Per-cell orientation confidence κ ∈ [0, 1] from z₁ polarity agreement."""
    n, _ = z1_patches.shape
    theta = theta.reshape(n, 1)
    q = q.reshape(n, 1)
    device = z1_patches.device
    half = (P - 1) / 2.0
    ys = torch.arange(P, device=device, dtype=torch.float64) - half
    xs = torch.arange(P, device=device, dtype=torch.float64) - half
    dy, dx = torch.meshgrid(ys, xs, indexing="ij")
    di_f, dj_f = dy.reshape(-1), dx.reshape(-1)
    rd = (di_f.pow(2) + dj_f.pow(2)).sqrt().clamp_min(eps)
    oi, oj = di_f / rd, dj_f / rd

    z1_c = z1_patches.to(torch.complex128)
    zr = z1_c.real
    zi = z1_c.imag

    sgn_q = torch.sign(q)
    sgn_q = torch.where(sgn_q == 0, torch.ones_like(sgn_q), sgn_q)
    nx = sgn_q * torch.cos(theta)
    ny = -sgn_q * torch.sin(theta)

    proj = nx.unsqueeze(-1) * zr.unsqueeze(1) + ny.unsqueeze(-1) * zi.unsqueeze(1)
    signed_mean = proj.mean(dim=-1)
    abs_mean = proj.abs().mean(dim=-1).clamp_min(eps)
    return (signed_mean.abs() / abs_mean).clamp(0.0, 1.0).to(torch.float32).reshape(n)


def _sum_pool2d(field: torch.Tensor, P: int, S: int) -> torch.Tensor:
    """Sum over P×P windows; field is (H, W)."""
    x = field.unsqueeze(0).unsqueeze(0)
    return F.avg_pool2d(x, kernel_size=P, stride=S).squeeze(0).squeeze(0) * (P * P)


def _sum_pool_bins(e: torch.Tensor, P: int, S: int) -> torch.Tensor:
    """Sum-pool K bin channels e (H, W, K) → (nH, nW, K)."""
    ek = e.permute(2, 0, 1).unsqueeze(0)
    pooled = F.avg_pool2d(ek, kernel_size=P, stride=S) * (P * P)
    return pooled.squeeze(0).permute(1, 2, 0)


def _bin_tuning_weight(
    diff: torch.Tensor,
    *,
    mode: str,
    cos_power: int,
    von_mises_kappa: float | torch.Tensor,
) -> torch.Tensor:
    """Orientation weight per bin; ``diff`` is ``θ_{2m} − bar_θ_k`` (period π)."""
    m = str(mode).lower().strip()
    if m == "von_mises":
        return torch.exp(von_mises_kappa * torch.cos(2.0 * diff))
    if m in ("cos_pow", "cos", "cos2"):
        return torch.cos(diff).pow(max(int(cos_power), 1))
    raise ValueError(f"unknown COL_BIN_TUNING mode {mode!r}; use 'von_mises' or 'cos_pow'")


def _pixel_bin_energy(
    h2m: torch.Tensor,
    z2: torch.Tensor,
    K: int,
    *,
    bin_tuning: str = "von_mises",
    cos_power: int = 2,
    von_mises_kappa: float | torch.Tensor = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return e (H,W,K) and bar_theta (K,)."""
    device = h2m.device
    dtype = h2m.dtype
    theta_2m = (0.5 * torch.angle(z2)).to(dtype)
    bar_theta = torch.linspace(0, math.pi, K + 1, device=device, dtype=dtype)[:-1]
    diff = theta_2m.unsqueeze(-1) - bar_theta
    tuning = _bin_tuning_weight(
        diff,
        mode=bin_tuning,
        cos_power=cos_power,
        von_mises_kappa=von_mises_kappa,
    )
    e = h2m.unsqueeze(-1) * tuning
    return e, bar_theta


def run_l1(
    h2m: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    P: int,
    border_mask: torch.Tensor,
    patch_overlap: int,
    border_patch_max_frac: float,
    eps: float,
    K: int | None = None,
    bin_tuning: str | None = None,
    cos_power: int | None = None,
    von_mises_kappa: float | torch.Tensor | None = None,
    device: torch.device | str | None = None,
    verbose: bool = True,
    return_torch: bool = False,
) -> dict:
    dev = torch.device(device) if device is not None else h2m.device
    h2m = h2m.to(dev, dtype=torch.float32)
    z1 = z1.to(dev)
    z2 = z2.to(dev)
    border_mask = border_mask.to(dev)

    K = int(L1.K if K is None else K)
    tuning = str(
        L1.COL_BIN_TUNING if bin_tuning is None else bin_tuning
    )
    cos_power = int(L1.COL_COS_POWER if cos_power is None else cos_power)
    if von_mises_kappa is None:
        kappa: float | torch.Tensor = L1.COL_VON_MISES_KAPPA
    else:
        kappa = von_mises_kappa

    H, W = h2m.shape
    S = stride_from_patch_overlap(P, patch_overlap)
    nH = (H - P) // S + 1 if H >= P else 0
    nW = (W - P) // S + 1 if W >= P else 0
    n = nH * nW

    if verbose:
        print(
            f"  grid {nH}x{nW} = {n} cells, K={K} bins, "
            f"{tuning} projection"
            + (
                f" (κ={float(kappa.detach() if isinstance(kappa, torch.Tensor) else kappa):g})"
                if tuning == "von_mises"
                else f" (cos^{cos_power})"
            )
        )

    e, bar_theta = _pixel_bin_energy(
        h2m, z2, K,
        bin_tuning=tuning,
        cos_power=cos_power,
        von_mises_kappa=kappa,
    )
    rho_bins = _sum_pool_bins(e, P, S)
    del e

    bm_float = border_mask.float()
    bm_patches = _extract_patches_torch(bm_float, nH, nW, P, S)
    is_border_flat = bm_patches.mean(dim=-1) > border_patch_max_frac

    rho_total = rho_bins.sum(dim=-1)
    rho_peak, k_star = rho_bins.max(dim=-1)
    k_left = (k_star - 1) % K
    k_right = (k_star + 1) % K
    r_left = rho_bins.gather(-1, k_left.unsqueeze(-1)).squeeze(-1)
    r_right = rho_bins.gather(-1, k_right.unsqueeze(-1)).squeeze(-1)
    denom = 2.0 * rho_peak - r_left - r_right
    frac = (r_left - r_right) / (denom + eps) * 0.5
    frac = frac.clamp(-0.5, 0.5)
    theta_2d = bar_theta[k_star] + frac * (math.pi / K)
    theta_flat = theta_2d.reshape(-1)

    delta_2d = (rho_peak / (rho_total + eps)).clamp(0.0, 1.0)
    delta_flat = delta_2d.reshape(-1)

    z1_patches = _extract_patches_torch(z1, nH, nW, P, S)
    Z1_sum = z1_patches.sum(dim=-1)
    z1_abs_sum = z1_patches.abs().sum(dim=-1).to(torch.float32)

    exp_ith = torch.complex(
        torch.cos(theta_flat), torch.sin(theta_flat),
    )
    q_flat = (Z1_sum.conj() * exp_ith).real.to(torch.float32)

    kappa_flat = _patch_orientation_kappa(
        z1_patches,
        theta_flat,
        q_flat,
        P,
        eps,
    )
    del z1_patches

    n_pix_patch_f = float(P * P)
    z1_bar = Z1_sum / n_pix_patch_f
    z1_bar_abs_flat = z1_bar.abs().to(torch.float32)
    phi_z1_flat = torch.angle(z1_bar).to(torch.float32)

    rho_peak_flat = rho_peak.reshape(-1)
    rho_total_flat = rho_total.reshape(-1)
    theta_flat = torch.where(is_border_flat, torch.zeros_like(theta_flat), theta_flat)
    q_flat = torch.where(is_border_flat, torch.zeros_like(q_flat), q_flat)
    delta_flat = torch.where(is_border_flat, torch.zeros_like(delta_flat), delta_flat)
    rho_peak_flat = torch.where(is_border_flat, torch.zeros_like(rho_peak_flat), rho_peak_flat)
    rho_total_flat = torch.where(is_border_flat, torch.zeros_like(rho_total_flat), rho_total_flat)
    ib_grid = is_border_flat.reshape(nH, nW)
    rho_bins = torch.where(
        ib_grid.unsqueeze(-1), torch.zeros_like(rho_bins), rho_bins,
    )
    kappa_flat = kappa_flat.masked_fill(is_border_flat, 0.0)
    z1_bar_abs_flat = torch.where(
        is_border_flat, torch.zeros_like(z1_bar_abs_flat), z1_bar_abs_flat,
    )
    phi_z1_flat = torch.where(is_border_flat, torch.zeros_like(phi_z1_flat), phi_z1_flat)

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
        rp = rho_peak_flat.cpu().numpy()
        rt = rho_total_flat.cpu().numpy()
        n_border = int(is_border_flat.sum().item())
        print(f"  active={n - n_border}  border={n_border}")
        print(f"  rho_peak: mean={rp.mean():.2f} max={rp.max():.2f}")
        print(f"  rho_total: mean={rt.mean():.2f} max={rt.max():.2f}")
        print(f"  delta (peak/total): mean={float(delta_flat.mean()):.3f}")
        print(f"  |q|: mean={float(q_flat.abs().mean()):.3f}")
        print(f"  kappa: mean={float(kappa_flat.mean()):.3f}")

    z2_bar_abs = (
        _sum_pool2d(z2.abs().to(torch.float32), P, S) / n_pix_patch_f
    ).reshape(nH, nW)

    if return_torch:
        return {
            "nH": nH,
            "nW": nW,
            "S": S,
            "P": P,
            "K": K,
            "k_star": k_star.reshape(nH, nW).to(torch.int64),
            "theta": theta_flat.reshape(nH, nW),
            "q": q_flat.reshape(nH, nW),
            "delta": delta_flat.reshape(nH, nW),
            "kappa": kappa_flat.reshape(nH, nW),
            "rho_peak": rho_peak_flat.reshape(nH, nW),
            "rho_bins": rho_bins,
            "z0": rho_total_flat.reshape(nH, nW),
            "cx": cx_grid,
            "cy": cy_grid,
            "cx_z2": cx_anchor,
            "cy_z2": cy_anchor,
            "z1_bar_abs": z1_bar_abs_flat.reshape(nH, nW),
            "z1_abs_sum": z1_abs_sum.reshape(nH, nW),
            "z2_bar_abs": z2_bar_abs,
            "phi_z1": phi_z1_flat.reshape(nH, nW),
            "is_border": is_border_flat.reshape(nH, nW),
        }

    def _to_np(t):
        return t.detach().cpu().numpy()

    return {
        "nH": nH,
        "nW": nW,
        "S": S,
        "P": P,
        "K": K,
        "k_star": _to_np(k_star.reshape(nH, nW).to(torch.int32)),
        "theta": _to_np(theta_flat.reshape(nH, nW)),
        "q": _to_np(q_flat.reshape(nH, nW)),
        "delta": _to_np(delta_flat.reshape(nH, nW)),
        "kappa": _to_np(kappa_flat.reshape(nH, nW)),
        "rho_peak": _to_np(rho_peak_flat.reshape(nH, nW)),
        "rho_bins": _to_np(rho_bins),
        "z0": _to_np(rho_total_flat.reshape(nH, nW)),
        "cx": _to_np(cx_grid),
        "cy": _to_np(cy_grid),
        "cx_z2": _to_np(cx_anchor),
        "cy_z2": _to_np(cy_anchor),
        "z1_bar_abs": _to_np(z1_bar_abs_flat.reshape(nH, nW)),
        "z1_abs_sum": _to_np(z1_abs_sum.reshape(nH, nW)),
        "z2_bar_abs": _to_np(z2_bar_abs),
        "phi_z1": _to_np(phi_z1_flat.reshape(nH, nW)),
        "is_border": _to_np(is_border_flat.reshape(nH, nW)),
    }
