r"""Renderer — anisotropic Gaussian splat + MLP thinning head (paper §2.5).

Pipeline:
  1. Cell grid: ρ-weighted θ combing → ρ-gated anchor smoothing.
  2. 2D anisotropic Gaussian splat of ρ at z₂ anchors → ρ̄(p) = Σ_c ρ_c φ_c(p).
  3. Per-pixel feature vector F_p = [ρ̄, coh, tang9, norm9] ∈ R²⁰.
  4. Thinning head: B̂(p) = ρ̄(p) · σ(W₂ ReLU(W₁ F_p + b₁) + b₂).
  NMS at inference thins to single-pixel width.

Learned: σ⊥, σ∥ (splat widths), s_t and s_n (stencil spacings),
         MLP 20→12→1 (265 params).  Total: 269 scalars.
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
_SIGMA_PAR_INIT = getattr(RENDER, "SIGMA_PAR_INIT", 2.0)
_SIGMA_PAR_MAX = getattr(RENDER, "SIGMA_PAR_MAX", 32.0)
_SPLAT_RADIUS_SIGMAS = getattr(RENDER, "SPLAT_RADIUS_SIGMAS", 3.0)


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


# ═══════════════════════════════════════════════════════════════
# Cell-grid smoothing
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
# Anisotropic 2D Gaussian splat (vectorized scatter_add)
# ═══════════════════════════════════════════════════════════════

def _aniso_kernel_phi(
    ox: torch.Tensor,
    oy: torch.Tensor,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    sig_perp: torch.Tensor,
    sig_par: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """φ = exp(−d⊥²/2σ⊥² − d∥²/2σ∥²) with offsets (ox, oy) from anchor to pixel."""
    d_perp = ox.unsqueeze(0) * cos_a.unsqueeze(1) - oy.unsqueeze(0) * sin_a.unsqueeze(1)
    d_par = ox.unsqueeze(0) * sin_a.unsqueeze(1) + oy.unsqueeze(0) * cos_a.unsqueeze(1)
    return torch.exp(
        -d_perp * d_perp / (2.0 * sig_perp * sig_perp + eps)
        - d_par * d_par / (2.0 * sig_par * sig_par + eps)
    )


def _splat_footprint_radius(sig_perp: torch.Tensor, sig_par: torch.Tensor, S: int) -> int:
    sp = float(sig_perp.detach().clamp(min=0.3, max=_SIGMA_PERP_MAX))
    sa = float(sig_par.detach().clamp(min=0.3, max=_SIGMA_PAR_MAX))
    r = max(
        int(math.ceil(_SPLAT_RADIUS_SIGMAS * sp)),
        int(math.ceil(_SPLAT_RADIUS_SIGMAS * sa)),
        int(math.ceil(_SPLAT_RADIUS_SIGMAS * S)),
        1,
    )
    return r


def _anisotropic_gaussian_splat(
    values: torch.Tensor,
    cx: torch.Tensor, cy: torch.Tensor,
    theta: torch.Tensor, is_border: torch.Tensor,
    sigma_perp: torch.Tensor,
    sigma_par: torch.Tensor,
    H: int, W: int, S: int,
    radius_sigmas: float = _SPLAT_RADIUS_SIGMAS,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deposit ρ̄ = Σ_c ρ_c φ_c; θ★ = scatter-max argmax ρ_c φ_c.

    φ_c is a finite 2D anisotropic Gaussian centered at anchor (c_x, c_y).
    """
    _ = radius_sigmas
    device, dtype = values.device, values.dtype
    sig_perp = sigma_perp.to(dtype=dtype, device=device).clamp(
        min=0.3, max=_SIGMA_PERP_MAX,
    )
    sig_par = sigma_par.to(dtype=dtype, device=device).clamp(
        min=0.3, max=_SIGMA_PAR_MAX,
    )
    foot_r = _splat_footprint_radius(sig_perp, sig_par, S)
    cut_perp = _SPLAT_RADIUS_SIGMAS * sig_perp
    cut_par = _SPLAT_RADIUS_SIGMAS * sig_par

    active = (~is_border) & (values.abs() > eps)
    active_idx = active.nonzero(as_tuple=True)[0]
    A = active_idx.shape[0]
    if A == 0:
        z = torch.zeros(H, W, device=device, dtype=dtype)
        return z, z

    val_a = values[active_idx]
    cx_a, cy_a = cx[active_idx], cy[active_idx]
    cos_a = torch.cos(theta[active_idx])
    sin_a = torch.sin(theta[active_idx])
    th_a = theta[active_idx]

    offsets = torch.arange(-foot_r, foot_r + 1, device=device, dtype=dtype)
    oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
    oy, ox = oy.reshape(-1), ox.reshape(-1)
    P = oy.shape[0]
    n_pix = H * W

    sum_rho_phi = torch.zeros(n_pix, device=device, dtype=dtype)
    max_rho_phi = torch.full((n_pix,), -1.0, device=device, dtype=dtype)
    theta_star = torch.zeros(n_pix, device=device, dtype=dtype)

    max_batch = max(1, 4_000_000 // P)
    for b0 in range(0, A, max_batch):
        b1 = min(b0 + max_batch, A)
        py_b = cy_a[b0:b1].unsqueeze(1) + oy.unsqueeze(0)
        px_b = cx_a[b0:b1].unsqueeze(1) + ox.unsqueeze(0)
        d_perp = ox.unsqueeze(0) * cos_a[b0:b1].unsqueeze(1) - oy.unsqueeze(0) * sin_a[b0:b1].unsqueeze(1)
        d_par = ox.unsqueeze(0) * sin_a[b0:b1].unsqueeze(1) + oy.unsqueeze(0) * cos_a[b0:b1].unsqueeze(1)
        in_bounds = (
            (py_b >= 0) & (py_b < H) & (px_b >= 0) & (px_b < W)
            & (d_perp.abs() <= cut_perp) & (d_par.abs() <= cut_par)
        )
        phi = _aniso_kernel_phi(
            ox, oy, cos_a[b0:b1], sin_a[b0:b1], sig_perp, sig_par, eps,
        ) * in_bounds.to(dtype=dtype)
        flat_idx = (py_b.long() * W + px_b.long()).clamp(0, n_pix - 1)

        rho_phi = val_a[b0:b1].unsqueeze(1) * phi
        sum_rho_phi.scatter_add_(0, flat_idx.reshape(-1), rho_phi.reshape(-1))

        th_expand = th_a[b0:b1].unsqueeze(1).expand_as(rho_phi)
        rp_flat = rho_phi.reshape(-1)
        th_flat = th_expand.reshape(-1)
        fi_flat = flat_idx.reshape(-1)
        update = rp_flat > max_rho_phi[fi_flat]
        update_idx = fi_flat[update]
        if update_idx.numel() > 0:
            max_rho_phi[update_idx] = rp_flat[update]
            theta_star[update_idx] = th_flat[update]

    return sum_rho_phi.reshape(H, W), theta_star.reshape(H, W)


# ═══════════════════════════════════════════════════════════════
# Bilinear sampling for stencil taps
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


def _sample_stencil_9(
    field: torch.Tensor,
    theta: torch.Tensor,
    spacing: torch.Tensor,
    direction: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """9-tap stencil along tangent or normal (j ∈ {-4,…,4}). Returns (H, W, 9)."""
    _ = eps
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
    for k in range(-4, 5):
        r_off = rows + k * sp * dr
        c_off = cols + k * sp * dc
        taps.append(_bilinear_sample_2d(field, r_off, c_off))
    return torch.stack(taps, dim=-1)


# ═══════════════════════════════════════════════════════════════
# Coherence diagnostic
# ═══════════════════════════════════════════════════════════════

def _compute_coherence(
    rho_bar: torch.Tensor,
    theta_star: torch.Tensor,
    values: torch.Tensor,
    cx: torch.Tensor, cy: torch.Tensor,
    theta: torch.Tensor, is_border: torch.Tensor,
    sigma_perp: torch.Tensor,
    sigma_par: torch.Tensor,
    H: int, W: int, S: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """coh(p) = Σ_c ρ_c φ_c cos²(θ_c − θ★) / (Σ_c ρ_c φ_c + ε)."""
    _ = rho_bar
    device, dtype = values.device, values.dtype
    sig_perp = sigma_perp.to(dtype=dtype, device=device).clamp(
        min=0.3, max=_SIGMA_PERP_MAX,
    )
    sig_par = sigma_par.to(dtype=dtype, device=device).clamp(
        min=0.3, max=_SIGMA_PAR_MAX,
    )
    foot_r = _splat_footprint_radius(sig_perp, sig_par, S)
    cut_perp = _SPLAT_RADIUS_SIGMAS * sig_perp
    cut_par = _SPLAT_RADIUS_SIGMAS * sig_par

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
    n_pix = H * W

    sum_rho_cos2 = torch.zeros(n_pix, device=device, dtype=dtype)
    sum_rho = torch.zeros(n_pix, device=device, dtype=dtype)

    max_batch = max(1, 4_000_000 // P)
    for b0 in range(0, A, max_batch):
        b1 = min(b0 + max_batch, A)
        py_b = cy_a[b0:b1].unsqueeze(1) + oy.unsqueeze(0)
        px_b = cx_a[b0:b1].unsqueeze(1) + ox.unsqueeze(0)
        d_perp = ox.unsqueeze(0) * cos_a[b0:b1].unsqueeze(1) - oy.unsqueeze(0) * sin_a[b0:b1].unsqueeze(1)
        d_par = ox.unsqueeze(0) * sin_a[b0:b1].unsqueeze(1) + oy.unsqueeze(0) * cos_a[b0:b1].unsqueeze(1)
        in_bounds = (
            (py_b >= 0) & (py_b < H) & (px_b >= 0) & (px_b < W)
            & (d_perp.abs() <= cut_perp) & (d_par.abs() <= cut_par)
        )
        phi = _aniso_kernel_phi(
            ox, oy, cos_a[b0:b1], sin_a[b0:b1], sig_perp, sig_par, eps,
        ) * in_bounds.to(dtype=dtype)
        flat_idx = (py_b.long() * W + px_b.long()).clamp(0, n_pix - 1)

        th_star_at_pix = theta_star.reshape(-1)[flat_idx]
        cos2_diff = torch.cos(th_a[b0:b1].unsqueeze(1) - th_star_at_pix).pow(2)

        rho_phi = val_a[b0:b1].unsqueeze(1) * phi
        sum_rho_cos2.scatter_add_(0, flat_idx.reshape(-1), (rho_phi * cos2_diff).reshape(-1))
        sum_rho.scatter_add_(0, flat_idx.reshape(-1), rho_phi.reshape(-1))

    return (sum_rho_cos2 / (sum_rho + eps)).reshape(H, W)


# Feature layout: [ρ̄, coh, tang_{-4}…tang_4, norm_{-4}…norm_4] ∈ R^20
_FEAT_DIM = int(getattr(RENDER, "THINNING_IN", 20))
_TANG_SLICE = slice(2, 11)
_NORM_SLICE = slice(11, 20)
_THIN_HIDDEN = int(getattr(RENDER, "THINNING_HIDDEN", 12))


# ═══════════════════════════════════════════════════════════════
# Thinning head (20 → 12 → 1 MLP)
# ═══════════════════════════════════════════════════════════════

class ThinningHead(nn.Module):
    """20→12→1 MLP: σ(W₂ ReLU(W₁ F + b₁) + b₂).

    F = [ρ̄, coh, tang9(9), norm9(9)] ∈ R²⁰.
    Initialised with structural priors per spec.
    """

    def __init__(
        self,
        in_dim: int = _FEAT_DIM,
        hidden_dim: int = _THIN_HIDDEN,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)
        self._init_priors()

    def _init_priors(self) -> None:
        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.bias.zero_()
            self.fc2.weight.zero_()

            # Unit 0: Mexican-hat on norm9 (non-max suppression)
            mex_hat = torch.full((9,), -0.25)
            mex_hat[4] = 1.0
            self.fc1.weight[0, _NORM_SLICE] = mex_hat

            # Unit 1: flat 1/9 on tang9 (along-contour smooth)
            self.fc1.weight[1, _TANG_SLICE] = 1.0 / 9.0

            # W₂ = 0, b₂ = 2 → gate ≈ σ(2) ≈ 0.88 at init (near-identity)
            self.fc2.bias.fill_(2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, 20). Returns: (N,) gate in (0, 1)."""
        h = F.relu(self.fc1(features))
        return torch.sigmoid(self.fc2(h).squeeze(-1))


# ═══════════════════════════════════════════════════════════════
# ModulationRenderer
# ═══════════════════════════════════════════════════════════════

class ModulationRenderer(nn.Module):
    """Anisotropic Gaussian splat + thinning head (paper §2.5).

    Learned: σ⊥ (1), σ∥ (1), s_t (1), s_n (1), ThinningHead 20→12→1 (265).
    Total: 269 parameters.
    """

    def __init__(self, hidden: int | None = None, **kwargs):
        super().__init__()
        _ = (hidden, kwargs)
        self._sigma_perp_raw = nn.Parameter(
            torch.tensor(_inv_softplus(_SIGMA_PERP_INIT), dtype=torch.float32)
        )
        self._sigma_par_raw = nn.Parameter(
            torch.tensor(_inv_softplus(_SIGMA_PAR_INIT), dtype=torch.float32)
        )
        self.s_t = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.s_n = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.thinning = ThinningHead()

    @property
    def sigma_perp(self) -> torch.Tensor:
        return F.softplus(self._sigma_perp_raw)

    @property
    def sigma_par(self) -> torch.Tensor:
        return F.softplus(self._sigma_par_raw)

    # Legacy compat
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


def upgrade_renderer_state_dict(state_dict: dict, prefix: str = "") -> dict:
    remove = {
        f"{prefix}_sigma_pre_raw",
        f"{prefix}_eta_h_raw",
        f"{prefix}_smooth_sigma_raw",
        f"{prefix}perp_conv.conv.weight",
        f"{prefix}perp_conv.conv.bias",
        f"{prefix}perp_conv.fc.weight",
        f"{prefix}perp_conv.fc.bias",
    }
    out = {k: v for k, v in state_dict.items() if k not in remove}
    par_key = f"{prefix}_sigma_par_raw"
    if par_key not in out:
        out[par_key] = torch.tensor(
            _inv_softplus(_SIGMA_PAR_INIT), dtype=torch.float32,
        )
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
    l0_pix: dict[str, torch.Tensor] | None = None,
    eps: float = 1e-6,
    training: bool = False,
    branch_pick: torch.Tensor | None = None,
    content_h: int | None = None,
    content_w: int | None = None,
    return_dominant_theta: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

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

    # ── Step 1: Cell-grid operations ──────────────────────────
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

    # Detach anchors and θ: no coordinate gradients back to seed
    cx_det, cy_det = _cx.detach(), _cy.detach()
    theta_det = theta_c.detach()
    rho_splat = torch.where(is_border, torch.zeros_like(rho_flat), rho_flat)

    # ── Step 2: Anisotropic Gaussian splat → ρ̄(p), θ★(p) ─────
    rho_bar, theta_star = _anisotropic_gaussian_splat(
        rho_splat, cx_det, cy_det, theta_det, is_border,
        sigma_perp=renderer.sigma_perp,
        sigma_par=renderer.sigma_par,
        H=H, W=W, S=S, eps=eps,
    )

    # ── Step 3: Coherence ─────────────────────────────────────
    coh = _compute_coherence(
        rho_bar, theta_star,
        rho_splat, cx_det, cy_det, theta_det, is_border,
        renderer.sigma_perp, renderer.sigma_par, H, W, S, eps=eps,
    )

    # ── Step 4: Feature vector F_p ────────────────────────────
    tang9 = _sample_stencil_9(rho_bar, theta_star, renderer.s_t, "tangent", eps)
    norm9 = _sample_stencil_9(rho_bar, theta_star, renderer.s_n, "normal", eps)
    # F_p = [ρ̄, coh, tang9(9), norm9(9)] ∈ R²⁰
    features = torch.cat([
        rho_bar.unsqueeze(-1),
        coh.unsqueeze(-1),
        tang9,
        norm9,
    ], dim=-1)  # (H, W, 20)

    # ── Step 5: Thinning head → B̂(p) = ρ̄(p) · gate(p) ──────
    feat_flat = features.reshape(H * W, _FEAT_DIM)
    gate = renderer.thinning(feat_flat).reshape(H, W)
    bmap = rho_bar * gate

    # ── Crop ──────────────────────────────────────────────────
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
    with torch.no_grad():
        bmap_t = render_boundary_map_torch(
            rho_t, proj_dev, renderer, cf_dev, proj["H"], proj["W"], None,
            eps=eps, training=False, branch_pick=bp,
            content_h=content_h, content_w=content_w,
        )
    return bmap_t.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# NMS
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
