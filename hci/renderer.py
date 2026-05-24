r"""Renderer — harmonic contour readout (interp → tang/norm stencils → gate).

θ-combing on the cell grid, bilinear interpolation of cached cell fields (ρ, θ,
κ_col from L1), then a 14-D feature stack
``[h2m_lum, h2m_chr, ρ̄, κ̄_col, tang₅, norm₅]`` with tangential / normal
``h2m`` samples at learned pixel spacings ``s_t``, ``s_n``, and
``B̂ = (h2m_lum+h2m_chr)·ρ̄·σ(MLP(F))``.  GABA recurrence lives in ``hci.L1``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import RENDER


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


# ═══════════════════════════════════════════════════════════════
# Cell-grid θ combing (unchanged)
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


# ═══════════════════════════════════════════════════════════════
# Bilinear interpolation: cell grid → pixel coordinates
# ═══════════════════════════════════════════════════════════════

def _interp_cell_to_pixel(
    field_grid: torch.Tensor,
    nH: int, nW: int,
    H: int, W: int,
    S: int, P: int,
) -> torch.Tensor:
    """Bilinear-interpolate a (nH, nW) or (nH, nW, C) cell-grid field
    to (H, W) or (H, W, C) pixel resolution.

    Cell centres sit at pixel coords (j*S + P/2, i*S + P/2).
    """
    device, dtype = field_grid.device, field_grid.dtype
    half_P = P / 2.0

    if field_grid.ndim == 2:
        field_4d = field_grid.unsqueeze(0).unsqueeze(0)  # (1,1,nH,nW)
    else:
        # (nH, nW, C) -> (1, C, nH, nW)
        field_4d = field_grid.permute(2, 0, 1).unsqueeze(0)

    # Pixel coords -> normalised grid coords for grid_sample
    # Cell i is at pixel y = i*S + half_P, cell j at pixel x = j*S + half_P
    # grid_sample wants coords in [-1, 1]
    py = torch.arange(H, device=device, dtype=dtype)
    px = torch.arange(W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(py, px, indexing="ij")

    # Map pixel coords to cell-grid coords (fractional cell index)
    cell_j = (gx - half_P) / max(S, 1)  # fractional column in cell grid
    cell_i = (gy - half_P) / max(S, 1)  # fractional row in cell grid

    # Normalise to [-1, 1] for grid_sample (align_corners=True)
    norm_x = 2.0 * cell_j / max(nW - 1, 1) - 1.0
    norm_y = 2.0 * cell_i / max(nH - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    out = F.grid_sample(
        field_4d, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )

    if field_grid.ndim == 2:
        return out.squeeze(0).squeeze(0)  # (H, W)
    else:
        return out.squeeze(0).permute(1, 2, 0)  # (H, W, C)


# ═══════════════════════════════════════════════════════════════
# Thinning head: MLP on 14-D stencil + contrast features (HCI readout)
# ═══════════════════════════════════════════════════════════════

class ThinningHead(nn.Module):
    """14 → hidden → 1 gate: σ(W₂ ReLU(W₁ F + b₁) + b₂).

    ``F = [h2m_lum, h2m_chr, ρ̄, κ̄_col, tang₋₂…tang₂, norm₋₂…norm₂]``.
    Init near-identity gate (``b₂=2`` on last layer, mild first-row coupling).
    """

    def __init__(self, in_dim: int = 14, hidden: int = 8):
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
            self.fc2.bias.fill_(2.0)
            for j in range(4):
                self.fc1.weight[0, j] = 0.05
            self.fc1.weight[:, 4:].normal_(0.0, 0.02)
            self.fc2.weight[0, 0] = 0.1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, in_dim). Returns: (N,) gate in (0, 1)."""
        h = F.relu(self.fc1(features))
        return torch.sigmoid(self.fc2(h).squeeze(-1))


# ═══════════════════════════════════════════════════════════════
# ModulationRenderer — harmonic-native
# ═══════════════════════════════════════════════════════════════

class ModulationRenderer(nn.Module):
    """Contour renderer: ``(h2m_lum+h2m_chr)·ρ̄·gate`` with 14→8→1 ThinningHead.

    Learned tangential / normal stencil spacings ``s_t``, ``s_n`` (softplus).
    Cell-grid κ_col comes from L1 (``cells_flat``).
    """

    def __init__(self, hidden: int | None = None, **kwargs):
        super().__init__()
        h = int(hidden if hidden is not None else RENDER.PIXEL_HIDDEN)
        self.thinning = ThinningHead(in_dim=14, hidden=max(h, 8))
        self._s_t_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0), dtype=torch.float32))
        self._s_n_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0), dtype=torch.float32))

    @property
    def s_t(self) -> torch.Tensor:
        return F.softplus(self._s_t_raw)

    @property
    def s_n(self) -> torch.Tensor:
        return F.softplus(self._s_n_raw)

    # Legacy compat — code that reads sigma_perp for diagnostics
    @property
    def sigma_perp(self) -> torch.Tensor:
        return torch.tensor(1.0, dtype=torch.float32)
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
    """Best-effort upgrade from old splat- or stencil-based state dicts.

    Drops ``thinning.*`` tensors whose shapes do not match the current 14→8→1
    head (legacy 4→4 checkpoints then re-init the gate from ``ThinningHead``
    defaults).
    """
    state_dict = dict(state_dict)
    wkey = f"{prefix}thinning.fc1.weight"
    if wkey in state_dict and tuple(state_dict[wkey].shape) != (8, 14):
        for k in list(state_dict):
            if f"{prefix}thinning." in k:
                del state_dict[k]

    remove = {
        f"{prefix}_sigma_par_raw",
        f"{prefix}_sigma_pre_raw",
        f"{prefix}_eta_h_raw",
        f"{prefix}_smooth_sigma_raw",
        f"{prefix}_sigma_perp_raw",
        f"{prefix}perp_conv.conv.weight",
        f"{prefix}perp_conv.conv.bias",
        f"{prefix}perp_conv.fc.weight",
        f"{prefix}perp_conv.fc.bias",
        f"{prefix}s_t",
        f"{prefix}s_n",
    }
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
        # Drop thinning tensors with unknown shapes (fc1 is upgraded above).
        if "thinning." in k and v.shape != _expected_shape(k):
            continue
        out[k] = v
    return out


def _expected_shape(key: str):
    """Expected shapes for the 14→8→1 thinning head."""
    shapes = {
        "thinning.fc1.weight": (8, 14),
        "thinning.fc1.bias": (8,),
        "thinning.fc2.weight": (1, 8),
        "thinning.fc2.bias": (1,),
    }
    for k, s in shapes.items():
        if key.endswith(k):
            return s
    return None


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


# ═══════════════════════════════════════════════════════════════
# render_boundary_map_torch — tang/norm stencils + 14-D gate
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
    return_rho_cell_grids: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):

    _ = training
    want_cell_grids = bool(return_rho_cell_grids)
    H, W = proj_dev["H"], proj_dev["W"]
    device, dtype = rho_cell.device, rho_cell.dtype

    nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
    n_cells = int(proj_dev["n_cells"])
    theta_all = cells_flat["theta"].to(device=device, dtype=dtype)
    theta_c = _theta_on_branch(theta_all, branch_pick, n_cells, device)
    S = int(cells_flat.get("S", max(1, W // max(nW, 1))))
    P = int(cells_flat.get("P", S + (S - 1)))  # reconstruct P from context

    # ── Step 1: Cell-grid θ combing ──────────────────────────
    rho_grid = rho_cell.reshape(nH, nW).to(dtype=dtype)
    ib_grid = cells_flat["is_border"].to(device=device).reshape(nH, nW).bool()
    theta_combed = _smooth_theta_rho_double_angle(
        theta_c.reshape(nH, nW), rho_grid, ib_grid, eps=eps,
    )

    is_border = cells_flat["is_border"].to(device=device).reshape(-1).bool()
    rho_flat = rho_cell.reshape(-1).to(dtype=dtype)

    active = (rho_flat > 0) & (~is_border)
    if not active.any().item():
        z = (rho_cell.sum() * 0.0).to(dtype=dtype, device=device)
        out = z.expand(H, W)[:Hp, :Wp]
        th_z = torch.zeros_like(out)
        zs_grid = torch.zeros(nH, nW, device=device, dtype=dtype)
        if return_dominant_theta and want_cell_grids:
            return out, th_z, zs_grid, zs_grid
        if return_dominant_theta:
            return out, th_z
        if return_rho_cell_grids:
            return out, zs_grid, zs_grid
        return out

    # ── Step 2: L1-supplied κ_col (cell grid); ρ unchanged here ──
    kappa_col_grid = cells_flat["kappa_col_cell"].to(
        device=device, dtype=dtype,
    ).reshape(nH, nW)
    rho_mod_grid = rho_grid.detach()

    rho_seed_cell = torch.where(ib_grid, torch.zeros_like(rho_grid), rho_grid)

    # ── Step 3: Interpolate cell-grid fields to pixel res ────
    # theta_combed is detached — orientation is a geometric feature, not
    # a learned signal.  Gradient flows through ρ̄, h2m, and the thinning MLP.
    theta_combed_det = theta_combed.detach()
    rho_grid_m = torch.where(ib_grid, torch.zeros_like(rho_mod_grid), rho_mod_grid)
    theta_cos2 = torch.where(ib_grid, torch.zeros_like(theta_combed_det),
                              rho_grid_m * torch.cos(2.0 * theta_combed_det))
    theta_sin2 = torch.where(ib_grid, torch.zeros_like(theta_combed_det),
                              rho_grid_m * torch.sin(2.0 * theta_combed_det))

    # Stack fields for a single grid_sample call: (nH, nW, C)
    # channels: [ρ, cos2θ·ρ, sin2θ·ρ, κ_col]
    cell_stack = torch.stack([
        rho_grid_m,
        theta_cos2,
        theta_sin2,
        kappa_col_grid,
    ], dim=-1)  # (nH, nW, 4)

    pix_stack = _interp_cell_to_pixel(cell_stack, nH, nW, H, W, S, P)
    # (H, W, 4)

    rho_pix = pix_stack[..., 0]
    # Recover θ from interpolated double-angle (handles π-wraparound)
    cos2_pix = pix_stack[..., 1]
    sin2_pix = pix_stack[..., 2]
    # Normalise by interpolated ρ to get unit double-angle
    rho_pix_safe = rho_pix.clamp_min(eps)
    cos2_norm = cos2_pix / rho_pix_safe
    sin2_norm = sin2_pix / rho_pix_safe
    theta_pix = 0.5 * torch.atan2(sin2_norm, cos2_norm)

    kappa_col_pix = pix_stack[..., 3]

    # ── Step 4: Pixel-native h2m fields ──────────────────────
    if l0_pix is not None and "h2m_lum" in l0_pix:
        h2m_lum = l0_pix["h2m_lum"].to(device=device, dtype=dtype)
        if h2m_lum.shape != (H, W):
            h2m_lum = h2m_lum[:H, :W]
    else:
        h2m_lum = torch.zeros(H, W, device=device, dtype=dtype)

    if l0_pix is not None and "h2m_chr" in l0_pix:
        h2m_chr = l0_pix["h2m_chr"].to(device=device, dtype=dtype)
        if h2m_chr.shape != (H, W):
            h2m_chr = h2m_chr[:H, :W]
    else:
        h2m_chr = torch.zeros(H, W, device=device, dtype=dtype)

    # Combined h2m as the base edge signal (pixel-native, smooth)
    h2m_combined = h2m_lum + h2m_chr

    # ── Step 5: Tangential / normal h2m stencils → 14-D gate ──
    theta_geom = theta_pix.detach()
    h4 = h2m_combined.unsqueeze(0).unsqueeze(0)
    st = renderer.s_t.to(device=device, dtype=dtype)
    sn = renderer.s_n.to(device=device, dtype=dtype)
    py = torch.arange(H, device=device, dtype=dtype)
    px = torch.arange(W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(py, px, indexing="ij")
    c = torch.cos(theta_geom)
    s = torch.sin(theta_geom)
    denom_x = max(W - 1, 1)
    denom_y = max(H - 1, 1)
    tang_parts: list[torch.Tensor] = []
    norm_parts: list[torch.Tensor] = []
    for j in (-2, -1, 0, 1, 2):
        fj = float(j)
        sx_t = gx + fj * st * c
        sy_t = gy + fj * st * s
        grid_t = torch.stack(
            [2.0 * sx_t / denom_x - 1.0, 2.0 * sy_t / denom_y - 1.0], dim=-1,
        ).unsqueeze(0)
        tang_parts.append(
            F.grid_sample(
                h4, grid_t, mode="bilinear", padding_mode="border", align_corners=True,
            ).squeeze(0).squeeze(0),
        )
        sx_n = gx + fj * sn * (-s)
        sy_n = gy + fj * sn * c
        grid_n = torch.stack(
            [2.0 * sx_n / denom_x - 1.0, 2.0 * sy_n / denom_y - 1.0], dim=-1,
        ).unsqueeze(0)
        norm_parts.append(
            F.grid_sample(
                h4, grid_n, mode="bilinear", padding_mode="border", align_corners=True,
            ).squeeze(0).squeeze(0),
        )
    tang_stack = torch.stack(tang_parts, dim=-1)
    norm_stack = torch.stack(norm_parts, dim=-1)

    theta_pix = theta_geom  # geometric; grad through ρ̄, h2m, gate, stencils

    fdim = int(renderer.thinning.in_dim)
    features = torch.cat([
        h2m_lum.unsqueeze(-1),
        h2m_chr.unsqueeze(-1),
        rho_pix.unsqueeze(-1),
        kappa_col_pix.unsqueeze(-1),
        tang_stack,
        norm_stack,
    ], dim=-1)

    if features.shape[-1] != fdim:
        raise RuntimeError(
            f"feature dim {features.shape[-1]} != thinning.in_dim {fdim}"
        )
    feat_flat = features.reshape(H * W, fdim)
    gate = renderer.thinning(feat_flat).reshape(H, W)
    bmap = h2m_combined * rho_pix * gate

    # ── Crop ─────────────────────────────────────────────────
    ch = Hp if content_h is None else content_h
    cw = Wp if content_w is None else content_w
    ch, cw = min(ch, H), min(cw, W)
    if content_h is not None and content_w is not None:
        crop = torch.ones_like(bmap)
        if ch < H: crop[ch:, :] = 0.0
        if cw < W: crop[:, cw:] = 0.0
        bmap = bmap * crop
        theta_pix = theta_pix * crop

    out = bmap[:Hp, :Wp]
    th_out = theta_pix[:Hp, :Wp]
    if return_dominant_theta and want_cell_grids:
        return out, th_out, rho_seed_cell, rho_grid_m
    if return_dominant_theta:
        return out, th_out
    if want_cell_grids:
        return out, rho_seed_cell, rho_grid_m
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