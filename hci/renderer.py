r"""Renderer — back-projection with learned 1D kernels"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from params import L1, RENDER


_DEPOSIT_HALF_WIDTH_STRIDES = float(getattr(RENDER, "DEPOSIT_HALF_WIDTH_STRIDES", 2.0))
_DEPOSIT_HALF_WIDTH_MIN = int(getattr(RENDER, "DEPOSIT_HALF_WIDTH_MIN", 4))
_DEPOSIT_HALF_WIDTH_MAX = int(getattr(RENDER, "DEPOSIT_HALF_WIDTH_MAX", 24))

_DEFAULT_S = max(1, int(getattr(L1, "PATCH_SIZE", 5)) - int(getattr(L1, "PATCH_OVERLAP", 3)))


def _deposit_half_width(S: int) -> int:
    h = int(math.ceil(_DEPOSIT_HALF_WIDTH_STRIDES * max(S, 1)))
    return max(_DEPOSIT_HALF_WIDTH_MIN, min(_DEPOSIT_HALF_WIDTH_MAX, h))


_KERNEL_H_W = _deposit_half_width(_DEFAULT_S)

_FEATURE_DIM = 5
_HIDDEN_DIM = int(getattr(RENDER, "CORR_HIDDEN", 12))
_OUT_DIM = 4

_SIGMA_PERP_INIT = float(getattr(RENDER, "SIGMA_PERP_INIT", 0.6))
_SIGMA_PAR_INIT  = float(getattr(RENDER, "SIGMA_PAR_INIT", 2.0))
_RAMP_CUTOFF_INIT = float(getattr(RENDER, "RAMP_CUTOFF_INIT", 0.5))

_KAPPA_MAX_INIT    = float(getattr(RENDER, "KAPPA_MAX_INIT", 0.1))
_EXT_MAX_INIT      = float(getattr(RENDER, "EXT_MAX_INIT", 1.0))
_DELTA_N_MAX_INIT  = float(getattr(RENDER, "DELTA_N_MAX_INIT", 1.0))
_ALPHA_RANGE_INIT  = float(getattr(RENDER, "ALPHA_RANGE_INIT", 0.5))

_BIN_GATE_TEMP_INIT = float(getattr(RENDER, "BIN_GATE_TEMP_INIT", 0.08))

_GATE_ACTIVE_THRESHOLD = 1e-3
_RHO_ACTIVE_FLOOR = 1e-4

_CLAIM_CLIP = 1.0 - 1e-5

_FEAT_SOFTFLOOR = 5e-2


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


def _init_phi_perp_raw_hann(kernel_h_w: int, omega_c: float) -> torch.Tensor:
    H = kernel_h_w
    L = 2 * H + 1
    oc = max(min(float(omega_c), 0.5), 1e-3)
    mags = torch.arange(H + 1, dtype=torch.float32) / L
    ratio = (mags / oc).clamp(max=1.0)
    W = torch.cos(0.5 * math.pi * ratio).clamp(min=0.0) ** 2
    decay = (W[1:] / W[:-1].clamp(min=1e-8)).clamp(1e-6, 1.0 - 1e-6)
    raw = torch.empty(H + 1, dtype=torch.float32)
    raw[0] = _inv_softplus(float(W[0]))
    raw[1:] = torch.log(decay / (1.0 - decay))
    return raw


def _windowed_ramp_from_logits(
    kernel_h_w: int,
    phi_perp_raw: torch.Tensor,
) -> torch.Tensor:
    H = kernel_h_w
    device = phi_perp_raw.device
    dtype = phi_perp_raw.dtype
    raw = phi_perp_raw
    L = 2 * H + 1

    W0 = F.softplus(raw[0:1])
    decay = torch.sigmoid(raw[1:])
    W_half = W0 * torch.cat([
        torch.ones(1, device=device, dtype=dtype),
        torch.cumprod(decay, dim=0),
    ], dim=0)

    idx_abs = torch.arange(-H, H + 1, device=device).abs()
    W_full = W_half[idx_abs]
    omega = torch.arange(-H, H + 1, device=device, dtype=dtype) / L
    H_omega = omega.abs() * W_full

    n_idx = torch.arange(-H, H + 1, device=device, dtype=dtype)
    cos_basis = torch.cos(2.0 * math.pi * omega.unsqueeze(0) * n_idx.unsqueeze(1))
    return (H_omega.unsqueeze(0) * cos_basis).sum(dim=1) / L


def _per_bin_features(
    rho_bins: torch.Tensor,
    bar_theta: torch.Tensor,
    ib_g: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    _ = eps
    nH, nW, K = rho_bins.shape
    dtype = rho_bins.dtype

    cos_b = torch.cos(bar_theta).view(1, 1, K)
    sin_b = torch.sin(bar_theta).view(1, 1, K)

    def _pad0_hwk(t: torch.Tensor) -> torch.Tensor:
        x = t.permute(2, 0, 1).unsqueeze(0)
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        return x.squeeze(0).permute(1, 2, 0)

    rho_p = _pad0_hwk(rho_bins)

    sum_pos_rho     = torch.zeros_like(rho_bins)
    sum_pos         = torch.zeros_like(rho_bins)
    sum_t_rho       = torch.zeros_like(rho_bins)
    sum_abs_t_rho   = torch.zeros_like(rho_bins)
    sum_n_rho       = torch.zeros_like(rho_bins)
    sum_abs_n_rho   = torch.zeros_like(rho_bins)

    sum_all_bins = rho_bins.sum(dim=-1, keepdim=True)
    f3 = sum_all_bins - rho_bins

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sl = (slice(1 + dy, 1 + dy + nH), slice(1 + dx, 1 + dx + nW))
            rn = rho_p[sl]
            t_proj =  float(dy) * cos_b + float(dx) * sin_b
            n_proj = -float(dy) * sin_b + float(dx) * cos_b
            d2 = float(dy * dy + dx * dx)
            pos_t = (t_proj * t_proj) / d2

            sum_pos_rho   = sum_pos_rho   + rn * pos_t
            sum_pos       = sum_pos       + pos_t.expand_as(rn)
            sum_t_rho     = sum_t_rho     + rn * t_proj
            sum_abs_t_rho = sum_abs_t_rho + rn * t_proj.abs()
            sum_n_rho     = sum_n_rho     + rn * n_proj
            sum_abs_n_rho = sum_abs_n_rho + rn * n_proj.abs()

    f1 = sum_pos_rho / (sum_pos       + _FEAT_SOFTFLOOR)
    f2 = sum_t_rho   / (sum_abs_t_rho + _FEAT_SOFTFLOOR)
    f4 = sum_n_rho   / (sum_abs_n_rho + _FEAT_SOFTFLOOR)

    use = (~ib_g).to(dtype=dtype).unsqueeze(-1)
    f0 = rho_bins * use
    f1 = f1 * use
    f2 = f2 * use
    f3 = f3 * use
    f4 = f4 * use

    return torch.stack([f0, f1, f2, f3, f4], dim=-1)


class _InterpKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, h: torch.Tensor, u: torch.Tensor, H_w: int) -> torch.Tensor:
        L = 2 * H_w
        u_floor = torch.floor(u)
        u_frac = u - u_floor
        m_lo = u_floor.to(torch.long) + H_w
        m_hi = m_lo + 1
        out_range = (m_lo < 0) | (m_hi > L)
        m_lo.clamp_(0, L)
        m_hi.clamp_(0, L)
        val = h[m_lo] + u_frac * (h[m_hi] - h[m_lo])
        val = val.masked_fill(out_range, 0.0)
        ctx.save_for_backward(h, u)
        ctx.H_w = H_w
        return val

    @staticmethod
    def backward(ctx, grad_val: torch.Tensor):
        h, u = ctx.saved_tensors
        H_w = ctx.H_w
        L = 2 * H_w

        u_floor = torch.floor(u)
        u_frac = u - u_floor
        m_lo = u_floor.to(torch.long) + H_w
        m_hi = m_lo + 1
        out_range = (m_lo < 0) | (m_hi > L)
        m_lo.clamp_(0, L)
        m_hi.clamp_(0, L)

        gv = grad_val.masked_fill(out_range, 0.0)

        grad_h = None
        grad_u = None
        if ctx.needs_input_grad[1]:
            grad_u = gv * (h[m_hi] - h[m_lo])
        if ctx.needs_input_grad[0]:
            grad_h = torch.zeros_like(h)
            w_lo = (1.0 - u_frac) * gv
            w_hi = u_frac * gv
            grad_h.scatter_add_(0, m_lo.reshape(-1), w_lo.reshape(-1))
            grad_h.scatter_add_(0, m_hi.reshape(-1), w_hi.reshape(-1))
        return grad_h, grad_u, None


def _interp_kernel(h: torch.Tensor, u: torch.Tensor, H_w: int) -> torch.Tensor:
    return _InterpKernel.apply(h, u, H_w)


def _chunk_log_neg(
    rho_c: torch.Tensor,
    gate_c: torch.Tensor,
    alpha_c: torch.Tensor,
    ax_f: torch.Tensor,
    ay_f: torch.Tensor,
    ca: torch.Tensor,
    sa: torch.Tensor,
    kappa_c: torch.Tensor,
    ext_s_c: torch.Tensor,
    ox_l: torch.Tensor,
    oy_l: torch.Tensor,
    h_perp_l: torch.Tensor,
    h_par_l: torch.Tensor,
    in_bounds_l: torch.Tensor,
    half_w: int,
) -> torch.Tensor:
    dx = ox_l.unsqueeze(0) - ax_f
    dy = oy_l.unsqueeze(0) - ay_f

    s =  dy * ca + dx * sa
    n = -dy * sa + dx * ca

    s_tilde = s - ext_s_c.unsqueeze(1)
    n_curv  = n - 0.5 * kappa_c.unsqueeze(1) * (s_tilde * s_tilde)

    h_perp_val = _interp_kernel(h_perp_l, n_curv,  half_w)
    h_par_val  = _interp_kernel(h_par_l,  s_tilde, half_w)

    f_val = F.relu(h_perp_val * h_par_val)

    amp = (alpha_c * gate_c * rho_c).unsqueeze(1)
    claim = (amp * f_val).clamp(min=0.0, max=_CLAIM_CLIP)
    claim_safe = torch.where(in_bounds_l, claim, torch.zeros_like(claim))
    return torch.log1p(-claim_safe)


def _backproject_deposit(
    rho_active: torch.Tensor,
    gate_active: torch.Tensor,
    alpha_active: torch.Tensor,
    ax_active: torch.Tensor,
    ay_active: torch.Tensor,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    kappa_active: torch.Tensor,
    ext_s_active: torch.Tensor,
    delta_n_active: torch.Tensor,
    h_perp: torch.Tensor,
    h_par: torch.Tensor,
    kernel_h_w: int,
    H: int, W: int,
    eps: float = 1e-6,
    use_checkpoint: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = rho_active.shape[0]
    device, dtype = rho_active.device, rho_active.dtype
    n_pix = H * W

    if A == 0:
        z = torch.zeros(H, W, device=device, dtype=dtype)
        return z, z

    half_w = kernel_h_w
    offsets = torch.arange(-half_w, half_w + 1, device=device, dtype=dtype)
    oy_g, ox_g = torch.meshgrid(offsets, offsets, indexing="ij")
    oy = oy_g.reshape(-1)
    ox = ox_g.reshape(-1)
    P = oy.shape[0]

    log_neg_acc = torch.zeros(n_pix, device=device, dtype=dtype)
    mom_re = torch.zeros(n_pix, device=device, dtype=torch.float32)
    mom_im = torch.zeros(n_pix, device=device, dtype=torch.float32)

    ax_eff = ax_active + delta_n_active * cos_a
    ay_eff = ay_active - delta_n_active * sin_a

    ax_int = torch.floor(ax_eff).long()
    ay_int = torch.floor(ay_eff).long()
    ax_frac = ax_eff - ax_int.to(dtype=dtype)
    ay_frac = ay_eff - ay_int.to(dtype=dtype)

    bar_theta_active = torch.atan2(sin_a, cos_a)

    max_batch = max(1, 4_000_000 // P)

    do_ckpt = bool(use_checkpoint) and torch.is_grad_enabled()

    for b0 in range(0, A, max_batch):
        b1 = min(b0 + max_batch, A)
        bs = b1 - b0

        ax_f = ax_frac[b0:b1].unsqueeze(1)
        ay_f = ay_frac[b0:b1].unsqueeze(1)
        ca = cos_a[b0:b1].unsqueeze(1)
        sa = sin_a[b0:b1].unsqueeze(1)

        px = ax_int[b0:b1].unsqueeze(1) + ox.unsqueeze(0).long()
        py = ay_int[b0:b1].unsqueeze(1) + oy.unsqueeze(0).long()
        in_bounds = (py >= 0) & (py < H) & (px >= 0) & (px < W)
        flat_idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1))

        chunk_args = (
            rho_active[b0:b1], gate_active[b0:b1], alpha_active[b0:b1],
            ax_f, ay_f, ca, sa,
            kappa_active[b0:b1], ext_s_active[b0:b1],
            ox, oy,
            h_perp, h_par,
            in_bounds, half_w,
        )
        if do_ckpt:
            log_neg = checkpoint(_chunk_log_neg, *chunk_args, use_reentrant=False)
        else:
            log_neg = _chunk_log_neg(*chunk_args)

        log_neg_acc.scatter_add_(0, flat_idx.reshape(-1), log_neg.reshape(-1))

        with torch.no_grad():
            cl_det = (-torch.expm1(log_neg.detach())).to(torch.float32).reshape(-1)
            idx_flat = flat_idx.reshape(-1)
            cos2 = torch.cos(2.0 * bar_theta_active[b0:b1])
            sin2 = torch.sin(2.0 * bar_theta_active[b0:b1])
            cos2_expand = cos2.unsqueeze(1).expand(bs, P).reshape(-1)
            sin2_expand = sin2.unsqueeze(1).expand(bs, P).reshape(-1)
            mom_re.scatter_add_(0, idx_flat, cl_det * cos2_expand)
            mom_im.scatter_add_(0, idx_flat, cl_det * sin2_expand)

    bmap = -torch.expm1(log_neg_acc)
    theta_star = (0.5 * torch.atan2(mom_im, mom_re)).to(dtype=dtype)
    return bmap.reshape(H, W), theta_star.reshape(H, W)


class CorrectionMLP(nn.Module):

    def __init__(
        self,
        in_dim: int = _FEATURE_DIM,
        hidden_dim: int = _HIDDEN_DIM,
        out_dim: int = _OUT_DIM,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
            self.fc1.bias.zero_()
            nn.init.normal_(self.fc2.weight, std=0.01)
            self.fc2.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(features))
        return self.fc2(h)


class ModulationRenderer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        _ = kwargs

        self._kernel_h_w = int(_KERNEL_H_W)
        H = self._kernel_h_w

        self.correction = CorrectionMLP()

        self._phi_perp_raw = nn.Parameter(
            _init_phi_perp_raw_hann(H, _RAMP_CUTOFF_INIT)
        )

        sig_a = max(_SIGMA_PAR_INIT, 1e-6)
        ratios_init = torch.tensor([
            math.exp(-(2 * j + 1) / (2.0 * sig_a * sig_a))
            for j in range(H)
        ], dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        psi_init_raw = torch.log(ratios_init / (1.0 - ratios_init))
        self._psi_par_raw = nn.Parameter(psi_init_raw)
        self._h_par_peak_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0)))

        self._kappa_max_raw   = nn.Parameter(torch.tensor(_inv_softplus(_KAPPA_MAX_INIT)))
        self._ext_max_raw     = nn.Parameter(torch.tensor(_inv_softplus(_EXT_MAX_INIT)))
        self._delta_n_max_raw = nn.Parameter(torch.tensor(_inv_softplus(_DELTA_N_MAX_INIT)))
        self._alpha_range_raw = nn.Parameter(torch.tensor(_inv_softplus(_ALPHA_RANGE_INIT)))

        self._gate_temp_raw = nn.Parameter(
            torch.tensor(_inv_softplus(_BIN_GATE_TEMP_INIT))
        )


    @property
    def h_perp(self) -> torch.Tensor:
        return _windowed_ramp_from_logits(self._kernel_h_w, self._phi_perp_raw)

    @property
    def h_par(self) -> torch.Tensor:
        H = self._kernel_h_w
        peak = F.softplus(self._h_par_peak_raw)
        ratios = torch.sigmoid(self._psi_par_raw)
        cumprod = torch.cumprod(ratios, dim=0)
        tilde_h = torch.cat([
            torch.ones(1, device=ratios.device, dtype=ratios.dtype),
            cumprod,
        ], dim=0)
        idx = torch.arange(-H, H + 1, device=ratios.device).abs()
        return peak * tilde_h[idx]


    @property
    def kappa_max(self) -> torch.Tensor:
        return F.softplus(self._kappa_max_raw).view(())

    @property
    def ext_max(self) -> torch.Tensor:
        return F.softplus(self._ext_max_raw).view(())

    @property
    def delta_n_max(self) -> torch.Tensor:
        return F.softplus(self._delta_n_max_raw).view(())

    @property
    def alpha_range(self) -> torch.Tensor:
        return F.softplus(self._alpha_range_raw).view(())

    @property
    def gate_temp(self) -> torch.Tensor:
        return F.softplus(self._gate_temp_raw).view(())

    @property
    def kernel_h_w(self) -> int:
        return int(self._kernel_h_w)


    @property
    def sigma_perp(self) -> torch.Tensor:
        return torch.tensor(_SIGMA_PERP_INIT, dtype=torch.float32)

    @property
    def sigma_par(self) -> torch.Tensor:
        return torch.tensor(_SIGMA_PAR_INIT, dtype=torch.float32)

    @property
    def ext_scale(self) -> torch.Tensor:
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


def upgrade_renderer_state_dict(state_dict: dict, prefix: str = "") -> dict:
    legacy = {
        f"{prefix}_ramp_cutoff_raw",
        f"{prefix}_ramp_rolloff_raw",
        f"{prefix}_ramp_ram_lak_raw",
        f"{prefix}_ramp_gain_raw",
        f"{prefix}_sigma_perp_raw",
        f"{prefix}_sigma_par_raw",
        f"{prefix}_sigma_pre_raw",
        f"{prefix}_eta_h_raw",
        f"{prefix}_smooth_sigma_raw",
        f"{prefix}_ext_scale_raw",
        f"{prefix}s_t",
        f"{prefix}s_n",
        f"{prefix}perp_conv.conv.weight",
        f"{prefix}perp_conv.conv.bias",
        f"{prefix}perp_conv.fc.weight",
        f"{prefix}perp_conv.fc.bias",
        f"{prefix}thinning.fc1.weight",
        f"{prefix}thinning.fc1.bias",
        f"{prefix}thinning.fc2.weight",
        f"{prefix}thinning.fc2.bias",
        f"{prefix}deposit.fc1.weight",
        f"{prefix}deposit.fc1.bias",
        f"{prefix}deposit.fc2.weight",
        f"{prefix}deposit.fc2.bias",
    }
    expected_fc1_w = (_HIDDEN_DIM, _FEATURE_DIM)
    expected_fc1_b = (_HIDDEN_DIM,)
    expected_fc2_w = (_OUT_DIM, _HIDDEN_DIM)
    expected_fc2_b = (_OUT_DIM,)
    out = {}
    for k, v in state_dict.items():
        if k in legacy:
            continue
        if k == f"{prefix}correction.fc1.weight" and tuple(v.shape) != expected_fc1_w:
            continue
        if k == f"{prefix}correction.fc1.bias"   and tuple(v.shape) != expected_fc1_b:
            continue
        if k == f"{prefix}correction.fc2.weight" and tuple(v.shape) != expected_fc2_w:
            continue
        if k == f"{prefix}correction.fc2.bias"   and tuple(v.shape) != expected_fc2_b:
            continue
        out[k] = v
    return out


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
    _ = (rho_cell, training, l0_pix, branch_pick)
    H, W = proj_dev["H"], proj_dev["W"]
    device = next(renderer.parameters()).device
    dtype = renderer.h_perp.dtype

    if "rho_out_bins" not in cells_flat:
        raise ValueError(
            "Renderer requires cells_flat['rho_out_bins'] from the seed."
        )
    if "ax_bin" not in cells_flat or "ay_bin" not in cells_flat:
        raise ValueError("cells_flat must include ax_bin and ay_bin (from L1).")
    if "theta_bins" not in cells_flat:
        raise ValueError("cells_flat must include theta_bins (from L1).")

    nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
    N = nH * nW

    rho_out_bins = cells_flat["rho_out_bins"].to(device=device, dtype=dtype)
    if rho_out_bins.dim() == 2:
        rho_out_bins = rho_out_bins.reshape(nH, nW, -1)
    K = rho_out_bins.shape[-1]

    ax_bin = cells_flat["ax_bin"].to(device=device, dtype=dtype)
    ay_bin = cells_flat["ay_bin"].to(device=device, dtype=dtype)
    if ax_bin.dim() == 3:
        ax_bin = ax_bin.reshape(N, K)
        ay_bin = ay_bin.reshape(N, K)

    bar_theta = cells_flat["theta_bins"].to(device=device, dtype=dtype).reshape(K)
    is_border = cells_flat["is_border"].to(device=device).reshape(N).bool()
    ib_g = is_border.reshape(nH, nW)

    feats = _per_bin_features(rho_out_bins.detach(), bar_theta, ib_g, eps=eps)
    feats_flat = feats.reshape(N * K, _FEATURE_DIM)

    rho_flat_bins = rho_out_bins.reshape(N, K)
    rho_max_cell = rho_flat_bins.max(dim=-1, keepdim=True).values
    rho_rel = rho_flat_bins - rho_max_cell
    ok = (~is_border).to(dtype=dtype).unsqueeze(-1)
    T_gate = renderer.gate_temp.clamp_min(1e-6)
    gate_flat = torch.exp(rho_rel / T_gate) * ok

    gate_NK = gate_flat.reshape(-1)
    rho_NK = rho_flat_bins.reshape(-1)
    keep = (gate_NK > _GATE_ACTIVE_THRESHOLD) & (rho_NK > _RHO_ACTIVE_FLOOR)

    if not bool(keep.any().item()):
        h_perp = renderer.h_perp
        h_par  = renderer.h_par
        zero = (
            h_perp.sum() * 0.0 + h_par.sum() * 0.0
            + renderer.kappa_max * 0.0 + renderer.ext_max * 0.0
            + renderer.delta_n_max * 0.0 + renderer.alpha_range * 0.0
            + renderer.gate_temp * 0.0
            + 0.0 * rho_out_bins.sum()
            + 0.0 * renderer.correction(feats_flat[:1]).sum()
        )
        bmap = torch.zeros(H, W, device=device, dtype=dtype) + zero
        theta_star = torch.zeros(H, W, device=device, dtype=dtype)
        out = bmap[:Hp, :Wp]
        if return_dominant_theta:
            return out, theta_star[:Hp, :Wp]
        return out

    feats_active = feats_flat[keep]
    raw_active = renderer.correction(feats_active)
    kappa_active     = renderer.kappa_max   * torch.tanh(raw_active[:, 0])
    ext_s_active     = renderer.ext_max     * torch.tanh(raw_active[:, 1])
    delta_n_active   = renderer.delta_n_max * torch.tanh(raw_active[:, 2])
    log_alpha_active = renderer.alpha_range * torch.tanh(raw_active[:, 3])
    alpha_active = torch.exp(log_alpha_active)

    bin_idx_full = torch.arange(K, device=device).unsqueeze(0).expand(N, K).reshape(-1)
    active_bin = bin_idx_full[keep]

    ax_active   = ax_bin.reshape(-1)[keep].detach()
    ay_active   = ay_bin.reshape(-1)[keep].detach()
    rho_active  = rho_NK[keep]
    gate_active = gate_NK[keep]

    cos_a = torch.cos(bar_theta).index_select(0, active_bin)
    sin_a = torch.sin(bar_theta).index_select(0, active_bin)

    bmap, theta_star = _backproject_deposit(
        rho_active=rho_active,
        gate_active=gate_active,
        alpha_active=alpha_active,
        ax_active=ax_active,
        ay_active=ay_active,
        cos_a=cos_a,
        sin_a=sin_a,
        kappa_active=kappa_active,
        ext_s_active=ext_s_active,
        delta_n_active=delta_n_active,
        h_perp=renderer.h_perp,
        h_par=renderer.h_par,
        kernel_h_w=renderer._kernel_h_w,
        H=H, W=W, eps=eps,
    )

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


HarmonicThinRenderer = ModulationRenderer
StampRenderer = ModulationRenderer
AnisoDiffusionRenderer = ModulationRenderer