r"""L2 — cell-grid ρ refinement via recurrent Naka–Rushton on convolved geometry.

State: ρ(c, k) — K competing orientation hypotheses per cell.

Seed (IC only; peak-relative bin profile → ρ⁽⁰⁾):
       ρ_seed^(k) = ρ_bins^(k) / (ρ_peak + η_z + ε)   (anisotropy; learned floor η_z)

  Per iteration t (ρ⁽⁰⁾ = ρ_seed; lateral drive only; cross from evolving ρ):
       drive^(k) = b_coll·ρ̃_coll^(k)   (neighborhood consensus, no seed in drive)
       cross^(k) = mean_{k'≠k} (W_disk * ρ^(k',t))   (spatial other-bin pool)
       ρ̃^(k,t+1) = drive² / (drive² + b_iso·c_iso^(k) + b_cross·cross^(k) + η_p² + ε)
       ρ^(k,t+1) = (1−α)ρ^(k,t) + α·ρ̃^(k,t+1)   (α learned, softplus, init 0.5)

  Geometric pools: grouped conv2d over K channels directly (no one-hot scatter).
  Coll: conv2d(ρ, W_coll); iso: conv2d(ρ², W_iso); cross: conv2d(ρ, W_disk); count-normalized.

Learned: b_coll, b_iso, b_cross, η_p, η_z, α
         — **6** nonnegative scalars (softplus; α clamped to [0, 1]).
  Renderer receives max_k ρ^(k) and parabolic θ from final ρ bins.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from params import L2, TRAIN


# ═══════════════════════════════════════════════════════════════
# Conv kernel precompute (fixed geometry, no gradients)
# ═══════════════════════════════════════════════════════════════

def _build_conv_kernels(
    R_fac: int,
    R_sup: int,
    K: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """W_coll, W_iso, W_disk, W_count_coll, W_count — center = 0; coll uses R_fac, rest R_sup."""
    patch_fac = 2 * R_fac + 1
    patch_sup = 2 * R_sup + 1
    dev = device if device is not None else torch.device("cpu")

    bc = torch.linspace(0, math.pi, K + 1, device=dev, dtype=dtype)[:-1]
    cos_b = torch.cos(bc)
    sin_b = torch.sin(bc)

    off_fac = torch.arange(-R_fac, R_fac + 1, device=dev, dtype=dtype)
    di_f, dj_f = torch.meshgrid(off_fac, off_fac, indexing="ij")
    rd_f = (di_f.pow(2) + dj_f.pow(2)).sqrt().clamp_min(1e-12)
    d_i_f = di_f / rd_f
    d_j_f = dj_f / rd_f

    t_dot_d = cos_b[:, None, None] * d_i_f + sin_b[:, None, None] * d_j_f
    W_coll = t_dot_d.abs().unsqueeze(1)
    W_coll[:, :, R_fac, R_fac] = 0.0

    off = torch.arange(-R_sup, R_sup + 1, device=dev, dtype=dtype)
    di, dj = torch.meshgrid(off, off, indexing="ij")
    rd = (di.pow(2) + dj.pow(2)).sqrt().clamp_min(1e-12)
    d_i = di / rd
    d_j = dj / rd

    n_dot_d = -sin_b[:, None, None] * d_i + cos_b[:, None, None] * d_j
    W_iso = (n_dot_d * n_dot_d).unsqueeze(1)

    W_disk = torch.ones(K, 1, patch_sup, patch_sup, device=dev, dtype=dtype)
    W_count_coll = torch.ones(K, 1, patch_fac, patch_fac, device=dev, dtype=dtype)
    W_count = torch.ones(K, 1, patch_sup, patch_sup, device=dev, dtype=dtype)

    W_iso[:, :, R_sup, R_sup] = 0.0
    W_disk[:, :, R_sup, R_sup] = 0.0
    W_count_coll[:, :, R_fac, R_fac] = 0.0
    W_count[:, :, R_sup, R_sup] = 0.0

    return W_coll, W_iso, W_disk, W_count_coll, W_count


def _pool_cell_geometry(
    rho: torch.Tensor,
    ok_map: torch.Tensor,
    W_coll: torch.Tensor,
    W_iso: torch.Tensor,
    W_count_coll: torch.Tensor,
    W_count: torch.Tensor,
    R_fac: int,
    R_sup: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped conv over K-channel ρ; returns (nH, nW, K) coll and iso pools."""
    nH, nW, K = rho.shape
    dtype = rho.dtype
    dev = rho.device

    rho_bins = rho.permute(2, 0, 1).unsqueeze(0)
    rho_sq_bins = rho_bins * rho_bins
    ok_2d = ok_map.to(dtype=dtype, device=dev).reshape(1, 1, nH, nW)
    ok_bins = ok_2d.expand(1, K, nH, nW)

    W_coll = W_coll.to(device=dev, dtype=dtype)
    W_iso = W_iso.to(device=dev, dtype=dtype)
    W_count_coll = W_count_coll.to(device=dev, dtype=dtype)
    W_count = W_count.to(device=dev, dtype=dtype)

    coll_bins = Fn.conv2d(rho_bins, W_coll, padding=R_fac, groups=K)
    iso_bins = Fn.conv2d(rho_sq_bins, W_iso, padding=R_sup, groups=K)
    count_coll_bins = Fn.conv2d(ok_bins, W_count_coll, padding=R_fac, groups=K)
    count_bins = Fn.conv2d(ok_bins, W_count, padding=R_sup, groups=K)

    rho_coll = coll_bins / (count_coll_bins + eps)
    c_iso = iso_bins / (count_bins + eps)

    ok_f = ok_map.to(dtype=dtype, device=dev).unsqueeze(-1)
    rho_coll = rho_coll.squeeze(0).permute(1, 2, 0) * ok_f
    c_iso = c_iso.squeeze(0).permute(1, 2, 0) * ok_f
    return rho_coll, c_iso


def cross_from_rho(
    rho: torch.Tensor,
    ok_map: torch.Tensor,
    W_disk: torch.Tensor,
    W_count: torch.Tensor,
    R_sup: int,
    K: int | None = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Spatial mean other-bin ρ: avg_{k'≠k} count-norm conv(W_disk, ρ^(k'))."""
    nH, nW, k_ch = rho.shape
    if K is None:
        K = k_ch
    km1 = max(int(K) - 1, 1)
    dtype = rho.dtype
    dev = rho.device

    rho_bins = rho.permute(2, 0, 1).unsqueeze(0)
    ok_bins = ok_map.to(dtype=dtype, device=dev).reshape(1, 1, nH, nW).expand(1, K, nH, nW)
    W_disk = W_disk.to(device=dev, dtype=dtype)
    W_count = W_count.to(device=dev, dtype=dtype)

    rho_pooled_bins = Fn.conv2d(rho_bins, W_disk, padding=R_sup, groups=K)
    count_bins = Fn.conv2d(ok_bins, W_count, padding=R_sup, groups=K)
    rho_pooled = rho_pooled_bins / (count_bins + eps)
    pooled_total = rho_pooled.sum(dim=1, keepdim=True)
    cross_bins = (pooled_total - rho_pooled) / km1
    cross = cross_bins.squeeze(0).permute(1, 2, 0)
    return cross * ok_map.unsqueeze(-1).to(dtype=cross.dtype, device=dev)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _inv_softplus(x):
    x = max(float(x), 1e-8)
    if x > 20.0:
        return x  # softplus(y) ≈ y for large y
    return math.log(math.expm1(x))


def rho_seed_from_bins(
    rho_bins: torch.Tensor,
    eta_z: torch.Tensor,
    is_border: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Peak-relative seed: ρ_seed^(k) = ρ_bins^(k) / (ρ_peak + η_z + ε)."""
    if is_border.dim() > 1:
        is_border = is_border.reshape(-1)
    rho_peak = rho_bins.max(dim=-1, keepdim=True).values
    rho_seed = rho_bins / (rho_peak + eta_z + eps)
    mask = is_border.unsqueeze(-1) if is_border.dim() == 1 else is_border.unsqueeze(-1)
    return torch.where(mask, torch.zeros_like(rho_seed), rho_seed)


def collapse_rho_bins(
    rho_bins: torch.Tensor,
    K: int,
    eps: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parabolic sub-bin θ and max_k ρ from final (nH, nW, K) bin masses."""
    dev = device if device is not None else rho_bins.device
    rho_peak, k_star = rho_bins.max(dim=-1)
    k_left = (k_star - 1) % K
    k_right = (k_star + 1) % K
    r_left = rho_bins.gather(-1, k_left.unsqueeze(-1)).squeeze(-1)
    r_right = rho_bins.gather(-1, k_right.unsqueeze(-1)).squeeze(-1)
    denom = 2.0 * rho_peak - r_left - r_right
    frac = (r_left - r_right) / (denom + eps) * 0.5
    frac = frac.clamp(-0.5, 0.5)
    bar_theta = torch.linspace(0, math.pi, K + 1, device=dev, dtype=rho_bins.dtype)[:-1]
    theta = bar_theta[k_star] + frac * (math.pi / K)
    return rho_peak, theta, k_star


def l2_grad_window(
    t_refine: int,
    n_segments: int = TRAIN.L2_SNAPSHOT_MAX,
) -> int:
    """TBPTT detach interval: max(1, T_refine // n_segments)."""
    return max(1, int(t_refine) // max(1, int(n_segments)))


def l2_snapshot_steps(
    t_refine: int,
    max_snapshots: int = TRAIN.L2_SNAPSHOT_MAX,
) -> list[int]:
    """Up to max_snapshots L2 step indices evenly spaced in [0, t_refine] (inclusive)."""
    iters = int(t_refine)
    n = min(int(max_snapshots), iters + 1)
    if n <= 1:
        return [0]
    steps: list[int] = []
    seen: set[int] = set()
    for i in range(n):
        t = int(round(i * iters / (n - 1)))
        t = max(0, min(iters, t))
        if t not in seen:
            seen.add(t)
            steps.append(t)
    if steps[0] != 0:
        steps.insert(0, 0)
    if steps[-1] != iters:
        steps.append(iters)
    return steps


# ═══════════════════════════════════════════════════════════════
# Main module
# ═══════════════════════════════════════════════════════════════

class TileDynamics(nn.Module):
    def __init__(
        self,
        r_fac_pool=L2.R_FAC_POOL,
        r_sup_pool=L2.R_SUP_POOL,
        K=L2.K,
        t_refine=L2.T_REFINE,
        eps=L2.EPS,
        eta_z_init=L2.ETA_Z_INIT,
        alpha_init=L2.ALPHA_INIT,
        tbptt_n_segments: int = TRAIN.L2_SNAPSHOT_MAX,
        **kw,
    ):
        super().__init__()
        _ = kw
        self.R_fac, self.R_sup, self.K = r_fac_pool, r_sup_pool, K
        self.T_refine, self.eps = t_refine, eps
        self.tbptt_n_segments = max(1, int(tbptt_n_segments))

        self._eta_z_raw = nn.Parameter(torch.tensor(_inv_softplus(max(eta_z_init, 1e-6))))
        self._alpha_raw = nn.Parameter(
            torch.tensor(_inv_softplus(float(alpha_init)), dtype=torch.float32)
        )

        self._b_coll_raw = nn.Parameter(torch.tensor(_inv_softplus(L2.B_COLL_INIT)))
        self._b_iso_raw = nn.Parameter(torch.tensor(_inv_softplus(L2.B_ISO_INIT)))
        self._b_cross_raw = nn.Parameter(torch.tensor(_inv_softplus(L2.B_CROSS_INIT)))
        self._eta_p_raw = nn.Parameter(
            torch.tensor(_inv_softplus(float(L2.ETA_P_INIT)), dtype=torch.float32)
        )

        W_coll, W_iso, W_disk, W_count_coll, W_count = _build_conv_kernels(
            self.R_fac, self.R_sup, K,
        )
        self.register_buffer("W_coll", W_coll)
        self.register_buffer("W_iso", W_iso)
        self.register_buffer("W_disk", W_disk)
        self.register_buffer("W_count_coll", W_count_coll)
        self.register_buffer("W_count", W_count)

    @property
    def eta_z(self):
        return Fn.softplus(self._eta_z_raw)

    @property
    def b_coll(self): return Fn.softplus(self._b_coll_raw)
    @property
    def b_iso(self): return Fn.softplus(self._b_iso_raw)
    @property
    def b_cross(self): return Fn.softplus(self._b_cross_raw)
    @property
    def eta_p(self): return Fn.softplus(self._eta_p_raw)

    @property
    def alpha(self):
        return Fn.softplus(self._alpha_raw).clamp(0.0, 1.0)

    @property
    def grad_window(self) -> int:
        return l2_grad_window(self.T_refine, self.tbptt_n_segments)

    def _record_geometry_snapshot(
        self,
        store: dict[str, dict[str, np.ndarray]],
        tag: str,
        rho_coll: torch.Tensor,
        c_iso: torch.Tensor,
        rho_3d: torch.Tensor,
        ok_map: torch.Tensor,
    ) -> None:
        """Scalar cell maps: max-K coll/iso/cross; max-K ρ (renderer input)."""
        ok_f = ok_map.to(dtype=rho_3d.dtype, device=rho_3d.device)
        rho_peak = rho_3d.max(dim=-1).values * ok_f
        cross = cross_from_rho(
            rho_3d, ok_map, self.W_disk, self.W_count,
            self.R_sup, self.K, self.eps,
        )
        c_cross_peak = cross.max(dim=-1).values * ok_f
        store[tag] = {
            k: v.detach().cpu().numpy().astype(np.float64)
            for k, v in (
                ("rho_coll", rho_coll.max(dim=-1).values * ok_f),
                ("c_iso", c_iso.max(dim=-1).values * ok_f),
                ("c_cross", c_cross_peak),
                ("rho_peak", rho_peak),
            )
        }

    def _iterate_once(
        self,
        rho_prev,
        rho_coll,
        c_iso,
        cross,
        ok_map,
        return_drive_terms: bool = False,
    ):
        drive = self.b_coll * rho_coll
        drive_sq = drive * drive
        eta_p_sq = self.eta_p * self.eta_p
        inhib = self.b_iso * c_iso + self.b_cross * cross + eta_p_sq
        ok_k = ok_map.unsqueeze(-1)
        rho_nr = (drive_sq / (drive_sq + inhib + self.eps)) * ok_k
        a = self.alpha
        rho_next = (1.0 - a) * rho_prev + a * rho_nr

        if return_drive_terms:
            with torch.no_grad():
                m = ok_map > 0
                def _stat(t):
                    if not m.any():
                        return 0.0, 0.0
                    x = t[m] if t.dim() == 2 else t[m]
                    return float(x.mean()), float(x.max())
                return rho_next, {
                    "rho_coll_mean": _stat(rho_coll)[0],
                    "c_iso_mean": _stat(c_iso)[0],
                    "c_cross_mean": _stat(cross)[0],
                    "drive_mean": _stat(drive)[0],
                    "drive_sq_mean": _stat(drive_sq)[0],
                    "inhib_mean": _stat(inhib)[0],
                    "rho_nr_mean": _stat(rho_nr)[0],
                    "rho_next_mean": _stat(rho_next)[0],
                    "alpha": float(a.detach()),
                    "term_coll_mean": _stat(self.b_coll * rho_coll)[0],
                    "term_iso_mean": _stat(self.b_iso * c_iso)[0],
                    "term_cross_mean": _stat(self.b_cross * cross)[0],
                    "eta_p": float(self.eta_p.detach()),
                }
        return rho_next

    def forward(self, cells_flat, return_surface_diags=False, **kw):
        device = next(self.parameters()).device
        nH, nW = cells_flat["nH"], cells_flat["nW"]
        N = nH * nW
        K = self.K

        rho_bins_in = cells_flat["rho_bins"].to(device)
        is_border = cells_flat["is_border"].to(device)

        rho_seed = rho_seed_from_bins(
            rho_bins_in.reshape(N, K), self.eta_z, is_border, self.eps,
        )
        ok_map = (~is_border).reshape(nH, nW)
        rho = rho_seed.clone()
        snapshot_steps: set[int] | None = None
        bimodality_per_iter: list[float] | None = None
        if return_surface_diags:
            snapshot_steps = set(
                l2_snapshot_steps(self.T_refine, TRAIN.L2_SNAPSHOT_MAX)
            )
            bimodality_per_iter = []

        def _bimodality_sum(r: torch.Tensor) -> float:
            return float((r * (1.0 - r)).sum().detach())

        if bimodality_per_iter is not None and 0 in snapshot_steps:
            bimodality_per_iter.append(_bimodality_sum(rho))

        grad_window = self.grad_window
        drive_debug = None
        want_drive_debug = bool(kw.get("return_drive_debug", False))
        geometry_snapshots: dict[str, dict[str, np.ndarray]] | None = None

        def _pool(r: torch.Tensor):
            return _pool_cell_geometry(
                r.reshape(nH, nW, K), ok_map,
                self.W_coll, self.W_iso,
                self.W_count_coll, self.W_count,
                self.R_fac, self.R_sup, self.eps,
            )

        if return_surface_diags:
            geometry_snapshots = {}
            rho_3d = rho.reshape(nH, nW, K)
            rho_coll, c_iso = _pool(rho_3d)
            self._record_geometry_snapshot(
                geometry_snapshots, "t0", rho_coll, c_iso, rho_3d, ok_map,
            )

        for t_iter in range(self.T_refine):
            if self.training and t_iter > 0 and t_iter % grad_window == 0:
                rho = rho.detach().requires_grad_(True)

            rho_3d = rho.reshape(nH, nW, K)
            rho_coll, c_iso = _pool(rho_3d)
            cross = cross_from_rho(
                rho_3d, ok_map, self.W_disk, self.W_count,
                self.R_sup, K, self.eps,
            )
            last = want_drive_debug and (t_iter == self.T_refine - 1)
            out = self._iterate_once(
                rho_3d, rho_coll, c_iso, cross,
                ok_map.to(dtype=rho.dtype, device=device),
                return_drive_terms=last,
            )
            if last:
                rho_3d, drive_debug = out
            else:
                rho_3d = out
            rho = rho_3d.reshape(N, K)

            if bimodality_per_iter is not None and (t_iter + 1) in snapshot_steps:
                bimodality_per_iter.append(_bimodality_sum(rho))

        rho_out_bins = rho.reshape(nH, nW, K)
        rho_out_bins = rho_out_bins * ok_map.unsqueeze(-1).to(dtype=rho_out_bins.dtype)

        if geometry_snapshots is not None:
            rho_coll, c_iso = _pool(rho_out_bins)
            self._record_geometry_snapshot(
                geometry_snapshots, "t_last", rho_coll, c_iso, rho_out_bins, ok_map,
            )

        rho_scalar, theta_out, k_star_out = collapse_rho_bins(
            rho_out_bins, K, self.eps, device=device,
        )
        rho_out_flat = rho_scalar.reshape(N)

        cf_out = dict(cells_flat)
        cf_out["theta"] = theta_out.reshape(N, 1)
        cf_out["k_star"] = k_star_out.reshape(N)

        branch = torch.zeros(N, device=device, dtype=torch.long)
        z1 = torch.zeros(N, 1, device=device, dtype=rho_out_flat.dtype)

        diags = None
        if return_surface_diags or want_drive_debug:
            diags = {}
            if return_surface_diags:
                ra = rho_out_flat[~is_border]
                diags["iter_stats"] = [{
                    "rho_mean": float(rho_out_flat.mean().detach()),
                    "rho_max": float(rho_out_flat.max().detach()),
                    "mid_band_frac": float(
                        ((ra > .3) & (ra < .7)).float().mean().detach()
                    ),
                    "n_interior": int(ok_map.sum().item()),
                }]
                if bimodality_per_iter is not None:
                    diags["bimodality_per_iter"] = bimodality_per_iter
                if geometry_snapshots is not None:
                    diags["geometry"] = geometry_snapshots
            if drive_debug is not None:
                diags["drive_debug"] = drive_debug

        return rho_out_flat, branch, rho_out_flat, z1, z1, cf_out, diags
