r"""L2 — cell-grid ρ refinement via recurrent Naka–Rushton on convolved geometry.

Seed (once, fixed in drive):
       ρ_seed = ρ_peak / (ρ_total + η_z + ε)

  L1 exports ρ_peak = max_k ρ_raw^{(k)} and ρ_total = Σ_k ρ_raw^{(k)} as
  cells_flat["rho_peak"] and cells_flat["z0"].

  Per iteration t (ρ⁽⁰⁾ = ρ_seed for pooling; ρ_seed fixed in drive):
       drive = b_seed·ρ_seed + b_coll·ρ_coll
       ρ⁽ᵗ⁺¹⁾ = drive² / (drive² + b_iso·c_iso + b_cross·c_cross + η_p² + ε)

  Geometric pools (ρ_coll, c_iso, c_cross): grouped conv2d over (2R+1)² neighborhoods
  (collinear R_fac; iso/cross R_sup); ρ scattered into K channels by hard θ-bin before conv;
  per-cell gather at own bin; coll/iso count-normalized; cross = mean ρ² in other bins locally.

Learned: b_coll, b_seed, b_iso, b_cross, η_p, η_z
         — **6** nonnegative scalars (softplus).
  L1 κ is not used by the renderer.
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
    """W_coll, W_iso, W_cross, W_count_coll, W_count — center = 0; coll uses R_fac, rest use R_sup."""
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

    W_cross = torch.ones(K, 1, patch_sup, patch_sup, device=dev, dtype=dtype)
    W_count_coll = torch.ones(K, 1, patch_fac, patch_fac, device=dev, dtype=dtype)
    W_count = torch.ones(K, 1, patch_sup, patch_sup, device=dev, dtype=dtype)

    W_iso[:, :, R_sup, R_sup] = 0.0
    W_cross[:, :, R_sup, R_sup] = 0.0
    W_count_coll[:, :, R_fac, R_fac] = 0.0
    W_count[:, :, R_sup, R_sup] = 0.0

    return W_coll, W_iso, W_cross, W_count_coll, W_count


def _hard_bin_map(theta_flat: torch.Tensor, K: int, nH: int, nW: int) -> torch.Tensor:
    """Peak orientation bin per cell, shape (nH, nW)."""
    bin_width = math.pi / K
    return (
        ((theta_flat % math.pi) / bin_width)
        .long()
        .clamp(0, K - 1)
        .reshape(nH, nW)
    )


def _pool_cell_geometry(
    rho: torch.Tensor,
    ok_map: torch.Tensor,
    bin_map: torch.Tensor,
    W_coll: torch.Tensor,
    W_iso: torch.Tensor,
    W_cross: torch.Tensor,
    W_count_coll: torch.Tensor,
    W_count: torch.Tensor,
    R_fac: int,
    R_sup: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter ρ into K θ-bin channels, grouped conv, gather own bin; (nH, nW) pools."""
    nH, nW = ok_map.shape
    K = W_cross.shape[0]
    dtype = rho.dtype
    dev = rho.device

    rho_2d = rho.reshape(1, 1, nH, nW)
    rho_sq_2d = rho_2d * rho_2d
    ok_2d = ok_map.to(dtype=dtype, device=dev).reshape(1, 1, nH, nW)

    W_coll = W_coll.to(device=dev, dtype=dtype)
    W_iso = W_iso.to(device=dev, dtype=dtype)
    W_cross = W_cross.to(device=dev, dtype=dtype)
    W_count_coll = W_count_coll.to(device=dev, dtype=dtype)
    W_count = W_count.to(device=dev, dtype=dtype)

    bin_onehot = (
        Fn.one_hot(bin_map.reshape(-1).long(), K)
        .to(dtype=dtype, device=dev)
        .reshape(nH, nW, K)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    rho_bins = bin_onehot * rho_2d
    rho_sq_bins = bin_onehot * rho_sq_2d
    ok_bins = bin_onehot * ok_2d

    coll_bins = Fn.conv2d(rho_bins, W_coll, padding=R_fac, groups=K)
    iso_bins = Fn.conv2d(rho_sq_bins, W_iso, padding=R_sup, groups=K)
    count_coll_bins = Fn.conv2d(ok_bins, W_count_coll, padding=R_fac, groups=K)
    count_bins = Fn.conv2d(ok_bins, W_count, padding=R_sup, groups=K)

    b = bin_map.to(device=dev).view(1, 1, nH, nW)
    own_count_coll = count_coll_bins.gather(1, b).squeeze(0).squeeze(0)
    own_count = count_bins.gather(1, b).squeeze(0).squeeze(0)

    rho_coll = coll_bins.gather(1, b).squeeze(0).squeeze(0) / (own_count_coll + eps)
    c_iso = iso_bins.gather(1, b).squeeze(0).squeeze(0) / (own_count + eps)

    m0_bins = Fn.conv2d(rho_sq_bins, W_cross, padding=R_sup, groups=K)
    m0_total = m0_bins.sum(dim=1).squeeze(0)
    count_total = count_bins.sum(dim=1).squeeze(0)
    own_m0 = m0_bins.gather(1, b).squeeze(0).squeeze(0)
    count_own = count_bins.gather(1, b).squeeze(0).squeeze(0)
    c_cross = (m0_total - own_m0) / (count_total - count_own + eps)

    ok_f = ok_map.to(dtype=dtype, device=dev)
    rho_coll = rho_coll * ok_f
    c_iso = c_iso * ok_f
    c_cross = c_cross * ok_f
    return rho_coll, c_iso, c_cross


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _inv_softplus(x):
    return math.log(math.expm1(max(float(x), 1e-8)))


def rho_seed_from_peak_z0(
    rho_peak: torch.Tensor,
    z0: torch.Tensor,
    eta_z: torch.Tensor,
    is_border: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """ρ_seed = ρ_peak / (ρ_total + η_z + ε)."""
    denom = z0 + eta_z + eps
    rho = rho_peak / denom.clamp_min(eps)
    return torch.where(is_border, torch.zeros_like(rho), rho)


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
        tbptt_n_segments: int = TRAIN.L2_SNAPSHOT_MAX,
        **kw,
    ):
        super().__init__()
        _ = kw
        self.R_fac, self.R_sup, self.K = r_fac_pool, r_sup_pool, K
        self.T_refine, self.eps = t_refine, eps
        self.tbptt_n_segments = max(1, int(tbptt_n_segments))

        self._eta_z_raw = nn.Parameter(torch.tensor(_inv_softplus(max(eta_z_init, 1e-6))))

        self._b_coll_raw = nn.Parameter(torch.tensor(_inv_softplus(L2.B_COLL_INIT)))
        self._b_seed_raw = nn.Parameter(torch.tensor(_inv_softplus(L2.B_SEED_INIT)))
        self._b_iso_raw = nn.Parameter(torch.tensor(_inv_softplus(L2.B_ISO_INIT)))
        self._b_cross_raw = nn.Parameter(torch.tensor(_inv_softplus(L2.B_CROSS_INIT)))
        self._eta_p_raw = nn.Parameter(
            torch.tensor(_inv_softplus(float(L2.ETA_P_INIT)), dtype=torch.float32)
        )

        W_coll, W_iso, W_cross, W_count_coll, W_count = _build_conv_kernels(
            self.R_fac, self.R_sup, K,
        )
        self.register_buffer("W_coll", W_coll)
        self.register_buffer("W_iso", W_iso)
        self.register_buffer("W_cross", W_cross)
        self.register_buffer("W_count_coll", W_count_coll)
        self.register_buffer("W_count", W_count)

    @property
    def eta_z(self):
        return Fn.softplus(self._eta_z_raw)
    @property
    def b_coll(self): return Fn.softplus(self._b_coll_raw)
    @property
    def b_seed(self): return Fn.softplus(self._b_seed_raw)
    @property
    def b_iso(self): return Fn.softplus(self._b_iso_raw)
    @property
    def b_cross(self): return Fn.softplus(self._b_cross_raw)
    @property
    def eta_p(self): return Fn.softplus(self._eta_p_raw)

    @property
    def grad_window(self) -> int:
        return l2_grad_window(self.T_refine, self.tbptt_n_segments)

    def _record_geometry_snapshot(
        self,
        store: dict[str, dict[str, np.ndarray]],
        tag: str,
        rho_coll: torch.Tensor,
        c_iso: torch.Tensor,
        c_cross: torch.Tensor,
    ) -> None:
        """Record raw geometry pools (same signals that enter the drive)."""
        store[tag] = {
            k: v.detach().cpu().numpy().astype(np.float64)
            for k, v in (
                ("rho_coll", rho_coll),
                ("c_iso", c_iso),
                ("c_cross", c_cross),
            )
        }

    def _iterate_once(
        self,
        rho_coll,
        c_iso,
        c_cross,
        rho_seed,
        ok_map,
        return_drive_terms: bool = False,
    ):
        drive = self.b_seed * rho_seed + self.b_coll * rho_coll
        drive_sq = drive * drive
        eta_p_sq = self.eta_p * self.eta_p
        inhib = self.b_iso * c_iso + self.b_cross * c_cross + eta_p_sq
        rho_next = (drive_sq / (drive_sq + inhib + self.eps)) * ok_map

        if return_drive_terms:
            with torch.no_grad():
                m = ok_map > 0
                def _stat(t):
                    if not m.any():
                        return 0.0, 0.0
                    x = t[m]
                    return float(x.mean()), float(x.max())
                return rho_next, {
                    "rho_coll_mean": _stat(rho_coll)[0],
                    "c_iso_mean": _stat(c_iso)[0],
                    "c_cross_mean": _stat(c_cross)[0],
                    "drive_mean": _stat(drive)[0],
                    "drive_sq_mean": _stat(drive_sq)[0],
                    "inhib_mean": _stat(inhib)[0],
                    "rho_next_mean": _stat(rho_next)[0],
                    "term_seed_mean": _stat(self.b_seed * rho_seed)[0],
                    "term_coll_mean": _stat(self.b_coll * rho_coll)[0],
                    "term_iso_mean": _stat(self.b_iso * c_iso)[0],
                    "term_cross_mean": _stat(self.b_cross * c_cross)[0],
                    "eta_p": float(self.eta_p.detach()),
                }
        return rho_next

    def forward(self, cells_flat, return_surface_diags=False, **kw):
        device = next(self.parameters()).device
        nH, nW = cells_flat["nH"], cells_flat["nW"]
        N = nH * nW

        theta_full = cells_flat["theta"].to(device)
        rho_peak = cells_flat["rho_peak"].to(device)
        z0 = cells_flat["z0"].to(device)
        is_border = cells_flat["is_border"].to(device)
        theta_flat = theta_full.reshape(-1) if theta_full.dim() == 2 else theta_full
        if theta_flat.dim() == 2:
            theta_flat = theta_flat[:, 0]

        rho_seed = rho_seed_from_peak_z0(
            rho_peak, z0, self.eta_z, is_border, self.eps,
        )

        ok_map = (~is_border).reshape(nH, nW)
        k_star_flat = cells_flat.get("k_star")
        if k_star_flat is not None:
            bin_map = k_star_flat.to(device).reshape(nH, nW).long().clamp(0, self.K - 1)
        else:
            bin_map = _hard_bin_map(theta_flat, self.K, nH, nW)
        rho_seed_2d = rho_seed.reshape(nH, nW)

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
                r, ok_map, bin_map,
                self.W_coll, self.W_iso, self.W_cross,
                self.W_count_coll, self.W_count,
                self.R_fac, self.R_sup, self.eps,
            )

        if return_surface_diags:
            geometry_snapshots = {}
            rho_coll, c_iso, c_cross = _pool(rho.reshape(nH, nW))
            self._record_geometry_snapshot(
                geometry_snapshots, "t0", rho_coll, c_iso, c_cross,
            )

        for t_iter in range(self.T_refine):
            if self.training and t_iter > 0 and t_iter % grad_window == 0:
                rho = rho.detach().requires_grad_(True)

            rho_coll, c_iso, c_cross = _pool(rho.reshape(nH, nW))
            last = want_drive_debug and (t_iter == self.T_refine - 1)
            out = self._iterate_once(
                rho_coll, c_iso, c_cross, rho_seed_2d,
                ok_map.to(dtype=rho.dtype, device=device),
                return_drive_terms=last,
            )
            if last:
                rho_2d, drive_debug = out
            else:
                rho_2d = out
            rho = rho_2d.reshape(N)

            if bimodality_per_iter is not None and (t_iter + 1) in snapshot_steps:
                bimodality_per_iter.append(_bimodality_sum(rho))

        rho_out = rho.reshape(nH, nW)

        if geometry_snapshots is not None:
            rho_coll, c_iso, c_cross = _pool(rho_out)
            self._record_geometry_snapshot(
                geometry_snapshots, "t_last", rho_coll, c_iso, c_cross,
            )

        rho_out = rho_out * ok_map.to(dtype=rho_out.dtype)
        rho_out_flat = rho_out.reshape(N)

        cf_out = dict(cells_flat)

        branch = torch.zeros(N, device=device, dtype=torch.long)
        z1 = torch.zeros(N, 1, device=device, dtype=rho_out.dtype)

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
