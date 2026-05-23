r"""L1 → ρ_seed: eigendecomposition statistics and tile-interior mask.

No tile dynamics: ``ρ`` is fixed from the L1 cell grid as
``ρ = (λ₁/(z₀+η_z)) · interior`` (no NR pool on the seed).  Learned scalar
``η_z`` only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from params import SEED


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


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


class RhoSeedModule(nn.Module):
    """Learned ``η_z`` for raw ratio seed ``r = λ₁/(z₀+η_z)`` (no NR-pool param)."""

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

    @property
    def eta_z(self) -> torch.Tensor:
        return Fn.softplus(self._eta_z_raw)

    def rho_cell(self, cells_flat: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-cell ``ρ`` and branch index (always 0)."""
        device = next(self.parameters()).device
        nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
        N = nH * nW
        lam1 = cells_flat["lam"][..., 0].to(device)
        z0 = cells_flat["z0"].to(device)
        is_border = cells_flat["is_border"].to(device)
        r_raw = lam1 / (z0 + self.eta_z).clamp_min(self.eps)
        r_raw = torch.where(is_border, torch.zeros_like(r_raw), r_raw)
        rho_seed = r_raw

        ti, tj = _build_tile_grid(nH, nW, self.R, self.stride, device)
        mi = _build_tile_membership(ti, tj, nW, self.R, device)
        tile_cov = torch.zeros(N, dtype=torch.bool, device=device)
        tile_cov[mi.reshape(-1)] = True
        interior = (~is_border & tile_cov).to(dtype=rho_seed.dtype)
        rho_out = rho_seed * interior
        branch = torch.zeros(N, device=device, dtype=torch.long)
        return rho_out, branch

    def forward(
        self,
        cells_flat: dict,
        return_surface_diags: bool = False,
        **kw,
    ):
        """Match old ``TileDynamics`` call shape for drop-in training/inference."""
        _ = kw
        rho_out, branch = self.rho_cell(cells_flat)
        N = rho_out.shape[0]
        device = rho_out.device
        dtype = rho_out.dtype
        z1 = torch.zeros(N, 1, device=device, dtype=dtype)
        cf_out = dict(cells_flat)
        diags = None
        if return_surface_diags:
            nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
            is_border = cells_flat["is_border"].to(device)
            ti, tj = _build_tile_grid(nH, nW, self.R, self.stride, device)
            mi = _build_tile_membership(ti, tj, nW, self.R, device)
            ra = rho_out[~is_border]
            diags = {
                "iter_stats": [{
                    "rho_mean": float(rho_out.mean().detach()),
                    "rho_max": float(rho_out.max().detach()),
                    "mid_band_frac": float(
                        ((ra > 0.3) & (ra < 0.7)).float().mean().detach()
                    ) if ra.numel() else 0.0,
                    "n_tiles": int(mi.shape[0]),
                }],
                "seed_only": True,
            }
        return rho_out, branch, rho_out, z1, z1, cf_out, diags
