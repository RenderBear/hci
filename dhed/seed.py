r"""L1 → ρ_seed: eigendecomposition statistics, Naka–Rushton pool normalize, tile mask.

No tile dynamics: ``ρ`` is fixed from the L1 cell grid as in STRIATE's TileDynamics
initial condition (``r = λ₁/(z₀+η_z)``, then ``NR_pool``), times the same
tile-interior coverage mask used previously.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from params import SEED


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


def naka_rushton_pool_normalize(
    x_flat: torch.Tensor,
    nH: int,
    nW: int,
    pool_radius: int,
    eta: torch.Tensor,
    is_border: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Local Naka–Rushton on the cell grid (same as STRIATE L2 seed / recurrent NR)."""
    x_1d = x_flat.reshape(-1)
    ib_1d = is_border.reshape(-1)
    if x_1d.numel() != nH * nW:
        raise ValueError(
            f"naka_rushton_pool_normalize: x has {x_1d.numel()} elements, "
            f"expected nH*nW={nH * nW} (nH={nH}, nW={nW})"
        )
    if ib_1d.numel() != nH * nW:
        raise ValueError(
            f"naka_rushton_pool_normalize: is_border has {ib_1d.numel()} elements, "
            f"expected {nH * nW}"
        )
    R = int(pool_radius)
    k = 2 * R + 1
    x_sq = x_1d * x_1d
    x_sq_g = x_sq.view(nH, nW)
    mu_sq = Fn.avg_pool2d(
        x_sq_g.unsqueeze(0).unsqueeze(0),
        kernel_size=k,
        stride=1,
        padding=R,
    ).squeeze(0).squeeze(0).reshape(-1)
    et = eta.to(dtype=x_1d.dtype, device=x_1d.device)
    eta_sq = et * et
    denom = x_sq + mu_sq + eta_sq + eps
    out = x_sq / denom
    return torch.where(ib_1d, torch.zeros_like(out), out)


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
    """Learned ``η_z``, ``η_ρ`` for NR seed; fixed tile geometry ``R``, ``stride``."""

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
        self._eta_rho_raw = nn.Parameter(
            torch.tensor(_inv_softplus(float(SEED.ETA_RHO_INIT)), dtype=torch.float32)
        )

    @property
    def eta_z(self) -> torch.Tensor:
        return Fn.softplus(self._eta_z_raw)

    @property
    def eta_rho(self) -> torch.Tensor:
        return Fn.softplus(self._eta_rho_raw)

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
        rho_seed = naka_rushton_pool_normalize(
            r_raw, nH, nW, self.R, self.eta_rho, is_border, self.eps,
        )

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
