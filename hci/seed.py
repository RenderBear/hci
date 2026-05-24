r"""Seed module — thin wrapper providing renderer-compatible interface.

HypercolumnSeed holds **fixed** ``η_z`` (buffer; diagnostics only), learnable ``η₀`` plus a small
MLP for **spatial** NR ``η=η₀·σ(``MLP``(``pooled κ,z``))`` on collinear passes,
raw β weights, and collinear σ scales used in ``run_l1_hypercolumn``.  Stored ``κ`` in ``cells_flat`` is **cosine alignment**
``ρ·S/(‖ρ‖‖S‖)`` (per pass) for the readout MLP, not a multiplicative gate on ``ρ``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from params import L1

from .L1 import HypercolumnSeed


class RhoSeedModule(nn.Module):
    """Drop-in replacement using HypercolumnSeed.

    The forward() method expects cells_flat from ``run_l1_hypercolumn``
    (raw ``μ`` + collinear GABA with spatial ``η``, dominant-bin readout).
    It reads dominant ρ from ``lam[...,0]`` and passes through with the renderer
    return signature.  During training, ``train.prepare_batch`` builds that
    dict with ``cells_format="torch"`` so ``hc_seed`` participates in autograd.
    """

    def __init__(self, r_pool=10, stride=7, eps=1e-9, eta_z_init: float | None = None):
        super().__init__()
        self.hc_seed = HypercolumnSeed(
            r_pool=r_pool,
            stride=stride,
            eps=eps,
            n_gaba_passes=int(L1.COL_PASSES),
            eta_z_init=eta_z_init,
        )
        # Expose for compat
        self.R = self.hc_seed.R
        self.stride = self.hc_seed.stride
        self.eps = self.hc_seed.eps

    @property
    def eta_z(self):
        return self.hc_seed.eta_z

    def rho_cell(self, cells_flat):
        """Return pre-computed ρ and branch from cells_flat."""
        device = next(self.parameters()).device
        N = int(cells_flat["nH"]) * int(cells_flat["nW"])

        if "rho_precomputed" in cells_flat:
            rho = cells_flat["rho_precomputed"].to(device)
        else:
            # Fallback: use lam[:, 0] as ρ (set by run_l1_hypercolumn)
            rho = cells_flat["lam"][..., 0].to(device)

        is_border = cells_flat["is_border"].to(device)
        rho = torch.where(is_border, torch.zeros_like(rho), rho)
        branch = torch.zeros(N, device=device, dtype=torch.long)
        return rho, branch

    def forward(self, cells_flat, return_surface_diags=False, **kw):
        rho_out, branch = self.rho_cell(cells_flat)
        N = rho_out.shape[0]
        device = rho_out.device
        dtype = rho_out.dtype
        z1 = torch.zeros(N, 1, device=device, dtype=dtype)
        cf_out = dict(cells_flat)
        diags = None
        if return_surface_diags:
            is_border = cells_flat["is_border"].to(device)
            ra = rho_out[~is_border]
            diags = {
                "iter_stats": [{
                    "rho_mean": float(rho_out.mean().detach()),
                    "rho_max": float(rho_out.max().detach()),
                    "mid_band_frac": float(
                        ((ra > 0.3) & (ra < 0.7)).float().mean().detach()
                    ) if ra.numel() else 0.0,
                    "n_tiles": 0,
                }],
                "seed_only": True,
            }
        return rho_out, branch, rho_out, z1, z1, cf_out, diags
