r"""Cell-grid AND gate — surround-normalized coherence × energy.

From L1 moments (R, ρ_total):
  E_rel = ρ_total / (ε + ⟨ρ_total⟩_𝒩)   (Gaussian surround, center-excluded)
  g_R = σ(a(R − R₀)),  g_E = σ(b log E_rel)
  ρ(c) = g_R · g_E · ok(c)

Learned: R₀ (init 0.45), a, b (softplus, init 12 / 5).
θ passes through from L1 unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from params import SEED


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


def _surround_kernel(
    radius: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    size = 2 * int(radius) + 1
    coords = torch.arange(size, device=device, dtype=dtype) - float(radius)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) ** 2))
    g[int(radius), int(radius)] = 0.0
    g = g / g.sum().clamp_min(1e-8)
    return g


def surround_mean_rho_total(
    rho_total: torch.Tensor,
    nH: int,
    nW: int,
    *,
    radius: int = SEED.SURROUND_RADIUS,
    sigma: float = SEED.SURROUND_SIGMA,
) -> torch.Tensor:
    """⟨ρ_total⟩_𝒩 via center-excluded Gaussian conv (reflect-padded)."""
    dev, dtype = rho_total.device, rho_total.dtype
    grid = rho_total.reshape(nH, nW).to(dtype=dtype)
    k = _surround_kernel(radius, sigma, dev, dtype).unsqueeze(0).unsqueeze(0)
    pad = int(radius)
    x = grid.unsqueeze(0).unsqueeze(0)
    x_pad = Fn.pad(x, (pad, pad, pad, pad), mode="reflect")
    return Fn.conv2d(x_pad, k).squeeze(0).squeeze(0)


def relative_energy(
    rho_total: torch.Tensor,
    nH: int,
    nW: int,
    eps: float,
    *,
    radius: int = SEED.SURROUND_RADIUS,
    sigma: float = SEED.SURROUND_SIGMA,
) -> torch.Tensor:
    """E_rel(c) = ρ_total / (ε + ⟨ρ_total⟩_𝒩), shape (nH, nW)."""
    nb = surround_mean_rho_total(
        rho_total, nH, nW, radius=radius, sigma=sigma,
    )
    grid = rho_total.reshape(nH, nW)
    return grid / (float(eps) + nb)


class AndGateSeed(nn.Module):
    """Surround-normalized AND gate on (R, E_rel) → scalar ρ for renderer."""

    def __init__(
        self,
        eps: float = SEED.EPS,
        R0_init: float = SEED.R0_INIT,
        a_init: float = SEED.A_INIT,
        b_init: float = SEED.B_INIT,
        surround_radius: int = SEED.SURROUND_RADIUS,
        surround_sigma: float = SEED.SURROUND_SIGMA,
        **kw,
    ):
        super().__init__()
        _ = kw
        self.eps = float(eps)
        self.surround_radius = int(surround_radius)
        self.surround_sigma = float(surround_sigma)
        self._R0 = nn.Parameter(torch.tensor(float(R0_init), dtype=torch.float32))
        self._a_raw = nn.Parameter(
            torch.tensor(_inv_softplus(float(a_init)), dtype=torch.float32)
        )
        self._b_raw = nn.Parameter(
            torch.tensor(_inv_softplus(float(b_init)), dtype=torch.float32)
        )

    @property
    def R0(self) -> torch.Tensor:
        return self._R0.view(())

    @property
    def a(self) -> torch.Tensor:
        return Fn.softplus(self._a_raw).view(())

    @property
    def b(self) -> torch.Tensor:
        return Fn.softplus(self._b_raw).view(())

    def forward(
        self,
        cells_flat,
        return_surface_diags: bool = False,
        **kw,
    ):
        _ = kw
        device = next(self.parameters()).device
        nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
        N = nH * nW

        R_in = cells_flat["coherence_R"].to(device).reshape(N)
        rho_t = cells_flat["rho_total"].to(device).reshape(N)
        is_border = cells_flat["is_border"].to(device).reshape(N).bool()

        E_rel = relative_energy(
            rho_t, nH, nW, self.eps,
            radius=self.surround_radius,
            sigma=self.surround_sigma,
        ).reshape(N)

        g_R = torch.sigmoid(self.a * (R_in - self.R0))
        g_E = torch.sigmoid(self.b * torch.log(E_rel.clamp_min(1e-8)))
        ok = (~is_border).to(dtype=g_R.dtype)
        rho_out_flat = g_R * g_E * ok

        cf_out = dict(cells_flat)
        cf_out["E_rel"] = E_rel.reshape(nH, nW)
        cf_out["g_R"] = (g_R * ok).reshape(nH, nW)
        cf_out["g_E"] = (g_E * ok).reshape(nH, nW)

        branch = torch.zeros(N, device=device, dtype=torch.long)
        z1 = torch.zeros(N, 1, device=device, dtype=rho_out_flat.dtype)

        diags = None
        if return_surface_diags:
            ra = rho_out_flat[~is_border]
            diags = {
                "iter_stats": [{
                    "rho_mean": float(rho_out_flat.mean().detach()),
                    "rho_max": float(rho_out_flat.max().detach()),
                    "mid_band_frac": float(
                        ((ra > 0.3) & (ra < 0.7)).float().mean().detach()
                    ) if ra.numel() else 0.0,
                    "n_interior": int(ok.sum().item()),
                }],
            }

        return rho_out_flat, branch, rho_out_flat, z1, z1, cf_out, diags


# Legacy alias
CellSeed = AndGateSeed
