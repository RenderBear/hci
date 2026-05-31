r"""Cell-grid contour seed — Naka–Rushton on coherent magnitude |Z|.

L1 supplies ``rho_peak`` = |Σ z₂| and θ = ½ arg(Σ z₂). With R = |Z|·ok:

  ρ(c) = R(c)² / (R(c)² + η_z² + ε) · ok(c)     (learned η_z > 0 via softplus)

No collinear pool, surround, or second divisive stage. ``relative_energy`` for
infer diagnostics is kept as a standalone helper (Gaussian neighborhood of ρ_total).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

try:
    from params import SEED
except Exception:  # pragma: no cover
    class SEED:  # type: ignore
        EPS = 1e-6
        ETA_Z_INIT = 0.30


_ETA_Z_INIT = float(getattr(SEED, "ETA_Z_INIT", 0.30))
_SURROUND_RADIUS = int(getattr(SEED, "SURROUND_RADIUS_DIAG", 5))
_SURROUND_SIGMA = float(getattr(SEED, "SURROUND_SIGMA_DIAG", 2.0))


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


def surround_mean(
    field: torch.Tensor,
    nH: int,
    nW: int,
    *,
    radius: int = _SURROUND_RADIUS,
    sigma: float = _SURROUND_SIGMA,
) -> torch.Tensor:
    """⟨field⟩_𝒩 via center-excluded Gaussian conv (reflect-padded), shape (nH, nW)."""
    dev, dtype = field.device, field.dtype
    grid = field.reshape(nH, nW).to(dtype=dtype)
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
    radius: int = _SURROUND_RADIUS,
    sigma: float = _SURROUND_SIGMA,
) -> torch.Tensor:
    """E_rel(c) = ρ_total / (ε + ⟨ρ_total⟩_𝒩). Diagnostic only (infer prep)."""
    nb = surround_mean(rho_total, nH, nW, radius=radius, sigma=sigma)
    grid = rho_total.reshape(nH, nW)
    return grid / (float(eps) + nb)


class ContourSeed(nn.Module):
    """ρ = R²/(R²+η_z²) with R = |Z| from L1 ``rho_peak``."""

    def __init__(
        self,
        eps: float = SEED.EPS,
        eta_z_init: float = _ETA_Z_INIT,
        **kw,
    ):
        super().__init__()
        _ = kw
        self.eps = float(eps)
        self._eta_z_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_z_init)))

    @property
    def eta_z(self) -> torch.Tensor:
        return Fn.softplus(self._eta_z_raw).view(())

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
        eps = self.eps

        rho_peak = cells_flat["rho_peak"].to(device).reshape(nH, nW).float()
        rho_t = cells_flat["rho_total"].to(device).reshape(nH, nW).float()
        is_border = cells_flat["is_border"].to(device).reshape(nH, nW).bool()
        ok = (~is_border).to(rho_peak.dtype)

        Rb = rho_peak * ok
        Rsq = Rb * Rb
        eta_sq = self.eta_z * self.eta_z
        rho = (Rsq / (Rsq + eta_sq + eps)) * ok
        rho_flat = rho.reshape(N)

        cf_out = dict(cells_flat)
        cf_out["rho_seed"] = rho
        cf_out["drive"] = Rb
        cf_out["E_rel"] = relative_energy(
            rho_t, nH, nW, eps,
            radius=_SURROUND_RADIUS,
            sigma=_SURROUND_SIGMA,
        )

        branch = torch.zeros(N, device=device, dtype=torch.long)
        z1 = torch.zeros(N, 1, device=device, dtype=rho_flat.dtype)

        diags = None
        if return_surface_diags:
            ra = rho_flat[~is_border.reshape(N)]
            diags = {
                "rho_coll": None,
                "iter_stats": [{
                    "rho_mean": float(rho_flat.mean().detach()),
                    "rho_max": float(rho_flat.max().detach()),
                    "mid_band_frac": float(
                        ((ra > 0.3) & (ra < 0.7)).float().mean().detach()
                    ) if ra.numel() else 0.0,
                    "n_interior": int(ok.sum().item()),
                    "fac_mean": 0.0,
                    "coll_mean": 0.0,
                    "sur_mean": 0.0,
                }],
            }

        return rho_flat, branch, rho_flat, z1, z1, cf_out, diags


AndGateSeed = ContourSeed
CellSeed = ContourSeed
