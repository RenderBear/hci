r"""Cell-grid seed — NR orientation selectivity from L1 bin masses.

Per cell, from L1 ``rho_bins``:
  ρ̂^(k) = ρ_bins^(k) / (ρ_total + ε)
  ρ̃^(k) = ρ̂^(k) − min_j ρ̂^(j)
  ρ_seed^(k) = ρ̃^(k)² / (ρ̃^(k)² + η_z²)

Scalar export to renderer: ρ(c) = max_k ρ_seed^(k)(c), with parabolic θ from bins.

Learned: η_z only (softplus, init 0.1).
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


def rho_seed_from_bins(
    rho_bins: torch.Tensor,
    eta_z: float | torch.Tensor,
    is_border: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Total-normalize, min-subtract, NR squash → (N, K) ρ_seed."""
    if is_border.dim() > 1:
        is_border = is_border.reshape(-1)
    rho_total = rho_bins.sum(dim=-1, keepdim=True)
    rho_hat = rho_bins / (rho_total + eps)
    rho_tilde = rho_hat - rho_hat.min(dim=-1, keepdim=True).values
    rho_tilde_sq = rho_tilde * rho_tilde
    eta_z_sq = eta_z * eta_z
    rho_seed = rho_tilde_sq / (rho_tilde_sq + eta_z_sq)
    mask = is_border.unsqueeze(-1) if is_border.dim() == 1 else is_border.unsqueeze(-1)
    return torch.where(mask, torch.zeros_like(rho_seed), rho_seed)


def collapse_rho_bins(
    rho_bins: torch.Tensor,
    K: int,
    eps: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parabolic sub-bin θ and max_k ρ from (nH, nW, K) or (N, K) bin masses."""
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


class CellSeed(nn.Module):
    """L1 bin masses → scalar ρ for renderer (max over NR seed bins)."""

    def __init__(
        self,
        K: int = SEED.K,
        eps: float = SEED.EPS,
        eta_z_init: float = SEED.ETA_Z_INIT,
        **kw,
    ):
        super().__init__()
        _ = kw
        self.K = int(K)
        self.eps = float(eps)
        self._eta_z_raw = nn.Parameter(
            torch.tensor(_inv_softplus(float(eta_z_init)), dtype=torch.float32)
        )

    @property
    def eta_z(self) -> torch.Tensor:
        return Fn.softplus(self._eta_z_raw)

    def forward(
        self,
        cells_flat,
        return_surface_diags: bool = False,
        **kw,
    ):
        _ = kw
        device = next(self.parameters()).device
        nH, nW = cells_flat["nH"], cells_flat["nW"]
        N = nH * nW
        K = self.K

        rho_bins_in = cells_flat["rho_bins"].to(device)
        is_border = cells_flat["is_border"].to(device)

        rho_seed = rho_seed_from_bins(
            rho_bins_in.reshape(N, K), self.eta_z, is_border, self.eps,
        )
        rho_seed_3d = rho_seed.reshape(nH, nW, K)
        ok_map = (~is_border).reshape(nH, nW)

        rho_scalar, theta_out, k_star_out = collapse_rho_bins(
            rho_seed_3d, K, self.eps, device=device,
        )
        interior = ok_map.to(dtype=rho_scalar.dtype)
        rho_out_flat = rho_scalar.reshape(N) * interior.reshape(N)

        cf_out = dict(cells_flat)
        cf_out["theta"] = theta_out.reshape(N, 1)
        cf_out["k_star"] = k_star_out.reshape(N)

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
                    "n_interior": int(ok_map.sum().item()),
                }],
            }

        return rho_out_flat, branch, rho_out_flat, z1, z1, cf_out, diags
