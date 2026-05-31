r"""Cell-grid contour seed — NR-normalised peak + collinear readback.

From L1 moments (ρ_peak = max|z₂|, θ = ½ arg Z₂ʷ, ρ_total):

  ρ_seed(c) = ρ_peak(c)² / (ρ_peak(c)² + η_seed²) · ok(c)

  collinear readback  ρ_coll(c) = relu( Σ_𝒩 w·ρ_seed'·a_κ(θ'−θ) / Σ_𝒩 w )

  excitation          e(c) = β_seed·ρ_seed(c) + β_coll·ρ_coll(c)

  surround            S(c) = ⟨ρ_seed⟩_𝒩   (broadside or isotropic; center-excluded)

  divisive readout    ρ(c) = e(c)² / (e(c)² + η² + λ·S(c)² + ε) · ok(c)

Learned (softplus-positive): β_seed, β_coll, κ_θ, η_seed, η, λ, σ_f.
θ passes through from L1 unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

try:
    from params import SEED
except Exception:  # pragma: no cover - allow standalone import / testing
    class SEED:  # type: ignore
        EPS = 1e-6
        SURROUND_RADIUS = 4
        SURROUND_SIGMA = 2.5
        R0_INIT = 0.45
        A_INIT = 12.0
        B_INIT = 5.0


_BETA_SEED_INIT = float(getattr(SEED, "BETA_SEED_INIT", getattr(SEED, "BETA_INIT", 0.30)))
_BETA_COLL_INIT = float(getattr(SEED, "BETA_COLL_INIT", getattr(SEED, "KAPPA_INIT", 3.0)))
_KAPPA_THETA_INIT = float(getattr(SEED, "KAPPA_THETA_INIT", 2.5))
_ETA_SEED_INIT = float(getattr(SEED, "ETA_SEED_INIT", 0.30))
_ETA_INIT = float(getattr(SEED, "ETA_INIT", 0.30))
_LAMBDA_INIT = float(getattr(SEED, "LAMBDA_INIT", 1.5))
_SIGMA_F_INIT = float(getattr(SEED, "SIGMA_F_INIT", 1.3))
_FACIL_RADIUS = int(getattr(SEED, "FACIL_RADIUS", 2))


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
    radius: int = SEED.SURROUND_RADIUS,
    sigma: float = SEED.SURROUND_SIGMA,
) -> torch.Tensor:
    """⟨field⟩_𝒩 via center-excluded Gaussian conv (reflect-padded), shape (nH, nW)."""
    dev, dtype = field.device, field.dtype
    grid = field.reshape(nH, nW).to(dtype=dtype)
    k = _surround_kernel(radius, sigma, dev, dtype).unsqueeze(0).unsqueeze(0)
    pad = int(radius)
    x = grid.unsqueeze(0).unsqueeze(0)
    x_pad = Fn.pad(x, (pad, pad, pad, pad), mode="reflect")
    return Fn.conv2d(x_pad, k).squeeze(0).squeeze(0)


surround_mean_rho_total = surround_mean


def relative_energy(
    rho_total: torch.Tensor,
    nH: int,
    nW: int,
    eps: float,
    *,
    radius: int = SEED.SURROUND_RADIUS,
    sigma: float = SEED.SURROUND_SIGMA,
) -> torch.Tensor:
    """E_rel(c) = ρ_total / (ε + ⟨ρ_total⟩_𝒩). Retained for diagnostics."""
    nb = surround_mean(rho_total, nH, nW, radius=radius, sigma=sigma)
    grid = rho_total.reshape(nH, nW)
    return grid / (float(eps) + nb)


def _shift(t: torch.Tensor, dy: int, dx: int, pad: int) -> torch.Tensor:
    nH, nW = t.shape
    tp = Fn.pad(t[None, None], (pad, pad, pad, pad), mode="reflect").squeeze(0).squeeze(0)
    r0 = pad + dy
    c0 = pad + dx
    return tp[r0:r0 + nH, c0:c0 + nW]


def collinear_facilitation(
    R: torch.Tensor,
    theta: torch.Tensor,
    *,
    sigma_f: torch.Tensor,
    kappa_theta: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    """ρ_coll(c) = relu( Σ_𝒩 w·R'·a_κ(θ'−θ) / Σ_𝒩 w ),  w = G(|δ|)·pos(δ;θ_c)."""
    nH, nW = R.shape
    dtype, dev = R.dtype, R.device
    ct, st = torch.cos(theta), torch.sin(theta)
    kθ = kappa_theta

    sig2 = (2.0 * sigma_f * sigma_f).clamp_min(eps)
    pad = int(radius)

    num = torch.zeros_like(R)
    den = torch.zeros_like(R)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            d2 = float(dy * dy + dx * dx)
            G = torch.exp(torch.tensor(-d2, dtype=dtype, device=dev) / sig2)
            t_proj = float(dy) * ct + float(dx) * st
            pos = (t_proj * t_proj) / (d2 + eps)
            w = G * pos
            th_sh = _shift(theta, dy, dx, pad)
            R_sh = _shift(R, dy, dx, pad)
            dtheta = 0.5 * torch.atan2(
                torch.sin(2.0 * (th_sh - theta)),
                torch.cos(2.0 * (th_sh - theta)),
            )
            a_k = torch.exp(kθ * (torch.cos(dtheta) - 1.0))
            num = num + w * R_sh * a_k
            den = den + w
    return torch.relu(num / (den + eps))


def cocircular_facilitation(
    R: torch.Tensor,
    theta: torch.Tensor,
    *,
    sigma_f: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    nH, nW = R.shape
    dtype, dev = R.dtype, R.device
    sig2 = (2.0 * sigma_f * sigma_f).clamp_min(eps)
    pad = int(radius)

    num = torch.zeros_like(R)
    den = torch.zeros_like(R)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            d2 = float(dy * dy + dx * dx)
            G = torch.exp(torch.tensor(-d2, dtype=dtype, device=dev) / sig2)
            beta = math.atan2(float(dx), float(dy))
            cos_bm = torch.cos(beta - theta)
            w = G * cos_bm * cos_bm
            th_sh = _shift(theta, dy, dx, pad)
            R_sh = _shift(R, dy, dx, pad)
            agree = torch.cos(2.0 * (th_sh + theta - 2.0 * beta))
            num = num + w * R_sh * agree
            den = den + w
    return torch.relu(num / (den + eps))


def broadside_surround(
    field: torch.Tensor,
    theta: torch.Tensor,
    *,
    sigma: float,
    radius: int,
    eps: float,
) -> torch.Tensor:
    nH, nW = field.shape
    ct, st = torch.cos(theta), torch.sin(theta)
    sig2 = 2.0 * float(sigma) * float(sigma)
    pad = int(radius)
    Se = torch.zeros_like(field)
    Wm = torch.zeros_like(field)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            d2 = float(dy * dy + dx * dx)
            G = math.exp(-d2 / sig2)
            n_proj = -float(dy) * st + float(dx) * ct
            perp = (n_proj * n_proj) / (d2 + eps)
            w = G * perp
            Se = Se + w * _shift(field, dy, dx, pad)
            Wm = Wm + w
    return Se / (Wm + eps)


class ContourSeed(nn.Module):
    """NR-normalised peak + collinear readback + divisive surround suppression.

      ρ_seed = ρ_peak² / (ρ_peak² + η_seed²)
      ρ_coll = collinear pool of ρ_seed
      e      = β_seed·ρ_seed + β_coll·ρ_coll
      S      = ⟨ρ_seed⟩_𝒩
      ρ      = e² / (e² + η² + λ·S²) · ok
    """

    def __init__(
        self,
        eps: float = SEED.EPS,
        beta_seed_init: float = _BETA_SEED_INIT,
        beta_coll_init: float = _BETA_COLL_INIT,
        kappa_theta_init: float = _KAPPA_THETA_INIT,
        eta_seed_init: float = _ETA_SEED_INIT,
        eta_init: float = _ETA_INIT,
        lambda_init: float = _LAMBDA_INIT,
        sigma_f_init: float = _SIGMA_F_INIT,
        facil_radius: int = _FACIL_RADIUS,
        facil_mode: str = str(getattr(SEED, "FACIL_MODE", "collinear")),
        surround_radius: int = SEED.SURROUND_RADIUS,
        surround_sigma: float = SEED.SURROUND_SIGMA,
        surround_mode: str = str(getattr(SEED, "SURROUND_MODE", "broadside")),
        **kw,
    ):
        super().__init__()
        _ = kw
        self.eps = float(eps)
        self.facil_radius = int(facil_radius)
        self.facil_mode = str(facil_mode)
        self.surround_radius = int(surround_radius)
        self.surround_sigma = float(surround_sigma)
        self.surround_mode = str(surround_mode)
        self._beta_seed_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_seed_init)))
        self._beta_coll_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_coll_init)))
        self._kappa_theta_raw = nn.Parameter(torch.tensor(_inv_softplus(kappa_theta_init)))
        self._eta_seed_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_seed_init)))
        self._eta_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_init)))
        self._lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(lambda_init)))
        self._sigma_f_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_f_init)))

    @property
    def beta_seed(self) -> torch.Tensor:
        return Fn.softplus(self._beta_seed_raw).view(())

    @property
    def beta_coll(self) -> torch.Tensor:
        return Fn.softplus(self._beta_coll_raw).view(())

    @property
    def kappa_theta(self) -> torch.Tensor:
        return Fn.softplus(self._kappa_theta_raw).view(())

    @property
    def eta_seed(self) -> torch.Tensor:
        return Fn.softplus(self._eta_seed_raw).view(())

    @property
    def eta(self) -> torch.Tensor:
        return Fn.softplus(self._eta_raw).view(())

    @property
    def lam(self) -> torch.Tensor:
        return Fn.softplus(self._lambda_raw).view(())

    @property
    def sigma_f(self) -> torch.Tensor:
        return Fn.softplus(self._sigma_f_raw).view(()).clamp_min(0.3)

    @property
    def beta(self) -> torch.Tensor:
        return self.beta_seed

    @property
    def kappa(self) -> torch.Tensor:
        return self.beta_coll

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
        theta = cells_flat["theta"].to(device).reshape(nH, nW).float()
        rho_t = cells_flat["rho_total"].to(device).reshape(nH, nW).float()
        is_border = cells_flat["is_border"].to(device).reshape(nH, nW).bool()
        ok = (~is_border).to(rho_peak.dtype)

        Rb = rho_peak * ok
        Rsq = Rb * Rb
        eta_seed_sq = self.eta_seed * self.eta_seed
        rho_seed = (Rsq / (Rsq + eta_seed_sq + eps)) * ok

        if self.facil_mode == "collinear":
            rho_coll = collinear_facilitation(
                rho_seed, theta,
                sigma_f=self.sigma_f,
                kappa_theta=self.kappa_theta,
                radius=self.facil_radius,
                eps=eps,
            )
        else:
            rho_coll = cocircular_facilitation(
                rho_seed, theta, sigma_f=self.sigma_f, radius=self.facil_radius, eps=eps,
            )

        e = self.beta_seed * rho_seed + self.beta_coll * rho_coll

        if self.surround_mode == "isotropic":
            S = surround_mean(
                rho_seed, nH, nW, radius=self.surround_radius, sigma=self.surround_sigma,
            )
        else:
            S = broadside_surround(
                rho_seed, theta, sigma=self.surround_sigma,
                radius=self.surround_radius, eps=eps,
            )

        e2 = e * e
        surround_sq = S * S
        denom = e2 + (self.eta * self.eta) + self.lam * surround_sq + eps
        rho = (e2 / denom) * ok
        rho_flat = rho.reshape(N)

        cf_out = dict(cells_flat)
        cf_out["rho_seed"] = rho_seed
        cf_out["rho_coll"] = (rho_coll * ok).reshape(nH, nW)
        cf_out["fac"] = (rho_coll * ok).reshape(nH, nW)
        cf_out["exc"] = (e * ok).reshape(nH, nW)
        cf_out["sur"] = (S * ok).reshape(nH, nW)
        cf_out["drive"] = Rb
        cf_out["E_rel"] = relative_energy(
            rho_t, nH, nW, eps,
            radius=self.surround_radius, sigma=self.surround_sigma,
        )
        cf_out["g_R"] = ((self.beta_seed * rho_seed + self.beta_coll * rho_coll) * ok).reshape(nH, nW)
        cf_out["g_E"] = (S * ok).reshape(nH, nW)

        branch = torch.zeros(N, device=device, dtype=torch.long)
        z1 = torch.zeros(N, 1, device=device, dtype=rho_flat.dtype)

        diags = None
        if return_surface_diags:
            ra = rho_flat[~is_border.reshape(N)]
            rc = (rho_coll * ok).reshape(-1)[~is_border.reshape(N)]
            diags = {
                "rho_coll": (rho_coll * ok).detach().cpu().numpy(),
                "iter_stats": [{
                    "rho_mean": float(rho_flat.mean().detach()),
                    "rho_max": float(rho_flat.max().detach()),
                    "mid_band_frac": float(
                        ((ra > 0.3) & (ra < 0.7)).float().mean().detach()
                    ) if ra.numel() else 0.0,
                    "n_interior": int(ok.sum().item()),
                    "fac_mean": float(rc.mean().detach()) if rc.numel() else 0.0,
                    "coll_mean": float(rc.mean().detach()) if rc.numel() else 0.0,
                    "sur_mean": float((S * ok).mean().detach()),
                }],
            }

        return rho_flat, branch, rho_flat, z1, z1, cf_out, diags


AndGateSeed = ContourSeed
CellSeed = ContourSeed
