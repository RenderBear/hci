r"""Cell-grid contour seed — η_z NR on |Z|, then collinear + surround divisive readout.

L1 supplies ``rho_peak`` = |Σ z₂| and θ = ½ arg(Σ z₂). With R = |Z|·ok:

  ρ_nr(c) = R(c)² / (R(c)² + η_z² + ε) · ok(c)     (learned η_z)

  ρ_coll, e, S, then  ρ(c) = e² / (e² + η_readout² + λ·S² + ε) · ok   (learned β, κ_θ, η_readout, λ, σ_f)

``cf_out`` exposes ``rho_nr`` (row‑1 diagnostics) and the returned flat ``ρ`` is the
final map for the renderer. ``relative_energy`` reuses the same surround neighborhood as the seed readout.
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
        SURROUND_RADIUS = 5
        SURROUND_SIGMA = 2.0


_ETA_Z_INIT = float(getattr(SEED, "ETA_Z_INIT", 0.30))
_BETA_SEED_INIT = float(getattr(SEED, "BETA_SEED_INIT", 0.5))
_BETA_COLL_INIT = float(getattr(SEED, "BETA_COLL_INIT", 0.5))
_KAPPA_THETA_INIT = float(getattr(SEED, "KAPPA_THETA_INIT", 2.5))
_ETA_READOUT_INIT = float(
    getattr(SEED, "ETA_READOUT_INIT", getattr(SEED, "ETA_INIT", 0.30))
)
_LAMBDA_INIT = float(getattr(SEED, "LAMBDA_INIT", 0.5))
_SIGMA_F_INIT = float(getattr(SEED, "SIGMA_F_INIT", 1.3))
_FACIL_RADIUS = int(getattr(SEED, "FACIL_RADIUS", 2))
_SURROUND_RADIUS = int(getattr(SEED, "SURROUND_RADIUS", 5))
_SURROUND_SIGMA = float(getattr(SEED, "SURROUND_SIGMA", 2.0))


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
    radius: int,
    sigma: float,
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


def _shift(t: torch.Tensor, dy: int, dx: int, pad: int) -> torch.Tensor:
    nH, nW = t.shape
    tp = Fn.pad(t[None, None], (pad, pad, pad, pad), mode="reflect").squeeze(0).squeeze(0)
    r0 = pad + dy
    c0 = pad + dx
    return tp[r0 : r0 + nH, c0 : c0 + nW]


def collinear_facilitation(
    R: torch.Tensor,
    theta: torch.Tensor,
    *,
    sigma_f: torch.Tensor,
    kappa_theta: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    """ρ_coll(c) = relu( Σ_𝒩 w·R'·a_κ(θ'−θ) / Σ_𝒩 w )."""
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
    """ρ_nr = R²/(R²+η_z²); then collinear + surround + ρ = e²/(e²+η_readout²+λS²)."""

    def __init__(
        self,
        eps: float = SEED.EPS,
        eta_z_init: float = _ETA_Z_INIT,
        beta_seed_init: float = _BETA_SEED_INIT,
        beta_coll_init: float = _BETA_COLL_INIT,
        kappa_theta_init: float = _KAPPA_THETA_INIT,
        eta_readout_init: float = _ETA_READOUT_INIT,
        lambda_init: float = _LAMBDA_INIT,
        sigma_f_init: float = _SIGMA_F_INIT,
        facil_radius: int = _FACIL_RADIUS,
        facil_mode: str = str(getattr(SEED, "FACIL_MODE", "collinear")),
        surround_radius: int = int(getattr(SEED, "SURROUND_RADIUS", 5)),
        surround_sigma: float = float(getattr(SEED, "SURROUND_SIGMA", 2.0)),
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
        self._eta_z_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_z_init)))
        self._beta_seed_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_seed_init)))
        self._beta_coll_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_coll_init)))
        self._kappa_theta_raw = nn.Parameter(torch.tensor(_inv_softplus(kappa_theta_init)))
        self._eta_readout_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_readout_init)))
        self._lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(lambda_init)))
        self._sigma_f_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_f_init)))

    @property
    def eta_z(self) -> torch.Tensor:
        return Fn.softplus(self._eta_z_raw).view(())

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
    def eta_readout(self) -> torch.Tensor:
        return Fn.softplus(self._eta_readout_raw).view(())

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
        eta_z_sq = self.eta_z * self.eta_z
        rho_nr = (Rsq / (Rsq + eta_z_sq + eps)) * ok

        if self.facil_mode == "collinear":
            rho_coll = collinear_facilitation(
                rho_nr,
                theta,
                sigma_f=self.sigma_f,
                kappa_theta=self.kappa_theta,
                radius=self.facil_radius,
                eps=eps,
            )
        else:
            rho_coll = cocircular_facilitation(
                rho_nr,
                theta,
                sigma_f=self.sigma_f,
                radius=self.facil_radius,
                eps=eps,
            )

        e = self.beta_seed * rho_nr + self.beta_coll * rho_coll

        if self.surround_mode == "isotropic":
            S = surround_mean(
                rho_nr,
                nH,
                nW,
                radius=self.surround_radius,
                sigma=self.surround_sigma,
            )
        else:
            S = broadside_surround(
                rho_nr,
                theta,
                sigma=self.surround_sigma,
                radius=self.surround_radius,
                eps=eps,
            )

        e2 = e * e
        denom_readout = e2 + (self.eta_readout * self.eta_readout) + self.lam * (S * S) + eps
        rho = (e2 / denom_readout) * ok
        rho_flat = rho.reshape(N)

        cf_out = dict(cells_flat)
        cf_out["rho_nr"] = rho_nr
        cf_out["rho_seed"] = rho_nr
        cf_out["rho_coll"] = (rho_coll * ok).reshape(nH, nW)
        cf_out["fac"] = (rho_coll * ok).reshape(nH, nW)
        cf_out["exc"] = (e * ok).reshape(nH, nW)
        cf_out["sur"] = (S * ok).reshape(nH, nW)
        cf_out["drive"] = Rb
        cf_out["E_rel"] = relative_energy(
            rho_t, nH, nW, eps,
            radius=self.surround_radius,
            sigma=self.surround_sigma,
        )
        cf_out["g_R"] = ((self.beta_seed * rho_nr + self.beta_coll * rho_coll) * ok).reshape(nH, nW)
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
