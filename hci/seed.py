r"""Cell-grid contour seed — association-field pooling of coherence.

Coherence R is the base signal. Collinear support GATES it (rather than topping it
up), and an isotropic surround over the resulting drive normalises it DIVISIVELY —
so suppression is relative to the local neighbourhood, never a flat DC offset.

From L1 moments (R = |Z₂|/Σ|z₂|, θ = ½ arg Z₂, ρ_total):

  double-angle field      q(c) = R(c) · e^{i 2θ(c)}            (u = R cos2θ, v = R sin2θ)

  collinear facilitation  F(c) = relu( Σ_𝒩 w·R'·cos2(θ'−θ) / Σ_𝒩 w )
                          w(δ) = G(|δ|) · pos(δ;θ_c),  pos = (δ·t̂_c)²/|δ|²
                          (t̂_c = (cosθ, sinθ) tangent — renderer convention, so the
                           facilitation axis coincides with the splat's σ∥ axis)

  gated excitation        e(c) = R(c) · (β + κ·F(c))
                          (β = coherence-alone floor; no collinear support ⇒ ~β·R,
                           which removes the R≈⟨R⟩ noise pedestal at the source)

  selective surround      S(c) = ⟨e⟩_𝒩   (center-excluded Gaussian over the drive;
                           spatially varying because e is structured)

  divisive readout        ρ(c) = e² / (e² + η² + (λ·S)²) · ok(c)

Learned (softplus-positive): β (floor), κ (facilitation gain), λ (surround gain),
η (semi-saturation), σ_f (facilitation length, cell units).
θ passes through from L1 unchanged. ``branch`` is a vestigial all-zero index.
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
        # legacy AND-gate fields (ignored by ContourSeed, kept for ctor compat)
        R0_INIT = 0.45
        A_INIT = 12.0
        B_INIT = 5.0


# ── init defaults (read from params if present, else literal fallbacks) ──
_BETA_INIT = float(getattr(SEED, "BETA_INIT", 0.30))     # coherence-alone floor (no collinear support)
_KAPPA_INIT = float(getattr(SEED, "KAPPA_INIT", 3.0))    # collinear facilitation gain
_LAMBDA_INIT = float(getattr(SEED, "LAMBDA_INIT", 1.5))  # divisive surround gain
_ETA_INIT = float(getattr(SEED, "ETA_INIT", 0.30))       # semi-saturation (empty-surround knee)
_SIGMA_F_INIT = float(getattr(SEED, "SIGMA_F_INIT", 1.3))  # facilitation length (cells)
_FACIL_RADIUS = int(getattr(SEED, "FACIL_RADIUS", 2))      # neighbourhood radius (cells)


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


# ═══════════════════════════════════════════════════════════════
# Isotropic surround (center-excluded Gaussian) — suppression term
# ═══════════════════════════════════════════════════════════════

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
    g[int(radius), int(radius)] = 0.0          # center-excluded
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


# Legacy names kept so existing diagnostics keep importing cleanly.
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
    """E_rel(c) = ρ_total / (ε + ⟨ρ_total⟩_𝒩). Retained for (R, E_rel) diagnostics."""
    nb = surround_mean(rho_total, nH, nW, radius=radius, sigma=sigma)
    grid = rho_total.reshape(nH, nW)
    return grid / (float(eps) + nb)


# ═══════════════════════════════════════════════════════════════
# Oriented neighbour shift (reflect-padded)
# ═══════════════════════════════════════════════════════════════

def _shift(t: torch.Tensor, dy: int, dx: int, pad: int) -> torch.Tensor:
    """Return value at (i+dy, j+dx) aligned to (i, j); reflect-padded by ``pad``."""
    nH, nW = t.shape
    tp = Fn.pad(t[None, None], (pad, pad, pad, pad), mode="reflect").squeeze(0).squeeze(0)
    r0 = pad + dy
    c0 = pad + dx
    return tp[r0:r0 + nH, c0:c0 + nW]


# ═══════════════════════════════════════════════════════════════
# Collinear facilitation — oriented double-angle pooling (one pass)
# ═══════════════════════════════════════════════════════════════

def collinear_facilitation(
    R: torch.Tensor,
    theta: torch.Tensor,
    *,
    sigma_f: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    """F(c) = relu( Σ_𝒩 w·R'·cos2(θ'−θ) / Σ_𝒩 w ),  w = G(|δ|)·pos(δ;θ_c).

    Co-oriented neighbours lying along the cell's tangent add constructively;
    random-orientation neighbours cancel in the vector sum. Shapes (nH, nW).
    """
    nH, nW = R.shape
    dtype, dev = R.dtype, R.device
    c2, s2 = torch.cos(2.0 * theta), torch.sin(2.0 * theta)
    ct, st = torch.cos(theta), torch.sin(theta)          # tangent t̂ = (cosθ, sinθ)
    u, v = R * c2, R * s2                                 # q = R e^{i2θ}

    sig2 = (2.0 * sigma_f * sigma_f).clamp_min(eps)
    pad = int(radius)

    Vu = torch.zeros_like(R)
    Vv = torch.zeros_like(R)
    Wm = torch.zeros_like(R)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            d2 = float(dy * dy + dx * dx)
            # differentiable in σ_f:
            G = torch.exp(torch.tensor(-d2, dtype=dtype, device=dev) / sig2)
            # squared projection of δ onto the per-cell tangent  → collinearity in [0,1]
            t_proj = float(dy) * ct + float(dx) * st
            pos = (t_proj * t_proj) / (d2 + eps)
            w = G * pos                                  # (nH, nW)
            Vu = Vu + w * _shift(u, dy, dx, pad)
            Vv = Vv + w * _shift(v, dy, dx, pad)
            Wm = Wm + w
    fac_raw = (c2 * Vu + s2 * Vv) / (Wm + eps)           # ⟨R'·cos2(θ'−θ)⟩_w
    return torch.relu(fac_raw)


def broadside_surround(
    e: torch.Tensor,
    theta: torch.Tensor,
    *,
    sigma: float,
    radius: int,
    eps: float,
) -> torch.Tensor:
    """⟨e⟩ over a center-excluded surround weighted toward the NORMAL of θ.

    w_perp(δ) = G(|δ|) · (δ·n̂_c)²/|δ|²  — the complement of the collinear
    facilitation lobe. A thin contour's own continuation lies along the tangent
    and is down-weighted here, so the contour does not suppress itself; flanking
    (broadside) texture still does. Shapes (nH, nW).
    """
    nH, nW = e.shape
    dtype, dev = e.dtype, e.device
    ct, st = torch.cos(theta), torch.sin(theta)          # tangent t̂ = (cosθ, sinθ)
    sig2 = 2.0 * float(sigma) * float(sigma)
    pad = int(radius)
    Se = torch.zeros_like(e)
    Wm = torch.zeros_like(e)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            d2 = float(dy * dy + dx * dx)
            G = math.exp(-d2 / sig2)
            n_proj = -float(dy) * st + float(dx) * ct     # δ · n̂
            perp = (n_proj * n_proj) / (d2 + eps)         # broadside fraction in [0,1]
            w = G * perp
            Se = Se + w * _shift(e, dy, dx, pad)
            Wm = Wm + w
    return Se / (Wm + eps)


# ═══════════════════════════════════════════════════════════════
# Seed module
# ═══════════════════════════════════════════════════════════════

class ContourSeed(nn.Module):
    """Divisive association-field gate.

      e   = R · (β + κ·F)                     facilitation-GATED coherence (β = floor)
      S   = ⟨e⟩_𝒩                             selective surround over structured drive
      ρ   = e² / (e² + η² + (λ·S)²) · ok      divisive normalisation (relative, not DC)
    """

    def __init__(
        self,
        eps: float = SEED.EPS,
        beta_init: float = _BETA_INIT,
        kappa_init: float = _KAPPA_INIT,
        lambda_init: float = _LAMBDA_INIT,
        eta_init: float = _ETA_INIT,
        sigma_f_init: float = _SIGMA_F_INIT,
        facil_radius: int = _FACIL_RADIUS,
        surround_radius: int = SEED.SURROUND_RADIUS,
        surround_sigma: float = SEED.SURROUND_SIGMA,
        surround_mode: str = str(getattr(SEED, "SURROUND_MODE", "broadside")),
        **kw,  # absorbs legacy R0_init / a_init / b_init etc.
    ):
        super().__init__()
        _ = kw
        self.eps = float(eps)
        self.facil_radius = int(facil_radius)
        self.surround_radius = int(surround_radius)
        self.surround_sigma = float(surround_sigma)
        self.surround_mode = str(surround_mode)  # "broadside" (oriented) | "isotropic"
        self._beta_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
        self._kappa_raw = nn.Parameter(torch.tensor(_inv_softplus(kappa_init)))
        self._lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(lambda_init)))
        self._eta_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_init)))
        self._sigma_f_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_f_init)))

    @property
    def beta(self) -> torch.Tensor:
        return Fn.softplus(self._beta_raw).view(())

    @property
    def kappa(self) -> torch.Tensor:
        return Fn.softplus(self._kappa_raw).view(())

    @property
    def lam(self) -> torch.Tensor:
        return Fn.softplus(self._lambda_raw).view(())

    @property
    def eta(self) -> torch.Tensor:
        return Fn.softplus(self._eta_raw).view(())

    @property
    def sigma_f(self) -> torch.Tensor:
        return Fn.softplus(self._sigma_f_raw).view(()).clamp_min(0.3)

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

        R = cells_flat["coherence_R"].to(device).reshape(nH, nW).float()
        theta = cells_flat["theta"].to(device).reshape(nH, nW).float()
        rho_t = cells_flat["rho_total"].to(device).reshape(nH, nW).float()
        is_border = cells_flat["is_border"].to(device).reshape(nH, nW).bool()
        ok = (~is_border).to(R.dtype)

        Rb = R * ok  # don't let border coherence leak into the pooling

        F = collinear_facilitation(
            Rb, theta, sigma_f=self.sigma_f, radius=self.facil_radius, eps=eps,
        )

        # Excitation: coherence GATED (not topped up) by collinear support.
        # A high-R cell with no collinear neighbours (a noise fluctuation) yields
        # only β·R, so the R≈⟨R⟩ noise pedestal is removed at the source.
        e = Rb * (self.beta + self.kappa * F)

        # Selective surround: pool the *structured* excitation, applied DIVISIVELY.
        # ⟨e⟩_𝒩 varies spatially (low in empty regions, high in clutter), so this
        # is a relative normalisation, not the flat DC offset that ⟨R⟩ produced.
        # "broadside" pools off the tangent so a contour doesn't suppress itself.
        if self.surround_mode == "isotropic":
            S = surround_mean(
                e, nH, nW, radius=self.surround_radius, sigma=self.surround_sigma,
            )
        else:
            S = broadside_surround(
                e, theta, sigma=self.surround_sigma,
                radius=self.surround_radius, eps=eps,
            )

        e2 = e * e
        denom = e2 + (self.eta * self.eta) + (self.lam * S) * (self.lam * S) + eps
        rho = (e2 / denom) * ok

        rho_flat = rho.reshape(N)

        cf_out = dict(cells_flat)
        cf_out["fac"] = (F * ok).reshape(nH, nW)
        cf_out["exc"] = (e * ok).reshape(nH, nW)
        cf_out["sur"] = (S * ok).reshape(nH, nW)
        cf_out["drive"] = (e * ok).reshape(nH, nW)
        cf_out["rho_seed"] = rho
        # legacy keys kept so existing diagnostics don't KeyError
        cf_out["E_rel"] = relative_energy(
            rho_t, nH, nW, eps,
            radius=self.surround_radius, sigma=self.surround_sigma,
        )
        cf_out["g_R"] = ((self.beta + self.kappa * F) * ok).reshape(nH, nW)
        cf_out["g_E"] = (S * ok).reshape(nH, nW)

        branch = torch.zeros(N, device=device, dtype=torch.long)
        z1 = torch.zeros(N, 1, device=device, dtype=rho_flat.dtype)

        diags = None
        if return_surface_diags:
            ra = rho_flat[~is_border.reshape(N)]
            diags = {"iter_stats": [{
                "rho_mean": float(rho_flat.mean().detach()),
                "rho_max": float(rho_flat.max().detach()),
                "mid_band_frac": float(
                    ((ra > 0.3) & (ra < 0.7)).float().mean().detach()
                ) if ra.numel() else 0.0,
                "n_interior": int(ok.sum().item()),
                "fac_mean": float((F * ok).mean().detach()),
                "sur_mean": float((S * ok).mean().detach()),
            }]}

        return rho_flat, branch, rho_flat, z1, z1, cf_out, diags


# Aliases so existing imports (AndGateSeed, CellSeed) keep working.
AndGateSeed = ContourSeed
CellSeed = ContourSeed