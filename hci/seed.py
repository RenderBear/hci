r"""Cell-grid contour seed — per-orientation-bin NR, collinear, B-weighted surround, readout"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

try:
    from params import L1, SEED
except Exception:  # pragma: no cover
    class L1:  # type: ignore
        NUM_ORIENT_BINS = 8
        KAPPA_VM_INIT = 2.0

    class SEED:  # type: ignore
        EPS = 1e-6
        ETA_Z_INIT = 0.30
        CROSS_SURROUND_RADIUS = 5
        SURROUND_SIGMA = 2.0
        RHO_STE_TAU = 0.1


_ETA_Z_INIT = float(getattr(SEED, "ETA_Z_INIT", 0.30))
_BETA_SEED_INIT = float(getattr(SEED, "BETA_SEED_INIT", 0.5))
_BETA_COLL_INIT = float(getattr(SEED, "BETA_COLL_INIT", 0.5))
_KAPPA_THETA_INIT = float(getattr(SEED, "KAPPA_THETA_INIT", 2.5))
_ETA_READOUT_INIT = float(
    getattr(SEED, "ETA_READOUT_INIT", getattr(SEED, "ETA_INIT", 0.30))
)
_LAMBDA_INIT = float(getattr(SEED, "LAMBDA_INIT", 0.5))
_SIGMA_F_INIT = float(getattr(SEED, "SIGMA_F_INIT", 1.3))
_SIGMA_S_INIT = float(getattr(SEED, "SIGMA_S_INIT", getattr(SEED, "SURROUND_SIGMA", 2.0)))
_KAPPA_VM_INIT = float(getattr(L1, "KAPPA_VM_INIT", 2.0))
_K = int(getattr(L1, "NUM_ORIENT_BINS", 8))
_FACIL_RADIUS = int(getattr(SEED, "FACIL_RADIUS", 2))
_CROSS_SURROUND_RADIUS = int(
    getattr(SEED, "CROSS_SURROUND_RADIUS", getattr(SEED, "SURROUND_RADIUS", 5))
)
_SURROUND_SIGMA = float(getattr(SEED, "SURROUND_SIGMA", 2.0))
_RHO_STE_TAU = float(getattr(SEED, "RHO_STE_TAU", 0.1))


def _inv_softplus(x: float) -> float:
    x = max(float(x), 1e-8)
    if x > 20.0:
        return x
    return math.log(math.expm1(x))


def orientation_bin_centers(K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(K, device=device, dtype=dtype) * (math.pi / float(K))


def orientation_B_matrix(K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = orientation_bin_centers(K, device, dtype).view(K, 1)
    d = t - t.t()
    return torch.sin(d) ** 2


def _surround_kernel(
    radius: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    *,
    exclude_center: bool = True,
) -> torch.Tensor:
    size = 2 * int(radius) + 1
    coords = torch.arange(size, device=device, dtype=dtype) - float(radius)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    g = torch.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) ** 2))
    if exclude_center:
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
    dev, dtype = field.device, field.dtype
    grid = field.reshape(nH, nW).to(dtype=dtype)
    k = _surround_kernel(radius, sigma, dev, dtype, exclude_center=True).unsqueeze(0).unsqueeze(0)
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
    radius: int = _CROSS_SURROUND_RADIUS,
    sigma: float = _SURROUND_SIGMA,
) -> torch.Tensor:
    nb = surround_mean(rho_total, nH, nW, radius=radius, sigma=sigma)
    grid = rho_total.reshape(nH, nW)
    return grid / (float(eps) + nb)


def _shift(t: torch.Tensor, dy: int, dx: int, pad: int) -> torch.Tensor:
    nH, nW = t.shape
    tp = Fn.pad(t[None, None], (pad, pad, pad, pad), mode="reflect").squeeze(0).squeeze(0)
    r0 = pad + dy
    c0 = pad + dx
    return tp[r0 : r0 + nH, c0 : c0 + nW]


def _shift_hwk(t: torch.Tensor, dy: int, dx: int, pad: int) -> torch.Tensor:
    nH, nW, K = t.shape
    out = []
    for k in range(K):
        out.append(_shift(t[:, :, k], dy, dx, pad))
    return torch.stack(out, dim=-1)


def collinear_facilitation(
    R: torch.Tensor,
    theta: torch.Tensor,
    *,
    sigma_f: torch.Tensor,
    kappa_theta: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
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


def _reflect_conv_hwk(
    x_hwk: torch.Tensor,
    weight: torch.Tensor,
    *,
    groups: int = 1,
) -> torch.Tensor:
    k = int(weight.shape[-1])
    pad = k // 2
    x = x_hwk.permute(2, 0, 1).unsqueeze(0)
    x_pad = Fn.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = Fn.conv2d(x_pad, weight, groups=groups)
    return y.squeeze(0).permute(1, 2, 0)


def _spatial_offsets(radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    offs = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            offs.append((dy, dx))
    if not offs:
        return torch.zeros(0, 2, device=device, dtype=dtype)
    return torch.tensor(offs, device=device, dtype=dtype)


def _gaussian_exponent(
    d2: torch.Tensor,
    sigma_sq: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(-d2 / (2.0 * sigma_sq.clamp_min(1e-12)))


def collinear_facilitation_bins(
    rho_nr: torch.Tensor,
    bar_theta: torch.Tensor,
    *,
    sigma_f: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    nH, nW, K = rho_nr.shape
    dtype, dev = rho_nr.dtype, rho_nr.device
    ct = torch.cos(bar_theta)
    st = torch.sin(bar_theta)
    sigma_sq = (sigma_f * sigma_f).clamp_min(eps)
    ks = 2 * int(radius) + 1
    center = int(radius)
    kernel = torch.zeros(K, 1, ks, ks, device=dev, dtype=dtype)
    offs = _spatial_offsets(radius, dev, dtype)
    if offs.numel() == 0:
        return torch.zeros_like(rho_nr)
    dy = offs[:, 0]
    dx = offs[:, 1]
    d2 = (dy * dy + dx * dx).to(dtype=dtype)
    g = _gaussian_exponent(d2, sigma_sq)
    t_proj = dy.to(dtype=dtype).view(1, -1) * ct.view(K, 1) + dx.to(dtype=dtype).view(1, -1) * st.view(K, 1)
    w = g.view(1, -1) * (t_proj * t_proj) / (d2.view(1, -1) + eps)
    for m in range(offs.shape[0]):
        oy = int(dy[m].item()) + center
        ox = int(dx[m].item()) + center
        kernel[:, 0, oy, ox] = w[:, m]
    numer = _reflect_conv_hwk(rho_nr, kernel, groups=K)
    denom = kernel.sum(dim=(-2, -1)).view(1, 1, K) + eps
    return torch.relu(numer / denom)


def surround_bins_B_weighted(
    rho_nr: torch.Tensor,
    B: torch.Tensor,
    *,
    sigma_s: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    nH, nW, K = rho_nr.shape
    dtype, dev = rho_nr.dtype, rho_nr.device
    sigma_sq = (sigma_s * sigma_s).clamp_min(eps)
    ks = 2 * int(radius) + 1
    center = int(radius)
    B = B.to(dtype=dtype, device=dev)
    col_sum = B.sum(dim=0)

    spatial = torch.zeros(1, 1, ks, ks, device=dev, dtype=dtype)
    w_acc = torch.zeros((), device=dev, dtype=dtype)
    offs = _spatial_offsets(radius, dev, dtype)
    if offs.numel() == 0:
        return torch.zeros_like(rho_nr)
    dy = offs[:, 0]
    dx = offs[:, 1]
    d2 = (dy * dy + dx * dx).to(dtype=dtype)
    g = _gaussian_exponent(d2, sigma_sq)
    w_acc = g.sum()
    for m in range(offs.shape[0]):
        oy = int(dy[m].item()) + center
        ox = int(dx[m].item()) + center
        spatial[0, 0, oy, ox] = g[m]

    smoothed = _reflect_conv_hwk(rho_nr, spatial.expand(K, 1, ks, ks), groups=K)
    numer = torch.matmul(smoothed, B)
    denom = w_acc * col_sum.view(1, 1, K) + eps
    return numer / denom


def rho_ste_hard_max(rho_out: torch.Tensor, tau: float) -> torch.Tensor:
    w = Fn.softmax(rho_out / float(tau), dim=-1)
    hard = rho_out.max(dim=-1).values
    soft = (w * rho_out).sum(dim=-1)
    return hard - soft.detach() + soft


def soft_theta_double_angle(
    weights: torch.Tensor,
    bar_theta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    w = weights.clamp_min(0.0)
    kd = w.dim()
    br = bar_theta.to(device=w.device, dtype=w.dtype)
    c2 = torch.cos(2.0 * br).reshape(*([1] * (kd - 1)), -1)
    s2 = torch.sin(2.0 * br).reshape(*([1] * (kd - 1)), -1)
    num_c = (w * c2).sum(dim=-1)
    num_s = (w * s2).sum(dim=-1)
    return 0.5 * torch.atan2(num_s, num_c + float(eps))


class ContourSeed(nn.Module):

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
        sigma_s_init: float = _SIGMA_S_INIT,
        kappa_vm_init: float = _KAPPA_VM_INIT,
        num_orient_bins: int = _K,
        facil_radius: int = _FACIL_RADIUS,
        facil_mode: str = str(getattr(SEED, "FACIL_MODE", "collinear")),
        cross_surround_radius: int = _CROSS_SURROUND_RADIUS,
        surround_sigma: float = float(getattr(SEED, "SURROUND_SIGMA", 2.0)),
        surround_mode: str = str(getattr(SEED, "SURROUND_MODE", "broadside")),
        rho_ste_tau: float = _RHO_STE_TAU,
        *,
        surround_radius: int | None = None,
        **kw,
    ):
        super().__init__()
        _ = kw
        if surround_radius is not None:
            cross_surround_radius = int(surround_radius)
        self.eps = float(eps)
        self.K = int(num_orient_bins)
        self.facil_radius = int(facil_radius)
        self.facil_mode = str(facil_mode)
        self.cross_surround_radius = int(cross_surround_radius)
        self.surround_sigma = float(surround_sigma)
        self.surround_mode = str(surround_mode)
        self.rho_ste_tau = float(rho_ste_tau)

        dev0 = torch.device("cpu")
        tb = orientation_bin_centers(self.K, dev0, torch.float32)
        self.register_buffer("theta_bins", tb, persistent=False)
        self.register_buffer("B_orth", orientation_B_matrix(self.K, dev0, torch.float32), persistent=False)

        self._kappa_vm_raw = nn.Parameter(torch.tensor(_inv_softplus(kappa_vm_init)))
        self._eta_z_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_z_init)))
        self._beta_seed_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_seed_init)))
        self._beta_coll_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_coll_init)))
        self._kappa_theta_raw = nn.Parameter(torch.tensor(_inv_softplus(kappa_theta_init)))
        self._eta_readout_raw = nn.Parameter(torch.tensor(_inv_softplus(eta_readout_init)))
        self._lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(lambda_init)))
        self._sigma_f_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_f_init)))
        self._sigma_s_raw = nn.Parameter(torch.tensor(_inv_softplus(sigma_s_init)))

    @property
    def kappa_vm(self) -> torch.Tensor:
        return Fn.softplus(self._kappa_vm_raw).view(())

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
    def sigma_s(self) -> torch.Tensor:
        return Fn.softplus(self._sigma_s_raw).view(()).clamp_min(0.3)

    @property
    def surround_radius(self) -> int:
        return self.cross_surround_radius

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

        is_border = cells_flat["is_border"].to(device).reshape(nH, nW).bool()
        ok = (~is_border).to(torch.float32)

        if "rho_bin" not in cells_flat:
            return self._forward_legacy(cells_flat, return_surface_diags, device, nH, nW, N, eps, ok, is_border)

        rho_bin = cells_flat["rho_bin"].to(device).float()
        if rho_bin.dim() == 2:
            rho_bin = rho_bin.reshape(nH, nW, -1)
        K = rho_bin.shape[-1]
        if K != self.K:
            raise ValueError(f"cells_flat rho_bin last dim K={K} != seed.K={self.K}")

        ax_bin = cells_flat["ax_bin"].to(device).float()
        ay_bin = cells_flat["ay_bin"].to(device).float()
        if ax_bin.dim() == 2:
            ax_bin = ax_bin.reshape(nH, nW, K)
            ay_bin = ay_bin.reshape(nH, nW, K)

        rho_t = cells_flat["rho_total"].to(device).reshape(nH, nW).float()
        bar_theta = self.theta_bins.to(device=device, dtype=rho_bin.dtype)
        B = self.B_orth.to(device=device, dtype=rho_bin.dtype)

        R = rho_bin * ok.unsqueeze(-1)
        Rsq = R * R
        eta_z_sq = self.eta_z * self.eta_z
        rho_nr = (Rsq / (Rsq + eta_z_sq + eps)) * ok.unsqueeze(-1)

        rho_coll = collinear_facilitation_bins(
            rho_nr,
            bar_theta,
            sigma_f=self.sigma_f,
            radius=self.facil_radius,
            eps=eps,
        )

        S = surround_bins_B_weighted(
            rho_nr,
            B,
            sigma_s=self.sigma_s,
            radius=self.cross_surround_radius,
            eps=eps,
        )

        e = self.beta_seed * rho_nr + self.beta_coll * rho_coll
        e2 = e * e
        eta_r = self.eta_readout * self.eta_readout
        pool = self.lam * (S * S)
        denom_r = e2 + eta_r + pool + eps
        rho_out = (e2 / denom_r) * ok.unsqueeze(-1)

        rho_cell = rho_ste_hard_max(rho_out, self.rho_ste_tau)
        theta_grid = soft_theta_double_angle(rho_out, bar_theta, eps)
        k_star = rho_out.argmax(dim=-1)
        ax_star = torch.gather(ax_bin, dim=-1, index=k_star.unsqueeze(-1)).squeeze(-1)
        ay_star = torch.gather(ay_bin, dim=-1, index=k_star.unsqueeze(-1)).squeeze(-1)

        rho_flat = rho_cell.reshape(N)
        theta_flat = theta_grid.reshape(N, 1)

        cf_out = dict(cells_flat)
        cf_out["rho_nr"] = rho_nr.mean(dim=-1)
        cf_out["rho_seed"] = rho_nr.mean(dim=-1)
        cf_out["rho_nr_bins"] = rho_nr
        cf_out["rho_coll"] = (rho_coll * ok.unsqueeze(-1)).mean(dim=-1)
        cf_out["rho_coll_bins"] = rho_coll * ok.unsqueeze(-1)
        cf_out["fac"] = cf_out["rho_coll"]
        cf_out["exc"] = (e * ok.unsqueeze(-1)).mean(dim=-1)
        cf_out["sur"] = (S * ok.unsqueeze(-1)).mean(dim=-1)
        cf_out["sur_bins"] = S * ok.unsqueeze(-1)
        cf_out["drive"] = rho_bin.mean(dim=-1) * ok
        cf_out["E_rel"] = relative_energy(
            rho_t, nH, nW, eps,
            radius=self.cross_surround_radius,
            sigma=float(self.surround_sigma),
        )
        cf_out["g_R"] = ((self.beta_seed * rho_nr + self.beta_coll * rho_coll) * ok.unsqueeze(-1)).mean(
            dim=-1,
        )
        cf_out["g_E"] = cf_out["sur"]
        cf_out["rho_out_bins"] = rho_out
        cf_out["theta"] = theta_flat
        cf_out["cx_z2"] = ax_star.reshape(N)
        cf_out["cy_z2"] = ay_star.reshape(N)

        branch = torch.zeros(N, device=device, dtype=torch.long)
        z1 = torch.zeros(N, 1, device=device, dtype=rho_flat.dtype)

        diags = None
        if return_surface_diags:
            ra = rho_flat[~is_border.reshape(N)]
            rc = cf_out["rho_coll"].reshape(-1)[~is_border.reshape(N)]
            diags = {
                "rho_coll": cf_out["rho_coll"].detach().cpu().numpy(),
                "iter_stats": [{
                    "rho_mean": float(rho_flat.mean().detach()),
                    "rho_max": float(rho_flat.max().detach()),
                    "mid_band_frac": float(
                        ((ra > 0.3) & (ra < 0.7)).float().mean().detach()
                    ) if ra.numel() else 0.0,
                    "n_interior": int(ok.sum().item()),
                    "fac_mean": float(rc.mean().detach()) if rc.numel() else 0.0,
                    "coll_mean": float(rc.mean().detach()) if rc.numel() else 0.0,
                    "sur_mean": float(cf_out["sur"].mean().detach()),
                }],
            }

        return rho_flat, branch, rho_flat, z1, z1, cf_out, diags

    def _forward_legacy(
        self,
        cells_flat,
        return_surface_diags: bool,
        device: torch.device,
        nH: int,
        nW: int,
        N: int,
        eps: float,
        ok: torch.Tensor,
        is_border: torch.Tensor,
    ):
        rho_peak = cells_flat["rho_peak"].to(device).reshape(nH, nW).float()
        theta = cells_flat["theta"].to(device).reshape(nH, nW).float()
        rho_t = cells_flat["rho_total"].to(device).reshape(nH, nW).float()

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
                radius=self.cross_surround_radius,
                sigma=self.surround_sigma,
            )
        else:
            S = broadside_surround(
                rho_nr,
                theta,
                sigma=self.surround_sigma,
                radius=self.cross_surround_radius,
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
            radius=self.cross_surround_radius,
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
