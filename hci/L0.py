r"""L0 — split-channel harmonic projection"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from params import L0 as _L0_PARAMS
except Exception:  # pragma: no cover
    class _L0_PARAMS:  # type: ignore
        EPS = 1e-6

_L0_EPS = float(getattr(_L0_PARAMS, "EPS", 1e-6))

EtaArg = Union[float, int, np.floating, np.integer, Callable[[], float]]


def lum_axis() -> torch.Tensor:
    return torch.ones(3, dtype=torch.float32) / math.sqrt(3.0)


def chroma_basis() -> tuple[torch.Tensor, torch.Tensor]:
    L = lum_axis()
    seed = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
    B1 = seed - (seed @ L) * L
    B1 = B1 / B1.norm()
    B2 = torch.linalg.cross(L, B1)
    B2 = B2 / B2.norm()
    return B1, B2


def _softplus_inv(y: float) -> float:
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


class L0LearnedMetric(nn.Module):

    def __init__(
        self,
        *,
        learnable: bool = True,
        w_lum_init: float = 1.0,
        s_chr_init: float = 1.0,
        chr_angle_init: float = 0.0,
    ) -> None:
        super().__init__()

        L = lum_axis()
        B1, B2 = chroma_basis()
        self.register_buffer("L", L)
        self.register_buffer("B1", B1)
        self.register_buffer("B2", B2)

        w_raw = _softplus_inv(float(w_lum_init))
        s1_raw = _softplus_inv(float(s_chr_init))
        s2_raw = _softplus_inv(float(s_chr_init))
        a_raw = float(chr_angle_init)

        if learnable:
            self._w_lum_raw = nn.Parameter(torch.tensor(w_raw, dtype=torch.float32))
            self._s1_raw = nn.Parameter(torch.tensor(s1_raw, dtype=torch.float32))
            self._s2_raw = nn.Parameter(torch.tensor(s2_raw, dtype=torch.float32))
            self._chr_angle = nn.Parameter(torch.tensor(a_raw, dtype=torch.float32))
        else:
            self.register_buffer("_w_lum_raw", torch.tensor(w_raw, dtype=torch.float32))
            self.register_buffer("_s1_raw", torch.tensor(s1_raw, dtype=torch.float32))
            self.register_buffer("_s2_raw", torch.tensor(s2_raw, dtype=torch.float32))
            self.register_buffer("_chr_angle", torch.tensor(a_raw, dtype=torch.float32))


    @property
    def w_lum(self) -> torch.Tensor:
        return F.softplus(self._w_lum_raw)

    @property
    def s1(self) -> torch.Tensor:
        return F.softplus(self._s1_raw)

    @property
    def s2(self) -> torch.Tensor:
        return F.softplus(self._s2_raw)

    @property
    def chr_angle(self) -> torch.Tensor:
        return self._chr_angle

    def _rotated_chroma_basis(self) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.cos(self._chr_angle)
        s = torch.sin(self._chr_angle)
        b1 = c * self.B1 + s * self.B2
        b2 = -s * self.B1 + c * self.B2
        return b1, b2

    @property
    def W(self) -> torch.Tensor:
        b1, b2 = self._rotated_chroma_basis()
        return torch.stack(
            [self.w_lum * self.L, self.s1 * b1, self.s2 * b2], dim=0,
        )

    @property
    def M(self) -> torch.Tensor:
        W = self.W
        return W.t() @ W


    def project_diff(self, delta_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b1, b2 = self._rotated_chroma_basis()
        lum = (delta_rgb @ self.L) * self.w_lum
        c1 = (delta_rgb @ b1) * self.s1
        c2 = (delta_rgb @ b2) * self.s2
        d_lum = lum.abs()
        d_chr = (c1.square() + c2.square() + _L0_EPS).sqrt()
        return d_lum, d_chr

    def project_image(self, img_rgb: torch.Tensor) -> torch.Tensor:
        return torch.einsum("hwi,ji->hwj", img_rgb, self.W)

    def extra_repr(self) -> str:
        with torch.no_grad():
            return (
                f"w_lum={float(self.w_lum):.4f}, "
                f"s1={float(self.s1):.4f}, s2={float(self.s2):.4f}, "
                f"chr_angle={float(self._chr_angle):.4f} rad"
            )


class L0Notch(nn.Module):

    def __init__(
        self,
        *,
        half_width: int | None = None,
        omega_n_init: float | None = None,
        sigma_n_init: float | None = None,
        d_init: float | None = None,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        H = int(half_width if half_width is not None else getattr(_L0_PARAMS, "NOTCH_HALF_WIDTH", 4))
        wn = float(omega_n_init if omega_n_init is not None else getattr(_L0_PARAMS, "NOTCH_OMEGA_N_INIT", 1.0 / 8))
        sn = float(sigma_n_init if sigma_n_init is not None else getattr(_L0_PARAMS, "NOTCH_SIGMA_N_INIT", 1.0 / 32))
        dd = float(d_init if d_init is not None else getattr(_L0_PARAMS, "NOTCH_D_INIT", 0.8))
        self.H = H
        rho_omega = _logit(2.0 * wn)
        rho_sigma = _softplus_inv(sn)
        rho_d = _logit(dd)
        if learnable:
            self.rho_omega = nn.Parameter(torch.tensor(rho_omega, dtype=torch.float32))
            self.rho_sigma = nn.Parameter(torch.tensor(rho_sigma, dtype=torch.float32))
            self.rho_d = nn.Parameter(torch.tensor(rho_d, dtype=torch.float32))
        else:
            self.register_buffer("rho_omega", torch.tensor(rho_omega, dtype=torch.float32))
            self.register_buffer("rho_sigma", torch.tensor(rho_sigma, dtype=torch.float32))
            self.register_buffer("rho_d", torch.tensor(rho_d, dtype=torch.float32))

    @property
    def omega_n(self) -> torch.Tensor:
        return 0.5 * torch.sigmoid(self.rho_omega)

    @property
    def sigma_n(self) -> torch.Tensor:
        return F.softplus(self.rho_sigma)

    @property
    def d(self) -> torch.Tensor:
        return torch.sigmoid(self.rho_d)

    def build_kernel(self) -> torch.Tensor:
        H = self.H
        L = 2 * H + 1
        device = self.rho_omega.device
        dtype = self.rho_omega.dtype
        omega_n = self.omega_n.to(dtype=dtype)
        sigma_n = self.sigma_n.to(dtype=dtype)
        d = self.d.to(dtype=dtype)
        m = torch.arange(-H, H + 1, device=device, dtype=dtype)
        omega_m = m / float(L)
        N = 1.0 - d * torch.exp(-((omega_m.abs() - omega_n) ** 2) / (2.0 * sigma_n ** 2))
        n_idx = torch.arange(-H, H + 1, device=device, dtype=dtype)
        cos_basis = torch.cos(2.0 * math.pi * omega_m.unsqueeze(0) * n_idx.unsqueeze(1))
        return (N.unsqueeze(0) * cos_basis).sum(dim=1) / float(L)

    def filter_channels(self, u: torch.Tensor) -> torch.Tensor:
        h = self.build_kernel()
        return torch.stack(
            [_apply_notch_separable(u[..., i], h) for i in range(u.shape[-1])],
            dim=-1,
        )


def resolve_eta(eta: EtaArg) -> float:
    if isinstance(eta, bool):
        raise TypeError("eta must be numeric or callable, not bool")
    if isinstance(eta, (int, float)):
        return float(eta)
    if isinstance(eta, (np.integer, np.floating)):
        return float(eta)
    if callable(eta):
        return float(eta())
    raise TypeError(f"eta must be float or ()->float, got {type(eta).__name__}")


def _make_unit(offsets: list[tuple[int, int]]) -> torch.Tensor:
    u = torch.tensor([[dx, -dy] for (dy, dx) in offsets], dtype=torch.float32)
    return u / u.norm(dim=1, keepdim=True)


def _make_F(unit: torch.Tensor) -> torch.Tensor:
    th = torch.atan2(unit[:, 1], unit[:, 0])
    return torch.stack(
        [torch.cos(th), torch.sin(th), torch.cos(2 * th), torch.sin(2 * th)], dim=0
    )


def compute_valid(
    H: int, W: int, offsets: list[tuple[int, int]], device: torch.device
) -> torch.Tensor:
    N = len(offsets)
    v = torch.zeros(H, W, N, dtype=torch.bool, device=device)
    for k, (dy, dx) in enumerate(offsets):
        r0, r1 = max(0, -dy), min(H, H - dy)
        c0, c1 = max(0, -dx), min(W, W - dx)
        if r0 < r1 and c0 < c1:
            v[r0:r1, c0:c1, k] = True
    return v


def compute_interior(H: int, W: int, device: torch.device) -> torch.Tensor:
    interior = torch.ones(H, W, dtype=torch.bool, device=device)
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    return interior


def _apply_notch_separable(ch: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    H_k = (h.shape[0] - 1) // 2
    x = ch.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, [H_k, H_k, H_k, H_k], mode="replicate")
    h = h.to(device=ch.device, dtype=ch.dtype)
    h_row = h.view(1, 1, 1, -1)
    h_col = h.view(1, 1, -1, 1)
    x = F.conv2d(x, h_row)
    x = F.conv2d(x, h_col)
    return x.squeeze(0).squeeze(0)


def sf(
    a: torch.Tensor, k: int, offsets: list[tuple[int, int]], fill: float = 0.0
) -> torch.Tensor:
    dy, dx = offsets[k]
    ny, nx = -dy, -dx
    H, W = a.shape[:2]
    o = torch.full_like(a, fill)
    y0s, y1s = max(0, -ny), min(H, H - ny)
    x0s, x1s = max(0, -nx), min(W, W - nx)
    y0d, y1d = max(0, ny), min(H, H + ny)
    x0d, x1d = max(0, nx), min(W, W + nx)
    if y0s < y1s and x0s < x1s:
        o[y0d:y1d, x0d:x1d] = a[y0s:y1s, x0s:x1s]
    return o


def _compute_d(img: torch.Tensor, offsets: list[tuple[int, int]]) -> torch.Tensor:
    H, W, C = img.shape
    N = len(offsets)
    den = torch.tensor(
        max(float(C), 1.0), dtype=torch.float32, device=img.device
    ).sqrt()
    d = torch.empty(H, W, N, dtype=torch.float32, device=img.device)
    for k, (dy, dx) in enumerate(offsets):
        r0s, r1s = max(0, -dy), min(H, H - dy)
        c0s, c1s = max(0, -dx), min(W, W - dx)
        r0d, r1d = max(0, dy), min(H, H + dy)
        c0d, c1d = max(0, dx), min(W, W + dx)
        diff = img[r0s:r1s, c0s:c1s] - img[r0d:r1d, c0d:c1d]
        d[r0d:r1d, c0d:c1d, k] = (diff * diff).sum(dim=-1).sqrt() / den
        if dy > 0:
            d[:dy, :, k] = 0.0
        elif dy < 0:
            d[dy:, :, k] = 0.0
        if dx > 0:
            d[:, :dx, k] = 0.0
        elif dx < 0:
            d[:, dx:, k] = 0.0
    return d


def _compute_d_lum_chroma(
    img_rgb: torch.Tensor,
    offsets: list[tuple[int, int]],
    metric: L0LearnedMetric | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    H, W, _ = img_rgb.shape
    N = len(offsets)
    use_metric = metric is not None
    if not use_metric:
        L_ref = lum_axis().to(device=img_rgb.device, dtype=img_rgb.dtype)
        B1_ref, B2_ref = chroma_basis()
        B1_ref = B1_ref.to(device=img_rgb.device, dtype=img_rgb.dtype)
        B2_ref = B2_ref.to(device=img_rgb.device, dtype=img_rgb.dtype)

    d_lum = torch.empty(H, W, N, dtype=torch.float32, device=img_rgb.device)
    d_chr = torch.empty(H, W, N, dtype=torch.float32, device=img_rgb.device)

    for k, (dy, dx) in enumerate(offsets):
        r0s, r1s = max(0, -dy), min(H, H - dy)
        c0s, c1s = max(0, -dx), min(W, W - dx)
        r0d, r1d = max(0, dy), min(H, H + dy)
        c0d, c1d = max(0, dx), min(W, W + dx)
        if use_metric:
            diff = img_rgb[r0s:r1s, c0s:c1s] - img_rgb[r0d:r1d, c0d:c1d]
            d_l, d_c = metric.project_diff(diff)
            d_lum[r0d:r1d, c0d:c1d, k] = d_l
            d_chr[r0d:r1d, c0d:c1d, k] = d_c
        else:
            diff = img_rgb[r0s:r1s, c0s:c1s] - img_rgb[r0d:r1d, c0d:c1d]
            dL = diff @ L_ref
            dC1 = diff @ B1_ref
            dC2 = diff @ B2_ref
            d_lum[r0d:r1d, c0d:c1d, k] = dL.abs()
            d_chr[r0d:r1d, c0d:c1d, k] = (dC1.square() + dC2.square() + _L0_EPS).sqrt()
        if dy > 0:
            d_lum[:dy, :, k] = 0.0
            d_chr[:dy, :, k] = 0.0
        elif dy < 0:
            d_lum[dy:, :, k] = 0.0
            d_chr[dy:, :, k] = 0.0
        if dx > 0:
            d_lum[:, :dx, k] = 0.0
            d_chr[:, :dx, k] = 0.0
        elif dx < 0:
            d_lum[:, dx:, k] = 0.0
            d_chr[:, dx:, k] = 0.0
    return d_lum, d_chr


def _project_image_rgb(
    img_rgb: torch.Tensor,
    metric: L0LearnedMetric | None,
) -> torch.Tensor:
    if metric is not None:
        return metric.project_image(img_rgb)
    L = lum_axis().to(device=img_rgb.device, dtype=img_rgb.dtype)
    B1, B2 = chroma_basis()
    B1 = B1.to(device=img_rgb.device, dtype=img_rgb.dtype)
    B2 = B2.to(device=img_rgb.device, dtype=img_rgb.dtype)
    W = torch.stack([L, B1, B2], dim=0)
    return torch.einsum("hwi,ji->hwj", img_rgb, W)


def _compute_m_sq_lum_chroma(
    u_tilde: torch.Tensor,
    offsets: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    H, W, _ = u_tilde.shape
    N = len(offsets)
    lum = u_tilde[..., 0]
    chr = u_tilde[..., 1:]
    m_lum_sq = torch.empty(H, W, N, dtype=torch.float32, device=u_tilde.device)
    m_chr_sq = torch.empty(H, W, N, dtype=torch.float32, device=u_tilde.device)

    for k, (dy, dx) in enumerate(offsets):
        r0s, r1s = max(0, -dy), min(H, H - dy)
        c0s, c1s = max(0, -dx), min(W, W - dx)
        r0d, r1d = max(0, dy), min(H, H + dy)
        c0d, c1d = max(0, dx), min(W, W + dx)
        d_lum = lum[r0s:r1s, c0s:c1s] - lum[r0d:r1d, c0d:c1d]
        d_chr = chr[r0s:r1s, c0s:c1s] - chr[r0d:r1d, c0d:c1d]
        m_lum_sq[r0d:r1d, c0d:c1d, k] = d_lum.square()
        m_chr_sq[r0d:r1d, c0d:c1d, k] = d_chr.square().sum(dim=-1)
        if dy > 0:
            m_lum_sq[:dy, :, k] = 0.0
            m_chr_sq[:dy, :, k] = 0.0
        elif dy < 0:
            m_lum_sq[dy:, :, k] = 0.0
            m_chr_sq[dy:, :, k] = 0.0
        if dx > 0:
            m_lum_sq[:, :dx, k] = 0.0
            m_chr_sq[:, :dx, k] = 0.0
        elif dx < 0:
            m_lum_sq[:, dx:, k] = 0.0
            m_chr_sq[:, dx:, k] = 0.0
    return m_lum_sq, m_chr_sq


def _naka_squared(
    m_sq: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    eta_sq = float(eta) * float(eta)
    return m_sq / (eta_sq + m_sq)


def _naka_per_direction(
    d: torch.Tensor,
    eta: float,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    d_t = d - torch.amin(d, dim=-1, keepdim=True)
    eta_sq = float(eta) * float(eta)
    d_sq = d_t * d_t
    g = torch.tensor(float(gamma), dtype=torch.float32, device=device)
    return g * d_sq / (eta_sq + d_sq)


def compute_contrast_field(
    img: torch.Tensor,
    eta0: EtaArg,
    gamma: float,
    offsets: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    eta0_used = resolve_eta(eta0)
    gamma_used = float(gamma)
    img = img.float()
    if img.ndim == 2:
        img = img.unsqueeze(-1)
    H, W, C = img.shape
    vld = compute_valid(H, W, offsets, img.device)

    d = _compute_d(img, offsets)
    d = d - torch.amin(d, dim=-1, keepdim=True)

    eta_sq = eta0_used * eta0_used
    d_sq = d * d
    d_sum_sq = d_sq.sum(dim=-1, keepdim=True)
    gamma_t = torch.tensor(gamma_used, dtype=torch.float32, device=img.device)

    h = gamma_t * d_sq / (eta_sq + d_sum_sq)

    return h, vld, eta0_used, gamma_used


def compute_contrast_field_rgb(
    img_rgb: torch.Tensor,
    *,
    gamma: float,
    offsets: list[tuple[int, int]],
    eta_lum: EtaArg | None = None,
    eta_chr: EtaArg | None = None,
    eta0: EtaArg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if eta0 is not None:
        if eta_lum is None:
            eta_lum = eta0
        if eta_chr is None:
            eta_chr = eta0
    if eta_lum is None or eta_chr is None:
        raise TypeError(
            "compute_contrast_field_rgb requires eta_lum and eta_chr "
            "(or eta0 for both channels)."
        )
    h, vld, el, _, g, _, _, _, _, _ = compute_l0_rgb(
        img_rgb, eta_lum=eta_lum, eta_chr=eta_chr, gamma=gamma, offsets=offsets,
    )
    return h, vld, el, g


def _compute_l0_rgb_notched(
    img_rgb: torch.Tensor,
    *,
    eta_lum: float,
    eta_chr: float,
    gamma: float,
    offsets: list[tuple[int, int]],
    metric: L0LearnedMetric | None,
    notch: L0Notch,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    H, W, _ = img_rgb.shape
    device = img_rgb.device
    u = _project_image_rgb(img_rgb, metric)
    u_tilde = notch.filter_channels(u)
    m_lum_sq, m_chr_sq = _compute_m_sq_lum_chroma(u_tilde, offsets)

    h_lum = _naka_squared(m_lum_sq, eta_lum)
    h_chr = _naka_squared(m_chr_sq, eta_chr)
    h = h_lum + h_chr

    vld = compute_valid(H, W, offsets, device)
    s, h1m, h2m = compute_harmonics(h, offsets)
    _, _, h2m_lum = compute_harmonics(h_lum, offsets)
    _, _, h2m_chr = compute_harmonics(h_chr, offsets)

    if gamma != 1.0:
        g = float(gamma)
        h2m = h2m.pow(g)
        h2m_lum = h2m_lum.pow(g)
        h2m_chr = h2m_chr.pow(g)

    return h, vld, eta_lum, eta_chr, float(gamma), s, h1m, h2m, h2m_lum, h2m_chr


def compute_l0_rgb(
    img_rgb: torch.Tensor,
    *,
    eta_lum: EtaArg,
    eta_chr: EtaArg,
    gamma: float,
    offsets: list[tuple[int, int]],
    metric: L0LearnedMetric | None = None,
    notch: L0Notch | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {tuple(img_rgb.shape)}")
    img_rgb = img_rgb.float()
    H, W, _ = img_rgb.shape
    device = img_rgb.device
    eta_l = resolve_eta(eta_lum)
    eta_c = resolve_eta(eta_chr)
    gamma_used = float(gamma)

    if notch is not None:
        return _compute_l0_rgb_notched(
            img_rgb,
            eta_lum=eta_l,
            eta_chr=eta_c,
            gamma=gamma_used,
            offsets=offsets,
            metric=metric,
            notch=notch,
        )

    vld = compute_valid(H, W, offsets, device)
    d_lum, d_chr = _compute_d_lum_chroma(img_rgb, offsets, metric=metric)

    h_lum = _naka_per_direction(d_lum, eta_l, gamma_used, device)
    h_chr = _naka_per_direction(d_chr, eta_c, gamma_used, device)
    h = h_lum + h_chr

    s, h1m, h2m = compute_harmonics(h, offsets)
    _, _, h2m_lum = compute_harmonics(h_lum, offsets)
    _, _, h2m_chr = compute_harmonics(h_chr, offsets)

    return h, vld, eta_l, eta_c, gamma_used, s, h1m, h2m, h2m_lum, h2m_chr


def compute_harmonics(
    h: torch.Tensor,
    offsets: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H, W = h.shape[:2]
    N = len(offsets)
    unit = _make_unit(offsets).to(h.device)
    fm = _make_F(unit).to(h.device)
    s = (h.reshape(-1, N) @ fm.T).reshape(H, W, 4)
    h1m = (s[..., 0] ** 2 + s[..., 1] ** 2 + _L0_EPS).sqrt()
    h2m = (s[..., 2] ** 2 + s[..., 3] ** 2 + _L0_EPS).sqrt()
    return s, h1m, h2m


def compute_seed(
    s: torch.Tensor,
    h1m: torch.Tensor,
    h2m: torch.Tensor,
) -> torch.Tensor:
    return torch.stack([s[..., 2], s[..., 3], s[..., 0], s[..., 1]], dim=-1)


def load_image(
    input_dir: str, image_filename: str, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    import os
    from PIL import Image

    path = os.path.join(input_dir, image_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path!r}")
    pil = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(pil)).float().div(255.0).clamp(0, 1)
    return img.to(device)


def run_l0(
    ir: torch.Tensor,
    eta_lum: EtaArg,
    eta_chr: EtaArg,
    gamma: float,
    offsets: list[tuple[int, int]],
    verbose: bool = True,
    metric: L0LearnedMetric | None = None,
    notch: L0Notch | None = None,
) -> dict:
    H, W = ir.shape[:2]
    device = ir.device
    interior = compute_interior(H, W, device)
    border_mask = ~interior
    if verbose:
        print("L0...")
    if ir.ndim == 3 and ir.shape[-1] == 3:
        (
            contrast_field,
            vld,
            eta_l,
            eta_c,
            gamma_used,
            s,
            h1m,
            h2m,
            h2m_lum,
            h2m_chr,
        ) = compute_l0_rgb(
            ir,
            eta_lum=eta_lum,
            eta_chr=eta_chr,
            gamma=gamma,
            offsets=offsets,
            metric=metric,
            notch=notch,
        )
        eta_used_print = float(eta_l)
    else:
        contrast_field, vld, eta_u, gamma_used = compute_contrast_field(
            ir, eta_lum, gamma, offsets,
        )
        eta_l = eta_c = float(eta_u)
        eta_used_print = float(eta_u)
        s, h1m, h2m = compute_harmonics(contrast_field, offsets)
        h2m_lum = h2m.clone()
        h2m_chr = torch.zeros_like(h2m)

    h2m = h2m.clone()
    h2m[border_mask] = 0.0
    h2m_lum = h2m_lum.clone()
    h2m_lum[border_mask] = 0.0
    h2m_chr = h2m_chr.clone()
    h2m_chr[border_mask] = 0.0
    a0 = compute_seed(s, h1m, h2m)
    a0[border_mask] = 0.0
    if verbose:
        if ir.ndim == 3 and ir.shape[-1] == 3:
            extra = ""
            if metric is not None:
                with torch.no_grad():
                    extra = (
                        f"  [w_ℓ={float(metric.w_lum):.3f} "
                        f"s₁={float(metric.s1):.3f} s₂={float(metric.s2):.3f} "
                        f"θ_chr={float(metric.chr_angle):+.3f}]"
                    )
            print(
                f"  η_lum={float(eta_l):.4f}  η_chr={float(eta_c):.4f}  "
                f"γ={gamma_used:.3f}  (split RGB){extra}"
            )
        else:
            print(f"  η={eta_used_print:.4f}  γ={gamma_used:.3f}  (grayscale)")
        print(f"  excluded {border_mask.sum().item()} border pixels")
        print(f"  h2m: mean={h2m.mean().item():.4f} max={h2m.max().item():.4f}")
    return {
        "contrast_field": contrast_field,
        "vld": vld,
        "s": s,
        "h1m": h1m,
        "h2m": h2m,
        "h2m_lum": h2m_lum,
        "h2m_chr": h2m_chr,
        "a0": a0,
        "border_mask": border_mask,
        "interior": interior,
        "eta_lum_used": float(eta_l),
        "eta_chr_used": float(eta_c),
        "eta_used": eta_used_print,
        "gamma_used": gamma_used,
    }
