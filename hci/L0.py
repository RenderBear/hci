r"""L0 — split-channel harmonic projection (GPU, PyTorch).

RGB: orthonormal luminance / chrominance split
  L = (R+G+B)/3,   C = I − L·1  (per-pixel 3-vector).

8-connected offsets δ_k with bearing φ_k:
  d_k^lum = |L(p) − L(p+δ_k)|,   d_k^chr = ‖C(p) − C(p+δ_k)‖₂

Per-pixel minimum across directions, then independent Naka–Rushton per channel:
  h_k^lum = γ (d̃_k^lum)² / (η_lum² + (d̃_k^lum)²),   d̃_k^lum = d_k^lum − min_j d_j^lum
  h_k^chr = γ (d̃_k^chr)² / (η_chr² + (d̃_k^chr)²)

  h_k = h_k^lum + h_k^chr,   z_n = Σ_k h_k e^{inφ_k}  (n ∈ {1,2} via ``compute_harmonics``).

Second-harmonic magnitudes from split fields (renderer + L1 photometry):
  h_{2m}^lum = |Σ_k h_k^lum e^{2iφ_k}|,   h_{2m}^chr = |Σ_k h_k^chr e^{2iφ_k}|.

η_lum and η_chr are read from ``params.L0`` at precompute time only; they are not model
parameters and are not updated during training.

Optional learned RGB metric ``L0LearnedMetric``: ``d = ‖W Δc‖₂`` with ``M = WᵀW``,
row 0 = luminance, rows 1–2 = chrominance at init (orthonormal). Trained end-to-end
when ``params.L0.LEARNED_METRIC`` is enabled.

With ``L0Notch`` (``params.L0.NOTCH_ENABLED``): project ``u = Wc``, separable
JPEG notch on all three projected channels, then squared directional differences,
NR without per-direction min subtraction, and ``γ`` applied to ``h_{2m}`` only.

Non-RGB ``compute_contrast_field`` (divisive NR) is unchanged for grayscale / generic C.
"""

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


# ═══════════════════════════════════════════════════════════════
# Learned RGB difference metric (M = WᵀW, row-0 lum / rows 1–2 chr)
# ═══════════════════════════════════════════════════════════════

def orthonormal_lum_chroma_basis() -> torch.Tensor:
    """3×3 orthonormal rows: luminance, then two chrominance directions."""
    s3 = 3.0 ** -0.5
    s2 = 2.0 ** -0.5
    s6 = 6.0 ** -0.5
    return torch.tensor(
        [
            [s3, s3, s3],
            [s2, -s2, 0.0],
            [s6, s6, -2.0 * s6],
        ],
        dtype=torch.float32,
    )


class L0LearnedMetric(nn.Module):
    r"""Learned inner product on linear RGB differences.

    Directional distance uses ``d = ‖W Δc‖₂`` with ``M = WᵀW ≽ 0``.
    Row 0 spans luminance (|ΔL| at init); rows 1–2 span chrominance
    (‖ΔC‖₂ at init).  η_lum / η_chr Naka–Rushton channels are unchanged.
    """

    def __init__(self, *, learnable: bool = True) -> None:
        super().__init__()
        init = orthonormal_lum_chroma_basis()
        if learnable:
            self.W = nn.Parameter(init.clone())
        else:
            self.register_buffer("W", init)

    def project_diff(self, delta_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``(..., 3)`` RGB difference to ``(|lum|, ‖chr‖₂)``."""
        proj = torch.einsum("...i,ji->...j", delta_rgb, self.W)
        d_lum = proj[..., 0].abs()
        d_chr = (proj[..., 1:].square().sum(dim=-1) + _L0_EPS).sqrt()
        return d_lum, d_chr

    def project_image(self, img_rgb: torch.Tensor) -> torch.Tensor:
        """Map ``(H, W, 3)`` RGB to ``(H, W, 3)`` projected channels."""
        return torch.einsum("hwi,ji->hwj", img_rgb, self.W)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def _softplus_inv(y: float) -> float:
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))


class L0Notch(nn.Module):
    r"""Learnable JPEG notch — separable conv on projected channels before differences.

    Frequency response (cycles/pixel, real even):
      N(ω) = 1 − d · exp(−(|ω| − ω_n)² / (2σ_n²))

    Spatial kernel on matched lattice ω_m = m/L via cosine synthesis; applied
    separably in 2D before Naka–Rushton.
    """

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
        """Length ``L = 2H + 1`` spatial taps ``h_n[n]``."""
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
        """Separable notch on each channel of ``(H, W, C)``."""
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


# ═══════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════

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
    """``(h_n *_x (h_n *_y ch))`` with replicate padding at image borders."""
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


# ═══════════════════════════════════════════════════════════════
# Directional differences (legacy joint RGB)
# ═══════════════════════════════════════════════════════════════

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
    """Scalar lum and chroma distance per direction.

    Default: |ΔL| and ‖ΔC‖₂ on the orthonormal split.
    With ``metric``: |W₀·ΔRGB| and ‖W₁:₂·ΔRGB‖₂ (learned M = WᵀW).
    """
    H, W, _ = img_rgb.shape
    N = len(offsets)
    use_metric = metric is not None
    if not use_metric:
        R = img_rgb[..., 0]
        G = img_rgb[..., 1]
        B = img_rgb[..., 2]
        L = (R + G + B) * (1.0 / 3.0)
        C = torch.stack([R - L, G - L, B - L], dim=-1)

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
            dL = L[r0s:r1s, c0s:c1s] - L[r0d:r1d, c0d:c1d]
            dC = C[r0s:r1s, c0s:c1s] - C[r0d:r1d, c0d:c1d]
            d_lum[r0d:r1d, c0d:c1d, k] = dL.abs()
            d_chr[r0d:r1d, c0d:c1d, k] = (dC.square().sum(dim=-1) + _L0_EPS).sqrt()
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
    W = orthonormal_lum_chroma_basis().to(device=img_rgb.device, dtype=img_rgb.dtype)
    return torch.einsum("hwi,ji->hwj", img_rgb, W)


def _compute_m_sq_lum_chroma(
    u_tilde: torch.Tensor,
    offsets: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared directional magnitudes on notched projected channels."""
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
    """m² / (η² + m²) — no per-direction min subtraction, no γ."""
    eta_sq = float(eta) * float(eta)
    return m_sq / (eta_sq + m_sq)


def _naka_per_direction(
    d: torch.Tensor,
    eta: float,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """γ d̃² / (η² + d̃²) with d̃_k = d_k − min_j d_j (per pixel)."""
    d_t = d - torch.amin(d, dim=-1, keepdim=True)
    eta_sq = float(eta) * float(eta)
    d_sq = d_t * d_t
    g = torch.tensor(float(gamma), dtype=torch.float32, device=device)
    return g * d_sq / (eta_sq + d_sq)


# ═══════════════════════════════════════════════════════════════
# Contrast field — divisive normalization (non-RGB / legacy)
# ═══════════════════════════════════════════════════════════════

def compute_contrast_field(
    img: torch.Tensor,
    eta0: EtaArg,
    gamma: float,
    offsets: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """d_k ← d_k − min_k d_k;  h_k = γ · d_k² / (η₀² + Σ_k d_k²)."""
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
    """Returns (h, vld, η_lum, γ). Pass ``eta_lum``/``eta_chr`` or ``eta0`` for both."""
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
    """Project → notch → differences → NR → harmonics (γ on h₂m only)."""
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
    """Split-channel L0 on (H,W,3) RGB.

    Returns
      h, vld, eta_lum_used, eta_chr_used, gamma_used,
      s, h1m, h2m, h2m_lum, h2m_chr
    """
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


# ═══════════════════════════════════════════════════════════════
# Harmonics + seed (combined field)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Image loading (unchanged)
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

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
            print(
                f"  η_lum={float(eta_l):.4f}  η_chr={float(eta_c):.4f}  "
                f"γ={gamma_used:.3f}  (split RGB)"
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
