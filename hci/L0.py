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

Non-RGB ``compute_contrast_field`` (divisive NR) is unchanged for grayscale / generic C.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

import numpy as np
import torch
from hci.renderer import _interp_cell_to_pixel

EtaArg = Union[float, Callable[[], float]]


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar |ΔL| and vector chroma L2 ‖ΔC‖₂ (no /√3 normalization)."""
    H, W, _ = img_rgb.shape
    N = len(offsets)
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
        dL = L[r0s:r1s, c0s:c1s] - L[r0d:r1d, c0d:c1d]
        dC = C[r0s:r1s, c0s:c1s] - C[r0d:r1d, c0d:c1d]
        d_lum[r0d:r1d, c0d:c1d, k] = dL.abs()
        d_chr[r0d:r1d, c0d:c1d, k] = (dC * dC).sum(dim=-1).sqrt()
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


def _naka_per_direction(
    d: torch.Tensor,
    eta: float | torch.Tensor,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """γ d̃² / (η² + d̃²) with d̃_k = d_k − min_j d_j (per pixel).

    eta can be a scalar or an (H, W) tensor for spatially-varying
    semi-saturation (pass-2 η modulation).
    """
    d_t = d - torch.amin(d, dim=-1, keepdim=True)
    d_sq = d_t * d_t
    g = torch.tensor(float(gamma), dtype=torch.float32, device=device)
    if isinstance(eta, torch.Tensor):
        # eta is (H, W) → expand to (H, W, 1) for broadcasting with (H, W, N)
        eta_sq = (eta * eta).unsqueeze(-1)
    else:
        eta_sq = float(eta) * float(eta)
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


def compute_l0_rgb(
    img_rgb: torch.Tensor,
    *,
    eta_lum: EtaArg | torch.Tensor,
    eta_chr: EtaArg | torch.Tensor,
    gamma: float,
    offsets: list[tuple[int, int]],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    float | torch.Tensor,
    float | torch.Tensor,
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Split-channel L0 on (H,W,3) RGB.

    eta_lum / eta_chr can be scalars (pass 1) or (H, W) tensors (pass 2).

    Returns
      h, vld, eta_lum_used, eta_chr_used, gamma_used,
      s, h1m, h2m, h2m_lum, h2m_chr
    """
    if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {tuple(img_rgb.shape)}")
    img_rgb = img_rgb.float()
    H, W, _ = img_rgb.shape
    device = img_rgb.device
    gamma_used = float(gamma)

    # Resolve eta — scalar or per-pixel tensor
    if isinstance(eta_lum, torch.Tensor):
        eta_l = eta_lum.to(device=device, dtype=torch.float32)
    else:
        eta_l = resolve_eta(eta_lum)
    if isinstance(eta_chr, torch.Tensor):
        eta_c = eta_chr.to(device=device, dtype=torch.float32)
    else:
        eta_c = resolve_eta(eta_chr)

    vld = compute_valid(H, W, offsets, device)
    d_lum, d_chr = _compute_d_lum_chroma(img_rgb, offsets)

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
    h1m = (s[..., 0] ** 2 + s[..., 1] ** 2).sqrt()
    h2m = (s[..., 2] ** 2 + s[..., 3] ** 2).sqrt()
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
            ir, eta_lum=eta_lum, eta_chr=eta_chr, gamma=gamma, offsets=offsets,
        )
        eta_used_print = float(eta_l) if not isinstance(eta_l, torch.Tensor) else f"map[{eta_l.min():.4f},{eta_l.max():.4f}]"
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
            el_str = f"{eta_used_print}" if isinstance(eta_used_print, str) else f"{eta_used_print:.4f}"
            ec_str = (f"map[{eta_c.min():.4f},{eta_c.max():.4f}]"
                      if isinstance(eta_c, torch.Tensor) else f"{float(eta_c):.4f}")
            print(f"  η_lum={el_str}  η_chr={ec_str}  γ={gamma_used:.3f}  (split RGB)")
        else:
            print(f"  η={eta_used_print}  γ={gamma_used:.3f}  (grayscale)")
        print(f"  excluded {border_mask.sum().item()} border pixels")
        print(f"  h2m: mean={h2m.mean().item():.4f} max={h2m.max().item():.4f}")
    eta_l_out = float(eta_l) if not isinstance(eta_l, torch.Tensor) else eta_l
    eta_c_out = float(eta_c) if not isinstance(eta_c, torch.Tensor) else eta_c
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
        "eta_lum_used": eta_l_out,
        "eta_chr_used": eta_c_out,
        "eta_used": eta_used_print,
        "gamma_used": gamma_used,
    }


# ═══════════════════════════════════════════════════════════════
# η modulation from collinear coherence (pass-2 feedback)
# ═══════════════════════════════════════════════════════════════

def compute_eta_modulation(
    kappa_col_grid: torch.Tensor,
    e_col_grid: torch.Tensor,
    is_border_grid: torch.Tensor,
    nH: int, nW: int,
    H: int, W: int,
    S: int, P: int,
    eta0_lum: float,
    eta0_chr: float,
    a: torch.Tensor | float,
    b: torch.Tensor | float,
    c: torch.Tensor | float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute spatially-varying η maps for L0 pass 2.

    Learned modulation via sigmoid:

    .. math::
        \eta(p) = \eta_0 \cdot \sigma\!\bigl(a - b\cdot\bar\kappa(p)
                  + c\cdot\bar E_{\text{col}}(p)\bigr)

    Scalars (a, b, c) control the modulation; at init (a=2, b=c=0),
    :math:`\sigma(2)\approx 0.88` → near-identity.

    Args:
        kappa_col_grid: (nH, nW) normalized collinear coherence
        e_col_grid:     (nH, nW) raw (unnormalized) collinear energy
        is_border_grid: (nH, nW) bool
        nH, nW: cell grid dimensions
        H, W: pixel dimensions
        S, P: cell grid stride and patch size
        eta0_lum, eta0_chr: base η values from pass 1
        a, b, c: learned modulation parameters (scalars or 0-d tensors)

    Returns:
        eta_lum_map: (H, W) per-pixel η for luminance
        eta_chr_map: (H, W) per-pixel η for chrominance
    """
    device = kappa_col_grid.device
    dtype = kappa_col_grid.dtype

    # Mask borders
    kappa = torch.where(is_border_grid, torch.zeros_like(kappa_col_grid), kappa_col_grid)
    e_col = torch.where(is_border_grid, torch.zeros_like(e_col_grid), e_col_grid)

    # Normalize E_col to [0, 1] for stable sigmoid input
    e_max = e_col.max().clamp_min(eps)
    e_col_norm = e_col / e_max

    # Resolve scalars to tensors on the right device
    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(float(x), device=device, dtype=dtype)

    a_t, b_t, c_t = _t(a), _t(b), _t(c)

    # σ(a - b·κ + c·E_col_norm) on cell grid
    logit = a_t - b_t * kappa + c_t * e_col_norm
    mod_grid = torch.sigmoid(logit)

    # Interpolate to pixel resolution
    mod_pix = _interp_cell_to_pixel(mod_grid, nH, nW, H, W, S, P)

    eta_lum_map = eta0_lum * mod_pix
    eta_chr_map = eta0_chr * mod_pix

    return eta_lum_map.to(dtype=dtype, device=device), eta_chr_map.to(dtype=dtype, device=device)


def run_l0_two_pass(
    ir: torch.Tensor,
    eta_lum: float,
    eta_chr: float,
    gamma: float,
    offsets: list[tuple[int, int]],
    kappa_col_grid: torch.Tensor,
    e_col_grid: torch.Tensor,
    is_border_grid: torch.Tensor,
    nH: int, nW: int,
    S: int, P: int,
    a: torch.Tensor | float = 2.0,
    b: torch.Tensor | float = 0.0,
    c: torch.Tensor | float = 0.0,
    verbose: bool = True,
) -> dict:
    """Run L0 pass 2 with η modulated by collinear coherence from pass 1.

    η(p) = η₀ · σ(a - b·κ̄(p) + c·Ē_col(p))

    Pass 1 is assumed to have already been run (producing the collinear
    signals).  This function computes the η modulation map and re-runs
    L0 with spatially-varying η.

    Args:
        ir: (H, W, 3) RGB image
        eta_lum, eta_chr: base η values from pass 1
        gamma: L0 gamma
        offsets: L0 offsets
        kappa_col_grid: (nH, nW) from pass-1 collinear recurrence
        e_col_grid: (nH, nW) raw collinear energy from pass 1
        is_border_grid: (nH, nW) bool
        nH, nW, S, P: cell grid geometry
        a, b, c: learned modulation params (scalar or 0-d tensor)
        verbose: print diagnostics

    Returns:
        dict with same keys as run_l0, plus 'eta_lum_map' and 'eta_chr_map'
    """
    H, W = ir.shape[:2]

    # Compute per-pixel η maps
    eta_lum_map, eta_chr_map = compute_eta_modulation(
        kappa_col_grid, e_col_grid, is_border_grid,
        nH, nW, H, W, S, P,
        eta0_lum=eta_lum, eta0_chr=eta_chr,
        a=a, b=b, c=c,
    )

    if verbose:
        print("L0 pass 2 (η-modulated)...")
        print(f"  η_lum: [{eta_lum_map.min():.4f}, {eta_lum_map.max():.4f}]  "
              f"(base {eta_lum:.4f})")
        print(f"  η_chr: [{eta_chr_map.min():.4f}, {eta_chr_map.max():.4f}]  "
              f"(base {eta_chr:.4f})")
        a_v = a.item() if isinstance(a, torch.Tensor) else a
        b_v = b.item() if isinstance(b, torch.Tensor) else b
        c_v = c.item() if isinstance(c, torch.Tensor) else c
        print(f"  σ params: a={a_v:.3f}  b={b_v:.3f}  c={c_v:.3f}")

    # Re-run L0 with spatially-varying η
    result = run_l0(
        ir,
        eta_lum=eta_lum_map,
        eta_chr=eta_chr_map,
        gamma=gamma,
        offsets=offsets,
        verbose=verbose,
    )
    result["eta_lum_map"] = eta_lum_map
    result["eta_chr_map"] = eta_chr_map
    return result
