r"""Renderer — harmonic-inversion deposit + per-bin curvature correction (noisy-OR).

This renderer replaces the 6→16→8-basis dictionary deposit with a per-bin Gaussian
synthesis from the seed's ``rho_out_bins``.  The bin tensor IS the parameterization
of the deposit — angle, position, and multi-curve behavior come straight from L1
and the seed without dictionary quantization.  A small per-bin correction MLP adds
curvature and tangent shift on top of an otherwise straight stroke.

Pipeline (per image, read from ``cells_flat``):

  rho_out_bins  ∈ ℝ^(nH,nW,K)   per-bin seed readout (gradient-bearing)
  ax_bin        ∈ ℝ^(N,K)        per-bin sub-pixel anchor x  (detached)
  ay_bin        ∈ ℝ^(N,K)        per-bin sub-pixel anchor y  (detached)
  theta_bins    ∈ ℝ^K            fixed bin centers \bar θ_k

Per active (cell c, bin k) pair:

  F^(k)_c ∈ ℝ⁴ — [ρ^(k), <ρ^(k) pos_t^(k)>_𝒩, signed_tangent_asym_ρ^(k),
                  Σ_{j≠k} ρ^(j)]   (no coherence in this version)
  (κ^(k), e^(k)) = bounded(MLP_{4→8→2}(F^(k)))
  g^(k)_c        = σ(α_g (ρ^(k) − τ · max_j ρ^(j))) · ok(c)         sparsity gate

  Per pixel p in deposit window centered on integer floor of (a_x^(k), a_y^(k)):
    Δ_x = (p_x − a_x^(k)),  Δ_y = (p_y − a_y^(k))   pixel-unit col/row offsets
    s   =  Δ_y cos θ̄_k + Δ_x sin θ̄_k                tangent projection  (along edge)
    n   = −Δ_y sin θ̄_k + Δ_x cos θ̄_k                normal  projection  (perpendicular)
    s̃   = s − e^(k)
    n_c = n − ½ κ^(k) · s̃²
    f^(k)_c(p) = exp(− n_c² / 2σ⊥² − s̃² / 2σ∥²)
    claim^(k)_c(p) = g^(k)_c · ρ^(k) · f^(k)_c(p)

CONVENTION on θ:
  bar_theta_k comes from L1's ½ arg(z₂), which is the GRADIENT angle (not the
  tangent).  In (dy, dx) = (row, col) order, (cos θ, sin θ) IS the tangent — but
  in (col, row) order it would point along the gradient/normal.  The projection
  formulas above use (dy, dx) order to match seed.py's collinear_facilitation_bins.

Aggregation (single noisy-OR across ALL (c, k) pairs):

  log(1 − B̂(p)) = Σ_{c,k} log(1 − claim^(k)_c(p))
  B̂(p) = 1 − ∏_{c,k} (1 − claim^(k)_c(p))

Why this and not the soft-indicator basis renderer:
  - Multi-curve falls out for free: a cell with two strong bins synthesizes two
    crossing strokes, each at its own angle / anchor.  No mode collapse at junctions.
  - Sub-pixel positioning is exact (analytical Gaussian centered on continuous
    anchor) — no bilinear scatter required.
  - The L1 / seed tensor parameterizes the deposit; the MLP only carries curvature
    and asymmetric reach.  Parameter count drops from ~250 to ~70.

The renderer still applies no thinning, splat coherence, or divisive suppression.
Precision lives at the seed.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import RENDER


# ═══════════════════════════════════════════════════════════════
# Defaults (pulled from params.RENDER with safe fallbacks)
# ═══════════════════════════════════════════════════════════════

# Deposit footprint: half-width in pixels = clamp(ceil(STRIDES · S), [MIN, MAX]).
_DEPOSIT_HALF_WIDTH_STRIDES = float(getattr(RENDER, "DEPOSIT_HALF_WIDTH_STRIDES", 2.0))
_DEPOSIT_HALF_WIDTH_MIN = int(getattr(RENDER, "DEPOSIT_HALF_WIDTH_MIN", 4))
_DEPOSIT_HALF_WIDTH_MAX = int(getattr(RENDER, "DEPOSIT_HALF_WIDTH_MAX", 24))

# Correction MLP: 4 features → CORR_HIDDEN → 2 outputs (raw_κ, raw_e_s).
_FEATURE_DIM = 4
_HIDDEN_DIM = int(getattr(RENDER, "CORR_HIDDEN", 8))
_OUT_DIM = 2

# Synthesis-parameter inits (pixel units).
_SIGMA_PERP_INIT = float(getattr(RENDER, "SIGMA_PERP_INIT", 0.6))
_SIGMA_PAR_INIT  = float(getattr(RENDER, "SIGMA_PAR_INIT", 2.0))
_KAPPA_MAX_INIT  = float(getattr(RENDER, "KAPPA_MAX_INIT", 0.1))
_EXT_MAX_INIT    = float(getattr(RENDER, "EXT_MAX_INIT", 1.0))

# Sparsity gate inits.
_BIN_GATE_TAU_INIT   = float(getattr(RENDER, "BIN_GATE_TAU_INIT", 0.4))
_BIN_GATE_ALPHA_INIT = float(getattr(RENDER, "BIN_GATE_ALPHA_INIT", 10.0))

# Compute-saving thresholds: (c, k) pairs below either floor are skipped entirely.
# Critical for the noisy-OR accumulator — floating-point noise pairs would otherwise
# pollute log1p(-claim) across the whole image.
_GATE_ACTIVE_THRESHOLD = 1e-3
_RHO_ACTIVE_FLOOR = 1e-4

# Numerical clip on claims for log1p(-claim) stability.
_CLAIM_CLIP = 1.0 - 1e-5

# Softfloor on feature denominators to suppress empty-neighborhood divergence.
_FEAT_SOFTFLOOR = 5e-2


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


# ═══════════════════════════════════════════════════════════════
# Per-bin features (4 scalars per (cell, bin); no coherence in this version)
# ═══════════════════════════════════════════════════════════════

def _per_bin_features(
    rho_bins: torch.Tensor,    # (nH, nW, K)   — rho_out from seed (detached at use site)
    bar_theta: torch.Tensor,   # (K,)          — bin centers
    ib_g: torch.Tensor,        # (nH, nW)      — bool, True on L1-border cells
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-bin features F^(k)_c ∈ ℝ⁴.

      f0: ρ^(k)(c)                                                       self-bin energy
      f1: Σ_δ pos_t^(k)(δ) ρ^(k)(c+δ) / (Σ_δ pos_t^(k)(δ) + softfloor)   collinear support
      f2: Σ_δ (δ·t̂_k) ρ^(k)(c+δ) / (Σ_δ |δ·t̂_k| ρ^(k)(c+δ) + softfloor) signed tangent asym
      f3: Σ_{j≠k} ρ^(j)(c)                                                competing-bin energy

    pos_t^(k)(δ) = (δ·t̂_k)² / |δ|² is the same kernel the seed uses for collinear
    facilitation, so f1 is a direct echo of "how much same-bin energy lies along
    my tangent in the cell neighborhood."  f2's sign tells which end has more
    support (drives the tangent shift e).  f3 fires at junctions.

    Returns: (nH, nW, K, 4).
    """
    _ = eps
    nH, nW, K = rho_bins.shape
    dtype = rho_bins.dtype

    cos_b = torch.cos(bar_theta).view(1, 1, K)
    sin_b = torch.sin(bar_theta).view(1, 1, K)

    def _pad0_hwk(t: torch.Tensor) -> torch.Tensor:
        # t: (nH, nW, K) → pad spatial dims by 1 with zeros, keep K trailing.
        x = t.permute(2, 0, 1).unsqueeze(0)              # (1, K, nH, nW)
        x = F.pad(x, (1, 1, 1, 1), value=0.0)            # (1, K, nH+2, nW+2)
        return x.squeeze(0).permute(1, 2, 0)             # (nH+2, nW+2, K)

    rho_p = _pad0_hwk(rho_bins)

    sum_pos_rho     = torch.zeros_like(rho_bins)         # f1 numerator
    sum_pos         = torch.zeros_like(rho_bins)         # f1 denominator
    sum_t_rho       = torch.zeros_like(rho_bins)         # f2 numerator (signed)
    sum_abs_t_rho   = torch.zeros_like(rho_bins)         # f2 denominator (|.|-weighted)

    # f3 is purely local — no neighborhood needed.
    sum_all_bins = rho_bins.sum(dim=-1, keepdim=True)    # (nH, nW, 1)
    f3 = sum_all_bins - rho_bins                          # (nH, nW, K)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sl = (slice(1 + dy, 1 + dy + nH), slice(1 + dx, 1 + dx + nW))
            rn = rho_p[sl]                                # (nH, nW, K)
            # Tangent projection in (dy, dx) = (row, col) order — matches seed
            # convention: (cos θ, sin θ) is the tangent in (row, col), so
            # t_proj = dy·cos + dx·sin.  See the convention comment in
            # _harmonic_deposit for why this is NOT dx·cos + dy·sin.
            t_proj = float(dy) * cos_b + float(dx) * sin_b   # (1, 1, K)
            d2 = float(dy * dy + dx * dx)
            pos_t = (t_proj * t_proj) / d2                   # (1, 1, K)

            sum_pos_rho   = sum_pos_rho   + rn * pos_t
            sum_pos       = sum_pos       + pos_t.expand_as(rn)
            sum_t_rho     = sum_t_rho     + rn * t_proj
            sum_abs_t_rho = sum_abs_t_rho + rn * t_proj.abs()

    f1 = sum_pos_rho / (sum_pos + _FEAT_SOFTFLOOR)
    f2 = sum_t_rho   / (sum_abs_t_rho + _FEAT_SOFTFLOOR)

    use = (~ib_g).to(dtype=dtype).unsqueeze(-1)            # (nH, nW, 1)
    f0 = rho_bins * use
    f1 = f1 * use
    f2 = f2 * use
    f3 = f3 * use

    return torch.stack([f0, f1, f2, f3], dim=-1)           # (nH, nW, K, 4)


# ═══════════════════════════════════════════════════════════════
# Synthesis: per-(cell, bin) Gaussian stroke deposit + noisy-OR
# ═══════════════════════════════════════════════════════════════

def _deposit_half_width(S: int) -> int:
    h = int(math.ceil(_DEPOSIT_HALF_WIDTH_STRIDES * max(S, 1)))
    return max(_DEPOSIT_HALF_WIDTH_MIN, min(_DEPOSIT_HALF_WIDTH_MAX, h))


def _harmonic_deposit(
    rho_active: torch.Tensor,        # (A,) ρ^(k)_c                    grad-bearing
    gate_active: torch.Tensor,       # (A,) g^(k)_c   ∈ [0, 1]         grad-bearing
    ax_active: torch.Tensor,         # (A,) anchor x in pixels         detached
    ay_active: torch.Tensor,         # (A,) anchor y in pixels         detached
    cos_a: torch.Tensor,             # (A,) cos θ̄_k                    detached
    sin_a: torch.Tensor,             # (A,) sin θ̄_k                    detached
    kappa_active: torch.Tensor,      # (A,) κ^(k)_c  (bounded ±κ_max)  grad-bearing
    ext_s_active: torch.Tensor,      # (A,) e^(k)_c  (bounded ±e_max)  grad-bearing
    sigma_perp: torch.Tensor,        # ()  σ⊥                          grad-bearing
    sigma_par: torch.Tensor,         # ()  σ∥                          grad-bearing
    H: int, W: int, half_w: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(cell, bin) Gaussian-stroke deposit; aggregate via global noisy-OR.

    The Gaussian is sampled analytically at integer pixel positions inside a
    (2·half_w+1)² window around the integer floor of the sub-pixel anchor.  Sub-
    pixel anchor accuracy is preserved by using the floating-point distance to
    the continuous anchor (not the integer base) inside the Gaussian.  No
    bilinear scatter needed.

    Returns:
        bmap:       (H, W) ∈ [0, 1]
        theta_star: (H, W) — bar_theta_k of the dominant claimant per pixel
                    (for inference-time ridge NMS).
    """
    A = rho_active.shape[0]
    device, dtype = rho_active.device, rho_active.dtype
    n_pix = H * W

    if A == 0:
        z = torch.zeros(H, W, device=device, dtype=dtype)
        return z, z

    offsets = torch.arange(-half_w, half_w + 1, device=device, dtype=dtype)
    oy_g, ox_g = torch.meshgrid(offsets, offsets, indexing="ij")
    oy = oy_g.reshape(-1)                                  # (P,)
    ox = ox_g.reshape(-1)
    P = oy.shape[0]

    log_neg_acc = torch.zeros(n_pix, device=device, dtype=dtype)
    max_claim = torch.full((n_pix,), -1.0, device=device, dtype=torch.float32)
    theta_star = torch.zeros(n_pix, device=device, dtype=dtype)

    ax_int = torch.floor(ax_active).long()
    ay_int = torch.floor(ay_active).long()
    ax_frac = ax_active - ax_int.to(dtype=dtype)
    ay_frac = ay_active - ay_int.to(dtype=dtype)

    bar_theta_active = torch.atan2(sin_a, cos_a)           # (A,) — for θ★ tracking

    inv_two_sig_perp_sq = 0.5 / (sigma_perp * sigma_perp).clamp_min(eps)
    inv_two_sig_par_sq  = 0.5 / (sigma_par  * sigma_par ).clamp_min(eps)

    # Batch over active pairs to bound peak memory.
    max_batch = max(1, 4_000_000 // P)

    for b0 in range(0, A, max_batch):
        b1 = min(b0 + max_batch, A)
        bs = b1 - b0

        ax_f = ax_frac[b0:b1].unsqueeze(1)                 # (bs, 1)
        ay_f = ay_frac[b0:b1].unsqueeze(1)
        ca = cos_a[b0:b1].unsqueeze(1)
        sa = sin_a[b0:b1].unsqueeze(1)

        # Δ = (pixel) − (sub-pixel anchor)
        #   pixel_x = ax_int + ox; pixel_y = ay_int + oy
        #   Δ_x = (ax_int + ox) − (ax_int + ax_frac) = ox − ax_frac
        dx = ox.unsqueeze(0) - ax_f                        # (bs, P)  col offset
        dy = oy.unsqueeze(0) - ay_f                        #          row offset

        # Rotate into bin frame.  CONVENTION (must match seed.py):
        #   bar_theta_k is the GRADIENT angle (from ½ arg z₂), and (cos θ, sin θ)
        #   is the tangent in (dy, dx) = (row, col) order — not (col, row).
        #   So s (along tangent) = dy·cos + dx·sin
        #      n (along normal ) = −dy·sin + dx·cos.
        # The old soft-indicator renderer had the dy/dx swap reversed; with the
        # anisotropic Gaussian (σ∥ ≫ σ⊥) here the resulting 90° rotation becomes
        # visible as "perpendicular" strokes — hence this fix.
        s =  dy * ca + dx * sa
        n = -dy * sa + dx * ca

        s_tilde = s - ext_s_active[b0:b1].unsqueeze(1)
        n_curv  = n - 0.5 * kappa_active[b0:b1].unsqueeze(1) * (s_tilde * s_tilde)

        f_val = torch.exp(
            -(n_curv * n_curv) * inv_two_sig_perp_sq
            -(s_tilde * s_tilde) * inv_two_sig_par_sq
        )                                                   # (bs, P)

        amp = (gate_active[b0:b1] * rho_active[b0:b1]).unsqueeze(1)  # (bs, 1)
        claim = (amp * f_val).clamp(min=0.0, max=_CLAIM_CLIP)        # (bs, P)

        px = ax_int[b0:b1].unsqueeze(1) + ox.unsqueeze(0).long()     # (bs, P)
        py = ay_int[b0:b1].unsqueeze(1) + oy.unsqueeze(0).long()
        in_bounds = (py >= 0) & (py < H) & (px >= 0) & (px < W)

        claim_safe = torch.where(in_bounds, claim, torch.zeros_like(claim))
        log_neg = torch.log1p(-claim_safe)                  # ≤ 0; 0 where claim = 0

        flat_idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1))
        log_neg_acc.scatter_add_(0, flat_idx.reshape(-1), log_neg.reshape(-1))

        # θ★ tracking (no gradient).
        cl_det = claim_safe.detach().to(torch.float32).reshape(-1)
        idx_flat = flat_idx.reshape(-1)
        th_expand = bar_theta_active[b0:b1].unsqueeze(1).expand(bs, P).reshape(-1)
        cur = max_claim[idx_flat]
        upd = cl_det > cur
        if upd.any():
            upd_idx = idx_flat[upd]
            max_claim[upd_idx] = cl_det[upd]
            theta_star[upd_idx] = th_expand[upd].to(dtype=dtype)

    bmap = -torch.expm1(log_neg_acc)                        # ∈ [0, 1]
    return bmap.reshape(H, W), theta_star.reshape(H, W)


# ═══════════════════════════════════════════════════════════════
# Correction MLP (4 → 8 → 2, shared across bins)
# ═══════════════════════════════════════════════════════════════

class CorrectionMLP(nn.Module):
    """Per-(cell, bin) correction MLP: F^(k)_c → (raw_κ, raw_e_s).

    Output is bounded by tanh in the renderer's forward path; this module just
    emits unbounded scalars.  Initialized so that κ ≈ e ≈ 0 at t = 0 → straight
    strokes centered on the per-bin anchor by default.  Curvature and tangent
    shift only switch on when the features carry junction / asymmetry signal.
    """

    def __init__(
        self,
        in_dim: int = _FEATURE_DIM,
        hidden_dim: int = _HIDDEN_DIM,
        out_dim: int = _OUT_DIM,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
            self.fc1.bias.zero_()
            nn.init.normal_(self.fc2.weight, std=0.01)
            self.fc2.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, 4) → (N, 2)."""
        h = F.relu(self.fc1(features))
        return self.fc2(h)


# ═══════════════════════════════════════════════════════════════
# ModulationRenderer
# ═══════════════════════════════════════════════════════════════

class ModulationRenderer(nn.Module):
    """Harmonic-inversion deposit + per-bin curvature correction.

    Learned parameters (softplus-positive on raw, except τ via sigmoid):
        σ⊥, σ∥          Gaussian stroke widths (perp / along tangent)
        κ_max, e_max    bounds on per-(c,k) curvature & tangent shift (used via tanh)
        τ, α_g          sparsity gate threshold and sharpness
        CorrectionMLP   4 → 8 → 2 (66 params with defaults)

    Total: ~72 params (66 MLP + 6 scalars).
    """

    def __init__(self, **kwargs):
        super().__init__()
        _ = kwargs   # tolerate legacy kwargs from older callers

        self.correction = CorrectionMLP()

        self._sigma_perp_raw = nn.Parameter(torch.tensor(_inv_softplus(_SIGMA_PERP_INIT)))
        self._sigma_par_raw  = nn.Parameter(torch.tensor(_inv_softplus(_SIGMA_PAR_INIT)))
        self._kappa_max_raw  = nn.Parameter(torch.tensor(_inv_softplus(_KAPPA_MAX_INIT)))
        self._ext_max_raw    = nn.Parameter(torch.tensor(_inv_softplus(_EXT_MAX_INIT)))
        self._alpha_g_raw    = nn.Parameter(torch.tensor(_inv_softplus(_BIN_GATE_ALPHA_INIT)))
        # τ ∈ [0, 1] via sigmoid; init at _BIN_GATE_TAU_INIT.
        tau0 = max(min(_BIN_GATE_TAU_INIT, 1.0 - 1e-4), 1e-4)
        self._tau_raw = nn.Parameter(torch.tensor(math.log(tau0 / (1.0 - tau0))))

    @property
    def sigma_perp(self) -> torch.Tensor:
        return F.softplus(self._sigma_perp_raw).view(()).clamp_min(0.2)

    @property
    def sigma_par(self) -> torch.Tensor:
        return F.softplus(self._sigma_par_raw).view(()).clamp_min(0.2)

    @property
    def kappa_max(self) -> torch.Tensor:
        return F.softplus(self._kappa_max_raw).view(())

    @property
    def ext_max(self) -> torch.Tensor:
        return F.softplus(self._ext_max_raw).view(())

    @property
    def alpha_g(self) -> torch.Tensor:
        return F.softplus(self._alpha_g_raw).view(())

    @property
    def tau(self) -> torch.Tensor:
        return torch.sigmoid(self._tau_raw).view(())

    # ── Legacy compat properties (so older diag / print code does not crash) ───
    @property
    def ext_scale(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def sigma_pre(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def smooth_sigma(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def eta_h(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def s_t(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)

    @property
    def s_n(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32)


def upgrade_renderer_state_dict(state_dict: dict, prefix: str = "") -> dict:
    """Strip legacy renderer keys before load_state_dict(strict=False)."""
    base_legacy = {
        f"{prefix}_sigma_pre_raw",
        f"{prefix}_eta_h_raw",
        f"{prefix}_smooth_sigma_raw",
        f"{prefix}_ext_scale_raw",
        f"{prefix}s_t",
        f"{prefix}s_n",
        f"{prefix}perp_conv.conv.weight",
        f"{prefix}perp_conv.conv.bias",
        f"{prefix}perp_conv.fc.weight",
        f"{prefix}perp_conv.fc.bias",
        f"{prefix}thinning.fc1.weight",
        f"{prefix}thinning.fc1.bias",
        f"{prefix}thinning.fc2.weight",
        f"{prefix}thinning.fc2.bias",
        f"{prefix}deposit.fc1.weight",
        f"{prefix}deposit.fc1.bias",
        f"{prefix}deposit.fc2.weight",
        f"{prefix}deposit.fc2.bias",
    }
    # _sigma_perp_raw / _sigma_par_raw exist in BOTH the old soft-indicator renderer
    # (as legacy unused scalars) AND the new harmonic renderer (as the stroke widths).
    # Only strip them when loading an old checkpoint that ALSO has soft-indicator-only
    # keys (e.g. _ext_scale_raw or deposit.fc1.weight) — otherwise keep them.
    old_soft_indicator = any(
        f"{prefix}{k}" in state_dict
        for k in ("_ext_scale_raw", "deposit.fc1.weight", "deposit.fc2.weight")
    )
    legacy = set(base_legacy)
    if old_soft_indicator:
        legacy.update({f"{prefix}_sigma_perp_raw", f"{prefix}_sigma_par_raw"})

    return {k: v for k, v in state_dict.items() if k not in legacy}


# ═══════════════════════════════════════════════════════════════
# Proj helpers
# ═══════════════════════════════════════════════════════════════

def proj_to_device(proj: dict, device: torch.device) -> dict:
    return {
        "H": proj["H"], "W": proj["W"],
        "n_cells": proj["n_cells"],
        "nH": proj["nH"], "nW": proj["nW"],
    }


def compute_render_features(
    z2_image: np.ndarray, img: np.ndarray,
    cells: dict, border_mask: np.ndarray,
    eps: float = 1e-9, **kwargs,
) -> dict:
    _ = (z2_image, img, cells, border_mask, eps, kwargs)
    H, W = z2_image.shape
    nH, nW = cells["nH"], cells["nW"]
    return {"H": H, "W": W, "n_cells": nH * nW, "nH": nH, "nW": nW}


# ═══════════════════════════════════════════════════════════════
# render_boundary_map_torch — entry point (signature kept stable)
# ═══════════════════════════════════════════════════════════════

def render_boundary_map_torch(
    rho_cell: torch.Tensor,
    proj_dev: dict,
    renderer: ModulationRenderer,
    cells_flat: dict,
    Hp: int, Wp: int,
    l0_pix: Optional[dict[str, torch.Tensor]] = None,
    eps: float = 1e-6,
    training: bool = False,
    branch_pick: Optional[torch.Tensor] = None,
    content_h: Optional[int] = None,
    content_w: Optional[int] = None,
    return_dominant_theta: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """B̂(p) = 1 − ∏_{c,k} (1 − g^(k)_c · ρ^(k) · f^(k)_c(p)).

    The scalar ``rho_cell`` argument is kept for signature compatibility with
    the prior renderer but is unused — gradient flows through
    ``cells_flat['rho_out_bins']`` written by the seed.  ``branch_pick`` is
    likewise unused; orientation is selected per pixel via the harmonic max.
    """
    _ = (rho_cell, training, l0_pix, branch_pick)
    H, W = proj_dev["H"], proj_dev["W"]
    device = next(renderer.parameters()).device
    dtype = renderer.sigma_perp.dtype

    if "rho_out_bins" not in cells_flat:
        raise ValueError(
            "Harmonic renderer requires cells_flat['rho_out_bins'] from the seed. "
            "Run seed.forward before render_boundary_map_torch."
        )
    if "ax_bin" not in cells_flat or "ay_bin" not in cells_flat:
        raise ValueError("cells_flat must include ax_bin and ay_bin (from L1).")
    if "theta_bins" not in cells_flat:
        raise ValueError("cells_flat must include theta_bins (from L1).")

    nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
    N = nH * nW
    S = int(cells_flat.get("S", max(1, W // max(nW, 1))))

    rho_out_bins = cells_flat["rho_out_bins"].to(device=device, dtype=dtype)
    if rho_out_bins.dim() == 2:
        rho_out_bins = rho_out_bins.reshape(nH, nW, -1)
    K = rho_out_bins.shape[-1]

    ax_bin = cells_flat["ax_bin"].to(device=device, dtype=dtype)
    ay_bin = cells_flat["ay_bin"].to(device=device, dtype=dtype)
    if ax_bin.dim() == 3:
        ax_bin = ax_bin.reshape(N, K)
        ay_bin = ay_bin.reshape(N, K)

    bar_theta = cells_flat["theta_bins"].to(device=device, dtype=dtype).reshape(K)
    is_border = cells_flat["is_border"].to(device=device).reshape(N).bool()
    ib_g = is_border.reshape(nH, nW)

    # ── Step 1: per-bin features (MLP input; ρ detached for stability) ────
    # Detaching rho_out_bins for the features prevents the MLP path from injecting
    # huge Jacobian terms into the seed / L0 via the (sum_pos + soft) ratios when
    # neighborhoods are nearly empty — same rationale as the prior renderer's
    # ρ-and-θ detach pattern for its 6-feature MLP.
    feats = _per_bin_features(rho_out_bins.detach(), bar_theta, ib_g, eps=eps)
    feats_flat = feats.reshape(N * K, _FEATURE_DIM)

    # ── Step 2: correction MLP → bounded (κ, e_s) per (c, k) ──────────────
    raw = renderer.correction(feats_flat)                  # (N·K, 2)
    kappa_flat = renderer.kappa_max * torch.tanh(raw[:, 0])
    ext_s_flat = renderer.ext_max  * torch.tanh(raw[:, 1])

    # ── Step 3: sparsity gate (relative to per-cell max bin) ──────────────
    rho_flat_bins = rho_out_bins.reshape(N, K)             # grad-bearing
    rho_max_cell = rho_flat_bins.max(dim=-1, keepdim=True).values
    ok = (~is_border).to(dtype=dtype).unsqueeze(-1)
    gate_flat = (
        torch.sigmoid(renderer.alpha_g * (rho_flat_bins - renderer.tau * rho_max_cell))
        * ok
    )                                                      # (N, K)

    # ── Step 4: select active (c, k) pairs ────────────────────────────────
    gate_NK = gate_flat.reshape(-1)
    rho_NK = rho_flat_bins.reshape(-1)
    keep = (gate_NK > _GATE_ACTIVE_THRESHOLD) & (rho_NK > _RHO_ACTIVE_FLOOR)

    if not bool(keep.any().item()):
        # Nothing to deposit.  Return zero map but keep autograd graph alive on every
        # renderer parameter so optimizer.step() sees zero (not None) grads.
        zero = (
            renderer.sigma_perp * 0.0 + renderer.sigma_par * 0.0
            + renderer.kappa_max * 0.0 + renderer.ext_max * 0.0
            + renderer.alpha_g * 0.0 + renderer.tau * 0.0
            + 0.0 * raw.sum()
            + 0.0 * rho_out_bins.sum()
        )
        bmap = torch.zeros(H, W, device=device, dtype=dtype) + zero
        theta_star = torch.zeros(H, W, device=device, dtype=dtype)
        out = bmap[:Hp, :Wp]
        if return_dominant_theta:
            return out, theta_star[:Hp, :Wp]
        return out

    # Index helpers so we can recover (cell, bin) from the flat (N·K,) layout.
    bin_idx_full = torch.arange(K, device=device).unsqueeze(0).expand(N, K).reshape(-1)
    active_bin = bin_idx_full[keep]                        # (A,)

    ax_active = ax_bin.reshape(-1)[keep].detach()
    ay_active = ay_bin.reshape(-1)[keep].detach()
    rho_active   = rho_NK[keep]
    gate_active  = gate_NK[keep]
    kappa_active = kappa_flat[keep]
    ext_s_active = ext_s_flat[keep]

    cos_a = torch.cos(bar_theta).index_select(0, active_bin)
    sin_a = torch.sin(bar_theta).index_select(0, active_bin)

    # ── Step 5: harmonic Gaussian deposit + noisy-OR ──────────────────────
    half_w = _deposit_half_width(S)
    bmap, theta_star = _harmonic_deposit(
        rho_active=rho_active,
        gate_active=gate_active,
        ax_active=ax_active,
        ay_active=ay_active,
        cos_a=cos_a,
        sin_a=sin_a,
        kappa_active=kappa_active,
        ext_s_active=ext_s_active,
        sigma_perp=renderer.sigma_perp,
        sigma_par=renderer.sigma_par,
        H=H, W=W, half_w=half_w, eps=eps,
    )

    # ── Crop to content region (unchanged from prior) ─────────────────────
    ch = Hp if content_h is None else content_h
    cw = Wp if content_w is None else content_w
    ch, cw = min(ch, H), min(cw, W)
    if content_h is not None and content_w is not None:
        crop = torch.ones_like(bmap)
        if ch < H: crop[ch:, :] = 0.0
        if cw < W: crop[:, cw:] = 0.0
        bmap = bmap * crop
        theta_star = theta_star * crop

    out = bmap[:Hp, :Wp]
    if return_dominant_theta:
        return out, theta_star[:Hp, :Wp]
    return out


# ═══════════════════════════════════════════════════════════════
# NumPy wrapper (unchanged interface)
# ═══════════════════════════════════════════════════════════════

def render_boundary_map(
    rho_cell: np.ndarray, proj: dict,
    renderer: ModulationRenderer, cells_flat: dict,
    l0_pix: Optional[dict[str, np.ndarray]] = None,
    device: torch.device = torch.device("cpu"),
    eps: float = 1e-6,
    branch_pick: Optional[np.ndarray] = None,
    content_h: Optional[int] = None, content_w: Optional[int] = None,
) -> np.ndarray:
    proj_dev = proj_to_device(proj, device)
    rho_t = torch.from_numpy(np.asarray(rho_cell, dtype=np.float32)).to(device)
    cf_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in cells_flat.items()}
    bp = None
    if branch_pick is not None:
        bp = torch.from_numpy(np.asarray(branch_pick, dtype=np.int64).ravel()).to(device)
    with torch.no_grad():
        bmap_t = render_boundary_map_torch(
            rho_t, proj_dev, renderer, cf_dev, proj["H"], proj["W"], None,
            eps=eps, training=False, branch_pick=bp,
            content_h=content_h, content_w=content_w,
        )
    return bmap_t.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# NMS (unchanged from prior — bmap + theta_star feed in as before)
# ═══════════════════════════════════════════════════════════════

def _nms_unit_normal_from_theta(theta, eps=1e-8):
    _ = eps
    t = np.asarray(theta, dtype=np.float64)
    return np.cos(t).astype(np.float32), (-np.sin(t)).astype(np.float32)


def _nms_unit_normal_from_gradient(mag, eps=1e-8):
    m = np.asarray(mag, dtype=np.float64)
    gx, gy = ndimage.sobel(m, axis=1), ndimage.sobel(m, axis=0)
    norm = np.sqrt(gx * gx + gy * gy) + eps
    return (gx / norm).astype(np.float32), (gy / norm).astype(np.float32)


def _nms_bilinear_sample(mag, row_off, col_off):
    coords = np.stack([row_off.astype(np.float64), col_off.astype(np.float64)])
    return ndimage.map_coordinates(
        mag.astype(np.float64), coords, order=1, mode="nearest"
    ).astype(np.float32)


def ridge_nms(mag, *, theta=None, grad_norm_floor=1e-7):
    m = np.asarray(mag, dtype=np.float32)
    if m.ndim != 2:
        raise ValueError(f"ridge_nms expects 2D, got {m.shape}")
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m_work = m.copy()
    if theta is not None:
        nx, ny = _nms_unit_normal_from_theta(np.asarray(theta, dtype=np.float32))
        weak = np.zeros((H, W), dtype=bool)
    else:
        gx = ndimage.sobel(m_work.astype(np.float64), axis=1)
        gy = ndimage.sobel(m_work.astype(np.float64), axis=0)
        gnorm = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        nx, ny = _nms_unit_normal_from_gradient(m_work)
        weak = gnorm < grad_norm_floor
    ahead = _nms_bilinear_sample(m_work, yy + ny, xx + nx)
    behind = _nms_bilinear_sample(m_work, yy - ny, xx - nx)
    keep = ((m_work >= ahead) & (m_work >= behind)) | weak
    return np.where(keep, m_work, 0.0).astype(np.float32)


def ridge_nms_binary(mag, threshold, *, theta=None, grad_norm_floor=1e-7):
    return (
        ridge_nms(mag, theta=theta, grad_norm_floor=grad_norm_floor) >= threshold
    ).astype(np.uint8) * 255


def cell_rho_to_2branch(rho_cell, branch):
    out = np.zeros((*rho_cell.shape, 2), dtype=rho_cell.dtype)
    ii, jj = np.indices(rho_cell.shape)
    out[ii, jj, branch.astype(np.int64)] = rho_cell
    return out


# Legacy aliases (for callers still referencing the old class names)
HarmonicThinRenderer = ModulationRenderer
StampRenderer = ModulationRenderer
AnisoDiffusionRenderer = ModulationRenderer