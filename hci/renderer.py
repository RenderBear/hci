r"""Renderer — FBP-style filtered back-projection with learned 1D kernels.

The 82-param Gaussian-stroke renderer is generalized along three axes:

  1. Learned 1D reconstruction kernels (h_⊥, h_∥) replace the fixed Gaussian.
     h_⊥ is the FBP-analogue *ramp filter* (perpendicular to tangent): even
     symmetric, free-sign side-lobes around a positive peak.  h_∥ is the
     longitudinal profile along the tangent: even, monotone-decay from a
     positive peak.  Both are sampled at continuous query positions via linear
     interpolation, so geometry corrections (κ, e, δ_n) and kernel taps both
     receive gradient.
  2. 2D anchor correction.  In addition to tangent shift e (slides stroke
     vertex along the edge), the MLP now also emits δ_n: a normal-direction
     anchor shift.  This addresses anchor jitter perpendicular to the edge as
     a learned operation rather than as post-hoc smoothing.
  3. Per-(cell, bin) amplitude modulation α ∈ [e^{−α_range}, e^{+α_range}].
     Synthesis is no longer linear in ρ^(k); bins with strong contextual
     support can be boosted, weak/ambiguous bins damped.

Pipeline (per image, read from ``cells_flat``):

  rho_out_bins  ∈ ℝ^(nH,nW,K)   per-bin seed readout (gradient-bearing)
  ax_bin, ay_bin∈ ℝ^(N,K)       per-bin sub-pixel anchors  (detached)
  theta_bins    ∈ ℝ^K           bin centers θ̄_k

Per active (cell c, bin k):

  F^(k)_c ∈ ℝ⁵: [ρ^(k), <ρ^(k) pos_t^(k)>_𝒩, signed_tangent_asym,
                  Σ_{j≠k} ρ^(j), signed_normal_asym]
  (κ, e, δ_n, log α) = bounded(MLP_{5→12→4}(F^(k)))
  g^(k)_c = σ(α_g (ρ^(k) − τ · max_j ρ^(j))) · ok(c)             sparsity gate

  Anchor shifted in normal direction by δ_n:
    tilde_a^(k) = a^(k) + δ_n · n̂_k^img        n̂_k^img = (cos θ̄, −sin θ̄)

  Per pixel p in deposit window (anchor frame, (dy, dx)=(row, col) order):
    Δ_x = p_x − tilde_a_x,  Δ_y = p_y − tilde_a_y
    s   =  Δ_y cos θ̄_k + Δ_x sin θ̄_k         tangent component
    n   = −Δ_y sin θ̄_k + Δ_x cos θ̄_k         normal  component
    s̃   = s − e
    n_c = n − ½ κ s̃²
    f^(k)_c(p) = max(0, h_⊥(n_c) · h_∥(s̃))    (relu — claim is non-negative)
    claim^(k)_c(p) = α · g^(k)_c · ρ^(k) · f^(k)_c(p)

Aggregation (noisy-OR across ALL active (c, k)):
  B̂(p) = 1 − ∏_{(c,k)} (1 − claim^(k)_c(p))

CONVENTION on θ:
  θ̄_k comes from L1's ½ arg(z₂), which is the GRADIENT angle.  In (dy, dx) =
  (row, col) order, (cos θ̄, sin θ̄) IS the tangent — matching seed.py's
  collinear_facilitation_bins.  In image (col, row) order: tangent = (sin θ̄,
  cos θ̄); normal = (cos θ̄, −sin θ̄).  All projections below use (dy, dx) order.

WHY FBP-LIKE:
  The whole stack factors as forward-Radon (L0 + L1) → gain-control (seed) →
  inverse-Radon (this renderer).  The previous renderer was missing the *filter*
  slot in the inverse — synthesis was unfiltered back-projection of Gaussian
  mollifiers, which fattens edges.  Learning h_⊥ as the radial filter lets the
  renderer place arbitrary 1D profiles (sharp delta-like, Gaussian, ramp-with-
  side-lobes, etc.) and produces visibly thinner edges when the data warrants.

Param inventory (with H_w = 4 from L1.PATCH_SIZE=5, PATCH_OVERLAP=3 → S=2):
  Correction MLP 5→12→4         60 + 12 + 48 + 4 = 124
  4 correction bounds            (κ_max, e_max, δ_n_max, α_range)         4
  2 gate scalars                 (τ, α_g)                                 2
  h_⊥: phi_perp ∈ ℝ^(H_w+1) (free-sign sides, +peak softplus)             5
  h_∥: psi_par ∈ ℝ^H_w (monotone-decay logits) + peak softplus            5
  Total                                                                  140
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from params import L1, RENDER


# ═══════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════

_DEPOSIT_HALF_WIDTH_STRIDES = float(getattr(RENDER, "DEPOSIT_HALF_WIDTH_STRIDES", 2.0))
_DEPOSIT_HALF_WIDTH_MIN = int(getattr(RENDER, "DEPOSIT_HALF_WIDTH_MIN", 4))
_DEPOSIT_HALF_WIDTH_MAX = int(getattr(RENDER, "DEPOSIT_HALF_WIDTH_MAX", 24))

# Canonical stride for the cell grid (P=5, overlap=3 → S=2).  The kernel half-
# width is fixed at module construction time using this stride, so the learned
# 1D filters have a fixed length and can be saved / loaded as tensors.  If you
# change L1.PATCH_SIZE or PATCH_OVERLAP, fresh-init the renderer.
_DEFAULT_S = max(1, int(getattr(L1, "PATCH_SIZE", 5)) - int(getattr(L1, "PATCH_OVERLAP", 3)))


def _deposit_half_width(S: int) -> int:
    h = int(math.ceil(_DEPOSIT_HALF_WIDTH_STRIDES * max(S, 1)))
    return max(_DEPOSIT_HALF_WIDTH_MIN, min(_DEPOSIT_HALF_WIDTH_MAX, h))


_KERNEL_H_W = _deposit_half_width(_DEFAULT_S)        # canonical kernel half-width (=4)

# Correction MLP topology.  5 features → 12 hidden → 4 outputs.
_FEATURE_DIM = 5
_HIDDEN_DIM = int(getattr(RENDER, "CORR_HIDDEN", 12))
_OUT_DIM = 4

# Kernel init shape (Gaussian σ) — kernels learn freely from these starting points.
_SIGMA_PERP_INIT = float(getattr(RENDER, "SIGMA_PERP_INIT", 0.6))
_SIGMA_PAR_INIT  = float(getattr(RENDER, "SIGMA_PAR_INIT", 2.0))

# Correction bounds (signed, applied via tanh on raw MLP outputs).
_KAPPA_MAX_INIT    = float(getattr(RENDER, "KAPPA_MAX_INIT", 0.1))
_EXT_MAX_INIT      = float(getattr(RENDER, "EXT_MAX_INIT", 1.0))
_DELTA_N_MAX_INIT  = float(getattr(RENDER, "DELTA_N_MAX_INIT", 1.0))
_ALPHA_RANGE_INIT  = float(getattr(RENDER, "ALPHA_RANGE_INIT", 0.5))

# Sparsity gate.
_BIN_GATE_TAU_INIT   = float(getattr(RENDER, "BIN_GATE_TAU_INIT", 0.4))
_BIN_GATE_ALPHA_INIT = float(getattr(RENDER, "BIN_GATE_ALPHA_INIT", 10.0))

# Active-pair compute-saving thresholds.
_GATE_ACTIVE_THRESHOLD = 1e-3
_RHO_ACTIVE_FLOOR = 1e-4

# Numerical clip on claims for log1p(-claim) stability.
_CLAIM_CLIP = 1.0 - 1e-5

# Softfloor on feature denominators.
_FEAT_SOFTFLOOR = 5e-2


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


# ═══════════════════════════════════════════════════════════════
# Per-bin features (5 scalars per (c, k))
# ═══════════════════════════════════════════════════════════════

def _per_bin_features(
    rho_bins: torch.Tensor,    # (nH, nW, K) — rho_out from seed (detached at use site)
    bar_theta: torch.Tensor,   # (K,)
    ib_g: torch.Tensor,        # (nH, nW) bool
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-bin features F^(k)_c ∈ ℝ⁵.

      f0: ρ^(k)(c)                                                    self-bin energy
      f1: <ρ^(k) · pos_t^(k)>_𝒩 / <pos_t^(k)>_𝒩                       collinear support
      f2: signed tangent asymmetry of ρ^(k) (∈ [-1, 1])              drives ext shift e
      f3: Σ_{j≠k} ρ^(j)(c)                                            competing-bin energy
      f4: signed normal  asymmetry of ρ^(k) (∈ [-1, 1])              drives normal shift δ_n

    pos_t^(k)(δ) = (δ·t̂_k)² / |δ|².  Tangent in (dy, dx): t̂ = (cos θ̄, sin θ̄).
    Normal  in (dy, dx): n̂ = (−sin θ̄, cos θ̄).

    Returns: (nH, nW, K, 5).  Detached at the call site before feeding the MLP.
    """
    _ = eps
    nH, nW, K = rho_bins.shape
    dtype = rho_bins.dtype

    cos_b = torch.cos(bar_theta).view(1, 1, K)
    sin_b = torch.sin(bar_theta).view(1, 1, K)

    def _pad0_hwk(t: torch.Tensor) -> torch.Tensor:
        x = t.permute(2, 0, 1).unsqueeze(0)             # (1, K, nH, nW)
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        return x.squeeze(0).permute(1, 2, 0)            # (nH+2, nW+2, K)

    rho_p = _pad0_hwk(rho_bins)

    sum_pos_rho     = torch.zeros_like(rho_bins)
    sum_pos         = torch.zeros_like(rho_bins)
    sum_t_rho       = torch.zeros_like(rho_bins)
    sum_abs_t_rho   = torch.zeros_like(rho_bins)
    sum_n_rho       = torch.zeros_like(rho_bins)
    sum_abs_n_rho   = torch.zeros_like(rho_bins)

    sum_all_bins = rho_bins.sum(dim=-1, keepdim=True)
    f3 = sum_all_bins - rho_bins

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sl = (slice(1 + dy, 1 + dy + nH), slice(1 + dx, 1 + dx + nW))
            rn = rho_p[sl]                              # (nH, nW, K)
            # Tangent and normal projections in (dy, dx) order — matches seed.
            t_proj =  float(dy) * cos_b + float(dx) * sin_b
            n_proj = -float(dy) * sin_b + float(dx) * cos_b
            d2 = float(dy * dy + dx * dx)
            pos_t = (t_proj * t_proj) / d2

            sum_pos_rho   = sum_pos_rho   + rn * pos_t
            sum_pos       = sum_pos       + pos_t.expand_as(rn)
            sum_t_rho     = sum_t_rho     + rn * t_proj
            sum_abs_t_rho = sum_abs_t_rho + rn * t_proj.abs()
            sum_n_rho     = sum_n_rho     + rn * n_proj
            sum_abs_n_rho = sum_abs_n_rho + rn * n_proj.abs()

    f1 = sum_pos_rho / (sum_pos       + _FEAT_SOFTFLOOR)
    f2 = sum_t_rho   / (sum_abs_t_rho + _FEAT_SOFTFLOOR)
    f4 = sum_n_rho   / (sum_abs_n_rho + _FEAT_SOFTFLOOR)

    use = (~ib_g).to(dtype=dtype).unsqueeze(-1)
    f0 = rho_bins * use
    f1 = f1 * use
    f2 = f2 * use
    f3 = f3 * use
    f4 = f4 * use

    return torch.stack([f0, f1, f2, f3, f4], dim=-1)    # (nH, nW, K, 5)


# ═══════════════════════════════════════════════════════════════
# 1D kernel interpolation (analogue of FBP filter sampling)
# ═══════════════════════════════════════════════════════════════




class _InterpKernel(torch.autograd.Function):
    """Linear interpolation of 1D kernel ``h`` at continuous positions ``u``.

    Custom Function: ALL forward intermediates (u_floor, u_frac, m_lo, m_hi,
    out_range, h_lo, h_hi, etc.) are local to ``forward`` and freed on return.
    ``backward`` recomputes them from saved (h, u) — both of which are already
    alive in the surrounding autograd graph, so this saves ~12 retained
    intermediate tensors of shape ``u.shape`` per call in exchange for one
    recomputation pass during backward.

    In the renderer, ``_interp_kernel`` is called twice per chunk on
    ``(bs, P)`` tiles; the prior naive autograd implementation dominated peak
    activation memory.
    """

    @staticmethod
    def forward(ctx, h: torch.Tensor, u: torch.Tensor, H_w: int) -> torch.Tensor:
        L = 2 * H_w
        u_floor = torch.floor(u)
        u_frac = u - u_floor                            # ∈ [0, 1)
        m_lo = u_floor.to(torch.long) + H_w
        m_hi = m_lo + 1
        out_range = (m_lo < 0) | (m_hi > L)
        m_lo.clamp_(0, L)
        m_hi.clamp_(0, L)
        # h[m_lo] + u_frac * (h[m_hi] - h[m_lo])  — one fewer multiply than
        # (1 - u_frac) * h[m_lo] + u_frac * h[m_hi].
        val = h[m_lo] + u_frac * (h[m_hi] - h[m_lo])
        val = val.masked_fill(out_range, 0.0)
        ctx.save_for_backward(h, u)
        ctx.H_w = H_w
        return val

    @staticmethod
    def backward(ctx, grad_val: torch.Tensor):
        h, u = ctx.saved_tensors
        H_w = ctx.H_w
        L = 2 * H_w

        # Recompute geometry.  Cheaper than retaining ~12 tiles in forward.
        u_floor = torch.floor(u)
        u_frac = u - u_floor
        m_lo = u_floor.to(torch.long) + H_w
        m_hi = m_lo + 1
        out_range = (m_lo < 0) | (m_hi > L)
        m_lo.clamp_(0, L)
        m_hi.clamp_(0, L)

        gv = grad_val.masked_fill(out_range, 0.0)        # zero contribution outside support

        grad_h = None
        grad_u = None
        # needs_input_grad order matches forward's positional args: (h, u, H_w).
        if ctx.needs_input_grad[1]:                      # grad w.r.t. u
            # ∂val/∂u = (h[m_hi] − h[m_lo]) on the in-range support, 0 elsewhere.
            grad_u = gv * (h[m_hi] - h[m_lo])
        if ctx.needs_input_grad[0]:                      # grad w.r.t. h
            # ∂val/∂h[i] aggregates linear-interp weights from every (b, p)
            # whose m_lo or m_hi indexes i.
            grad_h = torch.zeros_like(h)
            w_lo = (1.0 - u_frac) * gv
            w_hi = u_frac * gv
            grad_h.scatter_add_(0, m_lo.reshape(-1), w_lo.reshape(-1))
            grad_h.scatter_add_(0, m_hi.reshape(-1), w_hi.reshape(-1))
        return grad_h, grad_u, None                      # H_w (int) has no gradient


def _interp_kernel(h: torch.Tensor, u: torch.Tensor, H_w: int) -> torch.Tensor:
    """Linear interpolation of 1D kernel ``h`` at continuous positions ``u``.

    Memory-efficient wrapper around ``_InterpKernel``.  Behavior identical to
    the previous naive-autograd implementation; verified by gradcheck against
    finite differences on h and u.
    """
    return _InterpKernel.apply(h, u, H_w)


# ═══════════════════════════════════════════════════════════════
# Synthesis: per-(c, k) FBP-style stroke + global noisy-OR
# ═══════════════════════════════════════════════════════════════

def _chunk_log_neg(
    rho_c: torch.Tensor,            # (bs,)
    gate_c: torch.Tensor,           # (bs,)
    alpha_c: torch.Tensor,          # (bs,)
    ax_f: torch.Tensor,             # (bs, 1)
    ay_f: torch.Tensor,             # (bs, 1)
    ca: torch.Tensor,               # (bs, 1)
    sa: torch.Tensor,               # (bs, 1)
    kappa_c: torch.Tensor,          # (bs,)
    ext_s_c: torch.Tensor,          # (bs,)
    ox_l: torch.Tensor,             # (P,)  detached
    oy_l: torch.Tensor,             # (P,)  detached
    h_perp_l: torch.Tensor,         # (2H+1,)
    h_par_l: torch.Tensor,          # (2H+1,)
    in_bounds_l: torch.Tensor,      # (bs, P) bool, no grad
    half_w: int,
) -> torch.Tensor:
    """Synthesis math for one chunk — wrapped in ``checkpoint`` from the caller.

    All intermediate ``(bs, P)`` tensors (``s, n, s_tilde, n_curv, h_perp_val,
    h_par_val, f_val, claim, claim_safe``) are local; on backward they're
    recomputed instead of retained.  Returns ``log1p(-claim_safe)`` of shape
    ``(bs, P)`` — the only tensor handed back across the checkpoint boundary.
    """
    dx = ox_l.unsqueeze(0) - ax_f                       # (bs, P)
    dy = oy_l.unsqueeze(0) - ay_f

    # Tangent / normal projection — (dy, dx) order, matches seed convention.
    s =  dy * ca + dx * sa
    n = -dy * sa + dx * ca

    s_tilde = s - ext_s_c.unsqueeze(1)
    n_curv  = n - 0.5 * kappa_c.unsqueeze(1) * (s_tilde * s_tilde)

    # Sample the learned 1D kernels at continuous (n_curv, s_tilde).
    h_perp_val = _interp_kernel(h_perp_l, n_curv,  half_w)
    h_par_val  = _interp_kernel(h_par_l,  s_tilde, half_w)

    # ReLU keeps claim non-negative for noisy-OR probabilistic semantics
    # (h_perp may have negative side-lobes → sharpening via suppression
    # rather than via subtraction).
    f_val = F.relu(h_perp_val * h_par_val)

    amp = (alpha_c * gate_c * rho_c).unsqueeze(1)       # (bs, 1)
    claim = (amp * f_val).clamp(min=0.0, max=_CLAIM_CLIP)
    claim_safe = torch.where(in_bounds_l, claim, torch.zeros_like(claim))
    return torch.log1p(-claim_safe)                     # (bs, P)


def _fbp_deposit(
    rho_active: torch.Tensor,       # (A,) ρ^(k)_c                   grad-bearing
    gate_active: torch.Tensor,      # (A,) g^(k)_c                   grad-bearing
    alpha_active: torch.Tensor,     # (A,) α^(k)_c                   grad-bearing
    ax_active: torch.Tensor,        # (A,) anchor x (pixels)         detached
    ay_active: torch.Tensor,        # (A,) anchor y (pixels)         detached
    cos_a: torch.Tensor,            # (A,) cos θ̄_k                   detached
    sin_a: torch.Tensor,            # (A,) sin θ̄_k                   detached
    kappa_active: torch.Tensor,     # (A,) κ^(k)_c                   grad-bearing
    ext_s_active: torch.Tensor,     # (A,) e^(k)_c                   grad-bearing
    delta_n_active: torch.Tensor,   # (A,) δ_n^(k)_c                 grad-bearing
    h_perp: torch.Tensor,           # (2*H_w+1,) radial filter       grad-bearing
    h_par: torch.Tensor,            # (2*H_w+1,) longitudinal profile grad-bearing
    kernel_h_w: int,
    H: int, W: int,
    eps: float = 1e-6,
    use_checkpoint: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(c, k) stroke synthesis via interpolated 1D kernels, then noisy-OR.

    Memory layout:
      * Synthesis math runs inside ``_chunk_log_neg``, wrapped per chunk in
        ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``.  Each
        chunk's ``(bs, P)`` intermediates (``s, n, s_tilde, n_curv,
        h_perp_val, h_par_val, f_val, claim, claim_safe``) are freed after
        forward and recomputed in backward.  Only the returned ``log_neg`` of
        shape ``(bs, P)`` is retained per chunk.
      * The in-place ``scatter_add_`` into ``log_neg_acc`` and the no-grad θ★
        tracking stay OUTSIDE the checkpoint so they're not re-executed.
      * Pixel-index arithmetic (``px, py, in_bounds, flat_idx``) is computed
        outside the checkpoint (no grad anyway) so backward doesn't redo it.

    With this layout, peak retained activation memory in the synthesis is
    roughly ``2·bs·P·sizeof(dtype)`` per live chunk (``log_neg`` + the bool
    mask) rather than the ~6× that figure under naive autograd.

    Returns:
        bmap:       (H, W) ∈ [0, 1].
        theta_star: (H, W) — bar_theta_k of the dominant claimant (for NMS).
    """
    A = rho_active.shape[0]
    device, dtype = rho_active.device, rho_active.dtype
    n_pix = H * W

    if A == 0:
        z = torch.zeros(H, W, device=device, dtype=dtype)
        return z, z

    half_w = kernel_h_w
    offsets = torch.arange(-half_w, half_w + 1, device=device, dtype=dtype)
    oy_g, ox_g = torch.meshgrid(offsets, offsets, indexing="ij")
    oy = oy_g.reshape(-1)
    ox = ox_g.reshape(-1)
    P = oy.shape[0]

    log_neg_acc = torch.zeros(n_pix, device=device, dtype=dtype)
    max_claim = torch.full((n_pix,), -1.0, device=device, dtype=torch.float32)
    theta_star = torch.zeros(n_pix, device=device, dtype=dtype)

    # 2D anchor correction: shift by δ_n along normal in image (col, row) coords.
    #   tangent in image (col, row) = (sin θ̄, cos θ̄)
    #   normal  in image (col, row) = (cos θ̄, −sin θ̄)
    ax_eff = ax_active + delta_n_active * cos_a
    ay_eff = ay_active - delta_n_active * sin_a

    # Integer base of CORRECTED anchor so the deposit window stays centered
    # on the actual stroke peak (not on the uncorrected anchor).
    ax_int = torch.floor(ax_eff).long()
    ay_int = torch.floor(ay_eff).long()
    ax_frac = ax_eff - ax_int.to(dtype=dtype)
    ay_frac = ay_eff - ay_int.to(dtype=dtype)

    bar_theta_active = torch.atan2(sin_a, cos_a)        # for θ★

    max_batch = max(1, 4_000_000 // P)

    # Checkpointing only matters under autograd; inside ``torch.no_grad()``
    # (inference) we skip the wrapper to avoid its overhead and keep behavior
    # bit-for-bit identical to the eager path.
    do_ckpt = bool(use_checkpoint) and torch.is_grad_enabled()

    for b0 in range(0, A, max_batch):
        b1 = min(b0 + max_batch, A)
        bs = b1 - b0

        ax_f = ax_frac[b0:b1].unsqueeze(1)              # (bs, 1)
        ay_f = ay_frac[b0:b1].unsqueeze(1)
        ca = cos_a[b0:b1].unsqueeze(1)
        sa = sin_a[b0:b1].unsqueeze(1)

        # Pixel-index arithmetic (no grad — computed once, outside checkpoint).
        px = ax_int[b0:b1].unsqueeze(1) + ox.unsqueeze(0).long()
        py = ay_int[b0:b1].unsqueeze(1) + oy.unsqueeze(0).long()
        in_bounds = (py >= 0) & (py < H) & (px >= 0) & (px < W)
        flat_idx = (py.clamp(0, H - 1) * W + px.clamp(0, W - 1))

        # Synthesis math — heavy intermediates live behind the checkpoint.
        chunk_args = (
            rho_active[b0:b1], gate_active[b0:b1], alpha_active[b0:b1],
            ax_f, ay_f, ca, sa,
            kappa_active[b0:b1], ext_s_active[b0:b1],
            ox, oy,
            h_perp, h_par,
            in_bounds, half_w,
        )
        if do_ckpt:
            log_neg = checkpoint(_chunk_log_neg, *chunk_args, use_reentrant=False)
        else:
            log_neg = _chunk_log_neg(*chunk_args)

        # In-place scatter into the global log-space accumulator — kept OUT of
        # the checkpoint because (a) it's in-place on a tensor that crosses
        # iterations, (b) its backward only needs ``flat_idx`` and the source
        # gradient, neither of which benefits from recomputation.
        log_neg_acc.scatter_add_(0, flat_idx.reshape(-1), log_neg.reshape(-1))

        # θ★ tracking (no gradient).  ``claim_safe`` is not retained from the
        # checkpoint; recover it from ``log_neg`` (precision irrelevant since
        # this branch carries no gradient and only feeds NMS at inference).
        with torch.no_grad():
            cl_det = (-torch.expm1(log_neg.detach())).to(torch.float32).reshape(-1)
            idx_flat = flat_idx.reshape(-1)
            th_expand = bar_theta_active[b0:b1].unsqueeze(1).expand(bs, P).reshape(-1)
            cur = max_claim[idx_flat]
            upd = cl_det > cur
            if upd.any():
                upd_idx = idx_flat[upd]
                max_claim[upd_idx] = cl_det[upd]
                theta_star[upd_idx] = th_expand[upd].to(dtype=dtype)

    bmap = -torch.expm1(log_neg_acc)
    return bmap.reshape(H, W), theta_star.reshape(H, W)


# ═══════════════════════════════════════════════════════════════
# Correction MLP (5 → 12 → 4, shared across (c, k))
# ═══════════════════════════════════════════════════════════════

class CorrectionMLP(nn.Module):
    """F^(k)_c → (raw_κ, raw_e, raw_δ_n, raw_log_α).

    Outputs are tanh-bounded in the renderer's forward path; this module just
    emits unbounded scalars.  Initialized so κ ≈ e ≈ δ_n ≈ 0 and log α ≈ 0 at
    t=0 → straight strokes at the un-corrected anchor with α = 1, recovering
    the prior renderer's geometry exactly.
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
        h = F.relu(self.fc1(features))
        return self.fc2(h)


# ═══════════════════════════════════════════════════════════════
# ModulationRenderer
# ═══════════════════════════════════════════════════════════════

class ModulationRenderer(nn.Module):
    """FBP-style renderer with learned 1D kernels + 2D anchor correction.

    Learned parameters (140 total at H_w=4):
        CorrectionMLP 5→12→4                     124
        κ_max, e_max, δ_n_max, α_range             4
        τ, α_g                                     2
        h_⊥ raw (H_w+1 entries)                    5
        h_∥ raw: H_w decay-ratio logits + peak     5
    """

    def __init__(self, **kwargs):
        super().__init__()
        _ = kwargs

        # Kernel half-width fixed at construction.  Filters have length 2*H_w + 1.
        self._kernel_h_w = int(_KERNEL_H_W)
        H = self._kernel_h_w

        self.correction = CorrectionMLP()

        # ── h_⊥ parameterization (radial filter; FBP-analogue) ────────────
        # Raw vector of length H+1, indexed by |m|.
        # phi_perp[0] = softplus(raw[0])           positive peak (peak amplitude)
        # phi_perp[m] = raw[m]  for |m| ≥ 1        free-sign side-lobes
        # h_⊥[m] = phi_perp[|m|]                   reflective even-symmetry
        # Init: Gaussian with σ = _SIGMA_PERP_INIT.
        sig_p = max(_SIGMA_PERP_INIT, 1e-6)
        phi_init = torch.tensor([
            math.exp(-(j * j) / (2.0 * sig_p * sig_p))
            for j in range(H + 1)
        ], dtype=torch.float32)
        phi_init_raw = phi_init.clone()
        phi_init_raw[0] = _inv_softplus(float(phi_init[0]))   # softplus⁻¹ of 1.0
        # Remaining (free-sign) entries pass through verbatim.
        self._phi_perp_raw = nn.Parameter(phi_init_raw)

        # ── h_∥ parameterization (longitudinal profile) ───────────────────
        # Even, monotone-decay from positive peak.
        # peak = softplus(_h_par_peak_raw)
        # ratios[j] = sigmoid(_psi_par_raw[j])  ∈ (0, 1)   for j = 0..H-1
        # tilde_h[0] = 1; tilde_h[m] = ∏_{i<|m|} ratios[i]   (decays in |m|)
        # h_∥[m] = peak · tilde_h[|m|]
        # Init: Gaussian with σ = _SIGMA_PAR_INIT.
        sig_a = max(_SIGMA_PAR_INIT, 1e-6)
        ratios_init = torch.tensor([
            math.exp(-(2 * j + 1) / (2.0 * sig_a * sig_a))
            for j in range(H)
        ], dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        # logit
        psi_init_raw = torch.log(ratios_init / (1.0 - ratios_init))
        self._psi_par_raw = nn.Parameter(psi_init_raw)
        self._h_par_peak_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0)))

        # ── Correction bounds ────────────────────────────────────────────
        self._kappa_max_raw   = nn.Parameter(torch.tensor(_inv_softplus(_KAPPA_MAX_INIT)))
        self._ext_max_raw     = nn.Parameter(torch.tensor(_inv_softplus(_EXT_MAX_INIT)))
        self._delta_n_max_raw = nn.Parameter(torch.tensor(_inv_softplus(_DELTA_N_MAX_INIT)))
        self._alpha_range_raw = nn.Parameter(torch.tensor(_inv_softplus(_ALPHA_RANGE_INIT)))

        # ── Sparsity gate ─────────────────────────────────────────────────
        self._alpha_g_raw = nn.Parameter(torch.tensor(_inv_softplus(_BIN_GATE_ALPHA_INIT)))
        tau0 = max(min(_BIN_GATE_TAU_INIT, 1.0 - 1e-4), 1e-4)
        self._tau_raw = nn.Parameter(torch.tensor(math.log(tau0 / (1.0 - tau0))))

    # ── Derived filter tensors ──────────────────────────────────────────────

    @property
    def h_perp(self) -> torch.Tensor:
        """Even-symmetric radial filter, length 2*H_w + 1."""
        H = self._kernel_h_w
        raw = self._phi_perp_raw
        peak = F.softplus(raw[0:1])                     # (1,)
        side = raw[1:]                                  # (H,) free sign
        phi = torch.cat([peak, side], dim=0)            # (H+1,)
        idx = torch.arange(-H, H + 1, device=raw.device).abs()
        return phi[idx]                                 # (2H+1,)

    @property
    def h_par(self) -> torch.Tensor:
        """Even, monotone-decay longitudinal profile, length 2*H_w + 1."""
        H = self._kernel_h_w
        peak = F.softplus(self._h_par_peak_raw)
        ratios = torch.sigmoid(self._psi_par_raw)       # (H,) ∈ (0, 1)
        # tilde_h[0] = 1; tilde_h[j] = ∏_{i<j} ratios[i]  for j = 1..H
        cumprod = torch.cumprod(ratios, dim=0)          # (H,)
        tilde_h = torch.cat([
            torch.ones(1, device=ratios.device, dtype=ratios.dtype),
            cumprod,
        ], dim=0)                                       # (H+1,)
        idx = torch.arange(-H, H + 1, device=ratios.device).abs()
        return peak * tilde_h[idx]                      # (2H+1,)

    # ── Scalar properties ───────────────────────────────────────────────────

    @property
    def kappa_max(self) -> torch.Tensor:
        return F.softplus(self._kappa_max_raw).view(())

    @property
    def ext_max(self) -> torch.Tensor:
        return F.softplus(self._ext_max_raw).view(())

    @property
    def delta_n_max(self) -> torch.Tensor:
        return F.softplus(self._delta_n_max_raw).view(())

    @property
    def alpha_range(self) -> torch.Tensor:
        return F.softplus(self._alpha_range_raw).view(())

    @property
    def alpha_g(self) -> torch.Tensor:
        return F.softplus(self._alpha_g_raw).view(())

    @property
    def tau(self) -> torch.Tensor:
        return torch.sigmoid(self._tau_raw).view(())

    @property
    def kernel_h_w(self) -> int:
        return int(self._kernel_h_w)

    # ── Legacy compat properties (display / printing) ───────────────────────
    # The previous renderer had σ⊥, σ∥ as direct scalar params.  In the FBP
    # version they're replaced by the learned 1D kernels, but external print
    # code may still read these.  We expose the *init* widths so the names
    # don't break — actual kernel shape lives in `h_perp` / `h_par`.

    @property
    def sigma_perp(self) -> torch.Tensor:
        return torch.tensor(_SIGMA_PERP_INIT, dtype=torch.float32)

    @property
    def sigma_par(self) -> torch.Tensor:
        return torch.tensor(_SIGMA_PAR_INIT, dtype=torch.float32)

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
    """Strip legacy renderer keys and shape-incompatible correction-MLP keys."""
    legacy = {
        # Older renderers (Gaussian-stroke 82-version, soft-indicator basis,
        # stencil-thinning splat, etc.).  None of these exist in the FBP version.
        f"{prefix}_sigma_perp_raw",
        f"{prefix}_sigma_par_raw",
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
    # Correction MLP shape changed (4→8→2  becomes  5→12→4).  Strip any
    # incompatible-shape entries so the new module fresh-inits its MLP.
    expected_fc1_w = (_HIDDEN_DIM, _FEATURE_DIM)
    expected_fc1_b = (_HIDDEN_DIM,)
    expected_fc2_w = (_OUT_DIM, _HIDDEN_DIM)
    expected_fc2_b = (_OUT_DIM,)
    out = {}
    for k, v in state_dict.items():
        if k in legacy:
            continue
        if k == f"{prefix}correction.fc1.weight" and tuple(v.shape) != expected_fc1_w:
            continue
        if k == f"{prefix}correction.fc1.bias"   and tuple(v.shape) != expected_fc1_b:
            continue
        if k == f"{prefix}correction.fc2.weight" and tuple(v.shape) != expected_fc2_w:
            continue
        if k == f"{prefix}correction.fc2.bias"   and tuple(v.shape) != expected_fc2_b:
            continue
        out[k] = v
    return out


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
    """B̂(p) = 1 − ∏_{(c,k)} (1 − α · g · ρ · h_⊥(n_c) · h_∥(s̃))_+.

    The scalar ``rho_cell`` arg is kept for signature compatibility but is
    unused — gradient flows through ``cells_flat['rho_out_bins']``.
    """
    _ = (rho_cell, training, l0_pix, branch_pick)
    H, W = proj_dev["H"], proj_dev["W"]
    device = next(renderer.parameters()).device
    dtype = renderer.h_perp.dtype

    if "rho_out_bins" not in cells_flat:
        raise ValueError(
            "FBP renderer requires cells_flat['rho_out_bins'] from the seed."
        )
    if "ax_bin" not in cells_flat or "ay_bin" not in cells_flat:
        raise ValueError("cells_flat must include ax_bin and ay_bin (from L1).")
    if "theta_bins" not in cells_flat:
        raise ValueError("cells_flat must include theta_bins (from L1).")

    nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
    N = nH * nW

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

    # ── Step 1: per-bin features (rho detached for MLP input) ──────────
    feats = _per_bin_features(rho_out_bins.detach(), bar_theta, ib_g, eps=eps)
    feats_flat = feats.reshape(N * K, _FEATURE_DIM)

    # ── Step 2: sparsity gate + active-set selection (BEFORE the MLP) ───
    # The previous order ran the MLP on all N·K features and discarded ≥90 %
    # of the output.  Selecting first means the MLP only sees the active
    # subset — substantially smaller forward + retained graph.
    rho_flat_bins = rho_out_bins.reshape(N, K)
    rho_max_cell = rho_flat_bins.max(dim=-1, keepdim=True).values
    ok = (~is_border).to(dtype=dtype).unsqueeze(-1)
    gate_flat = (
        torch.sigmoid(renderer.alpha_g * (rho_flat_bins - renderer.tau * rho_max_cell))
        * ok
    )

    gate_NK = gate_flat.reshape(-1)
    rho_NK = rho_flat_bins.reshape(-1)
    keep = (gate_NK > _GATE_ACTIVE_THRESHOLD) & (rho_NK > _RHO_ACTIVE_FLOOR)

    if not bool(keep.any().item()):
        # No-op fallback — keep autograd graph alive on every renderer param.
        # The correction MLP didn't run on a real input here, so a one-row
        # sentinel pass keeps its weights in the graph too.
        h_perp = renderer.h_perp
        h_par  = renderer.h_par
        zero = (
            h_perp.sum() * 0.0 + h_par.sum() * 0.0
            + renderer.kappa_max * 0.0 + renderer.ext_max * 0.0
            + renderer.delta_n_max * 0.0 + renderer.alpha_range * 0.0
            + renderer.tau * 0.0 + renderer.alpha_g * 0.0
            + 0.0 * rho_out_bins.sum()
            + 0.0 * renderer.correction(feats_flat[:1]).sum()
        )
        bmap = torch.zeros(H, W, device=device, dtype=dtype) + zero
        theta_star = torch.zeros(H, W, device=device, dtype=dtype)
        out = bmap[:Hp, :Wp]
        if return_dominant_theta:
            return out, theta_star[:Hp, :Wp]
        return out

    # ── Step 3: correction MLP on the active subset only ───────────────
    feats_active = feats_flat[keep]                     # (A, 5)
    raw_active = renderer.correction(feats_active)      # (A, 4)
    kappa_active     = renderer.kappa_max   * torch.tanh(raw_active[:, 0])
    ext_s_active     = renderer.ext_max     * torch.tanh(raw_active[:, 1])
    delta_n_active   = renderer.delta_n_max * torch.tanh(raw_active[:, 2])
    log_alpha_active = renderer.alpha_range * torch.tanh(raw_active[:, 3])
    alpha_active = torch.exp(log_alpha_active)          # α ∈ [e^{-range}, e^{+range}]

    # ── Step 4: gather remaining active inputs (anchors, ρ, gate, θ) ───
    bin_idx_full = torch.arange(K, device=device).unsqueeze(0).expand(N, K).reshape(-1)
    active_bin = bin_idx_full[keep]

    ax_active   = ax_bin.reshape(-1)[keep].detach()
    ay_active   = ay_bin.reshape(-1)[keep].detach()
    rho_active  = rho_NK[keep]
    gate_active = gate_NK[keep]

    cos_a = torch.cos(bar_theta).index_select(0, active_bin)
    sin_a = torch.sin(bar_theta).index_select(0, active_bin)

    # ── Step 5: FBP deposit + noisy-OR ─────────────────────────────────
    bmap, theta_star = _fbp_deposit(
        rho_active=rho_active,
        gate_active=gate_active,
        alpha_active=alpha_active,
        ax_active=ax_active,
        ay_active=ay_active,
        cos_a=cos_a,
        sin_a=sin_a,
        kappa_active=kappa_active,
        ext_s_active=ext_s_active,
        delta_n_active=delta_n_active,
        h_perp=renderer.h_perp,
        h_par=renderer.h_par,
        kernel_h_w=renderer._kernel_h_w,
        H=H, W=W, eps=eps,
    )

    # ── Crop to content region ─────────────────────────────────────────
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
# NMS (unchanged)
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


# Legacy aliases
HarmonicThinRenderer = ModulationRenderer
StampRenderer = ModulationRenderer
AnisoDiffusionRenderer = ModulationRenderer