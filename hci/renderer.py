r"""Renderer — harmonic-native gating + recurrent collinear facilitation.

Pipeline:
  1. Cell grid: ρ-weighted θ combing (unchanged).
  2. Recurrent V1-style collinear facilitation + cross-orientation
     suppression on cell grid (n_passes iterations, binned-θ conv).
     Each pass: κ = col/(col+cross+ε), ρ ← ρ·κ.
     Effective range ≈ n_passes × R.
  3. Bilinear interpolation of modulated cell-grid fields (ρ_mod, θ,
     κ_col) to pixel coordinates — no scatter-add splat.
  4. Per-pixel feature vector F_p ∈ R¹⁸:
       [h2m_lum, h2m_chr, ρ̄^(T), θ̄_cos2, θ̄_sin2, κ̄_col,
        ρ̄^(T)/(ρ̄^(0)+ε), tang5(5), norm5(5), η_lum_map]
     where ρ̄^(0) is pre–collinear-recurrence ρ (same bilinear interp as ρ̄^(T)),
     tang5/norm5 are sampled from h2m (pixel-native), and η_lum_map is the
     pass-2 luminance η field (optional key ``eta_mod_map`` in ``l0_pix``;
     zeros if absent).
  5. Thinning head: B̂(p) = h2m(p) · σ(W₂ ReLU(W₁ F_p + b₁) + b₂).
     18→16→1 MLP.

Collinear coherence (Step 2):
  Recurrent divisive normalization on the cell grid.  Each cell's θ-bin
  selects the collinear kernel; the orthogonal bin (θ + 90°) selects
  the cross-oriented kernel.  κ = col_energy / (col + cross + ε) is
  Naka–Rushton: co-aligned neighbors facilitate, cross-oriented suppress.
  ρ is modulated by κ at each pass, so facilitation propagates along
  contours and cross-orientation suppression cascades — gap bridging
  and junction resolution emerge from the recurrence.

Learned: s_t (1), s_n (1), ThinningHead 18→16→1 (321 params).
Fixed: collinear kernels (precomputed), recurrent dynamics (0 params).
Total: 323 learned scalars.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import RENDER


# ═══════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════

_SIGMA_PERP_INIT = getattr(RENDER, "SIGMA_PERP_INIT", 1.5)
_SIGMA_PERP_MAX = getattr(RENDER, "SIGMA_PERP_MAX", 8.0)
_SPLAT_RADIUS_SIGMAS = getattr(RENDER, "SPLAT_RADIUS_SIGMAS", 3.0)
_SPLAT_HALF_W_PERP = getattr(RENDER, "SPLAT_HALF_W_PERP", 3)

# Collinear coherence defaults
_COL_RADIUS = getattr(RENDER, "COL_RADIUS", 5)
_COL_K_BINS = getattr(RENDER, "COL_K_BINS", 24)
_COL_SIGMA_D = getattr(RENDER, "COL_SIGMA_D", None)
_COL_SIGMA_T = getattr(RENDER, "COL_SIGMA_T", 1.0)
_COL_PASSES = getattr(RENDER, "COL_PASSES", 3)


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(max(float(x), 1e-8)))


# ═══════════════════════════════════════════════════════════════
# Cell-grid θ combing (unchanged)
# ═══════════════════════════════════════════════════════════════

def _smooth_theta_rho_double_angle(
    theta: torch.Tensor,
    rho: torch.Tensor,
    is_border: torch.Tensor,
    n_passes: int = RENDER.THETA_SMOOTH_PASSES,
    eps: float = 1e-6,
) -> torch.Tensor:
    nH, nW = theta.shape
    th, rh, ib = theta, rho, is_border
    pad = lambda t: F.pad(t[None, None], (1, 1, 1, 1)).squeeze(0).squeeze(0)
    for _ in range(int(n_passes)):
        th_p, rh_p = pad(th), pad(rh)
        u_p = pad(rh * torch.cos(2.0 * th))
        v_p = pad(rh * torch.sin(2.0 * th))
        su = torch.zeros_like(th)
        sv = torch.zeros_like(th)
        sw = torch.zeros_like(th)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                s = slice(1 + dy, 1 + dy + nH), slice(1 + dx, 1 + dx + nW)
                w = rh_p[s]
                su += w * u_p[s]
                sv += w * v_p[s]
                sw += w
        th_new = 0.5 * torch.atan2(sv / sw.clamp_min(eps), su / sw.clamp_min(eps))
        use = (~ib) & (sw > eps)
        th = torch.where(use, th_new, th)
    return th


# ═══════════════════════════════════════════════════════════════
# Collinear coherence (binned-θ convolution on cell grid)
# ═══════════════════════════════════════════════════════════════

def _build_collinear_kernels(
    R: int, K: int, sigma_d: float | None, sigma_t: float,
    device: torch.device, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute K kernels of shape (2R+1, 2R+1).
    Returns: (K, 1, 2R+1, 2R+1) ready for F.conv2d.
    """
    if sigma_d is None:
        sigma_d = max(R / 2.0, 0.5)
    offsets = torch.arange(-R, R + 1, device=device, dtype=dtype)
    di, dj = torch.meshgrid(offsets, offsets, indexing="ij")
    dist_sq = di * di + dj * dj
    w_d = torch.exp(-dist_sq / (2.0 * sigma_d * sigma_d))
    w_d[R, R] = 0.0
    w_d[dist_sq > R * R] = 0.0

    kernels = torch.zeros(K, 2 * R + 1, 2 * R + 1, device=device, dtype=dtype)
    for k in range(K):
        theta_k = k * math.pi / K
        d_perp = dj * math.cos(theta_k) - di * math.sin(theta_k)
        w_t = torch.exp(-d_perp * d_perp / (2.0 * sigma_t * sigma_t))
        kernels[k] = w_d * w_t
    return kernels.unsqueeze(1)


def collinear_rho_snapshot_schedule(
    n_passes: int, *, max_timestamps: int = 5,
) -> list[int]:
    """Pick ``t_after`` indices in ``[0, n_passes]`` for ρ collinear trajectory.

    ``t_after`` counts completed collinear passes (``0`` = before the first pass).
    At most ``max_timestamps`` values, spanning the full recurrence when
    ``n_passes + 1`` exceeds the cap.
    """
    if n_passes < 0:
        raise ValueError("n_passes must be non-negative")
    if max_timestamps < 1:
        raise ValueError("max_timestamps must be >= 1")
    n_states = n_passes + 1
    if n_states <= max_timestamps:
        return list(range(n_states))
    if max_timestamps == 1:
        return [n_passes]
    times: list[int] = []
    denom = max_timestamps - 1
    for k in range(max_timestamps):
        x = (k * n_passes) / denom
        times.append(int(round(x)))
    times = [max(0, min(n_passes, t)) for t in times]
    out: list[int] = []
    for t in times:
        if not out or out[-1] != t:
            out.append(t)
    return out


def compute_collinear_coherence(
    theta_grid: torch.Tensor,
    rho_grid: torch.Tensor,
    is_border_grid: torch.Tensor,
    R: int = _COL_RADIUS,
    K: int = _COL_K_BINS,
    sigma_d: float | None = _COL_SIGMA_D,
    sigma_t: float = _COL_SIGMA_T,
    n_passes: int = _COL_PASSES,
    nr_pool_radius: int | None = None,  # unused, kept for API compat
    eps: float = 1e-6,
    rho_snapshots_t_after: set[int] | frozenset[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """Recurrent V1-style collinear facilitation with GABA-budget normalization.

    Each pass:
      1. Convolve ρ·cos2θ, ρ·sin2θ with K tangent-selective kernels.
      2. Each cell reads its own θ-bin (collinear energy E_col).
      3. Compute total oriented energy across ALL K bins (E_total).
      4. κ = E_col / (E_total + ε)          (GABA budget)
      5. ρ ← ρ · κ                           (modulate)
      6. Feed updated ρ into next pass.

    The GABA budget replaces both the old col-vs-cross gating and the
    separate NR pool in a single normalization.  The denominator is the
    total oriented energy summed over all K angle bins — untuned
    inhibition, matching V1 GABAergic interneuron pooling.

    A cell on a clean edge has most energy in its own bin → κ ≈ 1.
    A cell in texture has energy spread across many bins → κ ≈ 1/K.
    No separate NR pool needed — the energy budget is self-stabilizing
    across iterations because the denominator scales with ρ.

    Args:
        theta_grid: (nH, nW) combed orientations
        rho_grid:   (nH, nW) cell strengths
        is_border_grid: (nH, nW) bool
        R, K, sigma_d, sigma_t: kernel parameters
        n_passes: number of recurrent iterations (default 3)

    Returns:
        kappa_col: (nH, nW) final collinear coherence in [0, 1]
        rho_mod:   (nH, nW) modulated ρ after all passes

        If ``rho_snapshots_t_after`` is not ``None``, also returns a dict mapping
        ``t_after`` → detached cell-grid ``ρ`` snapshot (same shape as ``rho_mod``).
    """
    device, dtype = theta_grid.device, theta_grid.dtype
    nH, nW = theta_grid.shape

    kernels = _build_collinear_kernels(R, K, sigma_d, sigma_t, device, dtype)

    # Bin assignments (fixed across passes — θ doesn't change)
    bin_idx = ((theta_grid % math.pi) * (K / math.pi)).long().clamp(0, K - 1)
    bin_idx_4d = bin_idx.unsqueeze(0).unsqueeze(0)  # (1, 1, nH, nW)

    rho_m = torch.where(is_border_grid, torch.zeros_like(rho_grid), rho_grid)
    kappa_col = torch.zeros_like(rho_m)

    # Cell's own double-angle unit vector (fixed across passes)
    c2 = torch.cos(2.0 * theta_grid)  # cos(2θ_c)
    s2 = torch.sin(2.0 * theta_grid)  # sin(2θ_c)

    want_snaps = rho_snapshots_t_after is not None
    snaps: dict[int, torch.Tensor] = {}
    snap_want: frozenset[int] = frozenset()
    if want_snaps:
        snap_want = frozenset(
            int(x) for x in (rho_snapshots_t_after or ())
            if 0 <= int(x) <= n_passes
        )
        if 0 in snap_want:
            snaps[0] = rho_m.detach().clone()

    for t in range(n_passes):
        # Double-angle fields with current ρ
        u = rho_m * torch.cos(2.0 * theta_grid)
        v = rho_m * torch.sin(2.0 * theta_grid)

        u_4d = u.unsqueeze(0).unsqueeze(0)
        v_4d = v.unsqueeze(0).unsqueeze(0)

        # Convolve with all K kernels: (1, K, nH, nW)
        conv_u = F.conv2d(u_4d, kernels, padding=R)
        conv_v = F.conv2d(v_4d, kernels, padding=R)

        # Collinear energy: gather from own θ-bin, project onto cell's orientation
        su_col = torch.gather(conv_u, 1, bin_idx_4d).squeeze(0).squeeze(0)
        sv_col = torch.gather(conv_v, 1, bin_idx_4d).squeeze(0).squeeze(0)
        col_proj = (su_col * c2 + sv_col * s2).clamp_min(0.0)

        # GABA budget: total ρ pooled across all K bins (untuned inhibition).
        rho_4d = rho_m.unsqueeze(0).unsqueeze(0)
        conv_rho = F.conv2d(rho_4d, kernels, padding=R)  # (1, K, nH, nW)

        # Per-bin ρ from the cell's own bin (collinear share of the budget)
        sr_col = torch.gather(conv_rho, 1, bin_idx_4d).squeeze(0).squeeze(0)

        # Total ρ across all bins
        sr_total = conv_rho.sum(dim=1).squeeze(0)  # (nH, nW)

        # Fair-share normalization:
        #   κ = (E_col / sr_col) × (sr_col / sr_total) × K
        #     = E_col × K / sr_total
        # But E_col ≤ sr_col (projection ≤ magnitude), so we normalize
        # the collinear projection against the per-bin average:
        #   κ = E_col / (sr_total / K + ε)
        # When E_col equals the average per-bin energy, κ = 1.
        # When E_col dominates (clean edge), κ > 1 → clamped to 1.
        # When E_col is small (texture), κ < 1 → suppressed.
        per_bin_avg = sr_total / K
        kappa_col = col_proj / (per_bin_avg + eps)
        kappa_col = kappa_col.clamp(0.0, 1.0)
        kappa_col = torch.where(is_border_grid, torch.zeros_like(kappa_col), kappa_col)

        # Modulate ρ
        rho_m = rho_m * kappa_col
        rho_m = torch.where(is_border_grid, torch.zeros_like(rho_m), rho_m)

        if want_snaps and (t + 1) in snap_want:
            snaps[t + 1] = rho_m.detach().clone()

    # Return final-pass raw collinear energy alongside κ and ρ_mod
    if want_snaps:
        return kappa_col, rho_m, col_proj, snaps
    return kappa_col, rho_m, col_proj


# ═══════════════════════════════════════════════════════════════
# Bilinear interpolation: cell grid → pixel coordinates
# ═══════════════════════════════════════════════════════════════

def _interp_cell_to_pixel(
    field_grid: torch.Tensor,
    nH: int, nW: int,
    H: int, W: int,
    S: int, P: int,
) -> torch.Tensor:
    """Bilinear-interpolate a (nH, nW) or (nH, nW, C) cell-grid field
    to (H, W) or (H, W, C) pixel resolution.

    Cell centres sit at pixel coords (j*S + P/2, i*S + P/2).
    """
    device, dtype = field_grid.device, field_grid.dtype
    half_P = P / 2.0

    if field_grid.ndim == 2:
        field_4d = field_grid.unsqueeze(0).unsqueeze(0)  # (1,1,nH,nW)
    else:
        # (nH, nW, C) -> (1, C, nH, nW)
        field_4d = field_grid.permute(2, 0, 1).unsqueeze(0)

    # Pixel coords -> normalised grid coords for grid_sample
    # Cell i is at pixel y = i*S + half_P, cell j at pixel x = j*S + half_P
    # grid_sample wants coords in [-1, 1]
    py = torch.arange(H, device=device, dtype=dtype)
    px = torch.arange(W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(py, px, indexing="ij")

    # Map pixel coords to cell-grid coords (fractional cell index)
    cell_j = (gx - half_P) / max(S, 1)  # fractional column in cell grid
    cell_i = (gy - half_P) / max(S, 1)  # fractional row in cell grid

    # Normalise to [-1, 1] for grid_sample (align_corners=True)
    norm_x = 2.0 * cell_j / max(nW - 1, 1) - 1.0
    norm_y = 2.0 * cell_i / max(nH - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    out = F.grid_sample(
        field_4d, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )

    if field_grid.ndim == 2:
        return out.squeeze(0).squeeze(0)  # (H, W)
    else:
        return out.squeeze(0).permute(1, 2, 0)  # (H, W, C)


# ═══════════════════════════════════════════════════════════════
# Bilinear sampling for stencil taps
# ═══════════════════════════════════════════════════════════════

def _bilinear_sample_2d(
    field: torch.Tensor, row: torch.Tensor, col: torch.Tensor,
) -> torch.Tensor:
    H, W = field.shape
    r = row.reshape(1, 1, -1, 1)
    c = col.reshape(1, 1, -1, 1)
    gx = 2.0 * c / max(W - 1, 1) - 1.0
    gy = 2.0 * r / max(H - 1, 1) - 1.0
    grid = torch.cat([gx, gy], dim=-1)
    out = F.grid_sample(field.reshape(1, 1, H, W), grid,
                        mode="bilinear", padding_mode="border", align_corners=True)
    return out.reshape(row.shape)


def _sample_stencil_5(
    field: torch.Tensor,
    theta: torch.Tensor,
    spacing: torch.Tensor,
    direction: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    """5-tap stencil along tangent or normal. Returns (H, W, 5)."""
    H, W = field.shape
    device, dtype = field.device, field.dtype
    rows = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).expand(H, W)
    cols = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).expand(H, W)
    sp = spacing.to(dtype=dtype, device=device)
    if direction == "tangent":
        dr, dc = torch.cos(theta), torch.sin(theta)
    else:
        dr, dc = -torch.sin(theta), torch.cos(theta)
    taps = []
    for k in range(-2, 3):
        r_off = rows + k * sp * dr
        c_off = cols + k * sp * dc
        taps.append(_bilinear_sample_2d(field, r_off, c_off))
    return torch.stack(taps, dim=-1)


# ═══════════════════════════════════════════════════════════════
# Thinning head (18 → 16 → 1 MLP)
# ═══════════════════════════════════════════════════════════════

class ThinningHead(nn.Module):
    """18→16→1 MLP: σ(W₂ ReLU(W₁ F + b₁) + b₂).

    F = [h2m_lum, h2m_chr, ρ̄^(T), θ̄_cos2, θ̄_sin2, κ̄_col, ρ̄^(T)/(ρ̄^(0)+ε),
         tang5(5), norm5(5), η_lum_map] ∈ R¹⁸.
    """

    def __init__(self, in_dim: int = 18, hidden: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 1, bias=True)
        self._init_priors()

    def _init_priors(self) -> None:
        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.bias.zero_()
            self.fc2.weight.zero_()

            # Feature layout (in_dim=18):
            #   0: h2m_lum
            #   1: h2m_chr
            #   2: ρ̄^(T)  (after collinear recurrence, interpolated)
            #   3: θ̄_cos2
            #   4: θ̄_sin2
            #   5: κ̄_col
            #   6: ρ̄^(T) / (ρ̄^(0) + ε)  collinear preservation ratio
            #   7-11: tang5 (on h2m)
            #   12-16: norm5 (on h2m)
            #   17: η_lum map (pass-2 modulation; optional at inference)

            # Unit 0: Mexican-hat on norm5 (channels 12–16)
            mex_hat = torch.tensor([-0.25, -0.25, 1.0, -0.25, -0.25])
            self.fc1.weight[0, 12:17] = mex_hat

            # Unit 0: flat smoothing on tang5 (channels 7–11)
            self.fc1.weight[0, 7:12] = 0.2

            # Unit 1: ρ̄^(T) — cell-grid edge strength
            self.fc1.weight[1, 2] = 1.0

            # Unit 2: collinear preservation ratio (texture crushed → low)
            self.fc1.weight[2, 6] = 1.0

            # Unit 3: h2m evidence
            self.fc1.weight[3, 0] = 0.5
            self.fc1.weight[3, 1] = 0.5

            # Unit 4: collinear coherence
            self.fc1.weight[4, 5] = 1.0

            # b₂ = 2 → σ(2) ≈ 0.88 near-identity gate at init
            self.fc2.bias.fill_(2.0)
            # Positive weights on ρ̄^(T), preservation ratio, collinear units
            self.fc2.weight[0, 1] = 0.3  # ρ̄^(T) unit
            self.fc2.weight[0, 2] = 0.2  # preservation-ratio unit
            self.fc2.weight[0, 4] = 0.3  # collinear unit

            # η_lum map (index 17): tiny random coupling so ∂gate/∂η is not
            # identically zero when this column was all zeros after the priors.
            if self.in_dim > 17:
                self.fc1.weight[:, 17].normal_(0.0, 0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (N, in_dim). Returns: (N,) gate in (0, 1)."""
        h = F.relu(self.fc1(features))
        return torch.sigmoid(self.fc2(h).squeeze(-1))


# ═══════════════════════════════════════════════════════════════
# ModulationRenderer — harmonic-native
# ═══════════════════════════════════════════════════════════════

class ModulationRenderer(nn.Module):
    """Harmonic-native renderer: h2m · gate(MLP(F_p)).

    No Gaussian scatter-add splat.  Cell-grid fields are bilinearly
    interpolated to pixel coordinates via F.grid_sample.  The edge
    map is the pixel-native h2m, gated by a per-pixel MLP.

    Learned: s_t (1), s_n (1), ThinningHead 18→16→1 (321).
    Fixed: collinear kernels.
    Total: 323 learned scalars.
    """

    def __init__(
        self,
        hidden: int | None = None,
        col_radius: int = _COL_RADIUS,
        col_k_bins: int = _COL_K_BINS,
        col_sigma_d: float | None = _COL_SIGMA_D,
        col_sigma_t: float = _COL_SIGMA_T,
        col_passes: int = _COL_PASSES,
        col_nr_pool_radius: int | None = None,
        **kwargs,
    ):
        super().__init__()
        _ = (hidden, kwargs)
        self.s_t = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.s_n = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.thinning = ThinningHead(in_dim=18, hidden=16)

        self.col_radius = col_radius
        self.col_k_bins = col_k_bins
        self.col_sigma_d = col_sigma_d
        self.col_sigma_t = col_sigma_t
        self.col_passes = col_passes
        self.col_nr_pool_radius = col_nr_pool_radius  # default: same as col_radius

    # Legacy compat — code that reads sigma_perp for diagnostics
    @property
    def sigma_perp(self) -> torch.Tensor:
        return torch.tensor(1.0, dtype=torch.float32)
    @property
    def sigma_par(self) -> torch.Tensor:
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
    def pixel_map(self, profile: torch.Tensor) -> torch.Tensor:
        return torch.ones(profile.shape[0], device=profile.device, dtype=profile.dtype)
    @property
    def n_refine(self) -> int:
        return 0
    @property
    def refine_head(self):
        return None
    @property
    def alpha_refine(self):
        return None


def upgrade_renderer_state_dict(state_dict: dict, prefix: str = "") -> dict:
    """Best-effort upgrade from old splat-based state dicts.

    Also upgrades thinning ``fc1`` weight (16, 17) → (16, 18) by appending a
    small random η-feature column (same scale as ``ThinningHead`` init) so
    older checkpoints load without dropping the thinning head and receive
    nonzero ∂gate/∂η from the start.
    """
    state_dict = dict(state_dict)
    wkey = f"{prefix}thinning.fc1.weight"
    if wkey in state_dict:
        w = state_dict[wkey]
        if tuple(w.shape) == (16, 17):
            # Match fresh-init coupling on the new η column (see ThinningHead).
            gen = torch.Generator(device=w.device)
            gen.manual_seed(0)
            zcol = torch.randn(16, 1, dtype=w.dtype, device=w.device, generator=gen) * 0.02
            state_dict[wkey] = torch.cat([w, zcol], dim=1)

    remove = {
        f"{prefix}_sigma_par_raw",
        f"{prefix}_sigma_pre_raw",
        f"{prefix}_eta_h_raw",
        f"{prefix}_smooth_sigma_raw",
        f"{prefix}_sigma_perp_raw",
        f"{prefix}perp_conv.conv.weight",
        f"{prefix}perp_conv.conv.bias",
        f"{prefix}perp_conv.fc.weight",
        f"{prefix}perp_conv.fc.bias",
    }
    remove_prefixes = (
        f"{prefix}refine_head.",
        f"{prefix}_alpha_refine_raw",
    )
    out = {}
    for k, v in state_dict.items():
        if k in remove:
            continue
        if any(k.startswith(rp) or k == rp for rp in remove_prefixes):
            continue
        # Drop thinning tensors with unknown shapes (fc1 is upgraded above).
        if "thinning." in k and v.shape != _expected_shape(k):
            continue
        out[k] = v
    return out


def _expected_shape(key: str):
    """Expected shapes for the 18→16→1 thinning head."""
    shapes = {
        "thinning.fc1.weight": (16, 18),
        "thinning.fc1.bias": (16,),
        "thinning.fc2.weight": (1, 16),
        "thinning.fc2.bias": (1,),
    }
    for k, s in shapes.items():
        if key.endswith(k):
            return s
    return None


# ═══════════════════════════════════════════════════════════════
# Proj / feature helpers
# ═══════════════════════════════════════════════════════════════

def compute_render_features(
    z2_image: np.ndarray, img: np.ndarray,
    cells: dict, border_mask: np.ndarray,
    eps: float = 1e-9, **kwargs,
) -> dict:
    _ = (z2_image, img, cells, border_mask, eps, kwargs)
    H, W = z2_image.shape
    nH, nW = cells["nH"], cells["nW"]
    return {"H": H, "W": W, "n_cells": nH * nW, "nH": nH, "nW": nW}


def proj_to_device(proj: dict, device: torch.device) -> dict:
    return {"H": proj["H"], "W": proj["W"], "n_cells": proj["n_cells"],
            "nH": proj["nH"], "nW": proj["nW"]}


def _theta_on_branch(theta, branch_pick, n_cells, device):
    if branch_pick is not None:
        b = branch_pick.to(device=device, dtype=torch.long).view(-1)
        idx_n = torch.arange(n_cells, device=device, dtype=torch.long)
        return theta[idx_n, b]
    return theta[:, 0]


# ═══════════════════════════════════════════════════════════════
# render_boundary_map_torch — harmonic-native
# ═══════════════════════════════════════════════════════════════

def render_boundary_map_torch(
    rho_cell: torch.Tensor,
    proj_dev: dict,
    renderer: ModulationRenderer,
    cells_flat: dict,
    Hp: int, Wp: int,
    l0_pix: dict[str, torch.Tensor] | None = None,
    eps: float = 1e-6,
    training: bool = False,
    branch_pick: torch.Tensor | None = None,
    content_h: int | None = None,
    content_w: int | None = None,
    return_dominant_theta: bool = False,
    return_rho_cell_grids: bool = False,
    return_rho_collinear_iter_pix: bool = False,
    rho_collinear_iter_max_snapshots: int = 5,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, torch.Tensor]]]
):

    _ = training
    want_cell_grids = bool(return_rho_cell_grids or return_rho_collinear_iter_pix)
    H, W = proj_dev["H"], proj_dev["W"]
    device, dtype = rho_cell.device, rho_cell.dtype

    nH, nW = int(cells_flat["nH"]), int(cells_flat["nW"])
    n_cells = int(proj_dev["n_cells"])
    theta_all = cells_flat["theta"].to(device=device, dtype=dtype)
    theta_c = _theta_on_branch(theta_all, branch_pick, n_cells, device)
    S = int(cells_flat.get("S", max(1, W // max(nW, 1))))
    P = int(cells_flat.get("P", S + (S - 1)))  # reconstruct P from context

    # ── Step 1: Cell-grid θ combing ──────────────────────────
    rho_grid = rho_cell.reshape(nH, nW).to(dtype=dtype)
    ib_grid = cells_flat["is_border"].to(device=device).reshape(nH, nW).bool()
    theta_combed = _smooth_theta_rho_double_angle(
        theta_c.reshape(nH, nW), rho_grid, ib_grid, eps=eps,
    )

    is_border = cells_flat["is_border"].to(device=device).reshape(-1).bool()
    rho_flat = rho_cell.reshape(-1).to(dtype=dtype)

    active = (rho_flat > 0) & (~is_border)
    if not active.any().item():
        z = (rho_cell.sum() * 0.0).to(dtype=dtype, device=device)
        out = z.expand(H, W)[:Hp, :Wp]
        th_z = torch.zeros_like(out)
        zs_grid = torch.zeros(nH, nW, device=device, dtype=dtype)
        rho_iter_pix_list: list[tuple[int, torch.Tensor]] = []
        if return_dominant_theta and want_cell_grids:
            if return_rho_collinear_iter_pix:
                return out, th_z, zs_grid, zs_grid, rho_iter_pix_list
            return out, th_z, zs_grid, zs_grid
        if return_dominant_theta:
            return out, th_z
        if return_rho_cell_grids:
            return out, zs_grid, zs_grid
        return out

    # ── Step 2: Recurrent collinear facilitation + cross suppression ─
    rho_cell_snaps: dict[int, torch.Tensor] | None = None
    if return_rho_collinear_iter_pix:
        sched = collinear_rho_snapshot_schedule(
            int(renderer.col_passes),
            max_timestamps=int(rho_collinear_iter_max_snapshots),
        )
        kappa_col_grid, rho_mod_grid, e_col_grid, rho_cell_snaps = compute_collinear_coherence(
            theta_combed.detach(),
            rho_grid.detach(),
            ib_grid,
            R=renderer.col_radius,
            K=renderer.col_k_bins,
            sigma_d=renderer.col_sigma_d,
            sigma_t=renderer.col_sigma_t,
            n_passes=renderer.col_passes,
            nr_pool_radius=renderer.col_nr_pool_radius,
            eps=eps,
            rho_snapshots_t_after=frozenset(sched),
        )
    else:
        kappa_col_grid, rho_mod_grid, e_col_grid = compute_collinear_coherence(
            theta_combed.detach(),
            rho_grid.detach(),
            ib_grid,
            R=renderer.col_radius,
            K=renderer.col_k_bins,
            sigma_d=renderer.col_sigma_d,
            sigma_t=renderer.col_sigma_t,
            n_passes=renderer.col_passes,
            nr_pool_radius=renderer.col_nr_pool_radius,
            eps=eps,
        )

    # Cell grids for diagnostics: ρ before vs after recurrent collinear passes
    rho_seed_cell = torch.where(ib_grid, torch.zeros_like(rho_grid), rho_grid)

    ch_pre = Hp if content_h is None else content_h
    cw_pre = Wp if content_w is None else content_w
    ch_pre, cw_pre = min(ch_pre, H), min(cw_pre, W)

    rho_iter_pix_list: list[tuple[int, torch.Tensor]] = []
    if rho_cell_snaps is not None:
        for t_after in sorted(rho_cell_snaps.keys()):
            rho_snap = rho_cell_snaps[t_after]
            rho_grid_m_snap = torch.where(ib_grid, torch.zeros_like(rho_snap), rho_snap)
            rho_pix_s = _interp_cell_to_pixel(rho_grid_m_snap, nH, nW, H, W, S, P)
            if content_h is not None and content_w is not None:
                crop_s = torch.ones_like(rho_pix_s)
                if ch_pre < H:
                    crop_s[ch_pre:, :] = 0.0
                if cw_pre < W:
                    crop_s[:, cw_pre:] = 0.0
                rho_pix_s = rho_pix_s * crop_s
            rho_iter_pix_list.append((t_after, rho_pix_s[:Hp, :Wp].contiguous()))

    # ── Step 3: Interpolate cell-grid fields to pixel res ────
    # Use modulated ρ (after collinear facilitation + cross suppression)
    # theta_combed is detached — orientation is a geometric feature, not
    # a learned signal.  Gradient flows through ρ̄ and the thinning MLP.
    theta_combed_det = theta_combed.detach()
    rho_grid_m = torch.where(ib_grid, torch.zeros_like(rho_mod_grid), rho_mod_grid)
    theta_cos2 = torch.where(ib_grid, torch.zeros_like(theta_combed_det),
                              rho_grid_m * torch.cos(2.0 * theta_combed_det))
    theta_sin2 = torch.where(ib_grid, torch.zeros_like(theta_combed_det),
                              rho_grid_m * torch.sin(2.0 * theta_combed_det))

    # Stack fields for a single grid_sample call: (nH, nW, C)
    # channels: [ρ, cos2θ·ρ, sin2θ·ρ, κ_col]
    cell_stack = torch.stack([
        rho_grid_m,
        theta_cos2,
        theta_sin2,
        kappa_col_grid,
    ], dim=-1)  # (nH, nW, 4)

    pix_stack = _interp_cell_to_pixel(cell_stack, nH, nW, H, W, S, P)
    # (H, W, 4)

    rho_pix = pix_stack[..., 0]
    # Recover θ from interpolated double-angle (handles π-wraparound)
    cos2_pix = pix_stack[..., 1]
    sin2_pix = pix_stack[..., 2]
    # Normalise by interpolated ρ to get unit double-angle
    rho_pix_safe = rho_pix.clamp_min(eps)
    cos2_norm = cos2_pix / rho_pix_safe
    sin2_norm = sin2_pix / rho_pix_safe
    theta_pix = 0.5 * torch.atan2(sin2_norm, cos2_norm)
    theta_pix = theta_pix.detach()  # geometric feature; grad flows through s_t, s_n, h2m

    kappa_col_pix = pix_stack[..., 3]

    # Pre–collinear-recurrence ρ at pixels (same geometry as ρ̄^(T)); ratio
    # encodes how much collinear recurrence preserved each site (texture → low).
    rho0_pix = _interp_cell_to_pixel(rho_seed_cell, nH, nW, H, W, S, P)
    rho_pres_ratio = rho_pix / (rho0_pix + eps)
    rho_pres_ratio = rho_pres_ratio.clamp(max=10.0)

    # ── Step 4: Pixel-native h2m fields ──────────────────────
    if l0_pix is not None and "h2m_lum" in l0_pix:
        h2m_lum = l0_pix["h2m_lum"].to(device=device, dtype=dtype)
        if h2m_lum.shape != (H, W):
            h2m_lum = h2m_lum[:H, :W]
    else:
        h2m_lum = torch.zeros(H, W, device=device, dtype=dtype)

    if l0_pix is not None and "h2m_chr" in l0_pix:
        h2m_chr = l0_pix["h2m_chr"].to(device=device, dtype=dtype)
        if h2m_chr.shape != (H, W):
            h2m_chr = h2m_chr[:H, :W]
    else:
        h2m_chr = torch.zeros(H, W, device=device, dtype=dtype)

    # Combined h2m as the base edge signal (pixel-native, smooth)
    h2m_combined = h2m_lum + h2m_chr

    if l0_pix is not None and "eta_mod_map" in l0_pix:
        eta_mod_pix = l0_pix["eta_mod_map"].to(device=device, dtype=dtype)
        if eta_mod_pix.shape != (H, W):
            eta_mod_pix = eta_mod_pix[:H, :W]
    else:
        eta_mod_pix = torch.zeros(H, W, device=device, dtype=dtype)

    # ── Step 5: Stencils on h2m (pixel-native, no staircase) ─
    tang5 = _sample_stencil_5(h2m_combined, theta_pix, renderer.s_t, "tangent", eps)
    norm5 = _sample_stencil_5(h2m_combined, theta_pix, renderer.s_n, "normal", eps)

    # ── Step 6: Feature vector F_p ∈ R¹⁸ ────────────────────
    fdim = int(renderer.thinning.in_dim)
    features = torch.cat([
        h2m_lum.unsqueeze(-1),          # 0
        h2m_chr.unsqueeze(-1),          # 1
        rho_pix.unsqueeze(-1),          # 2  ρ̄^(T) interpolated
        cos2_norm.unsqueeze(-1),        # 3  (interpolated double-angle)
        sin2_norm.unsqueeze(-1),        # 4
        kappa_col_pix.unsqueeze(-1),    # 5  (interpolated)
        rho_pres_ratio.unsqueeze(-1),   # 6  ρ̄^(T)/(ρ̄^(0)+ε)
        tang5,                          # 7-11
        norm5,                          # 12-16
        eta_mod_pix.unsqueeze(-1),      # 17  η_lum map (grad to η_mod when set)
    ], dim=-1)  # (H, W, 18)

    # ── Step 7: Thinning head → B̂(p) = h2m(p) · gate(p) ────
    if features.shape[-1] != fdim:
        raise RuntimeError(
            f"feature dim {features.shape[-1]} != thinning.in_dim {fdim}"
        )
    feat_flat = features.reshape(H * W, fdim)
    gate = renderer.thinning(feat_flat).reshape(H, W)
    bmap = h2m_combined * gate

    # ── Crop ─────────────────────────────────────────────────
    ch = Hp if content_h is None else content_h
    cw = Wp if content_w is None else content_w
    ch, cw = min(ch, H), min(cw, W)
    if content_h is not None and content_w is not None:
        crop = torch.ones_like(bmap)
        if ch < H: crop[ch:, :] = 0.0
        if cw < W: crop[:, cw:] = 0.0
        bmap = bmap * crop
        theta_pix = theta_pix * crop

    out = bmap[:Hp, :Wp]
    th_out = theta_pix[:Hp, :Wp]
    if return_dominant_theta and want_cell_grids:
        if return_rho_collinear_iter_pix:
            return out, th_out, rho_seed_cell, rho_grid_m, rho_iter_pix_list
        return out, th_out, rho_seed_cell, rho_grid_m
    if return_dominant_theta:
        return out, th_out
    if want_cell_grids:
        if return_rho_collinear_iter_pix:
            return out, rho_seed_cell, rho_grid_m, rho_iter_pix_list
        return out, rho_seed_cell, rho_grid_m
    return out


# ═══════════════════════════════════════════════════════════════
# NumPy wrapper
# ═══════════════════════════════════════════════════════════════

def render_boundary_map(
    rho_cell: np.ndarray, proj: dict,
    renderer: ModulationRenderer, cells_flat: dict,
    l0_pix: dict[str, np.ndarray] | None = None,
    device: torch.device = torch.device("cpu"),
    eps: float = 1e-6,
    branch_pick: np.ndarray | None = None,
    content_h: int | None = None, content_w: int | None = None,
) -> np.ndarray:
    proj_dev = proj_to_device(proj, device)
    rho_t = torch.from_numpy(np.asarray(rho_cell, dtype=np.float32)).to(device)
    cf_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for k, v in cells_flat.items()}
    bp = None
    if branch_pick is not None:
        bp = torch.from_numpy(np.asarray(branch_pick, dtype=np.int64).ravel()).to(device)
    l0_dev = None
    if l0_pix is not None:
        l0_dev = {}
        for k, v in l0_pix.items():
            if isinstance(v, np.ndarray):
                l0_dev[k] = torch.from_numpy(v.astype(np.float32)).to(device)
            elif isinstance(v, torch.Tensor):
                l0_dev[k] = v.to(device)
            else:
                l0_dev[k] = v
    with torch.no_grad():
        bmap_t = render_boundary_map_torch(
            rho_t, proj_dev, renderer, cf_dev, proj["H"], proj["W"], l0_dev,
            eps=eps, training=False, branch_pick=bp,
            content_h=content_h, content_w=content_w,
        )
    return bmap_t.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# NMS  (unchanged)
# ═══════════════════════════════════════════════════════════════

def _nms_unit_normal_from_theta(theta, eps=1e-8):
    t = np.asarray(theta, dtype=np.float64)
    return np.cos(t).astype(np.float32), (-np.sin(t)).astype(np.float32)

def _nms_unit_normal_from_gradient(mag, eps=1e-8):
    m = np.asarray(mag, dtype=np.float64)
    gx, gy = ndimage.sobel(m, axis=1), ndimage.sobel(m, axis=0)
    norm = np.sqrt(gx*gx + gy*gy) + eps
    return (gx/norm).astype(np.float32), (gy/norm).astype(np.float32)

def _nms_bilinear_sample(mag, row_off, col_off):
    coords = np.stack([row_off.astype(np.float64), col_off.astype(np.float64)])
    return ndimage.map_coordinates(mag.astype(np.float64), coords, order=1, mode="nearest").astype(np.float32)

def ridge_nms(mag, *, theta=None, grad_norm_floor=1e-7):
    m = np.asarray(mag, dtype=np.float32)
    if m.ndim != 2: raise ValueError(f"ridge_nms expects 2D, got {m.shape}")
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m_work = m.copy()
    if theta is not None:
        nx, ny = _nms_unit_normal_from_theta(np.asarray(theta, dtype=np.float32))
        weak = np.zeros((H, W), dtype=bool)
    else:
        gx, gy = ndimage.sobel(m_work.astype(np.float64), axis=1), ndimage.sobel(m_work.astype(np.float64), axis=0)
        gnorm = np.sqrt(gx*gx + gy*gy).astype(np.float32)
        nx, ny = _nms_unit_normal_from_gradient(m_work)
        weak = gnorm < grad_norm_floor
    ahead = _nms_bilinear_sample(m_work, yy+ny, xx+nx)
    behind = _nms_bilinear_sample(m_work, yy-ny, xx-nx)
    keep = ((m_work >= ahead) & (m_work >= behind)) | weak
    return np.where(keep, m_work, 0.0).astype(np.float32)

def ridge_nms_binary(mag, threshold, *, theta=None, grad_norm_floor=1e-7):
    return (ridge_nms(mag, theta=theta, grad_norm_floor=grad_norm_floor) >= threshold).astype(np.uint8) * 255

def cell_rho_to_2branch(rho_cell, branch):
    out = np.zeros((*rho_cell.shape, 2), dtype=rho_cell.dtype)
    ii, jj = np.indices(rho_cell.shape)
    out[ii, jj, branch.astype(np.int64)] = rho_cell
    return out

HarmonicThinRenderer = ModulationRenderer
StampRenderer = ModulationRenderer
AnisoDiffusionRenderer = ModulationRenderer