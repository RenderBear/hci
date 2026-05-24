r"""Shared pipeline hyperparameters, module inits, and script defaults.

L1 builds cos² hypercolumns → **seed NR** (learned ``η_z=\\mathrm{softplus}(\\tilde\\eta_z)``) → multiplicative **tanh gate**
passes ``ρ ← ρ^{\\mathrm{seed}} \\odot (1+α\\tanh(u/τ))`` with ``u = β_c S - β_x I`` (no in-loop
divisive NR). Learned ``η_z``, ``τ``, ``α`` (``HypercolumnSeed``). Inhibition uses
**isotropic surround** per bin ``k``: LOO mean ``(1/\\max(K-1,1))\\sum_{j\\neq k}ρ_j`` then convolve with the same radial kernel as ``G_k`` (no tangential weight).
``κ`` is cosine alignment ``(ρ·S)/(‖ρ‖‖S‖)`` for diagnostics / readout.
The renderer interpolates ρ, θ, κ and applies ``h2m·ρ̄·gate`` (14-D readout, no η_mod).

Training disk cache: ``TRAIN.CACHE_VERSION`` invalidates stored L0 tensors used
for **live** L1 each step (``h2m``, ``theta_h``, masks, etc.); it does not cache
``cells_flat``.  ``L0.L0_DIST_CACHE_VERSION`` gates reuse of pre-NR ``d_lum``/``d_chr``
across bumps when geometry is unchanged (see ``train.precompute_image``).
"""

from __future__ import annotations

from types import SimpleNamespace

# ── L0: split-channel harmonic projection (lum / chroma Naka–Rushton) ─────────
L0 = SimpleNamespace(
    OFFSETS=[
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ],
    ETA_LUM=0.05,
    ETA_CHR=0.05,
    GAMMA=1.0,
    # Bump when ``_compute_d_lum_chroma`` / ``L0.OFFSETS`` semantics change.
    # Train cache can reuse stored ``d_lum``/``d_chr`` across ``TRAIN.CACHE_VERSION`` bumps.
    L0_DIST_CACHE_VERSION=1,
)

# ── L1: hypercolumn cell grid + collinear recurrence (depthwise GABA) ───────
L1 = SimpleNamespace(
    PATCH_SIZE=5,
    PATCH_OVERLAP=3,
    BORDER_PATCH_MAX_FRAC=0.2,
    EPS=1e-15,
    COL_RADIUS=5,
    COL_K_BINS=24,
    COL_SIGMA_D=None,       # default: R/2 inside L1
    COL_SIGMA_T=1.0,
    COL_PASSES=5,
    # Mean-pool radius for GABA η MLP inputs (kernel 2*r+1 = 21 → r=10).
    GABA_ETA_POOL_RADIUS=10,
)

# ── SEED: tile geometry + η_z + τ, α gate + β_coll/β_cross (HypercolumnSeed; η_z learned) ───
SEED = SimpleNamespace(
    R_POOL=10,
    STRIDE=7,
    EPS=1e-9,
    ETA_Z=5.0,
    TAU_INIT=1.0,
    GAIN_ALPHA_INIT=0.2,
    BETA_COLL_INIT=0.5,
    BETA_CROSS_INIT=0.3,
)

# ── Render: θ combing + bilinear interp + minimal gate (κ_col, E_col from L1) ─
RENDER = SimpleNamespace(
    CELL_HIDDEN=16,
    PIXEL_HIDDEN=6,
    THETA_SMOOTH_PASSES=4,
)

# ── Training ─────────────────────────────────────────────────────────────────
TRAIN = SimpleNamespace(
    LR=5e-2,
    EPOCHS=15,
    BATCH_SIZE=4,
    GRAD_CLIP=1.0,
    NUM_WORKERS=2,
    LAM_DICE=0.0,
    LAM_BCE=1.0,
    CACHE_VERSION=24,
)

# ── Inference ────────────────────────────────────────────────────────────────
INFER = SimpleNamespace(
    DEFAULT_THRESHOLD=0.5,
    SHAPE_THETA_BINS=12,
)

# ── test.py evaluation ───────────────────────────────────────────────────────
TEST = SimpleNamespace(
    THRESHOLD_COUNT=99,
)

# ── eval/eval.py BSR-style metrics ───────────────────────────────────────────
EVAL = SimpleNamespace(
    THRESHOLD_COUNT=99,
    MAX_DIST_FRAC=0.0075,
    MATCH="fast",
)

# ── Diagnostics / matplotlib styling ────────────────────────────────────────
VIZ = SimpleNamespace(
    BG="#0e0e0e",
    PANEL_BG="#111111",
    FG="#dddddd",
    ACCENT="#888888",
    EPS=1e-15,
    COMPASS={
        (-1, -1): "NW",
        (-1, 0): "N",
        (-1, 1): "NE",
        (0, -1): "W",
        (0, 1): "E",
        (1, -1): "SW",
        (1, 0): "S",
        (1, 1): "SE",
    },
    GRID_POS={
        (-1, -1): (0, 0),
        (-1, 0): (0, 1),
        (-1, 1): (0, 2),
        (0, -1): (1, 0),
        (0, 1): (1, 2),
        (1, -1): (2, 0),
        (1, 0): (2, 1),
        (1, 1): (2, 2),
    },
)
