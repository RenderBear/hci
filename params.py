r"""Shared pipeline hyperparameters, module inits, and script defaults.

L1 builds cos² hypercolumns → min-subtract + η_z NR (pre-GABA) → GABA recurrence
(κ from ρ–S cosine similarity by default; see ``L1.COL_KAPPA_NORM``).
The renderer interpolates ρ, θ, κ and applies ``h2m·ρ̄·gate``.

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
    ETA0=0.05,
    GAMMA=1.0,
    # Bump when ``_compute_d_lum_chroma`` / ``L0.OFFSETS`` semantics change.
    # Train cache can reuse stored ``d_lum``/``d_chr`` across ``TRAIN.CACHE_VERSION`` bumps.
    L0_DIST_CACHE_VERSION=1,
    # Regional η MLP: mean-pool cell stats over (2R_η+1)², R_η in cell units.
    ETA_POOL_RADIUS_CELLS=10,
    ETA_MLP_HIDDEN=8,
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
    # κ after depthwise collinear conv S_k:
    # "cosine" = scalar cos(ρ, S) per cell, same κ all bins (texture vs edge);
    # "max" = per-bin S_k/(max_j S_j+ε); "fair_share" = S_k/(E_total/K+ε).
    COL_KAPPA_NORM="cosine",
)

# ── SEED: tile geometry + learned η_z (HypercolumnSeed); no separate dynamics ─
SEED = SimpleNamespace(
    R_POOL=10,
    STRIDE=7,
    EPS=1e-9,
    # Softplus(·) → η_z for pre-GABA NR; keep O(1) vs typical cos² patch ρ_k (not ≫1).
    ETA_Z_INIT=2.0,
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
    BATCH_SIZE=8,
    GRAD_CLIP=1.0,
    NUM_WORKERS=2,
    LAM_DICE=0.0,
    LAM_BCE=1.0,
    CACHE_VERSION=16,
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
