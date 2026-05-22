r"""Shared pipeline hyperparameters, module inits, and script defaults.

L1 eigendecomposition → ``ρ_seed`` (NR pool on ``λ₁/(z₀+η_z)``) → renderer splat.
No L2 tile dynamics.
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
)

# ── L1: per-patch eigendecomposition → cell grid ────────────────────────────
L1 = SimpleNamespace(
    PATCH_SIZE=5,
    PATCH_OVERLAP=3,
    BORDER_PATCH_MAX_FRAC=0.2,
    EPS=1e-15,
    N_BRANCHES=2,
)

# ── SEED: NR-normalized ρ from L1 (tile coverage mask; no dynamics) ─────────
SEED = SimpleNamespace(
    R_POOL=10,
    STRIDE=7,
    EPS=1e-9,
    ETA_Z_INIT=5.0,
    ETA_RHO_INIT=0.1,
)

# ── Render: Gaussian-line splat + collinear coherence + thinning ─────────────
RENDER = SimpleNamespace(
    CELL_HIDDEN=16,
    PIXEL_HIDDEN=6,
    SIGMA_PAR_INIT=8.0,
    SIGMA_PERP_INIT=1.0,
    SIGMA_PAR_MAX=32.0,
    SIGMA_PERP_MAX=8.0,
    GATE_RADIUS_SIGMAS=3.0,
    GATE_ALPHA_INIT=1.0,
    GATE_TAU_INIT=0.0,
    SPLAT_RADIUS_SIGMAS=3.0,
    THETA_SMOOTH_PASSES=4,
    SIGMA_PRE_INIT=1.5,
    SIGMA_PRE_MAX=12.0,
    SMOOTH_SIGMA_INIT=2.0,
    SMOOTH_RADIUS=3,
    PRE_SMOOTH_RADIUS=3,
    COL_RADIUS=12,
    COL_K_BINS=24,
    COL_SIGMA_D=None,       # default: R/2
    COL_SIGMA_T=1.0,
)

# ── Training ─────────────────────────────────────────────────────────────────
TRAIN = SimpleNamespace(
    LR=5e-2,
    EPOCHS=15,
    BATCH_SIZE=4,
    GRAD_CLIP=1.0,
    NUM_WORKERS=2,
    LAM_DICE=1.0,
    LAM_BCE=0.0,
    CACHE_VERSION=1,
)

# ── Inference ────────────────────────────────────────────────────────────────
INFER = SimpleNamespace(
    # infer.py: default edge τ is Otsu on the soft map; pass -t to use a fixed value.
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
