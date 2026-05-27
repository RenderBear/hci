r"""Shared pipeline hyperparameters, module inits, and script defaults."""

from __future__ import annotations

import math
from types import SimpleNamespace

# ── L0: split-channel harmonic projection (lum / chroma Naka–Rushton) ─────────
L0 = SimpleNamespace(
    OFFSETS=[
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ],
    # Sole L0 sensitivity knobs: fixed scalars in this file only (never nn.Parameter / never trained).
    ETA_LUM=0.05,
    ETA_CHR=0.05,
    ETA0=0.05,  # legacy alias for scripts still printing “η₀”
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

# ── L2: cell-grid conv dynamics (ρ refinement) ───────────────────────────────
L2 = SimpleNamespace(
    R_FAC_POOL=5,   # collinear (facilitation) pool radius
    R_SUP_POOL=10,   # iso / cross (suppression) pool radius
    K=24,
    T_REFINE=5,
    EPS=1e-9,
    ETA_Z_INIT=2.0,  # seed denom: ρ_seed = λ₁/(λ₁+λ₂+η_z)
    LOGIT_CLAMP=1e-4,
    # drive / inhibition (softplus-positive, learned; constant over t)
    B_SEED_INIT=0.5,
    B_COLL_INIT=1.0,
    B_ISO_INIT=0.3,
    B_CROSS_INIT=0.3,
    ETA_COLL_INIT=0.3,   # NR half-sat on ρ_coll (raw ~0.5)
    ETA_ISO_INIT=0.3,    # NR half-sat on lateral ρ² numerator (stable, O(1–20))
    ETA_CROSS_INIT=0.3,  # NR half-sat on cross-bin ρ² sum (no count norm)
    ETA_P_INIT=0.1,  # NR floor η_p² in ρ update denominator
)

# ── Render: §2.5 anisotropic splat + ρ̄ gate G + perp conv (see striate/renderer.py) ─
RENDER = SimpleNamespace(
    CELL_HIDDEN=16,  # legacy StriateE2E arg (unused)
    PIXEL_HIDDEN=6,  # legacy StriateE2E arg (unused)
    SIGMA_PAR_INIT=8.0,
    SIGMA_PERP_INIT=1.0,
    SIGMA_PAR_MAX=32.0,
    SIGMA_PERP_MAX=8.0,
    GATE_RADIUS_SIGMAS=3.0,  # splat kernel radius in max(σ_∥, σ_⊥) units
    GATE_ALPHA_INIT=1.0,  # softplus → α_g in G = σ(α_g(ρ̄ − τ_g))
    GATE_TAU_INIT=0.0,  # τ_g (learned)
    SPLAT_RADIUS_SIGMAS=3.0,  # alias / legacy name
    THETA_SMOOTH_PASSES=4,
    SIGMA_PRE_INIT=1.5,  # tangent z₂ pre-smooth width (softplus, px)
    SIGMA_PRE_MAX=12.0,
    SMOOTH_SIGMA_INIT=2.0,  # σ_s along-contour (softplus → ~2 px)
    SMOOTH_RADIUS=3,
    PRE_SMOOTH_RADIUS=3,
)

# ── Training ───────────────────────────────────────────────────────────────
TRAIN = SimpleNamespace(
    LR=5e-2,
    EPOCHS=15,
    BATCH_SIZE=4,
    GRAD_CLIP=1.0,
    NUM_WORKERS=2,
    LAM_DICE=1.0,
    LAM_BCE=0.0,
    CACHE_VERSION=42,
    L2_SNAPSHOT_MAX=5,
)

# ── Inference ────────────────────────────────────────────────────────────────
INFER = SimpleNamespace(
    DEFAULT_THRESHOLD=0.5,
    L2_ITERS=None,
    SHAPE_THETA_BINS=12,
)

# ── test.py evaluation ───────────────────────────────────────────────────────
TEST = SimpleNamespace(
    BISTABLE_THRESHOLD=0.5,
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
