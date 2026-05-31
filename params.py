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
    ETA_LUM=0.01,
    ETA_CHR=0.01,
    ETA0=0.05,  # legacy alias for scripts still printing “η₀”
    GAMMA=1.0,
    # ε inside chroma/harmonic norms — avoids ∂√0 = ∞ when backpropping through L0.
    EPS=1e-6,
    # Learned RGB metric W (M = WᵀW); trained end-to-end, init = orthonormal lum/chr.
    LEARNED_METRIC=True,
)

# ── L1: patch geometry + z₂ moment pooling ─────────────────────────────────
L1 = SimpleNamespace(
    PATCH_SIZE=5,
    PATCH_OVERLAP=3,
    BORDER_PATCH_MAX_FRAC=0.2,
    EPS=1e-15,
)

# ── Seed: η_z NR on |Z|, then collinear + surround + divisive readout ─────────
SEED = SimpleNamespace(
    EPS=1e-9,
    ETA_Z_INIT=0.30,
    BETA_SEED_INIT=0.5,
    BETA_COLL_INIT=0.5,
    KAPPA_THETA_INIT=2.5,
    ETA_INIT=0.30,
    LAMBDA_INIT=0.5,
    SIGMA_F_INIT=1.3,
    FACIL_RADIUS=2,
    FACIL_MODE="collinear",
    SURROUND_RADIUS=5,
    SURROUND_SIGMA=2.0,
    SURROUND_MODE="broadside",
    SURROUND_RADIUS_DIAG=5,
    SURROUND_SIGMA_DIAG=2.0,
)

# ── Render: §2.5 anisotropic splat + ρ̄ gate G + perp conv (see striate/renderer.py) ─
RENDER = SimpleNamespace(
    CELL_HIDDEN=16,  # legacy StriateE2E arg (unused)
    PIXEL_HIDDEN=6,  # legacy StriateE2E arg (unused)
    SIGMA_PAR_INIT=2.0,  # along-edge width; init ≈ L1 stride S (= P − overlap)
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
    # Thinning head: F_p ∈ R^20 = [ρ̄, coh, tang9, norm9]; MLP 20→12→1
    THINNING_IN=20,
    THINNING_HIDDEN=12,
    STENCIL_TAPS=9,  # j ∈ {-4,…,4} along tangent and normal
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
    # Bump when L0 / pad / ``l0_pix`` / GT schema changes — not L1 binning or seed.
    L0_CACHE_VERSION=2,
    # Legacy full-cache tag (pre-rho split); kept so old ``.pt`` files are rejected cleanly.
    CACHE_VERSION=44,
)

# ── Inference ────────────────────────────────────────────────────────────────
INFER = SimpleNamespace(
    DEFAULT_THRESHOLD=0.5,
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
