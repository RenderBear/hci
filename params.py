r"""Shared pipeline hyperparameters, module inits, and script defaults."""

from __future__ import annotations

from types import SimpleNamespace

# ── L0: split-channel harmonic projection (lum / chroma Naka–Rushton) ─────────
L0 = SimpleNamespace(
    OFFSETS=[
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ],
    ETA_LUM=0.01,
    ETA_CHR=0.01,
    GAMMA=1.0,
    EPS=1e-6,
    NOTCH_ENABLED=True,
    NOTCH_HALF_WIDTH=4,
    NOTCH_OMEGA_N_INIT=1.0 / 8,
    NOTCH_SIGMA_N_INIT=1.0 / 32,
    NOTCH_D_INIT=0.8,
)

# ── L1: patch geometry + z₂ orientation bins (von Mises on θ_p) ───────────
L1 = SimpleNamespace(
    PATCH_SIZE=5,
    PATCH_OVERLAP=3,
    BORDER_PATCH_MAX_FRAC=0.2,
    EPS=1e-15,
    NUM_ORIENT_BINS=8,
    KAPPA_VM_INIT=2.0,
)

# ── Seed: η_z NR on |Z|, then collinear + surround + divisive readout (η_readout) ─
SEED = SimpleNamespace(
    EPS=1e-9,
    ETA_Z_INIT=10.0,
    BETA_SEED_INIT=0.5,
    BETA_COLL_INIT=0.5,
    KAPPA_THETA_INIT=2.5,
    ETA_READOUT_INIT=0.30,
    LAMBDA_INIT=0.5,
    SIGMA_F_INIT=0.5,
    FACIL_RADIUS=5,
    FACIL_MODE="collinear",
    CROSS_SURROUND_RADIUS=10,
    SURROUND_SIGMA=2.0,
    SURROUND_MODE="broadside",
    SIGMA_S_INIT=2.0,
    RHO_STE_TAU=0.1,
)

# ── Render: back-projection with learned 1D kernels ───────────────────────────
RENDER = SimpleNamespace(
    DEPOSIT_HALF_WIDTH_STRIDES=2.0,
    DEPOSIT_HALF_WIDTH_MIN=4,
    DEPOSIT_HALF_WIDTH_MAX=24,

    SIGMA_PERP_INIT=0.6,
    SIGMA_PAR_INIT=2.0,
    RAMP_CUTOFF_INIT=0.5,

    KAPPA_MAX_INIT=0.1,
    EXT_MAX_INIT=1.0,
    DELTA_N_MAX_INIT=1.0,
    ALPHA_RANGE_INIT=0.5,

    BIN_GATE_TEMP_INIT=0.08,

    CORR_HIDDEN=12,

    DEPOSIT_ENVELOPE_SIGMA=0.0,
    SIGMA_PAR_MAX=32.0,
    SIGMA_PERP_MAX=8.0,
    SPLAT_RADIUS_SIGMAS=3.0,
    THETA_SMOOTH_PASSES=0,
    THINNING_IN=20,
    THINNING_HIDDEN=12,
)

# ── Training ───────────────────────────────────────────────────────────────
TRAIN = SimpleNamespace(
    LR=5e-2,
    EPOCHS=15,
    BATCH_SIZE=4,
    GRAD_CLIP=1.0,
    NUM_WORKERS=2,
    GT_MIN_AGREEMENT=0.0,
    L0_CACHE_VERSION=3,
)

# ── Evaluation & inference defaults ───────────────────────────────────────────
EVAL = SimpleNamespace(
    DEFAULT_THRESHOLD=0.5,
    THRESHOLD_COUNT=99,
    MAX_DIST_FRAC=0.0075,
)

INFER = SimpleNamespace(
    SHAPE_THETA_BINS=12,
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