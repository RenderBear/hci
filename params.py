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
    # Sole L0 sensitivity knobs: fixed scalars in this file only (never nn.Parameter / never trained).
    ETA_LUM=0.01,
    ETA_CHR=0.01,
    GAMMA=1.0,
    # ε inside chroma/harmonic norms — avoids ∂√0 = ∞ when backpropping through L0.
    EPS=1e-6,
    # Learned RGB metric W (M = WᵀW); trained end-to-end, init = orthonormal lum/chr.
    LEARNED_METRIC=True,
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
    SIGMA_F_INIT=1.3,
    FACIL_RADIUS=2,
    FACIL_MODE="collinear",
    SURROUND_RADIUS=5,
    SURROUND_SIGMA=2.0,
    SURROUND_MODE="broadside",
    # Learned surround Gaussian scale for orientation-bin surround S^(k) (init ≈ SURROUND_SIGMA).
    SIGMA_S_INIT=2.0,
    # Straight-through ρ max: softmax temperature (forward = hard max).
    RHO_STE_TAU=0.1,
)

# ── Render: harmonic-inversion deposit + per-bin curvature correction ────────
# Pipeline (see hci/renderer.py):
#   per (cell c, bin k) — F^(k)_c ∈ ℝ⁴ → MLP_{4→8→2} → bounded (κ, e_s)
#   stroke f^(k)_c(p) = exp(−(n − ½ κ s̃²)² / 2σ⊥² − s̃² / 2σ∥²)   s̃ = s − e_s
#   gate g^(k)_c = σ(α_g (ρ^(k) − τ · max_j ρ^(j)))
#   noisy-OR aggregation across ALL (c, k) pairs.
RENDER = SimpleNamespace(
    # ── Deposit footprint (pixels): half_w = clamp(⌈STRIDES · S⌉, [MIN, MAX]) ──
    DEPOSIT_HALF_WIDTH_STRIDES=2.0,
    DEPOSIT_HALF_WIDTH_MIN=4,
    DEPOSIT_HALF_WIDTH_MAX=24,

    # ── Gaussian stroke widths in pixel units ──────────────────────────────
    SIGMA_PERP_INIT=0.6,        # edge thickness (perpendicular to tangent)
    SIGMA_PAR_INIT=2.0,         # stroke extent along tangent; init ≈ stride S

    # ── Per-(cell, bin) correction bounds (signed via tanh on MLP outputs) ──
    KAPPA_MAX_INIT=0.1,         # 1/pixel; |κ| ≤ κ_max — small to encourage straight strokes at init
    EXT_MAX_INIT=1.0,           # pixels; signed tangent shift e_s of stroke vertex

    # ── Sparsity gate: bin retained when ρ^(k) > τ · max_j ρ^(j) ───────────
    BIN_GATE_TAU_INIT=0.4,      # τ ∈ [0, 1] via sigmoid; inference-time data-awareness lever
    BIN_GATE_ALPHA_INIT=10.0,   # gate sharpness

    # ── Correction MLP topology ────────────────────────────────────────────
    CORR_HIDDEN=8,              # 4 features → 8 hidden → 2 outputs  (66 params)

    # ── Legacy / unused (kept so old configs don't crash on import) ────────
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
    LAM_DICE=0.0,
    LAM_BCE=1.0,
    # Bump when L0 / pad / ``l0_pix`` / GT schema changes — not L1 binning or seed.
    L0_CACHE_VERSION=2,
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