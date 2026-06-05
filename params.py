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

# ── Render: §2.5 anisotropic splat + thinning head (see hci/renderer.py) ─────
RENDER = SimpleNamespace(
    # Soft-indicator deposit (hci/renderer.py): isotropic Gaussian envelope on m_c
    # in half-width–normalized (s, n); σ≈0.4–0.6 kills cross-patch line streaks. 0 = off.
    DEPOSIT_ENVELOPE_SIGMA=0.52,
    SIGMA_PAR_INIT=2.0,  # along-edge width; init ≈ L1 stride S (= P − overlap)
    SIGMA_PERP_INIT=1.0,
    SIGMA_PAR_MAX=32.0,
    SIGMA_PERP_MAX=8.0,
    SPLAT_RADIUS_SIGMAS=3.0,  # splat kernel radius in max(σ_∥, σ_⊥) units
    THETA_SMOOTH_PASSES=4,
    # Thinning head: F_p ∈ R^20 = [ρ̄, coh, tang9, norm9]; MLP 20→12→1
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
