r"""L1 — per-patch eigendecomposition → cell grid (GPU-native)."""

from __future__ import annotations

import math

import numpy as np
import torch

from params import L1


def stride_from_patch_overlap(P: int, patch_overlap: int) -> int:
    o = int(patch_overlap)
    p = int(P)
    if o < 0 or o >= p:
        raise ValueError(f"overlap must be in [0, {p}), got {o}")
    return p - o


def pad_for_patch_grid(
    img: np.ndarray,
    patch_size: int,
    patch_overlap: int,
) -> tuple[np.ndarray, int, int]:
    H0, W0 = img.shape[:2]
    S = stride_from_patch_overlap(patch_size, patch_overlap)

    def pd(d0):
        if d0 <= patch_size:
            return int(patch_size)
        n = (d0 - patch_size + S - 1) // S + 1
        return (n - 1) * S + patch_size

    Hp, Wp = pd(H0), pd(W0)
    if Hp == H0 and Wp == W0:
        return img, H0, W0
    out = np.pad(img, ((0, Hp - H0), (0, Wp - W0), (0, 0)), mode="reflect")
    return out, H0, W0


def z_from_l0_harmonics(
    s: torch.Tensor,
    border_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z1 = torch.complex(s[..., 0], s[..., 1])
    z2 = torch.complex(s[..., 2], s[..., 3])
    z1[border_mask] = 0.0
    z2[border_mask] = 0.0
    return z1, z2


def _extract_patches_torch(
    t: torch.Tensor,
    nH: int,
    nW: int,
    P: int,
    S: int,
) -> torch.Tensor:
    patches = t.unfold(0, P, S).unfold(1, P, S)
    return patches.contiguous().reshape(nH * nW, P * P)


def _patch_orientation_kappa(
    z1_patches: torch.Tensor,
    theta: torch.Tensor,
    q: torch.Tensor,
    P: int,
    eps: float,
) -> torch.Tensor:
    """Per-cell per-branch orientation confidence κ ∈ [0, 1].

    κ = |mean(z₁ projections onto edge normal)| / mean(|projections|).
    High when patch z₁ agrees on bright/dark side relative to q; low at ridges/texture.
    """
    n, _ = z1_patches.shape
    device = z1_patches.device
    half = (P - 1) / 2.0
    ys = torch.arange(P, device=device, dtype=torch.float64) - half
    xs = torch.arange(P, device=device, dtype=torch.float64) - half
    dy, dx = torch.meshgrid(ys, xs, indexing="ij")
    di_f, dj_f = dy.reshape(-1), dx.reshape(-1)
    rd = (di_f.pow(2) + dj_f.pow(2)).sqrt().clamp_min(eps)
    oi, oj = di_f / rd, dj_f / rd

    z1_c = z1_patches.to(torch.complex128)
    zr = z1_c.real
    zi = z1_c.imag

    sgn_q = torch.sign(q)
    sgn_q = torch.where(sgn_q == 0, torch.ones_like(sgn_q), sgn_q)
    nx = sgn_q * torch.cos(theta)
    ny = -sgn_q * torch.sin(theta)

    proj = nx.unsqueeze(-1) * zr.unsqueeze(1) + ny.unsqueeze(-1) * zi.unsqueeze(1)
    signed_mean = proj.mean(dim=-1)
    abs_mean = proj.abs().mean(dim=-1).clamp_min(eps)
    return (signed_mean.abs() / abs_mean).clamp(0.0, 1.0).to(torch.float32)


def _build_moment_matrices(Z0, Z2, Z4):
    N = Z0.shape[0]
    device = Z0.device
    M = torch.zeros((N, 3, 3), dtype=torch.complex128, device=device)

    Z0_c = Z0.to(torch.complex128)
    Z2_conj = Z2.conj()
    Z4_conj = Z4.conj()

    M[:, 0, 0] = Z0_c
    M[:, 0, 1] = Z2_conj
    M[:, 0, 2] = Z4_conj
    M[:, 1, 0] = Z2
    M[:, 1, 1] = Z0_c
    M[:, 1, 2] = Z2_conj
    M[:, 2, 0] = Z4
    M[:, 2, 1] = Z2
    M[:, 2, 2] = Z0_c

    return M


def _eigh_3x3_hermitian(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eps_local = 1e-30
    dtype_c = M.dtype
    dtype_r = torch.float64 if dtype_c == torch.complex128 else torch.float32
    dev = M.device

    a00 = M[..., 0, 0].real.to(dtype_r)
    a11 = M[..., 1, 1].real.to(dtype_r)
    a22 = M[..., 2, 2].real.to(dtype_r)
    m01 = M[..., 0, 1]
    m02 = M[..., 0, 2]
    m12 = M[..., 1, 2]
    s01 = (m01.real * m01.real + m01.imag * m01.imag).to(dtype_r)
    s02 = (m02.real * m02.real + m02.imag * m02.imag).to(dtype_r)
    s12 = (m12.real * m12.real + m12.imag * m12.imag).to(dtype_r)

    p1 = a00 + a11 + a22
    p2 = (a00 * a11 - s01) + (a00 * a22 - s02) + (a11 * a22 - s12)
    triple = m01 * m12 * m02.conj()
    p3 = (
        a00 * (a11 * a22 - s12)
        - a22 * s01
        - a11 * s02
        + 2.0 * triple.real.to(dtype_r)
    )

    p1_3 = p1 / 3.0
    q = (p2 - p1 * p1 / 3.0) / 3.0
    r = (p3 - p1 * p2 / 3.0 + 2.0 * p1 * p1 * p1 / 27.0) / 2.0

    nq = (-q).clamp_min(0.0)
    nq_sqrt = nq.sqrt()
    nq_pow_3_2 = nq_sqrt * nq

    deg_eps = 1e-24 if dtype_r == torch.float64 else 1e-10
    matrix_scale_sq = (p1 * p1).clamp_min(deg_eps)
    degenerate = nq < deg_eps * matrix_scale_sq
    safe_denom = nq_pow_3_2.clamp_min(eps_local)
    cos_phi = (r / safe_denom).clamp(-1.0, 1.0)
    phi = torch.acos(cos_phi)

    two_sqrt_nq = 2.0 * nq_sqrt
    third = 1.0 / 3.0
    mu_max = two_sqrt_nq * torch.cos(phi * third)
    mu_mid = two_sqrt_nq * torch.cos((phi - 2.0 * math.pi) * third)
    mu_min = two_sqrt_nq * torch.cos((phi - 4.0 * math.pi) * third)

    lam_max = torch.where(degenerate, p1_3, p1_3 + mu_max)
    lam_mid = torch.where(degenerate, p1_3, p1_3 + mu_mid)
    lam_min = torch.where(degenerate, p1_3, p1_3 + mu_min)

    w = torch.stack([lam_min, lam_mid, lam_max], dim=-1)

    batch_shape = M.shape[:-2]
    M_flat = M.reshape(-1, 3, 3)
    w_flat = w.reshape(-1, 3)
    N = M_flat.shape[0]
    arange_N = torch.arange(N, device=dev)
    I3 = torch.eye(3, dtype=dtype_c, device=dev).unsqueeze(0)

    g_lo = w_flat[:, 1] - w_flat[:, 0]
    g_hi = w_flat[:, 2] - w_flat[:, 1]
    gaps = torch.stack([g_lo, torch.minimum(g_lo, g_hi), g_hi], dim=-1)
    order = gaps.argsort(dim=-1, descending=True)

    cp_eps = 1e-30 if dtype_r == torch.float64 else 1e-14
    matrix_scale = p1.reshape(-1).abs().clamp_min(1.0)
    cp_threshold = (
        cp_eps * matrix_scale * matrix_scale * matrix_scale * matrix_scale
    )

    lam_complex = w_flat.to(dtype_c)
    V_unsorted = torch.empty_like(M_flat)

    for j in range(3):
        k = order[:, j]
        lam_k = lam_complex.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        A = M_flat - lam_k.unsqueeze(-1).unsqueeze(-1) * I3
        r0, r1, r2 = A[:, 0, :], A[:, 1, :], A[:, 2, :]

        u01 = torch.stack([
            r0[:, 1] * r1[:, 2] - r0[:, 2] * r1[:, 1],
            r0[:, 2] * r1[:, 0] - r0[:, 0] * r1[:, 2],
            r0[:, 0] * r1[:, 1] - r0[:, 1] * r1[:, 0],
        ], dim=-1)
        u02 = torch.stack([
            r0[:, 1] * r2[:, 2] - r0[:, 2] * r2[:, 1],
            r0[:, 2] * r2[:, 0] - r0[:, 0] * r2[:, 2],
            r0[:, 0] * r2[:, 1] - r0[:, 1] * r2[:, 0],
        ], dim=-1)
        u12 = torch.stack([
            r1[:, 1] * r2[:, 2] - r1[:, 2] * r2[:, 1],
            r1[:, 2] * r2[:, 0] - r1[:, 0] * r2[:, 2],
            r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0],
        ], dim=-1)

        n01 = (u01.real * u01.real + u01.imag * u01.imag).sum(dim=-1)
        n02 = (u02.real * u02.real + u02.imag * u02.imag).sum(dim=-1)
        n12 = (u12.real * u12.real + u12.imag * u12.imag).sum(dim=-1)
        norms = torch.stack([n01, n02, n12], dim=-1)
        candidates = torch.stack([u01, u02, u12], dim=1)
        best = norms.argmax(dim=-1)
        v_cand = candidates[arange_N, best]
        best_norm_sq = norms.gather(-1, best.unsqueeze(-1)).squeeze(-1)

        for jj in range(j):
            vj = V_unsorted[:, :, jj]
            proj = (vj.conj() * v_cand).sum(dim=-1, keepdim=True)
            v_cand = v_cand - proj * vj

        v_norm_sq = (v_cand.real * v_cand.real + v_cand.imag * v_cand.imag).sum(dim=-1)
        good = (best_norm_sq > cp_threshold) & (v_norm_sq > cp_threshold)
        v_cand = v_cand / v_norm_sq.clamp_min(eps_local).sqrt().to(dtype_c).unsqueeze(-1)

        if j == 0:
            fb = torch.zeros_like(v_cand)
            fb[:, 0] = 1.0
        elif j == 1:
            v0 = V_unsorted[:, :, 0]
            min_idx = v0.abs().argmin(dim=-1)
            fb = torch.zeros_like(v_cand)
            fb[arange_N, min_idx] = 1.0
            proj = (v0.conj() * fb).sum(dim=-1, keepdim=True)
            fb = fb - proj * v0
            fb_n = (fb.real * fb.real + fb.imag * fb.imag).sum(
                dim=-1, keepdim=True
            ).clamp_min(eps_local).sqrt().to(dtype_c)
            fb = fb / fb_n
        else:
            v0, v1 = V_unsorted[:, :, 0], V_unsorted[:, :, 1]
            fb = torch.stack([
                v0[:, 1] * v1[:, 2] - v0[:, 2] * v1[:, 1],
                v0[:, 2] * v1[:, 0] - v0[:, 0] * v1[:, 2],
                v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0],
            ], dim=-1)
            fb_n = (fb.real * fb.real + fb.imag * fb.imag).sum(
                dim=-1, keepdim=True
            ).clamp_min(eps_local).sqrt().to(dtype_c)
            fb = fb / fb_n

        V_unsorted[:, :, j] = torch.where(good.unsqueeze(-1), v_cand, fb)

    inv_order = torch.argsort(order, dim=-1)
    inv_idx = inv_order.unsqueeze(1).expand(-1, 3, -1)
    V_flat = torch.gather(V_unsorted, -1, inv_idx)

    V = V_flat.reshape(*batch_shape, 3, 3)
    return w, V


def _theta_from_eigenvector(v_col: torch.Tensor) -> torch.Tensor:

    eps = 1e-15
    v0 = v_col[:, 0]
    v1 = v_col[:, 1]
    v2 = v_col[:, 2]
    use_first = v0.abs() >= eps
    ratio = torch.where(
        use_first,
        v1 / torch.where(use_first, v0, torch.ones_like(v0)),
        v2 / torch.where(~use_first & (v1.abs() >= eps), v1, torch.ones_like(v1)),
    )
    th = 0.5 * torch.angle(ratio)
    th = th % math.pi
    return th


def run_l1(
    z1: torch.Tensor,
    z2: torch.Tensor,
    P: int,
    border_mask: torch.Tensor,
    patch_overlap: int,
    border_patch_max_frac: float,
    eps: float,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict:

    dev = torch.device(device) if device is not None else z1.device
    z1 = z1.to(dev)
    z2 = z2.to(dev)
    border_mask = border_mask.to(dev)

    H, W = z1.shape
    S = stride_from_patch_overlap(P, patch_overlap)
    nH = (H - P) // S + 1 if H >= P else 0
    nW = (W - P) // S + 1 if W >= P else 0
    n = nH * nW

    if verbose:
        print(f"  grid {nH}x{nW} = {n} cells, {L1.N_BRANCHES} branches/cell (λ₁, λ₂)")

    z2_patches = _extract_patches_torch(z2, nH, nW, P, S)
    z1_patches = _extract_patches_torch(z1, nH, nW, P, S)

    bm_float = border_mask.float()
    bm_patches = _extract_patches_torch(bm_float, nH, nW, P, S)
    is_border_flat = bm_patches.mean(dim=-1) > border_patch_max_frac

    z2_abs = z2_patches.abs().to(torch.float64)
    local_y_64 = (
        torch.arange(P, device=dev, dtype=torch.float64)
        .unsqueeze(1)
        .expand(P, P)
        .reshape(-1)
    )
    local_x_64 = (
        torch.arange(P, device=dev, dtype=torch.float64)
        .unsqueeze(0)
        .expand(P, P)
        .reshape(-1)
    )

    z2_patches_c128 = z2_patches.to(torch.complex128)
    Z0 = z2_abs.sum(dim=-1)
    Z2 = z2_patches_c128.sum(dim=-1)
    Z4 = (z2_patches_c128 * z2_patches_c128).sum(dim=-1)

    z1_patches_c128 = z1_patches.to(torch.complex128)
    Z1_sum = z1_patches_c128.sum(dim=-1)
    z1_abs_sum = z1_patches.abs().sum(dim=-1).to(torch.float64)

    n_pix_patch_f = float(P * P)
    z1_bar = Z1_sum / n_pix_patch_f
    z2_bar = Z2 / n_pix_patch_f
    z1_bar_abs_flat = z1_bar.abs().to(torch.float32)
    z2_bar_abs_flat = z2_bar.abs().to(torch.float32)
    phi_z1_flat = torch.angle(z1_bar).to(torch.float32)
    z1_bar_abs_flat = torch.where(
        is_border_flat, torch.zeros_like(z1_bar_abs_flat), z1_bar_abs_flat
    )
    z2_bar_abs_flat = torch.where(
        is_border_flat, torch.zeros_like(z2_bar_abs_flat), z2_bar_abs_flat
    )
    phi_z1_flat = torch.where(is_border_flat, torch.zeros_like(phi_z1_flat), phi_z1_flat)

    M = _build_moment_matrices(Z0, Z2, Z4)
    del (
        Z0,
        Z2,
        Z4,
        z2_patches,
        z2_patches_c128,
        z1_patches_c128,
        bm_float,
        bm_patches,
    )

    w_all, v_all = _eigh_3x3_hermitian(M)
    del M
    if not torch.isfinite(w_all).all():
        w_all = torch.where(torch.isfinite(w_all), w_all, torch.zeros_like(w_all))
    if not torch.isfinite(v_all).all():
        v_all = torch.where(
            torch.isfinite(v_all.real).unsqueeze(-1).all(dim=-1, keepdim=True),
            v_all,
            torch.zeros_like(v_all),
        )

    lam3 = w_all[:, 0]
    lam2 = w_all[:, 1]
    lam1 = w_all[:, 2]

    th_b0 = _theta_from_eigenvector(v_all[:, :, 2])
    th_b1 = _theta_from_eigenvector(v_all[:, :, 1])

    def _q_for_theta(theta: torch.Tensor) -> torch.Tensor:
        exp_ith = torch.complex(torch.cos(theta), torch.sin(theta))
        return (Z1_sum.conj() * exp_ith).real

    q_b0 = _q_for_theta(th_b0)
    q_b1 = _q_for_theta(th_b1)

    delta_b0 = ((lam1 - lam3) / (lam1 + eps)).clamp(0.0, 1.0)
    delta_b1 = ((lam2 - lam3) / (lam2 + eps)).clamp(0.0, 1.0)

    theta_flat = torch.stack([th_b0, th_b1], dim=-1)
    q_flat = torch.stack([q_b0, q_b1], dim=-1)
    delta_flat = torch.stack([delta_b0, delta_b1], dim=-1)
    lam_flat = torch.stack([lam1, lam2], dim=-1)

    kappa_flat = _patch_orientation_kappa(
        z1_patches, theta_flat, q_flat, P, eps,
    )
    del z1_patches

    lam_pair_sum = lam1 + lam2
    lam_sum = lam1 + lam2 + lam3
    lam_bar = lam_sum / 3.0

    theta_flat[is_border_flat] = 0.0
    q_flat[is_border_flat] = 0.0
    delta_flat[is_border_flat] = 0.0
    lam_flat[is_border_flat] = 0.0
    kappa_flat[is_border_flat] = 0.0

    lam3_masked = lam3.clone()
    lam3_masked[is_border_flat] = 0.0

    pis = torch.arange(nH, device=dev).repeat_interleave(nW)
    pjs = torch.arange(nW, device=dev).repeat(nH)
    cx_flat = pjs.double() * S + P / 2.0
    cy_flat = pis.double() * S + P / 2.0

    sw = z2_abs.sum(dim=-1)
    has_mass = sw > eps

    # Global z₂ centroid (for renderer anchoring)
    glob_x_64 = (pjs.double() * S).unsqueeze(1) + local_x_64.unsqueeze(0)
    glob_y_64 = (pis.double() * S).unsqueeze(1) + local_y_64.unsqueeze(0)
    wx = (z2_abs * glob_x_64).sum(dim=-1)
    wy = (z2_abs * glob_y_64).sum(dim=-1)
    cx_z2_flat = torch.where(has_mass, wx / sw.clamp_min(eps), cx_flat)
    cy_z2_flat = torch.where(has_mass, wy / sw.clamp_min(eps), cy_flat)
    cx_z2_flat = torch.where(is_border_flat, cx_flat, cx_z2_flat)
    cy_z2_flat = torch.where(is_border_flat, cy_flat, cy_z2_flat)
    del z2_abs, glob_x_64, glob_y_64, local_x_64, local_y_64, sw, wx, wy

    if verbose:
        lam1_np = lam1.cpu().numpy()
        lam2_np = lam2.cpu().numpy()
        lam3_np = lam3.cpu().numpy()
        n_border = int(is_border_flat.sum().item())
        n_active = n - n_border
        print(f"  active={n_active}  border={n_border}")
        print(f"  λ₁: mean={lam1_np.mean():.2f} max={lam1_np.max():.2f}")
        print(f"  λ₂: mean={lam2_np.mean():.2f} max={lam2_np.max():.2f}")
        print(f"  λ₃: mean={lam3_np.mean():.4f}")
        print(
            f"  δ₀: mean={float(delta_b0.mean()):.3f}  "
            f"δ₁: mean={float(delta_b1.mean()):.3f}"
        )
        print(
            f"  |q|₀: mean={float(q_b0.abs().mean()):.3f}  "
            f"|q|₁: mean={float(q_b1.abs().mean()):.3f}"
        )
        print(
            f"  κ₀: mean={float(kappa_flat[:, 0].mean()):.3f}  "
            f"κ₁: mean={float(kappa_flat[:, 1].mean()):.3f}"
        )

    def _to_np(t):
        return t.cpu().numpy()

    out = {
        "nH": nH,
        "nW": nW,
        "S": S,
        "P": P,
        "theta": _to_np(theta_flat.reshape(nH, nW, L1.N_BRANCHES)),
        "q": _to_np(q_flat.reshape(nH, nW, L1.N_BRANCHES)),
        "delta": _to_np(delta_flat.reshape(nH, nW, L1.N_BRANCHES)),
        "kappa": _to_np(kappa_flat.reshape(nH, nW, L1.N_BRANCHES)),
        "lam": _to_np(lam_flat.reshape(nH, nW, L1.N_BRANCHES)),
        "lam3": _to_np(lam3_masked.reshape(nH, nW)),
        "z0": _to_np(lam_pair_sum.reshape(nH, nW)),
        "lam_bar": _to_np(lam_bar.reshape(nH, nW)),
        "cx": _to_np(cx_flat.reshape(nH, nW)),
        "cy": _to_np(cy_flat.reshape(nH, nW)),
        "cx_z2": _to_np(cx_z2_flat.reshape(nH, nW)),
        "cy_z2": _to_np(cy_z2_flat.reshape(nH, nW)),
        "z1_bar_abs": _to_np(z1_bar_abs_flat.reshape(nH, nW)),
        "z1_abs_sum": _to_np(z1_abs_sum.to(torch.float32).reshape(nH, nW)),
        "z2_bar_abs": _to_np(z2_bar_abs_flat.reshape(nH, nW)),
        "phi_z1": _to_np(phi_z1_flat.reshape(nH, nW)),
        "is_border": _to_np(is_border_flat.reshape(nH, nW)),
    }
    return out