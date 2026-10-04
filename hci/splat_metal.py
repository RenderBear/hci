r"""Fused back-projection splat for MPS — Metal kernels for the forward sums and their gradient"""

from __future__ import annotations

import torch

# Pairs are bucketed by the tile (in pixels) their anchor falls in.
_TILE = 4
# Floats per pair in the packed table the kernels read.
_NF = 9
# Per-pair gradients the backward kernel writes before the two kernel-tap blocks.
_NG = 5

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

// Linear interpolation of a (2*hw + 1)-tap kernel at u; zero outside [-hw, hw).
static float interp_kernel(device const float* h, float u, int hw) {
    float u_floor = floor(u);
    int m_lo = int(u_floor) + hw;
    int m_hi = m_lo + 1;
    if (m_lo < 0 || m_hi > 2 * hw) return 0.0f;
    return h[m_lo] + (u - u_floor) * (h[m_hi] - h[m_lo]);
}

// log(1 - c), accurate for small c where forming 1 - c loses the digits of c.
static float log1m(float c) {
    float u = 1.0f - c;
    return (u == 1.0f) ? -c : log(u) * (-c) / (u - 1.0f);
}

// pairs: per active (cell, bin), sorted by anchor tile:
//   ax, ay, cos, sin, kappa, ext_s, amplitude, cos 2θ, sin 2θ
// out:   per pixel: Σ log(1 - claim), Σ claim·cos 2θ, Σ claim·sin 2θ
kernel void splat(device float* out,
                  device const float* pairs,
                  device const int* tile_start,
                  device const float* h_perp,
                  device const float* h_par,
                  device const int* cfg,
                  constant float& claim_clip,
                  uint idx [[thread_position_in_grid]]) {
    int W = cfg[0], ntx = cfg[1], nty = cfg[2], T = cfg[3], hw = cfg[4], NF = cfg[5];
    int px = int(idx) % W;
    int py = int(idx) / W;
    int tx0 = max(px - hw, 0) / T, tx1 = min((px + hw) / T, ntx - 1);
    int ty0 = max(py - hw, 0) / T, ty1 = min((py + hw) / T, nty - 1);

    float acc = 0.0f, mom_re = 0.0f, mom_im = 0.0f;
    for (int ty = ty0; ty <= ty1; ty++) {
        // Tiles tx0..tx1 of one row are contiguous in the sorted table.
        int k1 = tile_start[ty * ntx + tx1 + 1];
        for (int k = tile_start[ty * ntx + tx0]; k < k1; k++) {
            int q = k * NF;
            float ax_floor = floor(pairs[q]);
            float ay_floor = floor(pairs[q + 1]);
            int ox = px - int(ax_floor);
            int oy = py - int(ay_floor);
            if (abs(ox) > hw || abs(oy) > hw) continue;

            float dx = float(ox) - (pairs[q] - ax_floor);
            float dy = float(oy) - (pairs[q + 1] - ay_floor);
            float ca = pairs[q + 2], sa = pairs[q + 3];
            float s = dy * ca + dx * sa;
            float n = -dy * sa + dx * ca;
            float s_tilde = s - pairs[q + 5];
            float n_curv = n - 0.5f * pairs[q + 4] * (s_tilde * s_tilde);

            float f = max(interp_kernel(h_perp, n_curv, hw) * interp_kernel(h_par, s_tilde, hw), 0.0f);
            float claim = clamp(pairs[q + 6] * f, 0.0f, claim_clip);
            acc += log1m(claim);
            mom_re += claim * pairs[q + 7];
            mom_im += claim * pairs[q + 8];
        }
    }
    out[3 * idx] = acc;
    out[3 * idx + 1] = mom_re;
    out[3 * idx + 2] = mom_im;
}

#define MAX_TAPS 64

// Gradient of the first sum, one thread per pair: walks the pair's window, reads dL/d(sum) at
// each pixel and chains back through the same sample formula as above.
// out: per pair: d/d amplitude, d/d ax, d/d ay, d/d kappa, d/d ext_s, then that pair's share
//      of d/d h_perp[0..ntap) and d/d h_par[0..ntap)
kernel void splat_grad(device float* out,
                       device const float* pairs,
                       device const float* g_sum,
                       device const float* h_perp,
                       device const float* h_par,
                       device const int* cfg,
                       constant float& claim_clip,
                       uint idx [[thread_position_in_grid]]) {
    int W = cfg[0], H = cfg[1], hw = cfg[2], NF = cfg[3], NO = cfg[4], NG = cfg[5];
    int ntap = 2 * hw + 1;
    int q = int(idx) * NF;
    float ax_floor = floor(pairs[q]);
    float ay_floor = floor(pairs[q + 1]);
    int ax_int = int(ax_floor), ay_int = int(ay_floor);
    float ax_f = pairs[q] - ax_floor, ay_f = pairs[q + 1] - ay_floor;
    float ca = pairs[q + 2], sa = pairs[q + 3];
    float kappa = pairs[q + 4], ext_s = pairs[q + 5], amp = pairs[q + 6];

    float g_amp = 0.0f, g_ax = 0.0f, g_ay = 0.0f, g_kappa = 0.0f, g_ext = 0.0f;
    float g_perp[MAX_TAPS], g_par[MAX_TAPS];
    for (int m = 0; m < ntap; m++) { g_perp[m] = 0.0f; g_par[m] = 0.0f; }

    for (int oy = -hw; oy <= hw; oy++) {
        int py = ay_int + oy;
        if (py < 0 || py >= H) continue;
        for (int ox = -hw; ox <= hw; ox++) {
            int px = ax_int + ox;
            if (px < 0 || px >= W) continue;
            float G = g_sum[py * W + px];
            if (G == 0.0f) continue;

            float dx = float(ox) - ax_f;
            float dy = float(oy) - ay_f;
            float s = dy * ca + dx * sa;
            float n = -dy * sa + dx * ca;
            float s_tilde = s - ext_s;
            float n_curv = n - 0.5f * kappa * (s_tilde * s_tilde);

            // interp_kernel, keeping the tap index, fraction and slope for the chain rule
            float pf = floor(n_curv);
            int p_lo = int(pf) + hw;
            if (p_lo < 0 || p_lo + 1 > 2 * hw) continue;
            float p_fr = n_curv - pf;
            float p_slope = h_perp[p_lo + 1] - h_perp[p_lo];
            float hp = h_perp[p_lo] + p_fr * p_slope;

            float qf = floor(s_tilde);
            int q_lo = int(qf) + hw;
            if (q_lo < 0 || q_lo + 1 > 2 * hw) continue;
            float q_fr = s_tilde - qf;
            float q_slope = h_par[q_lo + 1] - h_par[q_lo];
            float hq = h_par[q_lo] + q_fr * q_slope;

            // f = relu(hp * hq): where it is clipped to zero the claim is zero and nothing flows.
            float r = hp * hq;
            if (!(r > 0.0f)) continue;
            float x = amp * r;
            if (!(x >= 0.0f && x <= claim_clip)) continue;      // clamp passes no gradient outside
            float d_x = -G / (1.0f - x);                        // d/dx of log(1 - x)

            g_amp += d_x * r;
            float d_r = d_x * amp;
            float d_hp = d_r * hq;
            float d_hq = d_r * hp;
            g_perp[p_lo] += (1.0f - p_fr) * d_hp;
            g_perp[p_lo + 1] += p_fr * d_hp;
            g_par[q_lo] += (1.0f - q_fr) * d_hq;
            g_par[q_lo + 1] += q_fr * d_hq;

            float d_n = d_hp * p_slope;                          // n_curv = n - kappa * s_tilde^2 / 2
            float d_s = d_hq * q_slope - d_n * kappa * s_tilde;  // s_tilde = s - ext_s
            g_kappa -= d_n * 0.5f * (s_tilde * s_tilde);
            g_ext -= d_s;
            g_ax -= d_s * sa + d_n * ca;                         // dx = ox - frac(ax)
            g_ay -= d_s * ca - d_n * sa;
        }
    }

    int o = int(idx) * NO;
    out[o] = g_amp;
    out[o + 1] = g_ax;
    out[o + 2] = g_ay;
    out[o + 3] = g_kappa;
    out[o + 4] = g_ext;
    for (int m = 0; m < ntap; m++) {
        out[o + NG + m] = g_perp[m];
        out[o + NG + ntap + m] = g_par[m];
    }
}
"""

_lib = None


def _kernels():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SOURCE)
    return _lib


def _pack(amp, ax_eff, ay_eff, cos_a, sin_a, kappa, ext_s) -> torch.Tensor:
    bar_theta = torch.atan2(sin_a, cos_a)
    return torch.stack(
        [
            ax_eff, ay_eff, cos_a, sin_a, kappa, ext_s, amp,
            torch.cos(2.0 * bar_theta), torch.sin(2.0 * bar_theta),
        ],
        dim=1,
    ).to(torch.float32)


class _SplatSums(torch.autograd.Function):
    """Per pixel: Σ log(1 − claim) and the two orientation moments (the moments carry no gradient)."""

    @staticmethod
    def forward(ctx, amp, ax_eff, ay_eff, kappa, ext_s, h_perp, h_par, cos_a, sin_a, hw, H, W, clip):
        device = amp.device
        A = amp.shape[0]
        T = _TILE
        ntx, nty = (W + T - 1) // T, (H + T - 1) // T

        # An anchor outside the image is filed under the nearest edge tile, which every pixel
        # its window can reach also visits.
        tx = torch.div(torch.floor(ax_eff).long(), T, rounding_mode="floor").clamp(0, ntx - 1)
        ty = torch.div(torch.floor(ay_eff).long(), T, rounding_mode="floor").clamp(0, nty - 1)
        tile = ty * ntx + tx
        # Unique keys, so the order inside a tile (and with it the rounding) is the same every run.
        order = torch.argsort(tile * A + torch.arange(A, device=device))
        tile_start = torch.searchsorted(
            tile[order], torch.arange(ntx * nty + 1, device=device),
        ).to(torch.int32)

        pairs = _pack(amp, ax_eff, ay_eff, cos_a, sin_a, kappa, ext_s)
        h_perp = h_perp.to(torch.float32).contiguous()
        h_par = h_par.to(torch.float32).contiguous()
        cfg = torch.tensor([W, ntx, nty, T, hw, _NF], device=device, dtype=torch.int32)
        out = torch.zeros(H * W, 3, device=device, dtype=torch.float32)
        _kernels().splat(
            out, pairs[order].contiguous(), tile_start, h_perp, h_par, cfg, float(clip),
            threads=H * W,
        )

        ctx.save_for_backward(pairs, h_perp, h_par)
        ctx.geom = (hw, H, W, clip)
        log_neg, mom = out[:, 0].contiguous(), out[:, 1:]
        ctx.mark_non_differentiable(mom)
        return log_neg, mom

    @staticmethod
    def backward(ctx, g_log_neg, _g_mom):
        pairs, h_perp, h_par = ctx.saved_tensors
        hw, H, W, clip = ctx.geom
        device = pairs.device
        A = pairs.shape[0]
        ntap = 2 * hw + 1
        n_out = _NG + 2 * ntap

        cfg = torch.tensor([W, H, hw, _NF, n_out, _NG], device=device, dtype=torch.int32)
        out = torch.zeros(A, n_out, device=device, dtype=torch.float32)
        _kernels().splat_grad(
            out, pairs, g_log_neg.to(torch.float32).contiguous(), h_perp, h_par, cfg, float(clip),
            threads=A,
        )
        g_h_perp = out[:, _NG:_NG + ntap].sum(dim=0)
        g_h_par = out[:, _NG + ntap:].sum(dim=0)
        return (
            out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4], g_h_perp, g_h_par,
            None, None, None, None, None, None,
        )


def splat_fused(
    rho_active: torch.Tensor,
    gate_active: torch.Tensor,
    alpha_active: torch.Tensor,
    ax_eff: torch.Tensor,
    ay_eff: torch.Tensor,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    kappa_active: torch.Tensor,
    ext_s_active: torch.Tensor,
    h_perp: torch.Tensor,
    h_par: torch.Tensor,
    kernel_h_w: int,
    H: int, W: int,
    claim_clip: float,
    with_theta: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sums of renderer._backproject_deposit, gathered per pixel instead of scattered per pair.

    Forward: each pixel visits the pairs anchored in the tiles its (2·hw + 1)² neighbourhood
    touches and keeps those whose window reaches it, so the contributing samples are the same
    ones. Backward: each pair walks its window and applies the chain rule of the same sample
    formula, in place of autograd over the chunked tensor passes.
    """
    device, dtype = rho_active.device, rho_active.dtype
    log_neg, mom = _SplatSums.apply(
        alpha_active * gate_active * rho_active, ax_eff, ay_eff, kappa_active, ext_s_active,
        h_perp, h_par, cos_a, sin_a, int(kernel_h_w), H, W, float(claim_clip),
    )
    bmap = (-torch.expm1(log_neg)).to(dtype=dtype)
    if with_theta:
        theta_star = (0.5 * torch.atan2(mom[:, 1], mom[:, 0])).to(dtype=dtype)
    else:
        theta_star = torch.zeros(H * W, device=device, dtype=dtype)
    return bmap.reshape(H, W), theta_star.reshape(H, W)
