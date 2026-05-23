r"""Shared matplotlib diagnostics: reusable plots and infer.py figure helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

from params import L0, VIZ


def rho_heatmap_cmap():
    cdict = {
        "red": [(0.0, 0.0, 0.0), (0.33, 1.0, 1.0), (0.66, 1.0, 1.0), (1.0, 1.0, 1.0)],
        "green": [(0.0, 0.0, 0.0), (0.33, 0.0, 0.0), (0.66, 1.0, 1.0), (1.0, 1.0, 1.0)],
        "blue": [(0.0, 0.0, 0.0), (0.33, 0.0, 0.0), (0.66, 0.0, 0.0), (1.0, 1.0, 1.0)],
    }
    return mcolors.LinearSegmentedColormap("rho_bryw", cdict)


def apply_border_zero(g, is_border):
    out = np.asarray(g, dtype=np.float64).copy()
    b = np.asarray(is_border, dtype=bool)
    if out.shape != b.shape:
        # e.g. L2 tile-member arrays (N_T, M) vs cell-grid border mask (nH, nW)
        return out
    out[b] = 0.0
    return out


def branch_commit_rgb(branch_idx: np.ndarray, is_border: np.ndarray) -> np.ndarray:

    b = np.asarray(branch_idx, dtype=np.int32)
    interior = ~np.asarray(is_border, dtype=bool)
    rgb = np.zeros((*b.shape, 3), dtype=np.float32)
    rgb[interior & (b == 0)] = (1.0, 0.0, 0.0)
    rgb[interior & (b == 1)] = (0.0, 1.0, 0.0)
    return rgb


def viz_l0_pinwheel(h, img, out_path):

    H, W, N = h.shape
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), facecolor=VIZ.BG)
    fig.suptitle(
        "L0 pinwheel (directional contrast)",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    for ax in axes.ravel():
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.axis("off")
    axes[1, 1].imshow(np.clip(img, 0.0, 1.0))
    axes[1, 1].set_title("input", fontsize=8, color=VIZ.FG, fontfamily="monospace")
    h_max = max(h.max(), VIZ.EPS)
    for k, (dy, dx) in enumerate(L0.OFFSETS):
        r, c = VIZ.GRID_POS[(dy, dx)]
        ax = axes[r, c]
        ax.imshow(h[:, :, k], cmap="hot", vmin=0, vmax=h_max)
        ax.set_title(
            f"h_{VIZ.COMPASS[(dy, dx)]} (k={k})",
            fontsize=8,
            color=VIZ.FG,
            fontfamily="monospace",
        )
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def _interior_vmin_vmax(arr: np.ndarray, interior: np.ndarray) -> tuple[float, float]:
    if not interior.any():
        return 0.0, 1.0
    v = arr[interior]
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmin == vmax:
        vmax = vmin + max(abs(vmin), 1.0) * 1e-12 + 1e-20
    return vmin, vmax


def viz_l1_lambdas_three(
    lam1: np.ndarray,
    lam2: np.ndarray,
    lam3: np.ndarray,
    is_border: np.ndarray,
    out_path: str,
    suptitle: str,
) -> None:

    cmap = "coolwarm"
    a1 = apply_border_zero(lam1, is_border)
    a2 = apply_border_zero(lam2, is_border)
    a3 = apply_border_zero(lam3, is_border)
    interior = ~np.asarray(is_border, dtype=bool)

    panels = [
        (a1, r"$\lambda_1$ (branch 0)"),
        (a2, r"$\lambda_2$ (branch 1)"),
        (a3, r"$\lambda_3$ (smallest eigenvalue)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 5.2 * 3), facecolor=VIZ.BG)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (arr, title_base) in zip(axes_flat, panels):
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.axis("off")
        vmin, vmax = _interior_vmin_vmax(arr, interior)
        im = ax.imshow(
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(
            f"{title_base}\ninterior min={vmin:.4g}  max={vmax:.4g}",
            fontsize=9,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        cbar = fig.colorbar(
            im,
            ax=ax,
            fraction=0.035,
            pad=0.02,
            shrink=0.82,
        )
        cbar.outline.set_edgecolor(VIZ.ACCENT)
        cbar.ax.tick_params(colors=VIZ.FG, labelsize=7)

    fig.suptitle(
        f"{suptitle}  (each row: own min→max; red=max, blue=min)",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "Legend: coolwarm on interior cells — blue = minimum, red = maximum "
        "(separate scale per λ; border cells masked).",
        ha="center",
        va="bottom",
        fontsize=8,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_rho_branch_grid(
    grids,
    titles,
    is_border,
    out_path,
    suptitle,
    layout_rows_cols,
    *,
    per_panel_max: bool = True,
    branch_idx: np.ndarray | None = None,
    **kwargs: Any,
) -> None:

    cmap = rho_heatmap_cmap()
    has_branch = branch_idx is not None
    grids = list(grids)
    titles = list(titles)
    if has_branch:
        grids.append(branch_commit_rgb(branch_idx, is_border))
        titles.append("commitment (red=br1, green=br2)")

    n = len(grids)
    nrows, ncols = layout_rows_cols
    cleaned: list = []
    for g in grids:
        a = np.asarray(g)
        if a.ndim == 2:
            cleaned.append(apply_border_zero(a, is_border))
        else:
            cleaned.append(a.astype(np.float32))

    m_global = max(
        (float(np.max(c)) for c in cleaned if c.ndim == 2),
        default=VIZ.EPS,
    )
    m_global = max(m_global, VIZ.EPS)

    layout_rho_with_branch = has_branch and n == 6
    layout_221 = n == 5 and nrows == 3 and ncols == 2
    layout_211 = n == 3 and nrows == 2 and ncols == 2
    if layout_rho_with_branch:
        fig = plt.figure(figsize=(5 * 2, 5 * 3), facecolor=VIZ.BG)
        gs = fig.add_gridspec(3, 2)
        axes_flat = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
        ]
    elif layout_221:
        fig = plt.figure(figsize=(5 * ncols, 5 * nrows), facecolor=VIZ.BG)
        gs = fig.add_gridspec(nrows, ncols)
        axes_flat = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, :]),
        ]
    elif layout_211:
        fig = plt.figure(figsize=(5 * ncols, 5 * nrows), facecolor=VIZ.BG)
        gs = fig.add_gridspec(nrows, ncols)
        axes_flat = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :]),
        ]
    else:
        assert nrows * ncols >= n
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 5 * nrows), facecolor=VIZ.BG
        )
        axes_flat = np.atleast_1d(axes).ravel()
    n_axes = len(axes_flat)
    for i in range(n_axes):
        ax = axes_flat[i]
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.axis("off")
        if i < n:
            arr = cleaned[i]
            if arr.ndim == 2:
                m_i = max(float(np.max(arr)), VIZ.EPS)
                scale = m_i if per_panel_max else m_global
                ax.imshow(
                    arr / scale,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                ax.set_title(
                    f"{titles[i]}\nraw max={m_i:.4g}",
                    fontsize=9,
                    color=VIZ.FG,
                    fontfamily="monospace",
                )
            else:
                ax.imshow(
                    np.clip(arr, 0.0, 1.0),
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                ax.set_title(
                    titles[i],
                    fontsize=9,
                    color=VIZ.FG,
                    fontfamily="monospace",
                )
        else:
            ax.set_visible(False)
    scale_note = (
        "per-panel max → each panel 0…1 (magnitudes not comparable across panels)"
        if per_panel_max
        else f"shared scale max={m_global:.4g}"
    )
    fig.suptitle(
        f"{suptitle}  ({scale_note})", fontsize=10, color=VIZ.FG, fontfamily="monospace"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def default_edge_threshold(bmap, edge_frac=0.03, floor=0.0):
    sv = np.sort(np.asarray(bmap, dtype=np.float64).ravel())[::-1]
    idx = int(len(sv) * edge_frac)
    return max(float(sv[min(idx, len(sv) - 1)]), float(floor))


def report_rho_pooled_gt_separation(
    rho_pooled: np.ndarray | Any,
    gt: np.ndarray | Any,
    cell_cy: np.ndarray | Any,
    cell_cx: np.ndarray | Any,
    is_border: np.ndarray | Any,
    *,
    activated_floor: float = 0.05,
    dilate_iters: int = 1,
    print_report: bool = True,
) -> dict[str, float | int]:
    """Compare pre-bistable ρ_pooled on dilated-GT keep vs drop cells."""
    from scipy.ndimage import binary_dilation

    pooled = np.asarray(rho_pooled, dtype=np.float64)
    gt_arr = np.asarray(gt, dtype=np.float64)
    cy = np.rint(np.asarray(cell_cy, dtype=np.float64)).astype(np.int64)
    cx = np.rint(np.asarray(cell_cx, dtype=np.float64)).astype(np.int64)
    border = np.asarray(is_border, dtype=bool)
    if pooled.shape != border.shape:
        raise ValueError(
            f"rho_pooled and is_border must match; got {pooled.shape} vs {border.shape}"
        )
    if cy.shape != pooled.shape or cx.shape != pooled.shape:
        raise ValueError("cell anchor grids must match rho_pooled shape")

    gt_bin = gt_arr > 0.5
    if dilate_iters > 0:
        gt_bin = binary_dilation(gt_bin, iterations=int(dilate_iters))

    H, W = gt_bin.shape
    valid = ~border.ravel()
    pooled_flat = pooled.ravel()[valid]
    cy_flat = cy.ravel()[valid]
    cx_flat = cx.ravel()[valid]
    in_bounds = (
        (cy_flat >= 0) & (cy_flat < H) & (cx_flat >= 0) & (cx_flat < W)
    )
    keep = np.zeros(pooled_flat.shape[0], dtype=bool)
    keep[in_bounds] = gt_bin[cy_flat[in_bounds], cx_flat[in_bounds]]
    drop = ~keep
    activated = pooled_flat > activated_floor

    def _summarize(mask: np.ndarray) -> dict[str, float | int]:
        vals = pooled_flat[mask]
        if vals.size == 0:
            return {"n": 0, "median": float("nan"), "mean": float("nan")}
        return {
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
        }

    metrics = {
        "n_interior": int(pooled_flat.size),
        "n_keep": int(keep.sum()),
        "n_drop": int(drop.sum()),
        "keep_all": _summarize(keep),
        "drop_all": _summarize(drop),
        "keep_activated": _summarize(keep & activated),
        "drop_activated": _summarize(drop & activated),
    }
    if print_report:
        print("ρ_pooled vs dilated GT (interior cells)")
        print(
            f"  keep: n={metrics['keep_all']['n']}  "
            f"median={metrics['keep_all']['median']:.3f}  "
            f"mean={metrics['keep_all']['mean']:.3f}"
        )
        print(
            f"  drop: n={metrics['drop_all']['n']}  "
            f"median={metrics['drop_all']['median']:.3f}  "
            f"mean={metrics['drop_all']['mean']:.3f}"
        )
        print(
            f"  activated keep (ρ>{activated_floor:.2f}): "
            f"n={metrics['keep_activated']['n']}  "
            f"median={metrics['keep_activated']['median']:.3f}"
        )
        print(
            f"  activated drop (ρ>{activated_floor:.2f}): "
            f"n={metrics['drop_activated']['n']}  "
            f"median={metrics['drop_activated']['median']:.3f}"
        )
    return metrics


def viz_hist_cdf_columns(
    maps,
    titles,
    is_border,
    out_path,
    suptitle,
    n_bins=50,
):

    n = len(maps)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.5), facecolor=VIZ.BG)
    if n == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(suptitle, fontsize=10, color=VIZ.FG, fontfamily="monospace")
    for col, (raw, title) in enumerate(zip(maps, titles)):
        g = apply_border_zero(raw, is_border)
        flat = g.ravel()[~is_border.ravel()]

        ax_h = axes[0, col]
        ax_h.set_facecolor(VIZ.PANEL_BG)
        ax_h.hist(
            flat,
            bins=n_bins,
            range=(0.0, 1.0),
            color=VIZ.ACCENT,
            edgecolor=VIZ.PANEL_BG,
            linewidth=0.3,
        )
        ax_h.set_xlim(0.0, 1.0)
        ax_h.set_title(
            f"{title}\nρ vs count  n={flat.size}",
            fontsize=8,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        ax_h.set_ylabel("count", fontsize=8, color=VIZ.FG)
        ax_h.tick_params(colors=VIZ.FG, labelsize=7)
        for s in ax_h.spines.values():
            s.set_color(VIZ.ACCENT)

        ax_c = axes[1, col]
        ax_c.set_facecolor(VIZ.PANEL_BG)
        n_cells = int(flat.size)
        if n_cells > 0:
            xs = np.sort(flat.astype(np.float64, copy=False))
            ys = np.arange(1, n_cells + 1, dtype=np.float64)
            ax_c.plot(xs, ys, color=VIZ.FG, linewidth=1.2)
        ax_c.set_xlim(0.0, 1.0)
        ax_c.set_ylim(0.0, float(max(n_cells, 1)))
        ax_c.set_xlabel("ρ", fontsize=8, color=VIZ.FG)
        ax_c.set_ylabel("cumulative count", fontsize=8, color=VIZ.FG)
        ax_c.set_title(
            "empirical CDF (unnormalized y)",
            fontsize=8,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        ax_c.grid(True, alpha=0.25, color=VIZ.ACCENT)
        ax_c.tick_params(colors=VIZ.FG, labelsize=7)
        for s in ax_c.spines.values():
            s.set_color(VIZ.ACCENT)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_infer_rho_map_hist_cdf(
    rho: np.ndarray,
    is_border: np.ndarray,
    out_path: str,
    *,
    eta_z: float | None = None,
    n_bins: int = 50,
) -> None:
    """Single cell-grid ρ: heatmap (per-cell max scale) + interior histogram + CDF."""
    g = apply_border_zero(np.asarray(rho, dtype=np.float64), is_border)
    ib = np.asarray(is_border, dtype=bool)
    flat = g.ravel()[~ib.ravel()]

    fig = plt.figure(figsize=(11.5, 6.2), facecolor=VIZ.BG)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.38, 1.0], height_ratios=[1, 1],
        wspace=0.30, hspace=0.34,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_cdf = fig.add_subplot(gs[1, 1])

    cmap = rho_heatmap_cmap()
    m_i = max(float(np.max(g)) if g.size else 0.0, VIZ.EPS)
    scale = m_i
    ax_map.set_facecolor(VIZ.PANEL_BG)
    ax_map.imshow(
        g / scale,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax_map.set_title(
        rf"cell $\rho$  (NR seed; raw max={m_i:.4g})",
        fontsize=9,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    ax_map.axis("off")

    ax_hist.set_facecolor(VIZ.PANEL_BG)
    if flat.size:
        ax_hist.hist(
            flat,
            bins=n_bins,
            range=(0.0, 1.0),
            color=VIZ.ACCENT,
            edgecolor=VIZ.PANEL_BG,
            linewidth=0.3,
        )
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_title(
        f"interior ρ vs count  n={flat.size}",
        fontsize=8,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    ax_hist.set_ylabel("count", fontsize=8, color=VIZ.FG)
    ax_hist.tick_params(colors=VIZ.FG, labelsize=7)
    for s in ax_hist.spines.values():
        s.set_color(VIZ.ACCENT)

    ax_cdf.set_facecolor(VIZ.PANEL_BG)
    n_cells = int(flat.size)
    if n_cells > 0:
        xs = np.sort(flat.astype(np.float64, copy=False))
        ys = np.arange(1, n_cells + 1, dtype=np.float64)
        ax_cdf.plot(xs, ys, color=VIZ.FG, linewidth=1.2)
    ax_cdf.set_xlim(0.0, 1.0)
    ax_cdf.set_ylim(0.0, float(max(n_cells, 1)))
    ax_cdf.set_xlabel("ρ", fontsize=8, color=VIZ.FG)
    ax_cdf.set_ylabel("cumulative count", fontsize=8, color=VIZ.FG)
    ax_cdf.set_title(
        "empirical CDF (unnormalized y)",
        fontsize=8,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    ax_cdf.grid(True, alpha=0.25, color=VIZ.ACCENT)
    ax_cdf.tick_params(colors=VIZ.FG, labelsize=7)
    for s in ax_cdf.spines.values():
        s.set_color(VIZ.ACCENT)

    eta_note = (
        rf"  $\eta_z$={eta_z:.4g}" if eta_z is not None else ""
    )
    fig.suptitle(
        "Cell ρ — map, histogram (interior), empirical CDF" + eta_note,
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_infer_rho_seed_final_dual_maps(
    rho_seed: np.ndarray,
    rho_final: np.ndarray,
    is_border: np.ndarray,
    out_path: str,
    *,
    n_collinear_passes: int,
) -> None:
    """Side-by-side cell ρ: L1 seed input to renderer vs after recurrent collinear passes."""
    gs = apply_border_zero(np.asarray(rho_seed, dtype=np.float64), is_border)
    gf = apply_border_zero(np.asarray(rho_final, dtype=np.float64), is_border)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), facecolor=VIZ.BG)
    cmap = rho_heatmap_cmap()
    vmax = max(
        float(np.max(gs)) if gs.size else 0.0,
        float(np.max(gf)) if gf.size else 0.0,
        float(VIZ.EPS),
    )
    norm_seed = np.clip(gs / vmax, 0.0, 1.0)
    norm_fin = np.clip(gf / vmax, 0.0, 1.0)
    titles = (
        r"$\rho_{\mathrm{seed}}$ (cell, pre-collinear)",
        r"$\rho_{\mathrm{final}}$ (after %d collinear passes)" % int(n_collinear_passes),
    )
    for ax, arr, title in zip(axes, (norm_seed, norm_fin), titles):
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=9, color=VIZ.FG, fontfamily="monospace")
        ax.axis("off")
    fig.suptitle(
        "cell $\\rho$: seed vs after recurrent collinear modulation "
        f"(shared color scale, max={vmax:.4g})",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_infer_rho_seed_final_hist_cdf(
    rho_seed: np.ndarray,
    rho_final: np.ndarray,
    is_border: np.ndarray,
    out_path: str,
    *,
    n_bins: int = 64,
) -> None:
    """Interior histograms + empirical CDFs for ρ_seed and ρ_final (common ρ range)."""
    ib = np.asarray(is_border, dtype=bool)
    gs = apply_border_zero(np.asarray(rho_seed, dtype=np.float64), is_border)
    gf = apply_border_zero(np.asarray(rho_final, dtype=np.float64), is_border)
    flat_s = gs.ravel()[~ib.ravel()]
    flat_f = gf.ravel()[~ib.ravel()]
    mx_s = float(np.max(flat_s)) if flat_s.size else 0.0
    mx_f = float(np.max(flat_f)) if flat_f.size else 0.0
    xmax = max(mx_s, mx_f, float(VIZ.EPS))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), facecolor=VIZ.BG)
    fig.suptitle(
        r"Interior $\rho_{\mathrm{seed}}$ vs $\rho_{\mathrm{final}}$ — histograms & CDFs "
        f"(common range [0, {xmax:.4g}])",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )

    rows = (
        (flat_s, r"$\rho_{\mathrm{seed}}$ (pre-collinear)"),
        (flat_f, r"$\rho_{\mathrm{final}}$ (post-collinear)"),
    )
    for row_i, (flat, row_title) in enumerate(rows):
        ax_h = axes[row_i, 0]
        ax_c = axes[row_i, 1]
        ax_h.set_facecolor(VIZ.PANEL_BG)
        ax_c.set_facecolor(VIZ.PANEL_BG)
        n = int(flat.size)
        if n > 0:
            ax_h.hist(
                flat,
                bins=n_bins,
                range=(0.0, xmax),
                color=VIZ.ACCENT,
                edgecolor=VIZ.PANEL_BG,
                linewidth=0.3,
            )
            xs = np.sort(flat.astype(np.float64, copy=False))
            ys = np.arange(1, n + 1, dtype=np.float64)
            ax_c.plot(xs, ys, color=VIZ.FG, linewidth=1.2)
        ax_h.set_xlim(0.0, xmax)
        ax_h.set_title(
            f"{row_title}\nvs count  n={n}",
            fontsize=8,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        ax_h.set_ylabel("count", fontsize=8, color=VIZ.FG)
        ax_h.tick_params(colors=VIZ.FG, labelsize=7)
        for s in ax_h.spines.values():
            s.set_color(VIZ.ACCENT)

        ax_c.set_xlim(0.0, xmax)
        ax_c.set_ylim(0.0, float(max(n, 1)))
        ax_c.set_xlabel(r"$\rho$", fontsize=8, color=VIZ.FG)
        ax_c.set_ylabel("cumulative count", fontsize=8, color=VIZ.FG)
        ax_c.set_title(
            "empirical CDF (unnormalized y)",
            fontsize=8,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        ax_c.grid(True, alpha=0.25, color=VIZ.ACCENT)
        ax_c.tick_params(colors=VIZ.FG, labelsize=7)
        for s in ax_c.spines.values():
            s.set_color(VIZ.ACCENT)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_pixel_projection_and_edges(
    pix_proj: np.ndarray,
    edges_u8: np.ndarray,
    out_path,
    suptitle="learned pixel projection · edges (thresholded)",
):

    m = max(float(pix_proj.max()), VIZ.EPS)
    cmap = rho_heatmap_cmap()
    rgba = cmap(np.clip(pix_proj / m, 0.0, 1.0))
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=VIZ.BG)
    fig.suptitle(suptitle, fontsize=10, color=VIZ.FG, fontfamily="monospace")
    ax0, ax1 = axes
    ax0.set_facecolor(VIZ.PANEL_BG)
    ax0.imshow(rgb)
    ax0.set_title(
        "final ρ (ridge projection)", fontsize=9, color=VIZ.FG, fontfamily="monospace"
    )
    ax0.axis("off")

    ax1.set_facecolor(VIZ.PANEL_BG)
    ax1.imshow(edges_u8, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("thresholded edges", fontsize=9, color=VIZ.FG, fontfamily="monospace")
    ax1.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def save_rho_png(pix_proj: np.ndarray, out_path):

    m = max(float(pix_proj.max()), VIZ.EPS)
    cmap = rho_heatmap_cmap()
    rgba = cmap(np.clip(pix_proj / m, 0.0, 1.0))
    Image.fromarray((rgba[..., :3] * 255).astype(np.uint8), "RGB").save(out_path)


def viz_infer_rho_iters_snapshot(
    panels: list[tuple[int, np.ndarray]],
    out_path: str,
    *,
    n_collinear_passes: int,
) -> None:
    """Montage of pixel $\\rho$ from the renderer's cell→pixel bilinear path at
    up to five milestones along the recurrent collinear recurrence.
    """
    if not panels:
        return
    panels = sorted(panels, key=lambda x: x[0])
    vmax = max(float(np.max(p[1])) for p in panels)
    vmax = max(vmax, VIZ.EPS)
    cmap = rho_heatmap_cmap()
    n = len(panels)
    fig_w = max(12.0, 3.2 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.8), facecolor=VIZ.BG)
    if n == 1:
        axes = np.asarray([axes])
    fig.suptitle(
        (
            rf"$\rho$ pixel projection (renderer bilinear) vs collinear passes "
            rf"($N={int(n_collinear_passes)}$ total passes; "
            rf"up to 5 timeline samples)"
        ),
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    for ax, (t_after, pix) in zip(axes.ravel(), panels):
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.axis("off")
        rgba = cmap(
            np.clip(np.asarray(pix, dtype=np.float64) / vmax, 0.0, 1.0)
        )
        ax.imshow(rgba[..., :3])
        if int(t_after) == 0:
            title = r"$\rho_{\mathrm{pix}}$ (pre-collinear)"
        else:
            title = rf"$\rho_{{\mathrm{{pix}}}}$ after pass {int(t_after)}"
        ax.set_title(title, fontsize=8, color=VIZ.FG, fontfamily="monospace")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_infer_l0_pinwheel(
    h_np: np.ndarray, img_pinwheel: np.ndarray, out_path: str
) -> None:

    viz_l0_pinwheel(h_np, img_pinwheel, out_path)


def cell_photo_diff_grids(
    cells_flat: dict,
    nH: int,
    nW: int,
    *,
    branch: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed luminance and chrominance differences across the cut (branch 0 for L2)."""
    b = int(branch)
    Lp = cells_flat["L_plus"].detach().cpu().numpy().reshape(nH, nW, 2)[:, :, b]
    Lm = cells_flat["L_minus"].detach().cpu().numpy().reshape(nH, nW, 2)[:, :, b]
    Cp = cells_flat["C_plus"].detach().cpu().numpy().reshape(nH, nW, 2, 3)[:, :, b, :]
    Cm = cells_flat["C_minus"].detach().cpu().numpy().reshape(nH, nW, 2, 3)[:, :, b, :]
    dL = Lp.astype(np.float64) - Lm.astype(np.float64)
    dC_norm = np.linalg.norm(
        Cp.astype(np.float64) - Cm.astype(np.float64),
        axis=-1,
    )
    return dL, dC_norm


def _cell_signal_grid(
    cells_flat: dict,
    key: str,
    nH: int,
    nW: int,
    *,
    branch: int = 0,
) -> np.ndarray:
    """Extract a per-cell scalar signal from cells_flat by key."""
    b = int(branch)
    if key in cells_flat:
        s = cells_flat[key]
        if isinstance(s, torch.Tensor):
            s = s.detach().cpu().numpy()
        s = np.asarray(s, dtype=np.float64)
        if s.ndim == 3:
            # (nH, nW, K) — already spatial
            return s[:, :, b]
        if s.ndim == 2:
            N = nH * nW
            if s.shape[0] == N and s.shape[1] > 1:
                # (N, K) — flat with branch dim
                return s[:, b].reshape(nH, nW)
            return s.reshape(nH, nW)
        return s.reshape(nH, nW)
    return np.zeros((nH, nW), dtype=np.float64)


def cell_s_lum_grid(cells_flat, nH, nW, *, branch=0):
    """L1 luminance separability s_lum per cell."""
    return _cell_signal_grid(cells_flat, "s_lum", nH, nW, branch=branch)


def cell_s_chr_grid(cells_flat, nH, nW, *, branch=0):
    """L1 chrominance separability s_chr per cell."""
    return _cell_signal_grid(cells_flat, "s_chr", nH, nW, branch=branch)


def viz_infer_cell_photo(
    cells_flat: dict,
    nH: int,
    nW: int,
    is_border: np.ndarray,
    out_path: str,
    *,
    branch: int = 0,
) -> None:
    """ΔL, ‖ΔC‖, s_lum, s_chr, s_photo per cell (branch 0)."""
    dL, dC_norm = cell_photo_diff_grids(cells_flat, nH, nW, branch=branch)
    s_lum = cell_s_lum_grid(cells_flat, nH, nW, branch=branch)
    s_chr = cell_s_chr_grid(cells_flat, nH, nW, branch=branch)
    s_photo = _cell_signal_grid(cells_flat, "s_photo", nH, nW, branch=branch)
    interior = ~np.asarray(is_border, dtype=bool)
    panels = [
        (apply_border_zero(dL, is_border), r"$\Delta L = L^+ - L^-$", "coolwarm", None),
        (
            apply_border_zero(dC_norm, is_border),
            r"$\|\Delta C\| = \|C^+ - C^-\|$",
            "magma",
            None,
        ),
        (
            apply_border_zero(s_lum, is_border),
            r"$s_{\mathrm{lum}}$ ($\Delta L^2/(\eta_{\mathrm{lum}}^2+\Delta L^2)$, same $\eta$ as L0)",
            "viridis",
            (0.0, 1.0),
        ),
        (
            apply_border_zero(s_chr, is_border),
            r"$s_{\mathrm{chr}}$ ($\|\Delta C\|^2/(\eta_{\mathrm{chr}}^2+\|\Delta C\|^2)$, same $\eta$ as L0)",
            "viridis",
            (0.0, 1.0),
        ),
        (
            apply_border_zero(s_photo, is_border),
            r"$s_{\mathrm{photo}}$ (OR combination of lum/chr asymmetries)",
            "viridis",
            (0.0, 1.0),
        ),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(5.5, 5.2 * 5), facecolor=VIZ.BG)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (arr, title_base, cmap, vlim) in zip(axes_flat, panels):
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.axis("off")
        if vlim is not None:
            vmin, vmax = vlim
        elif cmap == "coolwarm":
            if interior.any():
                v = np.abs(arr[interior])
                vmax = float(np.max(v)) if v.size else 1.0
            else:
                vmax = 1.0
            vmax = max(vmax, 1e-12)
            vmin = -vmax
        else:
            vmin, vmax = _interior_vmin_vmax(arr, interior)
        im = ax.imshow(
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(
            f"{title_base}  (branch {branch})\n"
            f"interior min={vmin:.4g}  max={vmax:.4g}",
            fontsize=9,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.82)
        cbar.outline.set_edgecolor(VIZ.ACCENT)
        cbar.ax.tick_params(colors=VIZ.FG, labelsize=7)

    fig.suptitle(
        rf"L1 photometry — $\Delta L$, $\|\Delta C\|$, $s_{{\mathrm{{lum}}}}$, "
        rf"$s_{{\mathrm{{chr}}}}$, $s_{{\mathrm{{photo}}}}$ "
        rf"(z₂-centroid partition, branch {branch})",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)


def viz_infer_l1_lambdas(
    lam: np.ndarray,
    lam3: np.ndarray,
    is_border: np.ndarray,
    out_path: str,
) -> None:

    viz_l1_lambdas_three(
        lam[:, :, 0],
        lam[:, :, 1],
        lam3,
        is_border,
        out_path,
        suptitle=r"L1 $\lambda$ (not the L2 seed; seed is $\lambda_k/(\lambda_1+\lambda_2+\eta_z)$)",
    )


def viz_l2_bimodality_per_iter(
    bimodality: list[float] | np.ndarray,
    out_path: str,
    *,
    suptitle: str = "L2 refine — bimodality vs step",
) -> bool:
    """Plot Σ_i ρ_i(1−ρ_i) over the full cell map at each L2 step (0 = before refine)."""
    y = np.asarray(bimodality, dtype=np.float64).ravel()
    if y.size == 0:
        return False
    x = np.arange(y.size, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), facecolor=VIZ.BG)
    ax.set_facecolor(VIZ.PANEL_BG)
    ax.plot(x, y, "o-", color="#66aaff", lw=1.4, ms=5, mfc="#88ccff", mec="#4488cc")
    ax.set_xlabel(
        "L2 step (0 = ρ seed before refine)",
        fontsize=9,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    ax.set_ylabel(
        r"$\sum_i \rho_i(1-\rho_i)$",
        fontsize=9,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    ax.tick_params(colors=VIZ.FG, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.grid(True, alpha=0.2, color="#555")
    ax.set_xticks(x)
    fig.suptitle(suptitle, fontsize=10, color=VIZ.FG, fontfamily="monospace", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)
    return True


def viz_infer_texture_gate(
    texture_gate: np.ndarray | None,
    is_border: np.ndarray,
    out_path: str,
) -> bool:

    if texture_gate is None:
        return False
    tg = np.asarray(texture_gate)
    if tg.ndim == 3 and tg.shape[-1] == 2:
        panels = [tg[:, :, 0], tg[:, :, 1]]
        labels = [
            r"tex$_0$ (branch-0 gate)",
            r"tex$_1$ (branch-1 gate)",
        ]
        layout = (1, 2)
    else:
        panels = [tg]
        labels = [r"tex"]
        layout = (1, 1)
    viz_rho_branch_grid(
        panels,
        labels,
        is_border,
        out_path,
        suptitle="Texture gate (last refine)",
        layout_rows_cols=layout,
    )
    return True


def viz_infer_l2_facilitation_factors(
    surface_diags: dict | None,
    is_border: np.ndarray,
    out_path: str,
) -> bool:

    if surface_diags is None:
        return False
    rho_coll_nb = surface_diags.get(
        "rho_coll",
        surface_diags.get("support_B", surface_diags.get("sigma_mag")),
    )

    panels: list = []
    labels: list = []

    coll_added = False
    if rho_coll_nb is not None:
        panels.append(rho_coll_nb)
        rc = np.asarray(rho_coll_nb)
        if rc.ndim == 2 and rc.shape == is_border.shape:
            sub_b = r"cell grid"
        else:
            sub_b = r"tile $\times$ member"
        labels.append(rf"$\rho_{{\mathrm{{coll}}}}$ (collinear readback, {sub_b})")
        coll_added = True

    if not coll_added:
        return False

    n = len(panels)
    if n <= 2:
        layout = (1, n)
    elif n == 3:
        layout = (1, 3)
    else:
        layout = (2, 2)

    sig_suptitle = (
        r"L2 facilitation — collinear readback $\rho_{\mathrm{coll}}$ (last refine)"
    )
    viz_rho_branch_grid(
        panels,
        labels,
        is_border,
        out_path,
        suptitle=sig_suptitle,
        layout_rows_cols=layout,
    )
    return True


def viz_infer_l2_suppression_factors(
    surface_diags: dict | None,
    is_border: np.ndarray,
    out_path: str,
) -> bool:

    if surface_diags is None:
        return False
    iso_pool_nb = surface_diags.get("iso_pool")
    cross_pool_nb = surface_diags.get("cross_pool")

    panels: list = []
    labels: list = []

    if iso_pool_nb is not None:
        panels.append(iso_pool_nb)
        labels.append(r"iso pool $I$ (same θ-bin, cell avg)")
    if cross_pool_nb is not None:
        panels.append(cross_pool_nb)
        labels.append(r"cross pool $C$ (other θ-bins, cell avg)")

    if not panels:
        return False

    n = len(panels)
    layout = (1, n) if n <= 2 else (1, 3)

    sig_suptitle = (
        r"L2 suppression factors — iso pool $I$ and cross pool $C$ (first refine)"
    )
    viz_rho_branch_grid(
        panels,
        labels,
        is_border,
        out_path,
        suptitle=sig_suptitle,
        layout_rows_cols=layout,
    )
    return True


def viz_infer_l2_bin_dynamics(
    surface_diags: dict | None,
    is_border: np.ndarray,
    out_path: str,
    *,
    K: int = 24,
) -> bool:

    if surface_diags is None:
        return False
    D_tile = surface_diags.get("D_tile")
    P_star = surface_diags.get("P_star")
    B_cell = surface_diags.get("B_cell")
    Q_cell = surface_diags.get("Q_cell")
    k_star = surface_diags.get("k_star")
    if D_tile is None or P_star is None or B_cell is None or Q_cell is None:
        return False

    interior = ~np.asarray(is_border, dtype=bool)
    panels_cont = [
        (apply_border_zero(D_tile, is_border), r"tile dominance $D_T$"),
        (apply_border_zero(P_star, is_border), r"winning participation $P^\star$"),
        (apply_border_zero(B_cell, is_border), r"per-cell facilitation $B_{\mathrm{cell}}$"),
        (apply_border_zero(Q_cell, is_border), r"per-cell suppression $Q_{\mathrm{cell}}$"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor=VIZ.BG)
    axes_flat = axes.ravel()
    for ax in axes_flat:
        ax.set_facecolor(VIZ.PANEL_BG)
        ax.axis("off")

    for i, (arr, title) in enumerate(panels_cont):
        ax = axes_flat[i]
        vmin, vmax = _interior_vmin_vmax(arr, interior)
        im = ax.imshow(arr, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8, color=VIZ.FG, fontfamily="monospace")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(colors=VIZ.FG, labelsize=6)

    ax_k = axes_flat[4]
    if k_star is not None:
        kk = np.asarray(k_star, dtype=np.float32).copy()
        kk[is_border] = np.nan
        n_bins = max(2, int(K))
        imk = ax_k.imshow(
            kk,
            cmap=plt.get_cmap("tab20", n_bins),
            vmin=-0.5,
            vmax=float(n_bins) - 0.5,
            interpolation="nearest",
        )
        ax_k.set_title(
            r"winning bin $k^\star$ (per cell, tile vote)",
            fontsize=8,
            color=VIZ.FG,
            fontfamily="monospace",
        )
        cbar_k = fig.colorbar(imk, ax=ax_k, fraction=0.046, pad=0.02)
        cbar_k.ax.tick_params(colors=VIZ.FG, labelsize=6)
    else:
        ax_k.text(
            0.5,
            0.5,
            r"no $k^\star$",
            ha="center",
            va="center",
            color=VIZ.FG,
            transform=ax_k.transAxes,
        )

    fig.suptitle(
        "L2 bin dynamics — dominance, participation, B/Q modulators, committed bin",
        fontsize=10,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)
    return True


def viz_infer_base_edges_overlay(
    base_rgb: np.ndarray,
    edges_u8: np.ndarray,
    out_path: str,
    *,
    edge_rgb: tuple[float, float, float] = (0.0, 1.0, 0.95),
    edge_weight: float = 0.88,
) -> None:

    base = np.clip(np.asarray(base_rgb, dtype=np.float32), 0.0, 1.0)
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("base_rgb must be H×W×3")
    em = np.asarray(edges_u8)
    if em.ndim == 3 and em.shape[2] == 1:
        em = em[..., 0]
    mask = em.astype(np.float32) >= 127.5
    ec = np.array(edge_rgb, dtype=np.float32).reshape(1, 1, 3)
    w = float(np.clip(edge_weight, 0.0, 1.0))
    out = base.copy()
    if mask.any():
        m = mask[..., np.newaxis]
        out = np.where(m, (1.0 - w) * base + w * ec, base)
    Image.fromarray((np.clip(out, 0.0, 1.0) * 255).astype(np.uint8), "RGB").save(
        out_path
    )


def viz_infer_pixel_boundary(
    bmap: np.ndarray,
    out_path: str,
    *,
    threshold: float,
) -> None:

    edges_u8 = ((np.asarray(bmap) >= threshold).astype(np.uint8)) * 255
    viz_pixel_projection_and_edges(
        bmap,
        edges_u8,
        out_path,
        suptitle=f"final ρ (ridge render) · edges (t={threshold:.4f})",
    )


def viz_infer_shape_readout(
    theta_map: np.ndarray,
    bmap: np.ndarray,
    *,
    threshold: float,
    theta_bins: int = 12,
    out_path: str,
) -> None:

    th = np.asarray(theta_map, dtype=np.float32)
    edge_mask = np.asarray(bmap, dtype=np.float32) >= float(threshold)
    valid = edge_mask & np.isfinite(th)
    rgb = np.zeros((*th.shape, 3), dtype=np.float32)
    if valid.any():
        n_bins = max(2, int(theta_bins))

        th_mod = np.mod(th[valid], np.pi)
        bidx = np.floor((th_mod / np.pi) * n_bins).astype(np.int64)
        bidx = np.clip(bidx, 0, n_bins - 1)
        palette = plt.get_cmap("tab20", n_bins)(np.arange(n_bins))[..., :3]
        rgb_valid = palette[bidx]
        rgb[valid] = rgb_valid.astype(np.float32)
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), "RGB").save(
        out_path
    )
