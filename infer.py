r"""infer.py — STRIATE single-image inference.

Pipeline: L0 → L1 K-bin projection → L2 ρ-space cell-grid dynamics (optional; use
``--no-dynamics`` to skip L2 and feed ρ_seed through the renderer)
→ splat boundary map ($\hat B = \bar\rho \cdot \mathrm{gate}$),
optional ridge NMS along θ, then thresholded edge PNG (default τ = 0.5).
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import torch
from PIL import Image

from params import L0, L1, L2, RENDER, INFER, TRAIN
from hci.L0 import compute_l0_rgb, compute_interior
from hci.L1 import z_from_l0_harmonics, pad_for_patch_grid, run_l1
from hci.renderer import (
    compute_render_features,
    render_boundary_map_torch,
    proj_to_device,
    ridge_nms,
    upgrade_renderer_state_dict,
)
from hci.diagnostics_viz import (
    viz_infer_l0_pinwheel,
    viz_infer_l1_lambdas,
    viz_infer_cell_rho_maps,
    viz_infer_rho_hist_cdf,
    viz_l2_bimodality_per_iter,
    viz_infer_l2_bin_dynamics,
    viz_infer_l2_facilitation_factors,
    viz_infer_l2_geometry,
    viz_infer_l2_suppression_factors,
    save_rho_png,
    viz_infer_shape_readout,
    viz_infer_base_edges_overlay,
    viz_infer_iters_snapshot,
)
from hci.L2 import (
    l2_snapshot_steps,
    rho_seed_from_lam1_z0,
)
from train import (
    StriateE2E,
    build_cells_flat,
    build_l0_pix,
    format_l2_param_lines,
    format_model_param_counts,
    format_renderer_param_lines,
    report_checkpoint_compatibility,
)


def _rho_out_seed_only(
    model,
    cf_dev: dict,
    device: torch.device,
    *,
    collect_diags: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict | None]:
    """Match TileDynamics output with T=0: ρ_seed on interior cells (no L2 refine)."""
    d = model.dynamics
    nH, nW = int(cf_dev["nH"]), int(cf_dev["nW"])
    N = nH * nW
    lam1 = cf_dev["lam"][..., 0].to(device)
    z0 = cf_dev["z0"].to(device)
    is_border = cf_dev["is_border"].to(device)
    rho_seed = rho_seed_from_lam1_z0(
        lam1, z0, d.eta_z, is_border, d.eps,
    )

    interior = (~is_border).to(dtype=rho_seed.dtype)
    rho_out = rho_seed * interior

    branch = torch.zeros(N, device=device, dtype=torch.long)
    z1 = torch.zeros(N, 1, device=device, dtype=rho_out.dtype)

    surface_diags = None
    if collect_diags:
        ra = rho_out[~is_border]
        surface_diags = {
            "iter_stats": [{
                "rho_mean": float(rho_out.mean().detach()),
                "rho_max": float(rho_out.max().detach()),
                "mid_band_frac": float(
                    ((ra > 0.3) & (ra < 0.7)).float().mean().detach()
                ) if ra.numel() else 0.0,
                "n_interior": int((~is_border).sum().item()),
            }],
            "no_dynamics": True,
        }
    return rho_out, branch, z1, surface_diags


def build_model(ckpt, device):
    m = StriateE2E(
        r_fac_pool=L2.R_FAC_POOL,
        r_sup_pool=L2.R_SUP_POOL,
        K=L2.K,
        t_refine=L2.T_REFINE,
        eps=L2.EPS,
        eta_z_init=L2.ETA_Z_INIT,
        render_cell_hidden=RENDER.CELL_HIDDEN,
        render_pixel_hidden=RENDER.PIXEL_HIDDEN,
    )
    sd = ckpt["model_state"]
    sd = upgrade_renderer_state_dict(sd, prefix="renderer.")
    incompatible = m.load_state_dict(sd, strict=False)
    report_checkpoint_compatibility(incompatible, context="infer build_model")
    return m.to(device).eval()


def _sync(device):

    if device.type == "cuda":
        torch.cuda.synchronize()


def run_l0_l1(img_path, device):

    timings = {}

    _sync(device)
    t0 = time.perf_counter()
    ir_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    ir_p, H0, W0 = pad_for_patch_grid(ir_np, L1.PATCH_SIZE, L1.PATCH_OVERLAP)
    del ir_np
    gc.collect()

    ir_t = torch.from_numpy(ir_p).to(device)
    h, vld, _, _, _, s, h1m, h2m, h2m_lum, h2m_chr = compute_l0_rgb(
        ir_t,
        eta_lum=L0.ETA_LUM,
        eta_chr=L0.ETA_CHR,
        gamma=L0.GAMMA,
        offsets=L0.OFFSETS,
    )
    h_np = h.cpu().numpy()
    img_pinwheel = np.clip(ir_p[:H0, :W0].copy(), 0.0, 1.0)

    bm_t = ~compute_interior(ir_p.shape[0], ir_p.shape[1], device)
    z1, z2 = z_from_l0_harmonics(s, bm_t)

    s_np = s.cpu().numpy()
    bm_np = bm_t.cpu().numpy()
    z2_image = (s_np[..., 2] + 1j * s_np[..., 3]).astype(np.complex64)
    z2_image[bm_np] = 0.0
    l0_pix = build_l0_pix(
        s_np, h1m, h2m, bm_np, h2m_lum=h2m_lum, h2m_chr=h2m_chr,
    )
    del h, vld, s, h1m, h2m_lum, h2m_chr
    gc.collect()
    _sync(device)
    timings["l0"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    cells = run_l1(
        h2m,
        z1,
        z2,
        L1.PATCH_SIZE,
        border_mask=bm_t,
        patch_overlap=L1.PATCH_OVERLAP,
        border_patch_max_frac=L1.BORDER_PATCH_MAX_FRAC,
        eps=L1.EPS,
        K=L1.K,
        cos_power=L1.COL_COS_POWER,
        device=device,
        verbose=False,
    )
    del h2m, z1, z2, bm_t
    gc.collect()
    cells["is_border"] |= (cells["cy"] + cells["P"] / 2 > H0) | (
        cells["cx"] + cells["P"] / 2 > W0
    )
    _sync(device)
    timings["l1"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    nH, nW = cells["nH"], cells["nW"]
    proj_info = compute_render_features(z2_image, ir_p, cells, bm_np, eps=L2.EPS)
    del z2_image, bm_np
    gc.collect()

    N = nH * nW
    cells_flat = build_cells_flat(cells)

    lam_grid = cells["lam"].astype(np.float64, copy=True)
    lam3_grid = cells["lam3"].astype(np.float64, copy=True)
    z0_grid = cells["z0"].astype(np.float64, copy=True)
    is_border_grid = cells["is_border"].copy()
    Hp, Wp = ir_p.shape[:2]
    del cells, ir_p
    gc.collect()
    _sync(device)
    timings["render_precompute"] = time.perf_counter() - t2

    prep = {
        "proj_info": proj_info,
        "cells_flat": cells_flat,
        "l0_pix": l0_pix,
        "Hp": Hp,
        "Wp": Wp,
        "H0": H0,
        "W0": W0,
        "nH": nH,
        "nW": nW,
        "lam_grid": lam_grid,
        "lam3_grid": lam3_grid,
        "z0_grid": z0_grid,
        "is_border_grid": is_border_grid,
        "h_np": h_np,
        "img_pinwheel": img_pinwheel,
    }
    return prep, timings


def _infer_l2_and_render(
    model,
    prep,
    device,
    *,
    n_l2_iters: int,
    collect_diags: bool = False,
    apply_ridge_nms: bool = True,
    infer_l2_kw: dict | None = None,
    timings: dict[str, float] | None = None,
    no_dynamics: bool = False,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, np.ndarray, dict | None, np.ndarray]:
    H0, W0 = prep["H0"], prep["W0"]
    Hp, Wp = prep["Hp"], prep["Wp"]
    nH, nW = prep["nH"], prep["nW"]

    cf_dev = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in prep["cells_flat"].items()
    }
    l0_dev = {k: v.to(device) for k, v in prep["l0_pix"].items()}
    proj_dev = proj_to_device(prep["proj_info"], device)
    l2_kw = {} if infer_l2_kw is None else infer_l2_kw

    if no_dynamics:
        if timings is not None:
            _sync(device)
            t0 = time.perf_counter()
        with torch.no_grad():
            rho_out, branch, supp_nb, surface_diags = _rho_out_seed_only(
                model, cf_dev, device, collect_diags=collect_diags,
            )
        if timings is not None:
            _sync(device)
            timings["l2"] = time.perf_counter() - t0
            t1 = time.perf_counter()
        with torch.no_grad():
            bmap_t, theta_t = render_boundary_map_torch(
                rho_out,
                proj_dev,
                model.renderer,
                cf_dev,
                Hp,
                Wp,
                l0_dev,
                eps=model.render_eps,
                training=False,
                branch_pick=branch.reshape(-1).long(),
                content_h=H0,
                content_w=W0,
                return_dominant_theta=True,
            )
        if timings is not None:
            _sync(device)
            timings["render"] = time.perf_counter() - t1
    else:
        t_refine_saved = int(model.dynamics.T_refine)
        model.dynamics.T_refine = int(n_l2_iters)
        try:
            with torch.no_grad():
                if timings is not None:
                    _sync(device)
                    t0 = time.perf_counter()
                rho_out, branch, _, _, supp_nb, _, surface_diags = (
                    model.dynamics(
                        cells_flat=cf_dev,
                        return_surface_diags=collect_diags,
                        **l2_kw,
                    )
                )
                if timings is not None:
                    _sync(device)
                    timings["l2"] = time.perf_counter() - t0
                    t1 = time.perf_counter()
                bmap_t, theta_t = render_boundary_map_torch(
                    rho_out,
                    proj_dev,
                    model.renderer,
                    cf_dev,
                    Hp,
                    Wp,
                    l0_dev,
                    eps=model.render_eps,
                    training=False,
                    branch_pick=branch.reshape(-1).long(),
                    content_h=H0,
                    content_w=W0,
                    return_dominant_theta=True,
                )
                if timings is not None:
                    _sync(device)
                    timings["render"] = time.perf_counter() - t1
        finally:
            model.dynamics.T_refine = t_refine_saved

    bmap = bmap_t.cpu().numpy()[:H0, :W0]
    theta_map = theta_t.cpu().numpy()[:H0, :W0]
    if apply_ridge_nms:
        bmap = ridge_nms(bmap, theta=theta_map)
    rho_post_grid = rho_out.cpu().numpy().reshape(nH, nW)
    branch_grid = branch.cpu().numpy().reshape(nH, nW)
    supp_nb_np = supp_nb.cpu().numpy().reshape(nH, nW)
    return bmap, theta_map, rho_out, branch_grid, rho_post_grid, surface_diags, supp_nb_np


def collect_iters_snapshot_softmaps(
    model,
    prep,
    device,
    iters: int,
    bmap_final: np.ndarray,
    *,
    apply_ridge_nms: bool = True,
    max_snapshots: int = TRAIN.L2_SNAPSHOT_MAX,
    no_dynamics: bool = False,
) -> tuple[list[int], list[np.ndarray]]:
    """Softmaps at up to max_snapshots L2 steps evenly spaced in [0, iters]."""
    if no_dynamics:
        return [0], [np.asarray(bmap_final, dtype=np.float64)]
    steps = l2_snapshot_steps(iters, max_snapshots=max_snapshots)
    cache: dict[int, np.ndarray] = {int(iters): np.asarray(bmap_final, dtype=np.float64)}
    for t in sorted(set(steps)):
        if t in cache:
            continue
        bmap, *_ = _infer_l2_and_render(
            model,
            prep,
            device,
            n_l2_iters=t,
            collect_diags=False,
            apply_ridge_nms=apply_ridge_nms,
            no_dynamics=no_dynamics,
        )
        cache[t] = bmap
    return steps, [cache[t] for t in steps]


def forward_with_diagnostics(
    model,
    prep,
    device,
    *,
    infer_l2_kw,
    collect_diags,
    apply_ridge_nms=True,
    no_dynamics: bool = False,
):

    timings: dict[str, float] = {}
    (
        bmap,
        theta_map,
        _rho_out,
        branch_grid,
        rho_post_grid,
        surface_diags,
        supp_nb_np,
    ) = _infer_l2_and_render(
        model,
        prep,
        device,
        n_l2_iters=int(model.dynamics.T_refine),
        collect_diags=collect_diags,
        apply_ridge_nms=apply_ridge_nms,
        infer_l2_kw=infer_l2_kw,
        timings=timings,
        no_dynamics=no_dynamics,
    )

    return (
        bmap,
        theta_map,
        rho_post_grid,
        branch_grid,
        prep["is_border_grid"],
        surface_diags,
        supp_nb_np,
        timings,
    )


def _format_model_summary(
    model, n_tot, n_dyn, n_r, device, diagnostics, *, l2_iters=None,
):
    d = model.dynamics
    r = model.renderer
    lines = [
        "Model",
        f"  params:      {n_tot} ({n_dyn} dynamics, {n_r} renderer)",
        f"  device:      {device}",
        f"  diagnostics: {diagnostics}",
        "",
        "L2",
        *format_l2_param_lines(d),
    ]
    if l2_iters is not None and int(l2_iters) != int(d.T_refine):
        lines.append(f"  T_infer={int(l2_iters)}")
    lines += ["", "Renderer", *format_renderer_param_lines(r)]
    return lines


def main():
    ap = argparse.ArgumentParser(description="STRIATE single-image inference")
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("--input_dir", default="data/infer")
    ap.add_argument("--output_dir", default="output/results")
    ap.add_argument("--model", default="output/checkpoints/intermediate.pt")
    ap.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=INFER.DEFAULT_THRESHOLD,
        help=(
            "Edge threshold on the soft map in [0, 1] "
            f"(default {INFER.DEFAULT_THRESHOLD})."
        ),
    )
    ap.add_argument(
        "--ridge-nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Directional NMS along renderer θ normal (~1 px ridges). "
        "Use --no-ridge-nms for the raw boundary map.",
    )
    ap.add_argument("--shape_theta_bins", type=int, default=INFER.SHAPE_THETA_BINS)
    ap.add_argument(
        "--l2_iters",
        type=int,
        default=INFER.L2_ITERS,
        metavar="N",
        help=(
            "Number of L2 refine steps (default: checkpoint T_refine). "
            "All interior cells update every step; the loop runs T_refine iterations."
        ),
    )
    ap.add_argument(
        "-d",
        "--diagnostics",
        action="store_true",
        help="Save additional diagnostics: base, l0_pinwheel, l1_lambdas, geometry, "
        "l2_rho_seed_post, l2_rho_hist_cdf, l2_bimodality_per_iter, l2_facilitation_factors, "
        "l2_suppression_factors, l2_bin_dynamics, render_softmap, "
        f"iters_snapshot (up to {TRAIN.L2_SNAPSHOT_MAX} softmaps evenly spaced over L2 steps; "
        "single map if --no-dynamics), "
        "render_theta_bins, overlay (base RGB with thresholded edges).",
    )
    ap.add_argument(
        "--no-dynamics",
        action="store_true",
        help=(
            "Skip L2 refinement: use NR-normalized ρ_seed (same as TileDynamics IC), "
            "same interior mask as TileDynamics, pass through the bilinear renderer; "
            "threshold to edges and save softmap as usual."
        ),
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    stem = os.path.splitext(args.image)[0]

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = build_model(ckpt, device)
    n_tot, n_dyn, n_r = format_model_param_counts(model)
    t_refine_ckpt = int(model.dynamics.T_refine)
    l2_iters = (
        int(args.l2_iters) if args.l2_iters is not None else t_refine_ckpt
    )
    if not args.no_dynamics and l2_iters < 1:
        ap.error("--l2_iters must be >= 1")

    if args.verbose:
        for line in _format_model_summary(
            model, n_tot, n_dyn, n_r, device, args.diagnostics, l2_iters=l2_iters,
        ):
            print(line)
        if args.no_dynamics:
            print("  mode:       --no-dynamics (ρ_seed → renderer, no L2)")
        print()

    img_path = os.path.join(args.input_dir, args.image)
    prep, prep_t = run_l0_l1(img_path, device)

    collect_diags = args.verbose or args.diagnostics
    model.dynamics.T_refine = l2_iters
    try:
        (bmap, theta_map, rho_post, branch_grid, is_border, diags, _, fwd_t) = (
            forward_with_diagnostics(
                model,
                prep,
                device,
                infer_l2_kw={},
                collect_diags=collect_diags,
                apply_ridge_nms=args.ridge_nms,
                no_dynamics=args.no_dynamics,
            )
        )
    finally:
        model.dynamics.T_refine = t_refine_ckpt

    d = model.dynamics
    lam_np = prep["lam_grid"]
    z0_np = prep["z0_grid"]
    nH, nW = int(prep["nH"]), int(prep["nW"])
    lam_t = torch.from_numpy(lam_np[..., 0].astype(np.float32)).to(device)
    z0_t = torch.from_numpy(z0_np.astype(np.float32)).to(device)
    ib_t = torch.from_numpy(np.asarray(is_border, dtype=np.bool_)).to(device)
    rho_seed_vis = (
        rho_seed_from_lam1_z0(lam_t, z0_t, d.eta_z, ib_t, float(d.eps))
        .detach()
        .cpu()
        .numpy()
        .reshape(nH, nW)
    )
    eta_z = float(d.eta_z.detach().cpu().item())

    if args.verbose and args.no_dynamics:
        print("L2: skipped (--no-dynamics); ρ_seed on interior cells → bilinear ρ map.")
    elif args.verbose and diags is not None and "iter_stats" in diags:
        stats = diags["iter_stats"]
        if stats and all(
            k in stats[0]
            for k in (
                "iter",
                "rho_mean",
                "rho_min",
                "rho_max",
                "rho_delta_mean",
                "rho_delta_max",
                "mid_band_frac",
            )
        ):
            print("L2 iter")
            print(
                "  iter   rho_mean   rho_min   rho_max   drho_mean   drho_max   midband"
            )
            for st in stats:
                print(
                    f"  {st['iter']:>4d}   "
                    f"{st['rho_mean']:>9.6f}  "
                    f"{st['rho_min']:>9.6f}  "
                    f"{st['rho_max']:>9.6f}  "
                    f"{st['rho_delta_mean']:>9.6f}  "
                    f"{st['rho_delta_max']:>9.6f}  "
                    f"{st['mid_band_frac']:>9.6f}"
                )
        elif stats:
            st0 = stats[0]
            if "n_interior" in st0:
                print(f"  n_interior={st0['n_interior']}")
            if "rho_mean" in st0:
                print(f"  rho_mean={st0['rho_mean']:.6f}")
            if "rho_max" in st0:
                print(f"  rho_max={st0['rho_max']:.6f}")
            if "mid_band_frac" in st0:
                print(f"  midband={st0['mid_band_frac']:.6f}")

    iters_snapshot: tuple[list[int], list[np.ndarray]] | None = None
    if args.diagnostics:
        iters_snapshot = collect_iters_snapshot_softmaps(
            model,
            prep,
            device,
            l2_iters,
            bmap,
            apply_ridge_nms=args.ridge_nms,
            no_dynamics=args.no_dynamics,
        )

    del model, ckpt
    gc.collect()

    od = args.output_dir
    saved_files = []
    t_save = time.perf_counter()

    if args.diagnostics:
        p_base = os.path.join(od, f"{stem}_base.png")
        Image.open(img_path).convert("RGB").save(p_base)
        saved_files.append(p_base)

        p_pin = os.path.join(od, f"{stem}_l0_pinwheel.png")
        viz_infer_l0_pinwheel(prep["h_np"], prep["img_pinwheel"], p_pin)
        saved_files.append(p_pin)

        p_lam = os.path.join(od, f"{stem}_l1_lambdas.png")
        viz_infer_l1_lambdas(prep["lam_grid"], prep["lam3_grid"], is_border, p_lam)
        saved_files.append(p_lam)

        p_maps = os.path.join(od, f"{stem}_l2_rho_seed_post.png")
        viz_infer_cell_rho_maps(
            rho_seed_vis, rho_post, is_border, p_maps, eta_z=eta_z,
        )
        saved_files.append(p_maps)

        p_hist = os.path.join(od, f"{stem}_l2_rho_hist_cdf.png")
        viz_infer_rho_hist_cdf(rho_seed_vis, rho_post, is_border, p_hist)
        saved_files.append(p_hist)

        if diags is not None and diags.get("geometry"):
            p_geom = os.path.join(od, f"{stem}_geometry.png")
            if viz_infer_l2_geometry(diags, is_border, p_geom):
                saved_files.append(p_geom)

        if diags is not None and diags.get("bimodality_per_iter"):
            p_bimod = os.path.join(od, f"{stem}_l2_bimodality_per_iter.png")
            if viz_l2_bimodality_per_iter(diags["bimodality_per_iter"], p_bimod):
                saved_files.append(p_bimod)

        if diags is not None and (
            "support_B" in diags
            or "sigma_mag" in diags
            or "rho_coll" in diags
        ):
            p_sig = os.path.join(od, f"{stem}_l2_facilitation_factors.png")
            if viz_infer_l2_facilitation_factors(diags, is_border, p_sig):
                saved_files.append(p_sig)

        if diags is not None and (
            "iso_pool" in diags or "cross_pool" in diags
        ):
            p_supp = os.path.join(od, f"{stem}_l2_suppression_factors.png")
            if viz_infer_l2_suppression_factors(diags, is_border, p_supp):
                saved_files.append(p_supp)

        if diags is not None and diags.get("D_tile") is not None:
            p_bin = os.path.join(od, f"{stem}_l2_bin_dynamics.png")
            if viz_infer_l2_bin_dynamics(diags, is_border, p_bin, K=L2.K):
                saved_files.append(p_bin)
    threshold = float(args.threshold)

    edges_u8 = ((np.asarray(bmap) >= threshold).astype(np.uint8)) * 255
    p_pix = os.path.join(od, f"{stem}_edges.png")
    Image.fromarray(edges_u8, mode="L").save(p_pix)
    saved_files.append(p_pix)

    if args.diagnostics:
        p_ov = os.path.join(od, f"{stem}_overlay.png")
        viz_infer_base_edges_overlay(prep["img_pinwheel"], edges_u8, p_ov)
        saved_files.append(p_ov)

        p_ridge = os.path.join(od, f"{stem}_render_softmap.png")
        save_rho_png(bmap, p_ridge)
        saved_files.append(p_ridge)

        if iters_snapshot is not None:
            snap_steps, snap_maps = iters_snapshot
            p_snap = os.path.join(od, f"{stem}_iters_snapshot.png")
            viz_infer_iters_snapshot(snap_maps, snap_steps, p_snap)
            saved_files.append(p_snap)

        p_orient = os.path.join(od, f"{stem}_render_theta_bins.png")
        viz_infer_shape_readout(
            theta_map,
            bmap,
            threshold=threshold,
            theta_bins=args.shape_theta_bins,
            out_path=p_orient,
        )
        saved_files.append(p_orient)
    save_s = time.perf_counter() - t_save

    inference_s = (
        prep_t["l0"]
        + prep_t["l1"]
        + prep_t["render_precompute"]
        + fwd_t["l2"]
        + fwd_t["render"]
    )
    elapsed_s = inference_s + save_s

    if args.verbose:
        print("Timings")
        print(f"  L0={prep_t['l0']:.3f}s  L1={prep_t['l1']:.3f}s  "
              f"render_pre={prep_t['render_precompute']:.3f}s  "
              f"L2={fwd_t['l2']:.3f}s  render={fwd_t['render']:.3f}s")
        print(f"  inference={inference_s:.3f}s  save={save_s:.3f}s  elapsed={elapsed_s:.3f}s")
        print(f"  ridge_nms={int(args.ridge_nms)}  threshold={threshold:.4f}")
        print()

    print(f"Outputs -> {args.output_dir}")
    for p in saved_files:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()