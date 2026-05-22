r"""infer.py — harmonic-contour-integration single-image inference.

Pipeline: L0 → L1 (eigendecomposition on patches) → ``ρ_seed`` (NR pool on
``λ₁/(z₀+η_z)``, tile-interior mask) → renderer splat to pixels, optional ridge
NMS along θ, then thresholded edge PNG (default: Otsu on the soft map; use ``-t``
for a fixed threshold in ``[0, 1]``).
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import torch
from PIL import Image

from params import L0, L1, RENDER, INFER, SEED
from hce.L0 import compute_l0_rgb, compute_interior
from hce.L1 import z_from_l0_harmonics, pad_for_patch_grid, run_l1
from hce.renderer import (
    compute_render_features,
    render_boundary_map_torch,
    proj_to_device,
    ridge_nms,
    upgrade_renderer_state_dict,
)
from hce.diagnostics_viz import (
    viz_infer_l0_pinwheel,
    viz_infer_l1_lambdas,
    viz_infer_cell_photo,
    viz_infer_rho_map_hist_cdf,
    save_rho_png,
    viz_infer_shape_readout,
    viz_infer_base_edges_overlay,
)
from train import (
    HarmonicContourE2E,
    build_cells_flat,
    build_l0_pix,
    format_seed_param_lines,
    format_model_param_counts,
    format_renderer_param_lines,
    remap_checkpoint_state_dict,
    report_checkpoint_compatibility,
)


def otsu_threshold_softmap(bmap: np.ndarray, *, nbins: int = 256) -> float:
    """Otsu threshold on ``[0, 1]`` soft map (same idea as OpenCV Canny auto τ on magnitude).

    Returns a value in ``(0, 1)`` suitable for ``bmap >= τ`` binarization.
    """
    x = np.clip(np.asarray(bmap, dtype=np.float64).ravel(), 0.0, 1.0)
    if x.size == 0:
        return 0.5
    hist, bin_edges = np.histogram(x, bins=nbins, range=(0.0, 1.0))
    total = float(hist.sum())
    if total < 1.0:
        return 0.5
    p = hist.astype(np.float64) / total
    omega = np.cumsum(p)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mu = np.cumsum(p * bin_centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    idx = int(np.argmax(sigma_b2))
    t = float(bin_centers[idx])
    return float(min(max(t, 1e-6), 1.0 - 1e-6))


def build_model(ckpt, device):
    m = HarmonicContourE2E(
        r_pool=SEED.R_POOL,
        stride=SEED.STRIDE,
        eps=SEED.EPS,
        eta_z_init=SEED.ETA_Z_INIT,
        render_cell_hidden=RENDER.CELL_HIDDEN,
        render_pixel_hidden=RENDER.PIXEL_HIDDEN,
    )
    sd = remap_checkpoint_state_dict(ckpt["model_state"])
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
    del h, vld, s, h1m, h2m, h2m_lum, h2m_chr
    gc.collect()
    _sync(device)
    timings["l0"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    cells = run_l1(
        z1,
        z2,
        L1.PATCH_SIZE,
        border_mask=bm_t,
        patch_overlap=L1.PATCH_OVERLAP,
        border_patch_max_frac=L1.BORDER_PATCH_MAX_FRAC,
        eps=L1.EPS,
        img=ir_t,
        device=device,
        verbose=False,
    )
    del z1, z2, bm_t, ir_t
    gc.collect()
    cells["is_border"] |= (cells["cy"] + cells["P"] / 2 > H0) | (
        cells["cx"] + cells["P"] / 2 > W0
    )
    _sync(device)
    timings["l1"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    nH, nW = cells["nH"], cells["nW"]
    proj_info = compute_render_features(z2_image, ir_p, cells, bm_np, eps=SEED.EPS)
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


def _infer_seed_and_render(
    model,
    prep,
    device,
    *,
    collect_diags: bool = False,
    apply_ridge_nms: bool = True,
    timings: dict[str, float] | None = None,
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

    if timings is not None:
        _sync(device)
        t0 = time.perf_counter()
    with torch.no_grad():
        rho_out, branch, _, supp_nb, _, _, surface_diags = model.seed(
            cells_flat=cf_dev,
            return_surface_diags=collect_diags,
        )
    if timings is not None:
        _sync(device)
        timings["seed"] = time.perf_counter() - t0
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

    bmap = bmap_t.cpu().numpy()[:H0, :W0]
    theta_map = theta_t.cpu().numpy()[:H0, :W0]
    if apply_ridge_nms:
        bmap = ridge_nms(bmap, theta=theta_map)
    rho_post_grid = rho_out.cpu().numpy().reshape(nH, nW)
    branch_grid = branch.cpu().numpy().reshape(nH, nW)
    supp_nb_np = supp_nb.cpu().numpy().reshape(nH, nW)
    return bmap, theta_map, rho_out, branch_grid, rho_post_grid, surface_diags, supp_nb_np


def forward_with_diagnostics(
    model,
    prep,
    device,
    *,
    collect_diags,
    apply_ridge_nms=True,
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
    ) = _infer_seed_and_render(
        model,
        prep,
        device,
        collect_diags=collect_diags,
        apply_ridge_nms=apply_ridge_nms,
        timings=timings,
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
    model, n_tot, n_seed, n_r, device, diagnostics,
):
    s = model.seed
    r = model.renderer
    lines = [
        "Model",
        f"  params:      {n_tot} ({n_seed} seed NR, {n_r} renderer)",
        f"  device:      {device}",
        f"  diagnostics: {diagnostics}",
        "",
        "L1 seed",
        *format_seed_param_lines(s),
        "",
        "Renderer",
        *format_renderer_param_lines(r),
    ]
    return lines


def main():
    ap = argparse.ArgumentParser(
        description="harmonic-contour-integration single-image inference",
    )
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("--input_dir", default="data/infer")
    ap.add_argument("--output_dir", default="output/results")
    ap.add_argument("--model", default="output/checkpoints/intermediate.pt")
    ap.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        metavar="τ",
        help=(
            "If set, use this fixed edge threshold on the soft map in [0, 1]. "
            "If omitted, τ is chosen by Otsu on the soft map (same class of rule as "
            "OpenCV Canny when thresholds are computed automatically)."
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
        "-d",
        "--diagnostics",
        action="store_true",
        help="Save additional diagnostics: base, l0_pinwheel, l1_lambdas, photo, "
        "rho (map+histogram+CDF), render_softmap, render_theta_bins, "
        "overlay (base RGB with thresholded edges).",
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
    n_tot, n_seed, n_r = format_model_param_counts(model)

    if args.verbose:
        for line in _format_model_summary(
            model, n_tot, n_seed, n_r, device, args.diagnostics,
        ):
            print(line)
        print()

    img_path = os.path.join(args.input_dir, args.image)
    prep, prep_t = run_l0_l1(img_path, device)

    collect_diags = args.verbose or args.diagnostics
    (bmap, theta_map, rho_post, branch_grid, is_border, diags, _, fwd_t) = (
        forward_with_diagnostics(
            model,
            prep,
            device,
            collect_diags=collect_diags,
            apply_ridge_nms=args.ridge_nms,
        )
    )

    eta_z = float(model.seed.eta_z.detach().cpu().item())

    if args.verbose and diags is not None and "iter_stats" in diags:
        stats = diags["iter_stats"]
        if stats:
            st0 = stats[0]
            print("ρ (seed path)")
            if "n_tiles" in st0:
                print(f"  n_tiles={st0['n_tiles']}")
            if "rho_mean" in st0:
                print(f"  rho_mean={st0['rho_mean']:.6f}")
            if "rho_max" in st0:
                print(f"  rho_max={st0['rho_max']:.6f}")
            if "mid_band_frac" in st0:
                print(f"  midband={st0['mid_band_frac']:.6f}")

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

        p_photo = os.path.join(od, f"{stem}_photo.png")
        viz_infer_cell_photo(
            prep["cells_flat"],
            prep["nH"],
            prep["nW"],
            is_border,
            p_photo,
        )
        saved_files.append(p_photo)

        p_rho = os.path.join(od, f"{stem}_rho.png")
        viz_infer_rho_map_hist_cdf(rho_post, is_border, p_rho, eta_z=eta_z)
        saved_files.append(p_rho)
    bmap_np = np.asarray(bmap, dtype=np.float64)
    if args.threshold is not None:
        threshold = float(args.threshold)
        if not (0.0 <= threshold <= 1.0):
            raise SystemExit(f"error: -t/--threshold must be in [0, 1], got {threshold}")
        thresh_mode = "fixed"
    else:
        threshold = otsu_threshold_softmap(bmap_np)
        thresh_mode = "Otsu"

    edges_u8 = ((bmap_np >= threshold).astype(np.uint8)) * 255
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
        + fwd_t["seed"]
        + fwd_t["render"]
    )
    elapsed_s = inference_s + save_s

    if args.verbose:
        print("Timings")
        print(f"  L0={prep_t['l0']:.3f}s  L1={prep_t['l1']:.3f}s  "
              f"render_pre={prep_t['render_precompute']:.3f}s  "
              f"seed={fwd_t['seed']:.3f}s  render={fwd_t['render']:.3f}s")
        print(f"  inference={inference_s:.3f}s  save={save_s:.3f}s  elapsed={elapsed_s:.3f}s")
        print(
            f"  ridge_nms={int(args.ridge_nms)}  threshold={thresh_mode}  τ={threshold:.4f}"
        )
        print()

    print(f"Outputs -> {args.output_dir}")
    for p in saved_files:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()