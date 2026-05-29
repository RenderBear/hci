r"""infer.py — STRIATE single-image inference.

Pipeline: L0 → L1 K-bin projection → seed NR (η_z) → splat boundary map
($\hat B = \bar\rho \cdot \mathrm{gate}$), optional ridge NMS along θ,
then thresholded edge PNG (default τ = 0.5).
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import torch
from PIL import Image

from params import L0, L1, SEED, RENDER, INFER
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
    viz_infer_l1_rho_masses,
    viz_infer_cell_rho,
    save_rho_png,
    viz_infer_shape_readout,
    viz_infer_base_edges_overlay,
)
from train import (
    StriateE2E,
    build_cells_flat,
    build_l0_pix,
    format_l1_param_lines,
    format_seed_param_lines,
    format_model_param_summary,
    format_renderer_param_lines,
    report_checkpoint_compatibility,
    upgrade_model_state_dict,
)


def build_model(ckpt, device):
    m = StriateE2E(
        K=SEED.K,
        eps=SEED.EPS,
        render_cell_hidden=RENDER.CELL_HIDDEN,
        render_pixel_hidden=RENDER.PIXEL_HIDDEN,
    )
    sd = upgrade_model_state_dict(ckpt["model_state"])
    sd = upgrade_renderer_state_dict(sd, prefix="renderer.")
    incompatible = m.load_state_dict(sd, strict=False)
    report_checkpoint_compatibility(incompatible, context="infer build_model")
    return m.to(device).eval()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_l0_l1(img_path, device, *, von_mises_kappa: float | None = None):
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
        bin_tuning=L1.COL_BIN_TUNING,
        cos_power=L1.COL_COS_POWER,
        von_mises_kappa=(
            L1.COL_VON_MISES_KAPPA if von_mises_kappa is None else von_mises_kappa
        ),
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
    proj_info = compute_render_features(z2_image, ir_p, cells, bm_np, eps=SEED.EPS)
    del z2_image, bm_np
    gc.collect()

    cells_flat = build_cells_flat(cells)

    rho_peak_grid = cells["rho_peak"].astype(np.float64, copy=True)
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
        "rho_peak_grid": rho_peak_grid,
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

    with torch.no_grad():
        if timings is not None:
            _sync(device)
            t0 = time.perf_counter()
        rho_out, branch, supp_nb, _, _, cf_out, surface_diags = model.seed(
            cells_flat=cf_dev,
            return_surface_diags=collect_diags,
        )
        if timings is not None:
            _sync(device)
            timings["seed"] = time.perf_counter() - t0
            t1 = time.perf_counter()
        bmap_t, theta_t = render_boundary_map_torch(
            rho_out,
            proj_dev,
            model.renderer,
            cf_out,
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


def _format_model_summary(model, device, diagnostics):
    r = model.renderer
    lines = [
        "Model",
        f"  params:      {format_model_param_summary(model)}",
        f"  device:      {device}",
        f"  diagnostics: {diagnostics}",
        "",
        "L1",
        *format_l1_param_lines(model),
        "",
        "Seed",
        *format_seed_param_lines(model.seed),
        "",
        "Renderer",
        *format_renderer_param_lines(r),
    ]
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
        "-d",
        "--diagnostics",
        action="store_true",
        help="Save additional diagnostics: base, l0_pinwheel, l1_rho_masses, "
        "cell_rho, render_softmap, render_theta_bins, overlay.",
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

    if args.verbose:
        for line in _format_model_summary(model, device, args.diagnostics):
            print(line)
        print()

    img_path = os.path.join(args.input_dir, args.image)
    prep, prep_t = run_l0_l1(
        img_path,
        device,
        von_mises_kappa=float(model.l1_von_mises_kappa.detach().cpu()),
    )

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

    if args.verbose and diags is not None and "iter_stats" in diags:
        st0 = diags["iter_stats"][0] if diags["iter_stats"] else {}
        if "n_interior" in st0:
            print(f"  n_interior={st0['n_interior']}")
        if "rho_mean" in st0:
            print(f"  cell ρ mean={st0['rho_mean']:.6f}")
        if "rho_max" in st0:
            print(f"  cell ρ max={st0['rho_max']:.6f}")

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

        p_rho = os.path.join(od, f"{stem}_l1_rho_masses.png")
        viz_infer_l1_rho_masses(
            prep["rho_peak_grid"], prep["z0_grid"], is_border, p_rho,
        )
        saved_files.append(p_rho)

        p_cell = os.path.join(od, f"{stem}_cell_rho.png")
        viz_infer_cell_rho(rho_post, is_border, p_cell)
        saved_files.append(p_cell)

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
        print(f"  ridge_nms={int(args.ridge_nms)}  threshold={threshold:.4f}")
        print()

    print(f"Outputs -> {args.output_dir}")
    for p in saved_files:
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
