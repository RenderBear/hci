r"""test.py — HCI test-set evaluation on the BSDS500 boundary benchmark (ODS, OIS, AP)"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image

from params import L0, L1, SEED, EVAL
from hci.L0 import compute_l0_rgb, compute_interior
from hci.L1 import z_from_l0_harmonics, pad_for_patch_grid, compute_cell_moments
from hci.renderer import (
    compute_render_features,
    render_boundary_map_torch,
    proj_to_device,
    ridge_nms,
    upgrade_renderer_state_dict,
)
from hci.boundary_bench import collect, evaluate_image, score_image, write_bsr_files
from hci.diagnostics_viz import viz_infer_rho, viz_infer_geometry
from train import (
    HCIE2E,
    build_cells_flat,
    build_l0_pix,
    report_checkpoint_compatibility,
    upgrade_model_state_dict,
)


def build_model(ckpt, device):
    m = HCIE2E(eps=SEED.EPS)
    sd = upgrade_model_state_dict(ckpt["model_state"])
    sd = upgrade_renderer_state_dict(sd, prefix="renderer.")
    incompatible = m.load_state_dict(sd, strict=False)
    report_checkpoint_compatibility(incompatible, context="test build_model")
    return m.to(device).eval()


def _detect_gt_format(gt_dir):
    if glob.glob(os.path.join(gt_dir, "*.mat")):
        return "mat"
    return "png"


def _find_gt(gt_dir, stem, gt_format):
    if gt_format == "mat":
        p = os.path.join(gt_dir, f"{stem}.mat")
        return p if os.path.exists(p) else None
    for ext in [".png", ".jpg"]:
        p = os.path.join(gt_dir, stem + ext)
        if os.path.exists(p):
            return p
    matches = glob.glob(os.path.join(gt_dir, f"{stem}*"))
    return matches[0] if matches else None


def _load_gt(gt_path, gt_format):
    """Boundary maps to match against: one per annotator (BSDS .mat) or a single PNG map."""
    if gt_format == "mat":
        import scipy.io as sio

        gt = sio.loadmat(gt_path)["groundTruth"]
        return [gt[0, i]["Boundaries"][0, 0].astype(bool) for i in range(gt.shape[1])]
    return [np.array(Image.open(gt_path).convert("L")).astype(np.float32) / 255.0 >= 0.5]


def run_image_inference(
    model,
    img_path,
    device,
    *,
    diagnostics_dir: str | None = None,
    stem: str | None = None,
):

    ir_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    ir_p, H0, W0 = pad_for_patch_grid(ir_np, L1.PATCH_SIZE, L1.PATCH_OVERLAP)
    del ir_np
    gc.collect()

    ir_t = torch.from_numpy(ir_p).to(device)
    with torch.no_grad():
        h, vld, _, _, _, s, h1m, h2m, h2m_lum, h2m_chr = compute_l0_rgb(
            ir_t,
            eta_lum=L0.ETA_LUM,
            eta_chr=L0.ETA_CHR,
            gamma=L0.GAMMA,
            offsets=L0.OFFSETS,
            metric=getattr(model, "l0_metric", None),
            notch=getattr(model, "l0_notch", None),
        )
    bm_t = ~compute_interior(ir_p.shape[0], ir_p.shape[1], device)
    z1, z2 = z_from_l0_harmonics(s, bm_t)
    _ = z1

    s_np = s.cpu().numpy()
    bm_np = bm_t.cpu().numpy()
    z2_img = (s_np[..., 2] + 1j * s_np[..., 3]).astype(np.complex64)
    z2_img[bm_np] = 0.0
    l0_pix = build_l0_pix(
        s_np, h1m, h2m, bm_np, h2m_lum=h2m_lum, h2m_chr=h2m_chr,
    )
    del h, vld, s, h1m, h2m_lum, h2m_chr
    gc.collect()

    cells = compute_cell_moments(
        h2m,
        z2,
        L1.PATCH_SIZE,
        border_mask=bm_t,
        patch_overlap=L1.PATCH_OVERLAP,
        border_patch_max_frac=L1.BORDER_PATCH_MAX_FRAC,
        eps=L1.EPS,
        device=device,
        verbose=False,
        kappa_vm=model.seed.kappa_vm.detach(),
        num_orient_bins=int(L1.NUM_ORIENT_BINS),
    )
    del h2m, z1, z2, bm_t, ir_t
    gc.collect()
    cells["is_border"] |= (cells["cy"] + cells["P"] / 2 > H0) | (
        cells["cx"] + cells["P"] / 2 > W0
    )

    nH, nW = cells["nH"], cells["nW"]
    proj = compute_render_features(z2_img, ir_p, cells, bm_np, eps=SEED.EPS)
    del z2_img, bm_np
    gc.collect()

    Hp, Wp = ir_p.shape[:2]
    cells_flat = build_cells_flat(cells)
    del cells, ir_p
    gc.collect()

    cf_dev = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in cells_flat.items()
    }
    l0_dev = {k: v.to(device) for k, v in l0_pix.items()}
    proj_dev = proj_to_device(proj, device)

    with torch.no_grad():
        rho_out, branch, _, _, _, cf_out, _ = model.seed(cells_flat=cf_dev)
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

        if diagnostics_dir is not None and stem is not None:
            os.makedirs(diagnostics_dir, exist_ok=True)
            nH = int(cf_dev["nH"])
            nW = int(cf_dev["nW"])
            rho_nr = cf_out["rho_nr"].detach().cpu().numpy().reshape(nH, nW)
            rho_post = rho_out.detach().cpu().numpy().reshape(nH, nW)
            is_b = cf_dev["is_border"].detach().cpu().numpy().reshape(nH, nW).astype(bool)
            p_rho = os.path.join(diagnostics_dir, f"{stem}_rho.png")
            viz_infer_rho(rho_nr, rho_post, is_b, p_rho)
            rho_coll = cf_out["rho_coll"].detach().cpu().numpy().reshape(nH, nW)
            sur = cf_out["sur"].detach().cpu().numpy().reshape(nH, nW)
            p_geom = os.path.join(diagnostics_dir, f"{stem}_geometry.png")
            viz_infer_geometry(rho_coll, sur, is_b, p_geom)

    bmap = bmap_t.cpu().numpy()[:H0, :W0]
    theta = theta_t.cpu().numpy()[:H0, :W0]
    del cf_dev, proj_dev, rho_out, branch, bmap_t, theta_t, cf_out
    gc.collect()
    return bmap, theta, H0, W0


def main():
    ap = argparse.ArgumentParser(description="HCI test-set metrics (BSDS500 boundary benchmark)")
    ap.add_argument("--images", default="data/test/imgs")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--test_gt", default="data/test/gt")
    ap.add_argument("--gt_format", default=None)
    ap.add_argument("--model", default="pretrained/final.pt")
    ap.add_argument("--output_dir", default="output/test")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="Save infer-style diagnostics under output_dir/diagnostics/ ({stem}_rho.png, {stem}_geometry.png).",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=EVAL.MAX_DIST_FRAC,
        help="Matching distance as a fraction of the image diagonal "
        "(BSDS maxDist: 0.0075; NYUD convention: 0.011).",
    )
    ap.add_argument(
        "--nthresh",
        type=int,
        default=EVAL.THRESHOLD_COUNT,
        help="Number of thresholds (BSDS: 99).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Processes running the benchmark; inference stays on --device.",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    gt_format = args.gt_format or _detect_gt_format(args.test_gt)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model = build_model(ckpt, device)

    img_files = sorted(
        glob.glob(os.path.join(args.images, "*.jpg"))
        + glob.glob(os.path.join(args.images, "*.png"))
    )
    if args.max_images is not None:
        img_files = img_files[: max(0, args.max_images)]

    pairs = []
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = _find_gt(args.test_gt, stem, gt_format)
        if gt_path is not None:
            pairs.append((stem, img_path, gt_path))

    if not pairs:
        print(f"error: no image/gt pairs found in {args.images} + {args.test_gt}")
        return

    n_img = len(img_files)
    n_pairs = len(pairs)
    if n_img > n_pairs:
        paired = {s for s, _, _ in pairs}
        skipped = [
            os.path.splitext(os.path.basename(p))[0]
            for p in img_files
            if os.path.splitext(os.path.basename(p))[0] not in paired
        ]
        tail = "" if len(skipped) <= 15 else f" … (+{len(skipped) - 15} more)"
        print(
            f"warning: {n_img - n_pairs} of {n_img} image(s) have no GT under "
            f"{args.test_gt} — skipped (no preds/gt written for these): "
            f"{', '.join(skipped[:15])}{tail}"
        )

    cap = f"  max_images={args.max_images}" if args.max_images is not None else ""
    print(
        f"model={args.model}  device={device}  gt_format={gt_format}  "
        f"paired={len(pairs)}/{n_img} images-with-GT{cap}"
    )

    eval_modes = ("c_eval", "s_eval")
    pred_dirs = {
        m: os.path.join(args.output_dir, m, "preds") for m in eval_modes
    }
    gt_dir = os.path.join(args.output_dir, "gt")
    for d in list(pred_dirs.values()) + [gt_dir]:
        os.makedirs(d, exist_ok=True)

    diag_dir = os.path.join(args.output_dir, "diagnostics") if args.diagnostics else None
    if diag_dir:
        print(f"diagnostics -> {diag_dir}  (<stem>_rho.png, <stem>_geometry.png per image)")

    per_image = {m: [] for m in eval_modes}
    stems = []
    t_total = time.perf_counter()

    def _report(idx, stem, dt, futures):
        line_parts = [f"  [{idx + 1}/{len(pairs)}] {stem}"]
        for m in eval_modes:
            ev = futures[m].result()
            per_image[m].append(ev)
            sc = score_image(ev)
            tag = "C" if m == "c_eval" else "S"
            line_parts.append(f"{tag}: OIS={sc['F']:.3f}@{sc['T']:.2f} AP={sc['AP']:.3f}")
        stems.append(stem)
        line_parts.append(f"infer {dt:.2f}s")
        print("  ".join(line_parts), flush=True)

    n_workers = max(1, args.workers)
    pending = deque()
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        for idx, (stem, img_path, gt_path) in enumerate(pairs):
            t0 = time.perf_counter()
            bmap_c, theta, H0, W0 = run_image_inference(
                model,
                img_path,
                device,
                diagnostics_dir=diag_dir,
                stem=stem if diag_dir else None,
            )
            bmap_s = ridge_nms(bmap_c, theta=theta)
            dt = time.perf_counter() - t0
            gts = _load_gt(gt_path, gt_format)

            H = min(bmap_c.shape[0], gts[0].shape[0])
            W = min(bmap_c.shape[1], gts[0].shape[1])
            gts = [g[:H, :W] for g in gts]

            futures = {}
            for m, bmap in (("c_eval", bmap_c), ("s_eval", bmap_s)):
                png = (np.clip(bmap[:H, :W], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                Image.fromarray(png, mode="L").save(os.path.join(pred_dirs[m], f"{stem}.png"))
                # Score the saved 8-bit map, as the MATLAB benchmark reads it back.
                futures[m] = pool.submit(
                    evaluate_image, png / 255.0, gts, nthresh=args.nthresh, max_dist=args.tol,
                )

            gt_png = np.logical_or.reduce(gts).astype(np.uint8) * 255
            Image.fromarray(gt_png, mode="L").save(os.path.join(gt_dir, f"{stem}.png"))

            pending.append((idx, stem, dt, futures))
            while pending and (
                len(pending) > n_workers
                or all(f.done() for f in pending[0][3].values())
            ):
                _report(*pending.popleft())

            del bmap_c, bmap_s, theta, gts
            gc.collect()

        while pending:
            _report(*pending.popleft())

    dt_total = time.perf_counter() - t_total

    gt_kind = "per-annotator .mat" if gt_format == "mat" else "single-map PNG"
    print(f"\n{'=' * 50}")
    print(f"BSDS500 boundary benchmark  nthresh={args.nthresh}  maxDist={args.tol:g}  GT: {gt_kind}")
    for mode in eval_modes:
        s = collect(per_image[mode])
        mode_dir = os.path.join(args.output_dir, mode)
        write_bsr_files(mode_dir, s)
        summary = {
            "mode": mode,
            "protocol": "BSDS500 boundary benchmark (bwmorph thin + one-to-one correspondPixels)",
            "nthresh": args.nthresh,
            "max_dist": args.tol,
            "gt_format": gt_format,
            "ODS_F1": s["ODS"]["F"],
            "ODS_P": s["ODS"]["P"],
            "ODS_R": s["ODS"]["R"],
            "ODS_t": s["ODS"]["T"],
            "OIS_F1": s["OIS"]["F"],
            "OIS_P": s["OIS"]["P"],
            "OIS_R": s["OIS"]["R"],
            "AP": s["AP"],
            "n_images": len(stems),
            "time": dt_total,
            "max_images": args.max_images,
            "images_dir": args.images,
            "model": args.model,
            "preds_dir": pred_dirs[mode],
            "gt_dir": gt_dir,
            "per_image": [
                {
                    "stem": stem,
                    "OIS_F1": sc["F"],
                    "OIS_P": sc["P"],
                    "OIS_R": sc["R"],
                    "OIS_t": sc["T"],
                    "AP": sc["AP"],
                    "counts": {k: ev[k].tolist() for k in ("cntR", "sumR", "cntP", "sumP")},
                }
                for stem, sc, ev in zip(stems, s["per_image"], per_image[mode])
            ],
            "pr_curve": s["pr_curve"],
        }
        out_path = os.path.join(mode_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        label = "C_EVAL (raw)" if mode == "c_eval" else "S_EVAL (NMS-thinned)"
        print(f"[{label}]")
        print(
            f"  ODS  F1={summary['ODS_F1']:.4f}  P={summary['ODS_P']:.4f}  "
            f"R={summary['ODS_R']:.4f}  @t={summary['ODS_t']:.3f}"
        )
        print(
            f"  OIS  F1={summary['OIS_F1']:.4f}  P={summary['OIS_P']:.4f}  "
            f"R={summary['OIS_R']:.4f}"
        )
        print(f"  AP   {summary['AP']:.4f}")
        print(f"  preds -> {pred_dirs[mode]}")
        print(f"  json  -> {out_path}")
        print(f"  bsr   -> {mode_dir}/eval_bdry.txt, eval_bdry_thr.txt, eval_bdry_img.txt")

    print(f"[GT (aligned to preds, binary PNG; union of annotators for .mat)]")
    print(f"  gt    -> {gt_dir}  ({len(pairs)} files)")

    print(f"{'=' * 50}")
    print(f"{len(pairs)} images  {dt_total:.1f}s total")


if __name__ == "__main__":
    main()