r"""test.py — STRIATE test-set evaluation (ODS, OIS, AP).

Per image: same L0 → moment pooling → association-field seed → render path as inference
(raw boundary map). Two prediction tracks: ``c_eval`` without ridge NMS; ``s_eval`` with ridge NMS
along dominant theta. BSDS-style matching uses about 0.75% of the image
diagonal.

Writes predictions under ``output_dir/{c_eval,s_eval,b_eval}/preds/`` and
aligned binary GT PNGs under ``output_dir/gt/`` (same stems), matching the
layout expected by ``eval/eval.py``. Only images with a matching GT file under
``--test_gt`` are processed; if more image files are listed than have GT, the
extras are skipped and a warning prints their stems.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation

from params import L0, L1, SEED, RENDER, TEST
from hci.L0 import compute_l0_rgb, compute_interior
from hci.L1 import z_from_l0_harmonics, pad_for_patch_grid, compute_cell_moments
from hci.renderer import (
    compute_render_features,
    render_boundary_map_torch,
    proj_to_device,
    ridge_nms,
    upgrade_renderer_state_dict,
)
from train import (
    StriateE2E,
    build_cells_flat,
    build_l0_pix,
    report_checkpoint_compatibility,
    upgrade_model_state_dict,
)

EVAL_THRESHOLDS = np.linspace(0.01, 0.99, TEST.THRESHOLD_COUNT)


def build_model(ckpt, device):
    m = StriateE2E(
        eps=SEED.EPS,
        render_cell_hidden=RENDER.CELL_HIDDEN,
        render_pixel_hidden=RENDER.PIXEL_HIDDEN,
    )
    sd = upgrade_model_state_dict(ckpt["model_state"])
    sd = upgrade_renderer_state_dict(sd, prefix="renderer.")
    incompatible = m.load_state_dict(sd, strict=False)
    report_checkpoint_compatibility(incompatible, context="test build_model")
    return m.to(device).eval()


def precision_max_dist(H, W, tol=None):

    d = (0.0075 if tol is None else float(tol)) * float(np.hypot(H, W))
    return max(1, int(round(d)))


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
    if gt_format == "mat":
        import scipy.io as sio

        m = sio.loadmat(gt_path)
        gt = m["groundTruth"]
        combined = None
        for i in range(gt.shape[1]):
            b = gt[0, i]["Boundaries"][0, 0].astype(np.float32)
            combined = b if combined is None else np.maximum(combined, b)
        return combined
    return np.array(Image.open(gt_path).convert("L")).astype(np.float32) / 255.0


def _eval_at_thresholds(bmap, gt, thresholds, max_dist):
    gt_bin = gt >= 0.5
    struct = np.ones((2 * max_dist + 1, 2 * max_dist + 1), dtype=bool)
    gt_dilated = binary_dilation(gt_bin, structure=struct)
    n_gt = int(gt_bin.sum())
    results = []
    for t in thresholds:
        pred = bmap >= t
        pred_dilated = binary_dilation(pred, structure=struct)
        tp_p = int((pred & gt_dilated).sum())
        tp_r = int((gt_bin & pred_dilated).sum())
        n_pred = int(pred.sum())
        prec = tp_p / max(n_pred, 1)
        rec = tp_r / max(n_gt, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-15)
        results.append({
            "t": float(t), "P": prec, "R": rec, "F1": f1,
            "tp_p": tp_p, "tp_r": tp_r, "n_pred": n_pred, "n_gt": n_gt,
        })
    return results


def _ap_from_pr(results):
    recs = np.array([r["R"] for r in results])
    precs = np.array([r["P"] for r in results])
    order = np.argsort(recs)
    recs = recs[order]
    precs = precs[order]
    mrec = np.concatenate([[0.0], recs, [1.0]])
    mpre = np.concatenate([[0.0], precs, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def run_image_inference(model, img_path, device):

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

    bmap = bmap_t.cpu().numpy()[:H0, :W0]
    theta = theta_t.cpu().numpy()[:H0, :W0]
    del cf_dev, proj_dev, rho_out, branch, bmap_t, theta_t
    gc.collect()
    return bmap, theta, H0, W0


def main():
    ap = argparse.ArgumentParser(description="STRIATE test-set metrics")
    ap.add_argument("--images", default="data/test/imgs")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--test_gt", default="data/test/gt")
    ap.add_argument("--gt_format", default=None)
    ap.add_argument("--model", default="output/checkpoints/intermediate.pt")
    ap.add_argument("--output_dir", default="output/test")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Tolerance factor for precision matching radius: max_dist = tol * diagonal "
        "(default: 0.0075).",
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

    eval_modes = ("c_eval", "s_eval", "b_eval")
    pred_dirs = {
        m: os.path.join(args.output_dir, m, "preds") for m in eval_modes
    }
    gt_dir = os.path.join(args.output_dir, "gt")
    for d in list(pred_dirs.values()) + [gt_dir]:
        os.makedirs(d, exist_ok=True)

    state = {
        m: {
            "per_image": [],
            "agg_tp_p": np.zeros(len(EVAL_THRESHOLDS)),
            "agg_tp_r": np.zeros(len(EVAL_THRESHOLDS)),
            "agg_n_pred": np.zeros(len(EVAL_THRESHOLDS)),
            "agg_n_gt": 0,
            "ois_tp_p_sum": 0,
            "ois_tp_r_sum": 0,
            "ois_n_pred_sum": 0,
            "ois_n_gt_sum": 0,
        }
        for m in ("c_eval", "s_eval")
    }
    # b_eval uses a single fixed threshold — no sweep, separate accumulators
    state["b_eval"] = {
        "per_image": [],
        "agg_tp_p": 0,
        "agg_tp_r": 0,
        "agg_n_pred": 0,
        "agg_n_gt": 0,
    }
    t_total = time.perf_counter()

    for idx, (stem, img_path, gt_path) in enumerate(pairs):
        t0 = time.perf_counter()
        bmap_c, theta, H0, W0 = run_image_inference(
            model, img_path, device,
        )
        bmap_s = ridge_nms(bmap_c, theta=theta)
        gt = _load_gt(gt_path, gt_format)

        H = min(bmap_c.shape[0], gt.shape[0])
        W = min(bmap_c.shape[1], gt.shape[1])
        bmaps = {"c_eval": bmap_c[:H, :W], "s_eval": bmap_s[:H, :W]}
        bmap_b = (bmap_s[:H, :W] >= TEST.BISTABLE_THRESHOLD).astype(np.float32)
        bmaps["b_eval"] = bmap_b
        gt = gt[:H, :W]

        eval_max_dist = precision_max_dist(H, W, tol=args.tol)

        # Save predictions as 8-bit PNGs.
        for m in ("c_eval", "s_eval"):
            png = np.clip(bmaps[m], 0.0, 1.0)
            png = (png * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(png, mode="L").save(
                os.path.join(pred_dirs[m], f"{stem}.png")
            )
        # b_eval: binary PNG (0 or 255)
        Image.fromarray((bmap_b * 255).astype(np.uint8), mode="L").save(
            os.path.join(pred_dirs["b_eval"], f"{stem}.png")
        )

        # Aligned GT (binary), same crop as preds — for eval/eval.py and archives.
        gt_png = ((gt >= 0.5).astype(np.uint8)) * 255
        Image.fromarray(gt_png, mode="L").save(os.path.join(gt_dir, f"{stem}.png"))

        line_parts = [f"  [{idx + 1}/{len(pairs)}] {stem}"]
        for m in ("c_eval", "s_eval"):
            results = _eval_at_thresholds(bmaps[m], gt, EVAL_THRESHOLDS, eval_max_dist)
            st = state[m]

            n_gt_img = results[0]["n_gt"]
            st["agg_n_gt"] += n_gt_img
            for ti, r in enumerate(results):
                st["agg_tp_p"][ti] += r["tp_p"]
                st["agg_tp_r"][ti] += r["tp_r"]
                st["agg_n_pred"][ti] += r["n_pred"]

            best_per_img = max(results, key=lambda r: r["F1"])
            st["ois_tp_p_sum"] += best_per_img["tp_p"]
            st["ois_tp_r_sum"] += best_per_img["tp_r"]
            st["ois_n_pred_sum"] += best_per_img["n_pred"]
            st["ois_n_gt_sum"] += n_gt_img

            ap_i = _ap_from_pr(results)
            st["per_image"].append(
                {
                    "stem": stem,
                    "OIS_F1": best_per_img["F1"],
                    "OIS_P": best_per_img["P"],
                    "OIS_R": best_per_img["R"],
                    "OIS_t": best_per_img["t"],
                    "AP": ap_i,
                }
            )
            tag = "C" if m == "c_eval" else "S"
            line_parts.append(
                f"{tag}: OIS={best_per_img['F1']:.3f}@{best_per_img['t']:.2f} "
                f"AP={ap_i:.3f}"
            )

        # b_eval: single fixed threshold, no sweep
        b_results = _eval_at_thresholds(
            bmaps["b_eval"], gt, [TEST.BISTABLE_THRESHOLD], eval_max_dist,
        )[0]
        b_st = state["b_eval"]
        b_st["agg_tp_p"] += b_results["tp_p"]
        b_st["agg_tp_r"] += b_results["tp_r"]
        b_st["agg_n_pred"] += b_results["n_pred"]
        b_st["agg_n_gt"] += b_results["n_gt"]
        b_st["per_image"].append(
            {
                "stem": stem,
                "F1": b_results["F1"],
                "P": b_results["P"],
                "R": b_results["R"],
            }
        )
        line_parts.append(
            f"B: F1={b_results['F1']:.3f} P={b_results['P']:.3f} R={b_results['R']:.3f}"
        )

        dt = time.perf_counter() - t0
        line_parts.append(f"{dt:.2f}s")
        print("  ".join(line_parts))

        del bmap_c, bmap_s, theta, gt, bmaps
        gc.collect()

    dt_total = time.perf_counter() - t_total

    def _finalize(mode):
        st = state[mode]
        agg_prec = st["agg_tp_p"] / np.maximum(st["agg_n_pred"], 1)
        agg_rec = st["agg_tp_r"] / max(st["agg_n_gt"], 1)
        agg_f1 = 2 * agg_prec * agg_rec / np.maximum(agg_prec + agg_rec, 1e-15)

        ods_idx = int(np.argmax(agg_f1))
        ods_f1 = float(agg_f1[ods_idx])
        ods_p = float(agg_prec[ods_idx])
        ods_r = float(agg_rec[ods_idx])
        ods_t = float(EVAL_THRESHOLDS[ods_idx])

        ois_prec = st["ois_tp_p_sum"] / max(st["ois_n_pred_sum"], 1)
        ois_rec = st["ois_tp_r_sum"] / max(st["ois_n_gt_sum"], 1)
        ois_f1 = 2 * ois_prec * ois_rec / max(ois_prec + ois_rec, 1e-15)
        mean_ois_macro = float(np.mean([r["OIS_F1"] for r in st["per_image"]]))

        agg_results = [
            {
                "t": float(EVAL_THRESHOLDS[i]),
                "P": float(agg_prec[i]),
                "R": float(agg_rec[i]),
                "F1": float(agg_f1[i]),
            }
            for i in range(len(EVAL_THRESHOLDS))
        ]
        ap_global = _ap_from_pr(agg_results)

        summary = {
            "mode": mode,
            "ODS_F1": ods_f1,
            "ODS_P": ods_p,
            "ODS_R": ods_r,
            "ODS_t": ods_t,
            "OIS_F1": ois_f1,
            "OIS_P": ois_prec,
            "OIS_R": ois_rec,
            "OIS_F1_macro": mean_ois_macro,
            "AP": ap_global,
            "n_images": len(pairs),
            "time": dt_total,
            "max_images": args.max_images,
            "images_dir": args.images,
            "model": args.model,
            "preds_dir": pred_dirs[mode],
            "gt_dir": gt_dir,
            "per_image": st["per_image"],
            "pr_curve": agg_results,
        }
        out_path = os.path.join(args.output_dir, mode, "results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary, out_path

    print(f"\n{'=' * 50}")
    for mode in ("c_eval", "s_eval"):
        summary, out_path = _finalize(mode)
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

    print(f"[GT (aligned to preds, binary PNG)]")
    print(f"  gt    -> {gt_dir}  ({len(pairs)} files)")

    # b_eval: fixed-threshold binary output — deployment metric
    b_st = state["b_eval"]
    b_prec = b_st["agg_tp_p"] / max(b_st["agg_n_pred"], 1)
    b_rec = b_st["agg_tp_r"] / max(b_st["agg_n_gt"], 1)
    b_f1 = 2 * b_prec * b_rec / max(b_prec + b_rec, 1e-15)
    b_f1_macro = float(np.mean([r["F1"] for r in b_st["per_image"]]))

    b_summary = {
        "mode": "b_eval",
        "threshold": TEST.BISTABLE_THRESHOLD,
        "F1": b_f1,
        "P": b_prec,
        "R": b_rec,
        "F1_macro": b_f1_macro,
        "n_images": len(pairs),
        "time": dt_total,
        "max_images": args.max_images,
        "images_dir": args.images,
        "model": args.model,
        "preds_dir": pred_dirs["b_eval"],
        "gt_dir": gt_dir,
        "per_image": b_st["per_image"],
    }
    b_out_path = os.path.join(args.output_dir, "b_eval", "results.json")
    os.makedirs(os.path.dirname(b_out_path), exist_ok=True)
    with open(b_out_path, "w") as f:
        json.dump(b_summary, f, indent=2)

    print(f"[B_EVAL (bistable, t={TEST.BISTABLE_THRESHOLD})]")
    print(f"  F1={b_f1:.4f}  P={b_prec:.4f}  R={b_rec:.4f}  (fixed threshold, no tuning)")
    print(f"  F1_macro={b_f1_macro:.4f}")
    print(f"  preds -> {pred_dirs['b_eval']}")
    print(f"  json  -> {b_out_path}")

    print(f"{'=' * 50}")
    print(f"{len(pairs)} images  {dt_total:.1f}s total")


if __name__ == "__main__":
    main()