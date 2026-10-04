r"""compare.py — score Canny and Sobel on a test split with the same benchmark as test.py, then
print them next to HCI.

Image/GT pairing, GT loading, 8-bit quantisation and scoring are test.py's own, so the numbers are
directly comparable with `uv run test`. Run HCI with test.py into ``<output_dir>/hci`` first (or
point ``--hci_dir`` at an existing run) to get it in the table.

    uv run python baselines/compare.py
    uv run python baselines/compare.py --canny_sigmas 1 2 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from params import EVAL  # noqa: E402
from hci.boundary_bench import collect, evaluate_image, exit_on_interrupt, write_bsr_files  # noqa: E402
from baselines import canny, sobel  # noqa: E402


def _load_test_module():
    # Load test.py by path: `import test` can resolve to the standard library's test package.
    spec = importlib.util.spec_from_file_location("hci_test", ROOT / "test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def methods(canny_sigmas):
    out = {}
    for s in canny_sigmas:
        out[f"canny_s{s:g}"] = lambda g, n, s=s: canny.soft_map(g, sigma=s, nthresh=n)
    out["sobel"] = lambda g, n: sobel.soft_map(g)
    out["sobel_nms"] = lambda g, n: sobel.soft_map_nms(g)
    return out


def run_method(name, fn, pairs, args, T, pool):
    mode_dir = os.path.join(args.output_dir, name)
    pred_dir = os.path.join(mode_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    per_image, stems, times = [], [], []
    pending = deque()
    t_total = time.perf_counter()
    for stem, img_path, gt_path in pairs:
        rgb = np.array(Image.open(img_path).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        t0 = time.perf_counter()
        pb = fn(gray, args.nthresh)
        times.append(time.perf_counter() - t0)

        gts = T._load_gt(gt_path, args.gt_format)
        H = min(pb.shape[0], gts[0].shape[0])
        W = min(pb.shape[1], gts[0].shape[1])
        gts = [g[:H, :W] for g in gts]
        png = (np.clip(pb[:H, :W], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(png, mode="L").save(os.path.join(pred_dir, f"{stem}.png"))
        pending.append(pool.submit(evaluate_image, png / 255.0, gts, nthresh=args.nthresh, max_dist=args.tol))
        stems.append(stem)
    while pending:
        per_image.append(pending.popleft().result())

    s = collect(per_image)
    write_bsr_files(mode_dir, s)
    summary = {
        "mode": name,
        "dataset": args.dataset,
        "protocol": "Berkeley boundary benchmark (bwmorph thin + one-to-one correspondPixels)",
        "nthresh": args.nthresh,
        "max_dist": args.tol,
        "gt_format": args.gt_format,
        "ODS_F1": s["ODS"]["F"], "ODS_P": s["ODS"]["P"], "ODS_R": s["ODS"]["R"], "ODS_t": s["ODS"]["T"],
        "OIS_F1": s["OIS"]["F"], "OIS_P": s["OIS"]["P"], "OIS_R": s["OIS"]["R"],
        "AP": s["AP"],
        "n_images": len(stems),
        "time": time.perf_counter() - t_total,
        "detect_s_median": float(np.median(times)),
        "images_dir": args.images,
        "preds_dir": pred_dir,
        "per_image": [
            {"stem": st, "OIS_F1": sc["F"], "OIS_P": sc["P"], "OIS_R": sc["R"], "OIS_t": sc["T"], "AP": sc["AP"]}
            for st, sc in zip(stems, s["per_image"])
        ],
        "pr_curve": s["pr_curve"],
    }
    with open(os.path.join(mode_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"  {name:<12} ODS={summary['ODS_F1']:.4f} @t={summary['ODS_t']:.3f}  "
        f"OIS={summary['OIS_F1']:.4f}  AP={summary['AP']:.4f}  ({summary['time']:.0f}s)",
        flush=True,
    )
    return summary


def print_table(output_dir, hci_dir):
    rows = []
    for mode, label in (("s_eval", "HCI (s_eval, NMS)"), ("c_eval", "HCI (c_eval, raw)")):
        p = os.path.join(hci_dir, mode, "results.json")
        if os.path.exists(p):
            rows.append((label, json.load(open(p))))
    for p in sorted(Path(output_dir).glob("*/results.json")):
        rows.append((p.parent.name, json.load(open(p))))
    if not rows:
        return
    print(f"\n| Method | ODS | OIS | AP | ODS P | ODS R | ODS t |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for label, r in rows:
        print(
            f"| {label} | {r['ODS_F1']:.4f} | {r['OIS_F1']:.4f} | {r['AP']:.4f} | "
            f"{r['ODS_P']:.4f} | {r['ODS_R']:.4f} | {r['ODS_t']:.3f} |"
        )


def main():
    ap = argparse.ArgumentParser(description="Canny / Sobel baselines on the HCI benchmark")
    ap.add_argument("--dataset", default="BRIND")
    ap.add_argument("--images", default="data/test/imgs")
    ap.add_argument("--test_gt", default="data/test/gt")
    ap.add_argument("--gt_format", default=None)
    ap.add_argument("-n", type=int, default=None, help="Cap number of images")
    ap.add_argument("--output_dir", default="output/compare")
    ap.add_argument("--hci_dir", default=None, help="test.py output dir (default: <output_dir>/hci)")
    ap.add_argument("--canny_sigmas", type=float, nargs="+", default=[canny.SIGMA])
    ap.add_argument("--methods", nargs="+", default=None, help="Subset of method names to run")
    ap.add_argument("--table_only", action="store_true", help="Only print the table from saved results")
    ap.add_argument("--tol", type=float, default=EVAL.MAX_DIST_FRAC)
    ap.add_argument("--nthresh", type=int, default=EVAL.THRESHOLD_COUNT)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    hci_dir = args.hci_dir or os.path.join(args.output_dir, "hci")

    if not args.table_only:
        T = _load_test_module()
        args.gt_format = args.gt_format or T._detect_gt_format(args.test_gt)
        img_files = sorted(
            [str(p) for p in Path(args.images).glob("*.jpg")] + [str(p) for p in Path(args.images).glob("*.png")]
        )
        if args.n is not None:
            img_files = img_files[: max(0, args.n)]
        pairs = []
        for img_path in img_files:
            stem = Path(img_path).stem
            gt_path = T._find_gt(args.test_gt, stem, args.gt_format)
            if gt_path is not None:
                pairs.append((stem, img_path, gt_path))
        print(f"{args.dataset}: {len(pairs)} image/GT pairs  maxDist={args.tol:g}  nthresh={args.nthresh}")

        todo = methods(args.canny_sigmas)
        if args.methods:
            todo = {k: v for k, v in todo.items() if k in args.methods}
        with ProcessPoolExecutor(
            max_workers=max(1, args.workers), mp_context=multiprocessing.get_context("spawn"),
            initializer=exit_on_interrupt,
        ) as pool:
            for name, fn in todo.items():
                run_method(name, fn, pairs, args, T, pool)

    print_table(args.output_dir, hci_dir)


if __name__ == "__main__":
    main()
