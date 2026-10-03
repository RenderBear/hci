r"""speed.py — per-image time and peak working memory for Sobel, Canny and HCI.

Every (method, resolution, threads) cell runs in a fresh subprocess. Memory is the rise in peak RSS
over one cold call, measured after imports, model load and image decode, so it counts the
detector's working buffers (and, for HCI, PyTorch's first-call allocations) but not the runtime
itself. Time is the median of warm calls. Canny and Sobel time one run on the grayscale image (for
Canny at its ODS threshold from compare.py); HCI times test.py's ``run_image_inference`` plus
``ridge_nms``, i.e. image to final edge map, which includes decoding the input file.

    uv run python baselines/speed.py
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESOLUTIONS = {"native": None, "720p": (1280, 720), "1080p": (1920, 1080)}


def _peak_rss() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024  # bytes on macOS, KiB on Linux


def worker(args) -> dict:
    import numpy as np
    from PIL import Image

    imgs = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    size = RESOLUTIONS[args.res]
    rgbs = []
    for p in imgs:
        im = Image.open(p).convert("RGB")
        if size is not None:
            # Keep orientation: portrait images become 1080×1920.
            im = im.resize(size if im.width >= im.height else size[::-1], Image.BICUBIC)
        rgbs.append(np.array(im))

    if args.method == "hci":
        import torch

        torch.set_num_threads(args.threads or torch.get_num_threads())
        from baselines.compare import _load_test_module
        from hci.renderer import ridge_nms

        T = _load_test_module()
        ckpt = torch.load(ROOT / "pretrained/final.pt", map_location="cpu", weights_only=False)
        model = T.build_model(ckpt, torch.device("cpu"))
        tmp = tempfile.mkdtemp()
        paths = []
        for i, a in enumerate(rgbs):
            paths.append(os.path.join(tmp, f"{i}.bmp"))
            Image.fromarray(a).save(paths[-1])
        inputs = paths

        def run(path):
            bmap, theta, _, _ = T.run_image_inference(model, path, torch.device("cpu"))
            return ridge_nms(bmap, theta=theta)
    else:
        import cv2

        cv2.setNumThreads(args.threads if args.threads else -1)
        from baselines import canny, sobel

        inputs = [cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) for a in rgbs]
        if args.method == "canny":
            run = lambda g: canny.detect(g, args.canny_t, sigma=args.canny_sigma)  # noqa: E731
        elif args.method == "sobel":
            run = sobel.soft_map
        else:
            run = sobel.soft_map_nms

    base = _peak_rss()
    run(inputs[0])
    peak = _peak_rss() - base

    times = []
    for _ in range(args.repeats):
        for x in inputs:
            t0 = time.perf_counter()
            run(x)
            times.append(time.perf_counter() - t0)
    h, w = rgbs[0].shape[:2]
    return {
        "method": args.method,
        "res": args.res,
        "shape": [h, w],
        "threads": args.threads or "default",
        "n_calls": len(times),
        "median_ms": 1e3 * float(np.median(times)),
        "p90_ms": 1e3 * float(np.percentile(times, 90)),
        "peak_mb": peak / 2**20,
        "bytes_per_px": peak / (h * w),
    }


def main():
    ap = argparse.ArgumentParser(description="Speed and memory of Sobel, Canny and HCI")
    ap.add_argument("--images", default="data/test/imgs")
    ap.add_argument("--output", default="output/compare/profile.json")
    ap.add_argument("--canny_sigma", type=float, default=None)
    ap.add_argument("--canny_t", type=float, default=None, help="Default: ODS threshold from compare.py")
    ap.add_argument("--methods", nargs="+", default=["sobel", "sobel_nms", "canny", "hci"])
    ap.add_argument("--threads", type=int, nargs="+", default=[0, 1], help="0 = library default")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--method", help=argparse.SUPPRESS)
    ap.add_argument("--res", help=argparse.SUPPRESS)
    ap.add_argument("-n", type=int, default=20, help=argparse.SUPPRESS)
    ap.add_argument("--repeats", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        args.threads = args.threads[0]
        print(json.dumps(worker(args)))
        return

    from baselines import canny

    sigma = args.canny_sigma if args.canny_sigma is not None else canny.SIGMA
    t = args.canny_t
    if t is None:
        p = ROOT / f"output/compare/canny_s{sigma:g}/results.json"
        t = json.load(open(p))["ODS_t"] if p.exists() else 0.2
    print(f"canny: sigma={sigma:g}  t={t:.3f}")

    rows = []
    for method in args.methods:
        for res in RESOLUTIONS:
            for th in args.threads:
                # HCI is ~10^3× slower, so it gets fewer calls; the fast methods repeat for stable medians.
                n, rep = (20, 1) if method == "hci" else (20, 10)
                if method == "hci" and res != "native":
                    n = 5 if res == "720p" else 2
                env = dict(os.environ)
                if th:
                    env.update(OMP_NUM_THREADS=str(th), VECLIB_MAXIMUM_THREADS=str(th), MKL_NUM_THREADS=str(th))
                cmd = [
                    sys.executable, __file__, "--worker", "--method", method, "--res", res,
                    "--threads", str(th), "-n", str(n), "--repeats", str(rep),
                    "--images", args.images, "--canny_sigma", str(sigma), "--canny_t", str(t),
                ]
                out = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
                if out.returncode != 0:
                    print(out.stderr[-2000:], file=sys.stderr)
                    continue
                r = json.loads(out.stdout.strip().splitlines()[-1])
                rows.append(r)
                print(
                    f"  {method:<10} {res:<7} threads={str(r['threads']):<8} "
                    f"{r['median_ms']:9.2f} ms  peak +{r['peak_mb']:8.1f} MB  "
                    f"({r['bytes_per_px']:.1f} B/px)",
                    flush=True,
                )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"canny_sigma": sigma, "canny_t": t, "rows": rows}, f, indent=2)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
