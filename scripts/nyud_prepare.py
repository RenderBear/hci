r"""Lay out Gupta et al.'s NYUD v2 benchmark data (eccv14-data) for train.py / test.py.

Train = train + val (795 images), test = 654 images, as in the edge-detection literature. Ground truth
stays in the BSDS ``groundTruth`` .mat format that train.py and test.py read with ``--gt_format mat``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import scipy.io as sio
from PIL import Image

# nyu-hooks/cropIt.m: I(46:470, 41:600, :) — the 425x560 region inside the Kinect border.
CROP = (slice(45, 470), slice(40, 600))


def seg2bdry(seg: np.ndarray) -> np.ndarray:
    """BSDS ``seg2bdry(seg, 'imageSize')``."""
    v = seg[:-1, :] != seg[1:, :]
    h = seg[:, :-1] != seg[:, 1:]
    b = np.zeros(seg.shape, dtype=bool)
    b[:-1, :-1] = h[:-1, :] | h[1:, :] | v[:, :-1] | v[:, 1:]
    return b


def _dense(a) -> np.ndarray:
    return np.asarray(a.toarray() if hasattr(a, "toarray") else a)


def boundaries(gt_path: str) -> tuple[list[np.ndarray], bool]:
    """Per-annotator boundary maps, and whether any had to be derived from a segmentation."""
    gt = sio.loadmat(gt_path)["groundTruth"]
    out, derived = [], False
    for i in range(gt.shape[1]):
        s = gt[0, i]
        names = s.dtype.names or ()
        if "Boundaries" in names:
            out.append(_dense(s["Boundaries"][0, 0]).astype(bool))
        elif "Segmentation" in names:
            out.append(seg2bdry(_dense(s["Segmentation"][0, 0])))
            derived = True
        else:
            raise ValueError(f"{gt_path}: groundTruth has neither Boundaries nor Segmentation")
    return out, derived


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="extracted eccv14-data directory")
    ap.add_argument("--out", required=True, help="output root, e.g. NYUDv2")
    args = ap.parse_args()

    img_dir = os.path.join(args.src, "data", "images")
    gt_dir = os.path.join(args.src, "benchmarkData", "groundTruth")
    split_path = os.path.join(args.src, "benchmarkData", "metadata", "eccv14-splits.mat")
    for p in (img_dir, gt_dir, split_path):
        if not os.path.exists(p):
            sys.exit(f"missing {p} (is --src an extracted eccv14-data directory?)")

    splits = sio.loadmat(split_path)
    if "trainval" in splits:
        train = splits["trainval"].ravel()
    else:
        train = np.union1d(splits["train"].ravel(), splits["val"].ravel())
    sets = {"train": np.sort(train), "test": np.sort(splits["test"].ravel())}

    for name, ids in sets.items():
        out_img = os.path.join(args.out, name, "imgs")
        out_gt = os.path.join(args.out, name, "gt")
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_gt, exist_ok=True)
        n_crop = n_derived = 0
        for n in ids:
            stem = f"img_{int(n):04d}"
            img_path = os.path.join(img_dir, stem + ".png")
            gt_path = os.path.join(gt_dir, stem + ".mat")
            bounds, derived = boundaries(gt_path)
            gh, gw = bounds[0].shape
            img = Image.open(img_path)
            iw, ih = img.size
            crop = (ih, iw) != (gh, gw)
            if crop and not ((ih, iw) == (480, 640) and (gh, gw) == (425, 560)):
                sys.exit(f"{stem}: image {ih}x{iw} and ground truth {gh}x{gw} do not match")

            if crop:
                Image.fromarray(np.asarray(img)[CROP]).save(os.path.join(out_img, stem + ".png"))
                n_crop += 1
            else:
                shutil.copyfile(img_path, os.path.join(out_img, stem + ".png"))

            if derived:
                cells = np.empty((1, len(bounds)), dtype=object)
                for i, b in enumerate(bounds):
                    cells[0, i] = {"Boundaries": b.astype(np.uint8)}
                sio.savemat(os.path.join(out_gt, stem + ".mat"), {"groundTruth": cells})
                n_derived += 1
            else:
                shutil.copyfile(gt_path, os.path.join(out_gt, stem + ".mat"))

        note = []
        if n_crop:
            note.append(f"{n_crop} images cropped to 425x560 to match the ground truth")
        if n_derived:
            note.append(f"{n_derived} ground-truth files given Boundaries from their Segmentation")
        print(f"  {name}: {len(ids)} images -> {out_img}, {out_gt}" + (f"  ({'; '.join(note)})" if note else ""))


if __name__ == "__main__":
    main()
