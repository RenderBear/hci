r"""Boundary benchmark — the BSDS500 protocol (BSR bench: evaluation_bdry_image, correspondPixels,
collect_eval_bdry), ported to numpy/scipy.

Per image and threshold t the prediction is binarised at ``pb >= t``, thinned with
``bwmorph(·, 'thin', inf)`` and matched one-to-one against each human annotation within
``max_dist · diagonal`` pixels. Recall counts matched GT pixels summed over annotators; precision
counts predicted pixels matched to *any* annotator. Summary statistics follow collect_eval_bdry.m:
ODS with linear interpolation between thresholds, OIS from each image's best threshold, AP as the
area under the 0.01-resampled PR curve.

``correspond_pixels`` solves the same integer assignment as the Berkeley CSA matcher (distances
×100, rounded; an unmatched pixel costs ``outlier_cost · max_dist · diagonal``, so the match count
is maximised first and the total distance second), but with every outlier connection present where
the C++ code samples six at random.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

MAX_DIST = 0.0075
OUTLIER_COST = 100.0
NTHRESH = 99

# Integer costs as in match.cc; scipy's sparse LAPJV can stall on non-integral float costs.
_COST_SCALE = 100


def thresholds(nthresh: int = NTHRESH) -> np.ndarray:
    if nthresh <= 1:
        return np.array([0.5])
    return np.linspace(1.0 / (nthresh + 1), 1.0 - 1.0 / (nthresh + 1), nthresh)


# ── thinning: bwmorph(bmap, 'thin', inf) ─────────────────────────────────────

def _thin_luts() -> tuple[np.ndarray, np.ndarray]:
    # Lam, Lee & Suen (1992) conditions G1–G3; bit i of the index is neighbour x_{i+1}
    # in the order E, NE, N, NW, W, SW, S, SE.
    n = np.arange(256)
    b = [((n >> i) & 1).astype(bool) for i in range(8)]
    g1 = sum((~b[k] & (b[k + 1] | b[(k + 2) % 8])).astype(int) for k in (0, 2, 4, 6)) == 1
    n1 = sum((b[k] | b[k + 1]).astype(int) for k in (0, 2, 4, 6))
    n2 = sum((b[k] | b[(k + 1) % 8]).astype(int) for k in (1, 3, 5, 7))
    g2 = (np.minimum(n1, n2) >= 2) & (np.minimum(n1, n2) <= 3)
    g3 = ~((b[1] | b[2] | ~b[7]) & b[0])
    g3p = ~((b[5] | b[6] | ~b[3]) & b[4])
    return g1 & g2 & g3, g1 & g2 & g3p


_THIN_LUTS = _thin_luts()
_NEIGHBOUR_BITS = np.array([[8, 4, 2], [16, 0, 1], [32, 64, 128]], dtype=np.uint8)


def thin(bmap: np.ndarray) -> np.ndarray:
    skel = np.asarray(bmap, dtype=bool).astype(np.uint8)
    n_prev, n_cur = -1, int(skel.sum())
    while n_cur != n_prev:
        n_prev = n_cur
        for lut in _THIN_LUTS:
            code = ndimage.correlate(skel, _NEIGHBOUR_BITS, mode="constant")
            skel[lut[code]] = 0
        n_cur = int(skel.sum())
    return skel.astype(bool)


# ── correspondPixels ─────────────────────────────────────────────────────────

def correspond_pixels(
    bmap1: np.ndarray,
    bmap2: np.ndarray,
    max_dist: float = MAX_DIST,
    outlier_cost: float = OUTLIER_COST,
) -> tuple[np.ndarray, np.ndarray]:
    if bmap1.shape != bmap2.shape:
        raise ValueError(f"shape mismatch: {bmap1.shape} vs {bmap2.shape}")
    match1 = np.zeros(bmap1.shape, dtype=bool)
    match2 = np.zeros(bmap2.shape, dtype=bool)
    p1 = np.argwhere(bmap1)
    p2 = np.argwhere(bmap2)
    if len(p1) == 0 or len(p2) == 0:
        return match1, match2

    radius = float(max_dist) * float(np.hypot(*bmap1.shape))
    pairs = cKDTree(p1).sparse_distance_matrix(cKDTree(p2), radius, output_type="ndarray")
    if len(pairs) == 0:
        return match1, match2

    # Only pixels with a partner in range enter the assignment (the rest are outliers).
    u1, i = np.unique(pairs["i"], return_inverse=True)
    u2, j = np.unique(pairs["j"], return_inverse=True)
    n1, n2 = len(u1), len(u2)
    d = pairs["v"].astype(np.float64)

    # Square assignment: rows = [map1 pixels, map2 outlier slots], cols = [map2 pixels, map1
    # outlier slots]. Mirrored slot-to-slot edges let every partial matching extend to a perfect
    # one, so the optimum pays the outlier cost for each unmatched pixel on either side. The +1 on
    # real and slot edges keeps weights non-zero and adds the same constant to every matched pair.
    oc = float(np.ceil(float(outlier_cost) * radius * _COST_SCALE))
    k1 = np.arange(n1)
    k2 = np.arange(n2)
    rows = np.concatenate([i, k1, n1 + k2, n1 + j])
    cols = np.concatenate([j, n2 + k1, k2, n2 + i])
    w = np.concatenate([
        np.rint(d * _COST_SCALE) + 1.0,
        np.full(n1, oc),
        np.full(n2, oc),
        np.ones(len(i)),
    ])
    n = n1 + n2
    graph = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    r_ind, c_ind = min_weight_full_bipartite_matching(graph)

    real = (r_ind < n1) & (c_ind < n2)
    match1[tuple(p1[u1[r_ind[real]]].T)] = True
    match2[tuple(p2[u2[c_ind[real]]].T)] = True
    return match1, match2


# ── evaluation_bdry_image ────────────────────────────────────────────────────

def evaluate_image(
    pb: np.ndarray,
    gts: list[np.ndarray],
    *,
    nthresh: int = NTHRESH,
    max_dist: float = MAX_DIST,
    thin_pb: bool = True,
) -> dict[str, np.ndarray]:
    thr = thresholds(nthresh)
    cnt_r = np.zeros(len(thr), dtype=np.int64)
    sum_r = np.zeros(len(thr), dtype=np.int64)
    cnt_p = np.zeros(len(thr), dtype=np.int64)
    sum_p = np.zeros(len(thr), dtype=np.int64)
    gts = [np.asarray(g, dtype=bool) for g in gts]
    for t, th in enumerate(thr):
        bmap = pb >= th
        if thin_pb:
            bmap = thin(bmap)
        acc_p = np.zeros(bmap.shape, dtype=bool)
        for gt in gts:
            m1, m2 = correspond_pixels(bmap, gt, max_dist)
            acc_p |= m1
            sum_r[t] += int(gt.sum())
            cnt_r[t] += int(m2.sum())
        sum_p[t] = int(bmap.sum())
        cnt_p[t] = int(acc_p.sum())
    return {"thresh": thr, "cntR": cnt_r, "sumR": sum_r, "cntP": cnt_p, "sumP": sum_p}


# ── collect_eval_bdry ────────────────────────────────────────────────────────

def fmeasure(r: np.ndarray, p: np.ndarray) -> np.ndarray:
    s = p + r
    return 2.0 * p * r / (s + (s == 0))


def _ratio(cnt: np.ndarray, tot: np.ndarray) -> np.ndarray:
    return cnt / (tot + (tot == 0))


def max_f(thr: np.ndarray, r: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    d = np.linspace(0.0, 1.0, 100)[None, :]

    def lerp(v: np.ndarray) -> np.ndarray:
        return np.concatenate([v[:1], (v[1:, None] * d + v[:-1, None] * (1.0 - d)).ravel()])

    tt, rr, pp = lerp(thr), lerp(r), lerp(p)
    ff = fmeasure(rr, pp)
    k = int(np.argmax(ff))
    return float(tt[k]), float(rr[k]), float(pp[k]), float(ff[k])


def area_pr(r: np.ndarray, p: np.ndarray) -> float:
    ru, idx = np.unique(r, return_index=True)
    if len(ru) <= 1:
        return 0.0
    pi = np.interp(np.linspace(0.0, 1.0, 101), ru, p[idx], left=0.0, right=0.0)
    return float(pi.sum() * 0.01)


def score_image(ev: dict[str, np.ndarray]) -> dict[str, float]:
    r = _ratio(ev["cntR"], ev["sumR"])
    p = _ratio(ev["cntP"], ev["sumP"])
    bt, br, bp, bf = max_f(np.asarray(ev["thresh"], dtype=np.float64), r, p)
    return {"T": bt, "R": br, "P": bp, "F": bf, "AP": area_pr(r, p)}


def collect(per_image: list[dict[str, np.ndarray]]) -> dict:
    thr = np.asarray(per_image[0]["thresh"], dtype=np.float64)
    tot = {k: np.zeros(len(thr)) for k in ("cntR", "sumR", "cntP", "sumP")}
    best = dict.fromkeys(("cntR", "sumR", "cntP", "sumP"), 0.0)
    images = []
    for ev in per_image:
        f = fmeasure(_ratio(ev["cntR"], ev["sumR"]), _ratio(ev["cntP"], ev["sumP"]))
        images.append(score_image(ev))
        for k in tot:
            tot[k] = tot[k] + ev[k]
        k_best = int(np.flatnonzero(f == f.max())[-1])
        for k in best:
            best[k] += float(ev[k][k_best])

    r = _ratio(tot["cntR"], tot["sumR"])
    p = _ratio(tot["cntP"], tot["sumP"])
    ods_t, ods_r, ods_p, ods_f = max_f(thr, r, p)
    ois_r = best["cntR"] / (best["sumR"] + (best["sumR"] == 0))
    ois_p = best["cntP"] / (best["sumP"] + (best["sumP"] == 0))
    return {
        "ODS": {"T": ods_t, "R": ods_r, "P": ods_p, "F": ods_f},
        "OIS": {"R": float(ois_r), "P": float(ois_p), "F": float(fmeasure(np.float64(ois_r), np.float64(ois_p)))},
        "AP": area_pr(r, p),
        "pr_curve": [
            {"t": float(t), "R": float(rv), "P": float(pv), "F": float(fv)}
            for t, rv, pv, fv in zip(thr, r, p, fmeasure(r, p))
        ],
        "per_image": images,
    }


def write_bsr_files(out_dir: str, summary: dict) -> None:
    """eval_bdry.txt, eval_bdry_thr.txt, eval_bdry_img.txt in the BSR column layout."""
    s = summary
    with open(os.path.join(out_dir, "eval_bdry.txt"), "w") as f:
        vals = (s["ODS"]["T"], s["ODS"]["R"], s["ODS"]["P"], s["ODS"]["F"],
                s["OIS"]["R"], s["OIS"]["P"], s["OIS"]["F"], s["AP"])
        f.write(" ".join(f"{v:10g}" for v in vals) + "\n")
    with open(os.path.join(out_dir, "eval_bdry_thr.txt"), "w") as f:
        for row in s["pr_curve"]:
            f.write(f"{row['t']:10g} {row['R']:10g} {row['P']:10g} {row['F']:10g}\n")
    with open(os.path.join(out_dir, "eval_bdry_img.txt"), "w") as f:
        for n, im in enumerate(s["per_image"], start=1):
            f.write(f"{n:10d} {im['T']:10g} {im['R']:10g} {im['P']:10g} {im['F']:10g}\n")
