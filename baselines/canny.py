r"""Canny baseline — OpenCV Canny on the grayscale image, swept over the benchmark thresholds.

Canny returns a binary map, so a single run gives one point on the PR curve. ``soft_map`` runs it
once per benchmark threshold ``t`` (high = ``t · gmax``, low = ``LOW_RATIO · high``) and stores, per
pixel, the highest ``t`` at which it survives hysteresis. Hysteresis is monotone in the threshold,
so binarising the soft map at ``t`` gives back exactly the Canny map at ``t``, and ODS is Canny at
its best single dataset-wide threshold — the protocol behind the BSDS Canny baseline.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from hci.boundary_bench import thresholds

SIGMA = 2.0
LOW_RATIO = 0.4  # MATLAB's edge(·, 'canny') default


def gmax(sigma: float) -> float:
    """Largest L2 gradient a 0→255 step reaches after the blur, in OpenCV's 3×3 Sobel units.

    Fixed per sigma, not per image, so one threshold means the same thing on every image.
    """
    step = 4.0 * 255.0  # unblurred Sobel response to an axis-aligned step
    if sigma <= 0:
        return step
    return min(step, 8.0 * 255.0 / (sigma * math.sqrt(2.0 * math.pi)))


def blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return gray
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)


def detect(gray: np.ndarray, t: float, sigma: float = SIGMA) -> np.ndarray:
    """One Canny run at benchmark threshold ``t``; returns a bool edge map."""
    high = t * gmax(sigma)
    return cv2.Canny(blur(gray, sigma), LOW_RATIO * high, high, L2gradient=True) > 0


def soft_map(gray: np.ndarray, sigma: float = SIGMA, nthresh: int = 99) -> np.ndarray:
    thr = thresholds(nthresh)
    # Each level sits halfway to the next, so 8-bit quantisation never drops it below its own t.
    level = np.append((thr[:-1] + thr[1:]) / 2.0, (thr[-1] + 1.0) / 2.0)
    g = blur(gray, sigma)
    out = np.zeros(gray.shape, dtype=np.float32)
    for t, lv in zip(thr, level):
        high = t * gmax(sigma)
        e = cv2.Canny(g, LOW_RATIO * high, high, L2gradient=True) > 0
        if not e.any():
            break
        out[e] = lv
    return out
