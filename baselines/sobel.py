r"""Sobel baseline — 3×3 Sobel gradient magnitude of the grayscale image, no smoothing.

``soft_map`` divides the L2 magnitude by the response to a unit step (4 for a [0, 1] image), a fixed
scale so one threshold means the same thing on every image. ``nms`` keeps only local maxima across
the gradient direction (Canny's four-direction suppression), the counterpart of HCI's ``s_eval``.
"""

from __future__ import annotations

import cv2
import numpy as np

STEP = 4.0  # 3×3 Sobel response to an axis-aligned 0→1 step


def gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def soft_map(gray: np.ndarray) -> np.ndarray:
    gx, gy = gradients(gray)
    return np.clip(cv2.magnitude(gx, gy) / STEP, 0.0, 1.0)


def nms(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    # Quantise the gradient direction to 0°, 45°, 90°, 135° and compare with the two neighbours
    # along it.
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    sector = (((ang + 22.5) // 45.0).astype(np.int8)) % 4
    p = np.pad(mag, 1, mode="constant")
    H, W = mag.shape
    c = p[1:-1, 1:-1]
    # (dy, dx) of the neighbour in the gradient direction; image y points down.
    steps = ((0, 1), (1, 1), (1, 0), (1, -1))
    keep = np.zeros(mag.shape, dtype=bool)
    for s, (dy, dx) in enumerate(steps):
        a = p[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
        b = p[1 - dy : 1 - dy + H, 1 - dx : 1 - dx + W]
        keep |= (sector == s) & (c >= a) & (c >= b)
    return np.where(keep, mag, 0.0).astype(np.float32)


def soft_map_nms(gray: np.ndarray) -> np.ndarray:
    gx, gy = gradients(gray)
    mag = np.clip(cv2.magnitude(gx, gy) / STEP, 0.0, 1.0)
    return nms(mag, gx, gy)
