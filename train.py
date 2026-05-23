r"""train.py — harmonic-contour-integration: L1 hypercolumns + render.

Pipeline per image: L0 → L1 (K-bin hypercolumns, learned ``η_z``, GABA recurrence,
dominant θ/ρ/κ for the cell grid) → cache → **renderer** (interp + thinning MLP).
The seed module only forwards cached ``lam`` as scalar ρ for the renderer.

Loss combines soft-Dice and per-pixel BCE on the η± valid band.
"""

from __future__ import annotations

import argparse, gc, glob, json, os, time
from multiprocessing import Pool, cpu_count

import numpy as np
import scipy.io as sio
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hci.L0 import (
    load_image,
    compute_l0_rgb,
    compute_interior,
    _compute_d_lum_chroma,
    _naka_per_direction,
    compute_harmonics,
    compute_seed,
)
from hci.L1 import (
    HypercolumnSeed,
    pad_for_patch_grid,
    run_l1_hypercolumn,
    z_from_l0_harmonics,
)
from hci.seed import RhoSeedModule
from hci.renderer import (
    ModulationRenderer,
    compute_render_features,
    render_boundary_map_torch,
    proj_to_device,
)
from params import L0, L1, RENDER, SEED, TRAIN, VIZ


def build_l0_pix(
    s_np: np.ndarray,
    h1m: torch.Tensor,
    h2m: torch.Tensor,
    border_mask_np: np.ndarray,
    h2m_lum: torch.Tensor | None = None,
    h2m_chr: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    h1 = h1m.detach().cpu().numpy().astype(np.float32)
    h2 = h2m.detach().cpu().numpy().astype(np.float32)
    h1[border_mask_np] = 0.0
    h2[border_mask_np] = 0.0
    z1_re = s_np[..., 0].astype(np.float32)
    z1_im = s_np[..., 1].astype(np.float32)
    z2_re = s_np[..., 2].astype(np.float32)
    z2_im = s_np[..., 3].astype(np.float32)
    z1_re[border_mask_np] = 0.0
    z1_im[border_mask_np] = 0.0
    z2_re[border_mask_np] = 0.0
    z2_im[border_mask_np] = 0.0
    out: dict[str, torch.Tensor] = {
        "h1m": torch.from_numpy(h1),
        "h2m": torch.from_numpy(h2),
        "z1_re": torch.from_numpy(z1_re),
        "z1_im": torch.from_numpy(z1_im),
        "z2_re": torch.from_numpy(z2_re),
        "z2_im": torch.from_numpy(z2_im),
    }
    if h2m_lum is not None:
        hl = h2m_lum.detach().cpu().numpy().astype(np.float32)
        hl[border_mask_np] = 0.0
        out["h2m_lum"] = torch.from_numpy(hl)
    if h2m_chr is not None:
        hc = h2m_chr.detach().cpu().numpy().astype(np.float32)
        hc[border_mask_np] = 0.0
        out["h2m_chr"] = torch.from_numpy(hc)
    return out


def fast_l0_pass2(
    d_lum: torch.Tensor,
    d_chr: torch.Tensor,
    eta_lum_map: torch.Tensor,
    eta_chr_map: torch.Tensor,
    gamma: float,
    offsets: list[tuple[int, int]],
    border_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fast L0 pass 2: reuse precomputed d_lum/d_chr, apply new per-pixel η.

    Skips the expensive directional-difference computation (which depends
    only on the image, not on η).  Only reapplies the Naka–Rushton with
    spatially-varying η and recomputes harmonics.

    Returns a minimal ``l0_pix``-style dict with **differentiable**
    ``h2m_lum`` / ``h2m_chr`` tensors only.  ``render_boundary_map_torch``
    reads just those keys for the pixel feature stack; we intentionally do
    **not** call ``build_l0_pix`` here (its numpy round-trip would detach
    ``η_mod`` from the autograd graph during training).
    """
    device = d_lum.device

    h_lum = _naka_per_direction(d_lum, eta_lum_map, gamma, device)
    h_chr = _naka_per_direction(d_chr, eta_chr_map, gamma, device)
    _, _, h2m_lum = compute_harmonics(h_lum, offsets)
    _, _, h2m_chr = compute_harmonics(h_chr, offsets)

    h2m_lum = h2m_lum.clone()
    h2m_lum[border_mask] = 0.0
    h2m_chr = h2m_chr.clone()
    h2m_chr[border_mask] = 0.0

    return {"h2m_lum": h2m_lum, "h2m_chr": h2m_chr}


def build_cells_flat(cells: dict) -> dict:
    nH, nW = cells["nH"], cells["nW"]
    N = nH * nW
    result = {
        "nH": nH,
        "nW": nW,
        "theta": torch.from_numpy(cells["theta"].reshape(N, 2).astype(np.float32)),
        "q": torch.from_numpy(cells["q"].reshape(N, 2).astype(np.float32)),
        "kappa": torch.from_numpy(cells["kappa"].reshape(N, 2).astype(np.float32)),
        "z1_abs_sum": torch.from_numpy(cells["z1_abs_sum"].reshape(N).astype(np.float32)),
        "lam": torch.from_numpy(cells["lam"].reshape(N, 2).astype(np.float32)),
        "lam3": torch.from_numpy(cells["lam3"].reshape(N).astype(np.float32)),
        "z0": torch.from_numpy(cells["z0"].reshape(N).astype(np.float32)),
        "cx_z2": torch.from_numpy(cells["cx_z2"].reshape(N).astype(np.float32)),
        "cy_z2": torch.from_numpy(cells["cy_z2"].reshape(N).astype(np.float32)),
        "is_border": torch.from_numpy(cells["is_border"].reshape(N).astype(np.bool_)),
    }
    result["kappa_col_cell"] = torch.from_numpy(
        np.asarray(cells["kappa_col_cell"], dtype=np.float32)
    )
    result["e_col_cell"] = torch.from_numpy(
        np.asarray(cells["e_col_cell"], dtype=np.float32)
    )
    return result


class HarmonicContourE2E(nn.Module):
    def __init__(
        self,
        r_pool: int,
        stride: int,
        eps: float,
        eta_z_init: float = SEED.ETA_Z_INIT,
        render_cell_hidden: int = RENDER.CELL_HIDDEN,
        render_pixel_hidden: int = RENDER.PIXEL_HIDDEN,
        eta_mod_enabled: bool = True,
    ):
        super().__init__()
        _ = render_cell_hidden
        self.seed = RhoSeedModule(
            r_pool=r_pool,
            stride=stride,
            eps=eps,
            eta_z_init=eta_z_init,
        )
        self.renderer = ModulationRenderer(hidden=render_pixel_hidden)
        self.eps = eps
        self.render_eps = max(float(eps), 1e-6)

        # Learned η modulation: η₀·σ(a - b·κ + c·Ē_col)
        # Init: a=2, b=c=0 → σ(2) ≈ 0.88 → near-identity
        self.eta_mod_enabled = eta_mod_enabled
        if eta_mod_enabled:
            self.eta_mod_a = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
            self.eta_mod_b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.eta_mod_c = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward_batch(self, meta_list):
        from hci.L0 import compute_eta_modulation

        bmaps = []
        for m in meta_list:
            cf_flat = m["cells_flat_dev"]
            rho_out, branch, _, _, _, _, _ = self.seed(cells_flat=cf_flat)

            # Pass 1: render with precomputed l0_pix to get κ_col and E_col
            bmap = render_boundary_map_torch(
                rho_out,
                m["proj_dev"],
                self.renderer,
                cf_flat,
                m["Hp"],
                m["Wp"],
                m["l0_pix"],
                eps=self.render_eps,
                training=self.training,
                branch_pick=branch.reshape(-1).long(),
                content_h=m["H0"],
                content_w=m["W0"],
            )

            # Pass 2: η modulation if enabled and data available
            has_pass2_data = (
                self.eta_mod_enabled
                and "d_lum" in m
                and "d_chr" in m
                and "border_mask" in m
            )
            if has_pass2_data:
                nH = int(cf_flat["nH"])
                nW = int(cf_flat["nW"])
                S = int(cf_flat.get("S", max(1, m["proj_dev"]["W"] // max(nW, 1))))
                P = int(cf_flat.get("P", S + (S - 1)))
                device = rho_out.device
                dtype = rho_out.dtype
                ib_grid = cf_flat["is_border"].to(device=device).reshape(nH, nW).bool()

                kappa_col = cf_flat["kappa_col_cell"].to(
                    device=device, dtype=dtype,
                ).reshape(nH, nW)
                e_col = cf_flat["e_col_cell"].to(
                    device=device, dtype=dtype,
                ).reshape(nH, nW)

                # Compute per-pixel η maps
                H, W = m["proj_dev"]["H"], m["proj_dev"]["W"]
                eta_lum_map, eta_chr_map = compute_eta_modulation(
                    kappa_col, e_col, ib_grid,
                    nH, nW, H, W, S, P,
                    eta0_lum=L0.ETA_LUM,
                    eta0_chr=L0.ETA_CHR,
                    a=self.eta_mod_a,
                    b=self.eta_mod_b,
                    c=self.eta_mod_c,
                )

                # Fast L0 pass 2: reuse d_lum/d_chr, apply modulated η
                l0_pix_pass2 = fast_l0_pass2(
                    m["d_lum"], m["d_chr"],
                    eta_lum_map, eta_chr_map,
                    L0.GAMMA, L0.OFFSETS,
                    m["border_mask"],
                )
                # Detach NR/harmonics path (avoids fragile grad through atan2/NR);
                # keep η map in-graph for thinning MLP feature 17 → η_mod a,b,c.
                l0_pix_pass2 = {
                    k: v.to(device).detach() for k, v in l0_pix_pass2.items()
                }
                l0_pix_pass2["eta_mod_map"] = eta_lum_map.to(
                    device=device, dtype=rho_out.dtype,
                )

                # Re-render with updated h2m + η feature
                bmap = render_boundary_map_torch(
                    rho_out,
                    m["proj_dev"],
                    self.renderer,
                    cf_flat,
                    m["Hp"],
                    m["Wp"],
                    l0_pix_pass2,
                    eps=self.render_eps,
                    training=self.training,
                    branch_pick=branch.reshape(-1).long(),
                    content_h=m["H0"],
                    content_w=m["W0"],
                )

            bmaps.append(bmap)
        return bmaps


def prepare_batch(items, device):
    meta = []
    for item in items:
        (gi, gt, Hp, Wp, Hg, Wg, H0, W0, cells_flat, l0_pix, img,
         d_lum, d_chr, border_mask) = item
        cf_dev = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in cells_flat.items()
        }
        l0_dev = {
            k: v.to(device) for k, v in l0_pix.items()
        }
        p_dev = proj_to_device(gi, device)
        m = {
            "nH": cells_flat["nH"],
            "nW": cells_flat["nW"],
            "proj_dev": p_dev,
            "gt": gt.to(device),
            "Hp": Hp,
            "Wp": Wp,
            "Hg": Hg,
            "Wg": Wg,
            "H0": H0,
            "W0": W0,
            "cells_flat_dev": cf_dev,
            "l0_pix": l0_dev,
            "img": img,
        }
        # Pass-2 η modulation data (may be None when absent)
        if d_lum is not None:
            m["d_lum"] = d_lum.to(device)
        if d_chr is not None:
            m["d_chr"] = d_chr.to(device)
        if border_mask is not None:
            m["border_mask"] = border_mask.to(device)
        meta.append(m)
    return meta


def load_bsds_gt(gt_path):
    m = sio.loadmat(gt_path)
    gt = m["groundTruth"]
    combined = None
    for i in range(gt.shape[1]):
        b = gt[0, i]["Boundaries"][0, 0].astype(np.float32)
        combined = b if combined is None else np.maximum(combined, b)
    return combined


def load_png_gt(gt_path):
    return np.array(Image.open(gt_path).convert("L")).astype(np.float32) / 255.0


def _find_gt_path_png(gt_dir, stem):
    for ext in [".png", ".jpg"]:
        p = os.path.join(gt_dir, stem + ext)
        if os.path.exists(p):
            return p
    matches = glob.glob(os.path.join(gt_dir, f"{stem}*"))
    return matches[0] if matches else None


def precompute_image(img_path, gt_path, gt_format):
    ir_np = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
    ir_np = np.clip(ir_np, 0, 1)
    ir_p, H0, W0 = pad_for_patch_grid(ir_np, L1.PATCH_SIZE, L1.PATCH_OVERLAP)
    del ir_np

    ir_t = torch.from_numpy(ir_p)

    # Compute directional differences (image-dependent, η-independent)
    d_lum, d_chr = _compute_d_lum_chroma(ir_t, L0.OFFSETS)

    h, vld, _, _, _, s, h1m, h2m, h2m_lum, h2m_chr = compute_l0_rgb(
        ir_t,
        eta_lum=L0.ETA_LUM,
        eta_chr=L0.ETA_CHR,
        gamma=L0.GAMMA,
        offsets=L0.OFFSETS,
    )
    border_mask_t = ~compute_interior(ir_p.shape[0], ir_p.shape[1], ir_t.device)

    z1, z2 = z_from_l0_harmonics(s, border_mask_t)
    theta_h = (0.5 * torch.atan2(s[..., 3], s[..., 2])).to(torch.float32)
    theta_h = torch.where(border_mask_t, torch.zeros_like(theta_h), theta_h)

    s_np = s.cpu().numpy()
    border_mask_np = border_mask_t.cpu().numpy()
    z2_image = (s_np[..., 2] + 1j * s_np[..., 3]).astype(np.complex64)
    z2_image[border_mask_np] = 0.0

    l0_pix = build_l0_pix(
        s_np, h1m, h2m, border_mask_np, h2m_lum=h2m_lum, h2m_chr=h2m_chr,
    )
    del h, vld, s, h1m, h2m_lum, h2m_chr
    gc.collect()

    # Standalone η_z (same init as SEED); cache is geometry-only vs trained ckpt.
    hc_seed = HypercolumnSeed(
        r_pool=SEED.R_POOL,
        stride=SEED.STRIDE,
        eps=SEED.EPS,
        eta_z_init=SEED.ETA_Z_INIT,
    )
    cells = run_l1_hypercolumn(
        h2m,
        theta_h,
        border_mask_t,
        hc_seed,
        P=L1.PATCH_SIZE,
        patch_overlap=L1.PATCH_OVERLAP,
        border_patch_max_frac=L1.BORDER_PATCH_MAX_FRAC,
        verbose=False,
        eps=float(L1.EPS),
    )
    del z1, z2, h2m, theta_h, ir_t
    gc.collect()

    P = cells["P"]
    cells["is_border"] |= (cells["cy"] + P / 2 > H0) | (cells["cx"] + P / 2 > W0)

    nH, nW = cells["nH"], cells["nW"]

    proj_info = compute_render_features(
        z2_image,
        ir_p,
        cells,
        border_mask_np,
        eps=SEED.EPS,
    )
    del z2_image, border_mask_np
    gc.collect()

    if gt_format == "mat":
        gt = load_bsds_gt(gt_path)
    else:
        gt = load_png_gt(gt_path)

    H_p, W_p = ir_p.shape[:2]
    H_gt, W_gt = gt.shape

    cells_flat = build_cells_flat(cells)

    img_cached = ir_p.astype(np.float32)
    del cells, ir_p
    gc.collect()

    return {
        "cache_version": TRAIN.CACHE_VERSION,
        "proj_info": proj_info,
        "gt": torch.from_numpy(gt),
        "H_p": H_p,
        "W_p": W_p,
        "H_gt": H_gt,
        "W_gt": W_gt,
        "H0": H0,
        "W0": W0,
        "cells_flat": cells_flat,
        "l0_pix": l0_pix,
        "img": img_cached,
        "d_lum": d_lum.cpu(),
        "d_chr": d_chr.cpu(),
        "border_mask": border_mask_t.cpu(),
    }


def _precompute_one(args):
    img_path, gt_path, gt_format, cache_path = args
    try:
        data = precompute_image(img_path, gt_path, gt_format)
        torch.save(data, cache_path)
        return os.path.splitext(os.path.basename(img_path))[0]
    except Exception as e:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  {stem}: error ({e})")
        return None


def _cache_entry_valid(data: dict) -> bool:
    if data.get("cache_version") != TRAIN.CACHE_VERSION:
        return False
    pi = data.get("proj_info")
    if not isinstance(pi, dict):
        return False
    need_pi = ("H", "W", "n_cells", "nH", "nW")
    if not all(k in pi for k in need_pi):
        return False
    cf = data.get("cells_flat")
    if not isinstance(cf, dict):
        return False
    for key in (
        "cx_z2",
        "cy_z2",
        "z0",
        "q",
        "kappa",
        "kappa_col_cell",
        "e_col_cell",
    ):
        if key not in cf:
            return False
    l0 = data.get("l0_pix")
    if not isinstance(l0, dict):
        return False
    for key in ("h1m", "h2m", "h2m_lum", "h2m_chr", "z1_re", "z1_im", "z2_re", "z2_im"):
        if key not in l0:
            return False
    return True


def _load_cache_entry(cache_path: str) -> dict | None:
    try:
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def precompute_split(
    image_dir,
    gt_dir,
    cache_dir,
    gt_format,
    max_images=None,
    n_workers=None,
):
    os.makedirs(cache_dir, exist_ok=True)
    img_files = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
    )
    if gt_format == "png" and os.path.abspath(image_dir) == os.path.abspath(gt_dir):
        img_files = [f for f in img_files if f.endswith(".jpg")]
    if max_images is not None:
        img_files = img_files[:max_images]

    stems = []
    work = []
    stale = 0
    for img_path in img_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        cache_path = os.path.join(cache_dir, f"{stem}.pt")
        if os.path.exists(cache_path):
            cached = _load_cache_entry(cache_path)
            if cached is not None and _cache_entry_valid(cached):
                stems.append(stem)
                continue
            stale += 1
        if gt_format == "mat":
            gt_path = os.path.join(gt_dir, f"{stem}.mat")
            if not os.path.exists(gt_path):
                continue
        else:
            gt_path = _find_gt_path_png(gt_dir, stem)
            if gt_path is None:
                continue
        work.append((img_path, gt_path, gt_format, cache_path))

    if stale:
        print(f"  {stale} stale cache entries will be rebuilt")

    if work:
        if n_workers is None:
            n_workers = min(cpu_count(), len(work), 8)
        print(f"  {len(work)} images to precompute ({n_workers} workers)")
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_precompute_one, work):
                if result is not None:
                    stems.append(result)
                    print(f"  cached {result} ({len(stems)} done)")
    elif not stems:
        n_imgs = len(img_files)
        sample = [os.path.basename(f) for f in img_files[:3]]
        print(f"  found {n_imgs} images in {image_dir} (e.g. {sample})")
        print(f"  but no matching GT in {gt_dir} (format={gt_format})")
        gt_sample = (
            sorted(os.listdir(gt_dir))[:5]
            if os.path.isdir(gt_dir)
            else ["<dir not found>"]
        )
        print(f"  GT dir contents: {gt_sample}")
    return stems


class StriateDataset(Dataset):
    def __init__(self, cache_dir, stems):
        self.cache_dir = cache_dir
        self.stems = stems

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        data = torch.load(
            os.path.join(self.cache_dir, f"{stem}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if not _cache_entry_valid(data):
            raise RuntimeError(
                f'Cache for "{stem}" is stale or incomplete '
                f"(cache_version={data.get('cache_version')!r}, "
                f"expected {TRAIN.CACHE_VERSION}; "
                f"proj_info must include render grid keys). "
                f"Delete {self.cache_dir} and re-run training."
            )
        H0 = data.get("H0", data["H_gt"])
        W0 = data.get("W0", data["W_gt"])
        return (
            data["proj_info"],
            data["gt"],
            data["H_p"],
            data["W_p"],
            data["H_gt"],
            data["W_gt"],
            H0,
            W0,
            data["cells_flat"],
            data["l0_pix"],
            data.get("img"),
            data.get("d_lum"),
            data.get("d_chr"),
            data.get("border_mask"),
        )


def collate_fn(batch):
    return batch


def soft_dice_loss_with_ignore(
    pred: torch.Tensor,
    target: torch.Tensor,
    eta_pos: float = 0.5,
    eta_neg: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable soft Dice on non-ignored pixels (η± edge band).

    Valid = (target≥η_pos) ∨ (target<η_neg); y = 1 on positives, 0 on negatives.
    D = 2·Σ(v·p·y) / (Σ(v·p) + Σ(v·y) + ε);  L = 1 − D.
    """
    pred = pred.nan_to_num(0.0).clamp(eps, 1.0 - eps)
    pos_mask = target >= eta_pos
    neg_mask = target < eta_neg
    valid = pos_mask | neg_mask
    if not valid.any():
        return pred.sum() * 0.0
    v = valid.float()
    y = pos_mask.float()
    vp = v * pred
    inter = (vp * y).sum()
    denom = vp.sum() + (v * y).sum() + eps
    dice = (2.0 * inter) / denom
    return 1.0 - dice


def bce_loss_with_ignore(
    pred: torch.Tensor,
    target: torch.Tensor,
    eta_pos: float = 0.5,
    eta_neg: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean BCE on non-ignored pixels (same η± edge band as soft Dice).

    Valid = (target≥η_pos) ∨ (target<η_neg); label y = 1 on positives, 0 on negatives.
    """
    pred = pred.nan_to_num(0.0).clamp(eps, 1.0 - eps)
    pos_mask = target >= eta_pos
    neg_mask = target < eta_neg
    valid = pos_mask | neg_mask
    if not valid.any():
        return pred.sum() * 0.0
    y = pos_mask.float()
    v = valid.float()
    bce = F.binary_cross_entropy(pred, y, reduction="none")
    return (bce * v).sum() / v.sum().clamp_min(eps)


def remap_checkpoint_state_dict(sd: dict) -> dict:
    """Load STRIATE ``dynamics._eta_z`` into ``seed.hc_seed``; drop other dynamics keys.

    ``seed._eta_rho_raw`` / ``dynamics._eta_rho_raw`` are omitted — NR pool on
    the seed was removed; those tensors are not loaded.  ``eta_mod_d`` is
    dropped if present (removed from the model).

    Learned η_z lives on ``HypercolumnSeed`` inside ``RhoSeedModule``; legacy
    checkpoints may use ``dynamics._eta_z_raw`` or flat ``seed._eta_z_raw``.
    """
    skip = frozenset({"seed._eta_rho_raw", "dynamics._eta_rho_raw"})
    out: dict = {}
    for k, v in sd.items():
        if k in skip:
            continue
        if k == "eta_mod_d":
            continue
        if k == "dynamics._eta_z_raw":
            out["seed.hc_seed._eta_z_raw"] = v
        elif k == "seed._eta_z_raw":
            out["seed.hc_seed._eta_z_raw"] = v
        elif k.startswith("dynamics."):
            continue
        else:
            out[k] = v
    return out


def debug_drive_batch(model, meta_list, device, *, lam_dice=1.0, lam_bce=0.0):
    """One training batch: seed/renderer grads sanity check."""
    model.train()
    s = model.seed
    meta = meta_list[0]
    cf = meta["cells_flat_dev"]

    model.zero_grad(set_to_none=True)
    rho_out, _, _, _, _, cf_out, _ = s(cells_flat=cf)
    bmap = render_boundary_map_torch(
        rho_out,
        meta["proj_dev"],
        model.renderer,
        cf_out,
        meta["Hp"],
        meta["Wp"],
        meta["l0_pix"],
        eps=model.render_eps,
        training=True,
        content_h=meta["H0"],
        content_w=meta["W0"],
    )
    Hg, Wg = meta["Hg"], meta["Wg"]
    loss = lam_dice * soft_dice_loss_with_ignore(
        bmap[:Hg, :Wg], meta["gt"][:Hg, :Wg],
    )
    if lam_bce:
        loss = loss + lam_bce * bce_loss_with_ignore(
            bmap[:Hg, :Wg], meta["gt"][:Hg, :Wg],
        )
    loss.backward()

    print("\n--- seed + renderer grad debug ---")
    print("\n  learned (value):")
    print(f"    eta_z={float(s.eta_z.detach()):.4f}")
    print("\n  |grad| on raw params:")
    for raw, label in (("_eta_z_raw", "eta_z"),):
        t = getattr(s, raw, None)
        if t is None or t.grad is None:
            print(f"    {label}: grad=None")
        else:
            print(f"    {label}: |grad|={t.grad.abs().mean().item():.2e}")
    for label, t in (
        ("s_t", model.renderer.s_t),
        ("s_n", model.renderer.s_n),
    ):
        if t.grad is None:
            print(f"    {label}: grad=None")
        else:
            print(f"    {label}: |grad|={t.grad.abs().mean().item():.2e}")
    for name, t in model.renderer.thinning.named_parameters():
        if t.grad is None:
            print(f"    thinning.{name}: grad=None")
        else:
            print(f"    thinning.{name}: |grad|={t.grad.abs().mean().item():.2e}")
    if hasattr(model, "eta_mod_a"):
        for label in ("eta_mod_a", "eta_mod_b", "eta_mod_c"):
            t = getattr(model, label, None)
            if t is None:
                continue
            g = "None" if t.grad is None else f"|grad|={t.grad.abs().mean().item():.2e}"
            print(f"    {label}={t.item():.3f}  {g}")
    print(f"\n  loss={loss.item():.4f}  requires_grad={loss.requires_grad}\n")


def plot_training_curves(history, out_dir):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=VIZ.BG)
    fig.suptitle(
        "harmonic-contour-integration  L1 seed + render",
        fontsize=12,
        color=VIZ.FG,
        fontfamily="monospace",
    )
    for ax in axes.ravel():
        ax.set_facecolor("#111")
        ax.tick_params(labelsize=7, colors="#888")
        ax.grid(True, alpha=0.15, color="#444")
        for s in ax.spines.values():
            s.set_color("#333")
    dice_key = "loss_soft_dice"
    if history and dice_key not in history[0] and "loss_soft_f1" in history[0]:
        dice_key = "loss_soft_f1"
    axes[0].plot(epochs, losses, color="#66aaff", lw=1.3, label="combined")
    if history and dice_key in history[0]:
        axes[0].plot(
            epochs,
            [h[dice_key] for h in history],
            color="#ffaa88",
            lw=1.0,
            label="soft Dice (mean)",
        )
    if history and "loss_bce" in history[0]:
        axes[0].plot(
            epochs,
            [h["loss_bce"] for h in history],
            color="#88ffaa",
            lw=1.0,
            label="BCE (mean)",
        )
    axes[0].legend(fontsize=7, facecolor="#111", edgecolor="#333", labelcolor=VIZ.FG)
    axes[0].set_title("train loss", fontsize=9, color=VIZ.FG, fontfamily="monospace")
    axes[1].plot(epochs, lrs, color="#ffaa44", lw=1.2)
    axes[1].set_title("LR", fontsize=9, color=VIZ.FG, fontfamily="monospace")
    axes[1].set_yscale("log")
    plt.tight_layout()
    p = os.path.join(out_dir, "training_curves.png")
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor=VIZ.BG)
    plt.close(fig)
    print(f"  saved {p}")


def format_seed_param_lines(seed: RhoSeedModule, *, indent: str = "  ") -> list[str]:
    """Learned NR seed scalars (infer / train logging)."""
    return [
        f"{indent}tile geometry:  R={seed.R}  stride={seed.stride}",
        f"{indent}seed ratio:  η_z={seed.eta_z.item():.3f}",
    ]


def format_renderer_param_lines(r: ModulationRenderer, *, indent: str = "  ") -> list[str]:
    n_th = sum(p.numel() for p in r.thinning.parameters())
    return [
        f"{indent}harmonic-native:  h2m · gate(MLP)  (no Gaussian splat)",
        f"{indent}stencil spacings:  s_t={r.s_t.item():.3f}  s_n={r.s_n.item():.3f}",
        f"{indent}thinning head:  18→16→1 MLP  ({n_th} params)",
        f"{indent}κ_col / E_col:  supplied from L1 hypercolumn (cached in cells_flat)",
    ]


def _format_seed_block(model: HarmonicContourE2E) -> str:
    s = model.seed
    parts = [
        "\n--- L1 seed (NR) ---\n",
        *[ln + "\n" for ln in format_seed_param_lines(s, indent="")],
        "\nρ = NR_pool(λ₁/(z₀+η_z)) × tile_interior  (no Allen–Cahn refine)\n",
    ]
    return "".join(parts)


def format_model_param_counts(model: HarmonicContourE2E):
    n_seed = sum(p.numel() for p in model.seed.parameters())
    n_renderer = sum(p.numel() for p in model.renderer.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    return n_total, n_seed, n_renderer


def _format_render_params(model: HarmonicContourE2E):
    r = model.renderer
    n_r = sum(p.numel() for p in r.parameters())
    eta_str = ""
    if hasattr(model, "eta_mod_a"):
        eta_str = (f" + η-mod σ(a={model.eta_mod_a.item():.1f},"
                   f"b={model.eta_mod_b.item():.1f},"
                   f"c={model.eta_mod_c.item():.1f})")
    return (f"renderer={n_r} params  (harmonic-native, s_t, s_n, "
            f"thinning 18→16→1{eta_str})")


def save_checkpoint(model, path):
    torch.save({"model_state": model.state_dict()}, path)


_ETA_MOD_STATE_KEYS = frozenset({"eta_mod_a", "eta_mod_b", "eta_mod_c"})


def report_checkpoint_compatibility(incompatible, context="checkpoint load"):
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if not missing and not unexpected:
        return
    print(f"[{context}] state_dict compatibility:")
    if missing:
        print(f"  missing_keys ({len(missing)}): {missing}")
    if unexpected:
        print(f"  unexpected_keys ({len(unexpected)}): {unexpected}")
    miss_eta = sorted(_ETA_MOD_STATE_KEYS.intersection(missing))
    if miss_eta:
        print(
            f"[{context}] WARNING: checkpoint is missing η-mod weights {miss_eta}. "
            "Those parameters were not loaded and still use PyTorch init values "
            "(a=2.0, b=0.0, c=0.0). Pass-2 η maps then reflect only that default σ, "
            "not a trained modulation. Retrain with the current train.py to learn "
            "eta_mod_a/b/c and write them into the checkpoint."
        )


def _detect_gt_format(gt_dir):
    if glob.glob(os.path.join(gt_dir, "*.mat")):
        return "mat"
    return "png"


def _format_hms(seconds: float) -> str:
    s = int(round(max(0.0, float(seconds))))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def main():
    wall_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_imgs", default="data/train/imgs")
    ap.add_argument("--train_gt", default="data/train/gt")
    ap.add_argument("--gt_format", default=None)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--checkpoints_dir", default="output/checkpoints")
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=TRAIN.EPOCHS)
    ap.add_argument("--lr", type=float, default=TRAIN.LR)
    ap.add_argument("--batch_size", type=int, default=TRAIN.BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=TRAIN.NUM_WORKERS)
    ap.add_argument("--grad_clip", type=float, default=TRAIN.GRAD_CLIP)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--lam_dice",
        type=float,
        default=TRAIN.LAM_DICE,
        help="weight on soft-Dice term (0 disables)",
    )
    ap.add_argument(
        "--lam_bce",
        type=float,
        default=TRAIN.LAM_BCE,
        help="weight on BCE term (0 disables)",
    )
    ap.add_argument(
        "--debug-drive",
        action="store_true",
        help="Run one batch, print drive term stats and param gradients, then exit",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    gt_format = args.gt_format or _detect_gt_format(args.train_gt)

    mt = args.max_train if args.max_train is not None else "all"
    print(
        f"device={device}  gt_format={gt_format}  batch={args.batch_size}"
        f"  max_train={mt}"
    )
    print(
        f"Seed: R={SEED.R_POOL}  stride={SEED.STRIDE}  "
        f"ρ = (λ₁/(z₀+η_z)) × tile_interior  (learned η_z; no NR pool on seed; no dynamics)"
    )
    print(
        f"Render: harmonic-native h2m·gate  "
        f"(L1 collinear GABA: R={L1.COL_RADIUS}, K={L1.COL_K_BINS}, passes={L1.COL_PASSES})"
    )
    print(
        f"Loss: λ_dice·soft-Dice + λ_bce·BCE (η± edge band)  "
        f"λ_dice={args.lam_dice:g}  λ_bce={args.lam_bce:g}"
    )
    if args.lam_dice == 0.0 and args.lam_bce == 0.0:
        print("  warning: both lambdas are 0 — loss is identically zero")
    print(
        f"L0 (precompute, fixed): η_lum={L0.ETA_LUM}  η_chr={L0.ETA_CHR}  "
        f"γ={L0.GAMMA}  (tune in params.py)"
    )

    train_cache = os.path.join(args.cache_dir, "train")
    print(f"\nprecomputing train...")
    fit_stems = precompute_split(
        args.train_imgs,
        args.train_gt,
        train_cache,
        gt_format,
        max_images=args.max_train,
    )
    print(f"  {len(fit_stems)} cached")
    if not fit_stems:
        print(f"\nerror: no training images found.")
        return
    print(f"  training on all {len(fit_stems)} images (no held-out val)")

    model = HarmonicContourE2E(
        r_pool=SEED.R_POOL,
        stride=SEED.STRIDE,
        eps=SEED.EPS,
        eta_z_init=SEED.ETA_Z_INIT,
        render_cell_hidden=RENDER.CELL_HIDDEN,
        render_pixel_hidden=RENDER.PIXEL_HIDDEN,
    ).to(device)

    active_params = list(model.parameters())
    optimizer = torch.optim.Adam(active_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    n_tot, n_seed, n_r = format_model_param_counts(model)
    print(
        f"\nmodel: {n_tot} params total  Adam: {n_tot} "
        f"(seed {n_seed} + renderer {n_r})"
    )

    train_ds = StriateDataset(train_cache, fit_stems)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if args.debug_drive:
        batch = next(iter(train_loader))
        moved = []
        for item in batch:
            (gi, gt, Hp, Wp, Hg, Wg, H0, W0, cells_flat, l0_pix, img, d_lum, d_chr, border_mask) = item
            moved.append((gi, gt, Hp, Wp, Hg, Wg, H0, W0, cells_flat, l0_pix, img, d_lum, d_chr, border_mask))
        debug_drive_batch(
            model,
            prepare_batch(moved, device),
            device,
            lam_dice=args.lam_dice,
            lam_bce=args.lam_bce,
        )
        return

    print(f"\ntraining ({args.epochs} epochs)...\n")
    history = []
    n_fit = len(fit_stems)
    dbg_img_step = max(1, n_fit // 5)

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        ep_soft_dice = 0.0
        ep_bce = 0.0
        n_img = 0
        n_batch = 0
        t0 = time.time()
        next_dbg_img = dbg_img_step

        for batch in train_loader:
            moved = []
            for item in batch:
                (gi, gt, Hp, Wp, Hg, Wg, H0, W0, cells_flat, l0_pix, img, d_lum, d_chr, border_mask) = item
                moved.append((gi, gt, Hp, Wp, Hg, Wg, H0, W0, cells_flat, l0_pix, img, d_lum, d_chr, border_mask))

            meta_list = prepare_batch(moved, device)
            bmaps = model.forward_batch(meta_list)

            dice_sum = None
            bce_sum = None
            for bmap, m in zip(bmaps, meta_list):
                Hg, Wg = m["Hg"], m["Wg"]
                bc = bmap[:Hg, :Wg]
                gc_ = m["gt"][:Hg, :Wg]
                loss_dice = soft_dice_loss_with_ignore(bc, gc_)
                loss_bce = bce_loss_with_ignore(bc, gc_)
                dice_sum = loss_dice if dice_sum is None else dice_sum + loss_dice
                bce_sum = loss_bce if bce_sum is None else bce_sum + loss_bce
            n_bm = len(bmaps)
            mean_dice = dice_sum / n_bm
            mean_bce = bce_sum / n_bm
            loss = args.lam_dice * mean_dice + args.lam_bce * mean_bce
            soft_dice_mean = mean_dice
            bce_mean = mean_bce

            if not loss.requires_grad:
                raise RuntimeError(
                    "loss has no grad_fn — check λ_dice/λ_bce, GT η band, ρ/renderer graph"
                )

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(active_params, args.grad_clip)
            optimizer.step()

            n_img += len(meta_list)
            ep_loss += loss.item()
            ep_soft_dice += soft_dice_mean.item()
            ep_bce += bce_mean.item()
            n_batch += 1
            loss_item = loss.item()
            while n_img >= next_dbg_img:
                dt_p = time.time() - t0
                print(
                    f"    [dbg] epoch {epoch + 1}/{args.epochs}  ~img {next_dbg_img}/{n_fit}"
                    f"  batch {n_batch}  loss(run)={ep_loss / n_batch:.4f}"
                    f"  batch_loss={loss_item:.4f}"
                    f"  dice={soft_dice_mean.item():.4f}  bce={bce_mean.item():.4f}"
                    f"  {dt_p:.1f}s elapsed",
                    flush=True,
                )
                next_dbg_img += dbg_img_step
            del meta_list, bmaps, loss, moved

        scheduler.step()
        al = ep_loss / max(n_batch, 1)
        al_sdice = ep_soft_dice / max(n_batch, 1)
        al_bce = ep_bce / max(n_batch, 1)
        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        print(
            f"  epoch {epoch + 1:3d}/{args.epochs}:  "
            f"loss={al:.4f}  dice={al_sdice:.4f}  bce={al_bce:.4f}  "
            f"lr={lr_now:.2e}  {dt:.1f}s"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": al,
                "loss_soft_dice": al_sdice,
                "loss_bce": al_bce,
                "lam_dice": args.lam_dice,
                "lam_bce": args.lam_bce,
                "lr": lr_now,
                "time": dt,
            }
        )

        intermediate_path = os.path.join(args.checkpoints_dir, "intermediate.pt")
        save_checkpoint(model, intermediate_path)
        print(f"  saved {intermediate_path}")

    model_path = os.path.join(args.checkpoints_dir, "final.pt")
    save_checkpoint(model, model_path)
    print(f"\nsaved {model_path}")

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, args.output_dir)

    wall_elapsed = time.time() - wall_start
    print(f"\nelapsed time: {_format_hms(wall_elapsed)} ({wall_elapsed:.1f}s)")
    print(f"\ndone -> {args.output_dir}/")


if __name__ == "__main__":
    main()