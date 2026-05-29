r"""train.py — STRIATE training: cell-grid conv dynamics + harmonic render.

Pipeline per image: L0 contrast/harmonics (cached) → L1 K-bin projection live each step
→ L2 cell-grid ρ refinement (ρ_raw + per-bin NR seed; lateral refine + mixing)
→ splat renderer ($\hat B = \bar\rho \cdot \mathrm{gate}$).
Loss combines soft-Dice and per-pixel BCE on the same η± valid band
(target≥η_pos or target<η_neg), weighted by ``--lam_dice`` and ``--lam_bce``
(each can be 0).

Trains on the full train split (no held-out val). Saves `intermediate.pt`
each epoch and `final.pt` when training completes.
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
)
from hci.L1 import (
    stride_from_patch_overlap,
    z_from_l0_harmonics,
    pad_for_patch_grid,
    run_l1,
)
from hci.L2 import TileDynamics, _inv_softplus
from hci.renderer import (
    ModulationRenderer,
    render_boundary_map_torch,
    proj_to_device,
    upgrade_renderer_state_dict,
)
from params import L0, L1, L2, RENDER, TRAIN, VIZ


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


def proj_info_from_grid(H: int, W: int, P: int, patch_overlap: int) -> dict:
    """Render/L2 grid metadata without running L1."""
    S = stride_from_patch_overlap(P, patch_overlap)
    nH = (H - P) // S + 1 if H >= P else 0
    nW = (W - P) // S + 1 if W >= P else 0
    return {"H": H, "W": W, "n_cells": nH * nW, "nH": nH, "nW": nW}


def build_cells_flat_torch(cells: dict) -> dict:
    """Flatten ``run_l1(..., return_torch=True)`` grids for L2 / renderer."""
    nH, nW = int(cells["nH"]), int(cells["nW"])
    N = nH * nW
    return {
        "nH": nH,
        "nW": nW,
        "theta": cells["theta"].reshape(N, 1),
        "k_star": cells["k_star"].reshape(N),
        "q": cells["q"].reshape(N, 1),
        "kappa": cells["kappa"].reshape(N, 1),
        "z1_abs_sum": cells["z1_abs_sum"].reshape(N),
        "rho_peak": cells["rho_peak"].reshape(N),
        "rho_bins": cells["rho_bins"].reshape(N, -1),
        "z0": cells["z0"].reshape(N),
        "cx_z2": cells["cx_z2"].reshape(N),
        "cy_z2": cells["cy_z2"].reshape(N),
        "is_border": cells["is_border"].reshape(N).bool(),
    }


def run_l1_cells_flat(
    l0_pix: dict[str, torch.Tensor],
    border_mask: torch.Tensor,
    H0: int,
    W0: int,
    device: torch.device,
    *,
    von_mises_kappa: torch.Tensor,
) -> dict:
    """Live L1 from cached L0 (differentiable w.r.t. learned von Mises κ)."""
    h2m = l0_pix["h2m"].to(device=device, dtype=torch.float32)
    z1 = torch.complex(
        l0_pix["z1_re"].to(device=device, dtype=torch.float32),
        l0_pix["z1_im"].to(device=device, dtype=torch.float32),
    )
    z2 = torch.complex(
        l0_pix["z2_re"].to(device=device, dtype=torch.float32),
        l0_pix["z2_im"].to(device=device, dtype=torch.float32),
    )
    bm = border_mask.to(device=device).bool()
    cells = run_l1(
        h2m,
        z1,
        z2,
        L1.PATCH_SIZE,
        border_mask=bm,
        patch_overlap=L1.PATCH_OVERLAP,
        border_patch_max_frac=L1.BORDER_PATCH_MAX_FRAC,
        eps=L1.EPS,
        K=L1.K,
        bin_tuning=L1.COL_BIN_TUNING,
        cos_power=L1.COL_COS_POWER,
        von_mises_kappa=von_mises_kappa,
        device=device,
        verbose=False,
        return_torch=True,
    )
    P = int(cells["P"])
    cells["is_border"] = cells["is_border"] | (
        (cells["cy"] + P / 2 > H0) | (cells["cx"] + P / 2 > W0)
    )
    return build_cells_flat_torch(cells)


def build_cells_flat(cells: dict) -> dict:
    nH, nW = cells["nH"], cells["nW"]
    N = nH * nW
    return {
        "nH": nH,
        "nW": nW,
        "theta": torch.from_numpy(cells["theta"].reshape(N, 1).astype(np.float32)),
        "k_star": torch.from_numpy(cells["k_star"].reshape(N).astype(np.int64)),
        "q": torch.from_numpy(cells["q"].reshape(N, 1).astype(np.float32)),
        "kappa": torch.from_numpy(cells["kappa"].reshape(N, 1).astype(np.float32)),
        "z1_abs_sum": torch.from_numpy(cells["z1_abs_sum"].reshape(N).astype(np.float32)),
        "rho_peak": torch.from_numpy(cells["rho_peak"].reshape(N).astype(np.float32)),
        "rho_bins": torch.from_numpy(cells["rho_bins"].reshape(N, -1).astype(np.float32)),
        "z0": torch.from_numpy(cells["z0"].reshape(N).astype(np.float32)),
        "cx_z2": torch.from_numpy(cells["cx_z2"].reshape(N).astype(np.float32)),
        "cy_z2": torch.from_numpy(cells["cy_z2"].reshape(N).astype(np.float32)),
        "is_border": torch.from_numpy(cells["is_border"].reshape(N).astype(np.bool_)),
    }


class StriateE2E(nn.Module):
    def __init__(
        self,
        r_fac_pool: int,
        r_sup_pool: int,
        K: int,
        t_refine: int,
        eps: float,
        render_cell_hidden: int = RENDER.CELL_HIDDEN,
        render_pixel_hidden: int = RENDER.PIXEL_HIDDEN,
        **kw,
    ):
        super().__init__()
        _ = kw  # legacy kwargs (e.g. stride, r_pool, eta_z_init) ignored
        self.dynamics = TileDynamics(
            r_fac_pool=r_fac_pool,
            r_sup_pool=r_sup_pool,
            K=K,
            t_refine=t_refine,
            eps=eps,
            tbptt_n_segments=TRAIN.L2_SNAPSHOT_MAX,
        )
        _ = render_cell_hidden
        self.renderer = ModulationRenderer(hidden=render_pixel_hidden)
        self._l1_von_mises_kappa_raw = nn.Parameter(
            torch.tensor(
                _inv_softplus(float(L1.COL_VON_MISES_KAPPA)),
                dtype=torch.float32,
            )
        )
        self.eps = eps
        self.render_eps = max(float(eps), 1e-6)

    @property
    def l1_von_mises_kappa(self) -> torch.Tensor:
        return F.softplus(self._l1_von_mises_kappa_raw)

    def forward_batch(self, meta_list):
        bmaps = []
        for m in meta_list:
            cf_flat = m["cells_flat_dev"]
            rho_out, branch, _, _, _, cf_out, _ = self.dynamics(cells_flat=cf_flat)
            bmap = render_boundary_map_torch(
                rho_out,
                m["proj_dev"],
                self.renderer,
                cf_out,
                m["Hp"],
                m["Wp"],
                m["l0_pix"],
                eps=self.render_eps,
                training=self.training,
                branch_pick=branch.reshape(-1).long(),
                content_h=m["H0"],
                content_w=m["W0"],
            )
            bmaps.append(bmap)
        return bmaps


def prepare_batch(items, device, model: StriateE2E):
    meta = []
    for item in items:
        (gi, gt, Hp, Wp, Hg, Wg, H0, W0, l0_pix, border_mask, img) = item
        cf_dev = run_l1_cells_flat(
            l0_pix,
            border_mask,
            H0,
            W0,
            device,
            von_mises_kappa=model.l1_von_mises_kappa,
        )
        l0_dev = {
            k: v.to(device) for k, v in l0_pix.items()
        }
        p_dev = proj_to_device(gi, device)
        meta.append(
            {
                "nH": cf_dev["nH"],
                "nW": cf_dev["nW"],
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
        )
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
    h, vld, _, _, _, s, h1m, h2m, h2m_lum, h2m_chr = compute_l0_rgb(
        ir_t,
        eta_lum=L0.ETA_LUM,
        eta_chr=L0.ETA_CHR,
        gamma=L0.GAMMA,
        offsets=L0.OFFSETS,
    )
    border_mask_t = ~compute_interior(ir_p.shape[0], ir_p.shape[1], ir_t.device)

    s_np = s.cpu().numpy()
    border_mask_np = border_mask_t.cpu().numpy()

    l0_pix = build_l0_pix(
        s_np, h1m, h2m, border_mask_np, h2m_lum=h2m_lum, h2m_chr=h2m_chr,
    )
    del h, vld, s, h1m, h2m, h2m_lum, h2m_chr, border_mask_t
    gc.collect()

    H_p, W_p = ir_p.shape[:2]
    proj_info = proj_info_from_grid(
        H_p, W_p, L1.PATCH_SIZE, L1.PATCH_OVERLAP,
    )

    if gt_format == "mat":
        gt = load_bsds_gt(gt_path)
    else:
        gt = load_png_gt(gt_path)

    H_gt, W_gt = gt.shape

    img_cached = ir_p.astype(np.float32)
    del ir_p
    gc.collect()

    return {
        "l0_cache_version": TRAIN.L0_CACHE_VERSION,
        "proj_info": proj_info,
        "gt": torch.from_numpy(gt),
        "H_p": H_p,
        "W_p": W_p,
        "H_gt": H_gt,
        "W_gt": W_gt,
        "H0": H0,
        "W0": W0,
        "border_mask": torch.from_numpy(border_mask_np),
        "l0_pix": l0_pix,
        "img": img_cached,
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
    if data.get("l0_cache_version") != TRAIN.L0_CACHE_VERSION:
        return False
    pi = data.get("proj_info")
    if not isinstance(pi, dict):
        return False
    need_pi = ("H", "W", "n_cells", "nH", "nW")
    if not all(k in pi for k in need_pi):
        return False
    bm = data.get("border_mask")
    if not isinstance(bm, torch.Tensor):
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
                f"(l0_cache_version={data.get('l0_cache_version')!r}, "
                f"expected {TRAIN.L0_CACHE_VERSION}; "
                f"need l0_pix + border_mask + proj_info). "
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
            data["l0_pix"],
            data["border_mask"],
            data.get("img"),
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


def debug_drive_batch(model, meta_list, device, *, lam_dice=1.0, lam_bce=0.0):
    """One training batch: drive term stats + ∂loss/∂(b_coll, η_iso, …)."""
    model.train()
    d = model.dynamics
    meta = meta_list[0]
    cf = meta["cells_flat_dev"]

    model.zero_grad(set_to_none=True)
    rho_out, _, _, _, _, cf_out, diags = d(
        cf, return_drive_debug=True,
    )
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

    print("\n--- L2 drive debug (last refine step, interior cells) ---")
    if diags and "drive_debug" in diags:
        for k, v in diags["drive_debug"].items():
            print(f"  {k}: {v:.4f}")
    print("\n  learned (value):")
    for name in (
        "b_coll", "b_seed", "b_iso", "b_cross",
        "eta_p", "alpha", "l1_von_mises_kappa",
    ):
        p = getattr(d, name)
        val = float(p.detach()) if not isinstance(p, torch.Tensor) else float(p.item())
        print(f"    {name}={val:.4f}")
    print("\n  |grad| on raw params:")
    for raw, label in (
        ("_b_coll_raw", "b_coll"),
        ("_b_iso_raw", "b_iso"),
        ("_b_cross_raw", "b_cross"),
        ("_eta_p_raw", "eta_p"),
        ("_alpha_raw", "alpha"),
        ("_l1_von_mises_kappa_raw", "l1_von_mises_kappa"),
    ):
        t = getattr(d, raw, None)
        if t is None or t.grad is None:
            print(f"    {label}: grad=None")
        else:
            print(f"    {label}: |grad|={t.grad.abs().mean().item():.2e}")
    print(f"\n  loss={loss.item():.4f}  requires_grad={loss.requires_grad}\n")


def plot_training_curves(history, out_dir):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=VIZ.BG)
    fig.suptitle(
        "STRIATE unified dynamics  learned ridge render",
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


def format_l1_param_lines(model: StriateE2E, *, indent: str = "  ") -> list[str]:
    return [
        f"{indent}von Mises κ={model.l1_von_mises_kappa.item():.3f} (learned; "
        f"exp(κ·cos2Δθ) bin sharpness — not per-cell orientation κ)",
    ]


def format_l2_param_lines(d, *, indent: str = "  ") -> list[str]:
    """Multi-line L2 learned-parameter summary (infer / train logging)."""
    sub = indent + "  "
    return [
        f"{indent}geometry:  K={d.K}  R_fac={d.R_fac}  R_sup={d.R_sup}  T_refine={d.T_refine}",
        f"{indent}binning:   K-channel ρ state (per-orientation hypotheses)",
        f"{indent}drive:",
        f"{sub}b_seed={d.b_seed.item():.3f}  b_coll={d.b_coll.item():.3f}",
        f"{indent}inhibition:",
        f"{sub}b_iso={d.b_iso.item():.3f}  b_cross={d.b_cross.item():.3f}  "
        f"η_p={d.eta_p.item():.3f}  α={d.alpha.item():.3f}",
        f"{indent}seed:  η_z={d.eta_z:.4g} (fixed)  "
        f"ρ̃=ρ_raw−min_k ρ_raw; ρ_seed=ρ̃²/(ρ̃²+η_z²)  "
        f"cross^(k)=mean_{{k'≠k}} W_disk*ρ^(k'))",
    ]


def format_renderer_param_lines(r, *, indent: str = "  ") -> list[str]:
    _ = (r, indent)
    return [
        f"{indent}cell→pixel: bilinear resize (ρ and θ via double-angle); no render params",
    ]


def _format_dynamics_params(model):
    d = model.dynamics
    parts = [
        "\n--- L2 ---\n",
        *[ln + "\n" for ln in format_l2_param_lines(d, indent="")],
        "\n--- dynamics ---\n",
        "drive = b_seed·ρ_seed^(k) + b_coll·ρ_coll^(k)\n",
        "ρ̃ = NR(drive², b_iso·c_iso + b_cross·cross + η_p²); "
        "ρ ← (1−α)ρ + α·ρ̃\n",
        "cross^(k) = mean_{k'≠k} count-norm(W_disk * ρ^(k')); grouped conv2d pools each step\n",
    ]
    return "".join(parts)


def format_model_param_counts(model: StriateE2E) -> tuple[int, int, int, int]:
    """Returns (total, L2 dynamics, L1 von Mises κ, renderer)."""
    n_l2 = sum(p.numel() for p in model.dynamics.parameters())
    n_l1 = model._l1_von_mises_kappa_raw.numel()
    n_renderer = sum(p.numel() for p in model.renderer.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    return n_total, n_l2, n_l1, n_renderer


def format_model_param_summary(model: StriateE2E) -> str:
    n_tot, n_l2, n_l1, n_r = format_model_param_counts(model)
    return (
        f"{n_tot} total = L2 dynamics {n_l2} + L1 von Mises κ {n_l1} "
        f"(bin sharpness, not per-cell κ) + renderer {n_r}"
    )


def _format_render_params(model):
    r = model.renderer
    n_r = sum(p.numel() for p in r.parameters())
    return (
        f"renderer={n_r} params  "
        f"(bilinear ρ / θ upsample; no learned σ)"
    )


def save_checkpoint(model, path):
    torch.save({"model_state": model.state_dict()}, path)


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
        f"Dynamics: R_fac={L2.R_FAC_POOL}  R_sup={L2.R_SUP_POOL}  K={L2.K}  "
        f"T={L2.T_REFINE}  "
        f"ρ←(1−α)ρ+α·NR(drive²/(drive²+b_iso·c_iso+b_cross·cross+η_p²)); "
        f"ρ_seed=ρ_raw²/(ρ_raw²+η_z²), ρ_raw=ρ_bins/(ρ_total+ε); α init={L2.ALPHA_INIT}; "
        f"cross^(k)=mean_{{k'≠k}} W_disk*ρ^(k'); "
        f"  TBPTT: max {TRAIN.L2_SNAPSHOT_MAX} refine steps/segment; "
        f"detach every {TRAIN.L2_SNAPSHOT_MAX} steps only if T>{TRAIN.L2_SNAPSHOT_MAX}"
    )
    print(
        f"Render: bilinear cell grid → pixel ρ (and θ); no render MLP / no splat"
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

    model = StriateE2E(
        r_fac_pool=L2.R_FAC_POOL,
        r_sup_pool=L2.R_SUP_POOL,
        K=L2.K,
        t_refine=L2.T_REFINE,
        eps=L2.EPS,
        render_cell_hidden=RENDER.CELL_HIDDEN,
        render_pixel_hidden=RENDER.PIXEL_HIDDEN,
    ).to(device)

    active_params = list(model.parameters())
    optimizer = torch.optim.Adam(active_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    print(f"\nmodel: {format_model_param_summary(model)}  (Adam on all)")

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
            (gi, gt, Hp, Wp, Hg, Wg, H0, W0, l0_pix, border_mask, img) = item
            moved.append((gi, gt, Hp, Wp, Hg, Wg, H0, W0, l0_pix, border_mask, img))
        debug_drive_batch(
            model,
            prepare_batch(moved, device, model),
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
                (gi, gt, Hp, Wp, Hg, Wg, H0, W0, l0_pix, border_mask, img) = item
                moved.append((gi, gt, Hp, Wp, Hg, Wg, H0, W0, l0_pix, border_mask, img))

            meta_list = prepare_batch(moved, device, model)
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