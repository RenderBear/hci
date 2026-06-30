r"""train.py — HCI training pipeline: z₂ moments + association-field seed + harmonic render"""

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
    L0LearnedMetric,
    L0Notch,
)
from hci.L1 import (
    stride_from_patch_overlap,
    z_from_l0_harmonics,
    pad_for_patch_grid,
    compute_cell_moments,
)
from hci.seed import AndGateSeed
from hci.renderer import (
    ModulationRenderer,
    render_boundary_map_torch,
    proj_to_device,
    upgrade_renderer_state_dict,
)
from params import L0, L1, SEED, TRAIN, VIZ


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


def build_l0_pix_live(
    img: torch.Tensor,
    border_mask: torch.Tensor,
    metric: L0LearnedMetric,
    device: torch.device,
    notch: L0Notch | None = None,
) -> dict[str, torch.Tensor]:
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(np.asarray(img), dtype=torch.float32)
    img_dev = img.to(device=device, dtype=torch.float32)
    bm = border_mask.to(device=device).bool()
    _, _, _, _, _, s, h1m, h2m, h2m_lum, h2m_chr = compute_l0_rgb(
        img_dev,
        eta_lum=L0.ETA_LUM,
        eta_chr=L0.ETA_CHR,
        gamma=L0.GAMMA,
        offsets=L0.OFFSETS,
        metric=metric,
        notch=notch,
    )
    z1_re, z1_im = s[..., 0], s[..., 1]
    z2_re, z2_im = s[..., 2], s[..., 3]
    inv = (~bm).float()
    return {
        "h1m": h1m * inv,
        "h2m": h2m * inv,
        "h2m_lum": h2m_lum * inv,
        "h2m_chr": h2m_chr * inv,
        "z1_re": z1_re * inv,
        "z1_im": z1_im * inv,
        "z2_re": z2_re * inv,
        "z2_im": z2_im * inv,
    }


def proj_info_from_grid(H: int, W: int, P: int, patch_overlap: int) -> dict:
    S = stride_from_patch_overlap(P, patch_overlap)
    nH = (H - P) // S + 1 if H >= P else 0
    nW = (W - P) // S + 1 if W >= P else 0
    return {"H": H, "W": W, "n_cells": nH * nW, "nH": nH, "nW": nW}


def build_cells_flat_torch(cells: dict) -> dict:
    nH, nW = int(cells["nH"]), int(cells["nW"])
    N = nH * nW
    rho_t = cells["rho_total"].reshape(N)
    K = int(cells.get("K", getattr(L1, "NUM_ORIENT_BINS", 8)))
    out = {
        "nH": nH,
        "nW": nW,
        "S": int(cells["S"]),
        "P": int(cells["P"]),
        "K": K,
        "theta": cells["theta"].reshape(N, 1),
        "rho_peak": cells["rho_peak"].reshape(N),
        "rho_total": rho_t,
        "z0": rho_t,
        "cx_z2": cells["cx_z2"].reshape(N),
        "cy_z2": cells["cy_z2"].reshape(N),
        "is_border": cells["is_border"].reshape(N).bool(),
    }
    if "rho_bin" in cells:
        out["rho_bin"] = cells["rho_bin"].reshape(N, K)
        out["ax_bin"] = cells["ax_bin"].reshape(N, K)
        out["ay_bin"] = cells["ay_bin"].reshape(N, K)
        out["rho_bin_coh"] = cells["rho_bin_coh"].reshape(N, K)
        out["theta_bins"] = cells["theta_bins"].reshape(K)
    return out


def run_moments_cells_flat(
    l0_pix: dict[str, torch.Tensor],
    border_mask: torch.Tensor,
    H0: int,
    W0: int,
    device: torch.device,
    *,
    kappa_vm: torch.Tensor | float | None = None,
) -> dict:
    h2m = l0_pix["h2m"].to(device=device, dtype=torch.float32)
    z2 = torch.complex(
        l0_pix["z2_re"].to(device=device, dtype=torch.float32),
        l0_pix["z2_im"].to(device=device, dtype=torch.float32),
    )
    bm = border_mask.to(device=device).bool()
    if kappa_vm is None:
        kappa_vm = float(getattr(L1, "KAPPA_VM_INIT", 2.0))
    cells = compute_cell_moments(
        h2m,
        z2,
        L1.PATCH_SIZE,
        bm,
        patch_overlap=L1.PATCH_OVERLAP,
        border_patch_max_frac=L1.BORDER_PATCH_MAX_FRAC,
        eps=L1.EPS,
        device=device,
        verbose=False,
        return_torch=True,
        kappa_vm=kappa_vm,
        num_orient_bins=int(getattr(L1, "NUM_ORIENT_BINS", 8)),
    )
    P = int(cells["P"])
    cells["is_border"] = cells["is_border"] | (
        (cells["cy"] + P / 2 > H0) | (cells["cx"] + P / 2 > W0)
    )
    return build_cells_flat_torch(cells)


run_l1_cells_flat = run_moments_cells_flat


def build_cells_flat(cells: dict) -> dict:
    nH, nW = cells["nH"], cells["nW"]
    N = nH * nW
    rho_t = cells["rho_total"].reshape(N)
    K = int(cells.get("K", getattr(L1, "NUM_ORIENT_BINS", 8)))
    out = {
        "nH": nH,
        "nW": nW,
        "S": int(cells["S"]),
        "P": int(cells["P"]),
        "K": K,
        "theta": torch.from_numpy(cells["theta"].reshape(N, 1).astype(np.float32)),
        "rho_peak": torch.from_numpy(
            cells["rho_peak"].reshape(N).astype(np.float32)
        ),
        "rho_total": torch.from_numpy(rho_t.astype(np.float32)),
        "z0": torch.from_numpy(rho_t.astype(np.float32)),
        "cx_z2": torch.from_numpy(cells["cx_z2"].reshape(N).astype(np.float32)),
        "cy_z2": torch.from_numpy(cells["cy_z2"].reshape(N).astype(np.float32)),
        "is_border": torch.from_numpy(cells["is_border"].reshape(N).astype(np.bool_)),
    }
    if "rho_bin" in cells:
        out["rho_bin"] = torch.from_numpy(
            np.asarray(cells["rho_bin"], dtype=np.float32).reshape(N, K)
        )
        out["ax_bin"] = torch.from_numpy(
            np.asarray(cells["ax_bin"], dtype=np.float32).reshape(N, K)
        )
        out["ay_bin"] = torch.from_numpy(
            np.asarray(cells["ay_bin"], dtype=np.float32).reshape(N, K)
        )
        out["rho_bin_coh"] = torch.from_numpy(
            np.asarray(cells["rho_bin_coh"], dtype=np.float32).reshape(N, K)
        )
        out["theta_bins"] = torch.from_numpy(
            np.asarray(cells["theta_bins"], dtype=np.float32).reshape(K)
        )
    return out


class HCIE2E(nn.Module):
    def __init__(
        self,
        eps: float = SEED.EPS,
        **kw,
    ):
        super().__init__()
        _ = kw
        self.l0_metric = L0LearnedMetric()
        self.l0_notch = L0Notch() if bool(getattr(L0, "NOTCH_ENABLED", False)) else None
        self.seed = AndGateSeed(eps=eps)
        self.renderer = ModulationRenderer()
        self.eps = eps
        self.render_eps = max(float(eps), 1e-6)

    def forward_batch(self, meta_list):
        bmaps = []
        for m in meta_list:
            cf_flat = m["cells_flat_dev"]
            rho_out, branch, _, _, _, cf_out, _ = self.seed(cells_flat=cf_flat)
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


def prepare_batch(items, device, model: HCIE2E):
    meta = []
    for item in items:
        (gi, gt, Hp, Wp, Hg, Wg, H0, W0, border_mask, img) = item
        if img is None:
            raise RuntimeError(
                "learned L0 metric requires cached RGB ``img``; "
                "rebuild cache (L0_CACHE_VERSION bump)."
            )
        l0_dev = build_l0_pix_live(
            img, border_mask, model.l0_metric, device, notch=model.l0_notch,
        )
        cf_dev = run_moments_cells_flat(
            l0_dev,
            border_mask,
            H0,
            W0,
            device,
            kappa_vm=model.seed.kappa_vm,
        )
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

    border_mask_np = ~compute_interior(
        ir_p.shape[0], ir_p.shape[1], torch.device("cpu"),
    ).numpy()

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
    img = data.get("img")
    if img is None:
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


class HCIDataset(Dataset):
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
                f"need border_mask + img + proj_info). "
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
            data["border_mask"],
            data.get("img"),
        )


def collate_fn(batch):
    return batch


def mask_gt_by_agreement(
    target: torch.Tensor,
    min_agreement: float,
) -> torch.Tensor:
    if min_agreement <= 0.0:
        return target
    y = target.clamp(0.0, 1.0)
    return torch.where(y >= min_agreement, y, torch.zeros_like(y))


def bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = pred.nan_to_num(0.0).clamp(eps, 1.0 - eps)
    y = target.clamp(0.0, 1.0)
    n = y.numel()
    n_pos = (y >= pos_threshold).sum().to(y.dtype)
    n_neg = n - n_pos
    if n_pos <= 0 or n_neg <= 0:
        return F.binary_cross_entropy(pred, y, reduction="mean")
    beta = n_neg / n
    one_m_beta = n_pos / n
    loss = -(beta * y * pred.log() + one_m_beta * (1.0 - y) * (1.0 - pred).log())
    return loss.mean()


def upgrade_model_state_dict(state_dict: dict) -> dict:
    drop = {
        "seed._eta_seed_raw",
        "seed._sigma_l1_raw",
        "seed._R0",
        "seed._a_raw",
        "seed._b_raw",
    }
    rename = {
        "seed._beta_raw": "seed._beta_seed_raw",
        "seed._kappa_raw": "seed._beta_coll_raw",
        "seed._eta_raw": "seed._eta_readout_raw",
        "_l1_von_mises_kappa_raw": "seed._kappa_vm_raw",
    }
    out: dict = {}
    for k, v in state_dict.items():
        if k.startswith("dynamics."):
            continue
        if k in drop:
            continue
        out[rename.get(k, k)] = v
    return out


def debug_seed_batch(
    model,
    meta_list,
    device,
    *,
    gt_min_agreement: float = 0.0,
):
    model.train()
    meta = meta_list[0]
    cf = meta["cells_flat_dev"]

    model.zero_grad(set_to_none=True)
    rho_out, _, _, _, _, cf_out, diags = model.seed(cf, return_surface_diags=True)
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
    gt = mask_gt_by_agreement(meta["gt"][:Hg, :Wg], gt_min_agreement)
    loss = bce_loss(bmap[:Hg, :Wg], gt)
    loss.backward()

    print("\n--- seed debug ---")
    if diags and "iter_stats" in diags and diags["iter_stats"]:
        st = diags["iter_stats"][0]
        for key in ("rho_mean", "rho_max", "mid_band_frac", "n_interior",
                    "fac_mean", "sur_mean"):
            if key in st:
                print(f"  {key}: {st[key]}")
    seed = model.seed
    print(
        f"\n  κ_vm={seed.kappa_vm.item():.4g}  η_z={seed.eta_z.item():.4g}  "
        f"β_seed={seed.beta_seed.item():.4g}  β_coll={seed.beta_coll.item():.4g}  "
        f"κ_θ={seed.kappa_theta.item():.4g}  η_readout={seed.eta_readout.item():.4g}  "
        f"λ={seed.lam.item():.4g}  σ_f={seed.sigma_f.item():.4g}  σ_S={seed.sigma_s.item():.4g}"
    )
    for name, t in (
        ("κ_vm", seed._kappa_vm_raw),
        ("η_z", seed._eta_z_raw),
        ("β_seed", seed._beta_seed_raw),
        ("β_coll", seed._beta_coll_raw),
        ("κ_θ", seed._kappa_theta_raw),
        ("η_readout", seed._eta_readout_raw),
        ("λ", seed._lambda_raw),
        ("σ_f", seed._sigma_f_raw),
        ("σ_S", seed._sigma_s_raw),
    ):
        if t.grad is None:
            print(f"  |grad| {name}: grad=None")
        else:
            print(f"  |grad| {name}: {t.grad.abs().mean().item():.2e}")
    print(f"\n  loss={loss.item():.4f}  requires_grad={loss.requires_grad}\n")


def plot_training_curves(history, out_dir):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=VIZ.BG)
    fig.suptitle(
        "HCI seed + learned ridge render",
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
    axes[0].plot(epochs, losses, color="#66aaff", lw=1.3, label="loss")
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


def format_l0_param_lines(model: HCIE2E, *, indent: str = "  ") -> list[str]:
    m = model.l0_metric
    w = m.W.detach().cpu()

    def _row(row: torch.Tensor) -> str:
        return "[" + ", ".join(f"{float(x):.4f}" for x in row.tolist()) + "]"

    lines = [
        f"{indent}W:",
        f"{indent}  [0] {_row(w[0])}",
        f"{indent}  [1] {_row(w[1])}",
        f"{indent}  [2] {_row(w[2])}",
    ]
    n = model.l0_notch
    if n is not None:
        lines.extend([
            "",
            f"{indent}notch: w_n={n.omega_n.item():.3g}  "
            f"σ_n={n.sigma_n.item():.3g}  d={n.d.item():.3g}  L={2 * n.H + 1}",
        ])
    return lines


def format_l1_param_lines(model: HCIE2E, *, indent: str = "  ") -> list[str]:
    K = int(getattr(L1, "NUM_ORIENT_BINS", 8))
    kvm = float(model.seed.kappa_vm.detach().cpu())
    return [f"{indent}K={K}  κ_vm={kvm:.4g}"]


def format_seed_param_lines(seed, *, indent: str = "  ") -> list[str]:
    return [
        f"{indent}η_z={seed.eta_z.item():.4g}  β_seed={seed.beta_seed.item():.4g}  "
        f"β_coll={seed.beta_coll.item():.4g}  κ_θ={seed.kappa_theta.item():.4g}",
        f"{indent}η_readout={seed.eta_readout.item():.4g}  λ={seed.lam.item():.4g}  "
        f"σ_f={seed.sigma_f.item():.4g}  σ_S={seed.sigma_s.item():.4g}  R={seed.cross_surround_radius}",
    ]

def format_renderer_param_lines(r, *, indent: str = "  ") -> list[str]:
    from hci.renderer import ModulationRenderer

    if not isinstance(r, ModulationRenderer):
        return [f"{indent}{type(r).__name__}"]

    hw = r.kernel_h_w
    h_perp = r.h_perp.detach().cpu()
    h_par = r.h_par.detach().cpu()
    center = int(hw)
    return [
        f"{indent}H_w={hw}  T_gate={r.gate_temp.item():.4g}  "
        f"κ_max={r.kappa_max.item():.4g}  e_max={r.ext_max.item():.4g}  "
        f"δ_n={r.delta_n_max.item():.4g}  α_r={r.alpha_range.item():.4g}",
        f"{indent}h_perp: min={float(h_perp.min()):+.3f}  "
        f"ctr={float(h_perp[center]):+.3f}  max={float(h_perp.max()):+.3f}",
        f"{indent}h_par peak={float(h_par[center]):.4g}",
    ]


def format_model_param_counts(model: HCIE2E) -> tuple[int, int, int, int]:
    n_l0 = sum(p.numel() for p in model.l0_metric.parameters())
    if model.l0_notch is not None:
        n_l0 += sum(p.numel() for p in model.l0_notch.parameters())
    n_seed = sum(p.numel() for p in model.seed.parameters())
    n_renderer = sum(p.numel() for p in model.renderer.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    return n_total, n_l0, n_seed, n_renderer


def format_model_param_summary(model: HCIE2E) -> str:
    n_tot, n_l0, n_seed, n_r = format_model_param_counts(model)
    return f"{n_tot} (L0={n_l0} seed={n_seed} renderer={n_r})"


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
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=TRAIN.EPOCHS)
    ap.add_argument("--lr", type=float, default=TRAIN.LR)
    ap.add_argument("--batch_size", type=int, default=TRAIN.BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=TRAIN.NUM_WORKERS)
    ap.add_argument("--grad_clip", type=float, default=TRAIN.GRAD_CLIP)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--gt_min_agreement",
        type=float,
        default=TRAIN.GT_MIN_AGREEMENT,
        help="zero soft-GT pixels below this level before loss (0 disables)",
    )
    ap.add_argument(
        "--debug-seed",
        action="store_true",
        help="Run one batch, print seed stats and β_seed/β_coll/κ_θ/η_z/η_readout/λ/σ_f gradients, then exit",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    gt_format = args.gt_format or _detect_gt_format(args.train_gt)

    mt = args.max_images if args.max_images is not None else "all"
    print(
        f"device={device}  gt_format={gt_format}  batch={args.batch_size}"
        f"  max_images={mt}"
    )
    gt_agree_note = (
        f"  gt_min_agreement={args.gt_min_agreement:g}"
        if args.gt_min_agreement > 0.0
        else ""
    )
    print(
        f"Loss: balanced BCE (soft GT){gt_agree_note}"
    )

    train_cache = os.path.join(args.cache_dir, "train")
    print(f"\nprecomputing train...")
    fit_stems = precompute_split(
        args.train_imgs,
        args.train_gt,
        train_cache,
        gt_format,
        max_images=args.max_images,
    )
    print(f"  {len(fit_stems)} cached")
    if not fit_stems:
        print(f"\nerror: no training images found.")
        return
    print(f"  training on all {len(fit_stems)}")

    model = HCIE2E(
        eps=SEED.EPS,
    ).to(device)

    active_params = list(model.parameters())
    optimizer = torch.optim.Adam(active_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    print(f"\nmodel: {format_model_param_summary(model)}")

    train_ds = HCIDataset(train_cache, fit_stems)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if args.debug_seed:
        batch = next(iter(train_loader))
        moved = []
        for item in batch:
            (gi, gt, Hp, Wp, Hg, Wg, H0, W0, border_mask, img) = item
            moved.append((gi, gt, Hp, Wp, Hg, Wg, H0, W0, border_mask, img))
        debug_seed_batch(
            model,
            prepare_batch(moved, device, model),
            device,
            gt_min_agreement=args.gt_min_agreement,
        )
        return

    print(f"\ntraining ({args.epochs} epochs)...\n")
    history = []
    n_fit = len(fit_stems)
    dbg_img_step = max(1, n_fit // 5)

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        n_img = 0
        n_batch = 0
        t0 = time.time()
        next_dbg_img = dbg_img_step

        for batch in train_loader:
            moved = []
            for item in batch:
                (gi, gt, Hp, Wp, Hg, Wg, H0, W0, border_mask, img) = item
                moved.append((gi, gt, Hp, Wp, Hg, Wg, H0, W0, border_mask, img))

            meta_list = prepare_batch(moved, device, model)
            n_bm = len(meta_list)

            optimizer.zero_grad()
            batch_loss_val = 0.0
            skip_batch = False

            for m in meta_list:
                cf_flat = m["cells_flat_dev"]
                rho_out, branch, _, _, _, cf_out, _ = model.seed(cells_flat=cf_flat)
                bmap_i = render_boundary_map_torch(
                    rho_out,
                    m["proj_dev"],
                    model.renderer,
                    cf_out,
                    m["Hp"], m["Wp"], m["l0_pix"],
                    eps=model.render_eps,
                    training=model.training,
                    branch_pick=branch.reshape(-1).long(),
                    content_h=m["H0"],
                    content_w=m["W0"],
                )
                Hg_i, Wg_i = m["Hg"], m["Wg"]
                bc = bmap_i[:Hg_i, :Wg_i]
                gc_ = mask_gt_by_agreement(
                    m["gt"][:Hg_i, :Wg_i], args.gt_min_agreement,
                )
                loss_i = bce_loss(bc, gc_) / n_bm

                if not loss_i.requires_grad:
                    raise RuntimeError(
                        "loss_i has no grad_fn — check GT, ρ/renderer graph"
                    )
                if not torch.isfinite(loss_i):
                    print(
                        f"    [warn] non-finite per-image loss="
                        f"{loss_i.item()} — skipping batch"
                    )
                    skip_batch = True
                    break

                loss_i.backward()

                batch_loss_val += loss_i.item()

                del bmap_i, bc, gc_, loss_i
                del rho_out, branch, cf_out, cf_flat

            if skip_batch:
                optimizer.zero_grad()
                del meta_list, moved
                continue

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(active_params, args.grad_clip)

            bad_grad = [
                n for n, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()
            ]
            if bad_grad:
                print(f"    [warn] non-finite grad in {bad_grad[:3]} — skipping step")
                optimizer.zero_grad()
                del meta_list, moved
                continue

            optimizer.step()

            loss_item = batch_loss_val

            n_img += n_bm
            ep_loss += loss_item
            n_batch += 1

            while n_img >= next_dbg_img:
                dt_p = time.time() - t0
                print(
                    f"    [dbg] epoch {epoch + 1}/{args.epochs}  ~img {next_dbg_img}/{n_fit}"
                    f"  batch {n_batch}  loss(run)={ep_loss / n_batch:.4f}"
                    f"  batch_loss={loss_item:.4f}"
                    f"  {dt_p:.1f}s elapsed",
                    flush=True,
                )
                next_dbg_img += dbg_img_step
            del meta_list, moved

        scheduler.step()
        al = ep_loss / max(n_batch, 1)
        lr_now = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        print(
            f"  epoch {epoch + 1:3d}/{args.epochs}:  "
            f"loss={al:.4f}  "
            f"lr={lr_now:.2e}  {dt:.1f}s"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": al,
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