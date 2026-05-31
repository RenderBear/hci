# Harmonic Contour Integration

## Structure

```
STRIATE/
├── pyproject.toml
├── requirements.txt
├── params.py                # all hyperparameters
├── train.py                 # StriateE2E training
├── test.py                  # ODS, OIS, AP evaluation
├── infer.py                 # single-image inference + diagnostics
├── hci/
│   ├── L0.py                # pixel-level contrast
│   ├── L1.py                # cell-level z₂ moments (E, C, θ)
│   ├── seed.py              # |Z|²/(|Z|²+η_z²) → cell ρ for splat
│   ├── renderer.py          # learned ridge projection
│   └── diagnostics_viz.py   # visualisation utilities
├── data/                    # train, test, infer images (generic layout)
└── output/
    ├── checkpoints/         # best.pt, final.pt
    └── test/results.json
```

**Equation and notation reference:** `equations.md` (code-aligned STRIATE pipeline: L0 → L1 moments → seed → splat renderer) and `docs/docs.html` (open in a browser).

## Requirements

Python **3.12+**, PyTorch 2.1+.

```bash
pip install -r requirements.txt
```

### GPU acceleration 

All stages are PyTorch-native. The default install is CPU-only. For CUDA, check your driver version (`nvidia-smi`) and install the matching PyTorch wheel first:

```bash
# Example for CUDA 12.8 — adjust to your version
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Requirements (uv)

Python **3.12+**, PyTorch 2.1+. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

### GPU acceleration (optional)

Check your CUDA version: `nvidia-smi`

Then install the matching PyTorch wheel, e.g. for CUDA 12.8:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Usage

### Train

```bash
uv run train.py --train_imgs data/train/imgs --train_gt data/train/gt
```

Main flags: `--epochs`, `--lr` (default `5e-2`), `--batch_size`, `--max_val_ratio`, `--device`, `--output_dir`, `--checkpoints_dir`, `--cache_dir`, `--gt_format` (`png` / `mat`; auto-detected from GT dir if omitted).

### BIPED

`BIPED/` is in `.gitignore` — place the dataset at the repo root yourself. Training uses `train.py` with the train RGB and edge-map folders below. Layout expected by the default commands (RGB + PNG edge maps):

```
BIPED/edges/imgs/train/rgbr/real/           # training RGB (.jpg / .png)
BIPED/edges/edge_maps/train/rgbr/real/      # training GT edges (same stem per image)
BIPED/edges/imgs/test/rgbr/                 # test RGB
BIPED/edges/edge_maps/test/rgbr/            # test GT edges
```

If you have the Kaggle CLI set up (or you already downloaded/unzipped the dataset), you can generate this layout with:

```bash
sh scripts/biped.sh --version bipedv2 --kaggle   # recommended
# or: sh scripts/biped.sh --version biped --src-dir /path/to/extracted/BIPED
# optional: --data-root BIPEDv2
```

**Train** on the BIPED train split (writes checkpoints under `output/checkpoints` unless overridden):

```bash
uv run train.py \
  --train_imgs BIPED/edges/imgs/train/rgbr/real \
  --train_gt BIPED/edges/edge_maps/train/rgbr/real \
  --cache_dir cache/biped_train \
```

Use a dedicated `--cache_dir` so BIPED caches do not mix with other experiments. Lower `--batch_size` if you hit GPU memory limits.

**Test** on the BIPED test split (ODS / OIS / AP; pairs images to GT by filename stem):

```bash
uv run test.py \
  --images BIPED/edges/imgs/test/rgbr \
  --test_gt BIPED/edges/edge_maps/test/rgbr \
  --output_dir output/test_biped
```

Quick smoke test: add `--max_images 10`. If your GT folder uses BSDS-style `.mat` files instead of PNG, pass `--gt_format mat`.

### BRIND (edge maps)

BRIND is a BSDS-based edge benchmark annotated for different discontinuity types (reflectance, illuminance, normal, depth) plus a combined `all` edge map.

To remap BRIND into the STRIATE directory layout used by `train.py`/`test.py`:

```bash
sh scripts/brind.sh --src-dir /path/to/extracted/BRIND --data-root BRIND --gt-type all
```

**Train** on BRIND (combined `all` edges by default):

```bash
uv run train.py \
  --train_imgs BRIND/edges/imgs/train/rgbr/real \
  --train_gt BRIND/edges/edge_maps/train/rgbr/real \
  --cache_dir cache/brind_train
```

**Test** on BRIND:

```bash
uv run test.py \
  --images BRIND/edges/imgs/test/rgbr \
  --test_gt BRIND/edges/edge_maps/test/rgbr \
  --output_dir output/test_brind
```

If the ground truth maps are stored as `.mat` files in your BRIND extraction, pass `--gt_format mat` to `test.py` and `--gt_format mat` to `train.py`.

### BSDS500

Clone the mirror at the repo root (e.g. next to this project): [BIDS/BSDS500](https://github.com/BIDS/BSDS500).

```bash
git clone https://github.com/BIDS/BSDS500.git
```

Paths below assume the usual layout inside the clone: `BSDS500/BSDS500/data/images/{train,test}` and `BSDS500/BSDS500/data/groundTruth/{train,test}`.

**Train** on the BSDS500 train split (MAT ground truth):

```bash
uv run train.py \
  --train_imgs BSDS500/BSDS500/data/images/train \
  --train_gt BSDS500/BSDS500/data/groundTruth/train \
  --gt_format mat \
  --cache_dir cache/bsds_train \
```

Use a dedicated `--cache_dir` so BSDS caches do not mix with BIPED or other runs. Lower `--batch_size` if you run out of memory.

**Test** on the BSDS500 test split (MAT ground truth):

```bash
uv run test.py \
  --images BSDS500/BSDS500/data/images/test \
  --test_gt BSDS500/BSDS500/data/groundTruth/test \
  --gt_format mat \
  --output_dir output/test_bsds500
```

### NYUD v2 (edge maps)

NYUD v2 is primarily an RGB-D dataset, but it is often used for edge detection through derived edge annotations. STRIATE matches RGB images and GT by filename stem (for example `img_5001.png` in both folders).

Default layout after running `nyud.sh`:

```
NYUDv2/
  images/                            # RGB images (.jpg / .png)
  GT/                                # edge GT maps (.png)
```

**Train** on the downloaded NYUDv2 layout (writes checkpoints under `output/checkpoints` unless overridden):

```bash
uv run train.py \
  --train_imgs NYUDv2/images \
  --train_gt NYUDv2/GT \
  --cache_dir cache/nyudv2_train
```

Use a dedicated `--cache_dir` so NYUD caches do not mix with other experiments. Lower `--batch_size` if you hit GPU memory limits.

**Test** on the downloaded NYUDv2 layout:

```bash
uv run test.py \
  --images NYUDv2/images \
  --test_gt NYUDv2/GT \
  --output_dir output/test_nyudv2
```

Quick smoke test: add `--max_images 20`.

If your NYUD edge labels are stored as MAT files, add `--gt_format mat`.

Optional split-based layout (if you create your own train/test split):

```
NYUDv2/
  images/train/
  images/test/
  edges/train/
  edges/test/
```

Use the corresponding split paths with `train.py` and `test.py`.

### Test

Walks a single image directory, pairs each image with ground truth by matching filename stems:

```bash
uv run test.py --model output/checkpoints/best.pt
```

| Flag | Default | Role |
|------|---------|------|
| `--images` | `data/test/imgs` | RGB test images (`.jpg`/`.png`) |
| `--test_gt` | `data/test/gt` | Ground truth maps (`.png`/`.jpg`/`.mat`) |
| `--gt_format` | auto | `png` or `mat` (BSDS) |
| `--model` | `output/checkpoints/best.pt` | Checkpoint |
| `--output_dir` | `output/test` | Output directory |
| `--max_images` | all | Cap number of images |
| `--device` | CUDA if available | `cpu`, `cuda`, or `mps` |
| `--tol` | `0.0075` | Precision-match radius factor (`max_dist = tol * image_diagonal`) |

Checkpoints produced **before** the regional η MLP (`eta_mlp.*` in the state dict) do not contain those weights. `infer.py` / `test.py` load with `strict=False`, so missing keys keep their **PyTorch init** values until you **retrain** with the current `train.py`; a short **WARNING** is printed when `eta_mlp.*` keys are absent from the file. Older checkpoints may still list `eta_mod_a` / `eta_mod_b` / `eta_mod_c`; those are ignored on load.

### Infer

```bash
uv run infer.py --image photo.png --input_dir data/infer \
  --model output/checkpoints/best.pt
```

Writes diagnostic images (pinwheel, ρ maps, CDF, texture gate, edges) under `--output_dir`.
