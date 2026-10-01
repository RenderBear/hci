#!/bin/sh
# NYUD v2 edge benchmark: Gupta et al.'s ECCV 2014 release of the 1449 labelled NYU Depth v2 images,
# with per-image ground truth in BSDS .mat format. Evaluate with --tol 0.011 (the NYUD convention).
set -eu

URL="https://www.dropbox.com/s/upuvhbjh4nbji02/eccv14-data.tgz?dl=1"
HERE=$(cd "$(dirname "$0")" && pwd)

usage() {
  cat <<'EOF'
Usage: sh scripts/nyud.sh [--src-dir DIR] [--data-root DIR]

Downloads eccv14-data.tgz (~900 MB; extraction needs several GB of temporary space), or uses an
extracted copy, and writes:
  DATA_ROOT/train/imgs   795 train+val images (img_XXXX.png)
  DATA_ROOT/train/gt     ground truth, one BSDS-format groundTruth .mat per image
  DATA_ROOT/test/imgs    654 test images
  DATA_ROOT/test/gt

  --src-dir DIR    an extracted eccv14-data directory (contains data/ and benchmarkData/)
  --data-root DIR  output root (default: NYUDv2)

Source: https://github.com/s-gupta/rcnn-depth (Gupta, Girshick, Arbelaez, Malik, ECCV 2014).
EOF
}

SRC=""
ROOT="NYUDv2"
while [ $# -gt 0 ]; do
  case "$1" in
    --src-dir|--data-root)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
      if [ "$1" = "--src-dir" ]; then SRC=$2; else ROOT=$2; fi
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

TMP=""
cleanup() { if [ -n "$TMP" ]; then rm -rf "$TMP"; fi; }
trap cleanup EXIT

if [ -z "$SRC" ]; then
  command -v curl >/dev/null || { echo "curl is required to download NYUD" >&2; exit 1; }
  TMP=$(mktemp -d)
  echo "downloading eccv14-data.tgz (~900 MB) ..."
  curl -L --fail --progress-bar -o "$TMP/eccv14-data.tgz" "$URL"
  echo "extracting ..."
  tar -xzf "$TMP/eccv14-data.tgz" -C "$TMP"
  rm -f "$TMP/eccv14-data.tgz"
  BENCH=$(find "$TMP" -maxdepth 3 -type d -name benchmarkData | head -n 1)
  [ -n "$BENCH" ] || { echo "benchmarkData/ not found in the archive" >&2; exit 1; }
  SRC=$(dirname "$BENCH")
fi

uv run --project "$HERE/.." python "$HERE/nyud_prepare.py" --src "$SRC" --out "$ROOT"
echo "NYUD ready in $ROOT/{train,test} (test with --tol 0.011)"
