#!/bin/sh
# BRIND: BSDS500 images with RINDNet's edge annotations, merged into one map per image.
# Source: https://github.com/xavysp/BRIND (300 train / 200 test).
set -eu

usage() {
  cat <<'EOF'
Usage: sh scripts/brind.sh [--src-dir DIR] [--data-root DIR]

Clones https://github.com/xavysp/BRIND (or uses an existing clone) and writes:
  DATA_ROOT/train/imgs   300 train images (.jpg)
  DATA_ROOT/train/gt     300 train edge maps (.png)
  DATA_ROOT/test/imgs    200 test images (.jpg)
  DATA_ROOT/test/gt      200 test edge maps (.png)

  --src-dir DIR    existing BRIND clone (has train_imgs/, train_gt/, test_imgs/, test_gt/)
  --data-root DIR  output root (default: data, where train.py and test.py look by default)

Only the combined edge map is published there; the per-type maps (reflectance,
illumination, normal, depth) are at https://github.com/MengyangPu/RINDNet.
EOF
}

SRC=""
ROOT="data"
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
  command -v git >/dev/null || { echo "git is required to fetch BRIND" >&2; exit 1; }
  TMP=$(mktemp -d)
  echo "cloning https://github.com/xavysp/BRIND ..."
  git clone --depth 1 --quiet https://github.com/xavysp/BRIND.git "$TMP/BRIND"
  SRC="$TMP/BRIND"
fi

copy_set() {  # copy_set SRC_DIR DST_DIR EXT EXPECTED_COUNT
  [ -d "$1" ] || { echo "missing $1 (is --src-dir a BRIND clone?)" >&2; exit 1; }
  mkdir -p "$2"
  n=0
  for f in "$1"/*."$3"; do
    [ -e "$f" ] || continue
    cp "$f" "$2/"
    n=$((n + 1))
  done
  echo "  $2: $n files"
  [ "$n" -eq "$4" ] || echo "  warning: expected $4 files in $1" >&2
}

copy_set "$SRC/train_imgs/imgs_all" "$ROOT/train/imgs" jpg 300
copy_set "$SRC/train_gt/gt_all" "$ROOT/train/gt" png 300
copy_set "$SRC/test_imgs" "$ROOT/test/imgs" jpg 200
copy_set "$SRC/test_gt" "$ROOT/test/gt" png 200
echo "BRIND ready in $ROOT/{train,test}"
