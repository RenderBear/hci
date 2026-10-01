#!/bin/sh
# BIPED: 250 outdoor 1280x720 images with expert edge annotations (200 train / 50 test).
# Source: https://www.kaggle.com/datasets/xavysp/biped
set -eu

usage() {
  cat <<'EOF'
Usage: sh scripts/biped.sh (--kaggle | --src-dir DIR) [--data-root DIR]

Writes:
  DATA_ROOT/train/imgs   200 train images (.jpg)
  DATA_ROOT/train/gt     200 train edge maps (.png)
  DATA_ROOT/test/imgs    50 test images (.jpg)
  DATA_ROOT/test/gt      50 test edge maps (.png)

  --kaggle         download xavysp/biped with the Kaggle CLI (run through uvx). Needs a
                   Kaggle API token: ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY.
  --src-dir DIR    an already downloaded and extracted BIPED (any folder containing
                   edges/imgs/train/rgbr/real); if it holds BIPED and BIPEDv2, v2 is used
  --data-root DIR  output root (default: BIPED)
EOF
}

SRC=""
KAGGLE=0
ROOT="BIPED"
while [ $# -gt 0 ]; do
  case "$1" in
    --kaggle) KAGGLE=1; shift ;;
    --src-dir|--data-root)
      [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
      if [ "$1" = "--src-dir" ]; then SRC=$2; else ROOT=$2; fi
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$KAGGLE:${SRC:+set}" in
  1:|0:set) ;;
  *) echo "pass exactly one of --kaggle or --src-dir" >&2; usage >&2; exit 2 ;;
esac

TMP=""
cleanup() { if [ -n "$TMP" ]; then rm -rf "$TMP"; fi; }
trap cleanup EXIT

if [ "$KAGGLE" -eq 1 ]; then
  command -v uvx >/dev/null || { echo "uvx (from uv) is required for --kaggle" >&2; exit 1; }
  TMP=$(mktemp -d)
  echo "downloading xavysp/biped from Kaggle ..."
  uvx --from kaggle kaggle datasets download xavysp/biped -p "$TMP" --unzip
  SRC=$TMP
fi

[ -d "$SRC" ] || { echo "not a directory: $SRC" >&2; exit 1; }
PICK=""
while IFS= read -r d; do
  [ -n "$d" ] || continue
  case "$d" in *BIPEDv2*) PICK=$d; break ;; esac
  [ -n "$PICK" ] || PICK=$d
done <<EOF
$(find "$SRC" -type d -path '*/edges/imgs/train/rgbr/real')
EOF
[ -n "$PICK" ] || { echo "no edges/imgs/train/rgbr/real under $SRC" >&2; exit 1; }
EDGES=${PICK%/imgs/train/rgbr/real}
echo "using $EDGES"

copy_set() {  # copy_set SRC_DIR DST_DIR EXT EXPECTED_COUNT
  [ -d "$1" ] || { echo "missing $1" >&2; exit 1; }
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

copy_set "$EDGES/imgs/train/rgbr/real" "$ROOT/train/imgs" jpg 200
copy_set "$EDGES/edge_maps/train/rgbr/real" "$ROOT/train/gt" png 200
copy_set "$EDGES/imgs/test/rgbr" "$ROOT/test/imgs" jpg 50
copy_set "$EDGES/edge_maps/test/rgbr" "$ROOT/test/gt" png 50
echo "BIPED ready in $ROOT/{train,test}"
