#!/usr/bin/env bash
# Download D-FINE COCO checkpoints into $TCLD_CKPT (default ./checkpoints).
# Usage: scripts/download_dfine_ckpt.sh [s l m n x ...]   (default: s l)
set -euo pipefail

CKPT_DIR="${TCLD_CKPT:-./checkpoints}"
mkdir -p "$CKPT_DIR"
BASE="https://github.com/Peterande/storage/releases/download/dfinev1.0"
MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(s l)

for m in "${MODELS[@]}"; do
  f="dfine_${m}_coco.pth"
  if [ -s "$CKPT_DIR/$f" ]; then
    echo "exists: $CKPT_DIR/$f"
    continue
  fi
  echo "downloading $f -> $CKPT_DIR"
  wget -q --show-progress -O "$CKPT_DIR/$f.part" "$BASE/$f"
  mv "$CKPT_DIR/$f.part" "$CKPT_DIR/$f"
done
ls -la "$CKPT_DIR"
