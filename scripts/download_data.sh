#!/bin/bash
# ============================================================
# Download PhysioNet/CinC Challenge 2016 dataset
# ============================================================
# Requires: wget
# Output: data/physionet2016/training-{a..f}/
# ============================================================

set -e

DATA_DIR="data/physionet2016"
BASE_URL="https://physionet.org/files/challenge-2016/1.0.0"

mkdir -p "$DATA_DIR"

echo "============================================"
echo "  Downloading PhysioNet/CinC 2016 Dataset"
echo "============================================"

for subset in training-a training-b training-c training-d training-e training-f; do
    echo ""
    echo "Downloading ${subset}..."
    wget -r -np -nH --cut-dirs=3 -P "$DATA_DIR" \
         "${BASE_URL}/${subset}/" 2>&1 | tail -1
    echo "  ✓ ${subset} complete"
done

echo ""
echo "============================================"
echo "  Download complete!"
echo "  Data saved to: ${DATA_DIR}/"
echo "============================================"
echo ""
echo "  Verify with: ls ${DATA_DIR}/training-a/*.wav | head"
