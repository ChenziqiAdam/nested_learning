#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/rq1_pilot.yaml}"
OUTPUT_DIR="${2:-results/rq1}"
DEVICE="${3:-cuda:0}"

echo "=========================================="
echo " RQ1 Analysis"
echo " Config:     $CONFIG"
echo " Output:     $OUTPUT_DIR"
echo " Device:     $DEVICE"
echo "=========================================="

uv run python scripts/rq1_analysis.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --num-batches 50 \
  --num-grad-batches 5 \
  --seq-len 256 \
  --batch-size 4

echo ""
echo "Done. Results:"
ls -lh "$OUTPUT_DIR"
