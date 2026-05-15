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
  --hf-dataset "wikitext,wikitext-103-raw-v1,test" \
  --num-batches 100 \
  --num-grad-batches 10 \
  --seq-len 512 \
  --batch-size 16

echo ""
echo "Done. Results:"
ls -lh "$OUTPUT_DIR"
