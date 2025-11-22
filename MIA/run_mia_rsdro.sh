#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run confidence/loss MIA against an RS-DRO checkpoint.
# Usage: bash MIA/run_mia_rsdro.sh /path/to/rsdro_resnet20.pt [width_factor]

CHECKPOINT=${1:-runs/rsdro/rsdro_resnet20.pt}
WIDTH_FACTOR=${2:-0.5}

python MIA/run_attacks.py \
  --checkpoint "${CHECKPOINT}" \
  --width-factor "${WIDTH_FACTOR}" \
  --data-root data \
  --split-root MIA/splits \
  --output-dir MIA/outputs/rsdro \
  --member-count 10000 \
  --calibration-count 5000 \
  --per-class-thresholds
