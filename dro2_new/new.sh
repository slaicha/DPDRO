#!/usr/bin/env bash
# Convenience launcher for RS-DRO on CIFAR10-ST.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/../data}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/../runs/rsdro}"

DEVICE="${DEVICE:-$(python3 - <<'PY'
import torch
print('cuda' if torch.cuda.is_available() else 'cpu')
PY
)}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RHO="${RHO:-0.1}"
EPSILON="${EPSILON:-4.0}"
G_CONST="${G_CONST:-1.0}"
L_CONST="${L_CONST:-1.0}"
C_CONST="${C_CONST:-10.0}"
LAMBDA0="${LAMBDA0:-0.001}"
ETA_T_SQUARED="${ETA_T_SQUARED:-1e-4}"
PSI_WARMUP_STEPS="${PSI_WARMUP_STEPS:-200}"
PSI_WARMUP_LR="${PSI_WARMUP_LR:-5e-4}"
SEED="${SEED:-7}"
WIDTH_FACTOR="${WIDTH_FACTOR:-0.5}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-12}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-512}"
PRETRAIN_LR="${PRETRAIN_LR:-0.2}"
PRETRAIN_MOMENTUM="${PRETRAIN_MOMENTUM:-0.9}"
PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-5e-4}"
PRETRAIN_BALANCED="${PRETRAIN_BALANCED:-1}"
MAX_T="${MAX_T:-5}"
MAX_B1="${MAX_B1:-4096}"
MAX_B2="${MAX_B2:-64}"
MAX_Q="${MAX_Q:-5}"

EXTRA_ARGS=()
if [[ "$PRETRAIN_BALANCED" != "0" ]]; then
  EXTRA_ARGS+=("--pretrain-balanced")
else
  EXTRA_ARGS+=("--no-pretrain-balanced")
fi

python3 train_rsdro.py \
  --data-root "$DATA_ROOT" \
  --num-workers "$NUM_WORKERS" \
  --rho "$RHO" \
  --epsilon "$EPSILON" \
  --G "$G_CONST" \
  --L "$L_CONST" \
  --c "$C_CONST" \
  --lambda0 "$LAMBDA0" \
  --eta-t-squared "$ETA_T_SQUARED" \
  --psi-warmup-steps "$PSI_WARMUP_STEPS" \
  --psi-warmup-lr "$PSI_WARMUP_LR" \
  --seed "$SEED" \
  --width-factor "$WIDTH_FACTOR" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --pretrain-batch-size "$PRETRAIN_BATCH_SIZE" \
  --pretrain-lr "$PRETRAIN_LR" \
  --pretrain-momentum "$PRETRAIN_MOMENTUM" \
  --pretrain-weight-decay "$PRETRAIN_WEIGHT_DECAY" \
  --max-T "$MAX_T" \
  --max-b1 "$MAX_B1" \
  --max-b2 "$MAX_B2" \
  --max-q "$MAX_Q" \
  --device "$DEVICE" \
  --results-dir "$RESULTS_DIR" \
  "${EXTRA_ARGS[@]}" \
  "$@"
