#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/../CIFAR10/cifar10_st}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/cifar10_st_double_spider_quick}"

# Quick sanity defaults (override via env or CLI args)
EPSILON="${EPSILON:-4.0}"
DELTA="${DELTA:-}"
EPOCHS="${EPOCHS:-30}"
LAMBDA0="${LAMBDA0:-1e-3}"
ETA_MIN="${ETA_MIN:-1e-4}"
RHO="${RHO:-0.5}"
MODEL_DEPTH="${MODEL_DEPTH:-20}"
WIDTH_MULT="${WIDTH_MULT:-1.0}"
D0="${D0:-0.05}"
D1="${D1:-0.05}"
D2="${D2:-0.05}"
H_BOUND="${H_BOUND:-1.0}"
EST_BATCH="${EST_BATCH:-64}"
EVAL_BATCH="${EVAL_BATCH:-128}"
MAX_CONST_BATCHES="${MAX_CONST_BATCHES:-5}"
C_NOISE="${C_NOISE:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-1234}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-64}"
VERBOSE_FLAG="${VERBOSE_FLAG:-true}"
ALPHA_SCALE="${ALPHA_SCALE:-0.2}"
GRAD_CLIP="${GRAD_CLIP:-2.0}"
GRAD_CLIP_ETA="${GRAD_CLIP_ETA:-0.5}"
EXP_CLIP="${EXP_CLIP:-10.0}"
MAX_BATCH="${MAX_BATCH:-1024}"
MAX_Q="${MAX_Q:-32}"

EXTRA_ARGS=("$@")

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/training/train_cifar10_st_dp_double_spider.py"
  "--data-root" "${DATA_ROOT}"
  "--output-dir" "${OUTPUT_DIR}"
  "--epsilon" "${EPSILON}"
  "--epochs" "${EPOCHS}"
  "--lambda0" "${LAMBDA0}"
  "--eta-min" "${ETA_MIN}"
  "--rho" "${RHO}"
  "--model-depth" "${MODEL_DEPTH}"
  "--width-multiplier" "${WIDTH_MULT}"
  "--D0" "${D0}"
  "--D1" "${D1}"
  "--D2" "${D2}"
  "--H" "${H_BOUND}"
  "--estimation-batch-size" "${EST_BATCH}"
  "--eval-batch-size" "${EVAL_BATCH}"
  "--alpha-scale" "${ALPHA_SCALE}"
  "--grad-clip" "${GRAD_CLIP}"
  "--grad-clip-eta" "${GRAD_CLIP_ETA}"
  "--exp-clip" "${EXP_CLIP}"
  "--max-batch" "${MAX_BATCH:-1024}"
  "--max-q" "${MAX_Q:-32}"
  "--num-workers" "${NUM_WORKERS}"
  "--seed" "${SEED}"
  "--log-interval" "${LOG_INTERVAL}"
  "--eval-interval" "${EVAL_INTERVAL}"
  "--max-constant-batches" "${MAX_CONST_BATCHES}"
  "--c-noise" "${C_NOISE}"
)

if [[ -n "${DELTA}" ]]; then
  CMD+=("--delta" "${DELTA}")
fi

if [[ "${VERBOSE_FLAG}" == "true" ]]; then
  CMD+=("--verbose")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "[INFO] Quick sanity run: ${CMD[*]}"
exec "${CMD[@]}"
