#!/usr/bin/env bash
set -euo pipefail

# Root of the project (directory containing dro_alg1/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python executable (override by setting PYTHON=...)
PYTHON_BIN="${PYTHON:-python}"

# Dataset / output locations (can be overridden via env vars)
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/../CIFAR10/cifar10_st}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/cifar10_st_double_spider}"

# Algorithm constants (edit as needed; must match theoretical requirements)
EPSILON="${EPSILON:-1.0}"
DELTA="${DELTA:-1e-5}"
ITERATIONS="${ITERATIONS:-1024}"
LAMBDA0="${LAMBDA0:-1e-3}"
ETA_MIN="${ETA_MIN:-1e-4}"
RHO="${RHO:-0.5}"
MODEL_DEPTH="${MODEL_DEPTH:-20}"
WIDTH_MULT="${WIDTH_MULT:-1.0}"
D0="${D0:-1.0}"
D1="${D1:-1.0}"
D2="${D2:-1.0}"
H_BOUND="${H_BOUND:-1.0}"
EST_BATCH="${EST_BATCH:-128}"
EVAL_BATCH="${EVAL_BATCH:-256}"
MAX_CONST_BATCHES="${MAX_CONST_BATCHES:-20}"
C_NOISE="${C_NOISE:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
VERBOSE_FLAG="${VERBOSE_FLAG:-true}"
ALPHA_SCALE="${ALPHA_SCALE:-1.0}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
GRAD_CLIP_ETA="${GRAD_CLIP_ETA:-1.0}"
EXP_CLIP="${EXP_CLIP:-20.0}"

EXTRA_ARGS=("$@")

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/training/train_cifar10_st_dp_double_spider.py"
  "--data-root" "${DATA_ROOT}"
  "--output-dir" "${OUTPUT_DIR}"
  "--epsilon" "${EPSILON}"
  "--delta" "${DELTA}"
  "--iterations" "${ITERATIONS}"
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
  "--num-workers" "${NUM_WORKERS}"
  "--seed" "${SEED}"
  "--log-interval" "${LOG_INTERVAL}"
  "--eval-interval" "${EVAL_INTERVAL}"
  "--max-constant-batches" "${MAX_CONST_BATCHES}"
  "--c-noise" "${C_NOISE}"
)

if [[ "${VERBOSE_FLAG}" == "true" ]]; then
  CMD+=("--verbose")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "[INFO] Running: ${CMD[*]}"
exec "${CMD[@]}"
