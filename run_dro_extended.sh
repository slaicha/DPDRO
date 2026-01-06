#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/multi_results"
mkdir -p "${RESULTS_DIR}"

# Dataset locations (override via environment if needed).
CIFAR_TORCH_ROOT="${CIFAR_TORCH_ROOT:-${ROOT_DIR}/Baseline/SGDA/data}"

# Epsilon configuration (specifically requested values)
EPS_VALUES_DEFAULT="0.1 1 3 8"
EPS_VALUES_STR="${EPS_LIST:-${EPS_VALUES_DEFAULT}}"
IFS=' ' read -r -a EPS_VALUES <<< "${EPS_VALUES_STR}"
REPEAT_COUNT="${REPEAT_COUNT:-5}"

format_eps_tag() {
  echo "$1" | tr '.' 'p'
}

rand_seed() {
  python - <<'PY'
import secrets
print(secrets.randbelow(2**31))
PY
}

run_dro1_new() {
  local eps="$1"
  local tag
  tag="$(format_eps_tag "${eps}")"
  for run_idx in $(seq 1 "${REPEAT_COUNT}"); do
    local out_dir="${RESULTS_DIR}/dro1_new_eps${tag}_run${run_idx}"
    if [ -d "${out_dir}" ]; then
        echo "Skipping ${out_dir}, already exists."
        continue
    fi
    mkdir -p "${out_dir}"
    (
      cd "${ROOT_DIR}"
      python dro1_new/algorithm.py \
        --data-root "${CIFAR_TORCH_ROOT}" \
        --train-batch-size 128 \
        --test-batch-size 100 \
        --baseline-epochs 20 \
        --epsilon "${eps}" \
        --run-dp \
        --output-dir "${out_dir}" \
        --save-model \
        2>&1 | tee "${out_dir}/train.log"
    )
  done
}

run_dro2_new() {
  local eps="$1"
  local tag
  tag="$(format_eps_tag "${eps}")"
  for run_idx in $(seq 1 "${REPEAT_COUNT}"); do
    local out_dir="${RESULTS_DIR}/dro2_new_eps${tag}_run${run_idx}"
    if [ -d "${out_dir}" ]; then
        echo "Skipping ${out_dir}, already exists."
        continue
    fi
    mkdir -p "${out_dir}"
    local seed
    seed="$(rand_seed)"
    (
      cd "${ROOT_DIR}/dro2_new"
      DATA_ROOT="${CIFAR_TORCH_ROOT}" \
      RESULTS_DIR="${out_dir}" \
      SEED="${seed}" \
      EPSILON="${eps}" \
      MAX_T=5 \
      MAX_B1=4096 \
      MAX_B2=64 \
      MAX_Q=5 \
      bash new.sh \
        --save-model \
        2>&1 | tee "${out_dir}/train.log"
    )
  done
}

# Run experiments
for eps in "${EPS_VALUES[@]}"; do
  run_dro1_new "${eps}"
  run_dro2_new "${eps}"
done

echo "[INFO] Extended runs complete. Results gathered in ${RESULTS_DIR}/"
