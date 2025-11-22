#!/usr/bin/env bash
# Aggregate runner for Baseline/SGDA, Baseline/Diff, dro1_new, and dro2_new.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/multi_results"
mkdir -p "${RESULTS_DIR}"

# Dataset locations (override via environment if needed).
CIFAR_TORCH_ROOT="${CIFAR_TORCH_ROOT:-${ROOT_DIR}/Baseline/SGDA/data}"

# Epsilon sweep configuration (override via EPS_LIST env, e.g., "0.2 0.5")
EPS_VALUES_DEFAULT="5 10"
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

run_baseline_sgda() {
  local eps="$1"
  local tag
  tag="$(format_eps_tag "${eps}")"
  for run_idx in $(seq 1 "${REPEAT_COUNT}"); do
    local out_dir="${RESULTS_DIR}/Baseline_SGDA_eps${tag}_run${run_idx}"
    mkdir -p "${out_dir}"
    local seed
    seed="$(rand_seed)"
    (
      cd "${ROOT_DIR}/Baseline/SGDA"
      python main.py \
        --data_root "${CIFAR_TORCH_ROOT}" \
        --batch_size 128 \
        --epsilon "${eps}" \
        --total_epochs 30 \
        --lr_w 0.2 \
        --lr_v 0.2 \
        --seed "${seed}" \
        --output_dir "${out_dir}" \
        2>&1 | tee "${out_dir}/train.log"
    )
  done
}

run_baseline_diff() {
  local eps="$1"
  local tag
  tag="$(format_eps_tag "${eps}")"
  for run_idx in $(seq 1 "${REPEAT_COUNT}"); do
    local out_dir="${RESULTS_DIR}/Baseline_Diff_eps${tag}_run${run_idx}"
    mkdir -p "${out_dir}"
    local seed
    seed="$(rand_seed)"
    (
      cd "${ROOT_DIR}/Baseline/Diff"
      python main.py \
        --data_root "${CIFAR_TORCH_ROOT}" \
        --batch_size 128 \
        --epsilon "${eps}" \
        --total_epochs 30 \
        --lr 0.2 \
        --lr_alpha 0.2 \
        --seed "${seed}" \
        --output_dir "${out_dir}" \
        2>&1 | tee "${out_dir}/train.log"
    )
  done
}

run_dro1_new() {
  local eps="$1"
  local tag
  tag="$(format_eps_tag "${eps}")"
  for run_idx in $(seq 1 "${REPEAT_COUNT}"); do
    local out_dir="${RESULTS_DIR}/dro1_new_eps${tag}_run${run_idx}"
    mkdir -p "${out_dir}"
    (
      cd "${ROOT_DIR}"
      python dro1_new/algorithm.py \
        --data-root "${CIFAR_TORCH_ROOT}" \
        --train-batch-size 128 \
        --test-batch-size 100 \
        --baseline-epochs 20 \
        --epsilon "${eps}" \
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
        2>&1 | tee "${out_dir}/train.log"
    )
  done
}

write_summary() {
  local summary_file="${RESULTS_DIR}/summary.txt"
  : > "${summary_file}"

  EPS_VALUES_STR_ENV="${EPS_VALUES_STR}" REPEAT_COUNT_ENV="${REPEAT_COUNT}" python - <<'PY'
import json, os
from collections import defaultdict

eps_values = os.environ.get("EPS_VALUES_STR_ENV", "0.5 1 5 10").split()
repeat_count = int(os.environ.get("REPEAT_COUNT_ENV", "5"))
root = os.path.abspath("multi_results")

def tagify(eps):
    return eps.replace(".", "p")

def load_accuracy(path, field="best_accuracy"):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get(field)

def dro1_best_from_log(path):
    if not os.path.isfile(path):
        return None
    best = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if "Best Test Accuracy:" in line:
                try:
                    val = float(line.split("Best Test Accuracy:")[-1].split("%")[0].strip())
                    best = val / 100.0
                except ValueError:
                    continue
    return best

averages = defaultdict(dict)

for eps in eps_values:
    tag = tagify(eps)
    # Baseline SGDA
    s_acc = []
    for run_idx in range(1, repeat_count + 1):
        acc = load_accuracy(os.path.join(root, f"Baseline_SGDA_eps{tag}_run{run_idx}", "results.json"))
        if acc is not None:
            s_acc.append(float(acc))
    if s_acc:
        averages[eps]["Baseline/SGDA"] = sum(s_acc) / len(s_acc)

    # Baseline Diff
    d_acc = []
    for run_idx in range(1, repeat_count + 1):
        acc = load_accuracy(os.path.join(root, f"Baseline_Diff_eps{tag}_run{run_idx}", "results.json"))
        if acc is not None:
            d_acc.append(float(acc))
    if d_acc:
        averages[eps]["Baseline/Diff"] = sum(d_acc) / len(d_acc)

    # dro1
    dr1 = []
    for run_idx in range(1, repeat_count + 1):
        acc = dro1_best_from_log(os.path.join(root, f"dro1_new_eps{tag}_run{run_idx}", "train.log"))
        if acc is not None:
            dr1.append(float(acc))
    if dr1:
        averages[eps]["dro1_new"] = sum(dr1) / len(dr1)

    # dro2
    dr2 = []
    for run_idx in range(1, repeat_count + 1):
        summary_path = os.path.join(root, f"dro2_new_eps{tag}_run{run_idx}", "summary.json")
        acc = load_accuracy(summary_path, field="test_accuracy")
        if acc is not None:
            dr2.append(float(acc))
    if dr2:
        averages[eps]["dro2_new"] = sum(dr2) / len(dr2)

with open(os.path.join(root, "summary.txt"), "w", encoding="utf-8") as fh:
    for eps in eps_values:
        if eps not in averages:
            continue
        fh.write(f"epsilon={eps}\n")
        for alg, val in averages[eps].items():
            fh.write(f"  {alg}: {val:.4f}\n")
        fh.write("\n")
PY
}

for eps in "${EPS_VALUES[@]}"; do
  run_baseline_sgda "${eps}"
  run_baseline_diff "${eps}"
  run_dro1_new "${eps}"
  run_dro2_new "${eps}"
done
write_summary

echo "[INFO] All runs complete. Results gathered in ${RESULTS_DIR}/"
