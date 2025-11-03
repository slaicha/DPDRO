#!/usr/bin/env bash
# Aggregate runner for Baseline/SGDA, Baseline/Diff, dro1, and dro2.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${ROOT_DIR}/Results"
mkdir -p "${RESULTS_DIR}"

# Dataset locations (override via environment if needed).
CIFAR_TORCH_ROOT="${CIFAR_TORCH_ROOT:-${ROOT_DIR}/Baseline/SGDA/data}"
CIFAR10_ST_ROOT="${CIFAR10_ST_ROOT:-${ROOT_DIR}/CIFAR10/cifar10_st}"

ensure_cifar10_st() {
  if [[ ! -f "${CIFAR10_ST_ROOT}/train.npz" || ! -f "${CIFAR10_ST_ROOT}/test.npz" ]]; then
    cat <<EOF
[ERROR] CIFAR10-ST NPZ files not found.
Expected to see:
  ${CIFAR10_ST_ROOT}/train.npz
  ${CIFAR10_ST_ROOT}/test.npz
Populate these files (see dro1/README.md) or set CIFAR10_ST_ROOT.
EOF
    exit 1
  fi
}

run_baseline_sgda() {
  local out_dir="${RESULTS_DIR}/Baseline_SGDA"
  mkdir -p "${out_dir}"
  (
    cd "${ROOT_DIR}/Baseline/SGDA"
    python main.py \
      --data_root "${CIFAR_TORCH_ROOT}" \
      --batch_size 128 \
      --epsilon 4.0 \
      --total_epochs 30 \
      --lr_w 0.2 \
      --lr_v 0.2 \
      --output_dir "${out_dir}" \
      2>&1 | tee "${out_dir}/train.log"
  )
}

run_baseline_diff() {
  local out_dir="${RESULTS_DIR}/Baseline_Diff"
  mkdir -p "${out_dir}"
  (
    cd "${ROOT_DIR}/Baseline/Diff"
    python main.py \
      --data_root "${CIFAR_TORCH_ROOT}" \
      --batch_size 128 \
      --epsilon 4.0 \
      --total_epochs 30 \
      --lr 0.2 \
      --lr_alpha 0.2 \
      --output_dir "${out_dir}" \
      2>&1 | tee "${out_dir}/train.log"
  )
}

run_dro1_new() {
  local out_dir="${RESULTS_DIR}/dro1_new"
  mkdir -p "${out_dir}"
  (
    cd "${ROOT_DIR}"
    python dro1_new/algorithm.py \
      --data-root "${CIFAR_TORCH_ROOT}" \
      --train-batch-size 128 \
      --test-batch-size 100 \
      --baseline-epochs 20 \
      --epsilon 4.0 \
      2>&1 | tee "${out_dir}/train.log"
  )
}

run_dro2() {
  ensure_cifar10_st
  local out_dir="${RESULTS_DIR}/dro2"
  mkdir -p "${out_dir}"
  (
    cd "${ROOT_DIR}/dro2"
    python training/train_cifar10_st_ascdro.py \
      --data-root "${CIFAR10_ST_ROOT}" \
      --output-dir "${out_dir}/artifacts" \
      --epsilon 4.0 \
      --eta 0.2 \
      --epochs 30 \
      2>&1 | tee "${out_dir}/train.log"
  )
}

write_summary() {
  local summary_file="${RESULTS_DIR}/summary.txt"
  : > "${summary_file}"

  python - <<'PY'
import json, os
root = os.path.abspath("Results")
entries = []

def maybe_add(name, path, scale=1.0, field="best_accuracy"):
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    value = data.get(field)
    if value is None:
        return
    entries.append((name, float(value) * scale))

maybe_add("Baseline/SGDA", os.path.join(root, "Baseline_SGDA", "results.json"))
maybe_add("Baseline/Diff", os.path.join(root, "Baseline_Diff", "results.json"))

dro1_path = os.path.join(root, "dro1", "artifacts", "metrics_double_spider.json")
if os.path.isfile(dro1_path):
    with open(dro1_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    test = data.get("final_metrics", {}).get("test", {})
    acc = test.get("accuracy")
    if acc is not None:
        entries.append(("dro1", float(acc) / 100.0))

dro2_path = os.path.join(root, "dro2", "artifacts", "metrics_ascdro.json")
if os.path.isfile(dro2_path):
    with open(dro2_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    history = data.get("history", [])
    best = 0.0
    for entry in history:
        val_acc = entry.get("val_acc")
        if val_acc is not None:
            best = max(best, float(val_acc) / 100.0)
    if best > 0.0:
        entries.append(("dro2", best))

summary_path = os.path.join(root, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as fh:
    for name, val in entries:
        fh.write(f"{name}: {val:.4f}\n")
PY
}

run_baseline_sgda
run_baseline_diff
run_dro1_new
run_dro2
write_summary

echo "[INFO] All runs complete. Results gathered in ${RESULTS_DIR}/"
