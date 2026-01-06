#!/usr/bin/env bash
set -euo pipefail

# Iterate over epsilons and runs
# Iterate over epsilons and runs
for eps in 0.1 1 3 5 8 10; do
  # Format epsilon for directory names (0.1 -> 0p1)
  eps_tag="${eps//./p}"

  for run in {1..5}; do
    echo "=================================================================="
    echo "Processing Epsilon=${eps}, Run=${run}"
    echo "=================================================================="

    # --- DRO1 (Double-SPIDER) ---
    # Path format: multi_results/dro1_new_eps${eps}_run${run}/checkpoint.pt
    # Width Factor: 1.0
    
    DRO1_DIR="multi_results/dro1_new_eps${eps_tag}_run${run}"
    DRO1_CKPT="${DRO1_DIR}/checkpoint.pt"
    DRO1_OUT="MIA/outputs/dro1_eps${eps}_run${run}"

    if [[ -f "${DRO1_CKPT}" ]]; then
        echo "Running MIA on DRO1 (Double-SPIDER)..."
        python MIA/run_attacks.py \
            --checkpoint "${DRO1_CKPT}" \
            --width-factor 1.0 \
            --data-root data \
            --split-root MIA/splits \
            --output-dir "${DRO1_OUT}" \
            --member-count 10000 \
            --calibration-count 5000 \
            --per-class-thresholds > /dev/null
    else
        echo "Warning: DRO1 checkpoint not found at ${DRO1_CKPT}"
    fi

    # --- DRO2 (RS-DRO) ---
    # Path format: multi_results/dro2_new_eps${eps}_run${run}/rsdro_resnet20.pt
    # Width Factor: 0.5
    
    DRO2_DIR="multi_results/dro2_new_eps${eps_tag}_run${run}"
    DRO2_CKPT="${DRO2_DIR}/rsdro_resnet20.pt"
    DRO2_OUT="MIA/outputs/dro2_eps${eps}_run${run}"

    if [[ -f "${DRO2_CKPT}" ]]; then
        echo "Running MIA on DRO2 (RS-DRO)..."
        python MIA/run_attacks.py \
            --checkpoint "${DRO2_CKPT}" \
            --width-factor 0.5 \
            --data-root data \
            --split-root MIA/splits \
            --output-dir "${DRO2_OUT}" \
            --member-count 10000 \
            --calibration-count 5000 \
            --per-class-thresholds > /dev/null

    fi


    # --- Baseline (SGDA) ---
    # Path format: multi_results/Baseline_SGDA_eps${eps}_run${run}/checkpoint.pt
    # Width Factor: 1.0

    SGDA_DIR="multi_results/Baseline_SGDA_eps${eps_tag}_run${run}"
    SGDA_CKPT="${SGDA_DIR}/checkpoint.pt"
    SGDA_OUT="MIA/outputs/Baseline_SGDA_eps${eps}_run${run}"

    if [[ -f "${SGDA_CKPT}" ]]; then
        echo "Running MIA on Baseline (SGDA)..."
        python MIA/run_attacks.py \
            --checkpoint "${SGDA_CKPT}" \
            --width-factor 1.0 \
            --data-root data \
            --split-root MIA/splits \
            --output-dir "${SGDA_OUT}" \
            --member-count 10000 \
            --calibration-count 5000 \
            --per-class-thresholds > /dev/null
    else
        echo "Warning: SGDA checkpoint not found at ${SGDA_CKPT}"
    fi

    # --- Baseline (Diff) ---
    # Path format: multi_results/Baseline_Diff_eps${eps}_run${run}/checkpoint.pt
    # Width Factor: 1.0

    DIFF_DIR="multi_results/Baseline_Diff_eps${eps_tag}_run${run}"
    DIFF_CKPT="${DIFF_DIR}/checkpoint.pt"
    DIFF_OUT="MIA/outputs/Baseline_Diff_eps${eps}_run${run}"

    if [[ -f "${DIFF_CKPT}" ]]; then
        echo "Running MIA on Baseline (Diff)..."
        python MIA/run_attacks.py \
            --checkpoint "${DIFF_CKPT}" \
            --width-factor 1.0 \
            --data-root data \
            --split-root MIA/splits \
            --output-dir "${DIFF_OUT}" \
            --member-count 10000 \
            --calibration-count 5000 \
            --per-class-thresholds > /dev/null
    else
        echo "Warning: Diff checkpoint not found at ${DIFF_CKPT}"
    fi

  done
done

echo "Todos MIA experiments completed."
