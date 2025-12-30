# DPDRO Experiment Results Report

This document details the final performance metrics for all trained models, including both **Test Accuracy** (Utility) and **Membership Inference Attack (MIA)** success rates (Privacy Leakage).

## 1. Experimental Setup

- **Dataset**: CIFAR10-ST (Skewed Training Data).
- **Privacy Budget**: $\epsilon \in \{5, 10\}$.
- **Repeated Runs**: 5 independent runs per configuration.

### Algorithms Evaluated
1.  **Baseline (SGDA)**: Differentially Private Stochastic Gradient Descent with Adaptive Clipping.
2.  **Baseline (Diff)**: Private Diffusion-based Optimization.
3.  **DRO1 (Double-SPIDER)**: DP Distributionally Robust Optimization (initialized with pre-trained weights).
4.  **DRO2 (RS-DRO)**: Recursive Spider DRO (trained from scratch with small models).

---

## 2. Global Summary

| Algorithm | Epsilon | Test Accuracy (%) | MIA Leakage (AUC) |
| :--- | :---: | :---: | :---: |
| **Baseline (SGDA)** | 5 | 53.59 | 0.94 (High) |
| **Baseline (Diff)** | 5 | 53.34 | 0.96 (High) |
| **DRO1 (Double-SPIDER)** | 5 | **56.38** | **0.97 (Highest)** |
| **DRO2 (RS-DRO)** | 5 | 55.18 | **0.80 (Lowest)** |
| | | | |
| **Baseline (SGDA)** | 10 | 54.34 | 0.95 (High) |
| **Baseline (Diff)** | 10 | 54.17 | 0.96 (High) |
| **DRO1 (Double-SPIDER)** | 10 | **56.92** | **0.97 (Highest)** |
| **DRO2 (RS-DRO)** | 10 | 56.89 | **0.78 (Lowest)** |

**Key Finding**: **DRO2 (RS-DRO)** offers the best trade-off. It provides competitive accuracy (~56%) while significantly improving privacy protections (AUC 0.80) compared to all other methods (AUC > 0.94).

---

## 3. Detailed Training Results (Test Accuracy)

Values represent the mean accuracy of the final model on the held-out test set, averaged over 5 runs.

| Algorithm | Epsilon 5.0 | Epsilon 10.0 |
| :--- | :--- | :--- |
| **Baseline (SGDA)** | 53.59% | 54.34% |
| **Baseline (Diff)** | 53.34% | 54.17% |
| **DRO1 (Double-SPIDER)**| 56.38% | 56.92% |
| **DRO2 (RS-DRO)** | 55.18% | 56.89% |

---

## 4. Detailed MIA Results (Privacy Leakage)

We performed **Membership Inference Attacks (MIA)** using two metrics strings:
-   **Confidence-based**: Uses the model's prediction confidence on the true class.
-   **Loss-based**: Uses the request's cross-entropy loss (stronger attack).

**Metrics**:
-   **AUC (Area Under Curve)**: 0.5 indicates random guessing (perfect privacy). 1.0 indicates perfect attacker success (no privacy).
-   **TPR @ 1% FPR**: The percentage of training members identified when the False Positive Rate is fixed at 1%.

### Epsilon = 5.0

| Algorithm | Attack | AUC (Mean ± Std) | TPR @ 1% FPR (Mean ± Std) |
| :--- | :--- | :--- | :--- |
| **Baseline (SGDA)** | Confidence | 0.7870 ± 0.049 | 21.95 ± 6.0% |
| | Loss | 0.9365 ± 0.026 | 61.12 ± 10.5% |
| **Baseline (Diff)** | Confidence | 0.8443 ± 0.006 | 30.16 ± 2.4% |
| | Loss | 0.9633 ± 0.006 | 71.35 ± 10.9% |
| **DRO1** | Confidence | 0.8911 ± 0.003 | 39.42 ± 1.4% |
| | Loss | **0.9723 ± 0.002** | **73.59 ± 3.5%** |
| **DRO2** | Confidence | **0.6395 ± 0.025** | **1.47 ± 1.2%** |
| | Loss | **0.8014 ± 0.038** | **1.70 ± 1.5%** |

### Epsilon = 10.0

| Algorithm | Attack | AUC (Mean ± Std) | TPR @ 1% FPR (Mean ± Std) |
| :--- | :--- | :--- | :--- |
| **Baseline (SGDA)** | Confidence | 0.8081 ± 0.036 | 25.07 ± 6.2% |
| | Loss | 0.9480 ± 0.019 | 70.34 ± 10.0% |
| **Baseline (Diff)** | Confidence | 0.8295 ± 0.014 | 27.36 ± 3.8% |
| | Loss | 0.9575 ± 0.012 | 70.49 ± 7.1% |
| **DRO1** | Confidence | 0.8910 ± 0.001 | 40.87 ± 0.9% |
| | Loss | **0.9715 ± 0.002** | **72.39 ± 2.9%** |
| **DRO2** | Confidence | **0.6334 ± 0.011** | **0.90 ± 0.4%** |
| | Loss | **0.7825 ± 0.015** | **0.95 ± 0.5%** |

---

## 5. Analysis & Conclusions

1.  **Warm-Starting Compromises Privacy (DRO1)**:
    -   DRO1 was initialized with a pre-trained non-private model (accuracy ~55%) to solve convergence issues.
    -   While this achieved high final accuracy, the MIA results (AUC > 0.97) show that the subsequent DP training phase was insufficient to "hide" the memorable samples from the initialization.
    -   **Result**: DRO1 behaves almost like a non-private model in terms of leakage.

2.  **RS-DRO is Robust (DRO2)**:
    -   DRO2 was trained from scratch with a smaller architecture (`width-factor=0.5`).
    -   It demonstrates significantly lower leakage (AUC ~0.78-0.80) while matching the accuracy of the other methods.
    -   **Recommendation**: For applications requiring strict privacy guarantees, **DRO2** is the preferred method.
