# DPDRO Experiment Results Report

This document details the final performance metrics for all trained models, including both Test Accuracy and Membership Inference Attack (MIA) success rates.

## 1. Experimental Setup

- Dataset: CIFAR10-ST (Skewed Training Data).
- Privacy Budget: $\epsilon \in \{0.1, 1, 3, 5, 8, 10\}$.
- Repeated Runs: 5 independent runs per configuration.

### Algorithms Evaluated
1.  Baseline (SGDA): Differentially Private Stochastic Gradient Descent with Adaptive Clipping. (Only trained for $\epsilon \in \{5, 10\}$)
2.  Baseline (Diff): Private Diffusion-based Optimization. (Only trained for $\epsilon \in \{5, 10\}$)
3.  DRO1 (Double-SPIDER): DP Distributionally Robust Optimization (initialized with pre-trained weights).
4.  DRO2 (RS-DRO): Recursive Spider DRO (trained from scratch with small models).

---

## 2. Extended Results Summary

The table below summarizes the Test Accuracy (Utility) and MIA Privacy Leakage (Cross-Entropy Loss-based AUC; Lower AUC is better, 0.5 is ideal privacy).

| Algorithm | Epsilon | Test Acc (Mean ± Std) | MIA AUC (Mean ± Std) |
| :--- | :--- | :--- | :--- |
| **Baseline_Diff** | 5 | 53.34 ± 1.70% | 0.9633 ± 0.0062 |
| **Baseline_Diff** | 10 | 54.17 ± 0.96% | 0.9575 ± 0.0115 |
| **Baseline_SGDA** | 5 | 53.59 ± 1.99% | 0.9365 ± 0.0256 |
| **Baseline_SGDA** | 10 | 54.34 ± 0.99% | 0.9480 ± 0.0188 |
| | | | |
| **DRO1** | 0.1 | 55.85 ± 1.26% | 0.9719 ± 0.0059 |
| **DRO1** | 1 | 56.32 ± 0.48% | 0.9723 ± 0.0045 |
| **DRO1** | 3 | 56.55 ± 0.96% | 0.9717 ± 0.0025 |
| **DRO1** | 5 | 56.38 ± 0.55% | 0.9723 ± 0.0021 |
| **DRO1** | 8 | 55.79 ± 1.02% | 0.9724 ± 0.0033 |
| **DRO1** | 10 | 56.92 ± 0.86% | 0.9715 ± 0.0022 |
| | | | |
| **DRO2** | 0.1 | 55.42 ± 1.69% | 0.8234 ± 0.0127 |
| **DRO2** | 1 | 54.48 ± 1.83% | 0.7932 ± 0.0072 |
| **DRO2** | 3 | 54.97 ± 2.01% | 0.7958 ± 0.0240 |
| **DRO2** | 5 | 55.18 ± 2.18% | 0.8014 ± 0.0383 |
| **DRO2** | 8 | 55.74 ± 1.34% | 0.8001 ± 0.0302 |
| **DRO2** | 10 | 56.89 ± 1.18% | 0.7825 ± 0.0149 |

### Key Observations
1.  **Privacy Stability**:
    *   DRO1 consistently leaks almost all membership information (AUC > 0.97) regardless of epsilon.
    *   DRO2 maintains much better privacy (AUC ~0.80) across the entire epsilon range.
2.  **Accuracy robustness**:
    *   Both DRO1 and DRO2 generally maintain accuracy > 54% even at very low epsilons, outperforming or matching the baselines which achieved ~53.5-54.5% at higher epsilons.
    *   DRO2's privacy advantage comes at virtually no cost to utility compared to DRO1.

---

## 3. Analysis & Conclusions

1.  Warm-Starting Compromises Privacy (DRO1):
    -   DRO1 was initialized with a pre-trained non-private model (accuracy ~55%) to solve convergence issues.
    -   While this achieved high final accuracy, the MIA results (AUC > 0.97) show that the subsequent DP training phase was insufficient to "hide" the memorable samples from the initialization.
    -   DRO1 behaves almost like a non-private model in terms of leakage.

2.  RS-DRO is Robust (DRO2):
    -   DRO2 was trained from scratch with a smaller architecture (`width-factor=0.5`).
    -   It demonstrates significantly lower leakage (AUC ~0.78-0.80) while matching the accuracy of the other methods.
