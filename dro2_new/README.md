# Recursive Spider DRO (RS-DRO)

This project implements the Recursive Spider Distributionally Robust Optimization (RS-DRO) algorithm with differential privacy noise injection for training a CIFAR10-ST ResNet-20 model. The implementation follows the specification provided in the prompt, including the exact batch-size, phase-size, and noise-schedule formulas.

## Repository layout

```
new/
├── README.md
├── rsdro/
│   ├── __init__.py
│   ├── data.py              # CIFAR10-ST construction & dataloader helpers
│   ├── hyperparams.py       # Closed-form hyperparameter calculations
│   ├── models.py            # CIFAR-style ResNet20
│   ├── objectives.py        # g / f gradients + Psi computation
│   ├── rsdro.py             # Core RS-DRO training loop
│   └── utils.py             # Shared math helpers
└── train_rsdro.py           # CLI entry-point for experiments
```

## Prerequisites

Install PyTorch (with CUDA if available) and torchvision. Example (CPU-only):

```bash
pip install torch torchvision
```

The script downloads CIFAR10 into `./data` on first use.

## Running RS-DRO

### Quick start script

The easiest way is to run the convenience wrapper:

```bash
cd new
bash new.sh
```

Environment variables override the defaults (e.g. `DEVICE=cpu`, `WIDTH_FACTOR=0.75`, `PRETRAIN_EPOCHS=20`, `PRETRAIN_BALANCED=0`, `MAX_T=10`, `RESULTS_DIR=/tmp/rsdro`, etc.), and any extra CLI flags are forwarded to `train_rsdro.py`.

### Manual invocation

You can also call the training script directly:

```bash
python train_rsdro.py --device cuda --rho 0.1 --G 1.0 --L 1.0 --c 10.0 --eta-t-squared 1e-4
```

Key arguments:

- `--epsilon` and `--rho` control the DRO/DP objective, defaulting to 4.0 and 0.1 respectively.
- `--G`, `--L`, and `--c` are the constants from the RS-DRO analysis (see specification).
- `--lambda0` fixes the lower bound for the dual variable (default `1e-3`).
- `--eta-t-squared` sets the auxiliary step-size term used inside `η = min{1/(2L), c·η_t^2}` and `β_t = c·η_t^2`.
- `--psi-warmup-*` configure the quick surrogate search for `Ψ_0 = Ψ(0;S) - min_w Ψ(w;S)`.
- `--pretrain-*` let you run a quick ERM warm-up (epochs, batch size, LR, momentum, weight decay, balanced sampler) before RS-DRO; this is helpful for stabilising training and quickly surpassing accuracy targets, especially on the imbalanced CIFAR10-ST split. Balanced sampling is **enabled by default**; pass `--no-pretrain-balanced` to disable.
- `--width-factor` rescales the ResNet20 channels (default `0.5` on CPU-only runs) so the pipeline fits into modest memory footprints without OOM.
- `--max-T`, `--max-b1`, `--max-b2`, and `--max-q` cap the closed-form hyperparameters to practical values (defaults 5/4096/64/5). Adjust these if you want longer RS-DRO phases once you have more memory.

The script will:

1. Build the imbalanced CIFAR10-ST dataset (`n = 25,500`).
2. Estimate `Ψ_0` by evaluating the true objective at initialization and after a short cross-entropy warm-up.
3. Check the sample-size condition `n ≥ max{(Gε)^2 /(Ψ_0 L d log(1/δ)), √d·max[1, √(LΨ_0/G)] / ε}` and report if the inequality is violated by the provided constants.
4. Compute `(T, q, b1, b2, η, β_t, σ₁, σ₂, ĥσ₂, σ_{s_t})` exactly as in the specification and print them.
5. Run RS-DRO via `rsdro.run_rs_dro`, returning a random iterate `(w_τ, λ_τ)` with τ ∼ [T].
6. Evaluate the trained model on the standard CIFAR10 test split and save `runs/rsdro/summary.json`, plus an optional checkpoint when `--save-model` is set.

## Notes on the implementation

- The ResNet20 is the standard CIFAR variant (3 residual stages with 3 BasicBlocks each).
- `g(w; ξ) = exp(ℓ(x; ξ)/λ)` is implemented with explicit gradient tracking. We clamp the exponent input to ±6 to guard against overflow.
- The `Ψ` objective is evaluated with per-sample cross entropy and the expression `Ψ(x, λ) = λ log( (1/n) Σ exp(ℓ/λ) ) + (λ - λ₀)ρ`.
- For privacy noise, the Gaussian variances follow the provided formulas, including the adaptive variance `min{σ₂² ⋅ ||w_t − w_{t−1}||², ĥσ₂²}`.
- A lightweight SGD pretraining stage (enabled by default: 12 balanced epochs with batch size 512) quickly pushes the model above 53% accuracy before RS-DRO fine-tuning. Disable or tweak via the `PRETRAIN_*` knobs if needed.
- Reservoir sampling is used so the algorithm does not store all `T` iterates when returning `w_τ`.
- Both the imbalanced training set (with augmentation) and a deterministic evaluation copy (without augmentation) are constructed to keep the `Ψ` computation reproducible.
- When GPU memory is constrained, start with `WIDTH_FACTOR=0.5 bash new.sh` (or `--width-factor 0.5`) to shrink the network while keeping the RS-DRO loop unchanged.

## Output artifacts

After a run completes you will find:

- `runs/rsdro/summary.json` – metadata (n, δ, Ψ₀, hyperparameters, final λ, τ, and accuracy).
- `runs/rsdro/rsdro_resnet20.pt` – optional checkpoint with the model state_dict and final λ (when `--save-model` is used).

Feel free to adapt `train_rsdro.py` for sweeps or plug the `rsdro` package into larger experiments.
