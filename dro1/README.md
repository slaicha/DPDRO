# DP Double-Spider on CIFAR10-ST

This package provides a reproducible implementation of the DP Double-Spider algorithm for training a ResNet classifier on the imbalanced CIFAR10-ST dataset.

## Repository layout

- `ascdro/algorithms.py` – core DP Double-Spider trainer, including gradient tracking, noise calibration, numerical stabilisation, and per-step logging.
- `ascdro/datasets.py` – lightweight CIFAR10-ST dataset wrapper loading the prepared NPZ archives with standard augmentations.
- `ascdro/models.py` – ResNet20/32-style backbones tailored for CIFAR inputs.
- `ascdro/risk.py` – KL-based risk model used inside the min-max objective.
- `ascdro/utils.py` – utilities (metric meters, batch streamers, device helpers, etc.).
- `training/train_cifar10_st_dp_double_spider.py` – command-line entry point that estimates Lipschitz/gradient constants, calibrates sampling sizes/noise, launches training, and writes results to `metrics_double_spider.json` plus `process.json` (per-step trace).
- `scripts/run_cifar10_st_dp_double_spider.sh` – default run script with production hyper-parameters.
- `scripts/run_cifar10_st_dp_double_spider_quick.sh` – quick sanity script that uses reduced budgets for fast smoke tests.
- `__init__.py` – exposes the high-level API when importing `dro_alg1` as a module.

## Usage

1. **Prepare CIFAR10-ST** (if not already cached) so that `<DATA_ROOT>/train.npz` and `test.npz` exist. The scripts assume the default location at `../CIFAR10/cifar10_st` relative to this directory.
2. **Quick sanity check**
   ```bash
   bash dro_alg1/scripts/run_cifar10_st_dp_double_spider_quick.sh
   ```
   This generates `process.json` (per-step metrics) and `metrics_double_spider.json` (full summary) under `dro_alg1/runs/cifar10_st_double_spider_quick` by default.
3. **Full training**
   ```bash
   bash dro_alg1/scripts/run_cifar10_st_dp_double_spider.sh
   ```
   Override any hyper-parameter by exporting the matching environment variable (e.g. `EPSILON`, `ALPHA_SCALE`, `GRAD_CLIP`) or by adding extra CLI arguments to the script invocation.

Both scripts ultimately call `training/train_cifar10_st_dp_double_spider.py`; refer to `--help` for the full set of tunable arguments.
