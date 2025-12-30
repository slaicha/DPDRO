# Applied Patches and Fixes

This document tracks execution-critical changes applied to the DPDRO repository to make it runnable on this workstation.

## 1. Feature Additions & Bug Fixes (`apply_workstation_patches.py`)

### Baseline/SGDA/main.py
- **Added**: `--save-model` argument.
- **Added**: Logic to save `checkpoint.pt` at the end of training.

### Baseline/Diff/main.py
- **Added**: `--save-model` argument.
- **Added**: Logic to save `checkpoint.pt` at the end of training.

### dro1_new/algorithm.py
- **Renamed**: `self.linear` -> `self.fc` for consistency with ResNet definition.
- **Added**: `--save-model` argument and checkpoint saving logic.

### dro2_new/train_rsdro.py
- **Fixed**: Broken Python format string in error message (`max{...}` -> `max(...)`).

### run_all_projects.sh
- **Updated**: Added `--save-model` flag to all project invocations.

## 2. OOM Fix for dro1_new

### dro1_new/algorithm.py
- **Issue**: For `epsilon=5`, the algorithm calculates a batch size `N3` of ~1 billion (capped at dataset size 25,500). Loading all 25,500 images onto the GPU for a single gradient step caused `CUDA Out Of Memory`.
- **Fix**: Implemented **Gradient Accumulation** (Chunking).
    - Added `_compute_gradient` method to `DPDoubleSpiderTrainer`.
    - Modified training loop to split large batches into 256-sample chunks.
    - Chunks are moved to GPU sequentially, gradients are computed and accumulated on the accumulation tensor.
    - This allows processing arbitrarily large "logical" batches with constant peak VRAM usage.

## 3. Configuration Fixes

### run_dro.sh
- **Issue**: `dro1_new` was saving checkpoints to the current working directory (`./checkpoint.pt`) because the output directory argument was missing.
- **Fix**: Added `--output-dir "${out_dir}"` to the `run_dro1_new` function invocation.

## 4. Accuracy Fix (Warm Start)

### dro1_new/algorithm.py
- **Issue**: The code trained a baseline model (Standard SGD) to ~55% accuracy but discarded it, initializing a fresh random model for the DP phase. With strict privacy noise, this random model failed to learn (8% accuracy).
- **Fix**: Modified initialization of `dp_model` to copy weights from the pre-trained `model` (warm start).
