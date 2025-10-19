# Code Directory Overview

This repository hosts two ASCDRO codebases that share a common design but target
different experimental settings:

## `dro_alg1`

* Legacy ASCDRO implementation used for early CIFAR10-ST experiments.
* Contains:
  * `ascdro/` – algorithms, datasets, models and utilities for the original
    Double Spider DRO implementation.
  * `training/` – scripts to train CIFAR10-ST models with the legacy pipeline.
  * `scripts/` – helper shell wrappers (data preparation, parameter sweeps).
* Recommended when reproducing baseline experiments or comparing against the
  earlier algorithm variant.

## `dro_alg2`

* Current ASCDRO code (formerly named `dro_new`) used for the ImageNet-LT and
  CIFAR10-ST experiments in this project.
* Contains:
  * `ascdro/` – updated ASCDRO trainer, Private SpiderBoost estimator, dataset
    loaders, and model builders.
  * `training/` – training scripts (e.g., `train_cifar10_st_ascdro.py`).
  * `scripts/` – command-line wrappers for CIFAR10-ST runs and ε sweeps.
* Use this codebase together with `imagenet/train_imagenet_lt.py` to run the
  latest DSDRO experiments.

Both folders follow the same structure, allowing easy cross-referencing between
the legacy (`dro_alg1`) and updated (`dro_alg2`) implementations.
