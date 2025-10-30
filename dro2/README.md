# Project Overview

This repository collects the Differentially Private ASCDRO training pipeline and
utilities used for image classification experiments.  The codebase is split
into two major components:

* `dro_new/`: reusable ASCDRO (Double Spider DRO) implementation, training
  scripts, and supporting datasets/models/utilities.
* `imagenet/`: entry points and assets specific to the ImageNet long-tailed
  (ImageNetLT) experiments built on top of the ASCDRO stack.


## Repository Layout

| Path | Description |
|------|-------------|
| `dro_new/ascdro/algorithms.py` | Core ASCDRO + Private SpiderBoost trainer implementation. |
| `dro_new/ascdro/datasets.py` | Dataset helpers (CIFAR10-ST NPZ loader, ImageNet folder wrapper). |
| `dro_new/ascdro/models.py` | ResNet builders for CIFAR and ImageNet backbones. |
| `dro_new/ascdro/risk.py` | KL risk model used by ASCDRO. |
| `dro_new/ascdro/utils.py` | Shared utilities (averaging meters, device helpers, etc.). |
| `dro_new/training/train_cifar10_st_ascdro.py` | ASCDRO training entry point for CIFAR10-ST. |
| `dro_new/scripts/run_cifar10_st_ascdro.sh` | Bash wrapper that prepares data and launches the CIFAR10-ST trainer. |
| `dro_new/scripts/run_cifar10_st_experiment.sh` | Convenience launcher for b5 sweeps on CIFAR10-ST. |
| `dro_new/datasets/prepare_datasets.py` | Dataset preparation utilities (CIFAR10-ST, ImageNet-LT, iNat2018). |
| `imagenet/train_imagenet_lt.py` | ASCDRO/DSDRO training script for ImageNet-LT using long-tailed manifests. |
| `imagenet/ImageNet_LT/*.txt` | Manifest files describing the long-tailed train/val splits (generated via `prepare_imagenet_lt.py`). |


## Running Training

For the ImageNet-LT experiment, first make sure the ImageNet images are
available under `imagenet/train/`, then launch:

```bash
python imagenet/train_imagenet_lt.py \
  --epsilon 8 \
  --delta 0.1 \
  --batch-size 192 \
  --num-workers 8 \
  --eta 0.05 \
  --rho 0.5 \
  --lambda0 0.1 \
  --beta 0.5
```

The script supports many optional arguments (image size, warm-up steps,
debug-sample caps, etc.); run `python imagenet/train_imagenet_lt.py --help` for
details.  Training logs and checkpoints are written to
`imagenet/runs/<experiment>/`.

For the CIFAR10-ST baseline, you can trigger the equivalent ASCDRO pipeline via

```bash
python dro_new/training/train_cifar10_st_ascdro.py \
  --data-root /path/to/cifar10_st \
  --output-dir dro_new/runs/cifar10_st_demo \
  --epsilon 1
```

or use the helper shell wrapper:

```bash
./dro_new/scripts/run_cifar10_st_ascdro.sh 1
```


## Notes

* All training code shares the same ASCDRO backbone, ensuring consistency with
  the Double Spider DRO algorithm specification.
* Generated artifacts (logs, checkpoints) are written under the respective
  `runs/` directories and can be safely archived or removed.
