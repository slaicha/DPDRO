# Empirical MIA on CIFAR10-ST (DP RS-DRO + ResNet-20)

This folder wires up the full empirical membership inference workflow you requested, now targeting the DP RS-DRO implementation in `dro2_new` (ResNet-20, default `width_factor=0.5`):

- Construct **CIFAR10-ST** (skewed: classes 0–4 keep last 100 only; classes 5–9 keep all 5000), plus the **public complement (24,500 samples)**, balanced eval pairs, and a disjoint non-member calibration split.
- Run **confidence** and **loss** attacks with ROC-AUC / TPR@1% / TPR@0.1% / precision@FPR, optional per-class thresholds.
- Optional **LiRA** (shadow models) pipeline with manifest tooling and a simple ResNet-20 shadow trainer.

Paths are relative to `/home/Arnold/Code`.

## What got added

- `data_prep.py` – materialises CIFAR10-ST, public complement, eval pairs, and calibration NPZ files under `MIA/splits/`.
- `datasets.py` – NPZ-backed datasets/dataloaders with membership labels.
- `model_utils.py` – ResNet-20 loader that reuses `dro2_new/rsdro` (no algorithm reimplementation), supports `width_factor`.
- `attacks.py` – confidence/loss/logit-margin scoring, threshold calibration, ROC/TPR/precision helpers, LiRA score computation.
- `run_attacks.py` – CLI to run confidence + loss attacks end-to-end (splits -> inference -> metrics/ROC plots).
- `make_shadow_manifest.py` – helper to sample shadow training indices and emit a manifest for LiRA.
- `train_shadows.py` – quick non-DP ResNet-20 trainer for the shadow manifest entries.
- `run_lira.py` – compute LiRA scores/metrics from a target checkpoint and a trained shadow manifest.

## Dependencies

Python 3.10+, PyTorch/torchvision (already used in the repo), plus:

```bash
pip install numpy scikit-learn matplotlib
```

## 1) Prepare data splits (CIFAR10-ST + public + eval/calibration)

The scripts will auto-create splits if missing, but you can run explicitly:

```bash
python MIA/data_prep.py  # optional: run_attacks/run_lira will call prepare_splits() implicitly
```

Paths produced (default root `MIA/splits/`):

- `train_st.npz` – target training set (25,500 samples).
- `public.npz` – complement (24,500 samples; classes 0–4 only by construction).
- `eval_members.npz` / `eval_nonmembers.npz` – balanced eval pool (defaults: 10k each, disjoint).
- `calibration_nonmembers.npz` – for threshold tuning (default 5k, disjoint from eval).
- `splits_metadata.json` – counts and config.

Arguments (if run via `run_attacks.py`/`run_lira.py`): `--member-count`, `--non-member-count`, `--calibration-count`, `--seed`, `--split-root`, `--data-root`.

## 2) Target models (DP RS-DRO baseline)

Train with your existing DP RS-DRO implementation (no rework inside MIA):

```bash
python dro2_new/train_rsdro.py \
  --data-root ./data \
  --results-dir runs/rsdro \
  --width-factor 0.5 \
  --save-model
```

This writes `runs/rsdro/rsdro_resnet20.pt` (state_dict under the `model` key) and `runs/rsdro/summary.json`. If you trained with a different `width-factor`, pass the same value to the MIA scripts.

## 3) Run simple attacks (confidence / loss)

```bash
bash MIA/run_mia_rsdro.sh runs/rsdro/rsdro_resnet20.pt 0.5
```

Or call the Python entry directly (use the same `width-factor` you trained with):

```bash
python MIA/run_attacks.py \
  --checkpoint runs/rsdro/rsdro_resnet20.pt \
  --width-factor 0.5 \
  --output-dir MIA/outputs/rsdro \
  --member-count 10000 \
  --calibration-count 5000 \
  --per-class-thresholds
```

What it does:

- Ensures splits exist (creates under `MIA/splits/` if missing).
- Runs the target checkpoint on eval + calibration splits.
- Calibrates thresholds to hit FPR = 1% / 0.1% on non-member calibration scores.
- Saves `attack_metrics.json` plus optional ROC PNGs (`roc_confidence.png`, `roc_loss.png`) in `--output-dir`.

Metrics reported: ROC-AUC, TPR@1%FPR, TPR@0.1%FPR, Precision@1%/0.1%FPR, thresholds; optional per-class thresholds (only classes present in calibration, i.e., 0–4 given the public complement definition).

## 4) LiRA (shadow models, optional strong baseline)

1) **Sample shadow training indices** (from `public.npz` by default):

```bash
python MIA/make_shadow_manifest.py \
  --public-npz MIA/splits/public.npz \
  --output MIA/shadows/manifest.json \
  --num-shadows 16 \
  --train-fraction 0.5 \
  --mode proportional \
  --checkpoint-template MIA/shadows/shadow_{i:03d}/checkpoint.pt
```

This writes `shadow_XXX_indices.npy` files plus a manifest with checkpoint paths to fill.

2) **Train shadows** (non-DP ResNet-20, standard CIFAR10 aug):

```bash
python MIA/train_shadows.py --manifest MIA/shadows/manifest.json --epochs 160 --batch-size 128
```

3) **Run LiRA scoring**:

```bash
python MIA/run_lira.py \
  --target-checkpoint runs/rsdro/rsdro_resnet20.pt \
  --shadow-manifest MIA/shadows/manifest.json \
  --output-dir MIA/outputs/lira_dp
```

Outputs: `lira_metrics.json` (same metrics set as above) and `lira_scores.npz` (eval/calib LiRA scores for further plotting). Per-class thresholds are calibrated on LiRA scores (limited to classes present in calibration data).

Notes:

- The default shadow pool (`public.npz`) only covers classes 0–4 because the CIFAR10-ST complement has no class-5–9 samples. If you want shadows with full class coverage, build a custom manifest using different pools (e.g., extra held-out CIFAR-10 data) and point `make_shadow_manifest.py --public-npz` to that source.
- Checkpoints listed in the manifest must exist before `run_lira.py` runs.

## 5) Plots/tables suggestions

- ROC curves: `MIA/outputs/*/roc_*.png` (or create with saved scores).
- Tables: parse `attack_metrics.json` / `lira_metrics.json` into AUC, TPR@1%/0.1%, Precision@FPR for each target (Non-DP, DP-A/B/C).
- Accuracy vs. ε: rely on training logs from your DP trainer; align with Section 4.1 low-FPR reporting.

## 6) Quick checklist

- [ ] Train/collect Non-DP + DP (A/B/C) checkpoints.
- [ ] Run `run_attacks.py` for each checkpoint.
- [ ] (Optional) Generate shadow manifest → train shadows → `run_lira.py`.
- [ ] Aggregate `attack_metrics.json`/`lira_metrics.json` into tables/plots (TPR@1%/0.1%FPR, AUC, precision).

Everything stays ASCII; feel free to tweak hyper-parameters/epochs to match your final paper setup.
