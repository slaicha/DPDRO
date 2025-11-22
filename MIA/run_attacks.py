from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MIA.attacks import (  # noqa: E402
    FPR_TARGETS,
    AttackMetrics,
    calibrate_thresholds,
    calibrate_thresholds_per_class,
    confidence_scores,
    evaluate_attack,
    loss_scores,
    logit_margin,
)
from MIA.data_prep import SplitConfig, SplitPaths, prepare_splits  # noqa: E402
from MIA.datasets import EvalDataset, NPZDataset, build_loader, default_test_transform  # noqa: E402
from MIA.model_utils import collect_outputs, get_device, load_resnet20  # noqa: E402


def _splits_ready(paths: SplitPaths) -> bool:
    return all(
        path.exists()
        for path in [
            paths.train_st,
            paths.public,
            paths.eval_members,
            paths.eval_nonmembers,
            paths.calibration_nonmembers,
            paths.metadata,
        ]
    )


def _maybe_prepare_splits(args) -> SplitPaths:
    cfg = SplitConfig(
        data_root=Path(args.data_root),
        output_root=Path(args.split_root),
        seed=args.seed,
        member_count=args.member_count,
        non_member_count=args.non_member_count,
        calibration_count=args.calibration_count,
    )
    paths = SplitPaths(
        train_st=cfg.output_root / "train_st.npz",
        public=cfg.output_root / "public.npz",
        eval_members=cfg.output_root / "eval_members.npz",
        eval_nonmembers=cfg.output_root / "eval_nonmembers.npz",
        calibration_nonmembers=cfg.output_root / "calibration_nonmembers.npz",
        metadata=cfg.output_root / "splits_metadata.json",
    )
    if args.recreate_splits or not _splits_ready(paths):
        print(f"[data] preparing splits at {cfg.output_root} (seed={cfg.seed})")
        paths = prepare_splits(cfg)
    else:
        print(f"[data] using cached splits at {paths.metadata}")
    return paths


def _save_metrics(payload: Dict[str, AttackMetrics], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {name: metrics.to_json() for name, metrics in payload.items()}
    with (out_dir / "attack_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    print(f"[eval] metrics saved to {out_dir / 'attack_metrics.json'}")


def _plot_roc(scores: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; skipping ROC plot")
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=title)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlim([0.0, 0.1])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def run_simple_attacks(args) -> None:
    device = get_device()
    paths = _maybe_prepare_splits(args)
    transform = default_test_transform()

    eval_dataset = EvalDataset(paths.eval_members, paths.eval_nonmembers, transform=transform)
    calib_dataset = NPZDataset(paths.calibration_nonmembers, transform=transform)

    eval_loader = build_loader(eval_dataset, batch_size=args.batch_size, num_workers=args.workers)
    calib_loader = build_loader(calib_dataset, batch_size=args.batch_size, num_workers=args.workers)

    model = load_resnet20(Path(args.checkpoint), device=device, width_factor=args.width_factor)
    print(f"[model] loaded checkpoint from {args.checkpoint} on {device} (width_factor={args.width_factor})")

    eval_outputs = collect_outputs(model, eval_loader, device)
    calib_outputs = collect_outputs(model, calib_loader, device)

    logits_eval = torch.from_numpy(eval_outputs["logits"])
    logits_calib = torch.from_numpy(calib_outputs["logits"])
    targets_eval = torch.from_numpy(eval_outputs["targets"])
    targets_calib = torch.from_numpy(calib_outputs["targets"])

    labels_eval = eval_outputs["members"]
    class_labels = eval_outputs["targets"]

    results: Dict[str, AttackMetrics] = {}

    # Confidence attack
    conf_eval = confidence_scores(logits_eval).numpy()
    conf_calib = confidence_scores(logits_calib).numpy()
    conf_thresholds = calibrate_thresholds(conf_calib, FPR_TARGETS)
    per_class_thr = None
    if args.per_class_thresholds:
        per_class_thr = calibrate_thresholds_per_class(conf_calib, targets_calib.numpy(), FPR_TARGETS)
    metrics_conf = evaluate_attack(
        scores=conf_eval,
        labels=labels_eval,
        thresholds=conf_thresholds,
        class_labels=class_labels,
        per_class_thresholds=per_class_thr,
    )
    results["confidence"] = metrics_conf
    _plot_roc(conf_eval, labels_eval, Path(args.output_dir) / "roc_confidence.png", "Confidence attack")

    # Loss attack
    loss_eval = loss_scores(logits_eval, targets_eval).numpy()
    loss_calib = loss_scores(logits_calib, targets_calib).numpy()
    loss_thresholds = calibrate_thresholds(loss_calib, FPR_TARGETS)
    per_class_thr_loss = None
    if args.per_class_thresholds:
        per_class_thr_loss = calibrate_thresholds_per_class(loss_calib, targets_calib.numpy(), FPR_TARGETS)
    metrics_loss = evaluate_attack(
        scores=loss_eval,
        labels=labels_eval,
        thresholds=loss_thresholds,
        class_labels=class_labels,
        per_class_thresholds=per_class_thr_loss,
    )
    results["loss"] = metrics_loss
    _plot_roc(loss_eval, labels_eval, Path(args.output_dir) / "roc_loss.png", "Loss attack")

    _save_metrics(results, Path(args.output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run empirical MIA (confidence / loss) on CIFAR10-ST targets.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (DP or non-DP).")
    parser.add_argument("--data-root", type=str, default="data", help="Root containing CIFAR-10 files.")
    parser.add_argument("--split-root", type=str, default="MIA/splits", help="Where to cache split NPZ files.")
    parser.add_argument("--output-dir", type=str, default="MIA/outputs", help="Directory to store attack outputs.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width-factor", type=float, default=0.5, help="Width factor used by RS-DRO ResNet20.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--member-count", type=int, default=10_000)
    parser.add_argument("--non-member-count", type=int, default=None)
    parser.add_argument("--calibration-count", type=int, default=5_000)
    parser.add_argument("--recreate-splits", action="store_true", help="Force regeneration of split NPZ files.")
    parser.add_argument(
        "--per-class-thresholds",
        action="store_true",
        help="Calibrate thresholds per class in addition to the global threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simple_attacks(args)
