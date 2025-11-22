from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MIA.attacks import (
    FPR_TARGETS,
    calibrate_thresholds,
    calibrate_thresholds_per_class,
    evaluate_attack,
    lira_from_shadow_bags,
    logit_margin,
)
from MIA.data_prep import SplitConfig, SplitPaths, prepare_splits
from MIA.datasets import EvalDataset, NPZDataset, build_loader, default_test_transform
from MIA.model_utils import collect_outputs, get_device, load_resnet20


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError("Shadow manifest must be a list of entries.")
    for entry in payload:
        if "checkpoint" not in entry or "train_indices_path" not in entry:
            raise ValueError("Each shadow entry needs `checkpoint` and `train_indices_path`.")
    return payload


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


def _membership_matrix(example_indices: np.ndarray, shadow_indices: List[np.ndarray]) -> np.ndarray:
    membership = np.zeros((example_indices.shape[0], len(shadow_indices)), dtype=np.int64)
    for j, idxs in enumerate(shadow_indices):
        if idxs.size == 0:
            continue
        membership[:, j] = np.isin(example_indices, idxs).astype(np.int64)
    return membership


def run_lira(args) -> None:
    device = get_device()
    paths = _maybe_prepare_splits(args)
    manifest = _load_manifest(Path(args.shadow_manifest))
    transform = default_test_transform()

    eval_dataset = EvalDataset(paths.eval_members, paths.eval_nonmembers, transform=transform)
    calib_dataset = NPZDataset(paths.calibration_nonmembers, transform=transform)

    eval_loader = build_loader(eval_dataset, batch_size=args.batch_size, num_workers=args.workers)
    calib_loader = build_loader(calib_dataset, batch_size=args.batch_size, num_workers=args.workers)

    target_model = load_resnet20(Path(args.target_checkpoint), device=device, width_factor=args.width_factor)
    target_eval = collect_outputs(target_model, eval_loader, device)
    target_calib = collect_outputs(target_model, calib_loader, device)

    target_eval_scores = logit_margin(
        torch.from_numpy(target_eval["logits"]),
        torch.from_numpy(target_eval["targets"]),
    ).numpy()
    target_calib_scores = logit_margin(
        torch.from_numpy(target_calib["logits"]),
        torch.from_numpy(target_calib["targets"]),
    ).numpy()
    labels_eval = target_eval["members"]
    class_labels = target_eval["targets"]

    shadow_indices: List[np.ndarray] = []
    for entry in manifest:
        idxs = np.load(Path(entry["train_indices_path"]))
        shadow_indices.append(np.asarray(idxs, dtype=np.int64))
        ckpt_path = Path(entry["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing shadow checkpoint: {ckpt_path}")

    shadow_members_eval = _membership_matrix(target_eval["indices"], shadow_indices)
    shadow_members_calib = _membership_matrix(target_calib["indices"], shadow_indices)

    num_eval = target_eval["logits"].shape[0]
    num_calib = target_calib["logits"].shape[0]
    num_shadows = len(manifest)
    shadow_scores_eval = np.zeros((num_eval, num_shadows), dtype=np.float32)
    shadow_scores_calib = np.zeros((num_calib, num_shadows), dtype=np.float32)

    for j, entry in enumerate(manifest):
        print(f"[shadow] running shadow {j+1}/{num_shadows} from {entry['checkpoint']}")
        model = load_resnet20(Path(entry["checkpoint"]), device=device, width_factor=args.width_factor)
        out_eval = collect_outputs(model, eval_loader, device)
        out_calib = collect_outputs(model, calib_loader, device)
        shadow_scores_eval[:, j] = logit_margin(
            torch.from_numpy(out_eval["logits"]), torch.from_numpy(out_eval["targets"])
        ).numpy()
        shadow_scores_calib[:, j] = logit_margin(
            torch.from_numpy(out_calib["logits"]), torch.from_numpy(out_calib["targets"])
        ).numpy()

    lira_eval = lira_from_shadow_bags(target_eval_scores, shadow_scores_eval, shadow_members_eval)
    lira_calib = lira_from_shadow_bags(target_calib_scores, shadow_scores_calib, shadow_members_calib)

    thresholds = calibrate_thresholds(lira_calib, FPR_TARGETS)
    per_class_thr = calibrate_thresholds_per_class(lira_calib, target_calib["targets"], FPR_TARGETS)
    metrics = evaluate_attack(
        scores=lira_eval,
        labels=labels_eval,
        thresholds=thresholds,
        class_labels=class_labels,
        per_class_thresholds=per_class_thr,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "lira_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics.to_json(), fh, indent=2)
    np.savez_compressed(out_dir / "lira_scores.npz", eval_scores=lira_eval, calib_scores=lira_calib)
    print(f"[lira] saved metrics to {out_dir / 'lira_metrics.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LiRA scores using shadow models.")
    parser.add_argument("--target-checkpoint", type=str, required=True, help="Target model checkpoint.")
    parser.add_argument(
        "--shadow-manifest",
        type=str,
        required=True,
        help="JSON list with entries: {checkpoint, train_indices_path}.",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split-root", type=str, default="MIA/splits")
    parser.add_argument("--output-dir", type=str, default="MIA/outputs/lira")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--width-factor", type=float, default=0.5, help="Width factor used by RS-DRO ResNet20.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--member-count", type=int, default=10_000)
    parser.add_argument("--non-member-count", type=int, default=None)
    parser.add_argument("--calibration-count", type=int, default=5_000)
    parser.add_argument("--recreate-splits", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_lira(args)
