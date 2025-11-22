from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from torchvision.datasets import CIFAR10


@dataclass
class SplitConfig:
    """Configuration for CIFAR10-ST splits and MIA evaluation sets."""

    data_root: Path
    output_root: Path
    seed: int = 0
    member_count: int = 10_000
    non_member_count: int | None = None
    calibration_count: int = 5_000


@dataclass
class SplitPaths:
    train_st: Path
    public: Path
    eval_members: Path
    eval_nonmembers: Path
    calibration_nonmembers: Path
    metadata: Path


def _class_indices(labels: Sequence[int]) -> Dict[int, np.ndarray]:
    label_arr = np.asarray(labels)
    return {cls: np.flatnonzero(label_arr == cls) for cls in range(10)}


def _build_cifar10_st_indices(labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return train/public indices for CIFAR10-ST along with per-class maps."""
    by_class = _class_indices(labels)
    train_parts: List[np.ndarray] = []
    public_parts: List[np.ndarray] = []
    for cls, idxs in by_class.items():
        if cls < 5:
            keep = idxs[-100:]
            drop = idxs[:-100]
        else:
            keep = idxs
            drop = np.array([], dtype=int)
        train_parts.append(keep)
        public_parts.append(drop)
    train_idx = np.concatenate(train_parts)
    public_idx = np.concatenate(public_parts)
    train_by_class = {cls: (idxs[-100:] if cls < 5 else idxs) for cls, idxs in by_class.items()}
    public_by_class = {cls: (idxs[:-100] if cls < 5 else np.array([], dtype=int)) for cls, idxs in by_class.items()}
    return train_idx, public_idx, train_by_class, public_by_class


def _sample(pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or len(pool) == 0:
        return np.empty((0,), dtype=int)
    k = min(k, len(pool))
    return rng.choice(pool, size=k, replace=False)


def _save_npz(path: Path, images: np.ndarray, targets: np.ndarray, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=images, targets=targets, indices=indices)


def prepare_splits(cfg: SplitConfig) -> SplitPaths:
    """Materialise CIFAR10-ST, public set, eval sets, and calibration set."""
    cfg.non_member_count = cfg.member_count if cfg.non_member_count is None else cfg.non_member_count
    rng = np.random.default_rng(cfg.seed)
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    train_set = CIFAR10(root=cfg.data_root, train=True, download=True)
    _ = CIFAR10(root=cfg.data_root, train=False, download=True)  # ensure test is cached for downstream use
    labels = np.asarray(train_set.targets)
    images = np.asarray(train_set.data)

    train_idx, public_idx, train_by_class, public_by_class = _build_cifar10_st_indices(labels)

    members_eval = _sample(train_idx, cfg.member_count, rng)
    nonmembers_eval = _sample(public_idx, cfg.non_member_count, rng)

    remaining_public = np.setdiff1d(public_idx, nonmembers_eval, assume_unique=False)
    calibration = _sample(remaining_public, cfg.calibration_count, rng)

    paths = SplitPaths(
        train_st=cfg.output_root / "train_st.npz",
        public=cfg.output_root / "public.npz",
        eval_members=cfg.output_root / "eval_members.npz",
        eval_nonmembers=cfg.output_root / "eval_nonmembers.npz",
        calibration_nonmembers=cfg.output_root / "calibration_nonmembers.npz",
        metadata=cfg.output_root / "splits_metadata.json",
    )

    _save_npz(paths.train_st, images[train_idx], labels[train_idx], train_idx)
    _save_npz(paths.public, images[public_idx], labels[public_idx], public_idx)
    _save_npz(paths.eval_members, images[members_eval], labels[members_eval], members_eval)
    _save_npz(paths.eval_nonmembers, images[nonmembers_eval], labels[nonmembers_eval], nonmembers_eval)
    _save_npz(paths.calibration_nonmembers, images[calibration], labels[calibration], calibration)

    metadata = {
        "config": {
            "data_root": str(cfg.data_root),
            "output_root": str(cfg.output_root),
            "seed": cfg.seed,
            "member_count": cfg.member_count,
            "non_member_count": cfg.non_member_count,
            "calibration_count": cfg.calibration_count,
        },
        "counts": {
            "train_st": int(len(train_idx)),
            "public": int(len(public_idx)),
            "eval_members": int(len(members_eval)),
            "eval_nonmembers": int(len(nonmembers_eval)),
            "calibration_nonmembers": int(len(calibration)),
        },
        "class_histogram_train": {str(cls): int(len(train_by_class[cls])) for cls in range(10)},
        "class_histogram_public": {str(cls): int(len(public_by_class[cls])) for cls in range(10)},
        "note": "CIFAR10-ST keeps last 100 samples of classes 0-4 and all 5000 samples of classes 5-9.",
    }
    with paths.metadata.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return paths


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialise CIFAR10-ST splits for MIA.")
    parser.add_argument("--data-root", type=str, default="data", help="Root path containing CIFAR-10.")
    parser.add_argument("--output-root", type=str, default="MIA/splits", help="Directory to write NPZ splits.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--member-count", type=int, default=10_000)
    parser.add_argument("--non-member-count", type=int, default=None)
    parser.add_argument("--calibration-count", type=int, default=5_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = SplitConfig(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        seed=args.seed,
        member_count=args.member_count,
        non_member_count=args.non_member_count,
        calibration_count=args.calibration_count,
    )
    paths = prepare_splits(cfg)
    print(f"[data] splits saved to {cfg.output_root} (metadata: {paths.metadata})")
