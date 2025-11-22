from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def sample_indices(targets: np.ndarray, indices: np.ndarray, rng: np.random.Generator, fraction: float, mode: str) -> np.ndarray:
    unique_classes = np.unique(targets)
    if mode == "balanced":
        per_class = int(len(targets) * fraction / len(unique_classes))
        selected: List[int] = []
        for cls in unique_classes:
            cls_pool = indices[targets == cls]
            take = min(per_class, len(cls_pool))
            if take > 0:
                selected.extend(rng.choice(cls_pool, size=take, replace=False).tolist())
        return np.asarray(selected, dtype=np.int64)
    # proportional sampling
    total = int(len(indices) * fraction)
    total = min(total, len(indices))
    return rng.choice(indices, size=total, replace=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a LiRA shadow manifest with sampled training indices.")
    parser.add_argument("--public-npz", type=str, default="MIA/splits/public.npz")
    parser.add_argument("--output", type=str, default="MIA/shadows/manifest.json")
    parser.add_argument("--num-shadows", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--mode", type=str, default="proportional", choices=["proportional", "balanced"])
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default="MIA/shadows/shadow_{i:03d}/checkpoint.pt",
        help="Python format string that yields the checkpoint path for each shadow (e.g., shadow_{i}/ckpt.pt).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = np.load(args.public_npz)
    targets = np.asarray(payload["targets"])
    indices = np.asarray(payload["indices"])
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(args.num_shadows):
        rng = np.random.default_rng(args.seed + i)
        chosen = sample_indices(targets, indices, rng, args.train_fraction, args.mode)
        indices_path = out_dir / f"shadow_{i:03d}_indices.npy"
        np.save(indices_path, chosen)
        checkpoint_path = Path(args.checkpoint_template.format(i=i, idx=i, shadow=i))
        entries.append(
            {
                "name": f"shadow_{i:03d}",
                "checkpoint": str(checkpoint_path),
                "train_indices_path": str(indices_path),
                "mode": args.mode,
                "train_fraction": args.train_fraction,
            }
        )
    with Path(args.output).open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
    print(f"[manifest] wrote {len(entries)} shadow entries to {args.output}")


if __name__ == "__main__":
    main()
