from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class EvalBatch:
    images: torch.Tensor
    targets: torch.Tensor
    is_member: torch.Tensor
    indices: torch.Tensor


class NPZDataset(Dataset):
    """Generic loader for NPZ archives with `data`, `targets`, and optional `indices`."""

    def __init__(self, path: Path, transform: Optional[transforms.Compose] = None) -> None:
        payload = np.load(path)
        self.images = payload["data"]
        self.targets = payload["targets"]
        self.indices = payload["indices"] if "indices" in payload else np.arange(len(self.targets))
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:  # type: ignore[override]
        image = Image.fromarray(self.images[idx])
        image = self.transform(image) if self.transform is not None else transforms.ToTensor()(image)
        target = int(self.targets[idx])
        index = int(self.indices[idx])
        return image, target, index


class EvalDataset(Dataset):
    """Wraps member/non-member NPZ archives into a single dataset with membership labels."""

    def __init__(
        self,
        members_path: Path,
        nonmembers_path: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        members = NPZDataset(members_path, transform=transform)
        nonmembers = NPZDataset(nonmembers_path, transform=transform)
        self.images = np.concatenate([members.images, nonmembers.images])
        self.targets = np.concatenate([members.targets, nonmembers.targets])
        self.indices = np.concatenate([members.indices, nonmembers.indices])
        self.is_member = np.concatenate(
            [np.ones_like(members.targets, dtype=np.int64), np.zeros_like(nonmembers.targets, dtype=np.int64)]
        )
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:  # type: ignore[override]
        image = Image.fromarray(self.images[idx])
        image = self.transform(image) if self.transform is not None else transforms.ToTensor()(image)
        return image, int(self.targets[idx]), int(self.is_member[idx]), int(self.indices[idx])


def build_loader(dataset: Dataset, batch_size: int, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def default_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
