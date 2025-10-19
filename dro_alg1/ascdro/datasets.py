"""Dataset helpers for CIFAR10-ST."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10STNPZ(Dataset):
    """Wrap CIFAR10-ST data stored in compressed NumPy archives."""

    def __init__(self, path: Path, transform: transforms.Compose | None = None) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        payload = np.load(path)
        self.images = payload["data"]
        self.targets = payload["targets"]
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        image = Image.fromarray(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        target = int(self.targets[idx])
        return image, target


def build_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
