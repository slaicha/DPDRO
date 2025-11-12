"""Dataset helpers for CIFAR10-ST."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def _select_cifar10_st_indices(targets: np.ndarray) -> np.ndarray:
    indices = []
    for class_id in range(10):
        class_indices = np.where(targets == class_id)[0]
        if class_id < 5:
            indices.extend(class_indices[-100:])
        else:
            indices.extend(class_indices)
    return np.array(indices, dtype=np.int64)


def build_cifar10_st_dataset(
    data_root: str = "./data", download: bool = True, augment: bool = True
) -> Tuple[Subset, int]:
    if augment:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

    base_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download,
        transform=transform,
    )

    targets = np.array(base_dataset.targets)
    keep_indices = _select_cifar10_st_indices(targets)
    subset = Subset(base_dataset, keep_indices.tolist())
    return subset, len(keep_indices)


def build_cifar10_st_loaders(
    dataset: Subset,
    batch_size_b1: int,
    batch_size_b2: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    loader_b1 = DataLoader(
        dataset,
        batch_size=batch_size_b1,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    loader_b2 = DataLoader(
        dataset,
        batch_size=batch_size_b2,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return loader_b1, loader_b2


def build_full_dataset_loader(
    dataset: Subset,
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )


def build_cifar10_test_loader(
    data_root: str = "./data", batch_size: int = 256, num_workers: int = 4
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )


__all__ = [
    "build_cifar10_st_dataset",
    "build_cifar10_st_loaders",
    "build_full_dataset_loader",
    "build_cifar10_test_loader",
]
