"""Utility helpers for DP Double-Spider training."""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Track running averages for scalar metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> List[torch.Tensor]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def clone_model(model: nn.Module) -> nn.Module:
    import copy

    replica = copy.deepcopy(model)
    for p in replica.parameters():
        p.requires_grad = True
    return replica


def model_parameter_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def l2_distance_model(model_a: nn.Module, model_b: nn.Module) -> float:
    with torch.no_grad():
        vec_a = model_parameter_vector(model_a)
        vec_b = model_parameter_vector(model_b)
        return torch.norm(vec_a - vec_b, p=2).item()


def apply_update(model: nn.Module, grads: Iterable[torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            param.add_(-lr * grad)


def tensor_list_norm(grads: Iterable[torch.Tensor]) -> float:
    return math.sqrt(sum(g.detach().pow(2).sum().item() for g in grads))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class BatchStreamConfig:
    dataset: Dataset
    batch_size: int
    device: torch.device
    num_workers: int = 0
    pin_memory: bool = True


class BatchStream:
    """Endless randomised stream of mini-batches."""

    def __init__(self, cfg: BatchStreamConfig) -> None:
        self.cfg = cfg
        self.loader = DataLoader(
            cfg.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        )
        self.iterator = iter(self.loader)

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            images, targets = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            images, targets = next(self.iterator)
        return (
            images.to(self.cfg.device, non_blocking=True),
            targets.to(self.cfg.device, non_blocking=True),
        )
