"""Utility helpers for the RS-DRO implementation."""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable, List, Sequence

import torch


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    """Returns a deep copy of *model* preserving device placement."""
    clone = deepcopy(model)
    for p_clone, p_src in zip(clone.parameters(), model.parameters()):
        p_clone.data = p_clone.data.to(p_src.device)
    return clone


def count_parameters(model: torch.nn.Module) -> int:
    """Counts the number of learnable parameters in *model*."""
    return sum(p.numel() for p in model.parameters())


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    """Concatenates all parameters of *model* into a single 1-D tensor."""
    return torch.cat([p.reshape(-1) for p in model.parameters()])


def model_distance(
    model_a: torch.nn.Module,
    lambda_a: torch.Tensor,
    model_b: torch.nn.Module,
    lambda_b: torch.Tensor,
) -> torch.Tensor:
    """Computes the Euclidean distance between (model, lambda) pairs."""
    squared = (lambda_a - lambda_b).pow(2)
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        squared = squared + torch.sum((p_a - p_b) ** 2)
    return torch.sqrt(torch.clamp(squared, min=1e-12))


def add_gaussian_noise_to_tensors(
    tensors: Sequence[torch.Tensor], std: float, generator: torch.Generator | None = None
) -> List[torch.Tensor]:
    """Adds i.i.d. Gaussian noise with standard deviation *std* to every tensor."""
    if std <= 0:
        return [t.clone() for t in tensors]

    noisy = []
    for tensor in tensors:
        noise = torch.empty_like(tensor)
        noise.normal_(mean=0.0, std=std, generator=generator)
        noisy.append(tensor + noise)
    return noisy


def add_gaussian_noise_to_scalar(
    value: torch.Tensor, std: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Adds scalar Gaussian noise."""
    if std <= 0:
        return value.clone()
    noise = torch.empty(1, device=value.device)
    noise.normal_(mean=0.0, std=std, generator=generator)
    return value + noise.squeeze(0)


def safe_log(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable natural logarithm."""
    return torch.log(torch.clamp(tensor, min=eps))


def ensure_positive(tensor: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
    """Projects tensor to be at least *min_value*."""
    return torch.clamp(tensor, min=min_value)
