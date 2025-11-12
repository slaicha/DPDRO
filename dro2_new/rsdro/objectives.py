"""Objective-related helpers for RS-DRO."""
from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

from .utils import ensure_positive, safe_log


def g_value_and_grad(
    model: torch.nn.Module,
    lambda_param: torch.Tensor,
    data: torch.Tensor,
    target: torch.Tensor,
    loss_clip: float = 6.0,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Computes g(w; xi) and gradients w.r.t. model parameters and lambda."""
    model.zero_grad(set_to_none=True)
    lambda_leaf = lambda_param.detach().clone().requires_grad_(True)

    criterion = nn.CrossEntropyLoss()
    outputs = model(data)
    ce_loss = criterion(outputs, target)

    ratio = torch.clamp(ce_loss / lambda_leaf, min=-loss_clip, max=loss_clip)
    g_value = torch.exp(ratio)

    g_value.backward()

    grad_params = [p.grad.detach().clone() for p in model.parameters()]
    grad_lambda = lambda_leaf.grad.detach().clone()
    return g_value.detach(), grad_params, grad_lambda


def g_value_only(
    model: torch.nn.Module,
    lambda_param: torch.Tensor,
    data: torch.Tensor,
    target: torch.Tensor,
    loss_clip: float = 6.0,
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs = model(data)
        ce_loss = criterion(outputs, target)
        ratio = torch.clamp(ce_loss / lambda_param, min=-loss_clip, max=loss_clip)
        return torch.exp(ratio)


def f_grad_lambda(lambda_param: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
    return lambda_param / (s_t + 1e-12)


def compute_psi(
    model: torch.nn.Module,
    lambda_param: torch.Tensor,
    dataloader: Iterable,
    rho: float,
    lambda_0: float,
    loss_clip: float = 6.0,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction="none")

    exp_losses = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            losses = criterion(outputs, target)
            ratio = torch.clamp(losses / lambda_param, min=-loss_clip, max=loss_clip)
            exp_losses.append(torch.exp(ratio))

    g2 = torch.mean(torch.cat(exp_losses))
    psi = lambda_param * safe_log(g2) + (lambda_param - lambda_0) * rho
    model.train()
    return psi


def estimate_psi0(
    model: torch.nn.Module,
    lambda_param: torch.Tensor,
    dataloader: Iterable,
    rho: float,
    lambda_0: float,
    warmup_steps: int = 200,
    warmup_lr: float = 1e-3,
    device: torch.device | None = None,
) -> float:
    if device is None:
        device = next(model.parameters()).device

    psi_at_zero = compute_psi(model, lambda_param, dataloader, rho, lambda_0)

    if warmup_steps <= 0:
        return max(psi_at_zero.item(), 1e-8)

    aux_model = deepcopy(model)
    aux_model.to(device)
    optimizer = optim.SGD(aux_model.parameters(), lr=warmup_lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    iterator = itertools.cycle(dataloader)
    for _ in range(warmup_steps):
        data, target = next(iterator)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = aux_model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

    psi_after_warmup = compute_psi(aux_model, lambda_param, dataloader, rho, lambda_0)
    psi_min_est = torch.min(psi_at_zero, psi_after_warmup)

    psi0 = torch.clamp(psi_at_zero - psi_min_est, min=1e-8)
    return psi0.item()


__all__ = ["g_value_and_grad", "g_value_only", "f_grad_lambda", "compute_psi", "estimate_psi0"]
