"""Risk model utilities for DP Double-Spider."""
from __future__ import annotations

import torch


class RiskModel:
    """KL-based risk model used for distributionally robust objectives."""

    def __init__(self, rho: float = 0.5) -> None:
        self.rho = rho

    def grad_f(self, s: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return lam / (s + 1e-12)

    def f(self, s: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        return lam * torch.log(s + 1e-12) + lam * self.rho
