"""Hyperparameter calculations for RS-DRO."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RSDROHyperParams:
    T: int
    q: int
    b1: int
    b2: int
    eta: float
    beta_t: float
    sigma1: float
    sigma2: float
    hat_sigma2: float
    sigma_st: float


@dataclass
class SampleSizeCheck:
    required_n: float
    required_n_alt: float
    satisfied: bool


def check_sample_size_condition(
    n: int,
    epsilon: float,
    delta: float,
    G: float,
    L: float,
    Psi_0: float,
    d: int,
) -> SampleSizeCheck:
    log_inv_delta = math.log(1.0 / delta)
    requirement_1 = (G * epsilon) ** 2 / (Psi_0 * L * d * log_inv_delta)
    requirement_2 = (math.sqrt(d) * max(1.0, math.sqrt(L * Psi_0 / G))) / epsilon
    return SampleSizeCheck(
        required_n=requirement_1,
        required_n_alt=requirement_2,
        satisfied=n >= max(requirement_1, requirement_2),
    )


def compute_hyperparams(
    n: int,
    d: int,
    epsilon: float,
    delta: float,
    G: float,
    L: float,
    c_const: float,
    Psi_0: float,
    eta_t_squared: float,
) -> RSDROHyperParams:
    if L <= 0:
        raise ValueError("L must be positive to compute RS-DRO hyperparameters.")
    if Psi_0 <= 0:
        raise ValueError("Psi_0 must be positive.")
    if eta_t_squared <= 0:
        raise ValueError("eta_t_squared must be positive.")
    log_inv_delta = math.log(1.0 / delta)

    T_term1 = (((Psi_0 * L) ** 0.25 * n * epsilon) / math.sqrt(G * d * log_inv_delta)) ** (4.0 / 3.0)
    T_term2 = (n * epsilon) / math.sqrt(d * log_inv_delta)
    T = max(1, int(math.floor(max(T_term1, T_term2))))

    b2_term1 = ((G * n * epsilon) / math.sqrt(Psi_0 * L * d * log_inv_delta)) ** (2.0 / 3.0)
    b2_term2 = ((G * n * d * log_inv_delta) ** (1.0 / 3.0)) / (
        (L * Psi_0) ** (1.0 / 6.0) * (epsilon ** (2.0 / 3.0))
    )
    b2 = max(1, int(math.floor(max(b2_term1, b2_term2))))

    if L == 0 or T == 0:
        q = 1
    else:
        numerator = (n ** 2) * (epsilon ** 2)
        denominator = (L ** 2) * T * d * log_inv_delta
        q = max(1, int(math.floor(numerator / denominator)))

    b1 = n

    eta_candidate = c_const * eta_t_squared
    eta = min(1.0 / (2.0 * L), eta_candidate)
    beta_t = min(0.999, c_const * eta_t_squared)

    max_term = max(1.0 / b2, math.sqrt(T) / n)
    sigma1 = (c_const * G * math.sqrt(T * log_inv_delta)) / (n * q * epsilon)
    sigma2 = (c_const * L * math.sqrt(log_inv_delta) / epsilon) * max_term
    hat_sigma2 = (2.0 * c_const * G * math.sqrt(log_inv_delta) / epsilon) * max_term
    sigma_st = (c_const * G * math.sqrt(T * log_inv_delta)) / (n * epsilon)

    return RSDROHyperParams(
        T=T,
        q=q,
        b1=b1,
        b2=b2,
        eta=eta,
        beta_t=beta_t,
        sigma1=sigma1,
        sigma2=sigma2,
        hat_sigma2=hat_sigma2,
        sigma_st=sigma_st,
    )


__all__ = ["RSDROHyperParams", "SampleSizeCheck", "check_sample_size_condition", "compute_hyperparams"]
