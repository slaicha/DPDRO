"""Core Recursive Spider DRO algorithm."""
from __future__ import annotations

import random
from copy import deepcopy
from typing import Tuple

import torch
from torch.utils.data import Dataset

from .hyperparams import RSDROHyperParams
from .objectives import f_grad_lambda, g_value_and_grad, g_value_only
from .utils import (
    add_gaussian_noise_to_scalar,
    add_gaussian_noise_to_tensors,
    ensure_positive,
    model_distance,
    safe_log,
)


def run_rs_dro(
    model: torch.nn.Module,
    lambda_param: torch.Tensor,
    train_dataset: Dataset,
    loader_b1,
    loader_b2,
    hyperparams: RSDROHyperParams,
    rho: float,
    lambda_lower_bound: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tuple[torch.nn.Module, torch.Tensor, int]:
    model = model.to(device)
    model.train()

    lambda_t = lambda_param.detach().clone().to(device)
    lambda_t.requires_grad_(False)

    w_prev_model = deepcopy(model).to(device)
    w_prev_lambda = lambda_t.clone()

    v_t = [torch.zeros_like(p) for p in model.parameters()]
    u_t = torch.zeros(1, device=device)
    s_t = torch.ones(1, device=device)
    s_prev = s_t.clone()

    loader_b1_iter = iter(loader_b1)
    loader_b2_iter = iter(loader_b2)

    dataset_len = len(train_dataset)

    selected_state = None
    selected_iter = -1

    for t in range(1, hyperparams.T + 1):
        w_prev_model.load_state_dict(model.state_dict())
        w_prev_lambda = lambda_t.clone()
        s_prev = s_t.clone()

        idx = random.randrange(dataset_len)
        xi_data, xi_target = train_dataset[idx]
        if not torch.is_tensor(xi_data):
            xi_data = torch.tensor(xi_data)
        xi_data = xi_data.unsqueeze(0).to(device)
        if torch.is_tensor(xi_target):
            xi_target = xi_target.view(1).long().to(device)
        else:
            xi_target = torch.tensor([xi_target], device=device, dtype=torch.long)

        g_prev = g_value_only(w_prev_model, w_prev_lambda, xi_data, xi_target)

        full_refresh = (t % hyperparams.q == 0) or (t == 1)

        if full_refresh:
            try:
                data_b1, target_b1 = next(loader_b1_iter)
            except StopIteration:
                loader_b1_iter = iter(loader_b1)
                data_b1, target_b1 = next(loader_b1_iter)
            data_b1, target_b1 = data_b1.to(device), target_b1.to(device)

            g_val, grad_params, grad_lambda = g_value_and_grad(model, lambda_t, data_b1, target_b1)
            v_t = [grad.clone() for grad in grad_params]
            u_t = grad_lambda.clone()
            v_t = add_gaussian_noise_to_tensors(v_t, hyperparams.sigma1, generator)
            u_t = add_gaussian_noise_to_scalar(u_t, hyperparams.sigma1, generator)
        else:
            try:
                data_b2, target_b2 = next(loader_b2_iter)
            except StopIteration:
                loader_b2_iter = iter(loader_b2)
                data_b2, target_b2 = next(loader_b2_iter)
            data_b2, target_b2 = data_b2.to(device), target_b2.to(device)

            _, grad_x_t, grad_lambda_t = g_value_and_grad(model, lambda_t, data_b2, target_b2)
            _, grad_x_prev, grad_lambda_prev = g_value_and_grad(w_prev_model, w_prev_lambda, data_b2, target_b2)

            v_t = [g_t - g_prev for g_t, g_prev in zip(grad_x_t, grad_x_prev)]
            u_t = grad_lambda_t - grad_lambda_prev

            distance = model_distance(model, lambda_t, w_prev_model, w_prev_lambda)
            sigma_term = hyperparams.sigma2 * distance.item()
            kappa_std = min(sigma_term, hyperparams.hat_sigma2)
            v_t = add_gaussian_noise_to_tensors(v_t, kappa_std, generator)
            u_t = add_gaussian_noise_to_scalar(u_t, kappa_std, generator)

        g_curr = g_value_only(model, lambda_t, xi_data, xi_target)
        innovation = s_prev - g_prev
        s_t = g_curr + (1.0 - hyperparams.beta_t) * innovation
        s_t = add_gaussian_noise_to_scalar(s_t, hyperparams.sigma_st, generator)
        s_t = ensure_positive(s_t)

        grad_f = f_grad_lambda(lambda_t, s_t)
        z_t_params = [grad_f * tens for tens in v_t]
        z_t_lambda = grad_f * u_t + safe_log(s_t) + rho

        with torch.no_grad():
            for param, z in zip(model.parameters(), z_t_params):
                param.add_(z, alpha=-hyperparams.eta)
            lambda_t = lambda_t - hyperparams.eta * z_t_lambda
            lambda_t = torch.clamp(lambda_t, min=lambda_lower_bound)

        if random.randint(1, t) == 1:
            selected_state = (
                {k: v.clone() for k, v in model.state_dict().items()},
                lambda_t.clone(),
            )
            selected_iter = t

    if selected_state is None:
        selected_state = ({k: v.clone() for k, v in model.state_dict().items()}, lambda_t.clone())
        selected_iter = hyperparams.T

    model.load_state_dict(selected_state[0])
    lambda_t = selected_state[1]
    return model, lambda_t, selected_iter


__all__ = ["run_rs_dro"]
