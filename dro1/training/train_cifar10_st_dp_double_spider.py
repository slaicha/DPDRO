"""Train ResNet20 on CIFAR10-ST with the DP Double-Spider algorithm."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dro_alg1.ascdro import (
    DPDoubleSpiderConfig,
    DPDoubleSpiderTrainer,
    CIFAR10STNPZ,
    RiskModel,
    build_resnet_cifar,
    build_transforms,
    count_parameters,
    get_device,
    set_seed,
)


LOGGER = logging.getLogger("train_cifar10_st_dp_double_spider")


def prepare_dataloaders(
    data_root: Path,
    estimation_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    train_dataset = CIFAR10STNPZ(data_root / "train.npz", transform=build_transforms(train=True))
    test_dataset = CIFAR10STNPZ(data_root / "test.npz", transform=build_transforms(train=False))
    train_loader = DataLoader(
        train_dataset,
        batch_size=estimation_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        train_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return {
        "train": train_loader,
        "eval": eval_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
    }


def estimate_gradient_lipschitz(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20) -> float:
    model.train()
    max_norm = 0.0
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        loss = F.cross_entropy(model(images), targets, reduction="mean")
        grads = torch.autograd.grad(loss, list(model.parameters()), retain_graph=False, create_graph=False)
        norm = math.sqrt(sum(g.detach().pow(2).sum().item() for g in grads))
        max_norm = max(max_norm, norm)
        if batch_idx >= max_batches:
            break
    model.zero_grad(set_to_none=True)
    return max(max_norm, 1e-6)


def estimate_smoothness(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_builder,
    sigma: float = 1e-3,
    max_batches: int = 20,
) -> float:
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    model.train()
    grads_base: Optional[List[torch.Tensor]] = None
    count = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        grads = torch.autograd.grad(
            F.cross_entropy(model(images), targets, reduction="mean"),
            list(model.parameters()),
            retain_graph=False,
            create_graph=False,
        )
        grads = [g.detach() for g in grads]
        if grads_base is None:
            grads_base = grads
        else:
            grads_base = [gb + g for gb, g in zip(grads_base, grads)]
        count += 1
        if count >= max_batches:
            break
    if grads_base is None:
        raise RuntimeError("Unable to estimate smoothness: empty loader")
    grads_base = [g / count for g in grads_base]

    depth = getattr(base_model, "depth", 20)
    width_multiplier = getattr(base_model, "_width_multiplier", 1.0)
    num_classes = base_model.fc.out_features if hasattr(base_model, "fc") else 10
    perturbed = model_builder(depth=depth, width_multiplier=width_multiplier, num_classes=num_classes).to(device)
    perturbed.load_state_dict(base_model.state_dict())
    perturbed.train()
    with torch.no_grad():
        for p in perturbed.parameters():
            p.add_(torch.randn_like(p) * sigma)
    diff_norm = 0.0
    for p_orig, p_new in zip(model.parameters(), perturbed.parameters()):
        diff_norm += (p_orig.detach() - p_new.detach()).pow(2).sum().item()
    diff_norm = math.sqrt(max(diff_norm, 1e-12))

    grads_perturbed: Optional[List[torch.Tensor]] = None
    count = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        grads = torch.autograd.grad(
            F.cross_entropy(perturbed(images), targets, reduction="mean"),
            list(perturbed.parameters()),
            retain_graph=False,
            create_graph=False,
        )
        grads = [g.detach() for g in grads]
        if grads_perturbed is None:
            grads_perturbed = grads
        else:
            grads_perturbed = [gb + g for gb, g in zip(grads_perturbed, grads)]
        count += 1
        if count >= max_batches:
            break
    if grads_perturbed is None:
        raise RuntimeError("Unable to estimate smoothness: empty loader")
    grads_perturbed = [g / count for g in grads_perturbed]

    grad_diff = 0.0
    for g0, g1 in zip(grads_base, grads_perturbed):
        grad_diff += (g0 - g1).pow(2).sum().item()
    grad_diff = math.sqrt(max(grad_diff, 1e-12))

    return max(grad_diff / diff_norm, 1e-6)


def estimate_gradient_bounds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    risk: RiskModel,
    eta0: float,
    max_batches: int = 20,
) -> Tuple[float, float]:
    model.train()
    max_param_norm = 0.0
    max_eta_grad = 0.0
    params = list(model.parameters())
    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        eta_tensor = torch.tensor(float(eta0), device=device, requires_grad=True)
        logits = model(images)
        losses = F.cross_entropy(logits, targets, reduction="none")
        s_val = torch.exp(losses / (eta_tensor + 1e-12)).mean()
        risk_val = risk.f(s_val, eta_tensor)
        grad_params = torch.autograd.grad(risk_val, params, retain_graph=True, create_graph=False)
        grad_eta = torch.autograd.grad(risk_val, eta_tensor, retain_graph=False, create_graph=False)[0]
        param_norm = math.sqrt(sum(g.detach().pow(2).sum().item() for g in grad_params))
        max_param_norm = max(max_param_norm, param_norm)
        max_eta_grad = max(max_eta_grad, grad_eta.detach().abs().item())
        if batch_idx >= max_batches:
            break
    model.zero_grad(set_to_none=True)
    return max(max_param_norm, 1e-6), max(max_eta_grad, 1e-6)


def calibrate_sampling_parameters(
    *,
    L0: float,
    L1: float,
    L2: float,
    D0: float,
    D1: float,
    D2: float,
    eps: float,
    delta: float,
    n: int,
    d: int,
    max_batch_size: Optional[int] = None,
    max_q: Optional[int] = None,
) -> Dict[str, float]:
    tiny = 1e-12
    L0 = max(L0, tiny)
    L1 = max(L1, tiny)
    L2 = max(L2, tiny)
    eps = max(eps, tiny)
    delta = max(delta, tiny)

    c0 = max(32.0 * L2, 8.0 * L0)
    c2 = max((1.0 / max(8.0 * L2, tiny)) + (L1 / max(L0 ** 3, tiny)), 1.0)

    N1_min = (6.0 * D2 * c0 * c2) / max(eps ** 2, tiny)
    N1 = min(n, max(1, int(math.ceil(N1_min))))
    if max_batch_size is not None:
        N1 = min(N1, max_batch_size)

    c1 = (
        4.0
        + (8.0 * (L1 ** 2) * D2) / max(N1 * (L0 ** 2), tiny)
        + (32.0 * (L1 ** 2) * D2) / max(N1 * (L0 ** 2), tiny)
        + (16.0 * (L1 ** 2) * L2) / max(5.0 * D1 * (L0 ** 3), tiny)
    )

    c3 = (
        1.0
        + L2 / max(10.0 * L0, tiny)
        + (L0 * D1 + L0 + 2.0 * L0 * L2 * D2) / max(L2, tiny)
        + (33.0 * (L2 ** 2)) / max(5.0 * L0 * L2, tiny)
        + (L1 ** 2) / max(15.0 * (L2 ** 3), tiny)
        + (L1 ** 2) / max(2.0 * L0 * (L2 ** 2), tiny)
    )

    c4 = 17.0 / 4.0 + math.sqrt(max(c3, tiny)) + math.sqrt(1.0 / max(60.0 * L2, tiny))

    log_term = math.log(1.0 / delta)
    q_raw = (n * eps) / max(math.sqrt(d * max(log_term, tiny)), tiny)
    q = max(1, int(math.floor(q_raw)))
    q = min(q, n)
    if max_q is not None:
        q = max(1, min(q, int(max_q)))

    N2_min = max(
        (20.0 * q * D1 * L2) / max(L0, tiny),
        20.0 * q * c2 * L2,
        (12.0 * q * (L1 ** 2) * c0 * c2) / max(L0 ** 2, tiny),
        float(q),
    )
    N2 = min(n, max(1, int(math.ceil(N2_min))))
    if max_batch_size is not None:
        N2 = min(N2, max_batch_size)

    N3_min = max(
        (200.0 * D1 * L2) / max(L0, tiny),
        (3.0 * c0 * (D0 + 4.0 * D1 * D2) * n) / max(2.0 * L0, tiny),
    )
    N3 = min(n, max(1, int(math.ceil(N3_min))))
    if max_batch_size is not None:
        N3 = min(N3, max_batch_size)

    N4_min = max(
        (5.0 * q * L2) / max(L0, tiny),
        (6.0 * q * c1 * c0) / max(L0, tiny),
    )
    N4 = min(n, max(1, int(math.ceil(N4_min))))
    if max_batch_size is not None:
        N4 = min(N4, max_batch_size)

    return {
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
        "q": q,
        "N1": N1,
        "N2": N2,
        "N3": N3,
        "N4": N4,
    }


def compute_noise_scales(
    *,
    c: float,
    L0: float,
    L1: float,
    L2: float,
    G_bound: float,
    M_bound: float,
    L_aux: float,
    epsilon: float,
    delta: float,
    N1: int,
    N2: int,
    N3: int,
    N4: int,
    q: int,
    T: int,
    n: int,
    alpha: float,
    beta_cap_const: float,
    eta_min: float,
    eta_grad_bound: float,
    param_grad_bound: float,
    num_classes: int,
) -> Dict[str, float]:
    tiny = 1e-12
    log_term = math.sqrt(max(math.log(1.0 / max(delta, tiny)), tiny))

    scale_anchor = max(1.0 / max(float(N1), 1.0), math.sqrt(float(T)) / (max(float(n), 1.0) * math.sqrt(max(float(q), 1.0))))
    sigma1 = (c * L2 * log_term / max(epsilon, tiny)) * scale_anchor

    delta_eta = alpha * eta_grad_bound
    delta_x = beta_cap_const * param_grad_bound

    L_N2 = 2.0 * max(L2 * delta_eta, (G_bound * M_bound * delta_x) / max(eta_min, tiny))
    scale_follow_eta = max(1.0 / max(float(N2), 1.0), math.sqrt(float(T)) / (max(float(n), 1.0) * math.sqrt(max(float(q), 1.0))))
    sigma2 = (c * L_N2 * log_term / max(epsilon, tiny)) * scale_follow_eta

    coeff = L0 + L1 * math.sqrt(max(float(num_classes), 1.0))
    scale_anchor_x = max(1.0 / max(float(N3), 1.0), math.sqrt(float(T)) / (max(float(n), 1.0) * math.sqrt(max(float(q), 1.0))))
    sigma3 = (c * coeff * log_term / max(epsilon, tiny)) * scale_anchor_x

    L_N4 = 2.0 * max(
        (M_bound * L_aux * delta_eta) / max(eta_min, tiny),
        coeff * delta_x,
    )
    scale_follow_x = max(1.0 / max(float(N4), 1.0), math.sqrt(float(T)) / (max(float(n), 1.0) * math.sqrt(max(float(q), 1.0))))
    sigma4 = (c * L_N4 * log_term / max(epsilon, tiny)) * scale_follow_x

    return {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "sigma3": sigma3,
        "sigma4": sigma4,
        "L_N2": L_N2,
        "L_N4": L_N4,
        "delta_eta_bound": delta_eta,
        "delta_x_bound": delta_x,
        "scale_anchor": scale_anchor,
        "scale_anchor_x": scale_anchor_x,
        "scale_follow_eta": scale_follow_eta,
        "scale_follow_x": scale_follow_x,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DP Double-Spider training on CIFAR10-ST")
    default_data_root = PROJECT_ROOT / "CIFAR10" / "cifar10_st"
    default_output = PROJECT_ROOT / "dro_alg1" / "runs" / "cifar10_st_double_spider"
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="Path containing cifar10_st/train.npz and test.npz")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Where to store checkpoints and metrics")
    parser.add_argument("--epsilon", type=float, default=4.0, help="Privacy budget ε")
    parser.add_argument("--delta", type=float, default=None, help="Privacy parameter δ (defaults to n^-1.1)")
    parser.add_argument("--iterations", type=int, default=None, help="Override total iteration count T")
    parser.add_argument("--epochs", type=int, default=30, help="Number of anchor epochs (used if --iterations not set)")
    parser.add_argument("--lambda0", type=float, default=1e-3, help="Initial η (λ₀ in Algorithm)")
    parser.add_argument("--eta-min", type=float, default=1e-4, help="Lower bound enforced on η")
    parser.add_argument("--rho", type=float, default=0.5, help="Risk model parameter ρ")
    parser.add_argument("--model-depth", type=int, default=20, help="ResNet depth")
    parser.add_argument("--width-multiplier", type=float, default=1.0, help="ResNet width multiplier")
    parser.add_argument("--D0", type=float, required=True, help="Bound D0 used in sampling calibration")
    parser.add_argument("--D1", type=float, required=True, help="Bound D1 used in sampling calibration")
    parser.add_argument("--D2", type=float, required=True, help="Bound D2 used in sampling calibration")
    parser.add_argument("--H", type=float, required=True, help="Upper bound H appearing in β_t definition")
    parser.add_argument("--estimation-batch-size", type=int, default=256, help="Batch size for constant estimation routines")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Batch size for evaluation and metric tracking")
    parser.add_argument("--alpha-scale", type=float, default=1.0, help="Scaling applied to 1/(4L2) for η-step size")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Global norm clip applied to DP gradient estimates")
    parser.add_argument("--grad-clip-eta", type=float, default=1.0, help="Clamp applied to scalar eta gradient updates")
    parser.add_argument("--exp-clip", type=float, default=20.0, help="Clamp bound for exp(loss/eta) stabilisation")
    parser.add_argument("--max-batch", type=int, default=1024, help="Maximum batch size cap for N1..N4")
    parser.add_argument("--max-q", type=int, default=32, help="Maximum anchor interval q")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--max-constant-batches", type=int, default=20, help="Mini-batches used for constant estimation")
    parser.add_argument("--c-noise", type=float, default=1.0, help="Multiplier c in the DP noise calibration formulas")
    parser.add_argument("--L-aux", type=float, default=None, help="Override auxiliary Lipschitz constant L used in σ₄ (defaults to L1 estimate)")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )
    set_seed(args.seed)

    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    data_root = args.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    loaders = prepare_dataloaders(
        data_root,
        estimation_batch_size=args.estimation_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    train_loader = loaders["train"]
    eval_loader = loaders["eval"]
    test_loader = loaders["test"]
    train_dataset = loaders["train_dataset"]

    if args.delta is None:
        args.delta = float(len(train_dataset) ** (-1.1))

    model = build_resnet_cifar(depth=args.model_depth, width_multiplier=args.width_multiplier, num_classes=10).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        LOGGER.info("Using DataParallel across %d GPUs", torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    risk = RiskModel(rho=args.rho)

    LOGGER.info("Estimating smoothness constants...")
    with torch.enable_grad():
        L1_est = estimate_gradient_lipschitz(model, train_loader, device, max_batches=args.max_constant_batches)
        L2_est = estimate_smoothness(
            model,
            train_loader,
            device,
            build_resnet_cifar,
            sigma=1e-3,
            max_batches=args.max_constant_batches,
        )
    L0_est = max(L2_est, 1e-6)
    constants = {
        "L0": float(L0_est),
        "L1": float(L1_est),
        "L2": float(L2_est),
    }
    LOGGER.info("Estimated constants: %s", json.dumps(constants, indent=2))

    LOGGER.info("Estimating gradient bounds...")
    with torch.enable_grad():
        param_grad_bound, eta_grad_bound = estimate_gradient_bounds(
            model,
            train_loader,
            device,
            risk,
            eta0=args.lambda0,
            max_batches=args.max_constant_batches,
        )
    gradient_bounds = {
        "param_grad_bound": float(param_grad_bound),
        "eta_grad_bound": float(eta_grad_bound),
    }
    LOGGER.info("Gradient bounds: %s", json.dumps(gradient_bounds, indent=2))

    n = len(train_dataset)
    d = count_parameters(model)
    LOGGER.info("Dataset size n=%d | parameter dimension d=%d", n, d)

    sampling_params = calibrate_sampling_parameters(
        L0=constants["L0"],
        L1=constants["L1"],
        L2=constants["L2"],
        D0=float(args.D0),
        D1=float(args.D1),
        D2=float(args.D2),
        eps=float(args.epsilon),
        delta=float(args.delta),
        n=n,
        d=d,
        max_batch_size=args.max_batch,
        max_q=args.max_q,
    )
    LOGGER.info("Sampling calibration: %s", json.dumps(sampling_params, indent=2))

    q_raw = int(max(1, sampling_params["q"]))
    if args.iterations is not None:
        T_total = int(max(1, args.iterations))
    else:
        T_total = int(max(1, args.epochs * q_raw))
    q_effective = min(q_raw, T_total)
    sampling_params["q"] = q_effective
    args.iterations = T_total

    alpha = max(args.alpha_scale, 1e-12) / max(4.0 * constants["L2"], 1e-12)
    beta_cap_denom = 2.0 * constants["L0"] + constants["L1"] * math.sqrt(max(float(args.H), 1e-12))
    beta_cap_const = 1.0 / max(beta_cap_denom, 1e-12)
    beta_info = {
        "beta_cap_const": float(beta_cap_const),
        "beta_cap_denom": float(beta_cap_denom),
    }
    LOGGER.info("Step-size parameters: %s", json.dumps({"alpha": alpha, **beta_info}, indent=2))

    noise_params = compute_noise_scales(
        c=float(args.c_noise),
        L0=constants["L0"],
        L1=constants["L1"],
        L2=constants["L2"],
        G_bound=gradient_bounds["eta_grad_bound"],
        M_bound=gradient_bounds["param_grad_bound"],
        L_aux=float(args.L_aux) if args.L_aux is not None else float(L1_est),
        epsilon=float(args.epsilon),
        delta=float(args.delta),
        N1=int(sampling_params["N1"]),
        N2=int(sampling_params["N2"]),
        N3=int(sampling_params["N3"]),
        N4=int(sampling_params["N4"]),
        q=q_effective,
        T=int(args.iterations),
        n=n,
        alpha=float(alpha),
        beta_cap_const=float(beta_cap_const),
        eta_min=float(args.eta_min),
        eta_grad_bound=float(eta_grad_bound),
        param_grad_bound=float(param_grad_bound),
        num_classes=10,
    )
    LOGGER.info("Noise calibration: %s", json.dumps(noise_params, indent=2))

    trainer_cfg = DPDoubleSpiderConfig(
        alpha=float(alpha),
        eta0=float(args.lambda0),
        q=q_effective,
        N1=int(sampling_params["N1"]),
        N2=int(sampling_params["N2"]),
        N3=int(sampling_params["N3"]),
        N4=int(sampling_params["N4"]),
        sigma1=float(noise_params["sigma1"]),
        sigma2=float(noise_params["sigma2"]),
        sigma3=float(noise_params["sigma3"]),
        sigma4=float(noise_params["sigma4"]),
        T=int(args.iterations),
        L0=float(L0_est),
        n=n,
        beta_cap_const=float(beta_cap_const),
        exp_clip=float(args.exp_clip),
        grad_clip=float(args.grad_clip),
        grad_clip_eta=float(args.grad_clip_eta),
        num_workers=args.num_workers,
        eta_min=float(args.eta_min),
        log_interval=int(args.log_interval),
        eval_interval=int(args.eval_interval) if args.eval_interval > 0 else None,
    )

    trainer = DPDoubleSpiderTrainer(
        model=model,
        risk=risk,
        cfg=trainer_cfg,
        device=device,
        dataset=train_dataset,
        eval_loader=eval_loader,
        test_loader=test_loader,
        process_path=output_dir / "process.json",
    )

    trainer_config_dump = {
        **gradient_bounds,
        **constants,
        **sampling_params,
        **noise_params,
        "alpha": trainer_cfg.alpha,
        "eta0": trainer_cfg.eta0,
        "eta_min": trainer_cfg.eta_min,
        "beta_cap_const": trainer_cfg.beta_cap_const,
        "H": float(args.H),
        "T": trainer_cfg.T,
        "n": trainer_cfg.n,
        "estimation_batch_size": args.estimation_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "D0": float(args.D0),
        "D1": float(args.D1),
        "D2": float(args.D2),
        "alpha_scale": float(args.alpha_scale),
        "grad_clip": float(args.grad_clip),
        "exp_clip": float(args.exp_clip),
        "grad_clip_eta": float(args.grad_clip_eta),
    }
    LOGGER.info("Trainer configuration snapshot: %s", json.dumps(trainer_config_dump, indent=2))

    start_time = time.perf_counter()
    history = trainer.fit()
    training_seconds = time.perf_counter() - start_time
    LOGGER.info("Training completed in %.2f seconds over %d steps", training_seconds, trainer.global_step)
    LOGGER.info("Final metrics: %s", json.dumps(trainer.final_metrics, indent=2))
    test_metrics = trainer.final_metrics.get("test", {}) if isinstance(trainer.final_metrics, dict) else {}
    final_test_acc = float(test_metrics.get("accuracy", 0.0)) / 100.0

    val_acc_history = [
        float(entry.get("val_acc", float("nan"))) / 100.0
        for entry in history
        if entry.get("val_acc") is not None
    ]
    best_val_acc = max(val_acc_history) if val_acc_history else final_test_acc
    best_step = next(
        (entry.get("step") for entry in history if entry.get("val_acc") is not None and float(entry["val_acc"]) / 100.0 == best_val_acc),
        None,
    )
    avg_val_acc = sum(val_acc_history) / len(val_acc_history) if val_acc_history else final_test_acc

    summary = {
        "final_accuracy": final_test_acc,
        "best_accuracy": best_val_acc,
        "average_accuracy": avg_val_acc,
        "best_step": best_step,
        "runtime_seconds": training_seconds,
        "steps": trainer.global_step,
    }
    print(
        f"[dro1] final_acc={summary['final_accuracy']:.4f}, "
        f"best_acc={summary['best_accuracy']:.4f} (step={summary['best_step']}), "
        f"avg_acc={summary['average_accuracy']:.4f}, "
        f"runtime={summary['runtime_seconds']:.2f}s"
    )

    metrics_path = output_dir / "metrics_double_spider.json"
    args_payload = vars(args).copy()
    args_payload["data_root"] = str(args.data_root)
    args_payload["output_dir"] = str(args.output_dir)

    payload = {
        "history": history,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "constants": {
            "L0": float(L0_est),
            "L1": float(L1_est),
            "L2": float(L2_est),
            "param_grad_bound": float(param_grad_bound),
            "eta_grad_bound": float(eta_grad_bound),
        },
        "sampling": sampling_params,
        "noise": noise_params,
        "step_sizes": {
            "alpha": float(alpha),
            "beta_cap_const": float(beta_cap_const),
            "H": float(args.H),
            "alpha_scale": float(args.alpha_scale),
            "grad_clip": float(args.grad_clip),
            "grad_clip_eta": float(args.grad_clip_eta),
            "exp_clip": float(args.exp_clip),
        },
        "bounds": {
            "D0": float(args.D0),
            "D1": float(args.D1),
            "D2": float(args.D2),
        },
        "args": args_payload,
        "summary": trainer.summary(),
        "final_metrics": trainer.final_metrics,
        "aggregates": summary,
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "eta": float(trainer.eta.detach().cpu().item()),
        "args": args_payload,
        "noise": noise_params,
        "constants": payload["constants"],
        "sampling": sampling_params,
        "step_sizes": {
            "alpha": float(alpha),
            "beta_cap_const": float(beta_cap_const),
            "H": float(args.H),
            "alpha_scale": float(args.alpha_scale),
            "grad_clip": float(args.grad_clip),
            "grad_clip_eta": float(args.grad_clip_eta),
            "exp_clip": float(args.exp_clip),
        },
        "aggregates": summary,
    }
    torch.save(checkpoint, output_dir / "checkpoints" / "final_double_spider.pt")

    LOGGER.info("Training finished. Metrics saved to %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
