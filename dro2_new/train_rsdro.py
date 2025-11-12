"""Entry point to train ResNet20 on CIFAR10-ST with RS-DRO."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from rsdro import (
    ResNet20,
    build_cifar10_st_dataset,
    build_cifar10_st_loaders,
    build_cifar10_test_loader,
    build_full_dataset_loader,
    check_sample_size_condition,
    compute_hyperparams,
    estimate_psi0,
    run_rs_dro,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    model.train()
    return 100.0 * correct / max(1, total)


def build_balanced_sampler(subset, num_classes: int = 10) -> WeightedRandomSampler:
    if not isinstance(subset, torch.utils.data.Subset):
        raise ValueError("Balanced sampling requires a torch.utils.data.Subset.")
    base_dataset = subset.dataset
    targets = getattr(base_dataset, "targets", None)
    if targets is None:
        raise ValueError("Underlying dataset does not expose targets for balanced sampling.")
    subset_indices = np.array(subset.indices)
    subset_targets = np.array(targets)[subset_indices]
    class_counts = np.bincount(subset_targets, minlength=num_classes)
    class_counts[class_counts == 0] = 1
    sample_weights = 1.0 / class_counts[subset_targets]
    weights = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def pretrain_model(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    num_workers: int,
    val_loader=None,
    balanced_sampling: bool = False,
) -> None:
    if epochs <= 0:
        return

    sampler = build_balanced_sampler(dataset) if balanced_sampling else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = 100.0 * correct / max(1, total)
        log = f"[Pretrain] Epoch {epoch}/{epochs} | loss={train_loss:.4f} | acc={train_acc:.2f}%"
        if val_loader is not None:
            val_acc = evaluate(model, val_loader, device)
            log += f" | val_acc={val_acc:.2f}%"
        print(log)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive Spider DRO trainer")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--G", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--lambda0", type=float, default=1e-3)
    parser.add_argument("--eta-t-squared", type=float, default=1e-4)
    parser.add_argument("--psi-warmup-steps", type=int, default=200)
    parser.add_argument("--psi-warmup-lr", type=float, default=5e-4)
    parser.add_argument("--width-factor", type=float, default=0.5, help="Scales the channel count of ResNet20 to reduce memory.")
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--pretrain-batch-size", type=int, default=512)
    parser.add_argument("--pretrain-lr", type=float, default=0.2)
    parser.add_argument("--pretrain-momentum", type=float, default=0.9)
    parser.add_argument("--pretrain-weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--pretrain-balanced",
        action="store_true",
        default=True,
        help="Use a class-balanced sampler during pretraining.",
    )
    parser.add_argument(
        "--no-pretrain-balanced",
        action="store_false",
        dest="pretrain_balanced",
        help="Disable class-balanced sampling during pretraining.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-T", type=int, default=5, help="Cap RS-DRO iterations to avoid OOM.")
    parser.add_argument("--max-b1", type=int, default=4096, help="Cap b1 batch size to limit memory.")
    parser.add_argument("--max-b2", type=int, default=64, help="Cap b2 batch size to limit memory.")
    parser.add_argument("--max-q", type=int, default=5, help="Cap q to keep the refresh period manageable.")
    parser.add_argument("--results-dir", type=str, default="./runs/rsdro")
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    train_dataset, n = build_cifar10_st_dataset(args.data_root, download=True, augment=True)
    eval_dataset, _ = build_cifar10_st_dataset(args.data_root, download=False, augment=False)
    full_loader = build_full_dataset_loader(eval_dataset, batch_size=256, num_workers=args.num_workers)
    test_loader = build_cifar10_test_loader(args.data_root, batch_size=256, num_workers=args.num_workers)
    delta = n ** -1.1

    model = ResNet20(width_factor=args.width_factor).to(device)
    lambda_param = torch.tensor(args.lambda0, dtype=torch.float32, device=device)

    print(f"Dataset size n={n}, delta={delta:.4e}")

    if args.pretrain_epochs > 0:
        pretrain_model(
            model=model,
            dataset=train_dataset,
            device=device,
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            lr=args.pretrain_lr,
            momentum=args.pretrain_momentum,
            weight_decay=args.pretrain_weight_decay,
            num_workers=args.num_workers,
            val_loader=test_loader,
            balanced_sampling=args.pretrain_balanced,
        )
        pre_acc = evaluate(model, test_loader, device)
        print(f"Pretraining finished with test accuracy: {pre_acc:.2f}%")

    psi0 = estimate_psi0(
        model=deepcopy(model).to(device),
        lambda_param=lambda_param.clone(),
        dataloader=full_loader,
        rho=args.rho,
        lambda_0=args.lambda0,
        warmup_steps=args.psi_warmup_steps,
        warmup_lr=args.psi_warmup_lr,
        device=device,
    )
    print(f"Estimated Psi_0={psi0:.6f}")

    d = sum(p.numel() for p in model.parameters()) + 1
    print(f"Parameter dimension d={d}")

    sample_check = check_sample_size_condition(
        n=n,
        epsilon=args.epsilon,
        delta=delta,
        G=args.G,
        L=args.L,
        Psi_0=psi0,
        d=d,
    )
    if not sample_check.satisfied:
        print(
            "[Warning] Sample size condition violated: n must be >= max{:.2f, {:.2f}}".format(
                sample_check.required_n, sample_check.required_n_alt
            )
        )

    hyperparams = compute_hyperparams(
        n=n,
        d=d,
        epsilon=args.epsilon,
        delta=delta,
        G=args.G,
        L=args.L,
        c_const=args.c,
        Psi_0=psi0,
        eta_t_squared=args.eta_t_squared,
    )

    hyperparams.T = max(1, min(hyperparams.T, args.max_T))
    hyperparams.b1 = max(1, min(hyperparams.b1, args.max_b1))
    hyperparams.b2 = max(1, min(hyperparams.b2, args.max_b2))
    hyperparams.q = max(1, min(hyperparams.q, args.max_q))

    print("Hyperparameters:")
    print(hyperparams)

    loader_b1, loader_b2 = build_cifar10_st_loaders(
        dataset=train_dataset,
        batch_size_b1=hyperparams.b1,
        batch_size_b2=hyperparams.b2,
        num_workers=args.num_workers,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)

    final_model, final_lambda, selected_iter = run_rs_dro(
        model=model,
        lambda_param=lambda_param,
        train_dataset=train_dataset,
        loader_b1=loader_b1,
        loader_b2=loader_b2,
        hyperparams=hyperparams,
        rho=args.rho,
        lambda_lower_bound=args.lambda0,
        device=device,
        generator=generator,
    )

    print(f"Selected iterate {selected_iter} with lambda={final_lambda.item():.6f}")

    test_acc = evaluate(final_model, test_loader, device)
    print(f"Test accuracy: {test_acc:.2f}%")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "n": n,
        "delta": delta,
        "psi0": psi0,
        "hyperparams": hyperparams.__dict__,
        "test_accuracy": test_acc,
        "selected_iter": selected_iter,
        "final_lambda": float(final_lambda.item()),
    }
    results_path = results_dir / "summary.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {results_path}")

    if args.save_model:
        model_path = results_dir / "rsdro_resnet20.pt"
        torch.save({"model": final_model.state_dict(), "lambda": final_lambda}, model_path)
        print(f"Checkpoint written to {model_path}")


if __name__ == "__main__":
    main()
