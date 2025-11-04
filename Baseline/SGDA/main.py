import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import json
import os

from optimizer import DPSGDA
from utils import compute_accuracy, get_cifar10_st, resnet20_cifar, set_all_seeds


def build_cifar_setup(args, device):
    train_set, test_set = get_cifar10_st(args.data_root, download=True)
    n = len(train_set)
    delta_value = args.delta if args.delta is not None else n ** (-1.1)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = resnet20_cifar(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    total_steps = max(1, args.total_epochs * steps_per_epoch)
    noise_scale = math.sqrt(total_steps * math.log(1 / delta_value)) / (n * args.epsilon)
    sigma_w = args.clip_w * noise_scale
    sigma_v = args.clip_v * noise_scale

    optimizer = DPSGDA(
        model.parameters(),
        loss_fn=loss_fn,
        lr_w=args.lr_w,
        lr_v=args.lr_v,
        sigma_w=sigma_w,
        sigma_v=sigma_v,
        clip_w=args.clip_w,
        clip_v=args.clip_v,
    )

    return dict(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        dataset_kind="cifar10-st",
        num_samples=n,
        delta=delta_value,
    )


def train(setup, args, device):
    model = setup["model"]
    loss_fn = setup["loss_fn"]
    train_loader = setup["train_loader"]
    test_loader = setup["test_loader"]
    optimizer = setup["optimizer"]
    dataset_kind = setup["dataset_kind"]

    acc_history = []

    for epoch in range(args.total_epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        total_correct = 0.0
        total = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                # logits already match CE loss
                batch_acc = compute_accuracy(outputs, targets)
                total_correct += batch_acc * targets.size(0)
                total += targets.size(0)

        test_acc = total_correct / total if total else 0.0
        acc_history.append(test_acc)

        print(
            f"epoch: {epoch}, test_acc: {test_acc:.4f}, lr_w: {optimizer.lr_w:.4f}, lr_v: {optimizer.lr_v:.4f}"
        )

    best_acc = max(acc_history) if acc_history else 0.0
    print(f"best acc: {best_acc:.4f}")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "results.json")
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump({
                "dataset": setup["dataset_kind"],
                "epochs": args.total_epochs,
                "best_accuracy": best_acc,
                "accuracy_history": acc_history,
                "epsilon": args.epsilon,
                "delta": setup["delta"],
                "lr_w": args.lr_w,
                "lr_v": args.lr_v,
                "clip_w": args.clip_w,
                "clip_v": args.clip_v,
                "num_samples": setup["num_samples"],
            }, fh, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="DP-SGDA on CIFAR10-ST")
    parser.add_argument("--data_root", default="./data", type=str, help="Dataset storage root")
    parser.add_argument("--epsilon", default=4.0, type=float, help="Param of differential privacy")
    parser.add_argument("--delta", default=None, type=float, help="Param delta for DP (default n^-1.1)")
    parser.add_argument("--clip_w", default=1.0, type=float, help="clip threshold for model gradients")
    parser.add_argument("--clip_v", default=1.0, type=float, help="clip threshold for dual gradients")
    parser.add_argument("--lr_w", default=0.2, type=float, help="learning rate for model parameters")
    parser.add_argument("--lr_v", default=0.2, type=float, help="learning rate for dual variables")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size")
    parser.add_argument("--total_epochs", default=30, type=int, help="Number of training epochs")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--workers", default=2, type=int, help="Number of dataloader workers")
    parser.add_argument("--output_dir", default=None, type=str, help="Directory to store result summary")
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    if args.batch_size is None:
        args.batch_size = 128
    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup = build_cifar_setup(args, device)
    train(setup, args, device)


if __name__ == "__main__":
    main()
