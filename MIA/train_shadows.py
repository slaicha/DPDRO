from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RSDRO_ROOT = PROJECT_ROOT / "dro2_new"
for p in (PROJECT_ROOT, RSDRO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rsdro import ResNet20  # type: ignore  # noqa: E402


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a JSON list.")
    return payload


def train_one(
    entry: Dict[str, str],
    train_set,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    indices_path = Path(entry["train_indices_path"])
    if not indices_path.exists():
        raise FileNotFoundError(f"Missing indices file: {indices_path}")
    train_indices = np.load(indices_path)
    train_subset = Subset(train_set, train_indices.tolist())
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ResNet20(num_classes=10, width_factor=args.width_factor).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            acc = evaluate(model, test_loader, device)
            print(f"[shadow {entry.get('name','?')}] epoch {epoch+1}: test_acc={acc:.4f}")

    ckpt_path = Path(entry["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "train_indices_path": str(indices_path)}, ckpt_path)
    print(f"[shadow {entry.get('name','?')}] saved to {ckpt_path}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += int(targets.numel())
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shadow models listed in a manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest produced by make_shadow_manifest.py")
    parser.add_argument("--data-root", type=str, default="data", help="Root containing CIFAR-10 data.")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width-factor", type=float, default=0.5, help="Width factor used by RS-DRO ResNet20.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
    test_set = CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = _load_manifest(Path(args.manifest))

    for entry in manifest:
        train_one(entry, train_set, test_loader, device, args)


if __name__ == "__main__":
    main()
