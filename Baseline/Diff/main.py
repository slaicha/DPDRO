import argparse
import json
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from optimizer import PrivateDiff
from utils import compute_accuracy, get_cifar10_st, resnet20_cifar, set_all_seeds


def train(args):
    epsilon, delta = args.epsilon, args.delta
    c1, c2, cy = args.c1, args.c2, args.cy
    lr, lr_alpha = args.lr, args.lr_alpha
    t, t2 = args.T, args.T2
    seed = args.seed
    batch_size = args.batch_size
    total_epochs = args.total_epochs
    set_all_seeds(seed)

    train_set, test_set = get_cifar10_st(args.data_root, download=True)
    n = len(train_set)
    delta_value = args.delta if args.delta is not None else n ** (-1.1)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = resnet20_cifar(num_classes=10).to(args.device)

    loss_fn = nn.CrossEntropyLoss()
    sigma_1 = math.sqrt(total_epochs / t * math.log(1 / delta_value)) / (n * epsilon)
    sigma_2 = math.sqrt(math.log(1 / delta_value)) / (n * epsilon)
    sigma_alpha = math.sqrt(math.log(1 / delta_value)) / (n * epsilon)
    optimizer = PrivateDiff(
        model.parameters(),
        loss_fn=loss_fn,
        lr = lr,
        lr_alpha=lr_alpha,
        c1=c1,
        c2=c2,
        c_y=cy,
        sigma1=sigma_1,
        sigma2=sigma_2,
        sigma_alpha=sigma_alpha,
        inner_iters=t2,
        T = t,
    )

    best_acc = 0.0
    acc_log = []
    r = 0
    for epoch in range(total_epochs):

        model.train()
        for i, (data, targets) in enumerate(trainloader):
            data, targets = data.to(args.device), targets.to(args.device)

            def closure():
                logits = model(data)
                loss = loss_fn(logits, targets)
                loss.backward()
                return loss

            optimizer.zero_grad()
            logits = model(data)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step(r=r, closure=closure)
            optimizer.zero_grad()
            r += 1

        # NOTE: evaluation on train & test sets
        model.eval()

        total_correct = 0.0
        total = 0
        for test_data, test_targets in testloader:
            test_data = test_data.to(args.device)
            test_targets = test_targets.to(args.device)
            logits = model(test_data)
            batch_acc = compute_accuracy(logits, test_targets)
            total_correct += batch_acc * test_targets.size(0)
            total += test_targets.size(0)
        model.train()

        test_acc = total_correct / total if total else 0.0
        best_acc = max(best_acc, test_acc)
        acc_log.append(test_acc)

        # NOTE: print results
        print("epoch: %s, test_acc: %.4f, lr: %.4f"% (epoch, test_acc, optimizer.lr))

    best_acc = best_acc if acc_log else 0.0
    print("best acc: %.4f" % best_acc)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        result_path = os.path.join(args.output_dir, "results.json")
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump({
                "dataset": "cifar10-st",
                "best_accuracy": best_acc,
                "accuracy_history": acc_log,
                "epochs": total_epochs,
                "epsilon": epsilon,
                "delta": delta_value,
                "lr": lr,
                "lr_alpha": lr_alpha,
                "clip_c1": c1,
                "clip_c2": c2,
                "clip_cy": cy,
                "num_samples": n,
            }, fh, indent=2)


parser = argparse.ArgumentParser(description='DP PrivateDiff on CIFAR10-ST.')
parser.add_argument('--epsilon', default=4.0, type=float, help='Param of differential privacy')
parser.add_argument("--c1", default=1, type=float, help="value of gradient clip")
parser.add_argument("--c2", default=1, type=float, help="value of gradient clip")
parser.add_argument("--cy", default=1, type=float, help="value of gradient clip")
parser.add_argument("--lr", default=0.2, type=float, help="scale the lr")
parser.add_argument("--lr_alpha", default=0.2, type=float, help="scale the lr")
parser.add_argument("--T2", default="3", type=int, help="T2")
parser.add_argument("--T", default="2", type=int, help="T")
parser.add_argument("--batch_size", default=128, type=int, help="batch size")
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument("--delta", default=None, type=float, help="delta (default n^-1.1)")
parser.add_argument("--total_epochs", default=30, type=int, help="epoch")
parser.add_argument("--data_root", default="./data", type=str, help="CIFAR10 root directory")
parser.add_argument("--workers", default=2, type=int, help="dataloader workers")
parser.add_argument("--output_dir", default=None, type=str, help="Directory to store result summary")

if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    train(args)
