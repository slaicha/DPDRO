import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset

# --- ResNet20 Model Definition ---


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetImpl(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20():
    return ResNetImpl(BasicBlock, [3, 3, 3])


# --- Imbalanced CIFAR-10 Dataset ---


class CustomImbalancedCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.cifar_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
        self.targets = np.array(self.cifar_dataset.targets)
        self.indices = self._get_imbalanced_indices()

    def _get_imbalanced_indices(self):
        indices = []
        for c in range(10):
            class_indices = np.where(self.targets == c)[0]
            if c < 5:
                indices.extend(class_indices[-100:])
            else:
                indices.extend(class_indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.cifar_dataset[original_idx]


# --- Standard Training & Testing Loop ---


def train_standard(epoch, model, train_loader, optimizer, criterion, device):
    print(f"\nEpoch: {epoch}")
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(
                f"  Batch {batch_idx}/{len(train_loader)}: Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})"
            )


def test_standard(epoch, model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    print(f"Test Results for Epoch {epoch}: Loss: {test_loss/len(test_loader):.3f} | Acc: {acc:.3f}% ({correct}/{total})")
    return acc


# --- DP Double-SPIDER Algorithm (with clipping) ---


def clip_vector(x, C):
    norm = torch.norm(x)
    if norm <= C or C <= 0:
        return x
    return x * (C / norm)


class DPDoubleSpiderTrainer:
    def __init__(
        self,
        *,
        T,
        q,
        epsilon,
        delta,
        n,
        d,
        L0,
        L1,
        L2,
        D0,
        D1,
        D2,
        H,
        G,
        M,
        lambda_val,
        C1,
        C2,
        C3,
        C4,
        N1,
        N2,
        N3,
        N4,
        num_workers=2,
    ):
        self.T = T
        self.q = q
        self.epsilon = epsilon
        self.delta = delta
        self.n = n
        self.d = d
        self.L0, self.L1, self.L2 = L0, L1, L2
        self.D0, self.D1, self.D2 = D0, D1, D2
        self.H = H
        self.G = G
        self.M = M
        self.lambda_val = lambda_val
        self.C1, self.C2, self.C3, self.C4 = C1, C2, C3, C4
        self.N1, self.N2, self.N3, self.N4 = N1, N2, N3, N4
        self.num_workers = num_workers

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Step sizes
        self.alpha = 1.0 / (4.0 * self.L2)
        self.beta_term1 = 1.0 / (2.0 * self.L0 + self.L1 * math.sqrt(self.H))

        # Noise scales (big-O constants set to 1)
        log_delta = math.log(1.0 / self.delta)
        sqrt_term = math.sqrt(self.T) / (self.n * math.sqrt(self.q))
        self.sigma1 = (self.C1 * math.sqrt(self.T * log_delta)) / (self.n * math.sqrt(self.q) * self.epsilon)
        self.sigma2 = (self.C2 * math.sqrt(log_delta)) / (self.N2 * self.epsilon)
        self.sigma3 = (self.C3 * math.sqrt(self.T * log_delta)) / (self.n * math.sqrt(self.q) * self.epsilon)
        self.sigma4 = (self.C4 * math.sqrt(log_delta)) / self.epsilon * max(1.0 / self.N4, sqrt_term)

    def _get_loader(self, dataset, batch_size):
        bs = min(batch_size, self.n)
        return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    @staticmethod
    def _flatten_params(model):
        return torch.cat([p.detach().view(-1) for p in model.parameters()])

    @staticmethod
    def _load_params(model, flat):
        offset = 0
        with torch.no_grad():
            for param in model.parameters():
                numel = param.numel()
                param.data.copy_(flat[offset : offset + numel].view_as(param))
                offset += numel

    @staticmethod
    def _grad_wrt(loss, params):
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=False)
        return torch.cat([g.view(-1) for g in grads])

    @staticmethod
    def _psi_star(t):
        return 0.25 * t ** 2 + t

    def _dro_loss(self, model, eta, data, targets):
        outputs = model(data)
        ell = F.cross_entropy(outputs, targets, reduction="none")
        psi_in = (ell - eta) / self.lambda_val
        psi_val = self._psi_star(psi_in)
        return self.lambda_val * psi_val.mean() + eta

    def train(self, model, eta_init, dataset):
        print("Starting DP Double-SPIDER training loop")
        model = model.to(self.device)
        eta_t = eta_init.to(self.device).detach().clone()
        eta_t.requires_grad_(True)
        eta_prev = eta_t.detach().clone()

        x_t = self._flatten_params(model).to(self.device)
        x_prev = x_t.clone()

        g_t = torch.zeros_like(eta_t)
        v_t = torch.zeros_like(x_t)

        loader1 = self._get_loader(dataset, self.N1)
        loader2 = self._get_loader(dataset, self.N2)
        loader3 = self._get_loader(dataset, self.N3)
        loader4 = self._get_loader(dataset, self.N4)

        it1, it2, it3, it4 = iter(loader1), iter(loader2), iter(loader3), iter(loader4)

        def next_batch(it_obj, loader):
            try:
                return next(it_obj), it_obj
            except StopIteration:
                it_new = iter(loader)
                return next(it_new), it_new

        sampled_x = x_t.clone()
        sampled_eta = eta_t.detach().clone()

        for t in range(self.T):
            self._load_params(model, x_t)
            model.train()

            if t % self.q == 0:
                (data, targets), it1 = next_batch(it1, loader1)
                data, targets = data.to(self.device), targets.to(self.device)
                eta_t.requires_grad_(True)
                loss = self._dro_loss(model, eta_t, data, targets)
                grad_eta = self._grad_wrt(loss, [eta_t])
                g_new = clip_vector(grad_eta, self.C1)
                g_new = g_new + torch.normal(0.0, self.sigma1, size=g_new.shape, device=self.device)
            else:
                (data, targets), it2 = next_batch(it2, loader2)
                data, targets = data.to(self.device), targets.to(self.device)
                eta_t.requires_grad_(True)
                loss_curr = self._dro_loss(model, eta_t, data, targets)
                grad_eta_curr = self._grad_wrt(loss_curr, [eta_t])

                self._load_params(model, x_prev)
                eta_prev_var = eta_prev.detach().clone().to(self.device)
                eta_prev_var.requires_grad_(True)
                loss_prev = self._dro_loss(model, eta_prev_var, data, targets)
                grad_eta_prev = self._grad_wrt(loss_prev, [eta_prev_var])

                self._load_params(model, x_t)
                eta_t = eta_t.detach().clone()
                eta_t.requires_grad_(True)

                diff = grad_eta_curr - grad_eta_prev
                clipped = clip_vector(diff, self.C2)
                g_new = clipped + g_t + torch.normal(0.0, self.sigma2, size=clipped.shape, device=self.device)

            g_t = g_new

            eta_prev = eta_t.detach().clone()
            eta_t = (eta_t - self.alpha * g_t).detach()
            eta_t.requires_grad_(True)

            if t % self.q == 0:
                (data, targets), it3 = next_batch(it3, loader3)
                data, targets = data.to(self.device), targets.to(self.device)
                loss = self._dro_loss(model, eta_t, data, targets)
                grad_x = self._grad_wrt(loss, list(model.parameters()))
                v_new = clip_vector(grad_x, self.C3)
                v_new = v_new + torch.normal(0.0, self.sigma3, size=v_new.shape, device=self.device)
            else:
                (data, targets), it4 = next_batch(it4, loader4)
                data, targets = data.to(self.device), targets.to(self.device)

                loss_curr = self._dro_loss(model, eta_t, data, targets)
                grad_x_curr = self._grad_wrt(loss_curr, list(model.parameters()))

                self._load_params(model, x_prev)
                eta_prev_var = eta_prev.detach().clone().to(self.device)
                eta_prev_var.requires_grad_(True)
                loss_prev = self._dro_loss(model, eta_prev_var, data, targets)
                grad_x_prev = self._grad_wrt(loss_prev, list(model.parameters()))

                self._load_params(model, x_t)

                diff = grad_x_curr - grad_x_prev
                clipped = clip_vector(diff, self.C4)
                v_new = clipped + v_t + torch.normal(0.0, self.sigma4, size=clipped.shape, device=self.device)

            v_t = v_new

            v_norm = torch.norm(v_t).item()
            beta_dynamic = float("inf") if v_norm < 1e-12 else 1.0 / (self.L0 * math.sqrt(self.n) * v_norm)
            beta_t = min(self.beta_term1, beta_dynamic)

            x_prev = x_t.clone()
            x_t = x_t - beta_t * v_t
            self._load_params(model, x_t)

            if torch.rand(1, device=self.device).item() < 1.0 / (t + 1):
                sampled_x = x_t.clone()
                sampled_eta = eta_t.detach().clone()

            if t % max(1, self.q) == 0:
                print(
                    f"Iter {t+1}/{self.T} | ||g_t||={torch.norm(g_t):.4f} ||v_t||={torch.norm(v_t):.4f} beta={beta_t:.3e}"
                )

        self._load_params(model, sampled_x)
        return model, sampled_eta.detach()


# --- Main Execution ---


def parse_args():
    parser = argparse.ArgumentParser(description="DP Double-SPIDER for DRO (dro1_new)")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--baseline-lr", type=float, default=0.1)
    parser.add_argument("--baseline-momentum", type=float, default=0.9)
    parser.add_argument("--baseline-weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler-Tmax", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--delta-exponent", type=float, default=1.1)
    parser.add_argument("--M", type=float, default=0.5)
    parser.add_argument("--G", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--sigma-squared", type=float, default=None)
    parser.add_argument("--lambda-val", type=float, default=0.1)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--C1", type=float, default=1.0)
    parser.add_argument("--C2", type=float, default=1.0)
    parser.add_argument("--C3", type=float, default=1.0)
    parser.add_argument("--C4", type=float, default=1.0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--run-dp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading data...")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = CustomImbalancedCIFAR10(
        root=args.data_root, train=True, download=True, transform=transform_train
    )
    n = len(train_dataset)

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNet20().to(device)
    d = sum(p.numel() for p in model.parameters() if p.requires_grad)

    epsilon = args.epsilon
    delta = args.delta if args.delta is not None else 1.0 / (n ** args.delta_exponent)
    print(f"Dataset size (n): {n}")
    print(f"Parameter dim (d): {d}")
    print(f"DP params: epsilon={epsilon}, delta={delta:.3e}")

    criterion = nn.CrossEntropyLoss()
    if not args.skip_baseline:
        print("\n--- Starting Standard Training (baseline) ---")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.baseline_lr,
            momentum=args.baseline_momentum,
            weight_decay=args.baseline_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.scheduler_Tmax
        )
        best_acc = 0.0
        for epoch in range(args.baseline_epochs):
            train_standard(epoch, model, train_loader, optimizer, criterion, device)
            acc = test_standard(epoch, model, test_loader, criterion, device)
            best_acc = max(best_acc, acc)
            scheduler.step()
        print(f"\n--- Standard Training Finished ---\nBest Test Accuracy: {best_acc:.3f}%")

    # --- DP Double-SPIDER parameter setup ---
    G = args.G
    L = args.L
    M = args.M
    H = args.H
    lambda_val = args.lambda_val
    sigma_sq = args.sigma_squared if args.sigma_squared is not None else G ** 2

    L0 = G + (G ** 2 * M) / lambda_val
    L1 = L / G
    L2 = (G ** 2 * M) / lambda_val

    D0 = 8 * G ** 2 + 10 * G ** 2 * M ** 2 * (lambda_val ** -2) * sigma_sq
    D1 = 8.0
    D2 = G ** 2 * M ** 2 * (lambda_val ** -2) * sigma_sq

    q_calc = (n * epsilon / math.sqrt(d * math.log(1.0 / delta))) ** (2.0 / 3.0)
    q = args.q if args.q is not None else max(1, math.ceil(q_calc))

    c0 = max(32 * L2, 8 * L0)
    c2 = max(1.0 / (8 * L2) + L1 / (L0 ** 3), 1.0)
    N1 = math.ceil((6 * D2 * c0 * c2) / (epsilon ** 2))

    c1 = 4 + (8 * L1 ** 2 * D2) / (N1 * L0 ** 2) + (32 * L1 ** 2 * D2) / (N1 * L0 ** 2) + (16 * L1 ** 2 * L2) / (5 * D1 * L0 ** 3)
    c3 = 1 + (L2 / (10 * L0)) + (L0 * D1 + L0 + 2 * L0 * L2 * D2) / L2 + (33 * L2 ** 2) / (5 * L0 * L2) + (L1 ** 2) / (15 * L2 ** 3) + (L1 ** 2) / (2 * L0 * L2 ** 2)
    c4 = 17.0 / 4.0 + math.sqrt(c3) + math.sqrt(1.0 / (60 * L2))

    N2 = math.ceil(
        max(
            (20 * q * D1 * L2) / L0,
            20 * q * c2 * L2,
            (12 * q * L1 ** 2 * c0 * c2) / (L0 ** 2),
            q,
        )
    )
    N3 = math.ceil(
        max(
            (200 * D1 * L2) / L0,
            (3 * c0 * (D0 + 4 * D1 * D2) * n) / (2 * L0),
        )
    )
    N4 = math.ceil(max((5 * q * L2) / L0, (6 * q * c1 * c0) / L0))

    print("\n--- DP Double-SPIDER parameters ---")
    print(f"q={q}, T={args.T}")
    print(f"N1={N1}, N2={N2}, N3={N3}, N4={N4}")
    print(f"C1={args.C1}, C2={args.C2}, C3={args.C3}, C4={args.C4}")

    if args.run_dp:
        dp_model = ResNet20().to(device)
        eta0 = torch.tensor(0.0, device=device, requires_grad=True)
        trainer = DPDoubleSpiderTrainer(
            T=args.T,
            q=q,
            epsilon=epsilon,
            delta=delta,
            n=n,
            d=d,
            L0=L0,
            L1=L1,
            L2=L2,
            D0=D0,
            D1=D1,
            D2=D2,
            H=H,
            G=G,
            M=M,
            lambda_val=lambda_val,
            C1=args.C1,
            C2=args.C2,
            C3=args.C3,
            C4=args.C4,
            N1=N1,
            N2=N2,
            N3=N3,
            N4=N4,
            num_workers=args.num_workers,
        )

        dp_model, final_eta = trainer.train(dp_model, eta0, train_dataset)
        print(f"Final sampled eta: {final_eta.item():.6f}")
        print("\n--- Testing DP-trained model ---")
        test_standard("DP", dp_model, test_loader, criterion, device)
    else:
        print("\nDP training not run. Use --run-dp to execute DP Double-SPIDER.")
