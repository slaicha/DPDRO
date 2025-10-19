"""ResNet variants tailored for CIFAR10-style inputs."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """ResNet backbone for CIFAR10-ST experiments."""

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: Sequence[int],
        num_classes: int = 10,
        width_multiplier: float = 1.0,
    ) -> None:
        super().__init__()

        base_channels = [16, 32, 64]
        channels = [max(8, int(round(c * width_multiplier))) for c in base_channels]

        self._num_blocks = tuple(num_blocks)
        self._width_multiplier = width_multiplier
        self.depth = 6 * num_blocks[0] + 2

        self.in_planes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2] * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def build_resnet_cifar(depth: int = 20, width_multiplier: float = 1.0, num_classes: int = 10) -> ResNetCIFAR:
    depth_to_blocks = {
        20: (3, 3, 3),
        32: (5, 5, 5),
        44: (7, 7, 7),
        56: (9, 9, 9),
        110: (18, 18, 18),
    }
    if depth not in depth_to_blocks:
        raise ValueError(f"Unsupported ResNet depth {depth}. Choose from {sorted(depth_to_blocks.keys())}.")
    return ResNetCIFAR(
        BasicBlock,
        depth_to_blocks[depth],
        num_classes=num_classes,
        width_multiplier=width_multiplier,
    )
