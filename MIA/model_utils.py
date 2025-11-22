from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RSDRO_ROOT = PROJECT_ROOT / "dro2_new"
for p in (PROJECT_ROOT, RSDRO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rsdro import ResNet20  # type: ignore  # noqa: E402


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resnet20(checkpoint_path: Path, device: torch.device | None = None, width_factor: float = 0.5) -> torch.nn.Module:
    device = device or get_device()
    model = ResNet20(num_classes=10, width_factor=width_factor)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Ensure you ran dro2_new/train_rsdro.py with --save-model "
            "or pass the correct path to the saved rsdro_resnet20.pt."
        )
    payload = torch.load(checkpoint_path, map_location=device)
    if "model" in payload:
        state_dict = payload["model"]
    elif "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_outputs(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, np.ndarray]:
    """Run inference and return logits, targets, membership flags, and indices."""
    logits_list = []
    targets_list = []
    members_list = []
    indices_list = []
    for batch in loader:
        if len(batch) == 3:
            images, targets, indices = batch
            is_member = torch.zeros_like(targets, dtype=torch.long)
        else:
            images, targets, is_member, indices = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        logits_list.append(outputs.detach().cpu())
        targets_list.append(targets.detach().cpu())
        members_list.append(is_member.detach().cpu())
        indices_list.append(indices.detach().cpu())
    logits = torch.cat(logits_list, dim=0).numpy()
    targets = torch.cat(targets_list, dim=0).numpy()
    members = torch.cat(members_list, dim=0).numpy()
    indices = torch.cat(indices_list, dim=0).numpy()
    return {"logits": logits, "targets": targets, "members": members, "indices": indices}
