"""DP Double-Spider algorithm."""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .risk import RiskModel
from .utils import AverageMeter, BatchStream, BatchStreamConfig, apply_update, clone_model, tensor_list_norm


@dataclass
class DPDoubleSpiderConfig:
    """Hyper-parameters for DP Double-Spider optimisation."""

    alpha: float
    eta0: float
    q: int
    N1: int
    N2: int
    N3: int
    N4: int
    sigma1: float
    sigma2: float
    sigma3: float
    sigma4: float
    T: int
    L0: float
    n: int
    beta_cap_const: float
    exp_clip: float
    grad_clip: float
    grad_clip_eta: float
    num_workers: int = 0
    eta_min: float = 1e-4
    log_interval: int = 50
    eval_interval: Optional[int] = 200


class DPDoubleSpiderTrainer:
    """Trainer implementing Algorithm DP Double-Spider."""

    def __init__(
        self,
        model: nn.Module,
        risk: RiskModel,
        cfg: DPDoubleSpiderConfig,
        device: torch.device,
        dataset: torch.utils.data.Dataset,
        eval_loader: Optional[torch.utils.data.DataLoader] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        process_path: Optional[Path] = None,
    ) -> None:
        self.model = model.to(device)
        self.prev_model = clone_model(self.model).to(device)
        self.prev_model.load_state_dict(self.model.state_dict())
        self.risk = risk
        self.cfg = cfg
        self.device = device
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.process_path = process_path
        self.final_metrics: Dict[str, Dict[str, float] | float] = {}

        self.eta = torch.tensor(float(cfg.eta0), device=device)
        self.prev_eta = self.eta.detach().clone()
        self.g_prev: Optional[torch.Tensor] = None
        self.v_prev: Optional[List[torch.Tensor]] = None

        self.logger = logging.getLogger(f"{__name__}.trainer")

        self.stream_N1 = BatchStream(
            BatchStreamConfig(dataset=dataset, batch_size=max(1, cfg.N1), device=device, num_workers=cfg.num_workers)
        )
        self.stream_N2 = BatchStream(
            BatchStreamConfig(dataset=dataset, batch_size=max(1, cfg.N2), device=device, num_workers=cfg.num_workers)
        )
        self.stream_N3 = BatchStream(
            BatchStreamConfig(dataset=dataset, batch_size=max(1, cfg.N3), device=device, num_workers=cfg.num_workers)
        )
        self.stream_N4 = BatchStream(
            BatchStreamConfig(dataset=dataset, batch_size=max(1, cfg.N4), device=device, num_workers=cfg.num_workers)
        )

        self.global_step = 0
        self.random_state: Optional[Dict[str, object]] = None
        self.history: List[Dict[str, float]] = []
        self.beta_cap_const = cfg.beta_cap_const

    def _compute_gradients(
        self,
        model: nn.Module,
        images: torch.Tensor,
        targets: torch.Tensor,
        eta_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        eps = 1e-12
        eta = eta_value.detach().clone().requires_grad_(True)
        logits = model(images)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        losses = F.cross_entropy(logits, targets, reduction="none")
        scaled_raw = losses / (eta + eps)
        if self.cfg.exp_clip > 0:
            scaled = scaled_raw / (1.0 + scaled_raw.abs() / self.cfg.exp_clip)
        else:
            scaled = scaled_raw
        s_val = torch.exp(scaled).mean()
        risk_val = self.risk.f(s_val, eta)

        grad_params = torch.autograd.grad(risk_val, list(model.parameters()), retain_graph=True, create_graph=False)
        grad_eta = torch.autograd.grad(risk_val, eta, retain_graph=False, create_graph=False)[0]

        return grad_eta.detach(), [g.detach() for g in grad_params], s_val.detach(), risk_val.detach()

    @staticmethod
    def _add_noise_scalar(value: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return value
        return value + torch.randn_like(value) * sigma

    @staticmethod
    def _add_noise_vector(grads: Iterable[torch.Tensor], sigma: float) -> List[torch.Tensor]:
        if sigma <= 0:
            return [g.clone() for g in grads]
        return [g + torch.randn_like(g) * sigma for g in grads]

    def _update_random_state(self, t: int) -> None:
        probability = 1.0 / float(t + 1)
        if random.random() <= probability:
            self.random_state = {
                "step": t + 1,
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "eta": float(self.eta.detach().cpu().item()),
            }

    def fit(self) -> List[Dict[str, float]]:
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for t in range(self.cfg.T):
            anchor = (t % max(1, self.cfg.q) == 0)

            prev_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.prev_eta = self.eta.detach().clone()

            # eta update
            if anchor or self.g_prev is None:
                images_eta, targets_eta = self.stream_N1.next()
                grad_eta_curr, _, _, _ = self._compute_gradients(self.model, images_eta, targets_eta, self.eta)
                g_t = self._add_noise_scalar(grad_eta_curr, self.cfg.sigma1)
            else:
                images_eta, targets_eta = self.stream_N2.next()
                grad_eta_curr, _, _, _ = self._compute_gradients(self.model, images_eta, targets_eta, self.eta)
                grad_eta_prev, _, _, _ = self._compute_gradients(self.prev_model, images_eta, targets_eta, self.prev_eta)
                g_t = grad_eta_curr - grad_eta_prev
                if self.g_prev is not None:
                    g_t = g_t + self.g_prev
                g_t = self._add_noise_scalar(g_t, self.cfg.sigma2)

            if self.cfg.grad_clip_eta > 0:
                g_t = torch.clamp(g_t, min=-self.cfg.grad_clip_eta, max=self.cfg.grad_clip_eta)
            g_t = self._sanitize_scalar(g_t, "g_t")
            self.eta = self.eta - self.cfg.alpha * g_t
            self._ensure_finite_eta()
            if self.eta.item() < self.cfg.eta_min:
                self.eta.fill_(self.cfg.eta_min)

            # x update
            if anchor or self.v_prev is None:
                images_x, targets_x = self.stream_N3.next()
                _, grad_x_curr, _, _ = self._compute_gradients(self.model, images_x, targets_x, self.eta)
                v_t = self._add_noise_vector(grad_x_curr, self.cfg.sigma3)
            else:
                images_x, targets_x = self.stream_N4.next()
                _, grad_x_curr, _, _ = self._compute_gradients(self.model, images_x, targets_x, self.eta)
                _, grad_x_prev, _, _ = self._compute_gradients(self.prev_model, images_x, targets_x, self.prev_eta)
                delta_grad = [a - b for a, b in zip(grad_x_curr, grad_x_prev)]
                accum = [d + v for d, v in zip(delta_grad, self.v_prev)]
                v_t = self._add_noise_vector(accum, self.cfg.sigma4)

            v_t = self._sanitize_vector(v_t, "v_t")
            v_t, v_norm = self._clip_vector(v_t)
            beta_t = self._beta_step(v_norm)
            apply_update(self.model, v_t, lr=beta_t)
            self._ensure_model_finite()

            with torch.no_grad():
                logits = self.model(images_x)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                loss = F.cross_entropy(logits, targets_x, reduction="mean").item()
                acc = (logits.argmax(dim=1) == targets_x).float().mean().item() * 100.0
                if not math.isfinite(loss):
                    self.logger.warning("Non-finite training loss detected; substituting zero.")
                    loss = 0.0
            loss_meter.update(float(loss), n=targets_x.size(0))
            acc_meter.update(acc, n=targets_x.size(0))

            # prepare next iteration
            self.prev_model.load_state_dict(prev_state)
            self.g_prev = self._sanitize_scalar(g_t.detach().clone(), "g_prev")
            self.v_prev = self._sanitize_vector([v.detach().clone() for v in v_t], "v_prev")
            self.global_step += 1
            self._update_random_state(t)

            should_eval = (
                self.eval_loader is not None
                and self.cfg.eval_interval is not None
                and (self.global_step % self.cfg.eval_interval == 0)
                and self.global_step > 0
            )

            entry: Dict[str, float] = {
                "step": float(self.global_step),
                "batch_loss": float(loss),
                "train_loss_avg": loss_meter.avg,
                "batch_acc": float(acc),
                "train_acc_avg": acc_meter.avg,
                "eta": float(self.eta.item()),
                "g_abs": float(g_t.abs().item()),
                "v_norm": float(v_norm),
                "beta_t": float(beta_t),
            }
            if should_eval:
                val_stats = self.evaluate(self.eval_loader)  # type: ignore[arg-type]
                entry.update(val_stats)
                loss_meter.reset()
                acc_meter.reset()
            self.history.append(entry)
            self._flush_process_log()

        summary_metrics: Dict[str, Dict[str, float] | float] = {"steps": float(self.global_step)}
        if self.eval_loader is not None:
            train_eval = self.evaluate(self.eval_loader)  # type: ignore[arg-type]
            summary_metrics["train"] = {
                "loss": float(train_eval.get("val_loss", 0.0)),
                "accuracy": float(train_eval.get("val_acc", 0.0)),
            }
        if self.test_loader is not None:
            test_eval = self.evaluate(self.test_loader)  # type: ignore[arg-type]
            summary_metrics["test"] = {
                "loss": float(test_eval.get("val_loss", 0.0)),
                "accuracy": float(test_eval.get("val_acc", 0.0)),
            }
        self.final_metrics = summary_metrics
        self._flush_process_log()
        return self.history
        # Note: final metrics populated after training loop.

    def _beta_step(self, v_norm: float) -> float:
        if v_norm <= 0.0:
            return self.beta_cap_const
        adaptive = 1.0 / (self.cfg.L0 * math.sqrt(self.cfg.n) * v_norm)
        return min(self.beta_cap_const, adaptive)

    def _sanitize_scalar(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.isfinite(value):
            self.logger.warning("Non-finite scalar %s detected; reset to zero.", name)
            return torch.zeros_like(value)
        return value

    def _sanitize_vector(self, grads: Iterable[torch.Tensor], name: str) -> List[torch.Tensor]:
        sanitized: List[torch.Tensor] = []
        had_issue = False
        for idx, g in enumerate(grads):
            if not torch.isfinite(g).all():
                had_issue = True
                sanitized.append(torch.zeros_like(g))
            else:
                sanitized.append(g)
        if had_issue:
            self.logger.warning("Non-finite entries found in %s; zeroed.", name)
        return sanitized

    def _clip_vector(self, grads: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float]:
        total_norm = tensor_list_norm(grads)
        if self.cfg.grad_clip > 0 and total_norm > self.cfg.grad_clip:
            scale = self.cfg.grad_clip / (total_norm + 1e-12)
            for g in grads:
                g.mul_(scale)
            clipped_norm = tensor_list_norm(grads)
            self.logger.debug("Gradient norm clipped from %.6f to %.6f", total_norm, clipped_norm)
            return grads, clipped_norm
        return grads, total_norm

    def _ensure_finite_eta(self) -> None:
        if not torch.isfinite(self.eta):
            self.logger.warning("Dual variable eta became non-finite; resetting to eta_min.")
            self.eta.fill_(self.cfg.eta_min)

    def _ensure_model_finite(self) -> None:
        any_issue = False
        with torch.no_grad():
            for param in self.model.parameters():
                if not torch.isfinite(param).all():
                    any_issue = True
                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
        if any_issue:
            self.logger.warning("Model parameters contained non-finite values; applied nan_to_num sanitisation.")

    def _flush_process_log(self) -> None:
        if self.process_path is None:
            return
        try:
            self.process_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "history": self.history,
                "final_metrics": getattr(self, "final_metrics", {}),
            }
            with self.process_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            self.logger.error("Failed to write process log to %s: %s", self.process_path, exc)

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_val = F.cross_entropy(logits, targets, reduction="mean").item()
            if not math.isfinite(loss_val):
                self.logger.warning("Non-finite validation loss detected; substituting zero.")
                loss_val = 0.0
            acc = (logits.argmax(dim=1) == targets).float().mean().item() * 100.0
            loss_meter.update(float(loss_val), n=targets.size(0))
            acc_meter.update(acc, n=targets.size(0))
        self.model.train()
        return {"val_loss": loss_meter.avg, "val_acc": acc_meter.avg}

    def summary(self) -> Dict[str, object]:
        return {
            "steps": self.global_step,
            "eta": float(self.eta.item()),
            "random_state_step": None if self.random_state is None else self.random_state.get("step"),
            "final_metrics": self.final_metrics,
        }
