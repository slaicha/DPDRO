from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
except ImportError as exc:
    raise ImportError("scikit-learn is required; install via `pip install scikit-learn`.") from exc


FPR_TARGETS = (0.01, 0.001)


def confidence_scores(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1).max(dim=1).values


def loss_scores(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    losses = F.nll_loss(log_probs, targets, reduction="none")
    return -losses


def logit_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(logits, k=2, dim=1)
    target_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    competitor = torch.where(top2.indices[:, 0] == targets, top2.values[:, 1], top2.values[:, 0])
    return target_logits - competitor


def _threshold_for_fpr(nonmember_scores: np.ndarray, target_fpr: float) -> float:
    if nonmember_scores.size == 0:
        return float("inf")
    sorted_scores = np.sort(nonmember_scores)
    cutoff_idx = max(int(math.ceil((1 - target_fpr) * len(sorted_scores))) - 1, 0)
    cutoff_idx = min(cutoff_idx, len(sorted_scores) - 1)
    return float(sorted_scores[cutoff_idx])


def calibrate_thresholds(nonmember_scores: np.ndarray, fpr_targets: Iterable[float] = FPR_TARGETS) -> Dict[str, float]:
    return {f"fpr_{int(fpr * 10000)}": _threshold_for_fpr(nonmember_scores, fpr) for fpr in fpr_targets}


def calibrate_thresholds_per_class(
    nonmember_scores: np.ndarray, nonmember_labels: np.ndarray, fpr_targets: Iterable[float] = FPR_TARGETS
) -> Dict[int, Dict[str, float]]:
    per_class: Dict[int, Dict[str, float]] = {}
    for cls in np.unique(nonmember_labels):
        per_class[int(cls)] = calibrate_thresholds(nonmember_scores[nonmember_labels == cls], fpr_targets)
    return per_class


def _confusion(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float]:
    tp = float(np.logical_and(preds == 1, labels == 1).sum())
    fp = float(np.logical_and(preds == 1, labels == 0).sum())
    tn = float(np.logical_and(preds == 0, labels == 0).sum())
    fn = float(np.logical_and(preds == 0, labels == 1).sum())
    return tp, fp, tn, fn


def _tpr_fpr_precision(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    tp, fp, tn, fn = _confusion(preds, labels)
    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    return tpr, fpr, precision


@dataclass
class AttackMetrics:
    auc: float
    tpr_at_1: float
    tpr_at_0_1: float
    precision_at_1: float
    precision_at_0_1: float
    thresholds: Mapping[str, float]
    per_class: Dict[int, Dict[str, float]] | None = None

    def to_json(self) -> Dict[str, object]:
        payload = {
            "auc": self.auc,
            "tpr_at_1%_fpr": self.tpr_at_1,
            "tpr_at_0.1%_fpr": self.tpr_at_0_1,
            "precision_at_1%_fpr": self.precision_at_1,
            "precision_at_0.1%_fpr": self.precision_at_0_1,
            "thresholds": dict(self.thresholds),
        }
        if self.per_class:
            payload["per_class"] = self.per_class
        return payload


def _apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.int64)


def _metrics_with_threshold(scores: np.ndarray, labels: np.ndarray, thresholds: Mapping[str, float]) -> Tuple[float, float, Dict[str, float]]:
    tpr_1 = precision_1 = tpr_0_1 = precision_0_1 = 0.0
    preds_1 = _apply_threshold(scores, thresholds.get("fpr_100", float("inf")))
    preds_0_1 = _apply_threshold(scores, thresholds.get("fpr_10", float("inf")))
    tpr_1, _, precision_1 = _tpr_fpr_precision(preds_1, labels)
    tpr_0_1, _, precision_0_1 = _tpr_fpr_precision(preds_0_1, labels)
    return tpr_1, tpr_0_1, {
        "precision_at_1%": precision_1,
        "precision_at_0.1%": precision_0_1,
    }


def evaluate_attack(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: Mapping[str, float],
    class_labels: np.ndarray | None = None,
    per_class_thresholds: Dict[int, Dict[str, float]] | None = None,
) -> AttackMetrics:
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    tpr_at_1 = float(np.interp(0.01, fpr, tpr))
    tpr_at_0_1 = float(np.interp(0.001, fpr, tpr))
    tpr_1_thr, tpr_0_1_thr, precisions = _metrics_with_threshold(scores, labels, thresholds)

    per_class_metrics: Dict[int, Dict[str, float]] | None = None
    if class_labels is not None and per_class_thresholds is not None:
        per_class_metrics = {}
        for cls, cls_thresholds in per_class_thresholds.items():
            mask = class_labels == cls
            if not np.any(mask):
                continue
            cls_auc = roc_auc_score(labels[mask], scores[mask]) if len(np.unique(labels[mask])) > 1 else float("nan")
            cls_tpr, cls_tpr_low, cls_precisions = _metrics_with_threshold(scores[mask], labels[mask], cls_thresholds)
            per_class_metrics[int(cls)] = {
                "auc": cls_auc,
                "tpr_at_1%_fpr": cls_tpr,
                "tpr_at_0.1%_fpr": cls_tpr_low,
                "precision_at_1%": cls_precisions["precision_at_1%"],
                "precision_at_0.1%": cls_precisions["precision_at_0.1%"],
                "thresholds": cls_thresholds,
            }

    return AttackMetrics(
        auc=float(auc),
        tpr_at_1=float(tpr_at_1),
        tpr_at_0_1=float(tpr_at_0_1),
        precision_at_1=float(precisions["precision_at_1%"]),
        precision_at_0_1=float(precisions["precision_at_0.1%"]),
        thresholds=thresholds,
        per_class=per_class_metrics,
    )


def gaussian_params(scores: np.ndarray) -> Tuple[float, float]:
    mu = float(scores.mean()) if scores.size else 0.0
    sigma = float(scores.std(ddof=1) + 1e-12) if scores.size else 1.0
    return mu, sigma


def gaussian_logpdf(x: float, mu: float, sigma: float) -> float:
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(math.sqrt(2 * math.pi) * sigma)


def lira_score(target_score: float, in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    mu_in, sigma_in = gaussian_params(in_scores)
    mu_out, sigma_out = gaussian_params(out_scores)
    return gaussian_logpdf(target_score, mu_in, sigma_in) - gaussian_logpdf(target_score, mu_out, sigma_out)


def lira_from_shadow_bags(
    target_scores: Sequence[float], shadow_scores: np.ndarray, shadow_memberships: np.ndarray
) -> np.ndarray:
    """Compute LiRA scores for each example given shadow statistics.

    Args:
        target_scores: sequence of logit margins from target model, one per example.
        shadow_scores: array with shape (num_examples, num_shadows) holding the same statistic
            measured on each shadow model.
        shadow_memberships: binary array with shape (num_examples, num_shadows) indicating
            whether each example was in the corresponding shadow's training set.
    """
    assert shadow_scores.shape == shadow_memberships.shape, "shadow score and membership shapes must match"
    lira_vals = []
    for i, s_star in enumerate(target_scores):
        in_scores = shadow_scores[i][shadow_memberships[i] == 1]
        out_scores = shadow_scores[i][shadow_memberships[i] == 0]
        lira_vals.append(lira_score(float(s_star), in_scores, out_scores))
    return np.asarray(lira_vals)
