"""Loss functions and class imbalance utilities."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt).pow(self.gamma) * ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("weight", weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / max(num_classes - 1, 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(true_dist * log_probs).sum(dim=1)
        if self.weight is not None:
            loss = loss * self.weight[target]
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def make_class_weights(class_counts: Sequence[int], device: torch.device) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float32, device=device).clamp_min(1.0)
    weights = counts.sum() / (counts.numel() * counts)
    return weights


def build_loss(config: dict, class_weights: torch.Tensor | None = None) -> nn.Module:
    loss_cfg = config.get("loss", {})
    loss_type = str(loss_cfg.get("loss_type", "cross_entropy")).lower()
    if loss_type in {"ce", "cross_entropy", "crossentropy"}:
        label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        try:
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        except TypeError:
            if label_smoothing > 0:
                return LabelSmoothingCrossEntropy(smoothing=label_smoothing, weight=class_weights)
            return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type in {"focal", "focal_loss"}:
        return FocalLoss(gamma=float(loss_cfg.get("gamma", 2.0)), weight=class_weights)
    if loss_type in {"label_smoothing", "label_smoothing_ce", "smooth_ce"}:
        return LabelSmoothingCrossEntropy(
            smoothing=float(loss_cfg.get("label_smoothing", 0.1)),
            weight=class_weights,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")

