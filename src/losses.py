"""Loss functions: cross-entropy baseline + focal loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    `alpha` may be a scalar or a per-class tensor. gamma=0 degenerates to
    weighted cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | torch.Tensor | None = None,
                 reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.tensor(float(alpha))
        else:
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        if self.label_smoothing > 0:
            log_probs = (1 - self.label_smoothing) * log_probs \
                + self.label_smoothing / num_classes * (
                    1.0 - torch.exp(log_probs)).clamp_min(1e-9).log()
        gather = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = gather.exp()
        loss = -((1 - p_t) ** self.gamma) * gather
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets) \
                if self.alpha.numel() > 1 else self.alpha.to(logits.device)
            loss = loss * alpha_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(loss_name: str, class_weights: torch.Tensor | None = None,
                    gamma: float = 2.0, alpha: float | None = None,
                    label_smoothing: float = 0.0) -> nn.Module:
    """Factory used by train.py.

    loss_name: 'ce' | 'focal'
    class_weights: per-class CE weights (inverse frequency etc.)
    alpha: focal class weight (overrides class_weights for focal)
    """
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights,
                                   label_smoothing=label_smoothing)
    if loss_name == "focal":
        a = alpha if alpha is not None else class_weights
        return FocalLoss(gamma=gamma, alpha=a, label_smoothing=label_smoothing)
    raise ValueError(f"unknown loss {loss_name!r}")
