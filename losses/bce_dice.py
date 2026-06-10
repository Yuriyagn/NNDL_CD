from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        d_loss = dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * d_loss
