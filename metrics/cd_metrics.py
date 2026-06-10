from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


EPS = 1e-8


def confusion_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, int]:
    preds = probs >= threshold
    labels = targets >= 0.5
    tp = torch.logical_and(preds, labels).sum().item()
    tn = torch.logical_and(~preds, ~labels).sum().item()
    fp = torch.logical_and(preds, ~labels).sum().item()
    fn = torch.logical_and(~preds, labels).sum().item()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def scores_from_confusion(confusion: Dict[str, int]) -> Dict[str, float]:
    tp = float(confusion.get("tp", 0))
    tn = float(confusion.get("tn", 0))
    fp = float(confusion.get("fp", 0))
    fn = float(confusion.get("fn", 0))
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)
    oa = (tp + tn) / (tp + tn + fp + fn + EPS)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "oa": oa,
    }


@dataclass
class ChangeDetectionMetrics:
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    @torch.no_grad()
    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        batch_confusion = confusion_from_probs(
            probs.detach().cpu(),
            targets.detach().cpu(),
            threshold=self.threshold,
        )
        for key, value in batch_confusion.items():
            self.confusion[key] += value

    def compute(self) -> Dict[str, float]:
        scores = scores_from_confusion(self.confusion)
        return {**self.confusion, **scores}
