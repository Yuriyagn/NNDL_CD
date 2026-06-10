from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def otsu_threshold(gray: np.ndarray) -> float:
    values = np.clip(gray, 0.0, 1.0)
    hist, bin_edges = np.histogram(values.ravel(), bins=256, range=(0.0, 1.0))
    total = values.size
    sum_total = np.dot(hist, np.arange(256))
    weight_bg = 0.0
    sum_bg = 0.0
    max_between = -1.0
    threshold = 0

    for idx in range(256):
        weight_bg += hist[idx]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += idx * hist[idx]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = idx

    return float(bin_edges[threshold])


class DifferenceBaseline:
    """Traditional absolute-difference baseline with Otsu or fixed threshold."""

    def __init__(
        self,
        threshold_mode: Literal["otsu", "fixed"] = "otsu",
        fixed_threshold: float = 0.2,
    ) -> None:
        self.threshold_mode = threshold_mode
        self.fixed_threshold = fixed_threshold

    @torch.no_grad()
    def predict_proba(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(image_b - image_a).mean(dim=1, keepdim=True)
        preds = []
        for item in diff.detach().cpu().numpy():
            gray = item[0]
            threshold = self.fixed_threshold
            if self.threshold_mode == "otsu":
                threshold = otsu_threshold(gray)
            preds.append((gray >= threshold).astype(np.float32)[None, :, :])
        return torch.from_numpy(np.stack(preds, axis=0))
