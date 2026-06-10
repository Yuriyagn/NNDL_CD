from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def tensor_image_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().clamp(0, 1).numpy()
    if array.ndim == 3:
        array = array.transpose(1, 2, 0)
    return (array * 255.0).round().astype(np.uint8)


def mask_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().squeeze().numpy()
    return ((array >= 0.5).astype(np.uint8) * 255)


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_bin = gt > 127
    pred_bin = pred > 127
    out = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    out[np.logical_and(~gt_bin, ~pred_bin)] = (30, 30, 30)
    out[np.logical_and(gt_bin, pred_bin)] = (40, 180, 80)
    out[np.logical_and(~gt_bin, pred_bin)] = (220, 60, 60)
    out[np.logical_and(gt_bin, ~pred_bin)] = (60, 120, 240)
    return out


def add_label(panel: Image.Image, label: str) -> Image.Image:
    header = 24
    out = Image.new("RGB", (panel.width, panel.height + header), (255, 255, 255))
    out.paste(panel, (0, header))
    draw = ImageDraw.Draw(out)
    draw.text((6, 5), label, fill=(0, 0, 0))
    return out


def save_prediction_panel(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    mask: torch.Tensor,
    prob: torch.Tensor,
    output_path: str | Path,
) -> None:
    image_a_np = tensor_image_to_uint8(image_a)
    image_b_np = tensor_image_to_uint8(image_b)
    gt_np = mask_to_uint8(mask)
    pred_np = mask_to_uint8(prob)
    err_np = error_map(gt_np, pred_np)

    panels = [
        add_label(Image.fromarray(image_a_np), "T1"),
        add_label(Image.fromarray(image_b_np), "T2"),
        add_label(Image.fromarray(gt_np).convert("RGB"), "GT"),
        add_label(Image.fromarray(pred_np).convert("RGB"), "Prediction"),
        add_label(Image.fromarray(err_np), "Error Map"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a side-by-side prediction panel.")
    parser.add_argument("--help-only", action="store_true", help="This module is normally called from test.py.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    print("Use save_prediction_panel from test.py to create prediction visualizations.")


if __name__ == "__main__":
    main()
