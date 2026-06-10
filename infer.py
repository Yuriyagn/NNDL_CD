from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiment_utils import select_device
from models import build_model
from models.difference import DifferenceBaseline


def load_image(path: str | Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image pair.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image-a", required=True)
    parser.add_argument("--image-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    image_a = load_image(args.image_a).to(device)
    image_b = load_image(args.image_b).to(device)

    if args.model == "difference_otsu":
        probs = DifferenceBaseline().predict_proba(image_a.cpu(), image_b.cpu())
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for neural models.")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_name = checkpoint.get("model", args.model)
        model_kwargs = checkpoint.get("model_kwargs", {})
        model = build_model(model_name, **model_kwargs).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        probs = torch.sigmoid(model(image_a, image_b)).cpu()

    pred = (probs.squeeze().numpy() >= 0.5).astype(np.uint8) * 255
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(output)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
