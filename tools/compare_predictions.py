from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import LEVIRCDDataset
from experiment_utils import metadata_for, select_device
from models import build_model
from models.difference import DifferenceBaseline
from tools.visualize_results import mask_to_uint8, tensor_image_to_uint8


def add_label(image: Image.Image, label: str) -> Image.Image:
    header = 24
    out = Image.new("RGB", (image.width, image.height + header), (255, 255, 255))
    out.paste(image.convert("RGB"), (0, header))
    draw = ImageDraw.Draw(out)
    draw.text((6, 5), label, fill=(0, 0, 0))
    return out


def save_row(panels: list[tuple[str, Image.Image]], output_path: Path) -> None:
    labeled = [add_label(image, label) for label, image in panels]
    width = sum(panel.width for panel in labeled)
    height = max(panel.height for panel in labeled)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in labeled:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def load_checkpoint_model(path: Path, device: torch.device) -> tuple[str, torch.nn.Module, float]:
    checkpoint = torch.load(path, map_location=device)
    model_name = checkpoint["model"]
    model = build_model(model_name, **checkpoint.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    threshold = float(checkpoint.get("best_threshold", 0.5))
    return model_name, model, threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create comparison panels across models.")
    parser.add_argument("--data-root", default="data/LEVIR_CD_SUBSET_256")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--include-difference", action="store_true")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--selection", choices=["top-change", "first"], default="top-change")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/predictions/comparison")
    return parser.parse_args()


def select_indices(dataset: LEVIRCDDataset, max_samples: int, selection: str) -> list[int]:
    if selection == "first":
        return list(range(min(max_samples, len(dataset))))
    scored = []
    for index in range(len(dataset)):
        item = dataset[index]
        ratio = float(item["mask"].float().mean().item())
        scored.append((ratio, index))
    scored.sort(reverse=True)
    return [index for _, index in scored[:max_samples]]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dataset = LEVIRCDDataset(args.data_root, split=args.split, augment=False)
    models: list[tuple[str, object, float]] = []
    if args.include_difference:
        models.append(("difference_otsu", DifferenceBaseline(), 0.5))
    for checkpoint_path in args.checkpoints:
        models.append(load_checkpoint_model(Path(checkpoint_path), device))

    output_dir = Path(args.output_dir)
    for index in select_indices(dataset, args.max_samples, args.selection):
        item = dataset[index]
        image_a = item["image_a"].unsqueeze(0).to(device)
        image_b = item["image_b"].unsqueeze(0).to(device)
        panels = [
            ("T1", Image.fromarray(tensor_image_to_uint8(item["image_a"]))),
            ("T2", Image.fromarray(tensor_image_to_uint8(item["image_b"]))),
            ("GT", Image.fromarray(mask_to_uint8(item["mask"])).convert("RGB")),
        ]
        for model_name, model, threshold in models:
            if model_name == "difference_otsu":
                probs = model.predict_proba(image_a.cpu(), image_b.cpu())
            else:
                probs = torch.sigmoid(model(image_a, image_b)).cpu()
            pred = (probs.squeeze().numpy() >= threshold).astype("uint8") * 255
            label = metadata_for(model_name)["display"]
            panels.append((label, Image.fromarray(pred).convert("RGB")))
        save_row(panels, output_dir / f"{Path(item['name']).stem}.png")
    print(f"comparison_dir={output_dir}")


if __name__ == "__main__":
    main()
