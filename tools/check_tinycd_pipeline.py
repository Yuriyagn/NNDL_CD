from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import LEVIRCDDataset
from experiment_utils import count_parameters, format_params, load_yaml, select_device
from losses import BCEDiceLoss
from metrics import ChangeDetectionMetrics
from models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TinyCD data flow without training.")
    parser.add_argument("--config", default="configs/tinycd.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def tensor_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach().float().cpu()
    return (
        f"{name}: shape={tuple(tensor.shape)} min={tensor.min().item():.4f} "
        f"max={tensor.max().item():.4f} mean={tensor.mean().item():.4f}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_name = cfg.get("model", "tinycd")
    if model_name != "tinycd":
        raise ValueError(f"This diagnostic is for TinyCD; config has model={model_name}")

    device = select_device(args.device)
    data_root = cfg.get("data_root", "data/LEVIR_CD_SUBSET_256")
    model_kwargs = cfg.get("model_kwargs", {})
    dataset = LEVIRCDDataset(data_root, split=args.split, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    image_a = batch["image_a"].to(device)
    image_b = batch["image_b"].to(device)
    mask = batch["mask"].to(device)

    model = build_model("tinycd", **model_kwargs).to(device)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"loaded_checkpoint={checkpoint_path}")
        print(f"checkpoint_best_threshold={checkpoint.get('best_threshold', 'missing')}")

    model.train()
    logits = model(image_a, image_b)
    probs = torch.sigmoid(logits)
    criterion = BCEDiceLoss(pos_weight=1.0)
    loss = criterion(logits, mask)
    loss.backward()

    grad_norm = 0.0
    grad_params = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().norm().cpu())
            grad_params += 1

    print(f"data_root={data_root}")
    print(f"split={args.split} dataset_size={len(dataset)}")
    print(f"names={list(batch['name'])}")
    print(tensor_stats("image_a", image_a))
    print(tensor_stats("image_b", image_b))
    print(tensor_stats("mask", mask))
    print(f"mask_unique={sorted(mask.detach().cpu().unique().tolist())}")
    print(f"mask_change_ratio={mask.float().mean().item():.4f}")
    print(f"params={format_params(count_parameters(model))}")
    print(tensor_stats("logits", logits))
    print(tensor_stats("sigmoid(logits)", probs))
    print(f"logits_shape_matches_mask={tuple(logits.shape) == tuple(mask.shape)}")
    print(f"loss_is_finite={torch.isfinite(loss).item()} loss={loss.item():.4f}")
    print(f"grad_params={grad_params} grad_norm_sum={grad_norm:.4f}")

    model.eval()
    metrics = ChangeDetectionMetrics(threshold=0.5)
    with torch.no_grad():
        metrics.update(torch.sigmoid(model(image_a, image_b)), mask)
    scores = metrics.compute()
    print(
        "batch_metrics_at_0.50="
        f"precision={scores['precision']:.4f} recall={scores['recall']:.4f} "
        f"f1={scores['f1']:.4f} iou={scores['iou']:.4f} oa={scores['oa']:.4f}"
    )


if __name__ == "__main__":
    main()
