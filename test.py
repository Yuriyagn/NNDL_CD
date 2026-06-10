from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import LEVIRCDDataset
from experiment_utils import (
    count_parameters,
    format_params,
    load_yaml,
    metadata_for,
    select_device,
    write_metrics_summary,
)
from metrics import ChangeDetectionMetrics
from models import build_model
from models.difference import DifferenceBaseline
from tools.visualize_results import save_prediction_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a change detection model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default="data/LEVIR_CD_SUBSET_256")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", default=None, help="Float threshold, or 'val_best' to use the checkpoint threshold.")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-visuals", type=int, default=8)
    parser.add_argument("--save-masks", action="store_true", help="Also save binary prediction masks.")
    parser.add_argument("--no-summary", action="store_true", help="Do not update metrics_summary files.")
    parser.add_argument("--save-json", default=None, help="Write evaluation metrics to this JSON file.")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def load_neural_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[str, torch.nn.Module, Dict[str, Any], Dict[str, Any] | None, Path | None]:
    cfg = load_yaml(args.config)
    checkpoint = None
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "config" in checkpoint:
            cfg = checkpoint["config"]

    model_name = args.model or cfg.get("model")
    if not model_name and checkpoint:
        model_name = checkpoint.get("model")
    if not model_name:
        raise ValueError("Provide --model, --config, or --checkpoint.")

    model_kwargs = cfg.get("model_kwargs", {})
    if checkpoint and checkpoint.get("model_kwargs"):
        model_kwargs = checkpoint["model_kwargs"]
    model = build_model(model_name, **model_kwargs).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model_name, model, model_kwargs, checkpoint, checkpoint_path


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = select_device(args.device)
    dataset = LEVIRCDDataset(args.data_root, split=args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint = None
    checkpoint_path = None
    if args.model == "difference_otsu":
        model_name = "difference_otsu"
        model = DifferenceBaseline()
        params: int | str = "-"
    else:
        model_name, model, _, checkpoint, checkpoint_path = load_neural_model(args, device)
        params = count_parameters(model)

    threshold_arg = args.threshold
    if threshold_arg is None or str(threshold_arg).lower() == "val_best":
        threshold = 0.5
        if checkpoint is not None and checkpoint.get("best_threshold") is not None:
            threshold = float(checkpoint["best_threshold"])
    else:
        threshold = float(threshold_arg)

    metrics = ChangeDetectionMetrics(threshold=float(threshold))
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions" / model_name
    mask_dir = pred_dir / "masks"
    visual_count = 0

    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        image_a = batch["image_a"].to(device)
        image_b = batch["image_b"].to(device)
        mask = batch["mask"].to(device)

        if model_name == "difference_otsu":
            probs = model.predict_proba(image_a.cpu(), image_b.cpu()).to(device)
        else:
            logits = model(image_a, image_b)
            probs = torch.sigmoid(logits)
        metrics.update(probs, mask)

        for item_idx, name in enumerate(batch["name"]):
            if visual_count >= args.max_visuals:
                break
            save_prediction_panel(
                batch["image_a"][item_idx],
                batch["image_b"][item_idx],
                batch["mask"][item_idx],
                probs[item_idx].detach().cpu(),
                pred_dir / f"{Path(name).stem}.png",
            )
            visual_count += 1
        if args.save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)
            for item_idx, name in enumerate(batch["name"]):
                binary = (probs[item_idx].detach().cpu().squeeze().numpy() >= threshold).astype("uint8")
                Image.fromarray(binary * 255).save(mask_dir / f"{Path(name).stem}.png")

    scores = metrics.compute()
    meta = metadata_for(model_name)
    row = {
        "Model": meta["display"],
        "Year": meta["year"],
        "Params": format_params(params),
        "Precision": scores["precision"],
        "Recall": scores["recall"],
        "F1": scores["f1"],
        "IoU": scores["iou"],
        "OA": scores["oa"],
    }
    if not args.no_summary:
        write_metrics_summary(output_dir / "tables", row)
    print(
        f"model={row['Model']} data_root={args.data_root} split={args.split} params={row['Params']} "
        f"precision={scores['precision']:.4f} recall={scores['recall']:.4f} "
        f"f1={scores['f1']:.4f} iou={scores['iou']:.4f} oa={scores['oa']:.4f} "
        f"threshold={threshold:.2f}"
    )
    if checkpoint_path is not None:
        print(f"checkpoint={checkpoint_path}")
    if args.no_summary:
        print("summary=skipped")
    else:
        print(f"summary={output_dir / 'tables' / 'metrics_summary.md'}")
    print(f"predictions={pred_dir}")
    result = {
        **scores,
        **row,
        "model": model_name,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "data_root": str(args.data_root),
        "split": args.split,
        "threshold": float(threshold),
    }
    if args.save_json:
        save_json_path = Path(args.save_json)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        with save_json_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"json={save_json_path}")
    return result


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
