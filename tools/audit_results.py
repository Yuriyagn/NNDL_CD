from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import LEVIRCDDataset
from experiment_utils import count_parameters, format_params, metadata_for, select_device
from metrics import ChangeDetectionMetrics
from models import build_model
from models.difference import DifferenceBaseline


CHECKPOINTS = {
    "fc_ef": "results/checkpoints/fc_ef/best.pt",
    "fc_siam_diff": "results/checkpoints/fc_siam_diff/best.pt",
    "stanet": "results/checkpoints/stanet/best.pt",
    "snunet_cd": "results/checkpoints/snunet_cd/best.pt",
    "tinycd": "results/checkpoints/tinycd/best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit current change-detection results.")
    parser.add_argument("--data-root", default="data/LEVIR_CD_SUBSET_256")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/tables")
    return parser.parse_args()


def raw_label_audit(data_root: Path) -> list[Dict[str, Any]]:
    rows = []
    for split in ["train", "val", "test"]:
        label_dir = data_root / split / "label"
        values = set()
        positive = 0
        total = 0
        count = 0
        for path in sorted(label_dir.glob("*")):
            if not path.is_file():
                continue
            array = np.asarray(Image.open(path).convert("L"))
            values.update(np.unique(array).tolist())
            positive += int((array > 127).sum())
            total += int(array.size)
            count += 1
        rows.append(
            {
                "split": split,
                "count": count,
                "raw_values": sorted(values),
                "converted_values": [0.0, 1.0],
                "change_ratio": positive / max(total, 1),
            }
        )
    return rows


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[str, torch.nn.Module, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model"]
    model = build_model(model_name, **checkpoint.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    params = count_parameters(model)
    stored_threshold = float(checkpoint.get("best_threshold", 0.5))
    return model_name, model, params, stored_threshold


@torch.no_grad()
def evaluate_model(
    model_name: str,
    model: Any,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, Any]:
    metrics = ChangeDetectionMetrics(threshold=threshold)
    sample_ratios: list[float] = []
    pred_sizes_ok = True
    gt_sizes_ok = True

    for batch in loader:
        image_a = batch["image_a"].to(device)
        image_b = batch["image_b"].to(device)
        mask = batch["mask"].to(device)
        if model_name == "difference_otsu":
            probs = model.predict_proba(image_a.cpu(), image_b.cpu()).to(device)
        else:
            probs = torch.sigmoid(model(image_a, image_b))

        if probs.shape[-2:] != mask.shape[-2:]:
            pred_sizes_ok = False
        if image_a.shape[-2:] != mask.shape[-2:] or image_b.shape[-2:] != mask.shape[-2:]:
            gt_sizes_ok = False

        preds = (probs >= threshold).float()
        sample_ratios.extend(preds.flatten(1).mean(dim=1).detach().cpu().tolist())
        metrics.update(probs, mask)

    scores = metrics.compute()
    ratios = np.asarray(sample_ratios, dtype=np.float64)
    return {
        **scores,
        "pred_positive_ratio_mean": float(ratios.mean()) if ratios.size else 0.0,
        "pred_positive_ratio_min": float(ratios.min()) if ratios.size else 0.0,
        "pred_positive_ratio_max": float(ratios.max()) if ratios.size else 0.0,
        "all_black_samples": int((ratios <= 0.0).sum()) if ratios.size else 0,
        "all_white_samples": int((ratios >= 1.0).sum()) if ratios.size else 0,
        "pred_size_ok": pred_sizes_ok,
        "image_label_size_ok": gt_sizes_ok,
    }


def saved_mask_audit(pred_root: Path, model_name: str) -> Dict[str, Any]:
    mask_dir = pred_root / model_name / "masks"
    if not mask_dir.is_dir():
        return {"saved_mask_count": 0, "saved_mask_note": "missing masks dir"}
    ratios = []
    sizes = set()
    for path in sorted(mask_dir.glob("*.png")):
        array = np.asarray(Image.open(path).convert("L"))
        sizes.add(array.shape)
        ratios.append(float((array > 127).mean()))
    ratios_array = np.asarray(ratios, dtype=np.float64)
    return {
        "saved_mask_count": len(ratios),
        "saved_mask_sizes": sorted(sizes),
        "saved_mask_ratio_mean": float(ratios_array.mean()) if ratios_array.size else 0.0,
        "saved_mask_all_black": int((ratios_array <= 0.0).sum()) if ratios_array.size else 0,
        "saved_mask_all_white": int((ratios_array >= 1.0).sum()) if ratios_array.size else 0,
    }


def write_csv(path: Path, rows: list[Dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    label_rows: list[Dict[str, Any]],
    metric_rows: list[Dict[str, Any]],
    threshold: float,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Reliability Audit\n\n")
        handle.write(f"Fixed evaluation threshold: `{threshold:.2f}` after sigmoid.\n\n")
        handle.write("## Label Check\n\n")
        handle.write("| Split | Count | Raw Values | Converted Values | Change Ratio |\n")
        handle.write("|---|---:|---|---|---:|\n")
        for row in label_rows:
            handle.write(
                f"| {row['split']} | {row['count']} | {row['raw_values']} | "
                f"{row['converted_values']} | {row['change_ratio']:.4f} |\n"
            )
        handle.write("\n")
        handle.write("Dataset loader converts labels with `mask > 127`, so raw `0/255` labels become `0/1` masks.\n\n")

        handle.write("## Fixed-Threshold Metrics And Prediction Sanity\n\n")
        handle.write(
            "| Model | Params | Stored Val Threshold | Precision | Recall | F1 | IoU | OA | "
            "Pred Ratio | All Black | All White | Size OK |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in metric_rows:
            handle.write(
                f"| {row['Model']} | {row['Params']} | {row['StoredThreshold']:.2f} | "
                f"{row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | "
                f"{row['IoU']:.4f} | {row['OA']:.4f} | {row['PredRatio']:.4f} | "
                f"{row['AllBlack']} | {row['AllWhite']} | {row['SizeOK']} |\n"
            )
        handle.write("\n")
        handle.write("F1 and IoU above are computed from TP/FP/FN for the changed class only; OA additionally uses TN.\n")
        handle.write("`All Black` and `All White` count per-sample binary predictions at the fixed threshold.\n")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    dataset = LEVIRCDDataset(data_root, split=args.split, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    label_rows = raw_label_audit(data_root)
    metric_rows = []

    models: list[tuple[str, Any, str, float]] = [
        ("difference_otsu", DifferenceBaseline(), "-", args.threshold)
    ]
    for _, checkpoint in CHECKPOINTS.items():
        model_name, model, params, stored_threshold = load_model(Path(checkpoint), device)
        models.append((model_name, model, format_params(params), stored_threshold))

    for model_name, model, params, stored_threshold in models:
        scores = evaluate_model(model_name, model, loader, device, args.threshold)
        mask_info = saved_mask_audit(Path("results/predictions"), model_name)
        meta = metadata_for(model_name)
        metric_rows.append(
            {
                "Model": meta["display"],
                "Params": params,
                "StoredThreshold": stored_threshold,
                "Precision": scores["precision"],
                "Recall": scores["recall"],
                "F1": scores["f1"],
                "IoU": scores["iou"],
                "OA": scores["oa"],
                "PredRatio": scores["pred_positive_ratio_mean"],
                "PredRatioMin": scores["pred_positive_ratio_min"],
                "PredRatioMax": scores["pred_positive_ratio_max"],
                "AllBlack": scores["all_black_samples"],
                "AllWhite": scores["all_white_samples"],
                "SizeOK": scores["pred_size_ok"] and scores["image_label_size_ok"],
                **mask_info,
            }
        )

    metric_columns = [
        "Model",
        "Params",
        "StoredThreshold",
        "Precision",
        "Recall",
        "F1",
        "IoU",
        "OA",
        "PredRatio",
        "PredRatioMin",
        "PredRatioMax",
        "AllBlack",
        "AllWhite",
        "SizeOK",
        "saved_mask_count",
        "saved_mask_sizes",
        "saved_mask_ratio_mean",
        "saved_mask_all_black",
        "saved_mask_all_white",
    ]
    write_csv(output_dir / f"metrics_fixed_threshold_{args.threshold:.2f}.csv", metric_rows, metric_columns)
    write_markdown(output_dir / "reliability_audit.md", label_rows, metric_rows, args.threshold)
    print(f"audit_report={output_dir / 'reliability_audit.md'}")
    print(f"fixed_threshold_csv={output_dir / f'metrics_fixed_threshold_{args.threshold:.2f}.csv'}")


if __name__ == "__main__":
    main()
