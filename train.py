from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from datasets import LEVIRCDDataset
from experiment_utils import clone_default_config, deep_update, load_yaml, select_device, set_seed
from losses import BCEDiceLoss
from metrics import ChangeDetectionMetrics
from models import build_model


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "fc_ef",
    "run_name": None,
    "model_kwargs": {},
    "data_root": "data/LEVIR_CD_SUBSET_256",
    "output_dir": "results",
    "save_dir": None,
    "seed": 42,
    "train": {
        "epochs": 2,
        "batch_size": 2,
        "lr": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "loss": "bce_dice",
        "num_workers": 0,
        "pos_weight": "auto",
        "pos_weight_max": 20.0,
        "gradient_clip": 1.0,
        "scheduler": "cosine",
        "augment": True,
        "val_thresholds": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
        "max_train_batches": None,
        "max_val_batches": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a change detection model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset", default=None, help="Dataset name under data/ or an explicit data root.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-dir", default=None, help="Flat run directory for checkpoints, log.csv, and config.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adamw"], default=None)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default=None)
    parser.add_argument("--loss", choices=["bce_dice"], default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pos-weight", default=None, help="Float value, 'auto', or 'none'.")
    parser.add_argument("--pos-weight-max", type=float, default=None)
    parser.add_argument("--gradient-clip", type=float, default=None)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = deep_update(clone_default_config(DEFAULT_CONFIG), load_yaml(args.config))
    cfg["train"] = dict(DEFAULT_CONFIG["train"], **cfg.get("train", {}))
    cfg["model_kwargs"] = dict(DEFAULT_CONFIG["model_kwargs"], **cfg.get("model_kwargs", {}))

    if args.model is not None:
        cfg["model"] = args.model
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.dataset is not None:
        dataset_path = Path(args.dataset)
        cfg["data_root"] = str(dataset_path if dataset_path.exists() else Path("data") / args.dataset)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.base_channels is not None:
        cfg["model_kwargs"]["base_channels"] = args.base_channels

    for arg_name, cfg_name in [
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("lr", "lr"),
        ("weight_decay", "weight_decay"),
        ("optimizer", "optimizer"),
        ("scheduler", "scheduler"),
        ("loss", "loss"),
        ("num_workers", "num_workers"),
        ("pos_weight_max", "pos_weight_max"),
        ("gradient_clip", "gradient_clip"),
        ("max_train_batches", "max_train_batches"),
        ("max_val_batches", "max_val_batches"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            cfg["train"][cfg_name] = value
    if args.pos_weight is not None:
        if args.pos_weight.lower() == "none":
            cfg["train"]["pos_weight"] = None
        elif args.pos_weight.lower() == "auto":
            cfg["train"]["pos_weight"] = "auto"
        else:
            cfg["train"]["pos_weight"] = float(args.pos_weight)
    if args.no_augment:
        cfg["train"]["augment"] = False
    return cfg


def estimate_pos_weight(data_root: str | Path, split: str = "train", max_value: float = 20.0) -> float:
    label_dir = Path(data_root) / split / "label"
    positive = 0
    total = 0
    for path in sorted(label_dir.glob("*")):
        if not path.is_file():
            continue
        mask = np.asarray(Image.open(path).convert("L"))
        positive += int((mask > 127).sum())
        total += int(mask.size)
    negative = total - positive
    if positive == 0:
        return 1.0
    return min(float(negative / positive), float(max_value))


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    thresholds: list[float] | None = None,
    gradient_clip: float | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    thresholds = thresholds or [0.5]
    metrics_by_threshold = {threshold: ChangeDetectionMetrics(threshold=threshold) for threshold in thresholds}
    total_loss = 0.0
    total_batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        image_a = batch["image_a"].to(device)
        image_b = batch["image_b"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(image_a, image_b)
            loss = criterion(logits, mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip is not None and gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

        probs = torch.sigmoid(logits)
        for metric in metrics_by_threshold.values():
            metric.update(probs, mask)
        total_loss += float(loss.detach().cpu())
        total_batches += 1

    scored = {threshold: metric.compute() for threshold, metric in metrics_by_threshold.items()}
    best_threshold, scores = max(scored.items(), key=lambda item: item[1]["f1"])
    scores["loss"] = total_loss / max(total_batches, 1)
    scores["threshold"] = float(best_threshold)
    return scores


LOG_FIELDNAMES = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_iou",
    "val_oa",
    "val_threshold",
    "lr",
]


def init_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
        writer.writeheader()


def append_log(log_path: Path, row: Dict[str, float]) -> None:
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDNAMES)
        writer.writerow({key: row.get(key, "") for key in LOG_FIELDNAMES})


def save_curves(history: list[Dict[str, float]], output_dir: Path, model_name: str) -> None:
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    csv_path = curves_dir / f"{model_name}_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=LOG_FIELDNAMES,
        )
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in LOG_FIELDNAMES})

    cache_root = Path(".cache")
    cache_dir = cache_root / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val loss")
    plt.plot(epochs, [row["val_f1"] for row in history], label="val F1")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curves_dir / f"{model_name}_curve.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    if cfg["model"] == "difference_otsu":
        raise ValueError("Difference baseline does not need training. Use test.py instead.")

    set_seed(int(cfg["seed"]))
    device = select_device(args.device)
    train_cfg = cfg["train"]
    if train_cfg.get("optimizer", "adamw") != "adamw":
        raise ValueError("Only AdamW optimizer is supported.")
    if train_cfg.get("loss", "bce_dice") != "bce_dice":
        raise ValueError("Only BCE+Dice loss is supported.")

    train_set = LEVIRCDDataset(
        cfg["data_root"],
        split="train",
        augment=bool(train_cfg.get("augment", True)),
    )
    val_set = LEVIRCDDataset(cfg["data_root"], split="val", augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
    )

    model = build_model(cfg["model"], **cfg["model_kwargs"]).to(device)
    pos_weight = train_cfg["pos_weight"]
    if isinstance(pos_weight, str) and pos_weight.lower() == "auto":
        pos_weight = estimate_pos_weight(
            cfg["data_root"],
            split="train",
            max_value=float(train_cfg.get("pos_weight_max", 20.0)),
        )
        print(f"auto_pos_weight={pos_weight:.4f}")
    criterion = BCEDiceLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = None
    scheduler_name = train_cfg.get("scheduler")
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(train_cfg["epochs"]), 1),
        )
    elif scheduler_name not in (None, "none"):
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    output_dir = Path(cfg["output_dir"])
    run_name = cfg.get("run_name") or cfg["model"]
    checkpoint_dir = Path(cfg["save_dir"]) if cfg.get("save_dir") else output_dir / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "log.csv"
    init_log(log_path)
    with (checkpoint_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    with (checkpoint_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)

    best_f1 = -1.0
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_scores = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=train_cfg["max_train_batches"],
            thresholds=[0.5],
            gradient_clip=train_cfg.get("gradient_clip"),
        )
        val_scores = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            max_batches=train_cfg["max_val_batches"],
            thresholds=[float(t) for t in train_cfg.get("val_thresholds", [0.5])],
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_scores["loss"],
            "val_loss": val_scores["loss"],
            "val_precision": val_scores["precision"],
            "val_recall": val_scores["recall"],
            "val_f1": val_scores["f1"],
            "val_iou": val_scores["iou"],
            "val_oa": val_scores["oa"],
            "val_threshold": val_scores["threshold"],
            "lr": current_lr,
        }
        history.append(row)
        append_log(log_path, row)
        state = {
            "model": cfg["model"],
            "model_kwargs": cfg["model_kwargs"],
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_scores,
            "best_threshold": val_scores["threshold"],
            "best_val_f1": max(best_f1, val_scores["f1"]),
            "config": cfg,
        }
        torch.save(state, checkpoint_dir / "last.pt")
        torch.save(state, checkpoint_dir / "last.pth")
        if val_scores["f1"] > best_f1:
            best_f1 = val_scores["f1"]
            state["best_val_f1"] = best_f1
            torch.save(state, checkpoint_dir / "best.pt")
            torch.save(state, checkpoint_dir / "best_val_f1.pth")
        print(
            f"epoch={epoch} train_loss={train_scores['loss']:.4f} "
            f"val_loss={val_scores['loss']:.4f} val_f1={val_scores['f1']:.4f} "
            f"val_iou={val_scores['iou']:.4f} threshold={val_scores['threshold']:.2f}"
        )

    curve_root = checkpoint_dir if cfg.get("save_dir") else output_dir
    save_curves(history, curve_root, run_name)
    print(f"best_f1={best_f1:.4f}")
    print(f"checkpoint={checkpoint_dir / 'best.pt'}")
    print(f"log={log_path}")


if __name__ == "__main__":
    main()
