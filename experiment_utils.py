from __future__ import annotations

import csv
import copy
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


MODEL_METADATA = {
    "difference_otsu": {"display": "Difference + Otsu", "year": "-"},
    "fc_ef": {"display": "FC-EF", "year": "2018"},
    "fc_siam_diff": {"display": "FC-Siam-Diff", "year": "2018"},
    "stanet": {"display": "STANet", "year": "2020"},
    "snunet_cd": {"display": "SNUNet-CD", "year": "2021"},
    "tinycd": {"display": "TinyCD", "year": "2022/2023"},
}

SUMMARY_ORDER = [
    "Difference + Otsu",
    "FC-EF",
    "FC-Siam-Diff",
    "STANet",
    "SNUNet-CD",
    "TinyCD",
]

LEGACY_SUMMARY_NAMES = {
    "STANet-lite",
    "SNUNet-CD-lite",
    "TinyCD-lite",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def clone_default_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(config)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(params: int | str) -> str:
    if isinstance(params, str):
        return params
    if params >= 1_000_000:
        return f"{params / 1_000_000:.2f}M"
    if params >= 1_000:
        return f"{params / 1_000:.1f}K"
    return str(params)


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def metadata_for(model_name: str) -> Dict[str, str]:
    return MODEL_METADATA.get(model_name, {"display": model_name, "year": ""})


def write_metrics_summary(
    table_dir: str | Path,
    row: Dict[str, str | float],
) -> None:
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "metrics_summary.csv"
    md_path = table_dir / "metrics_summary.md"
    columns = ["Model", "Year", "Params", "Precision", "Recall", "F1", "IoU", "OA"]

    rows: Dict[str, Dict[str, str]] = {}
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for old_row in csv.DictReader(handle):
                if old_row["Model"] in LEGACY_SUMMARY_NAMES:
                    continue
                rows[old_row["Model"]] = old_row

    formatted = {}
    for column in columns:
        value = row.get(column, "")
        if isinstance(value, float):
            formatted[column] = f"{value:.4f}"
        else:
            formatted[column] = str(value)
    rows[formatted["Model"]] = formatted

    ordered = []
    for name in SUMMARY_ORDER:
        if name in rows:
            ordered.append(rows.pop(name))
    ordered.extend(rows.values())

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ordered)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| Model | Year | Params | Precision | Recall | F1 | IoU | OA |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for item in ordered:
            handle.write(
                "| {Model} | {Year} | {Params} | {Precision} | {Recall} | "
                "{F1} | {IoU} | {OA} |\n".format(**item)
            )
