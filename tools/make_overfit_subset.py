from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def change_ratio(label_path: Path) -> float:
    mask = np.asarray(Image.open(label_path).convert("L"))
    return float((mask > 127).mean())


def select_samples(source_root: Path, source_split: str, num_samples: int) -> list[tuple[str, float]]:
    label_dir = source_root / source_split / "label"
    rows = []
    for path in sorted(label_dir.glob("*.png")):
        rows.append((path.name, change_ratio(path)))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:num_samples]


def copy_sample(source_root: Path, source_split: str, dst_root: Path, dst_split: str, name: str) -> None:
    for folder in ["A", "B", "label"]:
        src = source_root / source_split / folder / name
        dst = dst_root / dst_split / folder / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny overfit subset for model diagnostics.")
    parser.add_argument("--source-root", default="data/LEVIR_CD_SUBSET_256")
    parser.add_argument("--source-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--dst", default="data/LEVIR_CD_TINYCD_OVERFIT_256")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    dst_root = Path(args.dst)
    if args.overwrite and dst_root.exists():
        shutil.rmtree(dst_root)

    selected = select_samples(source_root, args.source_split, args.num_samples)
    if not selected:
        raise RuntimeError(f"No samples found in {source_root / args.source_split / 'label'}")

    manifest_rows = []
    for dst_split in ["train", "val", "test"]:
        for name, ratio in selected:
            copy_sample(source_root, args.source_split, dst_root, dst_split, name)
            manifest_rows.append(
                {
                    "split": dst_split,
                    "source_split": args.source_split,
                    "name": name,
                    "change_ratio": f"{ratio:.6f}",
                }
            )

    manifest = dst_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "source_split", "name", "change_ratio"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"created={dst_root}")
    print(f"samples_per_split={len(selected)}")
    print("top_samples:")
    for name, ratio in selected[:10]:
        print(f"{name} change_ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
