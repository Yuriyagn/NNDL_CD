from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image
import numpy as np


DEFAULT_WINDOWS_ROOT = Path("H:/Code/Data/LEVIR-CD")
DEFAULT_WSL_ROOT = Path("/mnt/h/Code/Data/LEVIR-CD")
DEFAULT_DST = Path("data/LEVIR_CD_SUBSET_256")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.stem)
    key: List[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    key.append(path.suffix)
    return key


def resolve_source_root(src: str | None) -> Path:
    candidates = []
    if src:
        candidates.append(Path(src))
    candidates.extend([DEFAULT_WSL_ROOT, DEFAULT_WINDOWS_ROOT])

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    joined = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find LEVIR-CD root. Checked: {joined}")


def list_split_images(src_root: Path, split: str, limit: int | None) -> List[Path]:
    a_dir = src_root / split / "A"
    if not a_dir.is_dir():
        raise FileNotFoundError(f"Missing source directory: {a_dir}")

    images = [
        path
        for path in a_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    images = sorted(images, key=natural_key)
    return images[:limit] if limit else images


def crop_positions(width: int, height: int, patch_size: int, stride: int) -> Iterable[tuple[int, int]]:
    if width < patch_size or height < patch_size:
        raise ValueError(f"Image size {width}x{height} is smaller than patch {patch_size}")
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            yield x, y


def ensure_output_dirs(dst_root: Path, split: str) -> Dict[str, Path]:
    dirs = {
        "A": dst_root / split / "A",
        "B": dst_root / split / "B",
        "label": dst_root / split / "label",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def crop_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    limit: int | None,
    patch_size: int,
    stride: int,
    min_change_ratio: float,
    keep_empty_ratio: float,
) -> List[Dict[str, object]]:
    out_dirs = ensure_output_dirs(dst_root, split)
    rows: List[Dict[str, object]] = []
    images = list_split_images(src_root, split, limit)

    for a_path in images:
        b_path = src_root / split / "B" / a_path.name
        label_path = src_root / split / "label" / a_path.name
        if not b_path.is_file():
            raise FileNotFoundError(f"Missing B image: {b_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label image: {label_path}")

        with (
            Image.open(a_path).convert("RGB") as image_a,
            Image.open(b_path).convert("RGB") as image_b,
            Image.open(label_path).convert("L") as label,
        ):
            if image_a.size != image_b.size or image_a.size != label.size:
                raise ValueError(f"Size mismatch for sample {a_path.name}")
            width, height = image_a.size

            for x, y in crop_positions(width, height, patch_size, stride):
                suffix = f"y{y:04d}_x{x:04d}{a_path.suffix.lower()}"
                out_name = f"{a_path.stem}_{suffix}"
                box = (x, y, x + patch_size, y + patch_size)
                label_patch = label.crop(box)
                change_ratio = float((np.asarray(label_patch) > 127).mean())
                if change_ratio < min_change_ratio and random.random() > keep_empty_ratio:
                    continue
                image_a.crop(box).save(out_dirs["A"] / out_name)
                image_b.crop(box).save(out_dirs["B"] / out_name)
                label_patch.save(out_dirs["label"] / out_name)
                rows.append(
                    {
                        "split": split,
                        "source": a_path.name,
                        "patch": out_name,
                        "x": x,
                        "y": y,
                        "change_ratio": f"{change_ratio:.6f}",
                    }
                )
    return rows


def write_manifest(dst_root: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    manifest = dst_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "source", "patch", "x", "y", "change_ratio"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop LEVIR-CD into 256x256 patches.")
    parser.add_argument("--src", default=None, help="Original LEVIR-CD root.")
    parser.add_argument("--dst", default=str(DEFAULT_DST), help="Output subset root.")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=10)
    parser.add_argument("--val-limit", type=int, default=5)
    parser.add_argument("--test-limit", type=int, default=5)
    parser.add_argument(
        "--min-change-ratio",
        type=float,
        default=0.0,
        help="Patch is considered positive when changed-pixel ratio reaches this value.",
    )
    parser.add_argument(
        "--keep-empty-ratio",
        type=float,
        default=1.0,
        help="Probability of keeping patches below --min-change-ratio.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Remove output subset before cropping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    src_root = resolve_source_root(args.src)
    dst_root = Path(args.dst)

    if args.overwrite and dst_root.exists():
        shutil.rmtree(dst_root)

    limits = {
        "train": args.train_limit,
        "val": args.val_limit,
        "test": args.test_limit,
    }
    rows: List[Dict[str, object]] = []
    for split, limit in limits.items():
        rows.extend(
            crop_split(
                src_root,
                dst_root,
                split,
                limit,
                args.patch_size,
                args.stride,
                args.min_change_ratio,
                args.keep_empty_ratio,
            )
        )

    write_manifest(dst_root, rows)
    print(f"Source: {src_root}")
    print(f"Output: {dst_root}")
    for split in limits:
        count = len(list((dst_root / split / "A").glob("*")))
        print(f"{split}: {count} patches")


if __name__ == "__main__":
    main()
