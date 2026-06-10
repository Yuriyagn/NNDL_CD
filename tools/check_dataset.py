from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def inspect_split(root: Path, split: str) -> Dict[str, object]:
    split_dir = root / split
    dirs = {name: split_dir / name for name in ["A", "B", "label"]}
    for name, directory in dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {split}/{name}: {directory}")

    names = sorted(
        path.name for path in dirs["A"].iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    if not names:
        raise RuntimeError(f"No images found in {dirs['A']}")

    missing = []
    sizes = set()
    label_values = set()
    changed_pixels = 0
    total_pixels = 0

    for name in names:
        paths = {kind: directory / name for kind, directory in dirs.items()}
        for kind, path in paths.items():
            if not path.is_file():
                missing.append(f"{kind}/{name}")
        if missing:
            continue

        with (
            Image.open(paths["A"]) as image_a,
            Image.open(paths["B"]) as image_b,
            Image.open(paths["label"]).convert("L") as label,
        ):
            if image_a.size != image_b.size or image_a.size != label.size:
                raise ValueError(f"Size mismatch for {split}/{name}")
            sizes.add(image_a.size)
            label_array = np.asarray(label)
            label_values.update(np.unique(label_array).tolist())
            changed_pixels += int((label_array > 127).sum())
            total_pixels += int(label_array.size)

    if missing:
        raise FileNotFoundError(f"Missing paired files: {missing[:10]}")

    return {
        "split": split,
        "count": len(names),
        "sizes": sorted(sizes),
        "label_values": sorted(label_values),
        "change_ratio": changed_pixels / max(total_pixels, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LEVIR-CD subset integrity.")
    parser.add_argument("--root", default="data/LEVIR_CD_SUBSET_256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    for split in ["train", "val", "test"]:
        info = inspect_split(root, split)
        print(
            f"{info['split']}: count={info['count']} sizes={info['sizes']} "
            f"label_values={info['label_values']} change_ratio={info['change_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
