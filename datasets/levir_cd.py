from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (mask > 127).astype(np.float32)


def _to_chw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class LEVIRCDDataset(Dataset):
    """LEVIR-CD paired image dataset.

    Expected layout:
        root/split/A/*.png
        root/split/B/*.png
        root/split/label/*.png
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.split_dir = self.root / split

        self.a_dir = self.split_dir / "A"
        self.b_dir = self.split_dir / "B"
        self.label_dir = self.split_dir / "label"
        for directory in (self.a_dir, self.b_dir, self.label_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing dataset directory: {directory}")

        self.names = self._collect_names()
        if not self.names:
            raise RuntimeError(f"No samples found under {self.split_dir}")

    def _collect_names(self) -> List[str]:
        names = []
        for path in sorted(self.a_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            if not (self.b_dir / path.name).is_file():
                raise FileNotFoundError(f"Missing B image for {path.name}")
            if not (self.label_dir / path.name).is_file():
                raise FileNotFoundError(f"Missing label for {path.name}")
            names.append(path.name)
        return names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        name = self.names[index]
        image_a = _load_rgb(self.a_dir / name)
        image_b = _load_rgb(self.b_dir / name)
        mask = _load_mask(self.label_dir / name)

        if self.augment:
            image_a, image_b, mask = self._augment(image_a, image_b, mask)

        return {
            "image_a": _to_chw(image_a),
            "image_b": _to_chw(image_b),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None, :, :])),
            "name": name,
        }

    @staticmethod
    def _augment(
        image_a: np.ndarray,
        image_b: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image_a = np.flip(image_a, axis=1)
            image_b = np.flip(image_b, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() < 0.5:
            image_a = np.flip(image_a, axis=0)
            image_b = np.flip(image_b, axis=0)
            mask = np.flip(mask, axis=0)

        k = random.randint(0, 3)
        if k:
            image_a = np.rot90(image_a, k, axes=(0, 1))
            image_b = np.rot90(image_b, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))

        return image_a, image_b, mask
