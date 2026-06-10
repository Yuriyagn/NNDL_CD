from __future__ import annotations

from typing import Any

from .fc_ef import FCEF
from .fc_siam_diff import FCSiamDiff
from .snunet_cd import SNUNetCD
from .stanet import STANet
from .tinycd import TinyCD


MODEL_REGISTRY = {
    "fc_ef": FCEF,
    "fc_siam_diff": FCSiamDiff,
    "stanet": STANet,
    "snunet_cd": SNUNetCD,
    "tinycd": TinyCD,
}


def build_model(name: str, **kwargs: Any):
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = ["build_model", "MODEL_REGISTRY"]
