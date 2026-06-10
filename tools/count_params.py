from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_utils import count_parameters, format_params
from models import MODEL_REGISTRY, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count trainable parameters.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [args.model] if args.model else sorted(MODEL_REGISTRY)
    for name in names:
        kwargs = {}
        if args.base_channels is not None:
            kwargs["base_channels"] = args.base_channels
        model = build_model(name, **kwargs)
        params = count_parameters(model)
        print(f"{name}: {params} ({format_params(params)})")


if __name__ == "__main__":
    main()
