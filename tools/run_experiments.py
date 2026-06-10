from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    "configs/fc_ef.yaml",
    "configs/fc_siam_diff.yaml",
    "configs/stanet.yaml",
    "configs/snunet_cd.yaml",
    "configs/tinycd.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test all configured models.")
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-masks", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def model_name_from_config(config_path: str) -> str:
    return Path(config_path).stem


def main() -> None:
    args = parse_args()
    for config in args.configs:
        train_cmd = [sys.executable, "train.py", "--config", config, "--device", args.device]
        if args.batch_size is not None:
            train_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])
        if not args.skip_train:
            run(train_cmd)

        if not args.skip_test:
            model_name = model_name_from_config(config)
            checkpoint = f"results/checkpoints/{model_name}/best.pt"
            test_cmd = [
                sys.executable,
                "test.py",
                "--checkpoint",
                checkpoint,
                "--device",
                args.device,
                "--max-visuals",
                "8",
            ]
            if args.batch_size is not None:
                test_cmd.extend(["--batch-size", str(args.batch_size)])
            if args.save_masks:
                test_cmd.append("--save-masks")
            run(test_cmd)


if __name__ == "__main__":
    main()
