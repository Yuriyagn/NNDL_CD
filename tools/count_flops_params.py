from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_utils import count_parameters, format_params, select_device
from models import MODEL_REGISTRY, build_model


def conv_flops(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> int:
    if not isinstance(output, torch.Tensor):
        return 0
    batch = output.shape[0]
    out_channels = output.shape[1]
    out_h = output.shape[2]
    out_w = output.shape[3]
    if isinstance(module, nn.Conv2d):
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
        return int(batch * out_channels * out_h * out_w * (in_channels // groups) * kernel_h * kernel_w)
    if isinstance(module, nn.ConvTranspose2d):
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
        return int(batch * in_channels * out_h * out_w * (out_channels // groups) * kernel_h * kernel_w)
    return 0


@torch.no_grad()
def profile_model(model: nn.Module, device: torch.device, size: int, repeats: int) -> tuple[int, float]:
    flops = 0
    handles = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        flops += conv_flops(module, inputs, output)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(module.register_forward_hook(hook))

    image_a = torch.randn(1, 3, size, size, device=device)
    image_b = torch.randn(1, 3, size, size, device=device)
    model.eval()
    model(image_a, image_b)
    for handle in handles:
        handle.remove()

    for _ in range(3):
        model(image_a, image_b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        model(image_a, image_b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(repeats, 1)
    return flops, elapsed_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate params, Conv FLOPs, and latency.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    names = [args.model] if args.model else sorted(MODEL_REGISTRY)
    for name in names:
        kwargs = {}
        if args.base_channels is not None:
            kwargs["base_channels"] = args.base_channels
        model = build_model(name, **kwargs).to(device)
        params = count_parameters(model)
        flops, latency_ms = profile_model(model, device, args.size, args.repeats)
        print(
            f"{name}: params={format_params(params)} conv_flops={flops / 1e9:.3f}G "
            f"latency={latency_ms:.2f}ms device={device}"
        )


if __name__ == "__main__":
    main()
