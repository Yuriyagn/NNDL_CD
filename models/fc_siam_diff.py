from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConvBlock, UpBlock


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc1 = ConvBlock(in_channels, channels[0])
        self.enc2 = ConvBlock(channels[0], channels[1])
        self.enc3 = ConvBlock(channels[1], channels[2])
        self.enc4 = ConvBlock(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return [e1, e2, e3, e4]


class FCSiamDiff(nn.Module):
    """Shared encoder with absolute feature differences and U-Net decoder."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.encoder = SharedEncoder(in_channels, base_channels)
        self.dec3 = UpBlock(channels[3], channels[2], channels[2])
        self.dec2 = UpBlock(channels[2], channels[1], channels[1])
        self.dec1 = UpBlock(channels[1], channels[0], channels[0])
        self.out = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        feats_a = self.encoder(image_a)
        feats_b = self.encoder(image_b)
        diffs = [torch.abs(b - a) for a, b in zip(feats_a, feats_b)]
        d3 = self.dec3(diffs[3], diffs[2])
        d2 = self.dec2(d3, diffs[1])
        d1 = self.dec1(d2, diffs[0])
        return self.out(d1)
