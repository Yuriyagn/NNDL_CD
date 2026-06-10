from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock, UpBlock


class FCEF(nn.Module):
    """Fully convolutional early fusion network."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.enc1 = ConvBlock(in_channels * 2, channels[0])
        self.enc2 = ConvBlock(channels[0], channels[1])
        self.enc3 = ConvBlock(channels[1], channels[2])
        self.bottleneck = ConvBlock(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)
        self.dec3 = UpBlock(channels[3], channels[2], channels[2])
        self.dec2 = UpBlock(channels[2], channels[1], channels[1])
        self.dec1 = UpBlock(channels[1], channels[0], channels[0])
        self.out = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image_a, image_b], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        logits = self.out(d1)
        if logits.shape[-2:] != image_a.shape[-2:]:
            logits = F.interpolate(logits, size=image_a.shape[-2:], mode="bilinear", align_corners=False)
        return logits
