from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock, ECAM


class SiameseEncoder5(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, c0),
                ConvBlock(c0, c1),
                ConvBlock(c1, c2),
                ConvBlock(c2, c3),
                ConvBlock(c3, c4),
            ]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for index, block in enumerate(self.blocks):
            if index > 0:
                x = self.pool(x)
            x = block(x)
            features.append(x)
        return features


class SNUNetCD(nn.Module):
    """SNUNet-CD style densely connected Siamese Nested U-Net.

    The implementation follows the core SNUNet-CD ideas needed for the course:
    shared Siamese encoding, nested U-Net++ dense skip fusion, and ECAM channel
    selection before the final change map.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.encoder = SiameseEncoder5(in_channels, base_channels)

        self.x0_1 = ConvBlock(c0 + c1, c0)
        self.x1_1 = ConvBlock(c1 + c2, c1)
        self.x2_1 = ConvBlock(c2 + c3, c2)
        self.x3_1 = ConvBlock(c3 + c4, c3)

        self.x0_2 = ConvBlock(c0 * 2 + c1, c0)
        self.x1_2 = ConvBlock(c1 * 2 + c2, c1)
        self.x2_2 = ConvBlock(c2 * 2 + c3, c2)

        self.x0_3 = ConvBlock(c0 * 3 + c1, c0)
        self.x1_3 = ConvBlock(c1 * 3 + c2, c1)

        self.x0_4 = ConvBlock(c0 * 4 + c1, c0)
        self.ecam = ECAM(c0 * 4)
        self.out = nn.Conv2d(c0 * 4, 1, kernel_size=1)

    @staticmethod
    def upsample_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        feats_a = self.encoder(image_a)
        feats_b = self.encoder(image_b)
        x0_0, x1_0, x2_0, x3_0, x4_0 = [
            torch.abs(b - a) for a, b in zip(feats_a, feats_b)
        ]

        x0_1 = self.x0_1(torch.cat([x0_0, self.upsample_like(x1_0, x0_0)], dim=1))
        x1_1 = self.x1_1(torch.cat([x1_0, self.upsample_like(x2_0, x1_0)], dim=1))
        x2_1 = self.x2_1(torch.cat([x2_0, self.upsample_like(x3_0, x2_0)], dim=1))
        x3_1 = self.x3_1(torch.cat([x3_0, self.upsample_like(x4_0, x3_0)], dim=1))

        x0_2 = self.x0_2(torch.cat([x0_0, x0_1, self.upsample_like(x1_1, x0_0)], dim=1))
        x1_2 = self.x1_2(torch.cat([x1_0, x1_1, self.upsample_like(x2_1, x1_0)], dim=1))
        x2_2 = self.x2_2(torch.cat([x2_0, x2_1, self.upsample_like(x3_1, x2_0)], dim=1))

        x0_3 = self.x0_3(
            torch.cat([x0_0, x0_1, x0_2, self.upsample_like(x1_2, x0_0)], dim=1)
        )
        x1_3 = self.x1_3(
            torch.cat([x1_0, x1_1, x1_2, self.upsample_like(x2_2, x1_0)], dim=1)
        )

        x0_4 = self.x0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.upsample_like(x1_3, x0_0)], dim=1)
        )
        fused = self.ecam(torch.cat([x0_1, x0_2, x0_3, x0_4], dim=1))
        return self.out(fused)
