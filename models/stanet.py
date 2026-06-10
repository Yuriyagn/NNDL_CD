from __future__ import annotations

import torch
import torch.nn as nn

from .common import ChannelSpatialAttention, ConvBlock, SpatioTemporalAttention, UpBlock
from .fc_siam_diff import SharedEncoder


class STANet(nn.Module):
    """STANet-inspired Siamese CNN with explicit spatio-temporal attention.

    The official STANet contains BAM/PAM attention variants. This adaptation
    keeps the course project self-contained: it uses a shared FC-Siam encoder,
    applies self-attention jointly over T1/T2 bottleneck tokens, then decodes
    multiscale feature differences with attention-refined skip features.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        c0, c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.encoder = SharedEncoder(in_channels, base_channels)
        self.temporal_attention = SpatioTemporalAttention(c3)
        self.skip_attn1 = ChannelSpatialAttention(c0)
        self.skip_attn2 = ChannelSpatialAttention(c1)
        self.skip_attn3 = ChannelSpatialAttention(c2)
        self.bottleneck = ConvBlock(c3, c3)
        self.dec3 = UpBlock(c3, c2, c2)
        self.dec2 = UpBlock(c2, c1, c1)
        self.dec1 = UpBlock(c1, c0, c0)
        self.out = nn.Conv2d(c0, 1, kernel_size=1)

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        feats_a = self.encoder(image_a)
        feats_b = self.encoder(image_b)
        att_a, att_b = self.temporal_attention(feats_a[-1], feats_b[-1])

        diffs = [torch.abs(b - a) for a, b in zip(feats_a, feats_b)]
        diffs[0] = self.skip_attn1(diffs[0])
        diffs[1] = self.skip_attn2(diffs[1])
        diffs[2] = self.skip_attn3(diffs[2])
        diffs[3] = torch.abs(att_b - att_a)

        x = self.bottleneck(diffs[3])
        x = self.dec3(x, diffs[2])
        x = self.dec2(x, diffs[1])
        x = self.dec1(x, diffs[0])
        return self.out(x)
