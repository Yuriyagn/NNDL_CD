from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .common import ChannelSpatialAttention, SeparableBlock, SeparableUpBlock


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 12,
        )
        self.enc1 = SeparableBlock(in_channels, c0)
        self.enc2 = SeparableBlock(c0, c1)
        self.enc3 = SeparableBlock(c1, c2)
        self.enc4 = SeparableBlock(c2, c3)
        self.enc5 = SeparableBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        return [e1, e2, e3, e4, e5]


class CompactMixAttentionMaskBlock(nn.Module):
    """Earlier compact TinyCD-inspired block kept for checkpoint compatibility."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mix = SeparableBlock(channels * 4, channels)
        self.channel_spatial = ChannelSpatialAttention(channels)
        hidden = max(channels // 2, 4)
        self.mask = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(feat_b - feat_a)
        summation = feat_a + feat_b
        product = feat_a * feat_b
        mixed = self.mix(torch.cat([feat_a, feat_b, diff, product + summation], dim=1))
        attended = self.channel_spatial(mixed)
        return attended * self.mask(attended) + diff


class CompactTinyCD(nn.Module):
    """Self-contained compact TinyCD-style fallback used by older checkpoints."""

    def __init__(self, in_channels: int = 3, base_channels: int = 12) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 12,
        )
        self.encoder = TinyEncoder(in_channels, base_channels)
        self.mix1 = CompactMixAttentionMaskBlock(c0)
        self.mix2 = CompactMixAttentionMaskBlock(c1)
        self.mix3 = CompactMixAttentionMaskBlock(c2)
        self.mix4 = CompactMixAttentionMaskBlock(c3)
        self.mix5 = CompactMixAttentionMaskBlock(c4)
        self.dec4 = SeparableUpBlock(c4, c3, c3)
        self.dec3 = SeparableUpBlock(c3, c2, c2)
        self.dec2 = SeparableUpBlock(c2, c1, c1)
        self.dec1 = SeparableUpBlock(c1, c0, c0)
        self.head = nn.Sequential(
            SeparableBlock(c0, c0),
            nn.Conv2d(c0, 1, kernel_size=1),
        )

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        feats_a = self.encoder(image_a)
        feats_b = self.encoder(image_b)
        f1 = self.mix1(feats_a[0], feats_b[0])
        f2 = self.mix2(feats_a[1], feats_b[1])
        f3 = self.mix3(feats_a[2], feats_b[2])
        f4 = self.mix4(feats_a[3], feats_b[3])
        f5 = self.mix5(feats_a[4], feats_b[4])
        x = self.dec4(f5, f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        return self.head(x)


class PixelwiseLinear(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        last_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if len(in_channels) != len(out_channels):
            raise ValueError("in_channels and out_channels must have the same length.")
        last_idx = len(in_channels) - 1
        layers: list[nn.Module] = []
        for idx, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            activation = nn.PReLU() if idx < last_idx or last_activation is None else last_activation
            layers.append(nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=True), activation))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MixingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, groups=out_channels, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mixed = torch.stack((x, y), dim=2)
        mixed = torch.reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        return self.mix(mixed)


class MixingMaskAttentionBlock(nn.Module):
    """Mix and Attention Mask Block from the official TinyCD implementation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        linear_in: list[int],
        linear_out: list[int],
        generate_masked: bool = False,
    ) -> None:
        super().__init__()
        self.mixing = MixingBlock(in_channels, out_channels)
        self.linear = PixelwiseLinear(linear_in, linear_out)
        self.final_norm = nn.InstanceNorm2d(out_channels) if generate_masked else None
        self.mixing_out = MixingBlock(in_channels, out_channels) if generate_masked else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mixed = self.mixing(x, y)
        attention = self.linear(mixed)
        if self.final_norm is None or self.mixing_out is None:
            return attention
        return self.final_norm(self.mixing_out(x, y) * attention)


class UpMask(nn.Module):
    def __init__(self, scale_factor: float, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.PReLU(),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if mask is not None:
            if x.shape[-2:] != mask.shape[-2:]:
                x = F.interpolate(x, size=mask.shape[-2:], mode="bilinear", align_corners=True)
            x = x * mask
        return self.conv(x)


def _efficientnet_b4_weights(pretrained: bool):
    if not pretrained:
        return None
    weights_cls = getattr(torchvision.models, "EfficientNet_B4_Weights", None)
    if weights_cls is not None:
        return weights_cls.DEFAULT
    return "DEFAULT"


def _get_backbone(
    pretrained: bool = False,
    output_layer: str = "3",
    freeze_backbone: bool = False,
) -> nn.ModuleList:
    try:
        features = torchvision.models.efficientnet_b4(weights=_efficientnet_b4_weights(pretrained)).features
    except TypeError:
        features = torchvision.models.efficientnet_b4(pretrained=pretrained).features

    backbone = nn.ModuleList()
    for name, layer in features.named_children():
        backbone.append(layer)
        if name == output_layer:
            break

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    return backbone


class OfficialTinyCD(nn.Module):
    """TinyCD adaptation following the official EfficientNet + MAMB design.

    The official code ends with a sigmoid and trains with BCELoss. This project
    uses BCEWithLogits + Dice for all neural models, so the final sigmoid is
    intentionally removed and the forward method returns raw logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
        output_layer: str = "3",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("OfficialTinyCD currently expects 3-channel RGB inputs.")
        self.backbone = _get_backbone(
            pretrained=pretrained,
            output_layer=output_layer,
            freeze_backbone=freeze_backbone,
        )
        self.first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self.mixing_mask = nn.ModuleList(
            [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                MixingBlock(112, 56),
            ]
        )
        self.up = nn.ModuleList(
            [
                UpMask(2, 56, 64),
                UpMask(2, 64, 64),
                UpMask(2, 64, 32),
            ]
        )
        self.classify = PixelwiseLinear([32, 16, 8], [16, 8, 1])

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        features = self._encode(image_a, image_b)
        latents = self._decode(features)
        return self.classify(latents)

    def _encode(self, image_a: torch.Tensor, image_b: torch.Tensor) -> list[torch.Tensor]:
        features = [self.first_mix(image_a, image_b)]
        for layer_index, layer in enumerate(self.backbone):
            image_a = layer(image_a)
            image_b = layer(image_b)
            if layer_index != 0:
                features.append(self.mixing_mask[layer_index - 1](image_a, image_b))
        return features

    def _decode(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for up_block, skip_index in zip(self.up, range(-2, -5, -1)):
            x = up_block(x, features[skip_index])
        return x


class TinyCD(nn.Module):
    """TinyCD wrapper.

    By default this uses the official-style EfficientNet-B4 + MAMB structure.
    Passing ``base_channels`` or ``variant="compact"`` selects the earlier
    compact fallback so old project checkpoints can still be evaluated.
    """

    def __init__(
        self,
        in_channels: int = 3,
        variant: str = "official",
        base_channels: int | None = None,
        pretrained: bool = False,
        output_layer: str = "3",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if base_channels is not None:
            variant = "compact"
        if variant == "compact":
            self.impl = CompactTinyCD(in_channels=in_channels, base_channels=base_channels or 12)
        elif variant == "official":
            self.impl = OfficialTinyCD(
                in_channels=in_channels,
                pretrained=pretrained,
                output_layer=output_layer,
                freeze_backbone=freeze_backbone,
            )
        else:
            raise ValueError(f"Unknown TinyCD variant: {variant}")

    def forward(self, image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
        return self.impl(image_a, image_b)

    def load_state_dict(self, state_dict, strict: bool = True):
        if isinstance(self.impl, CompactTinyCD) and all(not key.startswith("impl.") for key in state_dict):
            state_dict = {f"impl.{key}": value for key, value in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)
