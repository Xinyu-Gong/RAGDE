"""Reusable neural-network building blocks."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def normalization(kind: str) -> Callable[[int], nn.Module]:
    if kind == "instance":
        return lambda channels: nn.InstanceNorm2d(channels, affine=False)
    if kind == "batch":
        return nn.BatchNorm2d
    if kind == "none":
        return lambda _channels: nn.Identity()
    raise ValueError(f"Unknown normalization: {kind}")


def initialize_weights(module: nn.Module, gain: float = 0.02) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, gain)
        nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "instance") -> None:
        super().__init__()
        norm_layer = normalization(norm)
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            norm_layer(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BasicResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(inputs) + self.skip(inputs))


class ResNet34Encoder(nn.Module):
    """ResNet-34 encoder accepting a configurable number of input channels."""

    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._layer(64, 64, 3)
        self.layer2 = self._layer(64, 128, 4, stride=2)
        self.layer3 = self._layer(128, 256, 6, stride=2)
        self.layer4 = self._layer(256, 512, 3, stride=2)

    @staticmethod
    def _layer(in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [BasicResidualBlock(in_channels, out_channels, stride)]
        layers.extend(BasicResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        return self.layer4(features)


def identity_affine(batch: int, reference: torch.Tensor) -> torch.Tensor:
    theta = reference.new_zeros(batch, 2, 3)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    return theta
