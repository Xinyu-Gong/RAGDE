"""Structure-preserving modality translation networks."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ResidualBlock, initialize_weights, normalization


class ModalityTranslationGenerator(nn.Module):
    """CycleGAN ResNet generator used as the paper's MTG."""

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        channels: int = 64,
        residual_blocks: int = 9,
    ) -> None:
        super().__init__()
        norm = normalization("instance")
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, channels, 7, bias=False),
            norm(channels),
            nn.ReLU(inplace=True),
        ]
        current = channels
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(current, current * 2, 3, stride=2, padding=1, bias=False),
                    norm(current * 2),
                    nn.ReLU(inplace=True),
                ]
            )
            current *= 2
        layers.extend(ResidualBlock(current) for _ in range(residual_blocks))
        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        current, current // 2, 3, stride=2, padding=1, output_padding=1, bias=False
                    ),
                    norm(current // 2),
                    nn.ReLU(inplace=True),
                ]
            )
            current //= 2
        layers.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(channels, output_channels, 7),
                nn.Tanh(),
            ]
        )
        self.network = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class PatchDiscriminator(nn.Module):
    """70 x 70 PatchGAN discriminator."""

    def __init__(self, input_channels: int = 1, channels: int = 64) -> None:
        super().__init__()
        norm = normalization("instance")
        layers: list[nn.Module] = [
            nn.Conv2d(input_channels, channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        previous = 1
        for level in range(1, 3):
            multiplier = min(2**level, 8)
            layers.extend(
                [
                    nn.Conv2d(
                        channels * previous,
                        channels * multiplier,
                        4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    norm(channels * multiplier),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            previous = multiplier
        multiplier = min(previous * 2, 8)
        layers.extend(
            [
                nn.Conv2d(channels * previous, channels * multiplier, 4, padding=1, bias=False),
                norm(channels * multiplier),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels * multiplier, 1, 4, padding=1),
            ]
        )
        self.network = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
