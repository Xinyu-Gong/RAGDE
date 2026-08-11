"""Radiometric-agnostic Geometric Discrepancy Evaluator."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import ResidualBlock, initialize_weights


class GeometricDiscrepancyEvaluator(nn.Module):
    """Fully convolutional dense discrepancy predictor from Eq. (8)."""

    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 1,
        channels: int = 64,
        residual_blocks: int = 9,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, channels, 7),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        current = channels
        for _ in range(2):
            layers.extend(
                [
                    nn.Conv2d(current, current * 2, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(current * 2),
                    nn.ReLU(inplace=True),
                ]
            )
            current *= 2
        layers.extend(ResidualBlock(current) for _ in range(residual_blocks))
        for _ in range(2):
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        current, current // 2, 3, stride=2, padding=1, output_padding=1
                    ),
                    nn.InstanceNorm2d(current // 2),
                    nn.ReLU(inplace=True),
                ]
            )
            current //= 2
        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(channels, output_channels, 7), nn.Tanh()])
        self.network = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, image_pair: torch.Tensor) -> torch.Tensor:
        return self.network(image_pair)

    @staticmethod
    def aggregate(response: torch.Tensor) -> torch.Tensor:
        """Non-negative spatial aggregation A+ from Eq. (11)."""
        return ((response + 1.0) * 0.5).mean(dim=(1, 2, 3))
