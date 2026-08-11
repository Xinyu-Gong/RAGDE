"""GDE-guided coarse-to-fine registration network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ragde.utils.geometry import warp_affine, warp_flow

from .blocks import ConvBlock, ResNet34Encoder, identity_affine, initialize_weights
from .gde import GeometricDiscrepancyEvaluator


class AffineRegressor(nn.Module):
    """ResNet-34 affine regressor from Eq. (14)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        features = self.encoder(torch.cat((moving, fixed), dim=1))
        delta = self.head(self.pool(features).flatten(1)).view(-1, 2, 3)
        return identity_affine(delta.shape[0], delta) + delta


class DownBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat((skip, inputs), dim=1))


class DeformableRegressor(nn.Module):
    """U-Net displacement regressor from Eqs. (18)-(19)."""

    def __init__(self, channels: int = 8, max_flow: float = 16.0) -> None:
        super().__init__()
        self.max_flow = float(max_flow)
        self.stem = ConvBlock(2, channels)
        self.down1 = DownBlock(channels, channels * 2)
        self.down2 = DownBlock(channels * 2, channels * 4)
        self.down3 = DownBlock(channels * 4, channels * 8)
        self.down4 = DownBlock(channels * 8, channels * 16)
        self.up1 = UpBlock(channels * 16, channels * 8, channels * 8)
        self.up2 = UpBlock(channels * 8, channels * 4, channels * 4)
        self.up3 = UpBlock(channels * 4, channels * 2, channels * 2)
        self.up4 = UpBlock(channels * 2, channels, channels)
        self.head = nn.Sequential(nn.Conv2d(channels, 2, 3, padding=1), nn.Tanh())
        self.apply(initialize_weights)
        nn.init.zeros_(self.head[0].weight)
        nn.init.zeros_(self.head[0].bias)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        level1 = self.stem(torch.cat((moving, fixed), dim=1))
        level2 = self.down1(level1)
        level3 = self.down2(level2)
        level4 = self.down3(level3)
        bottleneck = self.down4(level4)
        features = self.up1(bottleneck, level4)
        features = self.up2(features, level3)
        features = self.up3(features, level2)
        features = self.up4(features, level1)
        return self.head(features) * self.max_flow


class RAGDENet(nn.Module):
    """Affine and deformable registration sharing one frozen GDE."""

    def __init__(
        self,
        gde: GeometricDiscrepancyEvaluator,
        guidance_scale: float = 2.0,
        deformable_channels: int = 8,
        max_flow: float = 16.0,
    ) -> None:
        super().__init__()
        self.gde = gde
        self.guidance_scale = float(guidance_scale)
        self.affine = AffineRegressor()
        self.deformable = DeformableRegressor(deformable_channels, max_flow)
        self.gde.requires_grad_(False)
        self.gde.eval()

    def train(self, mode: bool = True) -> RAGDENet:
        super().train(mode)
        self.gde.eval()
        return self

    def response_guidance(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response = self.gde(torch.cat((moving, fixed), dim=1))
        guided = torch.clamp(moving + self.guidance_scale * response, -1.0, 1.0)
        return guided, response

    def estimate_affine(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        guided, response = self.response_guidance(moving, fixed)
        return self.affine(guided, fixed), response

    def estimate_flow(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        guided, response = self.response_guidance(moving, fixed)
        return self.deformable(guided, fixed), response

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> dict[str, torch.Tensor]:
        theta, affine_response = self.estimate_affine(moving, fixed)
        coarse = warp_affine(moving, theta)
        flow, deformable_response = self.estimate_flow(coarse, fixed)
        fine = warp_flow(coarse, flow)
        return {
            "theta": theta,
            "flow": flow,
            "coarse": coarse,
            "fine": fine,
            "affine_response": affine_response,
            "deformable_response": deformable_response,
        }
