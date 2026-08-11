"""Loss functions used by MTG, GDE, and registration training."""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class LeastSquaresGANLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if real else torch.zeros_like(prediction)
        return F.mse_loss(prediction, target)


class ImagePool:
    """History buffer used when updating CycleGAN discriminators."""

    def __init__(self, size: int = 50) -> None:
        self.size = int(size)
        self.images: list[torch.Tensor] = []

    def query(self, batch: torch.Tensor) -> torch.Tensor:
        if self.size <= 0:
            return batch.detach()
        selected = []
        for current in batch.detach().split(1):
            if len(self.images) < self.size:
                self.images.append(current)
                selected.append(current)
            elif random.random() < 0.5:
                index = random.randrange(self.size)
                selected.append(self.images[index].clone())
                self.images[index] = current
            else:
                selected.append(current)
        return torch.cat(selected)


def mind_descriptor(image: torch.Tensor, patch_size: int = 5, dilation: int = 2) -> torch.Tensor:
    """Modality-independent neighborhood descriptor from Eq. (4)."""
    offsets = ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
    radius = dilation + patch_size // 2
    padded = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    height, width = image.shape[-2:]
    center = padded[:, :, radius : radius + height, radius : radius + width]
    distances = []
    for dy, dx in offsets:
        shifted = padded[
            :,
            :,
            radius + dy * dilation : radius + dy * dilation + height,
            radius + dx * dilation : radius + dx * dilation + width,
        ]
        distance = F.avg_pool2d(
            (center - shifted).square(), patch_size, stride=1, padding=patch_size // 2
        )
        distances.append(distance)
    descriptor = torch.cat(distances, dim=1)
    descriptor = descriptor - descriptor.amin(dim=1, keepdim=True)
    variance = descriptor.mean(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.exp(-descriptor / variance)


def mmind_loss(
    source: torch.Tensor,
    translated: torch.Tensor,
    scales: Sequence[int] = (1, 2, 4),
    weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Multi-scale MIND structural consistency from Eq. (5)."""
    if len(scales) != len(weights):
        raise ValueError("MMIND scales and weights must have equal length")
    loss = source.new_zeros(())
    for scale, weight in zip(scales, weights):
        if scale > 1:
            source_scale = F.avg_pool2d(source, scale, stride=scale)
            translated_scale = F.avg_pool2d(translated, scale, stride=scale)
        else:
            source_scale, translated_scale = source, translated
        loss = loss + float(weight) * F.l1_loss(
            mind_descriptor(source_scale), mind_descriptor(translated_scale)
        )
    return loss


def flow_smoothness(flow: torch.Tensor) -> torch.Tensor:
    horizontal = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    vertical = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return horizontal.square().mean() + vertical.square().mean()


def affine_inverse_consistency(forward: torch.Tensor, backward: torch.Tensor) -> torch.Tensor:
    from .geometry import homogeneous

    product = homogeneous(forward).bmm(homogeneous(backward))
    identity = torch.eye(3, device=product.device, dtype=product.dtype).expand_as(product)
    return F.mse_loss(product, identity)


def flow_inverse_consistency(forward: torch.Tensor, backward: torch.Tensor) -> torch.Tensor:
    from .geometry import warp_flow

    residual = forward + warp_flow(backward, forward, padding_mode="border")
    return residual.square().mean()
