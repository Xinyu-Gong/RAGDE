"""Differentiable geometry utilities with explicit coordinate conventions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def warp_affine(
    image: torch.Tensor, theta: torch.Tensor, padding_mode: str = "border"
) -> torch.Tensor:
    """Backward-warp ``image`` with normalized output-to-input affine matrices."""
    theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(theta, image.shape, align_corners=True)
    return F.grid_sample(
        image, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    )


def pixel_grid(batch: int, height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    """Return a ``[B, 2, H, W]`` grid ordered as ``(x, y)`` in pixels."""
    y, x = torch.meshgrid(
        torch.arange(height, device=reference.device, dtype=reference.dtype),
        torch.arange(width, device=reference.device, dtype=reference.dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


def warp_flow(
    image: torch.Tensor, flow: torch.Tensor, padding_mode: str = "border"
) -> torch.Tensor:
    """Backward-warp with a pixel flow ordered as ``(dx, dy)``."""
    batch, _, height, width = image.shape
    sample = pixel_grid(batch, height, width, image) + flow
    x = 2.0 * sample[:, 0] / max(width - 1, 1) - 1.0
    y = 2.0 * sample[:, 1] / max(height - 1, 1) - 1.0
    return F.grid_sample(
        image,
        torch.stack((x, y), dim=-1),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def identity_affine(batch: int, reference: torch.Tensor) -> torch.Tensor:
    theta = reference.new_zeros(batch, 2, 3)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    return theta


def homogeneous(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view(-1, 2, 3)
    bottom = theta.new_zeros(theta.shape[0], 1, 3)
    bottom[:, 0, 2] = 1.0
    return torch.cat((theta, bottom), dim=1)


def invert_affine(theta: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(homogeneous(theta))[:, :2]


def compose_affine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose sampling matrices so ``first`` is applied before ``second``."""
    return homogeneous(first).bmm(homogeneous(second))[:, :2]


def random_affine(
    batch: int,
    reference: torch.Tensor,
    rotation: float = 30.0,
    translation: float = 0.2,
    scaling: float = 0.15,
    shear: float = 0.05,
) -> torch.Tensor:
    """Sample affine-grid matrices using paper perturbation ranges.

    ``translation`` is expressed as a fraction of image extent. It is doubled
    when converted to affine-grid coordinates, whose full range is ``[-1, 1]``.
    """

    def uniform(limit: float) -> torch.Tensor:
        values = torch.rand(batch, device=reference.device, dtype=reference.dtype)
        return (values * 2.0 - 1.0) * limit

    angle = uniform(math.radians(rotation))
    scale_x = 1.0 + uniform(scaling)
    scale_y = 1.0 + uniform(scaling)
    shear_x = uniform(shear)
    shear_y = uniform(shear)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    theta = reference.new_zeros(batch, 2, 3)
    theta[:, 0, 0] = scale_x * cos_angle - shear_y * sin_angle
    theta[:, 0, 1] = shear_x * cos_angle - scale_y * sin_angle
    theta[:, 1, 0] = scale_x * sin_angle + shear_y * cos_angle
    theta[:, 1, 1] = shear_x * sin_angle + scale_y * cos_angle
    theta[:, 0, 2] = 2.0 * uniform(translation)
    theta[:, 1, 2] = 2.0 * uniform(translation)
    return theta


def _gaussian_kernel(sigma: float, reference: torch.Tensor) -> torch.Tensor:
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    coordinates = torch.arange(-radius, radius + 1, device=reference.device, dtype=reference.dtype)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    return kernel


def random_smooth_flow(
    batch: int,
    height: int,
    width: int,
    reference: torch.Tensor,
    sigma: float = 4.0,
    alpha: float = 8.0,
) -> torch.Tensor:
    """Generate Gaussian-smoothed random displacement fields in pixels."""
    noise = (
        torch.rand(batch, 2, height, width, device=reference.device, dtype=reference.dtype)
        .mul_(2.0)
        .sub_(1.0)
    )
    kernel = _gaussian_kernel(float(sigma), reference)
    radius = kernel.numel() // 2
    horizontal = kernel.view(1, 1, 1, -1).expand(2, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(2, 1, -1, 1)
    flow = F.conv2d(noise, horizontal, padding=(0, radius), groups=2)
    flow = F.conv2d(flow, vertical, padding=(radius, 0), groups=2)
    scale = flow.flatten(2).std(dim=2, keepdim=True).view(batch, 2, 1, 1).clamp_min(1e-6)
    return flow / scale * float(alpha)


def invert_flow(flow: torch.Tensor, iterations: int = 12) -> torch.Tensor:
    """Numerically invert a backward displacement field by fixed-point iteration."""
    inverse = -flow
    for _ in range(iterations):
        inverse = -warp_flow(flow, inverse, padding_mode="border")
    return inverse


def affine_sampling_map(theta: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Return affine source coordinates in pixels as ``[B, 2, H, W]``."""
    theta = theta.view(-1, 2, 3)
    batch = theta.shape[0]
    base = pixel_grid(batch, height, width, theta)
    x = 2.0 * base[:, 0] / max(width - 1, 1) - 1.0
    y = 2.0 * base[:, 1] / max(height - 1, 1) - 1.0
    points = torch.stack((x, y, torch.ones_like(x)), dim=1).flatten(2)
    transformed = theta.bmm(points).view(batch, 2, height, width)
    transformed_x = (transformed[:, 0] + 1.0) * (width - 1) / 2.0
    transformed_y = (transformed[:, 1] + 1.0) * (height - 1) / 2.0
    return torch.stack((transformed_x, transformed_y), dim=1)


def composed_sampling_map(
    theta: torch.Tensor, flow: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Source coordinates for ``warp_flow(warp_affine(image, theta), flow)``."""
    theta = theta.view(-1, 2, 3)
    batch = theta.shape[0]
    sample = pixel_grid(batch, height, width, theta) + flow
    x = 2.0 * sample[:, 0] / max(width - 1, 1) - 1.0
    y = 2.0 * sample[:, 1] / max(height - 1, 1) - 1.0
    points = torch.stack((x, y, torch.ones_like(x)), dim=1).flatten(2)
    transformed = theta.bmm(points).view(batch, 2, height, width)
    transformed[:, 0] = (transformed[:, 0] + 1.0) * (width - 1) / 2.0
    transformed[:, 1] = (transformed[:, 1] + 1.0) * (height - 1) / 2.0
    return transformed


def displacement_map(theta: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    height, width = flow.shape[-2:]
    base = pixel_grid(theta.shape[0], height, width, theta)
    return composed_sampling_map(theta, flow, height, width) - base


def discrepancy_target(
    theta1: torch.Tensor,
    flow1: torch.Tensor,
    theta2: torch.Tensor,
    flow2: torch.Tensor,
    tau: float = 10.0,
) -> torch.Tensor:
    """Construct the signed pseudo discrepancy map from Eq. (9)."""
    height, width = flow1.shape[-2:]
    map1 = composed_sampling_map(theta1, flow1, height, width)
    map2 = composed_sampling_map(theta2, flow2, height, width)
    distance = torch.linalg.vector_norm(map1 - map2, dim=1, keepdim=True)
    return 2.0 * torch.clamp(distance / float(tau), max=1.0) - 1.0
