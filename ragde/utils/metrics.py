"""Registration metrics corresponding to Eqs. (27)-(29)."""

from __future__ import annotations

import torch

from .geometry import affine_sampling_map, composed_sampling_map


def corner_rmse(
    predicted: torch.Tensor, target: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    corners = predicted.new_tensor(
        [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    ).t()
    corners = corners.unsqueeze(0).expand(predicted.shape[0], -1, -1)
    predicted_points = predicted.view(-1, 2, 3).bmm(corners)
    target_points = target.view(-1, 2, 3).bmm(corners)
    difference = predicted_points - target_points
    difference[:, 0] *= (width - 1) / 2.0
    difference[:, 1] *= (height - 1) / 2.0
    return torch.linalg.vector_norm(difference, dim=1).mean()


def displacement_rmse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((predicted - target).square().sum(dim=1).mean())


def reprojection_error(
    registered: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    error = (registered - target).abs()
    if valid_mask is None:
        return error.mean()
    denominator = valid_mask.sum().clamp_min(1.0) * registered.shape[1]
    return (error * valid_mask).sum() / denominator


def affine_valid_mask(theta: torch.Tensor, height: int, width: int) -> torch.Tensor:
    coordinates = affine_sampling_map(theta, height, width)
    valid_x = (coordinates[:, 0] >= 0) & (coordinates[:, 0] <= width - 1)
    valid_y = (coordinates[:, 1] >= 0) & (coordinates[:, 1] <= height - 1)
    return (valid_x & valid_y).unsqueeze(1).to(theta.dtype)


def composed_valid_mask(theta: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Mask pixels whose affine-plus-flow samples remain inside the source image."""
    height, width = flow.shape[-2:]
    coordinates = composed_sampling_map(theta, flow, height, width)
    valid_x = (coordinates[:, 0] >= 0) & (coordinates[:, 0] <= width - 1)
    valid_y = (coordinates[:, 1] >= 0) & (coordinates[:, 1] <= height - 1)
    return (valid_x & valid_y).unsqueeze(1).to(theta.dtype)
