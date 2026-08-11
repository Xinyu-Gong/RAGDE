import torch

from ragde.utils.geometry import (
    discrepancy_target,
    identity_affine,
    invert_affine,
    invert_flow,
    warp_affine,
    warp_flow,
)


def test_identity_warps_are_exact() -> None:
    image = torch.rand(2, 1, 16, 20)
    theta = identity_affine(2, image)
    flow = torch.zeros(2, 2, 16, 20)
    assert torch.allclose(warp_affine(image, theta), image, atol=1e-6)
    assert torch.allclose(warp_flow(image, flow), image, atol=1e-6)


def test_affine_inverse_round_trip_away_from_boundaries() -> None:
    y, x = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32), indexing="ij")
    image = (x + y).unsqueeze(0).unsqueeze(0)
    theta = image.new_tensor([[[0.95, 0.0, 0.0], [0.0, 0.95, 0.0]]])
    restored = warp_affine(warp_affine(image, invert_affine(theta)), theta)
    assert (restored[:, :, 4:-4, 4:-4] - image[:, :, 4:-4, 4:-4]).abs().mean() < 1e-5


def test_zero_discrepancy_maps_to_minus_one() -> None:
    reference = torch.zeros(1, 1, 10, 12)
    theta = identity_affine(1, reference)
    flow = torch.zeros(1, 2, 10, 12)
    target = discrepancy_target(theta, flow, theta, flow, tau=10)
    assert torch.equal(target, torch.full_like(target, -1.0))


def test_constant_flow_inverse() -> None:
    flow = torch.zeros(1, 2, 12, 12)
    flow[:, 0] = 2.0
    flow[:, 1] = -1.0
    assert torch.allclose(invert_flow(flow), -flow)
