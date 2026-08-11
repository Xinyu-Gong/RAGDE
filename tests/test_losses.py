import torch

from ragde.utils.losses import flow_smoothness, mmind_loss


def test_mmind_is_zero_for_identical_images() -> None:
    image = torch.rand(2, 1, 32, 32)
    assert float(mmind_loss(image, image)) == 0.0


def test_mmind_has_gradient() -> None:
    source = torch.rand(1, 1, 24, 24)
    translated = torch.rand(1, 1, 24, 24, requires_grad=True)
    mmind_loss(source, translated, scales=(1, 2), weights=(1.0, 1.0)).backward()
    assert translated.grad is not None
    assert torch.isfinite(translated.grad).all()


def test_constant_flow_is_smooth() -> None:
    flow = torch.ones(1, 2, 12, 12)
    assert float(flow_smoothness(flow)) == 0.0
