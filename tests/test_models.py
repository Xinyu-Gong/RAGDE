import torch

from ragde.models import GeometricDiscrepancyEvaluator, ModalityTranslationGenerator, RAGDENet


def test_model_interfaces_and_frozen_gde() -> None:
    image = torch.rand(1, 1, 64, 64).mul(2).sub(1)
    generator = ModalityTranslationGenerator(channels=8, residual_blocks=1).eval()
    gde = GeometricDiscrepancyEvaluator(channels=8, residual_blocks=1).eval()
    network = RAGDENet(gde, deformable_channels=8, max_flow=4).eval()
    with torch.no_grad():
        translated = generator(image)
        output = network(translated, image)
    assert translated.shape == image.shape
    assert output["theta"].shape == (1, 2, 3)
    assert output["flow"].shape == (1, 2, 64, 64)
    assert output["fine"].shape == image.shape
    assert not any(parameter.requires_grad for parameter in network.gde.parameters())


def test_gde_aggregate_range() -> None:
    response = torch.tensor([[[[-1.0, 1.0]]]])
    assert torch.allclose(GeometricDiscrepancyEvaluator.aggregate(response), torch.tensor([0.5]))
