"""Command-line interface for training and evaluating RAGDE-Net."""

from __future__ import annotations

import argparse
from typing import Any

import torch

from ragde.models import GeometricDiscrepancyEvaluator, ModalityTranslationGenerator, RAGDENet
from ragde.trainers import GDETrainer, MTGTrainer, RegistrationTrainer
from ragde.trainers.common import write_metrics
from ragde.utils.config import load_config, seed_everything


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/ragde.yaml", help="YAML configuration")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted configuration value; repeat as needed",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragde", description="RAGDE-Net tools")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train one pipeline stage")
    train.add_argument("stage", choices=("mtg", "gde", "registration"))
    train.add_argument("--resume", default="", help="Resume from a stage checkpoint")
    _common(train)

    evaluate = commands.add_parser("evaluate", help="Evaluate a registration checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", default="")
    evaluate.add_argument("--max-batches", type=int, default=0)
    _common(evaluate)

    smoke = commands.add_parser("smoke", help="Run dataset-free model interface checks")
    _common(smoke)
    return parser


def _trainer(stage: str, config: dict[str, Any]) -> Any:
    classes = {"mtg": MTGTrainer, "gde": GDETrainer, "registration": RegistrationTrainer}
    return classes[stage](config)


def smoke(config: dict[str, Any]) -> None:
    device = torch.device("cpu")
    image = torch.randn(1, 1, 64, 64, device=device).clamp(-1, 1)
    mtg = ModalityTranslationGenerator(channels=8, residual_blocks=1).to(device).eval()
    gde = GeometricDiscrepancyEvaluator(channels=8, residual_blocks=1).to(device).eval()
    network = RAGDENet(gde, deformable_channels=8, max_flow=4).to(device).eval()
    with torch.no_grad():
        translated = mtg(image)
        response = gde(torch.cat((translated, image), dim=1))
        output = network(translated, image)
    assert translated.shape == image.shape
    assert response.shape == image.shape
    assert output["theta"].shape == (1, 2, 3)
    assert output["flow"].shape == (1, 2, 64, 64)
    print("Model interface smoke test passed.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, args.set)
    seed_everything(int(config.get("seed", 42)))
    if args.command == "train":
        trainer = _trainer(args.stage, config)
        if args.resume:
            trainer.resume(args.resume)
        trainer.train()
    elif args.command == "evaluate":
        trainer = RegistrationTrainer(config)
        trainer.load_weights(args.checkpoint)
        metrics = trainer.evaluate(max_batches=args.max_batches)
        print(" ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
        if args.output:
            write_metrics(args.output, metrics)
    else:
        smoke(config)


if __name__ == "__main__":
    main()
