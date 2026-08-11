"""Offline synthetic training for the GDE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ragde.models import GeometricDiscrepancyEvaluator, ModalityTranslationGenerator
from ragde.utils.checkpoint import load_checkpoint, load_module, save_checkpoint
from ragde.utils.config import resolve_device
from ragde.utils.geometry import (
    discrepancy_target,
    random_affine,
    random_smooth_flow,
    warp_affine,
    warp_flow,
)

from .common import make_loader, mean_metrics, print_metrics


class GDETrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.settings = config["gde"]
        self.device = resolve_device(config)
        self.train_loader = make_loader(config, "train", shuffle=True)
        self.val_loader = make_loader(config, "val", shuffle=False)

        mtg_settings = config["mtg"]
        self.mtg = ModalityTranslationGenerator(
            channels=int(mtg_settings["generator_channels"]),
            residual_blocks=int(mtg_settings["residual_blocks"]),
        ).to(self.device)
        load_module(self.mtg, mtg_settings["checkpoint"], "generator_xy", self.device)
        self.mtg.requires_grad_(False).eval()

        self.gde = GeometricDiscrepancyEvaluator(
            channels=int(self.settings["channels"]),
            residual_blocks=int(self.settings["residual_blocks"]),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.gde.parameters(), lr=float(self.settings["learning_rate"]), betas=(0.9, 0.999)
        )
        self.output_dir = Path(config["output_dir"]) / "gde"
        self.start_epoch = 1
        self.best_metric = float("inf")

    def resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path, self.device)
        self.gde.load_state_dict(checkpoint["gde"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))

    def _transforms(
        self, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = source.shape
        geometry = self.config["geometry"]
        theta1 = random_affine(
            batch,
            source,
            rotation=float(geometry["rotation"]),
            translation=float(geometry["translation"]),
            scaling=float(geometry["scaling"]),
            shear=float(geometry["shear"]),
        )
        theta2 = random_affine(
            batch,
            source,
            rotation=float(geometry["rotation"]),
            translation=float(geometry["translation"]),
            scaling=float(geometry["scaling"]),
            shear=float(geometry["shear"]),
        )
        if self.settings.get("include_local_perturbation", True):
            flow1 = random_smooth_flow(
                batch,
                height,
                width,
                source,
                sigma=float(geometry["flow_sigma"]),
                alpha=float(geometry["flow_alpha"]),
            )
            flow2 = random_smooth_flow(
                batch,
                height,
                width,
                source,
                sigma=float(geometry["flow_sigma"]),
                alpha=float(geometry["flow_alpha"]),
            )
        else:
            flow1 = source.new_zeros(batch, 2, height, width)
            flow2 = source.new_zeros(batch, 2, height, width)
        return theta1, flow1, theta2, flow2

    def _batch(self, batch: dict[str, object], training: bool) -> dict[str, float]:
        source = batch["fixed"].to(self.device, non_blocking=True)
        theta1, flow1, theta2, flow2 = self._transforms(source)
        image1 = warp_flow(warp_affine(source, theta1), flow1)
        image2 = warp_flow(warp_affine(source, theta2), flow2)
        with torch.no_grad():
            pseudo_modality = self.mtg(image2)
            target = discrepancy_target(
                theta1, flow1, theta2, flow2, tau=float(self.settings["tau"])
            )
        prediction = self.gde(torch.cat((pseudo_modality, image1), dim=1))
        loss = F.l1_loss(prediction, target)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        mae = (prediction.detach() - target).abs().mean()
        return {"loss": float(loss.detach()), "mae": float(mae)}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.gde.eval()
        rows = []
        maximum = int(self.config["runtime"].get("max_val_batches", 0))
        for index, batch in enumerate(self.val_loader):
            if maximum and index >= maximum:
                break
            rows.append(self._batch(batch, training=False))
        self.gde.train()
        return mean_metrics(rows)

    def _save(self, path: Path, epoch: int) -> None:
        save_checkpoint(
            path,
            stage="gde",
            epoch=epoch,
            best_metric=self.best_metric,
            config=self.config,
            gde=self.gde.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def train(self) -> None:
        epochs = int(self.settings["epochs"])
        maximum = int(self.config["runtime"].get("max_train_batches", 0))
        validate_every = int(self.config["runtime"]["validate_every"])
        save_every = int(self.config["runtime"]["save_every"])
        for epoch in range(self.start_epoch, epochs + 1):
            self.gde.train()
            rows = []
            for index, batch in enumerate(self.train_loader):
                if maximum and index >= maximum:
                    break
                rows.append(self._batch(batch, training=True))
            print_metrics("GDE", epoch, epochs, mean_metrics(rows))
            self._save(self.output_dir / "last.pt", epoch)
            if epoch % save_every == 0:
                self._save(self.output_dir / f"epoch_{epoch:04d}.pt", epoch)
            if epoch % validate_every == 0 or epoch == epochs:
                validation = self.validate()
                print_metrics("GDE/VAL", epoch, epochs, validation)
                if validation["mae"] < self.best_metric:
                    self.best_metric = validation["mae"]
                    self._save(self.output_dir / "best.pt", epoch)
