"""Training and evaluation for GDE-guided coarse-to-fine registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ragde.models import GeometricDiscrepancyEvaluator, RAGDENet
from ragde.utils.checkpoint import load_checkpoint, load_module, save_checkpoint
from ragde.utils.config import resolve_device
from ragde.utils.geometry import (
    displacement_map,
    invert_affine,
    invert_flow,
    random_affine,
    random_smooth_flow,
    warp_affine,
    warp_flow,
)
from ragde.utils.losses import (
    affine_inverse_consistency,
    flow_inverse_consistency,
    flow_smoothness,
)
from ragde.utils.metrics import (
    composed_valid_mask,
    corner_rmse,
    displacement_rmse,
    reprojection_error,
)

from .common import make_loader, mean_metrics, print_metrics


class RegistrationTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.settings = config["registration"]
        self.device = resolve_device(config)
        self.train_loader = make_loader(config, "train", shuffle=True)
        self.val_loader = make_loader(config, "val", shuffle=False)

        gde_settings = config["gde"]
        gde = GeometricDiscrepancyEvaluator(
            channels=int(gde_settings["channels"]),
            residual_blocks=int(gde_settings["residual_blocks"]),
        ).to(self.device)
        load_module(gde, gde_settings["checkpoint"], "gde", self.device)
        self.network = RAGDENet(
            gde,
            guidance_scale=float(self.settings["guidance_scale"]),
            deformable_channels=int(self.settings["deformable_channels"]),
            max_flow=float(self.settings["max_flow"]),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in self.network.parameters() if parameter.requires_grad),
            lr=float(self.settings["learning_rate"]),
            betas=(float(self.settings["beta1"]), float(self.settings["beta2"])),
        )
        self.output_dir = Path(config["output_dir"]) / "registration"
        self.start_epoch = 1
        self.best_metric = float("inf")

    def resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path, self.device)
        self.network.load_state_dict(checkpoint["registration"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))

    def load_weights(self, path: str | Path) -> None:
        load_module(self.network, path, "registration", self.device)

    def _synthetic_pair(
        self, moving: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = moving.shape
        geometry = self.config["geometry"]
        theta = random_affine(
            batch,
            moving,
            rotation=float(geometry["rotation"]),
            translation=float(geometry["translation"]),
            scaling=float(geometry["scaling"]),
            shear=float(geometry["shear"]),
        )
        flow = random_smooth_flow(
            batch,
            height,
            width,
            moving,
            sigma=float(geometry["flow_sigma"]),
            alpha=float(geometry["flow_alpha"]),
        )
        # Build an input whose known correction follows the network order:
        # affine first, then local flow.
        local_misaligned = warp_flow(moving, invert_flow(flow))
        misaligned = warp_affine(local_misaligned, invert_affine(theta))
        return misaligned, theta, flow

    def schedule(self, epoch: int) -> float:
        transition = max(int(self.settings["transition_epoch"]), 1)
        floor = float(self.settings["affine_loss_floor"])
        return max(floor, 1.0 - epoch / transition)

    def _batch(self, batch: dict[str, object], epoch: int, training: bool) -> dict[str, float]:
        fixed = batch["fixed"].to(self.device, non_blocking=True)
        moving = batch["moving"].to(self.device, non_blocking=True)
        misaligned, target_theta, target_flow = self._synthetic_pair(moving)
        output = self.network(misaligned, fixed)

        affine_parameter = F.l1_loss(output["theta"], target_theta)
        affine_gde = self.network.gde.aggregate(
            self.network.gde(torch.cat((output["coarse"], fixed), dim=1))
        ).mean()
        reverse_theta, _ = self.network.estimate_affine(fixed, misaligned)
        affine_symmetry = affine_inverse_consistency(output["theta"], reverse_theta)
        affine_loss = (
            float(self.settings["alpha_param"]) * affine_parameter
            + float(self.settings["alpha_gde"]) * affine_gde
            + float(self.settings["alpha_symmetry"]) * affine_symmetry
        )

        flow_loss = F.mse_loss(output["flow"], target_flow)
        deformable_gde = self.network.gde.aggregate(
            self.network.gde(torch.cat((output["fine"], fixed), dim=1))
        ).mean()
        reverse_flow, _ = self.network.estimate_flow(fixed, output["coarse"])
        deformable_symmetry = flow_inverse_consistency(output["flow"], reverse_flow)
        smoothness = flow_smoothness(output["flow"])
        deformable_loss = (
            float(self.settings["beta_flow"]) * flow_loss
            + float(self.settings["beta_gde"]) * deformable_gde
            + float(self.settings["beta_symmetry"]) * deformable_symmetry
            + float(self.settings["beta_smoothness"]) * smoothness
        )

        weight = self.schedule(epoch)
        total = weight * affine_loss + (1.0 - weight) * deformable_loss
        if training:
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            self.optimizer.step()

        predicted_displacement = displacement_map(output["theta"], output["flow"])
        target_displacement = displacement_map(target_theta, target_flow)
        valid_mask = composed_valid_mask(target_theta, target_flow)
        return {
            "loss": float(total.detach()),
            "affine": float(affine_loss.detach()),
            "deformable": float(deformable_loss.detach()),
            "re": float(reprojection_error(output["fine"].detach(), moving, valid_mask)),
            "rmse_cor": float(
                corner_rmse(
                    output["theta"].detach(),
                    target_theta,
                    moving.shape[-2],
                    moving.shape[-1],
                )
            ),
            "rmse_disp": float(
                displacement_rmse(predicted_displacement.detach(), target_displacement)
            ),
        }

    @torch.no_grad()
    def evaluate(self, epoch: int | None = None, max_batches: int = 0) -> dict[str, float]:
        self.network.eval()
        rows = []
        evaluation_epoch = epoch if epoch is not None else int(self.settings["transition_epoch"])
        for index, batch in enumerate(self.val_loader):
            if max_batches and index >= max_batches:
                break
            rows.append(self._batch(batch, evaluation_epoch, training=False))
        self.network.train()
        metrics = mean_metrics(rows)
        return {key: metrics[key] for key in ("re", "rmse_cor", "rmse_disp")}

    def _save(self, path: Path, epoch: int) -> None:
        save_checkpoint(
            path,
            stage="registration",
            epoch=epoch,
            best_metric=self.best_metric,
            config=self.config,
            registration=self.network.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def train(self) -> None:
        epochs = int(self.settings["epochs"])
        maximum = int(self.config["runtime"].get("max_train_batches", 0))
        validate_every = int(self.config["runtime"]["validate_every"])
        save_every = int(self.config["runtime"]["save_every"])
        for epoch in range(self.start_epoch, epochs + 1):
            self.network.train()
            rows = []
            for index, batch in enumerate(self.train_loader):
                if maximum and index >= maximum:
                    break
                rows.append(self._batch(batch, epoch, training=True))
            print_metrics("REG", epoch, epochs, mean_metrics(rows))
            self._save(self.output_dir / "last.pt", epoch)
            if epoch % save_every == 0:
                self._save(self.output_dir / f"epoch_{epoch:04d}.pt", epoch)
            if epoch % validate_every == 0 or epoch == epochs:
                validation = self.evaluate(
                    epoch, int(self.config["runtime"].get("max_val_batches", 0))
                )
                print_metrics("REG/VAL", epoch, epochs, validation)
                if validation["re"] < self.best_metric:
                    self.best_metric = validation["re"]
                    self._save(self.output_dir / "best.pt", epoch)
