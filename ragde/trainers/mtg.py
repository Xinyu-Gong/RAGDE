"""Training loop for the structure-preserving MTG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ragde.models import ModalityTranslationGenerator, PatchDiscriminator
from ragde.utils.checkpoint import load_checkpoint, save_checkpoint
from ragde.utils.config import resolve_device
from ragde.utils.losses import ImagePool, LeastSquaresGANLoss, mmind_loss

from .common import make_loader, mean_metrics, print_metrics, set_requires_grad


class MTGTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.settings = config["mtg"]
        self.device = resolve_device(config)
        self.train_loader = make_loader(config, "train", shuffle=True)
        self.val_loader = make_loader(config, "val", shuffle=False)

        channels = int(self.settings["generator_channels"])
        blocks = int(self.settings["residual_blocks"])
        discriminator_channels = int(self.settings["discriminator_channels"])
        self.generator_xy = ModalityTranslationGenerator(
            channels=channels, residual_blocks=blocks
        ).to(self.device)
        self.generator_yx = ModalityTranslationGenerator(
            channels=channels, residual_blocks=blocks
        ).to(self.device)
        self.discriminator_x = PatchDiscriminator(channels=discriminator_channels).to(self.device)
        self.discriminator_y = PatchDiscriminator(channels=discriminator_channels).to(self.device)

        betas = (float(self.settings["beta1"]), float(self.settings["beta2"]))
        learning_rate = float(self.settings["learning_rate"])
        self.generator_optimizer = torch.optim.Adam(
            list(self.generator_xy.parameters()) + list(self.generator_yx.parameters()),
            lr=learning_rate,
            betas=betas,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            list(self.discriminator_x.parameters()) + list(self.discriminator_y.parameters()),
            lr=learning_rate,
            betas=betas,
        )
        pool_size = int(self.settings["image_pool_size"])
        self.fake_x_pool = ImagePool(pool_size)
        self.fake_y_pool = ImagePool(pool_size)
        self.gan = LeastSquaresGANLoss()
        self.output_dir = Path(config["output_dir"]) / "mtg"
        self.start_epoch = 1
        self.best_metric = float("inf")

    def resume(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path, self.device)
        self.generator_xy.load_state_dict(checkpoint["generator_xy"])
        self.generator_yx.load_state_dict(checkpoint["generator_yx"])
        self.discriminator_x.load_state_dict(checkpoint["discriminator_x"])
        self.discriminator_y.load_state_dict(checkpoint["discriminator_y"])
        self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))

    def _mmind(self, source: torch.Tensor, translated: torch.Tensor) -> torch.Tensor:
        return mmind_loss(
            source,
            translated,
            scales=self.settings["mmind_scales"],
            weights=self.settings["mmind_weights"],
        )

    def _step(self, batch: dict[str, object]) -> dict[str, float]:
        real_x = batch["fixed"].to(self.device, non_blocking=True)
        real_y = batch["moving"].to(self.device, non_blocking=True)

        set_requires_grad(self.discriminator_x, False)
        set_requires_grad(self.discriminator_y, False)
        self.generator_optimizer.zero_grad(set_to_none=True)
        fake_y = self.generator_xy(real_x)
        fake_x = self.generator_yx(real_y)
        reconstructed_x = self.generator_yx(fake_y)
        reconstructed_y = self.generator_xy(fake_x)
        adversarial = self.gan(self.discriminator_y(fake_y), True) + self.gan(
            self.discriminator_x(fake_x), True
        )
        cycle = F.l1_loss(reconstructed_x, real_x) + F.l1_loss(reconstructed_y, real_y)
        structural = self._mmind(real_x, fake_y) + self._mmind(real_y, fake_x)
        generator_loss = (
            adversarial
            + float(self.settings["lambda_cycle"]) * cycle
            + float(self.settings["lambda_mmind"]) * structural
        )
        generator_loss.backward()
        self.generator_optimizer.step()

        set_requires_grad(self.discriminator_x, True)
        set_requires_grad(self.discriminator_y, True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss_x = 0.5 * (
            self.gan(self.discriminator_x(real_x), True)
            + self.gan(self.discriminator_x(self.fake_x_pool.query(fake_x)), False)
        )
        loss_y = 0.5 * (
            self.gan(self.discriminator_y(real_y), True)
            + self.gan(self.discriminator_y(self.fake_y_pool.query(fake_y)), False)
        )
        discriminator_loss = loss_x + loss_y
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        return {
            "generator": float(generator_loss.detach()),
            "discriminator": float(discriminator_loss.detach()),
            "cycle": float(cycle.detach()),
            "mmind": float(structural.detach()),
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.generator_xy.eval()
        self.generator_yx.eval()
        rows = []
        maximum = int(self.config["runtime"].get("max_val_batches", 0))
        for index, batch in enumerate(self.val_loader):
            if maximum and index >= maximum:
                break
            real_x = batch["fixed"].to(self.device, non_blocking=True)
            real_y = batch["moving"].to(self.device, non_blocking=True)
            rows.append(
                {
                    "mmind": float(
                        self._mmind(real_x, self.generator_xy(real_x))
                        + self._mmind(real_y, self.generator_yx(real_y))
                    )
                }
            )
        self.generator_xy.train()
        self.generator_yx.train()
        return mean_metrics(rows)

    def _save(self, path: Path, epoch: int) -> None:
        save_checkpoint(
            path,
            stage="mtg",
            epoch=epoch,
            best_metric=self.best_metric,
            config=self.config,
            generator_xy=self.generator_xy.state_dict(),
            generator_yx=self.generator_yx.state_dict(),
            discriminator_x=self.discriminator_x.state_dict(),
            discriminator_y=self.discriminator_y.state_dict(),
            generator_optimizer=self.generator_optimizer.state_dict(),
            discriminator_optimizer=self.discriminator_optimizer.state_dict(),
        )

    def train(self) -> None:
        epochs = int(self.settings["epochs"])
        maximum = int(self.config["runtime"].get("max_train_batches", 0))
        validate_every = int(self.config["runtime"]["validate_every"])
        save_every = int(self.config["runtime"]["save_every"])
        for epoch in range(self.start_epoch, epochs + 1):
            rows = []
            for index, batch in enumerate(self.train_loader):
                if maximum and index >= maximum:
                    break
                rows.append(self._step(batch))
            print_metrics("MTG", epoch, epochs, mean_metrics(rows))
            self._save(self.output_dir / "last.pt", epoch)
            if epoch % save_every == 0:
                self._save(self.output_dir / f"epoch_{epoch:04d}.pt", epoch)
            if epoch % validate_every == 0 or epoch == epochs:
                validation = self.validate()
                print_metrics("MTG/VAL", epoch, epochs, validation)
                if validation["mmind"] < self.best_metric:
                    self.best_metric = validation["mmind"]
                    self._save(self.output_dir / "best.pt", epoch)
