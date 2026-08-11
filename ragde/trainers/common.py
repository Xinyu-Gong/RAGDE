"""Shared training infrastructure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ragde.data import PairedImageDataset


def make_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    data = config["data"]
    root_key = "train_root" if split == "train" else "val_root"
    root = data.get(root_key)
    if not root:
        raise ValueError(f"data.{root_key} is required")
    dataset = PairedImageDataset(
        root,
        fixed_dir=data["fixed_dir"],
        moving_dir=data["moving_dir"],
        image_size=data["image_size"],
        strict_pairs=data.get("strict_pairs", True),
    )
    loader = config["loader"]
    return DataLoader(
        dataset,
        batch_size=int(loader.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=int(loader.get("num_workers", 4)),
        pin_memory=bool(loader.get("pin_memory", True)),
        drop_last=shuffle and len(dataset) >= int(loader.get("batch_size", 8)),
    )


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("No batches were processed")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def print_metrics(stage: str, epoch: int, epochs: int, metrics: dict[str, float]) -> None:
    values = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
    print(f"[{stage}] epoch={epoch}/{epochs} {values}", flush=True)


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
