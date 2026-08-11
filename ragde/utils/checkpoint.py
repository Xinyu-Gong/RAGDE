"""Checkpoint persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return checkpoint


def load_module(
    module: nn.Module,
    path: str | Path,
    key: str,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, device)
    if key not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain key '{key}'")
    module.load_state_dict(checkpoint[key], strict=strict)
    return checkpoint
