"""YAML configuration loading, overrides, and validation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _assign(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError(f"Cannot assign nested override: {dotted_key}")
    target[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use KEY=VALUE syntax: {override}")
        key, raw_value = override.split("=", 1)
        _assign(config, key, yaml.safe_load(raw_value))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "loader", "geometry", "mtg", "gde", "registration", "runtime"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")
    size = config["data"].get("image_size")
    if not isinstance(size, list) or len(size) != 2 or any(int(value) <= 0 for value in size):
        raise ValueError("data.image_size must contain two positive integers")
    if float(config["gde"].get("tau", 0)) <= 0:
        raise ValueError("gde.tau must be positive")


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
