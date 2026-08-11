"""Datasets for spatially aligned multimodal image pairs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in files:
            raise RuntimeError(f"Duplicate image stem '{path.stem}' in {directory}")
        files[path.stem] = path
    if not files:
        raise RuntimeError(f"No supported images found in {directory}")
    return files


def read_grayscale(path: str | Path, size: Iterable[int] | None = None) -> torch.Tensor:
    """Read an image as a ``[1, H, W]`` float tensor in ``[-1, 1]``."""
    with Image.open(path) as image:
        image = image.convert("L")
        if size is not None:
            height, width = (int(value) for value in size)
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).unsqueeze(0).div_(127.5).sub_(1.0)


class PairedImageDataset(Dataset):
    """Load aligned modality pairs matched by filename stem.

    ``root/fixed_dir/0001.png`` is paired with
    ``root/moving_dir/0001.png``. Different extensions are allowed.
    """

    def __init__(
        self,
        root: str | Path,
        fixed_dir: str = "fixed",
        moving_dir: str = "moving",
        image_size: Iterable[int] = (256, 256),
        strict_pairs: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        fixed = _images(self.root / fixed_dir)
        moving = _images(self.root / moving_dir)
        shared = sorted(fixed.keys() & moving.keys())
        if not shared:
            raise RuntimeError(
                f"No matching filename stems in {self.root / fixed_dir} and "
                f"{self.root / moving_dir}"
            )
        if strict_pairs and (set(fixed) != set(moving)):
            missing_fixed = sorted(set(moving) - set(fixed))[:5]
            missing_moving = sorted(set(fixed) - set(moving))[:5]
            raise RuntimeError(
                "Unpaired files detected. "
                f"Missing fixed stems: {missing_fixed}; missing moving stems: {missing_moving}"
            )
        self.pairs = [(fixed[key], moving[key]) for key in shared]
        self.image_size = tuple(int(value) for value in image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        fixed_path, moving_path = self.pairs[index]
        return {
            "fixed": read_grayscale(fixed_path, self.image_size),
            "moving": read_grayscale(moving_path, self.image_size),
            "name": fixed_path.stem,
            "fixed_path": str(fixed_path),
            "moving_path": str(moving_path),
        }
