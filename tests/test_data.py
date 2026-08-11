from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ragde.data import PairedImageDataset


def _write(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 10), value, dtype=np.uint8)).save(path)


def test_dataset_pairs_by_stem_and_normalizes(tmp_path: Path) -> None:
    _write(tmp_path / "visible" / "scene_2.png", 255)
    _write(tmp_path / "visible" / "scene_1.png", 0)
    _write(tmp_path / "sar" / "scene_1.jpg", 127)
    _write(tmp_path / "sar" / "scene_2.jpg", 128)
    dataset = PairedImageDataset(tmp_path, "visible", "sar", image_size=(8, 6))
    sample = dataset[0]
    assert len(dataset) == 2
    assert sample["name"] == "scene_1"
    assert sample["fixed"].shape == (1, 8, 6)
    assert -1.0 <= float(sample["fixed"].min()) <= float(sample["fixed"].max()) <= 1.0


def test_dataset_rejects_unpaired_files(tmp_path: Path) -> None:
    _write(tmp_path / "x" / "one.png", 0)
    _write(tmp_path / "y" / "two.png", 0)
    with pytest.raises(RuntimeError, match="No matching"):
        PairedImageDataset(tmp_path, "x", "y")
