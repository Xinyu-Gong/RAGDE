from pathlib import Path

from ragde.utils.config import load_config


def test_dotted_override() -> None:
    path = Path(__file__).parents[1] / "configs" / "ragde.yaml"
    config = load_config(path, ["loader.batch_size=2", "device=cpu"])
    assert config["loader"]["batch_size"] == 2
    assert config["device"] == "cpu"
