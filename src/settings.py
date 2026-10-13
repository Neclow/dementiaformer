"""Reading the settings files: of the data (synthetic or server) and of the models."""

from pathlib import Path
from typing import Any

import yaml

from src._config import DATA_CONFIG_DIR, MODEL_CONFIG_DIR


def read_data_settings(name: str) -> tuple[dict[str, Any], dict[str, Path]]:
    """A data settings file (synthetic or server), and its paths joined to its root."""
    # In UTF-8: Windows would otherwise read it in its own encoding, and garble
    # paths with æ, ø or å
    with open(DATA_CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    root = Path(settings["root"])
    return settings, {key: root / path for key, path in settings["paths"].items()}


def read_model_settings(name: str) -> dict[str, Any]:
    """A model's settings file."""
    with open(MODEL_CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)
