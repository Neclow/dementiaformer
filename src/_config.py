"""Project-wide constants."""

from pathlib import Path
from typing import Final

# Folder of the TOML settings files, relative to the repository root
DEFAULT_CONFIG_DIR: Final = Path("configs")
# Folder of the user's raw data, relative to the repository root
DEFAULT_RAW_DIR: Final = Path("data/raw")
