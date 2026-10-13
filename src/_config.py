"""Project-wide constants."""

import datetime as dt

from pathlib import Path
from typing import Final

# Folder of the TOML settings files, relative to the repository root
DEFAULT_CONFIG_DIR: Final = Path("configs")
# Folder of the user's raw data, relative to the repository root
DEFAULT_RAW_DIR: Final = Path("data/raw")
# Last register date, as in 5 predictions.R
REGISTER_END: Final = dt.date(2022, 12, 31)
# Dementia codes
DEMENTIA_CODES: Final = [
    "F000",
    "F001",
    "F009",
    "F010",
    "F019",
    "F020",
    "F039",
    "G30",
    "G301",
    "G309",
    "G318B",
    "G318E",
    "G319",
    "G310B",
]
# Random emigrants share in make_synthetic_data.R
EMIGRATION_SHARE: Final = 0.03
# Random seed
SEED: Final = 709816
