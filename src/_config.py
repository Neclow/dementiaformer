"""Project-wide constants."""

import datetime as dt

from pathlib import Path
from typing import Final

# Folder of the settings files, relative to the repository root
DEFAULT_CONFIG_DIR: Final = Path("configs")
# Last register date, as in 5 predictions.R
REGISTER_END: Final = dt.date(2022, 12, 31)
# The 22 ICD-10 chapters, as in 3 strings.R: (first code, last code, name). A
# three-character code belongs to the chapter whose range holds it
ICD10_CHAPTERS: Final = [
    ("A00", "B99", "INFECTIOUS"),
    ("C00", "D48", "NEOPLASMS"),
    ("D50", "D89", "BLOOD_IMMUNE"),
    ("E00", "E90", "ENDOCRINE"),
    ("F00", "F99", "MENTAL"),
    ("G00", "G99", "NERVOUS"),
    ("H00", "H59", "EYE"),
    ("H60", "H95", "EAR"),
    ("I00", "I99", "CIRCULATORY"),
    ("J00", "J99", "RESPIRATORY"),
    ("K00", "K93", "DIGESTIVE"),
    ("L00", "L99", "SKIN"),
    ("M00", "M99", "MUSCULOSKELETAL"),
    ("N00", "N99", "GENITOURINARY"),
    ("O00", "O99", "PREGNANCY"),
    ("P00", "P96", "PERINATAL"),
    ("Q00", "Q99", "CONGENITAL"),
    ("R00", "R99", "SYMPTOMS"),
    ("S00", "T98", "INJURY"),
    ("V01", "Y98", "EXTERNAL_CAUSES"),
    ("Z00", "Z99", "HEALTH_SERVICES"),
    ("U00", "U99", "SPECIAL_PURPOSES"),
]
# No event before this date, as in 3 strings.R (start of the Psychiatric
# Central Research Register)
EARLIEST_REGISTER_DATE: Final = dt.date(1969, 1, 1)
# A shorter period abroad gets no event, as in 3 strings.R
MINIMUM_SPELL_DAYS: Final = 180
# The fold of the training set held out to validate the models
VALIDATION_FOLD: Final = 5
