"""Project-wide constants."""

import datetime as dt

from pathlib import Path
from typing import Any, Final

# Folder of the settings files, relative to the repository root: of the data
# (one per environment: synthetic or server) and of the models
DEFAULT_CONFIG_DIR: Final = Path("configs")
DATA_CONFIG_DIR: Final = DEFAULT_CONFIG_DIR / "data"
MODEL_CONFIG_DIR: Final = DEFAULT_CONFIG_DIR / "models"
# Folder of the plots, relative to the repository root
DEFAULT_IMG_DIR: Final = Path("img")
# Delphi's vocabulary, relative to the repository root: labels.csv of
# github.com/gerstung-lab/Delphi at commit fb72166 (data/ukb_real_data/labels.csv)
DELPHI_LABELS: Final = Path("data/labels.csv")
# CPU threads for PyTorch, unless --threads says otherwise: the server is shared
DEFAULT_N_THREADS: Final = 16
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
# Tokens with a fixed number, in this order: PAD (0) fills the end of short
# timelines in a batch, NO_EVENT (1) marks an age at which nothing happened
SPECIAL_TOKENS: Final = [
    "PAD",
    "NO_EVENT",
    "DEATH",
    "DEMENTIA",
    "ABROAD_START",
    "ABROAD_END",
    "UNKNOWN_CODE",
    # Sex, at age 0
    "Female",
    "Male",
]
# A code in fewer training people shares its chapter's token (the minimum
# cell count of 5 predictions.R)
MIN_PEOPLE_PER_CODE: Final = 5
# Diagnoses that define dementia, as in the user's script 2 (Hvidberg et al.
# 2016): a code is a dementia code when it starts with one of these
DEMENTIA_CODE_PREFIXES: Final = [
    "F00",
    "F01",
    "F020",
    "F039",
    "G30",
    "G310B",
    "G318B",
    "G318E",
    "G319",
]
# The kinds of timeline events: Delphi's (diagnoses with dementia codes, sex,
# death) or ours (diagnoses without dementia codes, since DEMENTIA marks
# dementia; DEMENTIA, death, periods abroad, sex)
EVENT_KINDS: Final = {
    "delphi": ["diagnosis", "dementia_code", "sex", "death"],
    "ours": ["diagnosis", "dementia", "death", "abroad", "sex"],
}
# The tokenisation variants: whose events, codes (Delphi's list or all) and
# vocabulary each uses, and whether everyone's timeline stops before the index
# date. The files are written in the format of the vocabulary
VARIANTS: Final[dict[str, dict[str, Any]]] = {
    "delphi": {
        "events": "delphi",
        "codes": "delphi",
        "vocabulary": "delphi",
        "cut_at_index": False,
    },
    "delphi_min_people": {
        "events": "delphi",
        "codes": "delphi",
        "vocabulary": "ours",
        "cut_at_index": False,
    },
    "delphi_all_codes": {
        "events": "delphi",
        "codes": "ours",
        "vocabulary": "ours",
        "cut_at_index": False,
    },
    "whole_lives": {
        "events": "ours",
        "codes": "ours",
        "vocabulary": "ours",
        "cut_at_index": False,
    },
    "cut_at_index": {
        "events": "ours",
        "codes": "ours",
        "vocabulary": "ours",
        "cut_at_index": True,
    },
}
