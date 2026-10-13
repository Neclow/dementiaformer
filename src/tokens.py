"""The tokenisations: each variant's events, vocabulary, token numbers and files."""

from pathlib import Path

import numpy as np
import polars as pl

from src._config import EVENT_KINDS, ICD10_CHAPTERS, MIN_PEOPLE_PER_CODE, SPECIAL_TOKENS
from src.icd10 import icd10_chapter


def select_events(timelines: pl.DataFrame, choice: str) -> pl.DataFrame:
    """Delphi's events or ours (EVENT_KINDS): one row per PNR, AGE_DAYS, KIND and EVENT."""
    # Filter: the kinds of events of EVENT_KINDS[choice]
    # Create columns:
    #   - KIND: diagnosis for the dementia codes too (kept by Delphi's events)
    return timelines.filter(pl.col("KIND").is_in(EVENT_KINDS[choice])).with_columns(
        KIND=pl.when(pl.col("KIND") == "dementia_code")
        .then(pl.lit("diagnosis"))
        .otherwise(pl.col("KIND"))
    )


def select_codes(
    events: pl.DataFrame, choice: str, delphi_codes: list[str]
) -> pl.DataFrame:
    """The diagnoses with a code of Delphi's labels.csv only (delphi), or all of them (ours)."""
    # Filter: for delphi, diagnoses with a code of delphi_codes only, as Delphi
    #   drops the others
    if choice == "ours":
        return events
    return events.filter(
        (pl.col("KIND") != "diagnosis") | pl.col("EVENT").is_in(delphi_codes)
    )


def cut_at_index(events: pl.DataFrame, persons: pl.DataFrame) -> pl.DataFrame:
    """Everyone's events before the index date only (test people's already stop there)."""
    return (
        events.join(persons.select("PNR", "INDEX_AGE_DAYS"), on="PNR")
        .filter(pl.col("AGE_DAYS") < pl.col("INDEX_AGE_DAYS"))
        .select("PNR", "AGE_DAYS", "KIND", "EVENT")
    )


def make_vocabulary(
    events: pl.DataFrame,
    persons: pl.DataFrame,
    choice: str,
    labels: pl.DataFrame | None,
) -> pl.DataFrame:
    """Delphi's labels.csv (delphi), or ours from a variant's events: one row per token, TOKEN and NAME."""
    if choice == "delphi":
        assert labels is not None
        return labels
    # Ours. Create rows, in this order:
    #   - SPECIAL_TOKENS: PAD is 0, NO_EVENT is 1
    #   - RARE_<chapter>: one per ICD-10 chapter, for the rarer codes
    #   - every code that at least MIN_PEOPLE_PER_CODE training people have,
    #     in alphabetical order (validation and test people are not counted)
    # The codes with their own token, in alphabetical order
    codes = (
        events.filter(pl.col("KIND") == "diagnosis")
        .join(persons.filter(pl.col("SPLIT") == "train").select("PNR"), on="PNR")
        .group_by("EVENT")
        .agg(PEOPLE=pl.col("PNR").n_unique())
        .filter(pl.col("PEOPLE") >= MIN_PEOPLE_PER_CODE)
        .sort("EVENT")["EVENT"]
        .to_list()
    )
    rare = [f"RARE_{name}" for _, _, name in ICD10_CHAPTERS]
    names = [*SPECIAL_TOKENS, *rare, *codes]
    return pl.DataFrame(
        {"TOKEN": range(len(names)), "NAME": names},
        schema={"TOKEN": pl.Int32, "NAME": pl.String},
    )


def tokenise(events: pl.DataFrame, vocabulary: pl.DataFrame) -> pl.DataFrame:
    """A variant's events as token numbers of a vocabulary: one row per PNR, AGE_DAYS and TOKEN."""
    # Create columns:
    #   - NAME: the event, or for a code without its own token RARE_<chapter>,
    #     or UNKNOWN_CODE outside every chapter
    #   - TOKEN: the number of NAME in the vocabulary
    # Filter: each token at its first age only, since the rarer codes of a
    #   chapter share one token
    # Sort: by person, age and token (DEATH comes last: it ends the record)
    has_token = pl.col("EVENT").is_in(vocabulary["NAME"].to_list())
    chapter = icd10_chapter(pl.col("EVENT"))
    # One row per event: PNR, AGE_DAYS, KIND, EVENT, NAME, TOKEN
    named = events.with_columns(
        NAME=pl.when((pl.col("KIND") != "diagnosis") | has_token)
        .then(pl.col("EVENT"))
        .when(chapter.is_not_null())
        .then(pl.lit("RARE_") + chapter)
        .otherwise(pl.lit("UNKNOWN_CODE"))
    ).join(vocabulary, on="NAME", how="left")
    # Every event has a token
    assert named["TOKEN"].null_count() == 0, named.filter(pl.col("TOKEN").is_null())
    return (
        named.group_by("PNR", "TOKEN")
        .agg(pl.col("AGE_DAYS").min())
        .select("PNR", "AGE_DAYS", "TOKEN")
        .sort("PNR", "AGE_DAYS", "TOKEN")
    )


def read_delphi_labels(path: Path) -> pl.DataFrame:
    """Delphi's labels.csv: each line's NAME and TOKEN number."""
    # Create columns:
    #   - NAME: the first word of the line, as in Delphi (A00, Female), with
    #     Delphi's Death named DEATH, as in our timelines
    #   - TOKEN: the line's index, as in Delphi's batches: 0 is padding, 1 no
    #     event, Female 2 and Death 1269
    lines = path.read_text(encoding="utf-8").splitlines()
    return pl.DataFrame(
        {
            "NAME": [line.split(" ")[0] for line in lines],
            "TOKEN": range(len(lines)),
        },
        schema={"NAME": pl.String, "TOKEN": pl.Int32},
    ).with_columns(NAME=pl.col("NAME").replace("Death", "DEATH"))


def write(
    tokens: pl.DataFrame,
    vocabulary: pl.DataFrame,
    persons: pl.DataFrame,
    folder: Path,
    choice: str,
) -> None:
    """Write a variant's events.parquet and vocab.csv, and Delphi's files for Delphi's vocabulary."""
    folder.mkdir(exist_ok=True)
    tokens.write_parquet(folder / "events.parquet")
    vocabulary.write_csv(folder / "vocab.csv")
    if choice == "ours":
        return
    # Delphi's format, for Delphi's own code: train.bin, val.bin and test.bin,
    # uint32 rows of (patient id, age in days, token), the rows of a patient
    # together. The patient id is the person's row in persons (sorted by PNR);
    # patients.parquet maps it back to the PNR. Tokens are one less than ours
    # (Female 1, Death 1268): Delphi adds 1 when it builds batches
    # One row per person: PATIENT_ID (the row number), PNR, SPLIT
    patients = persons.select("PNR", "SPLIT").with_row_index("PATIENT_ID")
    patient_id = dict(patients.select("PNR", "PATIENT_ID").iter_rows())
    # One row per token of a person: PNR, AGE_DAYS, TOKEN, PATIENT_ID, SPLIT
    rows = tokens.with_columns(
        PATIENT_ID=pl.col("PNR").replace_strict(patient_id, return_dtype=pl.UInt32)
    ).join(patients.select("PNR", "SPLIT"), on="PNR", maintain_order="left")
    for split, name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        # One row per token of a person of the split: patient id, age in days,
        # token (uint32)
        array = (
            rows.filter(pl.col("SPLIT") == split)
            .select("PATIENT_ID", "AGE_DAYS", pl.col("TOKEN") - 1)
            .to_numpy()
            .astype(np.uint32)
        )
        array.tofile(folder / f"{name}.bin")
    patients.select("PATIENT_ID", "PNR").write_parquet(folder / "patients.parquet")
