"""Build the timelines the models read: every person's events with their ages."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src._config import (
    DEFAULT_CONFIG_DIR,
    DEMENTIA_CODE_PREFIXES,
    EARLIEST_REGISTER_DATE,
    EVENT_KINDS,
    ICD10_CHAPTERS,
    MIN_PEOPLE_PER_CODE,
    MINIMUM_SPELL_DAYS,
    REGISTER_END,
    SPECIAL_TOKENS,
    VALIDATION_FOLD,
    VARIANTS,
)
from src.icd10 import icd10_chapter


def age_days(date: str) -> pl.Expr:
    """The age in days on the date in column date (needs BIRTH_DATE)."""
    return (pl.col(date) - pl.col("BIRTH_DATE")).dt.total_days().cast(pl.Int32)


def inside_record(date: str) -> pl.Expr:
    """The date in column date is between WINDOW_START and RECORD_END_DATE."""
    return pl.col(date).is_between(pl.col("WINDOW_START"), pl.col("RECORD_END_DATE"))


def make_persons(
    per_person_raw: pl.DataFrame, split: pl.DataFrame, index_age: int
) -> pl.DataFrame:
    """One row per person: split, index date and observation window."""
    # Columns selected: PNR, sex, dates of birth, death, emigration and dementia
    # Create columns:
    #   - SPLIT: test outside the training set of split.R, validation in its
    #     fold VALIDATION_FOLD, train otherwise
    #   - INDEX_DATE: birthday at index_age (29 February becomes 28 February,
    #     as R's %m+% does)
    #   - WINDOW_START: birth date, or EARLIEST_REGISTER_DATE if later
    #   - RECORD_END_DATE: earliest of death, emigration and the registers end
    #   - INDEX_AGE_DAYS: age in days at the index date
    #   - OBSERVED_UNTIL_AGE_DAYS: age in days at the end of the record, or at
    #     the index date for test people
    is_test = pl.col("SPLIT") == "test"
    return (
        per_person_raw.select(
            "PNR",
            "SEX",
            "BIRTH_DATE",
            "DEATH_DATE",
            "EMIGRATION_DATE",
            "DEMENTIA_INCIDENT_DATE",
        )
        .join(split.select("PNR", "in_training", "fold"), on="PNR")
        .with_columns(
            SPLIT=pl.when(~pl.col("in_training"))
            .then(pl.lit("test"))
            .when(pl.col("fold") == VALIDATION_FOLD)
            .then(pl.lit("validation"))
            .otherwise(pl.lit("train")),
            INDEX_DATE=pl.col("BIRTH_DATE").dt.offset_by(f"{index_age}y"),
            WINDOW_START=pl.max_horizontal(
                "BIRTH_DATE", pl.lit(EARLIEST_REGISTER_DATE)
            ),
            RECORD_END_DATE=pl.min_horizontal(
                "DEATH_DATE", "EMIGRATION_DATE", pl.lit(REGISTER_END)
            ),
        )
        .with_columns(
            INDEX_AGE_DAYS=age_days("INDEX_DATE"),
            OBSERVED_UNTIL_AGE_DAYS=pl.when(is_test)
            .then(
                pl.min_horizontal(age_days("RECORD_END_DATE"), age_days("INDEX_DATE"))
            )
            .otherwise(age_days("RECORD_END_DATE")),
        )
        .drop("in_training", "fold")
        .sort("PNR")
    )


def make_diagnoses(
    per_diagnosis_raw: pl.DataFrame, persons: pl.DataFrame
) -> pl.DataFrame:
    """One row per person, kind and three-character code, at its first contact in the record."""
    # Columns selected: PNR, admission date, diagnosis
    # Filter 1: known admission date and diagnosis
    # Filter 2: admission inside the record
    # Create columns:
    #   - KIND: dementia_code for a code of the dementia definition of script 2
    #     (read on the full code: F02.0 is one, F02.1 is not), diagnosis
    #     otherwise
    #   - EVENT: the first three characters of the diagnosis
    #   - AGE_DAYS: age at the first contact with that kind and code
    is_dementia_code = pl.any_horizontal(
        pl.col("DIAG").str.starts_with(prefix) for prefix in DEMENTIA_CODE_PREFIXES
    )
    return (
        per_diagnosis_raw.select("PNR", "INDATE", "DIAG")
        .filter(pl.col("INDATE").is_not_null(), pl.col("DIAG").is_not_null())
        .join(persons, on="PNR")
        .filter(inside_record("INDATE"))
        .group_by(
            "PNR",
            KIND=pl.when(is_dementia_code)
            .then(pl.lit("dementia_code"))
            .otherwise(pl.lit("diagnosis")),
            EVENT=pl.col("DIAG").str.slice(0, 3),
        )
        .agg(AGE_DAYS=age_days("INDATE").min())
        .select("PNR", "AGE_DAYS", "KIND", "EVENT")
    )


def make_outcomes(persons: pl.DataFrame) -> pl.DataFrame:
    """DEMENTIA at DEMENTIA_INCIDENT_DATE and DEATH at DEATH_DATE, inside the record."""
    return pl.concat(
        [
            persons.filter(inside_record("DEMENTIA_INCIDENT_DATE")).select(
                "PNR",
                AGE_DAYS=age_days("DEMENTIA_INCIDENT_DATE"),
                KIND=pl.lit("dementia"),
                EVENT=pl.lit("DEMENTIA"),
            ),
            persons.filter(inside_record("DEATH_DATE")).select(
                "PNR",
                AGE_DAYS=age_days("DEATH_DATE"),
                KIND=pl.lit("death"),
                EVENT=pl.lit("DEATH"),
            ),
        ]
    )


def make_abroad(per_crossing_raw: pl.DataFrame, persons: pl.DataFrame) -> pl.DataFrame:
    """ABROAD_START and ABROAD_END of each period abroad before the index date, as in strings.R."""
    # Border crossings, as in section 5a of strings.R
    # Columns selected: PNR, date, code (U out, I in)
    # Filter 1: people of the cohort, known date and code
    # Filter 2: no crossing that repeats the previous one, so a departure is
    #   never paired with another departure
    # Create columns:
    #   - NEXT_CODE, NEXT_DATE: the person's next crossing
    #   - IS_FIRST: the person's first crossing
    previous_code = pl.col("INDUD_KODE").shift(1).over("PNR")
    # One row per border crossing: PNR, HAEND_DATO, INDUD_KODE, NEXT_CODE,
    # NEXT_DATE, IS_FIRST
    crossings = (
        per_crossing_raw.select("PNR", "HAEND_DATO", "INDUD_KODE")
        .join(persons.select("PNR"), on="PNR", how="semi")
        .filter(
            pl.col("HAEND_DATO").is_not_null(), pl.col("INDUD_KODE").is_in(["U", "I"])
        )
        .sort("PNR", "HAEND_DATO")
        .filter(previous_code.is_null() | (pl.col("INDUD_KODE") != previous_code))
        .with_columns(
            NEXT_CODE=pl.col("INDUD_KODE").shift(-1).over("PNR"),
            NEXT_DATE=pl.col("HAEND_DATO").shift(-1).over("PNR"),
            IS_FIRST=pl.int_range(pl.len()).over("PNR") == 0,
        )
    )

    # Periods, as in section 5b of strings.R: a departure opens one, closed by
    # the next return or left open; a first crossing that is a return means
    # abroad since before the registers
    # Create columns:
    #   - START_DATE: departure, or WINDOW_START if earlier or unknown
    #   - END_DATE: return, or the index date if later or unknown; a period
    #     still open a year before the index date ends there
    # Filter: periods of at least MINIMUM_SPELL_DAYS
    # One row per departure: PNR, FROM_DATE, TO_DATE (null without a return)
    departures = crossings.filter(pl.col("INDUD_KODE") == "U").select(
        "PNR",
        FROM_DATE="HAEND_DATO",
        TO_DATE=pl.when(pl.col("NEXT_CODE") == "I").then(pl.col("NEXT_DATE")),
    )
    # One row per person whose first crossing is a return: PNR, FROM_DATE (null),
    # TO_DATE
    abroad_from_start = crossings.filter(
        pl.col("IS_FIRST"), pl.col("INDUD_KODE") == "I"
    ).select("PNR", FROM_DATE=pl.lit(None, pl.Date), TO_DATE="HAEND_DATO")
    # One row per period abroad: PNR, FROM_DATE, TO_DATE, the columns of
    # persons, START_DATE, END_DATE
    periods = (
        pl.concat([departures, abroad_from_start])
        .join(persons, on="PNR")
        .with_columns(
            START_DATE=pl.max_horizontal(
                pl.coalesce("FROM_DATE", "WINDOW_START"), "WINDOW_START"
            ),
            END_DATE=pl.min_horizontal(
                pl.coalesce("TO_DATE", "INDEX_DATE"), "INDEX_DATE"
            ),
        )
        .with_columns(
            END_DATE=pl.when(pl.col("TO_DATE").is_null())
            .then(
                pl.min_horizontal("END_DATE", pl.col("INDEX_DATE").dt.offset_by("-1y"))
            )
            .otherwise(pl.col("END_DATE"))
        )
        .filter(
            (pl.col("END_DATE") - pl.col("START_DATE")).dt.total_days()
            >= MINIMUM_SPELL_DAYS
        )
    )
    return pl.concat(
        [
            periods.select(
                "PNR",
                AGE_DAYS=age_days("START_DATE"),
                KIND=pl.lit("abroad"),
                EVENT=pl.lit("ABROAD_START"),
            ),
            periods.select(
                "PNR",
                AGE_DAYS=age_days("END_DATE"),
                KIND=pl.lit("abroad"),
                EVENT=pl.lit("ABROAD_END"),
            ),
        ]
    )


def make_sex(persons: pl.DataFrame) -> pl.DataFrame:
    """The person's sex as an event at age 0, as in Delphi."""
    return persons.filter(pl.col("SEX").is_not_null()).select(
        "PNR", AGE_DAYS=pl.lit(0, pl.Int32), KIND=pl.lit("sex"), EVENT="SEX"
    )


def make_timelines(
    per_diagnosis_raw: pl.DataFrame,
    per_crossing_raw: pl.DataFrame,
    persons: pl.DataFrame,
) -> pl.DataFrame:
    """One row per event: diagnoses, dementia, death, periods abroad and sex."""
    # Filter: test people's events before the index date only, so that their
    #   outcome never reaches the models
    # One row per event, everyone's whole record: PNR, AGE_DAYS, KIND, EVENT
    events = pl.concat(
        [
            make_diagnoses(per_diagnosis_raw, persons),
            make_outcomes(persons),
            make_abroad(per_crossing_raw, persons),
            make_sex(persons),
        ]
    )
    return (
        events.join(persons.select("PNR", "SPLIT", "INDEX_AGE_DAYS"), on="PNR")
        .filter(
            (pl.col("SPLIT") != "test")
            | (pl.col("AGE_DAYS") < pl.col("INDEX_AGE_DAYS"))
        )
        .select("PNR", "AGE_DAYS", "KIND", "EVENT")
        .sort("PNR", "AGE_DAYS", "KIND", "EVENT")
    )


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
    """Delphi's labels.csv: each line's NAME and its TOKEN number in Delphi's files."""
    # Create columns:
    #   - NAME: the first word of the line, as in Delphi (A00, Female), with
    #     Delphi's Death named DEATH, as in our timelines
    #   - TOKEN: the line's index minus 1, as Delphi's data has it: Female is 1
    #     and Death 1268. Delphi adds 1 when it builds batches, so that 0 is
    #     padding and 1 is no event
    lines = path.read_text().splitlines()
    return pl.DataFrame(
        {
            "NAME": [line.split(" ")[0] for line in lines],
            "TOKEN": range(-1, len(lines) - 1),
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
    """Write a variant's files: Delphi's (delphi) or ours (events.parquet, vocab.csv)."""
    folder.mkdir(exist_ok=True)
    if choice == "ours":
        tokens.write_parquet(folder / "events.parquet")
        vocabulary.write_csv(folder / "vocab.csv")
        return
    # Delphi's format: train.bin, val.bin and test.bin, uint32 rows of (patient
    # id, age in days, token), the rows of a patient together. The patient id is
    # the person's row in persons (sorted by PNR); patients.parquet maps it back
    # to the PNR
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
            .select("PATIENT_ID", "AGE_DAYS", "TOKEN")
            .to_numpy()
            .astype(np.uint32)
        )
        array.tofile(folder / f"{name}.bin")
    patients.select("PATIENT_ID", "PNR").write_parquet(folder / "patients.parquet")


def parse_args() -> Namespace:
    """The command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", help="settings file in configs/, without .yaml: synthetic or server"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(DEFAULT_CONFIG_DIR / f"{args.config}.yaml") as file:
        config = yaml.safe_load(file)
    root = Path(config["root"])
    paths = {name: root / path for name, path in config["paths"].items()}

    print("Reading the data...", end=" ")
    # One row per person (script 2): PNR, SEX, BIRTH_DATE, DEATH_DATE,
    # EMIGRATION_DATE, DEMENTIA_INCIDENT_DATE, EVENT_TYPE, ...
    per_person_raw = pl.read_parquet(paths["persons"])
    # One row per diagnosis at a hospital contact (script 2): PNR, INDATE, DIAG
    per_diagnosis_raw = pl.read_parquet(
        paths["contacts"], columns=["PNR", "INDATE", "DIAG"]
    )
    # One row per border crossing (script 2): PNR, HAEND_DATO, INDUD_KODE (U out,
    # I in)
    per_crossing_raw = pl.read_parquet(paths["migration"])
    # One row per person (split.R): PNR, in_training, fold
    split = pl.read_parquet(paths["split"])
    print("Done")

    print("Making the timelines...", end=" ")
    # One row per person: PNR, SEX, BIRTH_DATE, DEATH_DATE, EMIGRATION_DATE,
    # DEMENTIA_INCIDENT_DATE, SPLIT, INDEX_DATE, WINDOW_START, RECORD_END_DATE,
    # INDEX_AGE_DAYS, OBSERVED_UNTIL_AGE_DAYS
    persons = make_persons(per_person_raw, split, config["index_age"])
    # One row per event: PNR, AGE_DAYS, KIND, EVENT
    timelines = make_timelines(per_diagnosis_raw, per_crossing_raw, persons)
    print("Done")

    # Every person has a split, and no test person has an event from the index
    # date on
    assert persons.height == per_person_raw.height
    assert (
        timelines.join(persons.filter(pl.col("SPLIT") == "test"), on="PNR")
        .filter(pl.col("AGE_DAYS") >= pl.col("INDEX_AGE_DAYS"))
        .is_empty()
    )

    paths["tokens"].mkdir(parents=True, exist_ok=True)
    persons.select(
        "PNR", "SPLIT", "INDEX_AGE_DAYS", "OBSERVED_UNTIL_AGE_DAYS"
    ).write_parquet(paths["tokens"] / "persons.parquet")
    timelines.write_parquet(paths["tokens"] / "timelines.parquet")

    print(f"{persons.height} people, {timelines.height} events -> {paths['tokens']}")
    print(persons["SPLIT"].value_counts().sort("SPLIT"))
    print(timelines["KIND"].value_counts().sort("KIND"))

    # Every variant: the five steps, with the choices of its row in VARIANTS
    # One row per line of Delphi's labels.csv: NAME, TOKEN
    labels = read_delphi_labels(paths["delphi_labels"])
    for variant, choices in VARIANTS.items():
        # One row per event of the variant: PNR, AGE_DAYS, KIND, EVENT
        events = select_events(timelines, choices["events"])
        events = select_codes(events, choices["codes"], labels["NAME"].to_list())
        if choices["cut_at_index"]:
            events = cut_at_index(events, persons)
        # One row per token of the variant's vocabulary: TOKEN, NAME
        vocabulary = make_vocabulary(events, persons, choices["vocabulary"], labels)
        # One row per token of a person: PNR, AGE_DAYS, TOKEN
        tokens = tokenise(events, vocabulary)
        folder = paths["tokens"] / variant
        write(tokens, vocabulary, persons, folder, choices["vocabulary"])
        print(
            f"{variant}: {tokens.height} events, vocabulary of {vocabulary.height} -> {folder}"
        )


if __name__ == "__main__":
    main()
