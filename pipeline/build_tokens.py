"""Build the timelines the models read: every person's events with their ages."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import polars as pl
import yaml

from src._config import (
    DEFAULT_CONFIG_DIR,
    EARLIEST_REGISTER_DATE,
    MINIMUM_SPELL_DAYS,
    REGISTER_END,
    VALIDATION_FOLD,
)


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
    """One row per person and three-character code, at its first contact in the record."""
    # Columns selected: PNR, admission date, diagnosis
    # Filter 1: known admission date and diagnosis
    # Filter 2: admission inside the record
    # Create columns:
    #   - EVENT: the first three characters of the diagnosis
    #   - AGE_DAYS: age at the first contact with that code
    return (
        per_diagnosis_raw.select("PNR", "INDATE", "DIAG")
        .filter(pl.col("INDATE").is_not_null(), pl.col("DIAG").is_not_null())
        .join(persons, on="PNR")
        .filter(inside_record("INDATE"))
        .group_by("PNR", EVENT=pl.col("DIAG").str.slice(0, 3))
        .agg(AGE_DAYS=age_days("INDATE").min())
        .select("PNR", "AGE_DAYS", KIND=pl.lit("diagnosis"), EVENT="EVENT")
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
    departures = crossings.filter(pl.col("INDUD_KODE") == "U").select(
        "PNR",
        FROM_DATE="HAEND_DATO",
        TO_DATE=pl.when(pl.col("NEXT_CODE") == "I").then(pl.col("NEXT_DATE")),
    )
    abroad_from_start = crossings.filter(
        pl.col("IS_FIRST"), pl.col("INDUD_KODE") == "I"
    ).select("PNR", FROM_DATE=pl.lit(None, pl.Date), TO_DATE="HAEND_DATO")
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


def parse_args() -> Namespace:
    """The command-line arguments."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "synthetic.yaml",
        help="(default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as file:
        config = yaml.safe_load(file)
    root = Path(config["root"])
    paths = {name: root / path for name, path in config["paths"].items()}

    print("Reading the data...", end=" ")
    per_person_raw = pl.read_parquet(paths["persons"])
    per_diagnosis_raw = pl.read_parquet(
        paths["contacts"], columns=["PNR", "INDATE", "DIAG"]
    )
    per_crossing_raw = pl.read_parquet(paths["migration"])
    split = pl.read_parquet(paths["split"])
    print("Done")

    print("Making the timelines...", end=" ")
    persons = make_persons(per_person_raw, split, config["index_age"])
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


if __name__ == "__main__":
    main()
