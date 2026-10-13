"""Make the synthetic data in the layout the pipeline reads, at a given index age."""

import io
import zipfile

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import polars as pl

from src._config import (
    DEFAULT_RAW_DIR,
    DEMENTIA_CODES,
    EMIGRATION_SHARE,
    REGISTER_END,
    SEED,
)

# Folder inside synthetic.zip that holds the person and migration files
ZIP_FOLDER = (
    "synthetic/data/workdata/709816/Personal folders/Mathilde/Projects/"
    "p1_transformers/Data/"
)


def read_from_synthetic_zip(name: str) -> pl.DataFrame:
    """Read one parquet file of synthetic.zip."""
    with zipfile.ZipFile(DEFAULT_RAW_DIR / "synthetic.zip") as archive:
        return pl.read_parquet(io.BytesIO(archive.read(ZIP_FOLDER + name)))


def first_dementia_code(
    per_diagnosis_raw: pl.DataFrame, until: pl.DataFrame
) -> pl.DataFrame:
    """Each person's first dementia code on or before their date in until (PNR, END)."""
    return (
        per_diagnosis_raw.filter(pl.col("DIAG").is_in(DEMENTIA_CODES))
        .join(until, on="PNR")
        .filter(pl.col("END").is_null() | (pl.col("INDATE") <= pl.col("END")))
        .group_by("PNR")
        .agg(DEMENTIA_INCIDENT_DATE=pl.col("INDATE").min())
    )


def make_persons(
    per_person_raw: pl.DataFrame,
    per_diagnosis_raw: pl.DataFrame,
    index_age: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """One row per person alive and without dementia at the index age."""
    birthday = f"BDAY{index_age}"

    # Person columns only, with BDAY<age>, and no death after the registers end
    # Columns selected: PNR, sex, birth date, birth cohort, education, death date
    # Create columns:
    #   - BDAY<age>: birth date + index_age years
    #   - DEATH_DATE: null if death after the registers end
    persons = per_person_raw.select(
        "PNR", "SEX", "BIRTH_DATE", "BIRTH_COHORT", "EDU_CAT3", "DEATH_DATE"
    ).with_columns(
        pl.col("BIRTH_DATE").dt.offset_by(f"{index_age}y").alias(birthday),
        DEATH_DATE=pl.when(pl.col("DEATH_DATE") <= REGISTER_END).then(
            pl.col("DEATH_DATE")
        ),
    )

    # The cohort
    # Filter 1: alive at the index age
    # Filter 2: no dementia code before the index age (codes after death ignored)
    until_death = persons.select("PNR", END="DEATH_DATE")
    persons = (
        persons.join(
            first_dementia_code(per_diagnosis_raw, until_death), on="PNR", how="left"
        )
        .filter(
            pl.col("DEATH_DATE").is_null() | (pl.col("DEATH_DATE") >= pl.col(birthday)),
            pl.col("DEMENTIA_INCIDENT_DATE").is_null()
            | (pl.col("DEMENTIA_INCIDENT_DATE") >= pl.col(birthday)),
        )
        .drop("DEMENTIA_INCIDENT_DATE")
    )

    # Emigration
    # Create columns:
    #   - EMIGRATION_DATE: for EMIGRATION_SHARE of the people, a random day
    #     between the index age and death or the registers end; null otherwise
    emigrants = persons.sample(fraction=EMIGRATION_SHARE, seed=SEED)
    days_left = pl.min_horizontal("DEATH_DATE", pl.lit(REGISTER_END)) - pl.col(birthday)
    emigration = emigrants.with_columns(
        DAYS=days_left.dt.total_days(),
        RANDOM=pl.Series(rng.random(emigrants.height)),
    ).select(
        "PNR",
        EMIGRATION_DATE=pl.col(birthday)
        + pl.duration(days=(pl.col("RANDOM") * pl.col("DAYS")).floor().cast(pl.Int64)),
    )
    persons = persons.join(emigration, on="PNR", how="left")

    # Outcome
    # Create columns:
    #   - DEMENTIA_INCIDENT_DATE: first dementia code before death and emigration
    #   - DEMENTIA: 1 if there is a DEMENTIA_INCIDENT_DATE, 0 otherwise
    #   - FOLLOW_UP_END_DATE: earliest of dementia, death, emigration, registers end
    #   - EVENT_TYPE: dementia, death or censored, whichever ends the follow-up
    until_death_or_emigration = persons.select(
        "PNR", END=pl.min_horizontal("DEATH_DATE", "EMIGRATION_DATE")
    )
    follow_up_end = pl.min_horizontal(
        "DEMENTIA_INCIDENT_DATE", "DEATH_DATE", "EMIGRATION_DATE", pl.lit(REGISTER_END)
    )
    return (
        persons.join(
            first_dementia_code(per_diagnosis_raw, until_death_or_emigration),
            on="PNR",
            how="left",
        )
        .with_columns(
            DEMENTIA=pl.col("DEMENTIA_INCIDENT_DATE").is_not_null().cast(pl.Int32),
            FOLLOW_UP_END_DATE=follow_up_end,
        )
        .with_columns(
            EVENT_TYPE=pl.when(
                pl.col("FOLLOW_UP_END_DATE") == pl.col("DEMENTIA_INCIDENT_DATE")
            )
            .then(pl.lit("dementia"))
            .when(pl.col("FOLLOW_UP_END_DATE") == pl.col("DEATH_DATE"))
            .then(pl.lit("death"))
            .otherwise(pl.lit("censored"))
        )
        .sort("PNR")
    )


def make_contacts(
    per_diagnosis_raw: pl.DataFrame, persons: pl.DataFrame
) -> pl.DataFrame:
    """One row per diagnosis at a hospital contact of the cohort."""
    # Columns selected: PNR, hospital admission and discharge dates, diagnosis, diagnosis type, patient type
    # Filter 1: no null admission date or diagnosis
    # Filter 2: people of the cohort only
    # Filter 3: no contact after death or emigration
    until_death_or_emigration = persons.select(
        "PNR", END=pl.min_horizontal("DEATH_DATE", "EMIGRATION_DATE")
    )
    return (
        per_diagnosis_raw.select(
            "PNR", "INDATE", "OUTDATE", "DIAG", "DIAG_TYPE", "PATIENT_TYPE"
        )
        .filter(pl.col("INDATE").is_not_null(), pl.col("DIAG").is_not_null())
        .join(until_death_or_emigration, on="PNR")
        .filter(pl.col("END").is_null() | (pl.col("INDATE") <= pl.col("END")))
        .drop("END")
    )


def make_migration(
    per_crossing_raw: pl.DataFrame, persons: pl.DataFrame
) -> pl.DataFrame:
    """One row per border crossing of the cohort."""
    # Filter 1: people of the cohort only
    # Add rows: a departure (INDUD_KODE U) on each EMIGRATION_DATE
    departures = persons.filter(pl.col("EMIGRATION_DATE").is_not_null()).select(
        "PNR", HAEND_DATO="EMIGRATION_DATE", INDUD_KODE=pl.lit("U")
    )
    return pl.concat(
        [per_crossing_raw.join(persons.select("PNR"), on="PNR"), departures]
    ).sort("PNR", "HAEND_DATO")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-age", type=int, default=70, help="(default: %(default)s)"
    )
    index_age = parser.parse_args().index_age
    output = Path(f"data/synthetic_age{index_age}")
    rng = np.random.default_rng(SEED)

    print("Reading the raw data...", end=" ")
    per_person_raw = read_from_synthetic_zip("df_born1923_1943_short.parquet")
    per_diagnosis_raw = pl.read_parquet(DEFAULT_RAW_DIR / "df_born1923_1943.parquet")
    per_crossing_raw = read_from_synthetic_zip("df_born1923_1943_migration.parquet")
    print("Done")

    print(f"Making the synthetic data for index age {index_age}...", end=" ")
    persons = make_persons(per_person_raw, per_diagnosis_raw, index_age, rng)
    contacts = make_contacts(per_diagnosis_raw, persons)
    migration = make_migration(per_crossing_raw, persons)
    print("Done")

    # None of the impossible dates may remain
    print("Checking the cohort and contacts for impossible dates...", end=" ")
    assert persons["PNR"].is_unique().all()
    assert persons.filter(pl.col("DEATH_DATE") > REGISTER_END).is_empty()
    assert persons.filter(
        pl.col("DEMENTIA_INCIDENT_DATE") > pl.col("DEATH_DATE")
    ).is_empty()
    assert persons.filter(
        pl.col("FOLLOW_UP_END_DATE") < pl.col(f"BDAY{index_age}")
    ).is_empty()
    assert (
        contacts.join(persons, on="PNR")
        .filter(
            (pl.col("INDATE") > pl.col("DEATH_DATE"))
            | (pl.col("INDATE") > pl.col("EMIGRATION_DATE"))
        )
        .is_empty()
    )
    print("Done")

    output.mkdir(parents=True, exist_ok=True)
    prefix = f"age{index_age}_df_born1923_1943"
    contacts.write_parquet(output / f"{prefix}.parquet")
    persons.write_parquet(output / f"{prefix}_short.parquet")
    migration.write_parquet(output / f"{prefix}_migration.parquet")

    print(f"{persons.height} people, {contacts.height} contacts -> {output}")
    print(persons["EVENT_TYPE"].value_counts().sort("EVENT_TYPE"))
    print("Done")


if __name__ == "__main__":
    main()
