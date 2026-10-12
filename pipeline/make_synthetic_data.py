"""Make the synthetic data in the layout the pipeline reads, at a given index age."""

# Writes data/synthetic_age<age>/. Decisions: docs/pipeline.md. Run from the
# repository root:
#   pixi run make_synthetic_data
#   python -m pipeline.make_synthetic_data --index-age 65

import datetime as dt
import io
import zipfile

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import polars as pl

RAW = Path("data/raw")
# Folder inside synthetic.zip that holds the person and migration files
ZIP_FOLDER = (
    "synthetic/data/workdata/709816/Personal folders/Mathilde/Projects/"
    "p1_transformers/Data/"
)
# Last register date, as in 5 predictions.R
REGISTER_END = dt.date(2022, 12, 31)
# The dementia codes of make_synthetic_data.R
# fmt: off
DEMENTIA_CODES = [
    "F000", "F001", "F009", "F010", "F019", "F020", "F039",
    "G30", "G301", "G309", "G318B", "G318E", "G319", "G310B",
]
# fmt: on
# Share of "out_after_65" movers in make_synthetic_data.R
EMIGRATION_SHARE = 0.03
SEED = 709816


def read_from_zip(name: str) -> pl.DataFrame:
    """Read one parquet file of synthetic.zip."""
    with zipfile.ZipFile(RAW / "synthetic.zip") as archive:
        return pl.read_parquet(io.BytesIO(archive.read(ZIP_FOLDER + name)))


def keep_until(
    contacts: pl.DataFrame, persons: pl.DataFrame, end_column: str
) -> pl.DataFrame:
    """The contacts on or before each person's date in end_column (all when null)."""
    return (
        contacts.join(persons.select("PNR", end_column), on="PNR")
        .filter(pl.col(end_column).is_null() | (pl.col("INDATE") <= pl.col(end_column)))
        .drop(end_column)
    )


def first_dementia_code(contacts: pl.DataFrame) -> pl.DataFrame:
    """DEMENTIA_INCIDENT_DATE: each person's first contact with a dementia code."""
    return (
        contacts.filter(pl.col("DIAG").is_in(DEMENTIA_CODES))
        .group_by("PNR")
        .agg(DEMENTIA_INCIDENT_DATE=pl.col("INDATE").min())
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-age", type=int, default=70, help="(default: %(default)s)"
    )
    index_age = parser.parse_args().index_age
    birthday = f"BDAY{index_age}"
    output = Path(f"data/synthetic_age{index_age}")
    rng = np.random.default_rng(SEED)

    # Person columns only, with BDAY<age>, and no death after the registers end
    persons = (
        read_from_zip("df_born1923_1943_short.parquet")
        .select("PNR", "SEX", "BIRTH_DATE", "BIRTH_COHORT", "EDU_CAT3", "DEATH_DATE")
        .with_columns(
            pl.col("BIRTH_DATE").dt.offset_by(f"{index_age}y").alias(birthday),
            DEATH_DATE=pl.when(pl.col("DEATH_DATE") <= REGISTER_END).then(
                pl.col("DEATH_DATE")
            ),
        )
    )
    # Contact columns only
    contacts = (
        pl.read_parquet(RAW / "df_born1923_1943.parquet")
        .select("PNR", "INDATE", "OUTDATE", "DIAG", "DIAG_TYPE", "PATIENT_TYPE")
        .filter(pl.col("INDATE").is_not_null(), pl.col("DIAG").is_not_null())
    )
    migration = read_from_zip("df_born1923_1943_migration.parquet")

    # No contact after death, then keep people alive and without dementia at
    # the index age
    contacts = keep_until(contacts, persons, "DEATH_DATE")
    persons = (
        persons.join(first_dementia_code(contacts), on="PNR", how="left")
        .filter(
            pl.col("DEATH_DATE").is_null() | (pl.col("DEATH_DATE") >= pl.col(birthday)),
            pl.col("DEMENTIA_INCIDENT_DATE").is_null()
            | (pl.col("DEMENTIA_INCIDENT_DATE") >= pl.col(birthday)),
        )
        .drop("DEMENTIA_INCIDENT_DATE")
    )

    # Departures on a random day between the index age and death or the end of
    # the registers
    emigrants = persons.sample(fraction=EMIGRATION_SHARE, seed=SEED)
    days_left = pl.min_horizontal("DEATH_DATE", pl.lit(REGISTER_END)) - pl.col(birthday)
    departures = emigrants.with_columns(
        DAYS=days_left.dt.total_days(),
        RANDOM=pl.Series(rng.random(emigrants.height)),
    ).select(
        "PNR",
        HAEND_DATO=pl.col(birthday)
        + pl.duration(days=(pl.col("RANDOM") * pl.col("DAYS")).floor().cast(pl.Int64)),
        INDUD_KODE=pl.lit("U"),
    )
    migration = pl.concat(
        [migration.join(persons.select("PNR"), on="PNR"), departures]
    ).sort("PNR", "HAEND_DATO")

    # No contact after emigration, then dementia, follow-up end and event type
    # from what is left
    persons = persons.join(
        departures.select("PNR", EMIGRATION_DATE="HAEND_DATO"), on="PNR", how="left"
    )
    contacts = keep_until(
        contacts.join(persons.select("PNR"), on="PNR"), persons, "EMIGRATION_DATE"
    )
    follow_up_end = pl.min_horizontal(
        "DEMENTIA_INCIDENT_DATE", "DEATH_DATE", "EMIGRATION_DATE", pl.lit(REGISTER_END)
    )
    persons = (
        persons.join(first_dementia_code(contacts), on="PNR", how="left")
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

    # None of the impossible dates may remain
    assert persons["PNR"].is_unique().all()
    assert persons.filter(pl.col("DEATH_DATE") > REGISTER_END).is_empty()
    assert persons.filter(
        pl.col("DEMENTIA_INCIDENT_DATE") > pl.col("DEATH_DATE")
    ).is_empty()
    assert persons.filter(pl.col("FOLLOW_UP_END_DATE") < pl.col(birthday)).is_empty()
    assert (
        contacts.join(persons, on="PNR")
        .filter(
            (pl.col("INDATE") > pl.col("DEATH_DATE"))
            | (pl.col("INDATE") > pl.col("EMIGRATION_DATE"))
        )
        .is_empty()
    )

    output.mkdir(parents=True, exist_ok=True)
    prefix = f"age{index_age}_df_born1923_1943"
    contacts.write_parquet(output / f"{prefix}.parquet")
    persons.write_parquet(output / f"{prefix}_short.parquet")
    migration.write_parquet(output / f"{prefix}_migration.parquet")

    print(f"{persons.height} people, {contacts.height} contacts -> {output}")
    print(persons["EVENT_TYPE"].value_counts().sort("EVENT_TYPE"))


if __name__ == "__main__":
    main()
