"""Make the synthetic data in the layout the pipeline reads, at a given index age."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src._config import (
    DEFAULT_CONFIG_DIR,
    DEMENTIA_CODES,
    EMIGRATION_SHARE,
    ICD10_CHAPTERS,
    REGISTER_END,
    SEED,
)


def icd10_chapter(code: pl.Expr) -> pl.Expr:
    """The ICD10_CHAPTERS name of a three-character code, null outside every chapter."""
    return pl.coalesce(
        pl.when(code.is_between(pl.lit(first), pl.lit(last))).then(pl.lit(name))
        for first, last, name in ICD10_CHAPTERS
    )


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
    persons = (
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
    )

    # Chapter flags, as script 2 writes them: 3 strings.R compares them with its
    # own in check 11f
    # Create columns:
    #   - CHAPTER_<name>: 1 if a contact before the index age has a code of that
    #     chapter, 0 otherwise
    chapters_before_index = (
        per_diagnosis_raw.join(persons.select("PNR", birthday), on="PNR")
        .filter(pl.col("INDATE") < pl.col(birthday))
        .select("PNR", CHAPTER=icd10_chapter(pl.col("DIAG").str.slice(0, 3)))
        .drop_nulls()
        .unique()
    )
    persons = persons.with_columns(
        pl.col("PNR")
        .is_in(chapters_before_index.filter(pl.col("CHAPTER") == name)["PNR"].to_list())
        .cast(pl.Int32)
        .alias(f"CHAPTER_{name}")
        for _, _, name in ICD10_CHAPTERS
    )
    return persons.sort("PNR")


def make_icd10_descriptions(icd10_coding: pl.DataFrame) -> pl.DataFrame:
    """One row per ICD-10 code with its English title, as 3 strings.R reads them."""
    # Filter: codes only, not the chapter and block rows
    # Create columns:
    #   - code: the code without a dot, as the registers write it
    #   - description: the title, without the code in front of it
    return icd10_coding.filter(pl.col("coding").str.contains(r"^[A-Z][0-9]")).select(
        code="coding", description=pl.col("meaning").str.replace(r"^\S+ ", "")
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
    index_age = config["index_age"]
    inputs = {name: root / path for name, path in config["synthetic"].items()}
    outputs = {name: root / path for name, path in config["paths"].items()}
    rng = np.random.default_rng(SEED)

    print("Reading the raw data...", end=" ")
    per_person_raw = pl.read_parquet(inputs["persons"])
    per_diagnosis_raw = pl.read_parquet(inputs["contacts"])
    per_crossing_raw = pl.read_parquet(inputs["migration"])
    icd10_coding = pl.read_csv(
        inputs["icd10_coding"], separator="\t", infer_schema=False
    )
    print("Done")

    print(f"Making the synthetic data for index age {index_age}...", end=" ")
    persons = make_persons(per_person_raw, per_diagnosis_raw, index_age, rng)
    contacts = make_contacts(per_diagnosis_raw, persons)
    migration = make_migration(per_crossing_raw, persons)
    icd10_descriptions = make_icd10_descriptions(icd10_coding)
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

    for name in ["contacts", "persons", "migration", "icd10_descriptions"]:
        outputs[name].parent.mkdir(parents=True, exist_ok=True)
    contacts.write_parquet(outputs["contacts"])
    persons.write_parquet(outputs["persons"])
    migration.write_parquet(outputs["migration"])
    icd10_descriptions.write_csv(outputs["icd10_descriptions"])

    print(
        f"{persons.height} people, {contacts.height} contacts -> {outputs['persons'].parent}"
    )
    print(persons["EVENT_TYPE"].value_counts().sort("EVENT_TYPE"))
    print("Done")


if __name__ == "__main__":
    main()
