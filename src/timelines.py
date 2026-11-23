"""Every person's timeline: their events with their ages, shared by every variant."""

import polars as pl

from src._config import (
    DEMENTIA_CODE_PREFIXES,
    EARLIEST_REGISTER_DATE,
    MINIMUM_SPELL_DAYS,
    REGISTER_END,
    RISK_FACTOR_CODES,
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
    # Columns selected: PNR, sex, dates of birth, death, emigration and dementia,
    #   and the other covariates of the Cox models (birth year, education,
    #   children alive at the index age, bereavements)
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
            "BIRTH_COHORT",
            "EDU_CAT3",
            "CHILDREN_ALIVE_AT_70",
            "CHILD_BEREAVEMENT",
            "SPOUSE_BEREAVEMENT",
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


def make_covariates(
    per_education_raw: pl.DataFrame, persons: pl.DataFrame
) -> pl.DataFrame:
    """The other covariates of the Cox models as events, kept at any age as sex."""
    # One row per covariate of a person, as an event of kind covariate:
    #   - BORN_<birth year> at age 0
    #   - EDUCATION_<level> (SHORT, MEDIUM, LONG) at the first date of the
    #     person's highest level in per_education_raw, at the latest the day
    #     before the index date, so that test people keep it. A date before
    #     birth cannot be true: it is left out, and without another date the
    #     level is undated
    #   - the day before the index date: EDUCATION_UNKNOWN without a level, an
    #     undated level, CHILDREN_ALIVE_<number> (0, 1, 2, 3+),
    #     CHILD_BEREAVEMENT and SPOUSE_BEREAVEMENT if the person had one
    before_index = (pl.col("INDEX_AGE_DAYS") - 1).cast(pl.Int32)
    education_dates = (
        per_education_raw.join(persons.select("PNR", "BIRTH_DATE"), on="PNR")
        .filter(pl.col("EDU_DATE") >= pl.col("BIRTH_DATE"))
        .group_by("PNR", "EDU_CAT3")
        .agg(pl.col("EDU_DATE").min())
    )
    education = persons.join(education_dates, on=["PNR", "EDU_CAT3"], how="left")

    def event(rows: pl.DataFrame, age: pl.Expr, name: pl.Expr) -> pl.DataFrame:
        """One event per row: PNR, AGE_DAYS, KIND, EVENT."""
        return rows.select("PNR", AGE_DAYS=age, KIND=pl.lit("covariate"), EVENT=name)

    return pl.concat(
        [
            event(
                persons.filter(pl.col("BIRTH_COHORT").is_not_null()),
                pl.lit(0, pl.Int32),
                pl.format("BORN_{}", pl.col("BIRTH_COHORT").cast(pl.Int32)),
            ),
            event(
                education,
                pl.when(pl.col("EDU_CAT3").is_null())
                .then(before_index)
                .otherwise(
                    pl.min_horizontal(
                        pl.coalesce(age_days("EDU_DATE"), before_index), before_index
                    )
                ),
                pl.lit("EDUCATION_")
                + pl.col("EDU_CAT3").fill_null("unknown").str.to_uppercase(),
            ),
            event(
                persons.filter(pl.col("CHILDREN_ALIVE_AT_70").is_not_null()),
                before_index,
                pl.lit("CHILDREN_ALIVE_") + pl.col("CHILDREN_ALIVE_AT_70"),
            ),
            event(
                persons.filter(pl.col("CHILD_BEREAVEMENT") == 1),
                before_index,
                pl.lit("CHILD_BEREAVEMENT"),
            ),
            event(
                persons.filter(pl.col("SPOUSE_BEREAVEMENT") == 1),
                before_index,
                pl.lit("SPOUSE_BEREAVEMENT"),
            ),
        ]
    )


def make_risks(diagnoses: pl.DataFrame) -> pl.DataFrame:
    """The risk factors of the tabular data, at the first contact with one of their codes."""
    # One row per code of a risk factor: CODE, RISK (RISK_<name>)
    codes = pl.DataFrame(
        [
            (code, f"RISK_{name}")
            for name, factor_codes in RISK_FACTOR_CODES.items()
            for code in factor_codes
        ],
        schema={"CODE": pl.String, "RISK": pl.String},
        orient="row",
    )
    return (
        diagnoses.filter(pl.col("KIND") == "diagnosis")
        .join(codes, left_on="EVENT", right_on="CODE")
        .group_by("PNR", "RISK")
        .agg(pl.col("AGE_DAYS").min())
        .select("PNR", "AGE_DAYS", KIND=pl.lit("risk"), EVENT="RISK")
    )


def make_timelines(
    per_diagnosis_raw: pl.DataFrame,
    per_crossing_raw: pl.DataFrame,
    per_education_raw: pl.DataFrame,
    persons: pl.DataFrame,
) -> pl.DataFrame:
    """One row per event: diagnoses, dementia, death, periods abroad, sex, covariates and risk factors."""
    # Filter: test people's events before the index date only, so that their
    #   outcome never reaches the models
    # One row per event, everyone's whole record: PNR, AGE_DAYS, KIND, EVENT
    diagnoses = make_diagnoses(per_diagnosis_raw, persons)
    events = pl.concat(
        [
            diagnoses,
            make_outcomes(persons),
            make_abroad(per_crossing_raw, persons),
            make_sex(persons),
            make_covariates(per_education_raw, persons),
            make_risks(diagnoses),
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
