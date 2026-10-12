"""Turn the client's registry files into token timelines, the input of the models.

A timeline is one person's events as (age in days, token) pairs: diagnoses,
dementia, death and periods abroad. The steps, in the order
pipeline/build_tokens.py runs them:

1. load_inputs       read the four input files, only the columns needed
2. build_persons     one row per person: split, index date, end of record
3. build_timelines   every person's events over the whole record
4. select_variant    cut timelines before the index age (test people always,
                     everyone in a cut_at_index variant), name the dementia event
5. build_vocabulary  the token number of every event name, from training people
6. to_tokens         the timelines as token numbers
7. summarise         totals for checking

Steps 1 to 3 do not depend on the variant: build_tokens runs them once. The
decisions behind each step are explained in docs/tokens.md.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, Literal, NamedTuple

import polars as pl

# The special tokens, in the order of their numbers. PAD (0) fills the end of
# short timelines in a batch and NO_EVENT (1) marks an age at which nothing
# happened; the models rely on these two numbers. UNKNOWN_CODE stands for a
# rare code outside every ICD-10 chapter.
SPECIAL_TOKENS: Final = (
    "PAD",
    "NO_EVENT",
    "DEATH",
    "ABROAD_START",
    "ABROAD_END",
    "UNKNOWN_CODE",
)
PAD: Final = 0
NO_EVENT: Final = 1

# The subtype of a dementia without any dementia code, e.g. a date from a
# prescription
UNSPECIFIED_SUBTYPE: Final = "UNSPECIFIED"

# The 22 ICD-10 chapters, as in 3 strings.R: (first code, last code, name). A
# three-character code belongs to the chapter whose range holds it. A letter is
# not a chapter: D is split between neoplasms and blood, H between eye and ear,
# and V to Y are one chapter.
ICD10_CHAPTERS: Final = (
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
)

# The columns read from each input file, and their types
INPUT_COLUMNS: Final = {
    "contacts": {"PNR": pl.String, "INDATE": pl.Date, "DIAG": pl.String},
    "persons": {
        "PNR": pl.String,
        "BIRTH_DATE": pl.Date,
        "DEATH_DATE": pl.Date,
        "EMIGRATION_DATE": pl.Date,
        "DEMENTIA_INCIDENT_DATE": pl.Date,
    },
    "migration": {"PNR": pl.String, "HAEND_DATO": pl.Date, "INDUD_KODE": pl.String},
    "split": {"PNR": pl.String, "in_training": pl.Boolean, "fold": pl.Int32},
}

# The columns of a timeline (build_timelines)
TIMELINE_SCHEMA: Final = {
    "PNR": pl.String,
    "AGE_DAYS": pl.Int32,
    "KIND": pl.String,
    "EVENT": pl.String,
    "CHAPTER": pl.String,
    "SUBTYPE": pl.String,
}


@dataclass(frozen=True)
class TokenSettings:
    """The [settings] table of a config file, shared by all its variants."""

    index_age: int
    earliest_register_date: date
    register_end_date: date
    minimum_spell_days: int
    min_people_per_code: int
    validation_fold: int
    first_occurrence_only: bool
    dementia_subtypes: dict[str, list[str]]


@dataclass(frozen=True)
class Variant:
    """A [variants.<name>] table of a config file: one folder of tokens."""

    name: str
    dementia: Literal["single", "subtype"] = "single"
    cut_at_index: bool = False

    def __post_init__(self) -> None:
        if self.dementia not in ("single", "subtype"):
            raise ValueError(
                f"variant {self.name}: dementia must be 'single' or 'subtype', "
                f"not {self.dementia!r}"
            )


class Inputs(NamedTuple):
    contacts: pl.DataFrame
    persons: pl.DataFrame
    migration: pl.DataFrame
    split: pl.DataFrame


def load_inputs(paths: Mapping[str, Path]) -> Inputs:
    """Read the columns of INPUT_COLUMNS from each input file, cast to their types."""

    def read(name: str) -> pl.DataFrame:
        columns = INPUT_COLUMNS[name]
        return (
            pl.scan_parquet(paths[name])
            .select(pl.col(column).cast(dtype) for column, dtype in columns.items())
            .collect()
        )

    return Inputs(read("contacts"), read("persons"), read("migration"), read("split"))


def age_days(date_column: str) -> pl.Expr:
    """The age in days on the date in date_column (needs BIRTH_DATE)."""
    return (pl.col(date_column) - pl.col("BIRTH_DATE")).dt.total_days().cast(pl.Int32)


def build_persons(
    persons: pl.DataFrame, split: pl.DataFrame, settings: TokenSettings
) -> pl.DataFrame:
    """One row per person: split, observation window, index date, end of record.

    SPLIT            "test" outside the training set of 5 predictions.R,
                     "validation" in its fold settings.validation_fold, "train"
                     otherwise. The model trains on "train" only
    WINDOW_START     the birth date, or the start of the registers if later, as
                     in 3 strings.R
    INDEX_DATE       the birthday at settings.index_age (a 29 February birthday
                     becomes 28 February, as R's %m+% does)
    RECORD_END_DATE  death, emigration or the end of the registers, whichever
                     comes first: no event is recorded after it
    *_AGE_DAYS       the ages in days at INDEX_DATE and RECORD_END_DATE
    """
    missing = persons.join(split, on="PNR", how="anti")
    if not missing.is_empty():
        raise ValueError(
            f"{missing.height} people are missing from the split file, "
            f"e.g. {missing['PNR'][0]}"
        )
    return (
        persons.join(split, on="PNR")
        .with_columns(
            SPLIT=pl.when(~pl.col("in_training"))
            .then(pl.lit("test"))
            .when(pl.col("fold") == settings.validation_fold)
            .then(pl.lit("validation"))
            .otherwise(pl.lit("train")),
            WINDOW_START=pl.max_horizontal(
                "BIRTH_DATE", pl.lit(settings.earliest_register_date)
            ),
            INDEX_DATE=pl.col("BIRTH_DATE").dt.offset_by(f"{settings.index_age}y"),
            RECORD_END_DATE=pl.min_horizontal(
                "DEATH_DATE", "EMIGRATION_DATE", pl.lit(settings.register_end_date)
            ),
        )
        .with_columns(
            INDEX_AGE_DAYS=age_days("INDEX_DATE"),
            RECORD_END_AGE_DAYS=age_days("RECORD_END_DATE"),
        )
        .drop("in_training", "fold")
        .sort("PNR")
    )


def build_timelines(
    contacts: pl.DataFrame,
    migration: pl.DataFrame,
    persons: pl.DataFrame,
    settings: TokenSettings,
) -> pl.DataFrame:
    """Every person's events over the whole record, one row per event.

    KIND     "diagnosis", "dementia", "death" or "abroad"
    EVENT    the three-character ICD-10 code of a diagnosis; DEMENTIA, DEATH,
             ABROAD_START or ABROAD_END otherwise
    CHAPTER  the ICD-10 chapter of a diagnosis, null outside every chapter
    SUBTYPE  the dementia subtype of a dementia event
    """
    events = [
        diagnosis_events(contacts, persons, settings),
        dementia_events(contacts, persons, settings),
        death_events(persons),
        abroad_events(migration, persons, settings),
    ]
    return pl.concat(
        event.select(
            (
                pl.col(column).cast(dtype)
                if column in event.columns
                else pl.lit(None, dtype).alias(column)
            )
            for column, dtype in TIMELINE_SCHEMA.items()
        )
        for event in events
    ).sort("PNR", "AGE_DAYS", "EVENT")


def dementia_subtype(settings: TokenSettings) -> pl.Expr:
    """The subtype of the code in DIAG, null when it is not a dementia code.

    A code is a dementia code when it starts with one of the prefixes of
    settings.dementia_subtypes; the first subtype that matches wins.
    """
    subtype = pl.lit(None, pl.String)
    for name, prefixes in reversed(list(settings.dementia_subtypes.items())):
        is_subtype = pl.any_horizontal(
            pl.col("DIAG").str.starts_with(prefix) for prefix in prefixes
        )
        subtype = pl.when(is_subtype).then(pl.lit(name)).otherwise(subtype)
    return subtype


def icd10_chapter(column: str) -> pl.Expr:
    """The ICD10_CHAPTERS name of the three-character code in column, or null."""
    return pl.coalesce(
        pl.when(pl.col(column).is_between(pl.lit(first), pl.lit(last))).then(
            pl.lit(name)
        )
        for first, last, name in ICD10_CHAPTERS
    )


def within_record(date_column: str) -> pl.Expr:
    """The date in date_column is inside the record: from WINDOW_START to RECORD_END_DATE."""
    return pl.col(date_column).is_between(
        pl.col("WINDOW_START"), pl.col("RECORD_END_DATE")
    )


def diagnosis_events(
    contacts: pl.DataFrame, persons: pl.DataFrame, settings: TokenSettings
) -> pl.DataFrame:
    """Diagnoses inside the record, without the dementia codes.

    EVENT is the first three characters of the code. With
    settings.first_occurrence_only, each code is kept at its first contact only.
    Dementia codes are left out: dementia_events places dementia.
    """
    diagnoses = (
        contacts.filter(pl.col("INDATE").is_not_null(), pl.col("DIAG").is_not_null())
        .filter(dementia_subtype(settings).is_null())
        .join(persons, on="PNR")
        .filter(within_record("INDATE"))
        .select(
            "PNR",
            AGE_DAYS=age_days("INDATE"),
            KIND=pl.lit("diagnosis"),
            EVENT=pl.col("DIAG").str.slice(0, 3),
        )
        .unique()
    )
    if settings.first_occurrence_only:
        diagnoses = diagnoses.group_by("PNR", "KIND", "EVENT").agg(
            pl.col("AGE_DAYS").min()
        )
    return diagnoses.with_columns(CHAPTER=icd10_chapter("EVENT"))


def dementia_events(
    contacts: pl.DataFrame, persons: pl.DataFrame, settings: TokenSettings
) -> pl.DataFrame:
    """One DEMENTIA event per person with dementia, at DEMENTIA_INCIDENT_DATE.

    The date is the client's outcome, which may come from a prescription rather
    than a diagnosis. SUBTYPE is the subtype of the person's first dementia code
    inside the record (codes in alphabetical order on the same day), or
    UNSPECIFIED_SUBTYPE without any.
    """
    first_codes = (
        contacts.with_columns(SUBTYPE=dementia_subtype(settings))
        .filter(pl.col("SUBTYPE").is_not_null())
        .join(persons, on="PNR")
        .filter(within_record("INDATE"))
        .sort("PNR", "INDATE", "DIAG")
        .unique(subset="PNR", keep="first")
        .select("PNR", "SUBTYPE")
    )
    return (
        persons.filter(within_record("DEMENTIA_INCIDENT_DATE"))
        .join(first_codes, on="PNR", how="left")
        .select(
            "PNR",
            AGE_DAYS=age_days("DEMENTIA_INCIDENT_DATE"),
            KIND=pl.lit("dementia"),
            EVENT=pl.lit("DEMENTIA"),
            SUBTYPE=pl.col("SUBTYPE").fill_null(UNSPECIFIED_SUBTYPE),
        )
    )


def death_events(persons: pl.DataFrame) -> pl.DataFrame:
    """One DEATH event per person who died inside the record.

    A death after emigration is not recorded: the record ends at emigration.
    """
    return persons.filter(within_record("DEATH_DATE")).select(
        "PNR",
        AGE_DAYS=age_days("DEATH_DATE"),
        KIND=pl.lit("death"),
        EVENT=pl.lit("DEATH"),
    )


def abroad_events(
    migration: pl.DataFrame, persons: pl.DataFrame, settings: TokenSettings
) -> pl.DataFrame:
    """ABROAD_START and ABROAD_END events of each period abroad before the index date.

    A port of section 5 of 3 strings.R, which writes the same periods as
    clauses:

    1. Each person's border crossings in date order (INDUD_KODE "U" out, "I"
       in). A crossing that repeats the previous one is dropped, so a departure
       is never paired with another departure.
    2. A departure opens a period, closed by the next return or left open. A
       first crossing that is a return means the person was abroad before the
       registers start.
    3. Each period is clipped to the window, from WINDOW_START to the index
       date. A period still open a year before the index date is cut there,
       because the cohort lives in Denmark in the last year before it.
    4. Only periods of at least settings.minimum_spell_days get events.
    """
    crossings = (
        migration.join(persons, on="PNR", how="semi")
        .filter(
            pl.col("HAEND_DATO").is_not_null(), pl.col("INDUD_KODE").is_in(["U", "I"])
        )
        .sort("PNR", "HAEND_DATO")
    )
    previous_code = pl.col("INDUD_KODE").shift(1).over("PNR")
    crossings = crossings.filter(
        previous_code.is_null() | (pl.col("INDUD_KODE") != previous_code)
    ).with_columns(
        NEXT_CODE=pl.col("INDUD_KODE").shift(-1).over("PNR"),
        NEXT_DATE=pl.col("HAEND_DATO").shift(-1).over("PNR"),
        IS_FIRST=pl.int_range(pl.len()).over("PNR") == 0,
    )

    departures = crossings.filter(pl.col("INDUD_KODE") == "U").select(
        "PNR",
        FROM_DATE=pl.col("HAEND_DATO"),
        TO_DATE=pl.when(pl.col("NEXT_CODE") == "I").then(pl.col("NEXT_DATE")),
    )
    abroad_from_start = crossings.filter(
        pl.col("IS_FIRST"), pl.col("INDUD_KODE") == "I"
    ).select("PNR", FROM_DATE=pl.lit(None, pl.Date), TO_DATE=pl.col("HAEND_DATO"))

    periods = (
        pl.concat([departures, abroad_from_start])
        .join(persons, on="PNR")
        .with_columns(
            YEAR_BEFORE_INDEX=pl.col("INDEX_DATE").dt.offset_by("-1y"),
            START_DATE=pl.max_horizontal(
                pl.coalesce("FROM_DATE", "WINDOW_START"), "WINDOW_START"
            ),
            END_DATE=pl.min_horizontal(
                pl.coalesce("TO_DATE", "INDEX_DATE"), "INDEX_DATE"
            ),
        )
        .with_columns(
            END_DATE=pl.when(
                pl.col("TO_DATE").is_null()
                & (pl.col("END_DATE") > pl.col("YEAR_BEFORE_INDEX"))
            )
            .then(pl.col("YEAR_BEFORE_INDEX"))
            .otherwise(pl.col("END_DATE"))
        )
        .filter(
            (pl.col("END_DATE") - pl.col("START_DATE")).dt.total_days()
            >= settings.minimum_spell_days
        )
    )
    return pl.concat(
        periods.select(
            "PNR", AGE_DAYS=age_days(date), KIND=pl.lit("abroad"), EVENT=pl.lit(event)
        )
        for date, event in (("START_DATE", "ABROAD_START"), ("END_DATE", "ABROAD_END"))
    )


def select_variant(
    timelines: pl.DataFrame, persons: pl.DataFrame, variant: Variant
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The timelines and people of one variant.

    Test people's timelines stop before the index date, so their outcome never
    reaches the model; with variant.cut_at_index, everyone's do.
    OBSERVED_UNTIL_AGE_DAYS is the age at which each timeline stops: the end of
    the record, or the index age where the timeline is cut. With
    variant.dementia "subtype", the dementia event is named after its subtype
    (DEMENTIA_ALZHEIMER, ...).
    """
    cut = (pl.col("SPLIT") == "test") | pl.lit(variant.cut_at_index)
    persons = persons.with_columns(
        OBSERVED_UNTIL_AGE_DAYS=pl.when(cut)
        .then(pl.min_horizontal("RECORD_END_AGE_DAYS", "INDEX_AGE_DAYS"))
        .otherwise(pl.col("RECORD_END_AGE_DAYS"))
    )
    timelines = (
        timelines.join(persons.select("PNR", "SPLIT", "INDEX_AGE_DAYS"), on="PNR")
        .filter(~cut | (pl.col("AGE_DAYS") < pl.col("INDEX_AGE_DAYS")))
        .drop("SPLIT", "INDEX_AGE_DAYS")
    )
    if variant.dementia == "subtype":
        timelines = timelines.with_columns(
            EVENT=pl.when(pl.col("KIND") == "dementia")
            .then(pl.lit("DEMENTIA_") + pl.col("SUBTYPE"))
            .otherwise(pl.col("EVENT"))
        )
    return timelines, persons


def build_vocabulary(
    timelines: pl.DataFrame,
    persons: pl.DataFrame,
    settings: TokenSettings,
    variant: Variant,
) -> pl.DataFrame:
    """One row per token: TOKEN (its number), NAME, KIND, CHAPTER, N_TRAIN_PEOPLE.

    In order: the special tokens (PAD = 0, NO_EVENT = 1, ...), the dementia
    token(s), one RARE_<chapter> token per ICD-10 chapter, then every
    three-character code that at least settings.min_people_per_code training
    people have, in alphabetical order. N_TRAIN_PEOPLE is that number of people
    (codes only). Validation and test people are not counted.
    """
    if variant.dementia == "single":
        dementia = ["DEMENTIA"]
    else:
        subtypes = dict.fromkeys([*settings.dementia_subtypes, UNSPECIFIED_SUBTYPE])
        dementia = [f"DEMENTIA_{subtype}" for subtype in subtypes]

    fixed = pl.DataFrame(
        [(name, "special", None) for name in SPECIAL_TOKENS]
        + [(name, "dementia", None) for name in dementia]
        + [(f"RARE_{name}", "rare code", name) for _, _, name in ICD10_CHAPTERS],
        schema={"NAME": pl.String, "KIND": pl.String, "CHAPTER": pl.String},
        orient="row",
    )
    training_people = persons.filter(pl.col("SPLIT") == "train").select("PNR")
    codes = (
        timelines.filter(pl.col("KIND") == "diagnosis")
        .join(training_people, on="PNR")
        .group_by("EVENT", "CHAPTER")
        .agg(N_TRAIN_PEOPLE=pl.col("PNR").n_unique().cast(pl.Int64))
        .filter(pl.col("N_TRAIN_PEOPLE") >= settings.min_people_per_code)
        .sort("EVENT")
        .select(
            NAME="EVENT",
            KIND=pl.lit("code"),
            CHAPTER="CHAPTER",
            N_TRAIN_PEOPLE="N_TRAIN_PEOPLE",
        )
    )
    return (
        pl.concat([fixed, codes], how="diagonal")
        .with_row_index("TOKEN")
        .with_columns(pl.col("TOKEN").cast(pl.Int32))
    )


def to_tokens(
    timelines: pl.DataFrame, vocabulary: pl.DataFrame, settings: TokenSettings
) -> pl.DataFrame:
    """The timelines as token numbers: one row per PNR, AGE_DAYS and TOKEN, sorted.

    A diagnosis gets the token of its code when the vocabulary has one, else the
    RARE_<chapter> token of its chapter, else UNKNOWN_CODE. With
    settings.first_occurrence_only, each diagnosis token is kept at its first
    age only, since several rare codes of a chapter share one token.
    """
    codes = vocabulary.filter(pl.col("KIND") == "code")["NAME"].to_list()
    named = timelines.with_columns(
        NAME=pl.when(pl.col("KIND") != "diagnosis")
        .then(pl.col("EVENT"))
        .when(pl.col("EVENT").is_in(codes))
        .then(pl.col("EVENT"))
        .when(pl.col("CHAPTER").is_not_null())
        .then(pl.lit("RARE_") + pl.col("CHAPTER"))
        .otherwise(pl.lit("UNKNOWN_CODE"))
    ).join(vocabulary.select("NAME", "TOKEN"), on="NAME", how="left")
    unnamed = named.filter(pl.col("TOKEN").is_null())["NAME"].unique().to_list()
    assert not unnamed, f"events without a token: {unnamed}"

    if settings.first_occurrence_only:
        is_diagnosis = pl.col("KIND") == "diagnosis"
        named = pl.concat(
            [
                named.filter(is_diagnosis)
                .group_by("PNR", "TOKEN")
                .agg(pl.col("AGE_DAYS").min()),
                named.filter(~is_diagnosis).select("PNR", "TOKEN", "AGE_DAYS"),
            ]
        )
    return (
        named.select("PNR", "AGE_DAYS", "TOKEN")
        .unique()
        .sort("PNR", "AGE_DAYS", "TOKEN")
    )


def summarise(
    events: pl.DataFrame,
    timelines: pl.DataFrame,
    persons: pl.DataFrame,
    vocabulary: pl.DataFrame,
    context_length: int = 96,
) -> dict[str, pl.DataFrame]:
    """Totals for checking one variant, one table per topic.

    Aggregates only, but as with the client's scripts, check that no count is
    below 5 before taking any of them off the server. context_length is
    Delphi's: the share of longer timelines tells whether it is long enough.
    """
    events_per_person = (
        persons.select("PNR", "SPLIT")
        .join(events.group_by("PNR").agg(EVENTS=pl.len()), on="PNR", how="left")
        .with_columns(pl.col("EVENTS").fill_null(0))
    )
    people = (
        events_per_person.group_by("SPLIT")
        .agg(
            PEOPLE=pl.len(),
            WITH_EVENTS=(pl.col("EVENTS") > 0).sum(),
            MEDIAN_EVENTS=pl.col("EVENTS").median(),
            P99_EVENTS=pl.col("EVENTS").quantile(0.99),
            SHARE_LONGER_THAN_CONTEXT=(pl.col("EVENTS") > context_length).mean(),
        )
        .sort("SPLIT")
    )

    # Diagnoses of training people, by three-character code
    training_people = persons.filter(pl.col("SPLIT") == "train").select("PNR")
    diagnoses = timelines.filter(pl.col("KIND") == "diagnosis").join(
        training_people, on="PNR"
    )
    people_per_code = diagnoses.group_by("EVENT").agg(PEOPLE=pl.col("PNR").n_unique())
    diagnoses = diagnoses.join(people_per_code, on="EVENT")
    thresholds = [5, 10, 25, 50, 100]
    rare_codes = pl.DataFrame(
        {
            "FEWER_PEOPLE_THAN": thresholds,
            "CODES": [(people_per_code["PEOPLE"] < t).sum() for t in thresholds],
            "SHARE_OF_DIAGNOSES": [
                (diagnoses["PEOPLE"] < t).mean() for t in thresholds
            ],
        }
    )

    # Codes Delphi-2M's vocabulary lacks: it only has chapters A to Q
    outside = ~pl.col("EVENT").str.slice(0, 1).is_in(list("ABCDEFGHIJKLMNOPQ"))
    outside_delphi = diagnoses.select(
        SHARE_OF_DIAGNOSES=outside.mean(),
        SHARE_OF_PEOPLE=pl.col("PNR").filter(outside).n_unique()
        / training_people.height,
    )

    # Outcome tokens of training people
    outcome_names = vocabulary.filter(
        (pl.col("KIND") == "dementia") | (pl.col("NAME") == "DEATH")
    ).select("TOKEN", "NAME")
    outcomes = (
        outcome_names.join(
            events.join(training_people, on="PNR")
            .group_by("TOKEN")
            .agg(TRAINING_PEOPLE=pl.col("PNR").n_unique()),
            on="TOKEN",
            how="left",
        )
        .with_columns(pl.col("TRAINING_PEOPLE").fill_null(0))
        .select("NAME", "TRAINING_PEOPLE")
    )

    return {
        "Events per person, by split": people,
        "Training diagnoses whose code has fewer training people than": rare_codes,
        "Training diagnoses outside Delphi's chapters (A-Q)": outside_delphi,
        "Training people with each outcome token": outcomes,
    }
